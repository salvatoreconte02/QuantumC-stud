# my_extensions/qar_to_quantum_mlir.py

"""
Lowering da QAR (QarModule) al quantum MLIR di QuantumC.

In questa versione:
- inizializza vettori e matrici con valori REALI se disponibili in:
    qar_module.const_arrays  +  qar_module.const_shapes
- supporta anche inizializzazione "fallback" a 0 se non ci sono costanti
- implementa lowering di:
    * QarMapAdd  (vec add)
    * QarDot     (vec dot)
    * QarMatMul  (matmul)

AGGIUNTA STRUTTURALE:
- costruisce una mappa semantica result_map che collega:
    ("C", i, j) -> SSAValue corrispondente a C[i][j]
  e per i vettori:
    ("c", i) -> SSAValue corrispondente a c[i]
  e per gli scalari (dot):
    ("acc",) -> SSAValue finale
Questa mappa serve per rendere il "return" generale e corretto nel merge.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any

from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.dialects.func import FuncOp
from xdsl.ir import Block, Region, SSAValue

from my_extensions.qar_ir import (
    QarModule,
    QarMapAdd,
    QarDot,
    QarMatMul,
)

from step4_mlir_to_quantum_mlir.quantum_dialect import (
    QuantumInitOp,
    QAddiOp,
    QMuliOp,
)


# ---------------------------------------------------------------------------
# Analisi del QarModule: raccolta info su vettori, matrici e scalari
# ---------------------------------------------------------------------------

def _collect_symbols(
    qar_module: QarModule,
) -> tuple[dict[str, tuple[tuple[int, ...], int]], set[str]]:
    """
    Raccoglie:
      - tensor_info: name -> (shape, elem_bits)
            shape = (L,) per vettori
            shape = (rows, cols) per matrici
      - scalar_names: nomi scalari (es. 'acc' in dot)
    """
    tensor_info: dict[str, tuple[tuple[int, ...], int]] = {}
    scalar_names: set[str] = set()

    def _numel(shape: tuple[int, ...]) -> int:
        t = 1
        for d in shape:
            t *= d
        return t

    def _update(name: str, shape: tuple[int, ...], elem_bits: int):
        old = tensor_info.get(name)
        if old is None:
            tensor_info[name] = (shape, elem_bits)
            return

        old_shape, _old_bits = old

        # Se la "dimensione" (rank) è diversa, non risolvibile qui:
        if len(shape) != len(old_shape):
            return

        # Stesso rank: tenere la shape con più elementi (se capita).
        if _numel(shape) > _numel(old_shape):
            tensor_info[name] = (shape, elem_bits)

    for fn in qar_module.functions:
        for op in fn.ops:
            if isinstance(op, QarMapAdd):
                shp = (op.length,)
                for name in (op.dest, op.lhs, op.rhs):
                    _update(name, shp, op.elem_bits)

            elif isinstance(op, QarDot):
                scalar_names.add(op.dest)
                shp = (op.length,)
                for name in (op.lhs, op.rhs):
                    _update(name, shp, op.elem_bits)

            elif isinstance(op, QarMatMul):
                _update(op.lhs, (op.m, op.k), op.elem_bits)
                _update(op.rhs, (op.k, op.n), op.elem_bits)
                _update(op.dest, (op.m, op.n), op.elem_bits)

    return tensor_info, scalar_names


# ---------------------------------------------------------------------------
# Entry point: QarModule -> ModuleOp (quantum dialect)
# ---------------------------------------------------------------------------

def from_qar_to_quantum_mlir(qar_module: QarModule, num_bits: int = 16) -> ModuleOp:
    """
    QarModule -> ModuleOp (quantum dialect)

    Inizializza registri anche quando non ci sono macro-op QAR:
    usa const_shapes/const_arrays per costruire tensor_info.

    NOTA: viene aggiunto module.result_map come side-table del compilatore.
    """
    tensor_info, scalar_names = _collect_symbols(qar_module)

    const_arrays = getattr(qar_module, "const_arrays", {}) or {}
    const_shapes = getattr(qar_module, "const_shapes", {}) or {}

    for name in set(const_arrays.keys()) | set(const_shapes.keys()):
        if name in tensor_info:
            continue
        if name in const_shapes:
            tensor_info[name] = (tuple(const_shapes[name]), num_bits)
        elif name in const_arrays:
            tensor_info[name] = ((len(const_arrays[name]),), num_bits)

    module = ModuleOp([])

    # Side-table per collegare elementi (vec/mat) ai rispettivi SSAValue finali.
    # Chiave:
    #   - vettore: (name, i)
    #   - matrice: (name, i, j)
    #   - scalare: (name,)
    module.result_map: Dict[Tuple[Any, ...], SSAValue] = {}

    entry_block = Block()
    func_region = Region([entry_block])
    func_type = ([], [])
    func = FuncOp("main", func_type, func_region)
    module.body.blocks[0].add_op(func)

    tensor_env: Dict[str, List[SSAValue]] = {}
    scalar_env: Dict[str, SSAValue] = {}

    _emit_tensor_and_scalar_inits(
        entry_block,
        tensor_info,
        scalar_names,
        tensor_env,
        scalar_env,
        qar_module,
    )

    _emit_qar_ops(entry_block, qar_module, tensor_env, scalar_env, module.result_map)

    return module


# ---------------------------------------------------------------------------
# PROLOGO: inizializzazione registri per tensori (vettori/matrici) e scalari
# ---------------------------------------------------------------------------

def _emit_tensor_and_scalar_inits(
    block: Block,
    tensor_info: dict[str, tuple[tuple[int, ...], int]],
    scalar_names: set[str],
    tensor_env: dict[str, list[SSAValue]],
    scalar_env: dict[str, SSAValue],
    qar_module: QarModule,
):
    const_arrays = getattr(qar_module, "const_arrays", {}) or {}
    const_shapes = getattr(qar_module, "const_shapes", {}) or {}

    def _numel(shape: tuple[int, ...]) -> int:
        t = 1
        for d in shape:
            t *= d
        return t

    for name, (inferred_shape, _bits) in tensor_info.items():
        shape = const_shapes.get(name, inferred_shape)

        if len(shape) not in (1, 2):
            raise ValueError(f"Shape non supportata per '{name}': {shape}")

        total = _numel(shape)
        init_vals = const_arrays.get(name)

        regs: list[SSAValue] = []
        for idx in range(total):
            v = int(init_vals[idx]) if (init_vals is not None and idx < len(init_vals)) else 0
            init_op = QuantumInitOp(v, i32)
            block.add_op(init_op)
            init_op.results[0].name_hint = f"q_{name}_{idx}"
            regs.append(init_op.results[0])

        tensor_env[name] = regs

    for name in scalar_names:
        init_op = QuantumInitOp(0, i32)
        block.add_op(init_op)
        init_op.results[0].name_hint = f"q_{name}"
        scalar_env[name] = init_op.results[0]


# ---------------------------------------------------------------------------
# Lowering delle op QAR in operazioni quantum.*
# ---------------------------------------------------------------------------

def _emit_qar_ops(
    block: Block,
    qar_module: QarModule,
    tensor_env: dict[str, list[SSAValue]],
    scalar_env: dict[str, SSAValue],
    result_map: Dict[Tuple[Any, ...], SSAValue],
):
    for fn in qar_module.functions:
        for op in fn.ops:
            if isinstance(op, QarMapAdd):
                _lower_vec_add(block, op, tensor_env, result_map)
            elif isinstance(op, QarDot):
                _lower_vec_dot(block, op, tensor_env, scalar_env, result_map)
            elif isinstance(op, QarMatMul):
                _lower_matmul(block, op, tensor_env, result_map)
            else:
                pass


def _lower_vec_add(
    block: Block,
    op: QarMapAdd,
    tensor_env: dict[str, list[SSAValue]],
    result_map: Dict[Tuple[Any, ...], SSAValue],
):
    dest_regs = tensor_env[op.dest]
    lhs_regs = tensor_env[op.lhs]
    rhs_regs = tensor_env[op.rhs]

    for i in range(op.length):
        add_op = QAddiOp(lhs_regs[i], rhs_regs[i])
        block.add_op(add_op)
        dest_regs[i] = add_op.results[0]

        # registra c[i] -> SSAValue
        result_map[(op.dest, i)] = dest_regs[i]


def _lower_vec_dot(
    block: Block,
    op: QarDot,
    tensor_env: dict[str, list[SSAValue]],
    scalar_env: dict[str, SSAValue],
    result_map: Dict[Tuple[Any, ...], SSAValue],
):
    a_regs = tensor_env[op.lhs]
    b_regs = tensor_env[op.rhs]
    acc = scalar_env[op.dest]

    for i in range(op.length):
        mul_op = QMuliOp(a_regs[i], b_regs[i])
        block.add_op(mul_op)

        add_op = QAddiOp(acc, mul_op.results[0])
        block.add_op(add_op)
        acc = add_op.results[0]

    scalar_env[op.dest] = acc
    result_map[(op.dest,)] = acc  # scalare finale


def _lower_matmul(
    block: Block,
    op: QarMatMul,
    tensor_env: dict[str, list[SSAValue]],
    result_map: Dict[Tuple[Any, ...], SSAValue],
):
    """
    C = A * B
    A: (m,k) row-major flatten
    B: (k,n) row-major flatten
    C: (m,n) row-major flatten

    C[i,j] = sum_{kk=0..k-1} A[i,kk] * B[kk,j]
    """
    A = tensor_env[op.lhs]
    B = tensor_env[op.rhs]
    C = tensor_env[op.dest]

    m, n, k = op.m, op.n, op.k

    def idx(row: int, col: int, ncols: int) -> int:
        return row * ncols + col

    for i in range(m):
        for j in range(n):
            acc = C[idx(i, j, n)]
            for kk in range(k):
                a_ik = A[idx(i, kk, k)]
                b_kj = B[idx(kk, j, n)]

                mul_op = QMuliOp(a_ik, b_kj)
                block.add_op(mul_op)

                add_op = QAddiOp(acc, mul_op.results[0])
                block.add_op(add_op)
                acc = add_op.results[0]

            C[idx(i, j, n)] = acc

            # registra C[i][j] -> SSAValue (semantica preservata)
            result_map[(op.dest, i, j)] = acc