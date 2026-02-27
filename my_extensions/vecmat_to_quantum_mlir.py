# my_extensions/vecmat_to_quantum_mlir.py

"""
Lowering da VecMatModule a quantum MLIR (dialetto QuantumC).

Questo modulo:
- inizializza vettori/matrici usando le costanti in `const_arrays/const_shapes` se presenti
  (altrimenti inizializza a 0)
- effettua il lowering di VecAddOp, VecDotOp e MatMulOp
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any

from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.dialects.func import FuncOp
from xdsl.ir import Block, Region, SSAValue

from my_extensions.vecmat_ir import (
    VecMatModule,
    VecAddOp,
    VecDotOp,
    MatMulOp,
)

from step4_mlir_to_quantum_mlir.quantum_dialect import (
    QuantumInitOp,
    QAddiOp,
    QMuliOp,
)


######### Analisi del VecMatModule: raccolta info su vettori, matrici e scalari ##########

def _collect_symbols(
    vecmat_module: VecMatModule,
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
        """
        Inserisce/aggiorna le info di un simbolo.
        Se il simbolo compare con shape diverse ma stesso rank, teniamo quella più grande.
        Se il rank è diverso, ignoriamo (caso ambiguo che non è gestito qui).
        """
        old = tensor_info.get(name)
        if old is None:
            tensor_info[name] = (shape, elem_bits)
            return

        old_shape, _old_bits = old

        # Se la "dimensione" (rank) è diversa, non risolviamo qui
        if len(shape) != len(old_shape):
            return

        # Stesso rank: tenere la shape con più elementi (se capita).
        if _numel(shape) > _numel(old_shape):
            tensor_info[name] = (shape, elem_bits)

    for fn in vecmat_module.functions:
        for op in fn.ops:
            if isinstance(op, VecAddOp):
                shp = (op.length,)
                for name in (op.dest, op.lhs, op.rhs):
                    _update(name, shp, op.elem_bits)

            elif isinstance(op, VecDotOp):
                scalar_names.add(op.dest)
                shp = (op.length,)
                for name in (op.lhs, op.rhs):
                    _update(name, shp, op.elem_bits)

            elif isinstance(op, MatMulOp):
                _update(op.lhs, (op.m, op.k), op.elem_bits)
                _update(op.rhs, (op.k, op.n), op.elem_bits)
                _update(op.dest, (op.m, op.n), op.elem_bits)

    return tensor_info, scalar_names


#### Entry point: VecMatModule -> ModuleOp (quantum dialect) ####

def from_vecmat_to_quantum_mlir(vecmat_module: VecMatModule, num_bits: int = 16) -> ModuleOp:
    """
    Traduce un VecMatModule in un ModuleOp (dialetto quantum).

    - deduce shape dei tensori dalle ops (e dalle costanti se presenti)
    - inizializza tensori/scalari
    - emette le macro-op (vec_add / vec_dot / matmul)
    """
    tensor_info, scalar_names = _collect_symbols(vecmat_module)

    const_arrays = getattr(vecmat_module, "const_arrays", {}) or {}
    const_shapes = getattr(vecmat_module, "const_shapes", {}) or {}

    # Completa tensor_info usando le costanti dichiarate nel C anche se non compaiono in ops.
    for name in set(const_arrays.keys()) | set(const_shapes.keys()):
        if name in tensor_info:
            continue
        if name in const_shapes:
            tensor_info[name] = (tuple(const_shapes[name]), num_bits)
        elif name in const_arrays:
            tensor_info[name] = ((len(const_arrays[name]),), num_bits)

    module = ModuleOp([])

    # Side-table: collega (nome, indici) allo SSAValue "corrente/finale" generato dal lowering.
    # Serve per recuperare risultati in modo stabile (return/merge), senza dipendere
    # dall'ultimo valore visto durante l'emissione.
    #
    # Chiavi:
    #   - vettore: (name, i)
    #   - matrice: (name, i, j)
    #   - scalare: (name,)

    module.result_map: Dict[Tuple[Any, ...], SSAValue] = {}

    # Crea la funzione main vuota e aggiunge il blocco di ingresso.
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
        vecmat_module,
    )

    _emit_vecmat_ops(entry_block, vecmat_module, tensor_env, scalar_env, module.result_map, const_arrays)

    return module


###### PROLOGO: inizializzazione registri per tensori (vettori/matrici) e scalari ######

def _emit_tensor_and_scalar_inits(
    block: Block,
    tensor_info: dict[str, tuple[tuple[int, ...], int]],
    scalar_names: set[str],
    tensor_env: dict[str, list[SSAValue]],
    scalar_env: dict[str, SSAValue],
    vecmat_module: VecMatModule,
):
    """
    Alloca e inizializza i registri quantum per tutti i simboli usati.

    - tensori: creati come lista piatta di SSAValue (row-major per matrici)
    - scalari: inizializzati a 0
    - se ci sono costanti in vecmat_module.const_arrays/const_shapes, vengono usate;
      altrimenti fallback a 0 e/o shape inferita
    """
    const_arrays = getattr(vecmat_module, "const_arrays", {}) or {}
    const_shapes = getattr(vecmat_module, "const_shapes", {}) or {}

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


###### Lowering delle op VecMat in operazioni quantum.* ########

def _emit_vecmat_ops(
    block: Block,
    vecmat_module: VecMatModule,
    tensor_env: dict[str, list[SSAValue]],
    scalar_env: dict[str, SSAValue],
    result_map: Dict[Tuple[Any, ...], SSAValue],
    const_arrays: dict[str, list[int]],
):
    """Emette il lowering delle ops VecMat in operazioni quantum.* sul block."""
    for fn in vecmat_module.functions:
        for op in fn.ops:
            if isinstance(op, VecAddOp):
                _lower_vec_add(block, op, tensor_env, result_map)
            elif isinstance(op, VecDotOp):
                _lower_vec_dot(block, op, tensor_env, scalar_env, result_map, const_arrays)
            elif isinstance(op, MatMulOp):
                _lower_matmul(block, op, tensor_env, result_map, const_arrays)
            else:
                pass


def _lower_vec_add(
    block: Block,
    op: VecAddOp,
    tensor_env: dict[str, list[SSAValue]],
    result_map: Dict[Tuple[Any, ...], SSAValue],
):
    """Lowering di c[i] = a[i] + b[i] per i in [0, L)."""
    dest_regs = tensor_env[op.dest]
    lhs_regs = tensor_env[op.lhs]
    rhs_regs = tensor_env[op.rhs]

    for i in range(op.length):
        add_op = QAddiOp(lhs_regs[i], rhs_regs[i])
        block.add_op(add_op)
        dest_regs[i] = add_op.results[0]

        # Aggiorna il registro destinazione con il nuovo SSAValue (stile SSA).
        result_map[(op.dest, i)] = dest_regs[i]


def _lower_vec_dot(
    block: Block,
    op: VecDotOp,
    tensor_env: dict[str, list[SSAValue]],
    scalar_env: dict[str, SSAValue],
    result_map: Dict[Tuple[Any, ...], SSAValue],
    const_arrays: dict[str, list[int]],
):
    """
    Lowering di acc = sum_i (a[i] * b[i]).
    Se sono disponibili i valori costanti di a e b, salta i prodotti con operandi zero.
    """
    a_regs = tensor_env[op.lhs]
    b_regs = tensor_env[op.rhs]
    acc = scalar_env[op.dest]

    for i in range(op.length):
       
        a_vals = const_arrays.get(op.lhs)
        b_vals = const_arrays.get(op.rhs)
        if a_vals is not None and b_vals is not None:
            if a_vals[i] == 0 or b_vals[i] == 0:
                continue

        mul_op = QMuliOp(a_regs[i], b_regs[i])
        block.add_op(mul_op)

        add_op = QAddiOp(acc, mul_op.results[0])
        block.add_op(add_op)
        acc = add_op.results[0]

    # Salva il valore finale dell'accumulatore.
    scalar_env[op.dest] = acc
    result_map[(op.dest,)] = acc  


def _lower_matmul(
    block: Block,
    op: MatMulOp,
    tensor_env: dict[str, list[SSAValue]],
    result_map: Dict[Tuple[Any, ...], SSAValue],
    const_arrays: dict[str, list[int]],
):
    """
    Lowering di C = A * B con tensori flatten in row-major.

    Indici:
      A: (m, k), B: (k, n), C: (m, n)
      C[i,j] = sum_{kk} A[i,kk] * B[kk,j]
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
                a_vals = const_arrays.get(op.lhs)
                b_vals = const_arrays.get(op.rhs)
                if a_vals is not None and b_vals is not None:
                    a_val = a_vals[idx(i, kk, k)]
                    b_val = b_vals[idx(kk, j, n)]
                    if a_val == 0 or b_val == 0:
                        continue

                a_ik = A[idx(i, kk, k)]
                b_kj = B[idx(kk, j, n)]

                mul_op = QMuliOp(a_ik, b_kj)
                block.add_op(mul_op)

                add_op = QAddiOp(acc, mul_op.results[0])
                block.add_op(add_op)
                acc = add_op.results[0]

            # Aggiorna C[i,j] col nuovo SSAValue e registra il risultato.
            C[idx(i, j, n)] = acc
            result_map[(op.dest, i, j)] = acc
