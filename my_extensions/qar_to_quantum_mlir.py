# my_extensions/qar_to_quantum_mlir.py

"""
Lowering da QAR (QarModule) al quantum MLIR di QuantumC.

ATTENZIONE: questo pass non "esegue" le operazioni numeriche su dati reali.
Come QuantumC, costruisce SOLO la STRUTTURA del circuito quantistico:

  - quante operazioni servono,
  - come sono composte (pattern di add/mul),
  - come cresce la complessità al variare di length, m, n, k.

I registri inizializzati con QuantumInitOp(0) sono PLACEHOLDER:
non rappresentano il valore 0 del programma C, ma slot quantistici
su cui agiscono le primitive (addi, muli, ecc.).

Rispetto alla versione precedente, qui facciamo anche una
inizializzazione "simbolica" dei vettori/matrici:
  - per ogni nome vettoriale (a, b, c, ...) creiamo una lista di registri,
  - per ogni scalare (s, ...) creiamo un registro,
e poi usiamo questi registri nelle op QAR (dot, map_add, ...).
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple

from xdsl.dialects.builtin import ModuleOp, i32, IntegerAttr
from xdsl.dialects.func import FuncOp
from xdsl.ir import Block, Region, SSAValue

from my_extensions.qar_ir import (
    QarModule,
    QarFunction,
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
# Analisi del QarModule: raccolta info su vettori e scalari
# ---------------------------------------------------------------------------

def _collect_qaar_symbols(qar_module: QarModule) -> tuple[dict[str, tuple[int, int]], set[str]]:
    """
    Scansiona il QarModule e raccoglie:
      - vec_info: mappa nome -> (max_length, elem_bits) per vettori/matrici
      - scalar_names: insieme di nomi scalari (es. 's' del dot)
    """
    vec_info: dict[str, tuple[int, int]] = {}
    scalar_names: set[str] = set()

    for fn in qar_module.functions:
        for op in fn.ops:
            if isinstance(op, QarMapAdd):
                # dest, lhs, rhs sono vettori
                for name in (op.dest, op.lhs, op.rhs):
                    old = vec_info.get(name)
                    L = op.length
                    if old is None or L > old[0]:
                        vec_info[name] = (L, op.elem_bits)

            elif isinstance(op, QarDot):
                # dest è scalare, lhs/rhs sono vettori
                scalar_names.add(op.dest)
                for name in (op.lhs, op.rhs):
                    old = vec_info.get(name)
                    L = op.length
                    if old is None or L > old[0]:
                        vec_info[name] = (L, op.elem_bits)

            elif isinstance(op, QarMatMul):
                # In futuro puoi usare qui info su matrici
                # per esempio memorizzare dimensioni, ecc.
                pass

    return vec_info, scalar_names


# ---------------------------------------------------------------------------
# Entry point: QarModule -> ModuleOp (quantum dialect)
# ---------------------------------------------------------------------------

def from_qar_to_quantum_mlir(qar_module: QarModule, num_bits: int = 16) -> ModuleOp:
    """
    Entry point usato dalla pipeline:

        QarModule  ->  ModuleOp (quantum dialect)

    Costruisce un ModuleOp con dentro una funzione @main che:
      - in un prologo inizializza i registri per tutti i vettori e scalari QAR,
      - poi emette le operazioni quantistiche corrispondenti a QarDot, QarMapAdd, ecc.

    Il parametro num_bits è mantenuto per coerenza con il resto della pipeline,
    ma al momento non viene usato in questo pass.
    """

    # 1) Raccogli informazioni su vettori e scalari usati nel QarModule
    vec_info, scalar_names = _collect_qaar_symbols(qar_module)

    # 2) Crea il ModuleOp esterno
    module = ModuleOp([])

    # 3) Crea la funzione @main senza argomenti.
    #    Tipo semplificato: ([], []) perché qui non imponiamo un tipo di ritorno;
    #    il backend guarda solo le operazioni nel blocco.
    entry_block = Block()
    func_region = Region([entry_block])
    func_type = ([], [])
    func = FuncOp("main", func_type, func_region)

    # Inserisci la func nel modulo
    module.body.blocks[0].add_op(func)

    # 4) Ambienti per vettori e scalari
    vec_env: Dict[str, List[SSAValue]] = {}
    scalar_env: Dict[str, SSAValue] = {}

    # 5) PROLOGO: inizializza i registri quantistici per vettori e scalari
    _emit_vector_and_scalar_inits(
        entry_block,
        vec_info,
        scalar_names,
        vec_env,
        scalar_env,
        qar_module,
    )

    # 6) Emissione delle operazioni QAR come quantum ops
    _emit_qar_ops(entry_block, qar_module, vec_env, scalar_env)

    # NOTA: qui NON aggiungiamo ReturnOp.
    #  - Nel caso ibrido, il ReturnOp viene dal percorso scalare e
    #    la merge-pipeline lo gestisce.
    #  - Nel caso puramente vettoriale, il QASM rappresenta solo lo
    #    schema di operazioni (nessuna misura obbligatoria).

    return module


# ---------------------------------------------------------------------------
# PROLOGO: inizializzazione dei registri per vettori e scalari
# ---------------------------------------------------------------------------

def _emit_vector_and_scalar_inits(
    block: Block,
    vec_info: dict[str, tuple[int, int]],
    scalar_names: set[str],
    vec_env: dict[str, list[SSAValue]],
    scalar_env: dict[str, SSAValue],
    qar_module: QarModule,
):
    const_arrays = getattr(qar_module, "const_arrays", {}) or {}
    const_shapes = getattr(qar_module, "const_shapes", {}) or {}

    # Vettori / matrici (flatten)
    for name, (length, elem_bits) in vec_info.items():
        regs: list[SSAValue] = []
        init_vals = const_arrays.get(name)

        # se c'è shape, usa quello per il numero di elementi
        shape = const_shapes.get(name)
        total = length
        if shape is not None:
            total = 1
            for d in shape:
                total *= d

        for idx in range(total):
            if init_vals is not None and idx < len(init_vals):
                v = int(init_vals[idx])
            else:
                v = 0

            init_op = QuantumInitOp(v, i32)
            block.add_op(init_op)
            init_op.results[0].name_hint = f"q_{name}_{idx}"
            regs.append(init_op.results[0])

        vec_env[name] = regs

    # Scalari
    for name in scalar_names:
        init_op = QuantumInitOp(0, i32)
        block.add_op(init_op)
        init_op.results[0].name_hint = f"q_{name}"
        scalar_env[name] = init_op.results[0]




# ---------------------------------------------------------------------------
# Lowering delle op QAR in operazioni del quantum dialect
# ---------------------------------------------------------------------------

def _emit_qar_ops(
    block: Block,
    qar_module: QarModule,
    vec_env: dict[str, list[SSAValue]],
    scalar_env: dict[str, SSAValue],
):
    """
    Per ogni op QAR genera la corrispondente sequenza di operazioni
    nel dialetto quantistico, riusando i registri di vec_env/scalar_env.
    """
    for fn in qar_module.functions:
        for op in fn.ops:
            if isinstance(op, QarMapAdd):
                _lower_vec_add(block, op, vec_env)
            elif isinstance(op, QarDot):
                _lower_vec_dot(block, op, vec_env, scalar_env)
            elif isinstance(op, QarMatMul):
                # TODO: in futuro puoi implementare qui anche il matmul
                pass
            else:
                # se in futuro aggiungi altre op QAR, gestiscile qui
                pass


def _lower_vec_add(
    block: Block,
    op: QarMapAdd,
    vec_env: dict[str, list[SSAValue]],
):
    """
    QarMapAdd(dest, lhs, rhs, length, elem_bits)
    implementata come:
        dest[i] = lhs[i] + rhs[i]
    usando QAddiOp per ogni elemento.
    """
    dest_regs = vec_env[op.dest]
    lhs_regs = vec_env[op.lhs]
    rhs_regs = vec_env[op.rhs]

    for i in range(op.length):
        lhs_q = lhs_regs[i]
        rhs_q = rhs_regs[i]

        # QAddiOp(lhs, rhs) come definito nel quantum_dialect
        add_op = QAddiOp(lhs_q, rhs_q)
        block.add_op(add_op)

        # aggiorna il registro di destinazione per questo elemento
        dest_regs[i] = add_op.results[0]


def _lower_vec_dot(
    block: Block,
    op: QarDot,
    vec_env: dict[str, list[SSAValue]],
    scalar_env: dict[str, SSAValue],
):
    """
    QarDot(dest='s', lhs='a', rhs='b', length, elem_bits)
    implementata come:
        s = 0 (già inizializzato nel prologo)
        per i in [0, length):
            tmp = a[i] * b[i]
            s   = s + tmp
    """
    a_regs = vec_env[op.lhs]
    b_regs = vec_env[op.rhs]

    acc = scalar_env[op.dest]

    for i in range(op.length):
        lhs_q = a_regs[i]
        rhs_q = b_regs[i]

        # 1) moltiplicazione: tmp = a[i] * b[i]
        mul_op = QMuliOp(lhs_q, rhs_q)
        block.add_op(mul_op)
        tmp_q = mul_op.results[0]

        # 2) accumulo: acc = acc + tmp
        add_op = QAddiOp(acc, tmp_q)
        block.add_op(add_op)
        acc = add_op.results[0]

    # aggiorna l'env, così se in futuro s viene riusato, hai l'ultima versione
    scalar_env[op.dest] = acc
