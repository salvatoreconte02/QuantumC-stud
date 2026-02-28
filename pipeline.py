"""CLI driving the QuantumC compilation pipeline."""
from __future__ import annotations
from pprint import pprint

import argparse
import json
import os
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from qiskit import transpile

from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.printer import Printer
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Operation, SSAValue

from step2_ast_to_dataclasses.c_ast import (
    parse_ast,
    TranslationUnit,
    pretty_print_translation_unit,
    FunctionDecl,
    CompoundStmt,
    ForStmt,
    VarDecl,
    InitList,
    ReturnStmt,
    ArrayAccess,
    DeclRef,
    IntegerLiteral,
    BinaryOperator,
)
from step3_dataclasses_to_mlir.mlir_generator import MLIRGenerator
from step4_mlir_to_quantum_mlir.quantum_mlir_generator import generate_quantum_mlir
from step5_quantum_mlir_to_qasm.qasm_generator import (
    generate_circuit,
    export_qasm,
    export_qasm_clifford_t,
    compute_t_count,
    compute_t_depth,
)
from step5_quantum_mlir_to_qasm.q_arithmetics import simulate
from step4_mlir_to_quantum_mlir.quantum_dialect import (
    QuantumInitOp,
    QAddiOp,
    QSubiOp,
    QMuliOp,
    QDivSOp,
)

# Estensioni nostre
from my_extensions.vecmat_lowering import (
    from_c_ast_to_vecmat,
    _match_vec_add_for,
    _match_vec_dot_for,
    _match_matmul_for,
    _match_matmul_for_direct,
)
from my_extensions.vecmat_to_quantum_mlir import from_vecmat_to_quantum_mlir

JSON_DIR = "json_out"
MLIR_DIR = "mlir_out"
QMLIR_DIR = "quantum_mlir_out"
QASM_DIR = "output"


def generate_json_ast(c_path: str) -> str:
    base = os.path.splitext(os.path.basename(c_path))[0]
    os.makedirs(JSON_DIR, exist_ok=True)
    json_path = os.path.join(JSON_DIR, f"{base}.json")
    with open(json_path, "w") as f:
        subprocess.run(
            ["clang", "-Xclang", "-ast-dump=json", "-g", "-fsyntax-only", c_path],
            stdout=f,
            check=True,
        )
    return json_path


def generate_mlir(tu: TranslationUnit) -> ModuleOp:
    """Percorso originale QuantumC: dataclass -> classical MLIR (solo scalari)."""
    generator = MLIRGenerator()
    module = ModuleOp([])
    block = module.body.blocks[0]
    for func in tu.decls:
        block.add_op(generator.generate_function(func))
    return module


def save_module(module: ModuleOp, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        Printer(stream=f).print_op(module)


def filter_out_recognized_vecmat_loops(tu: TranslationUnit, elem_bits: int) -> TranslationUnit:
    """
    Restituisce una copia della Translation Unit dove i loop già riconosciuti come macro-op
    (vec_add / vec_dot / matmul) vengono rimossi dal body delle funzioni.

    Serve per evitare che il percorso "scalare" ricompili anche i loop che
    abbiamo già abbassato come operazioni VecMat.
    """
    tu2 = deepcopy(tu)

    def _flatten_stmts(stmts):
        """Appiattisce liste annidate di statement (alcuni AST le producono)."""
        out = []
        for s in stmts:
            if isinstance(s, list):
                out.extend(_flatten_stmts(s))
            else:
                out.append(s)
        return out

    for decl in tu2.decls:
        if not isinstance(decl, FunctionDecl):
            continue
        if not isinstance(decl.body, CompoundStmt):
            continue

        # normalizza eventuali liste annidate
        decl.body.stmts = _flatten_stmts(decl.body.stmts)

        new_stmts = []
        for s in decl.body.stmts:
            if isinstance(s, ForStmt):
                # Se il loop matcha una macro-op, lo scartiamo.
                if _match_matmul_for(s, elem_bits) is not None:
                    continue
                if _match_matmul_for_direct(s, elem_bits) is not None:
                    continue
                if _match_vec_add_for(s, elem_bits) is not None:
                    continue
                if _match_vec_dot_for(s, elem_bits) is not None:
                    continue
            new_stmts.append(s)

        decl.body.stmts = new_stmts

    return tu2

def filter_out_vecmat_decls(tu: TranslationUnit) -> TranslationUnit:
    """
    Rimuove dal Translation Unit destinato al percorso scalare le dichiarazioni non-scalari
    (vettori/matrici) che il generatore scalare non gestisce correttamente.

    Regole:
      - VarDecl con init = InitList   -> rimuovere (es. int a={1,2};, int A={{..},{..}};)
      - VarDecl con init = None       -> rimuovere (buffer tipo 'int c;' o 'int C;' usato nei loop vec/mat)
    """
    tu2 = deepcopy(tu)

    for decl in tu2.decls:
        if not isinstance(decl, FunctionDecl):
            continue
        if not isinstance(decl.body, CompoundStmt):
            continue

        new_stmts = []
        for s in decl.body.stmts:
            if isinstance(s, VarDecl):
                if isinstance(s.init, InitList):
                    continue
                if s.init is None:
                    continue
            new_stmts.append(s)

        decl.body.stmts = new_stmts

    return tu2


def _ensure_scalar_returns(quantum_module: ModuleOp) -> None:
    """
    Garantisce che ogni FuncOp con tipo di ritorno non-void abbia un ReturnOp.

    Strategia:
      - se manca ReturnOp:
          * cerca l'ultimo SSAValue di tipo i32 prodotto nel blocco e ritorna quello
          * altrimenti crea quantum.init 0 : i32 e ritorna quello
    """
    top: Block = quantum_module.body.blocks[0]

    for fn_op in list(top.ops):
        if not isinstance(fn_op, FuncOp):
            continue

        body = fn_op.body.blocks[0]

        try:
            outputs = list(fn_op.function_type.outputs)
        except Exception:
            continue

        if len(outputs) == 0:
            if not any(isinstance(bop, ReturnOp) for bop in body.ops):
                body.add_op(ReturnOp())
            continue

        if any(isinstance(bop, ReturnOp) for bop in body.ops):
            continue

        ret_val: SSAValue | None = None
        for bop in reversed(list(body.ops)):
            if not getattr(bop, "results", None):
                continue
            for res in reversed(list(bop.results)):
                if getattr(res, "type", None) == i32:
                    ret_val = res
                    break
            if ret_val is not None:
                break

        if ret_val is None:
            init0 = QuantumInitOp(0, i32)
            body.add_op(init0)
            ret_val = init0.results[0]

        body.add_op(ReturnOp(ret_val))



####### Return hint extraction ##########

@dataclass(frozen=True)
class ReturnHint:
    # "vec" or "mat"
    kind: str
    # destination symbol in C (e.g., "c" or "C")
    name: str
    # linearized index into sinks (row-major for matrices)
    idx: int
    # debug/extra
    index: int | None = None
    row: int | None = None
    col: int | None = None
    n: int | None = None


@dataclass(frozen=True)
class HybridReturnHint:
    """Return che combina scalare + elemento array, es. `return z + C[1][1]`"""
    opcode: str           # "+", "-", "*", "/"
    scalar_side: str      # "lhs" o "rhs" (quale lato è lo scalare)
    array_hint: ReturnHint  # hint per l'elemento array


def _unwrap_int(x: object) -> int | None:
    """
    Converte un valore potenzialmente wrappato (es. .data) in int.
    Ritorna None se non convertibile.
    """
    try:
        if hasattr(x, "data"):
            return int(getattr(x, "data"))
        return int(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _declref_name(dr: DeclRef) -> str | None:
    try:
        n = dr.name
        if hasattr(n, "data"):
            return str(n.data)
        return str(n)
    except Exception:
        return None


def _int_lit_value(il: IntegerLiteral) -> int | None:
    try:
        return _unwrap_int(il.value)
    except Exception:
        return None


def _get_matmul_n_for_dest(vecmat_module, dest: str, fn_name: str) -> int | None:
    """
    Determina n (numero colonne di C) per una MatMulOp che produce 'dest'.
    Preferenze:
      1) const_shapes della rhs (B): (k, n) => n = shape[1]
      2) campo op.n
    """
    try:
        for f in getattr(vecmat_module, "functions", []):
            if getattr(f, "name", None) != fn_name:
                continue
            for op in getattr(f, "ops", []):
                if getattr(op, "dest", None) != dest:
                    continue
                rhs = getattr(op, "rhs", None)
                if rhs is not None:
                    shp = getattr(vecmat_module, "const_shapes", {}).get(rhs)
                    if shp is not None and len(shp) == 2:
                        n = _unwrap_int(shp[1])
                        if n is not None:
                            return n
                if hasattr(op, "n"):
                    n = _unwrap_int(getattr(op, "n"))
                    if n is not None:
                        return n
    except Exception:
        return None
    return None


def _is_array_access(expr) -> bool:
    """Controlla se l'espressione è un accesso ad array (1D o 2D)."""
    if not isinstance(expr, ArrayAccess):
        return False
    # c[i] - array 1D
    if isinstance(expr.array, DeclRef) and isinstance(expr.index, IntegerLiteral):
        return True
    # C[i][j] - array 2D
    if (
        isinstance(expr.array, ArrayAccess)
        and isinstance(expr.array.array, DeclRef)
        and isinstance(expr.array.index, IntegerLiteral)
        and isinstance(expr.index, IntegerLiteral)
    ):
        return True
    return False


def _is_scalar_expr(expr) -> bool:
    """Controlla se l'espressione è scalare (non un accesso ad array)."""
    # DeclRef semplice (variabile scalare)
    if isinstance(expr, DeclRef):
        return True
    # Letterale intero
    if isinstance(expr, IntegerLiteral):
        return True
    # Operazione binaria su scalari
    if isinstance(expr, BinaryOperator):
        return _is_scalar_expr(expr.lhs) and _is_scalar_expr(expr.rhs)
    return False


def _extract_array_hint_from_access(
    expr: ArrayAccess,
    vecmat_module,
    fn_name: str,
) -> ReturnHint | None:
    """Estrae un ReturnHint da un ArrayAccess (1D o 2D)."""
    # c[i] - array 1D
    if isinstance(expr.array, DeclRef) and isinstance(expr.index, IntegerLiteral):
        base = _declref_name(expr.array)
        idx0 = _int_lit_value(expr.index)
        if base is None or idx0 is None:
            return None
        return ReturnHint(kind="vec", name=base, idx=idx0, index=idx0)

    # C[i][j] - array 2D
    if (
        isinstance(expr.array, ArrayAccess)
        and isinstance(expr.array.array, DeclRef)
        and isinstance(expr.array.index, IntegerLiteral)
        and isinstance(expr.index, IntegerLiteral)
    ):
        base = _declref_name(expr.array.array)
        row = _int_lit_value(expr.array.index)
        col = _int_lit_value(expr.index)
        if base is None or row is None or col is None:
            return None

        n = _get_matmul_n_for_dest(vecmat_module, base, fn_name)
        if n is None or n <= 0:
            return None

        idx = row * n + col
        return ReturnHint(kind="mat", name=base, idx=idx, row=row, col=col, n=n)

    return None


def _extract_return_hint_for_function(
    fn: FunctionDecl,
    vecmat_module,
) -> ReturnHint | HybridReturnHint | None:
    """
    Estrae un hint dal return C:
      - return c[<const>]
      - return C[<const>][<const>]
      - return <scalar> + c[<const>]   (HybridReturnHint)
      - return C[<const>][<const>] + <scalar>  (HybridReturnHint)
    """
    if not isinstance(fn.body, CompoundStmt):
        return None

    ret_stmt: ReturnStmt | None = None
    for s in fn.body.stmts:
        if isinstance(s, ReturnStmt):
            ret_stmt = s  # ultimo return trovato
    if ret_stmt is None:
        return None

    v = ret_stmt.value

    # return c[i]
    if isinstance(v, ArrayAccess) and isinstance(v.array, DeclRef) and isinstance(v.index, IntegerLiteral):
        base = _declref_name(v.array)
        idx0 = _int_lit_value(v.index)
        if base is None or idx0 is None:
            return None
        return ReturnHint(kind="vec", name=base, idx=idx0, index=idx0)

    # return C[i][j]
    if (
        isinstance(v, ArrayAccess)
        and isinstance(v.array, ArrayAccess)
        and isinstance(v.array.array, DeclRef)
        and isinstance(v.array.index, IntegerLiteral)
        and isinstance(v.index, IntegerLiteral)
    ):
        base = _declref_name(v.array.array)
        row = _int_lit_value(v.array.index)
        col = _int_lit_value(v.index)
        if base is None or row is None or col is None:
            return None

        n = _get_matmul_n_for_dest(vecmat_module, base, fn.name)
        if n is None or n <= 0:
            # senza n non si può linearizzare stabilmente
            return None

        idx = row * n + col
        return ReturnHint(kind="mat", name=base, idx=idx, row=row, col=col, n=n)

    # return <scalar> OP <array[...]> oppure <array[...]> OP <scalar>
    if isinstance(v, BinaryOperator):
        lhs, rhs = v.lhs, v.rhs
        opcode = v.opcode

        # Caso: return scalar + C[i][j] (o c[i])
        if _is_scalar_expr(lhs) and _is_array_access(rhs):
            array_hint = _extract_array_hint_from_access(rhs, vecmat_module, fn.name)
            if array_hint is not None:
                return HybridReturnHint(opcode=opcode, scalar_side="lhs", array_hint=array_hint)

        # Caso: return C[i][j] + scalar (o c[i] + scalar)
        if _is_array_access(lhs) and _is_scalar_expr(rhs):
            array_hint = _extract_array_hint_from_access(lhs, vecmat_module, fn.name)
            if array_hint is not None:
                return HybridReturnHint(opcode=opcode, scalar_side="rhs", array_hint=array_hint)

    return None


def _extract_return_hints(tu: TranslationUnit, vecmat_module) -> dict[str, ReturnHint | HybridReturnHint]:
    hints: dict[str, ReturnHint | HybridReturnHint] = {}
    for decl in tu.decls:
        if not isinstance(decl, FunctionDecl):
            continue
        h = _extract_return_hint_for_function(decl, vecmat_module)
        if h is not None:
            hints[decl.name] = h
    return hints


#### Merge #####

def _merge_scalar_and_vec_quantum(
    scalar_quantum_module: ModuleOp,
    vec_quantum_module: ModuleOp,
    return_hints: dict[str, ReturnHint | HybridReturnHint] | None = None,
) -> ModuleOp:
    """
    Unisce il modulo quantum "scalare" con quello "vec/mat" (macro-op).

    Copia (clone) le operazioni del vec module dentro la funzione scalare corrispondente,
    rimappando gli SSAValue. Se il return della funzione scalare è un placeholder (0),
    lo sostituisce con il valore corretto preso da result_map o da euristiche.
    Supporta anche un return ibrido (scalare op array) se specificato in return_hints.
    """
    return_hints = return_hints or {}

    scalar_top: Block = scalar_quantum_module.body.blocks[0]
    vec_top: Block = vec_quantum_module.body.blocks[0]

    scalar_funcs: dict[str, FuncOp] = {
        op.sym_name.data: op for op in scalar_top.ops if isinstance(op, FuncOp)
    }

    # result_map collega (nome, indici) allo SSAValue finale generato nel lowering vec/mat.
    # Lo usiamo per scegliere in modo deterministico cosa ritornare (es. C[i][j]), senza euristiche fragili.
    vec_result_map = getattr(vec_quantum_module, "result_map", None)

    for vec_func in vec_top.ops:
        if not isinstance(vec_func, FuncOp):
            continue

        fname = vec_func.sym_name.data
        if fname not in scalar_funcs:
            continue

        scalar_func = scalar_funcs[fname]
        scalar_body: Block = scalar_func.body.blocks[0]
        vec_body: Block = vec_func.body.blocks[0]

        return_anchor: ReturnOp | None = None
        for op in scalar_body.ops:
            if isinstance(op, ReturnOp):
                return_anchor = op
                break

        prologue_anchor: Operation | None = None
        for op in scalar_body.ops:
            if isinstance(op, ReturnOp):
                continue
            if not isinstance(op, QuantumInitOp):
                prologue_anchor = op
                break
        if prologue_anchor is None:
            prologue_anchor = return_anchor

        vec_inits: list[Operation] = []
        vec_others: list[Operation] = []
        for op in vec_body.ops:
            if isinstance(op, ReturnOp):
                continue
            if isinstance(op, QuantumInitOp):
                vec_inits.append(op)
            else:
                vec_others.append(op)

        # Mappa SSA che associa ogni risultato del vec module
        # al corrispondente SSAValue clonato nella funzione scalare.
        # Serve per sostituire correttamente gli operandi dopo il clone.
        ssa_map: dict[SSAValue, SSAValue] = {}

        def insert_before(anchor: Operation | None, op_to_insert: Operation) -> None:
            if anchor is None:
                scalar_body.add_op(op_to_insert)
            else:
                scalar_body.insert_op_before(op_to_insert, anchor)

        def remap(v: SSAValue) -> SSAValue:
            return ssa_map.get(v, v)

        for init_op in vec_inits:
            new_init = init_op.clone()
            insert_before(prologue_anchor, new_init)
            if init_op.results and new_init.results:
                ssa_map[init_op.results[0]] = new_init.results[0]

        produced_results: list[SSAValue] = []
        for op in vec_others:
            new_op = op.clone()
            new_op.operands = [remap(o) for o in new_op.operands]
            insert_before(return_anchor, new_op)

            for old_res, new_res in zip(op.results, new_op.results):
                ssa_map[old_res] = new_res
                produced_results.append(new_res)

        # Se il return scalare è "0" , ovvero un placeholder, lo rimpiazziamo col risultato vec/mat.
        # Se non è placeholder, lo lasciamo (a meno di un return ibrido esplicito).
        if return_anchor is not None and return_anchor.operands:
            ret_val = return_anchor.operands[0]
            is_placeholder_zero = False

            if isinstance(ret_val.owner, QuantumInitOp):
                try:
                    is_placeholder_zero = int(ret_val.owner.value.value.data) == 0
                except Exception:
                    is_placeholder_zero = False

            hint = return_hints.get(fname)

            # Caso return ibrido: combiniamo il valore scalare già calcolato con un elemento di array
            # preso da result_map (es. z + C[1][1]), secondo la hint.
            if isinstance(hint, HybridReturnHint) and not is_placeholder_zero:
                # Il return scalare ha già il valore di z (scalare)
                scalar_val = ret_val

                # Trova il valore array dal result_map
                array_hint = hint.array_hint
                array_val: SSAValue | None = None
                if vec_result_map is not None:
                    try:
                        if array_hint.kind == "vec" and array_hint.index is not None:
                            key = (array_hint.name, array_hint.index)
                            if key in vec_result_map:
                                array_val = remap(vec_result_map[key])
                        elif array_hint.kind == "mat" and array_hint.row is not None and array_hint.col is not None:
                            key = (array_hint.name, array_hint.row, array_hint.col)
                            if key in vec_result_map:
                                array_val = remap(vec_result_map[key])
                    except Exception:
                        array_val = None

                if array_val is not None:
                    # Determina ordine operandi
                    if hint.scalar_side == "lhs":
                        lhs_val, rhs_val = scalar_val, array_val
                    else:
                        lhs_val, rhs_val = array_val, scalar_val

                    # Emetti l'operazione quantum corrispondente
                    new_op: Operation | None = None
                    if hint.opcode == "+":
                        new_op = QAddiOp(lhs_val, rhs_val)
                    elif hint.opcode == "-":
                        new_op = QSubiOp(lhs_val, rhs_val)
                    elif hint.opcode == "*":
                        new_op = QMuliOp(lhs_val, rhs_val)
                    elif hint.opcode == "/":
                        new_op = QDivSOp(lhs_val, rhs_val)

                    if new_op is not None:
                        insert_before(return_anchor, new_op)
                        return_anchor.operands = [new_op.results[0]]

            elif is_placeholder_zero:
                chosen: SSAValue | None = None

                if hint is not None and vec_result_map is not None and isinstance(hint, ReturnHint):
                    try:
                        if hint.kind == "vec" and hint.index is not None:
                            key = (hint.name, hint.index)
                            if key in vec_result_map:
                                chosen = remap(vec_result_map[key])
                        elif hint.kind == "mat" and hint.row is not None and hint.col is not None:
                            key = (hint.name, hint.row, hint.col)
                            if key in vec_result_map:
                                chosen = remap(vec_result_map[key])
                    except Exception:
                        chosen = None
                # Fallback: se non abbiamo hint/result_map, prova a scegliere un "sink" (valore non usato dopo).
                if chosen is None and produced_results:
                    sinks: list[SSAValue] = []
                    for r in produced_results:
                        try:
                            if len(list(r.uses)) == 0:
                                sinks.append(r)
                        except Exception:
                            pass

                    if hint is not None and isinstance(hint, ReturnHint) and sinks:
                        if 0 <= hint.idx < len(sinks):
                            chosen = sinks[hint.idx]

                    if chosen is None:
                        if sinks:
                            chosen = sinks[0]
                        else:
                            chosen = produced_results[-1]

                if chosen is not None:
                    return_anchor.operands = [chosen]

    return scalar_quantum_module


######### Metrics + scoring ########## da togliere gli score???

@dataclass(frozen=True)
class CircuitMetrics:
    """Metriche di un circuito dopo transpile su base Clifford+T"""
    mode: str
    num_qubits: int
    depth: int
    size: int
    count_ops_total: int
    count_cx: int
    count_t: int
    count_tdg: int
    count_h: int
    count_p: int
    count_rz: int
    # Metriche Clifford+T (T-count considera decomposizione Solovay-Kitaev)
    t_count: int              # T-count totale (esatto + approssimato)
    t_count_exact: int        # T-count da decomposizioni esatte (T, Tdg, Toffoli)
    t_count_approx: int       # T-count da approssimazioni (rotazioni arbitrarie)
    t_depth: int              # T-depth: numero di strati con T-gates
    arbitrary_rotations: int  # Numero di rotazioni che richiedono approssimazione
    score: float


def _compute_metrics_and_score(qc) -> CircuitMetrics:
    """
    Transpila il circuito su basis Clifford+T (+Rz) e calcola metriche/score.

    Stima T-count:
      - T/Tdg contano 1 ciascuno
      - Rz multipli di pi/4 contano come 1 T
      - Rz non-Clifford vengono approssimati con un costo fisso (default 150 T)
    """
    
    ###### Transpile a base Clifford+T  #########
    
    CLIFFORD_T_BASIS = ['h', 's', 'sdg', 't', 'tdg', 'cx', 'rz']
    transpiled = transpile(qc, basis_gates=CLIFFORD_T_BASIS, optimization_level=3)

    
    ###### Calcola metriche dal circuito transpilato #######
   
    num_qubits = transpiled.num_qubits
    depth = int(transpiled.depth())
    size = int(transpiled.size())

    counts = transpiled.count_ops()
    count_ops_total = int(sum(int(v) for v in counts.values()))
    count_cx = int(counts.get("cx", 0))
    count_t = int(counts.get("t", 0))
    count_tdg = int(counts.get("tdg", 0))
    count_h = int(counts.get("h", 0))
    count_p = int(counts.get("p", 0))
    count_rz = int(counts.get("rz", 0))

    
    ###### Calcola T-count analizzando i gate Rz ########
    
    T_GATES_PER_ARBITRARY_ROTATION = 150

    def _is_clifford_angle(angle: float, tol: float = 1e-10) -> bool:
        """Multiplo di pi/2 → Clifford (S, Z, I) → 0 T"""
        normalized = angle % (2 * np.pi)
        remainder = normalized % (np.pi / 2)
        return remainder < tol or (np.pi / 2 - remainder) < tol

    def _is_t_angle(angle: float, tol: float = 1e-10) -> bool:
        """Multiplo di pi/4 → T gate → 1 T"""
        normalized = angle % (2 * np.pi)
        remainder = normalized % (np.pi / 4)
        return remainder < tol or (np.pi / 4 - remainder) < tol

    #Conta T-gates espliciti 
    t_count_exact = count_t + count_tdg

    # Analizza i gate Rz 
    rz_clifford = 0      # multipli di pi/2 → 0 T
    rz_t_angle = 0       # multipli di pi/4 → 1 T
    rz_arbitrary = 0     # altri angoli → 150 T

    for inst in transpiled.data:
        if inst.operation.name == 'rz':
            angle = float(inst.operation.params[0])
            if _is_clifford_angle(angle):
                rz_clifford += 1
            elif _is_t_angle(angle):
                rz_t_angle += 1
            else:
                rz_arbitrary += 1

    t_count_exact += rz_t_angle
    t_count_approx = rz_arbitrary * T_GATES_PER_ARBITRARY_ROTATION
    t_count = t_count_exact + t_count_approx
    arbitrary_rotations = rz_arbitrary

    # T-depth: profondità effettiva delle T-gates, calcolata sul DAG del circuito.
    # Include T/Tdg e Rz non-Clifford.
    t_depth = compute_t_depth(transpiled)

    # Score basato principalmente su T-count
    score = (
        1.00 * num_qubits +
        0.02 * depth +
        0.001 * t_count +
        0.10 * count_cx +
        0.01 * count_h
    )

    return CircuitMetrics(
        mode="",
        num_qubits=num_qubits,
        depth=depth,
        size=size,
        count_ops_total=count_ops_total,
        count_cx=count_cx,
        count_t=count_t,
        count_tdg=count_tdg,
        count_h=count_h,
        count_p=count_p,
        count_rz=count_rz,
        t_count=t_count,
        t_count_exact=t_count_exact,
        t_count_approx=t_count_approx,
        t_depth=t_depth,
        arbitrary_rotations=arbitrary_rotations,
        score=float(score),
    )


def _print_metrics(m: CircuitMetrics) -> None:
    print("\n=== Circuit metrics ===")
    print(f"mode        : {m.mode}")
    print(f"num_qubits  : {m.num_qubits}")
    print(f"depth       : {m.depth}")
    print(f"size        : {m.size}")
    print(f"ops_total   : {m.count_ops_total}")
    print(f"cx          : {m.count_cx}")
    print(f"t           : {m.count_t}")
    print(f"tdg         : {m.count_tdg}")
    print(f"h           : {m.count_h}")
    print(f"p           : {m.count_p}")
    print(f"rz          : {m.count_rz}")
    print("--- Clifford+T metrics ---")
    print(f"T-count     : {m.t_count} (total)")
    print(f"  exact     : {m.t_count_exact} (from T/Tdg/Toffoli)")
    print(f"  approx    : {m.t_count_approx} (from arbitrary rotations)")
    print(f"T-depth     : {m.t_depth} (layers with T-gates)")
    print(f"  arb. rot. : {m.arbitrary_rotations} (rotations needing ~150 T each)")
    print(f"score       : {m.score:.3f} (lower is better)")


def _print_comparison(a: CircuitMetrics, b: CircuitMetrics) -> None:
    print("\n=== Comparison ===")
    better = a if a.score <= b.score else b
    worse = b if better is a else a
    print(f"Better score: {better.mode} ({better.score:.3f})")
    print(f"Worse score : {worse.mode} ({worse.score:.3f})")
    print("Note: lo score è una regola euristica interna basata su qubit/depth/porte.")


def compile_c_file(
    c_file: str,
    num_bits: int = 16,
    verbose: bool = False,
    pretty: bool = False,
    run: bool = False,
    adder: str = "qft",  # "qft" | "ripple" | "both"
) -> str:
    base = os.path.splitext(os.path.basename(c_file))[0]

    json_path = generate_json_ast(c_file)
    with open(json_path) as f:
        ast_json = json.load(f)

    tu = parse_ast(ast_json)

    if pretty:
        print("=== C Code ===")
        print(pretty_print_translation_unit(tu))
        print("================================")

    print("=== TranslationUnit (dataclass) ===")
    pprint(tu)

    vecmat_module = from_c_ast_to_vecmat(tu, num_bits)
    print("=== VecMatModule (dialetto vettoriale/matriciale) ===")
    pprint(vecmat_module)

    return_hints = _extract_return_hints(tu, vecmat_module)

    has_vecmat_ops = any(func.ops for func in vecmat_module.functions)
    has_vecmat_consts = bool(getattr(vecmat_module, "const_arrays", {}) or {})
    enable_vecmat_path = has_vecmat_ops or has_vecmat_consts

    if enable_vecmat_path and has_vecmat_ops:
        tu_filtered = filter_out_recognized_vecmat_loops(tu, num_bits)
        tu_filtered = filter_out_vecmat_decls(tu_filtered)
        mlir_module = generate_mlir(tu_filtered)
    else:
        mlir_module = generate_mlir(tu)

    classical_path = os.path.join(MLIR_DIR, f"{base}_classical.mlir")
    save_module(mlir_module, classical_path)

    scalar_quantum_module = generate_quantum_mlir(mlir_module)
    _ensure_scalar_returns(scalar_quantum_module)

    scalar_quantum_path = os.path.join(QMLIR_DIR, f"{base}_quantum_scalar.mlir")
    save_module(scalar_quantum_module, scalar_quantum_path)

    if enable_vecmat_path:
        if has_vecmat_ops:
            print("Rilevate macro-op vettoriali/matriciali: VecMat → quantum.")
        else:
            print(" Nessuna macro-op, ma presenti inizializzazioni costanti: VecMat → quantum (solo init).")

        print("VecMat const_arrays:", getattr(vecmat_module, "const_arrays", None))
        print("VecMat const_shapes:", getattr(vecmat_module, "const_shapes", None))

        vec_quantum_module = from_vecmat_to_quantum_mlir(vecmat_module, num_bits=num_bits)
        vec_quantum_path = os.path.join(QMLIR_DIR, f"{base}_quantum_vec.mlir")
        save_module(vec_quantum_module, vec_quantum_path)

        quantum_module = _merge_scalar_and_vec_quantum(
            scalar_quantum_module,
            vec_quantum_module,
            return_hints=return_hints,
        )
    else:
        print("Nessuna macro-op vettoriale/matriciale e nessuna inizializzazione costante: uso solo il percorso scalare.")
        quantum_module = scalar_quantum_module

    _ensure_scalar_returns(quantum_module)

    quantum_path = os.path.join(QMLIR_DIR, f"{base}_quantum_final.mlir")
    save_module(quantum_module, quantum_path)

    def _build(mode: str):
        circuit = generate_circuit(quantum_module, num_bits=num_bits, verbose=verbose, arithmetic_mode=mode)

        # Metriche sempre disponibili sul circuito "as built"
        m_raw = _compute_metrics_and_score(circuit)
        m_raw = CircuitMetrics(mode=mode, **{k: getattr(m_raw, k) for k in m_raw.__dataclass_fields__ if k != "mode"})

        # Transpile verso un gate set UNIFORME per confronto equo tra backend.
        # Usiamo {cx, rz, sx, x} che è un basis universale standard:
        # - cx: unico gate a 2 qubit (permette confronto diretto)
        # - rz, sx, x: gate single-qubit sufficienti per universalità
        # Questo decompone sia i 'cp' del QFT che i 'ccx' del ripple-carry.
        circuit_nisq = None
        try:
            from qiskit import transpile as _transpile
            uniform_basis = ["cx", "rz", "sx", "x", "measure"]
            circuit_nisq = _transpile(circuit, basis_gates=uniform_basis, optimization_level=1)

            m_nisq = _compute_metrics_and_score(circuit_nisq)
            m_nisq = CircuitMetrics(mode=mode, **{k: getattr(m_nisq, k) for k in m_nisq.__dataclass_fields__ if k != "mode"})
            return circuit, circuit_nisq, m_nisq

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\n[WARN] Transpile NISQ fallito o troppo costoso per mode={mode}.")
            print(f"       Uso metriche sul circuito non-transpilato. Dettaglio: {type(e).__name__}: {e}")
            return circuit, None, m_raw
    if adder not in ("qft", "ripple", "both"):
        raise ValueError("adder must be one of: qft, ripple, both")

    if adder == "both":
        circuit_qft, _circuit_qft_nisq, m_qft = _build("qft")
        circuit_rip, _circuit_rip_nisq, m_rip = _build("ripple")

        _print_metrics(m_qft)
        _print_metrics(m_rip)
        _print_comparison(m_qft, m_rip)

        qasm_path_qft = os.path.join(QASM_DIR, f"{base}_qft.qasm")
        qasm_path_rip = os.path.join(QASM_DIR, f"{base}_ripple.qasm")

        if run:
            export_qasm(circuit_qft, qasm_path_qft)
            export_qasm(circuit_rip, qasm_path_rip)
        else:
            export_qasm_clifford_t(circuit_qft, qasm_path_qft)
            export_qasm_clifford_t(circuit_rip, qasm_path_rip)

        return qasm_path_qft

    else:
        circuit, _circuit_nisq, m = _build(adder)
        _print_metrics(m)

        qasm_path = os.path.join(QASM_DIR, f"{base}_{adder}.qasm")
        if run:
            export_qasm(circuit, qasm_path)
        else:
            export_qasm_clifford_t(circuit, qasm_path)

        return qasm_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile C code to QASM and optionally simulate.")
    parser.add_argument("c_file", nargs="?", default=os.path.join("c_code", "try.c"), help="Path to the C file")
    parser.add_argument("--run", action="store_true", help="Run the resulting QASM file with simulation")
    parser.add_argument("--bits", type=int, default=16, help="Number of bits used for the quantum representation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output during circuit generation")
    parser.add_argument("--pretty", action="store_true", help="Print the parsed C code from the AST")
    parser.add_argument("--time", action="store_true", help="Print total compilation + simulation time")

    parser.add_argument(
        "--adder",
        choices=["qft", "ripple", "both"],
        default="qft",
        help="Choose arithmetic backend: qft, ripple, or both (runs both and prints comparison).",
    )

    args = parser.parse_args()

    start = time.time() if args.time else None

    qasm_path = compile_c_file(
        args.c_file,
        num_bits=args.bits,
        verbose=args.verbose,
        pretty=args.pretty,
        run=args.run,
        adder=args.adder,
    )

    if args.run:
        from qiskit import QuantumCircuit
        print(f"Running simulation for {qasm_path} ...")
        qc = QuantumCircuit.from_qasm_file(qasm_path)
        simulate(qc)

    if args.time:
        elapsed = time.time() - start
        print(f"\n[Pipeline completed in {elapsed:.2f} seconds]")


if __name__ == "__main__":
    main()