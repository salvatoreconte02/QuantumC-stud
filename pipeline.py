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
)
from step3_dataclasses_to_mlir.mlir_generator import MLIRGenerator
from step4_mlir_to_quantum_mlir.quantum_mlir_generator import generate_quantum_mlir
from step5_quantum_mlir_to_qasm.qasm_generator import (
    generate_circuit,
    export_qasm,
    export_qasm_clifford_t,
)
from step5_quantum_mlir_to_qasm.q_arithmetics import simulate
from step4_mlir_to_quantum_mlir.quantum_dialect import QuantumInitOp

# Estensioni nostre
from my_extensions.vecmat_lowering import (
    from_c_ast_to_vecmat,
    _match_vec_add_for,
    _match_vec_dot_for,
    _match_matmul_for,
)
from my_extensions.vecmat_to_qar import from_vecmat_to_qar
from my_extensions.qar_to_quantum_mlir import from_qar_to_quantum_mlir

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
    Mantiene l'ibrido evitando però che il percorso scalare compili anche i loop
    già riconosciuti come macro-op (vec_add / vec_dot / matmul).
    """
    tu2 = deepcopy(tu)

    for decl in tu2.decls:
        if not isinstance(decl, FunctionDecl):
            continue
        if not isinstance(decl.body, CompoundStmt):
            continue

        new_stmts = []
        for s in decl.body.stmts:
            if isinstance(s, ForStmt):
                if _match_matmul_for(s, elem_bits) is not None:
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
    Rimuove dal TU destinato al percorso scalare le dichiarazioni non-scalari
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


# -------------------------
# Return hint extraction
# -------------------------

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


def _extract_return_hint_for_function(
    fn: FunctionDecl,
    vecmat_module,
) -> ReturnHint | None:
    """
    Estrae un hint dal return C:
      - return c[<const>]
      - return C[<const>][<const>]
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

    return None


def _extract_return_hints(tu: TranslationUnit, vecmat_module) -> dict[str, ReturnHint]:
    hints: dict[str, ReturnHint] = {}
    for decl in tu.decls:
        if not isinstance(decl, FunctionDecl):
            continue
        h = _extract_return_hint_for_function(decl, vecmat_module)
        if h is not None:
            hints[decl.name] = h
    return hints


# -------------------------
# Merge
# -------------------------

def _merge_scalar_and_vec_quantum(
    scalar_quantum_module: ModuleOp,
    vec_quantum_module: ModuleOp,
    return_hints: dict[str, ReturnHint] | None = None,
) -> ModuleOp:
    """
    Merge + collegamento risultato vettoriale/matriciale al return (solo se il return scalare è placeholder).

    Logica corretta (generale):
      - se vec_quantum_module espone result_map (side-table) allora:
          * return c[i]    -> lookup key (name, i)
          * return C[r][c] -> lookup key (name, r, c)
        e si collega ESATTAMENTE quello SSAValue.
      - fallback solo se result_map non è disponibile o non contiene la chiave.
    """
    return_hints = return_hints or {}

    scalar_top: Block = scalar_quantum_module.body.blocks[0]
    vec_top: Block = vec_quantum_module.body.blocks[0]

    scalar_funcs: dict[str, FuncOp] = {
        op.sym_name.data: op for op in scalar_top.ops if isinstance(op, FuncOp)
    }

    # result_map prodotto da from_qar_to_quantum_mlir (se avete applicato la modifica al file qar_to_quantum_mlir.py)
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

        # ReturnOp scalare (prima occorrenza)
        return_anchor: ReturnOp | None = None
        for op in scalar_body.ops:
            if isinstance(op, ReturnOp):
                return_anchor = op
                break

        # Prologue anchor: prima op NON init e NON return; se non esiste, usare return
        prologue_anchor: Operation | None = None
        for op in scalar_body.ops:
            if isinstance(op, ReturnOp):
                continue
            if not isinstance(op, QuantumInitOp):
                prologue_anchor = op
                break
        if prologue_anchor is None:
            prologue_anchor = return_anchor  # può essere None

        # Separa init e non-init dal vettoriale
        vec_inits: list[Operation] = []
        vec_others: list[Operation] = []
        for op in vec_body.ops:
            if isinstance(op, ReturnOp):
                continue
            if isinstance(op, QuantumInitOp):
                vec_inits.append(op)
            else:
                vec_others.append(op)

        ssa_map: dict[SSAValue, SSAValue] = {}

        def insert_before(anchor: Operation | None, op_to_insert: Operation) -> None:
            if anchor is None:
                scalar_body.add_op(op_to_insert)
            else:
                scalar_body.insert_op_before(op_to_insert, anchor)

        def remap(v: SSAValue) -> SSAValue:
            return ssa_map.get(v, v)

        # 1) Inserisci init nel prologo
        for init_op in vec_inits:
            new_init = init_op.clone()
            insert_before(prologue_anchor, new_init)
            if init_op.results and new_init.results:
                ssa_map[init_op.results[0]] = new_init.results[0]

        # 2) Inserisci altre op prima del return e rimappa SSA
        produced_results: list[SSAValue] = []
        for op in vec_others:
            new_op = op.clone()
            new_op.operands = [remap(o) for o in new_op.operands]
            insert_before(return_anchor, new_op)

            for old_res, new_res in zip(op.results, new_op.results):
                ssa_map[old_res] = new_res
                produced_results.append(new_res)

        # 3) Se return scalare è placeholder quantum.init 0, sostituire con il valore corretto
        if return_anchor is not None and return_anchor.operands:
            ret_val = return_anchor.operands[0]
            is_placeholder_zero = False

            if isinstance(ret_val.owner, QuantumInitOp):
                try:
                    is_placeholder_zero = int(ret_val.owner.value.value.data) == 0
                except Exception:
                    is_placeholder_zero = False

            if is_placeholder_zero:
                chosen: SSAValue | None = None
                hint = return_hints.get(fname)

                # --- PRIMA SCELTA: result_map (corretto e generale) ---
                if hint is not None and vec_result_map is not None:
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

                # --- FALLBACK: sinks (solo se result_map non disponibile o chiave assente) ---
                if chosen is None and produced_results:
                    sinks: list[SSAValue] = []
                    for r in produced_results:
                        try:
                            if len(list(r.uses)) == 0:
                                sinks.append(r)
                        except Exception:
                            pass

                    if hint is not None and sinks:
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

def compile_c_file(
    c_file: str, num_bits: int = 16, verbose: bool = False, pretty: bool = False, run: bool = False
) -> str:
    base = os.path.splitext(os.path.basename(c_file))[0]

    json_path = generate_json_ast(c_file)
    with open(json_path) as f:
        ast_json = json.load(f)

    tu = parse_ast(ast_json)

    if pretty:
        print("=== Pretty Printed C Code ===")
        print(pretty_print_translation_unit(tu))
        print("================================")

    print("=== TranslationUnit (dataclass) ===")
    pprint(tu)

    vecmat_module = from_c_ast_to_vecmat(tu, num_bits)
    print("=== VecMatModule (dialetto vettoriale/matriciale) ===")
    pprint(vecmat_module)

    # Hint per collegare return (c[i], C[r][c]) quando il percorso scalare è stato filtrato
    return_hints = _extract_return_hints(tu, vecmat_module)

    has_vecmat_ops = any(func.ops for func in vecmat_module.functions)
    has_vecmat_consts = bool(getattr(vecmat_module, "const_arrays", {}) or {})
    enable_vecmat_path = has_vecmat_ops or has_vecmat_consts

    # Step 3a: percorso SCALARE originale QuantumC, filtrato se c'è vec/mat
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
            print(">>> Rilevate macro-op vettoriali/matriciali: VecMat → QAR → quantum.")
        else:
            print(">>> Nessuna macro-op, ma presenti inizializzazioni costanti: VecMat → QAR → quantum (solo init).")

        qar_module = from_vecmat_to_qar(vecmat_module)
        print("=== QarModule (dialetto QAR) ===")
        pprint(qar_module)

        print("QAR const_arrays:", getattr(qar_module, "const_arrays", None))
        print("QAR const_shapes:", getattr(qar_module, "const_shapes", None))

        vec_quantum_module = from_qar_to_quantum_mlir(qar_module, num_bits=num_bits)
        vec_quantum_path = os.path.join(QMLIR_DIR, f"{base}_quantum_vec.mlir")
        save_module(vec_quantum_module, vec_quantum_path)

        quantum_module = _merge_scalar_and_vec_quantum(
            scalar_quantum_module,
            vec_quantum_module,
            return_hints=return_hints,
        )
    else:
        print(">>> Nessuna macro-op vettoriale/matriciale e nessuna inizializzazione costante: uso solo il percorso scalare.")
        quantum_module = scalar_quantum_module

    _ensure_scalar_returns(quantum_module)

    quantum_path = os.path.join(QMLIR_DIR, f"{base}_quantum_final.mlir")
    save_module(quantum_module, quantum_path)

    circuit = generate_circuit(quantum_module, num_bits=num_bits, verbose=verbose)
    qasm_path = os.path.join(QASM_DIR, f"{base}.qasm")

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

    args = parser.parse_args()

    start = time.time() if args.time else None

    qasm_path = compile_c_file(
        args.c_file,
        num_bits=args.bits,
        verbose=args.verbose,
        pretty=args.pretty,
        run=args.run,
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