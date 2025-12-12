"""CLI driving the QuantumC compilation pipeline."""
from __future__ import annotations
from pprint import pprint

import argparse
import json
import os
import subprocess
import sys
import time

from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer
from xdsl.dialects.func import FuncOp, ReturnOp  # <-- nuovo import

from xdsl.ir import Block


from step2_ast_to_dataclasses.c_ast import parse_ast, TranslationUnit, pretty_print_translation_unit
from step3_dataclasses_to_mlir.mlir_generator import MLIRGenerator
from step4_mlir_to_quantum_mlir.quantum_mlir_generator import generate_quantum_mlir
from step5_quantum_mlir_to_qasm.qasm_generator import generate_circuit, export_qasm, export_qasm_clifford_t
from step5_quantum_mlir_to_qasm.q_arithmetics import simulate
from step4_mlir_to_quantum_mlir.quantum_dialect import QuantumInitOp

# Estensioni nostre
from my_extensions.vecmat_lowering import from_c_ast_to_vecmat
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


from xdsl.ir import Operation, SSAValue
from xdsl.dialects.func import FuncOp, ReturnOp
from step4_mlir_to_quantum_mlir.quantum_dialect import QuantumInitOp

def _merge_scalar_and_vec_quantum(
    scalar_quantum_module: ModuleOp,
    vec_quantum_module: ModuleOp,
) -> ModuleOp:

    scalar_top: Block = scalar_quantum_module.body.blocks[0]
    vec_top: Block = vec_quantum_module.body.blocks[0]

    # nome funzione -> FuncOp nel modulo scalare
    scalar_funcs: dict[str, FuncOp] = {
        op.sym_name.data: op for op in scalar_top.ops if isinstance(op, FuncOp)
    }

    # per ogni funzione vettoriale
    for vec_func in vec_top.ops:
        if not isinstance(vec_func, FuncOp):
            continue

        fname = vec_func.sym_name.data
        if fname not in scalar_funcs:
            continue

        scalar_func = scalar_funcs[fname]
        scalar_body: Block = scalar_func.body.blocks[0]
        vec_body: Block = vec_func.body.blocks[0]

        # trova ReturnOp scalare (anchor)
        anchor: Operation | None = None
        for op in scalar_body.ops:
            if isinstance(op, ReturnOp):
                anchor = op
                break
        if anchor is None:
            # se non c'è return, inseriamo in coda
            # (ma di solito main scalare ha return)
            anchor = None

        # 1) separa init e altre op dal vec body (ignorando ReturnOp)
        vec_inits: list[Operation] = []
        vec_others: list[Operation] = []
        for op in vec_body.ops:
            if isinstance(op, ReturnOp):
                continue
            if isinstance(op, QuantumInitOp):
                vec_inits.append(op)
            else:
                vec_others.append(op)

        # 2) SSA remap: vec SSA -> new SSA (dopo inserimento nel blocco scalare)
        ssa_map: dict[SSAValue, SSAValue] = {}

        def insert_op(op: Operation):
            if anchor is None:
                scalar_body.add_op(op)
            else:
                scalar_body.insert_op_before(op, anchor)

        # 3) inserisci init e costruisci mapping sui risultati
        for init_op in vec_inits:
            new_init = init_op.clone()
            insert_op(new_init)
            # mappa risultato 0 (quantum.init ha un solo result)
            ssa_map[init_op.results[0]] = new_init.results[0]

        # helper: rimappa un SSAValue se presente in ssa_map
        def remap(v: SSAValue) -> SSAValue:
            return ssa_map.get(v, v)

        # 4) inserisci le altre op con ricablaggio
        for op in vec_others:
            new_op = op.clone()

            # ricablaggio degli operandi
            # (xdsl: Operation.operands è una lista modificabile)
            new_operands = [remap(o) for o in new_op.operands]
            new_op.operands = new_operands

            insert_op(new_op)

            # aggiorna mapping anche per i risultati prodotti (catena add/mul)
            for old_res, new_res in zip(op.results, new_op.results):
                ssa_map[old_res] = new_res

    return scalar_quantum_module

def compile_c_file(
    c_file: str, num_bits: int = 16, verbose: bool = False, pretty: bool = False, run: bool = False
) -> str:
    base = os.path.splitext(os.path.basename(c_file))[0]

    # Step 1: Clang → JSON AST
    json_path = generate_json_ast(c_file)
    with open(json_path) as f:
        ast_json = json.load(f)

    # Step 2: JSON AST → TranslationUnit (dataclass)
    tu = parse_ast(ast_json)

    if pretty:
        print("=== Pretty Printed C Code ===")
        print(pretty_print_translation_unit(tu))
        print("================================")

    print("=== TranslationUnit (dataclass) ===")
    pprint(tu)

    # Step 3a: percorso SCALARE originale QuantumC
    mlir_module = generate_mlir(tu)
    classical_path = os.path.join(MLIR_DIR, f"{base}_classical.mlir")
    save_module(mlir_module, classical_path)

    scalar_quantum_module = generate_quantum_mlir(mlir_module)
    scalar_quantum_path = os.path.join(QMLIR_DIR, f"{base}_quantum_scalar.mlir")
    save_module(scalar_quantum_module, scalar_quantum_path)

    # Step 3b: percorso VETTORIALE/MATRICIALE (VecMat → QAR → quantum)
    vecmat_module = from_c_ast_to_vecmat(tu, num_bits)
    print("=== VecMatModule (dialetto vettoriale/matriciale) ===")
    pprint(vecmat_module)

    has_vecmat_ops = any(func.ops for func in vecmat_module.functions)

    if has_vecmat_ops:
        print(">>> Rilevate macro-op vettoriali/matriciali: VecMat → QAR → quantum.")
        qar_module = from_vecmat_to_qar(vecmat_module)
        print("=== QarModule (dialetto QAR) ===")
        pprint(qar_module)

        print("QAR const_arrays:", getattr(qar_module, "const_arrays", None))
        print("QAR const_shapes:", getattr(qar_module, "const_shapes", None))

        vec_quantum_module = from_qar_to_quantum_mlir(qar_module, num_bits=num_bits)
        vec_quantum_path = os.path.join(QMLIR_DIR, f"{base}_quantum_vec.mlir")
        save_module(vec_quantum_module, vec_quantum_path)

        # Fusione dei due quantum MLIR nello stesso ModuleOp
        quantum_module = _merge_scalar_and_vec_quantum(
            scalar_quantum_module,
            vec_quantum_module,
        )
    else:
        print(">>> Nessuna macro-op vettoriale/matriciale: uso solo il percorso scalare.")
        quantum_module = scalar_quantum_module

    # Salva il modulo quantistico finale (ibrido se necessario)
    quantum_path = os.path.join(QMLIR_DIR, f"{base}_quantum_final.mlir")
    save_module(quantum_module, quantum_path)

    # Step finale: quantum MLIR → circuito → QASM
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