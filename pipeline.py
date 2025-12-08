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


from xdsl.dialects.func import FuncOp, ReturnOp  # assicurarsi che sia presente in testa al file
from xdsl.ir import Block
from xdsl.dialects.builtin import ModuleOp

def _merge_scalar_and_vec_quantum(
    scalar_quantum_module: ModuleOp,
    vec_quantum_module: ModuleOp,
) -> ModuleOp:
    """
    Fonde il modulo quantistico scalare generato da QuantumC con quello
    vettoriale/matriciale generato dal nostro pass QAR → quantum.

    Strategia:
    - per ogni funzione con lo stesso nome,
      * si prendono tutte le op del corpo vettoriale,
      * si separano le QuantumInitOp dalle altre,
      * si clona tutto e si inserisce nel corpo scalare,
        prima dell'eventuale ReturnOp, con l'ordine:
          1) tutte le QuantumInitOp,
          2) tutte le altre op (QAddiOp, QMuliOp, ...).

    In questo modo, il backend step5 vede sempre i quantum.init
    prima delle operazioni che li usano.
    """

    scalar_block: Block = scalar_quantum_module.body.blocks[0]
    vec_block: Block = vec_quantum_module.body.blocks[0]

    # nome funzione -> FuncOp nel modulo scalare
    scalar_funcs: dict[str, FuncOp] = {}
    for op in scalar_block.ops:
        if isinstance(op, FuncOp):
            scalar_funcs[op.sym_name.data] = op

    # per ogni funzione vettoriale
    for op in vec_block.ops:
        if not isinstance(op, FuncOp):
            continue

        fname = op.sym_name.data
        if fname not in scalar_funcs:
            # per ora ignoriamo eventuali funzioni "solo vettoriali"
            continue

        scalar_func = scalar_funcs[fname]
        scalar_body_block: Block = scalar_func.body.blocks[0]
        vec_body_block: Block = op.body.blocks[0]

        # cerca eventuale ReturnOp nel blocco scalare
        last_ret: ReturnOp | None = None
        for s_op in scalar_body_block.ops:
            if isinstance(s_op, ReturnOp):
                last_ret = s_op

        # separa init e altre op nel corpo vettoriale
        init_ops = []
        other_ops = []
        for vop in vec_body_block.ops:
            if isinstance(vop, QuantumInitOp):
                init_ops.append(vop)
            else:
                other_ops.append(vop)

        def insert_before(anchor: Operation, ops_to_insert: list[Operation]):
            """
            Inserisce le operazioni clonate prima di 'anchor' nello stesso ordine
            in cui compaiono in ops_to_insert.
            """
            # per preservare l'ordine, si inserisce in reverse
            for vop in reversed(ops_to_insert):
                cloned = vop.clone()
                scalar_body_block.insert_op_before(cloned, anchor)

        if last_ret is not None:
            # inserisce prima tutti i quantum.init, poi le altre op, prima del return
            insert_before(last_ret, init_ops)
            insert_before(last_ret, other_ops)
        else:
            # nessun return: aggiunge semplicemente in coda al blocco
            for vop in init_ops + other_ops:
                cloned = vop.clone()
                scalar_body_block.add_op(cloned)

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