#!/usr/bin/env python3
"""Script per generare figura del circuito quantistico."""

import sys
import os
import json
import subprocess

sys.path.insert(0, '.')

from step2_ast_to_dataclasses.c_ast import parse_ast
from step5_quantum_mlir_to_qasm.qasm_generator import generate_circuit
from my_extensions.vecmat_lowering import from_c_ast_to_vecmat
from my_extensions.vecmat_to_quantum_mlir import from_vecmat_to_quantum_mlir

JSON_DIR = "json_out"

def generate_json_ast(c_path: str) -> str:
    """Genera JSON AST da file C usando clang."""
    base = os.path.splitext(os.path.basename(c_path))[0]
    os.makedirs(JSON_DIR, exist_ok=True)
    json_path = os.path.join(JSON_DIR, f"{base}.json")
    cmd = ["clang", "-Xclang", "-ast-dump=json", "-fsyntax-only", c_path]
    with open(json_path, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)
    return json_path

def generate_figure(c_file, num_bits, output_path):
    """Genera figura del circuito quantistico."""
    json_path = generate_json_ast(c_file)
    with open(json_path) as f:
        ast_json = json.load(f)

    tu = parse_ast(ast_json)
    vecmat_module = from_c_ast_to_vecmat(tu, elem_bits=num_bits)
    quantum_module = from_vecmat_to_quantum_mlir(vecmat_module, num_bits=num_bits)

    # Genera circuito
    circuit = generate_circuit(quantum_module, num_bits=num_bits, verbose=False, arithmetic_mode="qft")

    print(f"Qubits: {circuit.num_qubits}")
    print(f"Depth: {circuit.depth()}")
    print(f"Gates: {circuit.size()}")

    # Salva figura
    fig = circuit.draw(output='mpl', style='iqp', fold=-1, scale=0.6)
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\n✅ Figura salvata in {output_path}")

if __name__ == "__main__":
    generate_figure(
        c_file="tests/vector/add_small.c",
        num_bits=2,
        output_path="docs/latex/figures/circuit_vecadd.pdf"
    )
