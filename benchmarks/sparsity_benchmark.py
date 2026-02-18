#!/usr/bin/env python3
"""
Benchmark script for sparsity optimization.
Generates 4x4 matrices with varying sparsity levels and measures circuit metrics.
"""

import os
import sys
import csv
import json
import random
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    generate_json_ast,
    parse_ast,
    from_c_ast_to_vecmat,
    from_vecmat_to_quantum_mlir,
)
from step5_quantum_mlir_to_qasm.qasm_generator import generate_circuit
from step5_quantum_mlir_to_qasm.q_arithmetics import set_number_of_bits, set_arithmetic_mode


def generate_sparse_matrix(size: int, num_zeros: int, seed: int = None) -> list[list[int]]:
    """Generate a matrix with specified number of zeros."""
    if seed is not None:
        random.seed(seed)

    total = size * size
    if num_zeros > total:
        num_zeros = total

    # Create flat list with non-zero values
    values = [random.randint(1, 7) for _ in range(total - num_zeros)]
    values.extend([0] * num_zeros)
    random.shuffle(values)

    # Reshape to matrix
    return [values[i*size:(i+1)*size] for i in range(size)]


def matrix_to_c_init(matrix: list[list[int]]) -> str:
    """Convert matrix to C initializer syntax."""
    rows = ["{" + ", ".join(str(x) for x in row) + "}" for row in matrix]
    return "{" + ", ".join(rows) + "}"


def generate_matmul_c_code(A: list[list[int]], B: list[list[int]], size: int) -> str:
    """Generate C code for matrix multiplication."""
    return f'''int main() {{
    int A[{size}][{size}] = {matrix_to_c_init(A)};
    int B[{size}][{size}] = {matrix_to_c_init(B)};
    int C[{size}][{size}] = {matrix_to_c_init([[0]*size for _ in range(size)])};

    for (int i = 0; i < {size}; i++) {{
        for (int j = 0; j < {size}; j++) {{
            int s = 0;
            for (int k = 0; k < {size}; k++) {{
                s = s + A[i][k] * B[k][j];
            }}
            C[i][j] = s;
        }}
    }}

    return C[0][0];
}}
'''


def compile_and_measure(c_code: str, bits: int, backend: str) -> dict:
    """Compile C code and return circuit metrics."""
    set_number_of_bits(bits)
    set_arithmetic_mode(backend)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(c_code)
        c_path = f.name

    try:
        # Compile through pipeline
        json_path = generate_json_ast(c_path)
        with open(json_path) as f:
            ast_json = json.load(f)
        tu = parse_ast(ast_json)
        vecmat_module = from_c_ast_to_vecmat(tu, bits)
        quantum_module = from_vecmat_to_quantum_mlir(vecmat_module, bits)
        circuit = generate_circuit(quantum_module, num_bits=bits, arithmetic_mode=backend)

        return {
            'qubits': circuit.num_qubits,
            'gates': circuit.size(),
            'depth': circuit.depth(),
        }
    finally:
        os.unlink(c_path)
        if os.path.exists(json_path):
            os.unlink(json_path)


def run_benchmark():
    """Run the full sparsity benchmark."""
    SIZE = 4
    BITS = 8
    BACKENDS = ['qft', 'ripple']

    # Sparsity levels: number of zeros per matrix (out of 16 elements)
    # Using 0%, 25%, 50%, 75% for clean trends (extreme values cause anomalies)
    ZERO_COUNTS = [0, 4, 8, 12]

    results = []

    for num_zeros in ZERO_COUNTS:
        sparsity = num_zeros / (SIZE * SIZE)
        print(f"Testing sparsity {sparsity:.1%} ({num_zeros} zeros)...")

        # Generate matrices with same seed for reproducibility
        A = generate_sparse_matrix(SIZE, num_zeros, seed=42 + num_zeros)
        B = generate_sparse_matrix(SIZE, num_zeros, seed=123 + num_zeros)
        c_code = generate_matmul_c_code(A, B, SIZE)

        for backend in BACKENDS:
            print(f"  Backend: {backend}")
            metrics = compile_and_measure(c_code, BITS, backend)

            results.append({
                'sparsity': sparsity,
                'zeros': num_zeros,
                'backend': backend,
                'qubits': metrics['qubits'],
                'gates': metrics['gates'],
                'depth': metrics['depth'],
            })

    # Write CSV
    output_path = Path(__file__).parent / 'sparsity_results.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sparsity', 'zeros', 'backend', 'qubits', 'gates', 'depth'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Sparsity':<10} {'Zeros':<6} {'Backend':<8} {'Qubits':<8} {'Gates':<10} {'Depth':<8}")
    print("="*70)
    for r in results:
        print(f"{r['sparsity']:<10.1%} {r['zeros']:<6} {r['backend']:<8} {r['qubits']:<8} {r['gates']:<10} {r['depth']:<8}")


if __name__ == '__main__':
    run_benchmark()
