#!/usr/bin/env python3
"""
Benchmark script for sparsity optimization on 16x16 matrices.
Measures Clifford+T metrics: qubits, depth, t_count.

Sparsity levels chosen for ML relevance:
- 0%: Dense baseline
- 25%: Light sparsity
- 50%: Moderate sparsity
- 75%: Sparse
- 90%: Typical ML pruning

T-count methodology:
- QFT backend: Uses transpile + Solovay-Kitaev estimation (150T per arbitrary rotation)
- Ripple backend: Uses theoretical T-count from gate composition (CCX=7T, MCX deterministic)
  This avoids Qiskit's inconsistent MCX decomposition which depends on circuit size.
"""

import os
import sys
import csv
import json
import random
import tempfile
import time
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    generate_json_ast,
    parse_ast,
    from_c_ast_to_vecmat,
    from_vecmat_to_quantum_mlir,
)
from qiskit import transpile
from step5_quantum_mlir_to_qasm.qasm_generator import (
    generate_circuit,
    compute_t_count,
)
from step5_quantum_mlir_to_qasm.q_arithmetics import set_number_of_bits, set_arithmetic_mode

# Basis gates per transpile (solo per QFT)
UNIFORM_BASIS = ["cx", "rz", "sx", "x", "measure"]

# T-count per gate (standard Clifford+T decomposition)
T_PER_CCX = 7  # Toffoli gate: 7 T-gates


# =============================================================================
# Configuration
# =============================================================================
SIZE = 16           # Matrix dimension (16x16 = 256 elements)
BITS = 8            # Bit width for integers
BACKENDS = ['qft', 'ripple']

# Sparsity levels (percentage of zeros)
SPARSITY_LEVELS = [0.0, 0.25, 0.50, 0.75, 0.90]


def calculate_ripple_t_count(circuit) -> int:
    """
    Calculate theoretical T-count for ripple-carry circuits.

    Uses standard decomposition formulas:
    - CCX (Toffoli): 7 T-gates
    - MCX(n controls): V-chain decomposition = 2*(n-1) Toffoli = 14*(n-1) T

    This is deterministic and doesn't depend on Qiskit's transpiler behavior.
    Reference: Amy et al., "Polynomial-time T-depth optimization"
    """
    total_t = 0

    for inst in circuit.data:
        name = inst.operation.name
        if name == 'ccx':
            total_t += T_PER_CCX
        elif name == 'mcx':
            n_controls = len(inst.qubits) - 1
            # V-chain decomposition: 2*(n-1) Toffoli gates
            total_t += 2 * (n_controls - 1) * T_PER_CCX

    return total_t


def generate_sparse_matrix(size: int, num_zeros: int, seed: int = None) -> list[list[int]]:
    """Generate a matrix with specified number of zeros."""
    if seed is not None:
        random.seed(seed)

    total = size * size
    if num_zeros > total:
        num_zeros = total

    # Create flat list with non-zero values (1-7 to keep values small)
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
    """
    Compile C code and return circuit metrics.

    T-count methodology differs by backend:
    - QFT: transpile to basis gates + Solovay-Kitaev estimation (150T per arbitrary rotation)
    - Ripple: theoretical T-count from gate composition (deterministic, no Qiskit dependency)
    """
    set_number_of_bits(bits)
    set_arithmetic_mode(backend)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(c_code)
        c_path = f.name

    json_path = None
    try:
        # Compile through pipeline
        json_path = generate_json_ast(c_path)
        with open(json_path) as f:
            ast_json = json.load(f)
        tu = parse_ast(ast_json)
        vecmat_module = from_c_ast_to_vecmat(tu, bits)
        quantum_module = from_vecmat_to_quantum_mlir(vecmat_module, bits)
        circuit = generate_circuit(quantum_module, num_bits=bits, arithmetic_mode=backend)

        # Qubits dal circuito originale
        qubits = circuit.num_qubits

        # Depth da decompose (deterministico, senza anomalie)
        circuit_decomposed = circuit.decompose(reps=1)
        depth = circuit_decomposed.depth()

        # T-count: diverso per backend
        if backend == 'ripple':
            # Ripple-carry: T-count teorico deterministico
            # Evita l'inconsistenza di Qiskit nella decomposizione MCX
            t_count = calculate_ripple_t_count(circuit)
        else:
            # QFT: transpile + Solovay-Kitaev estimation
            circuit_transpiled = transpile(circuit, basis_gates=UNIFORM_BASIS, optimization_level=1)
            t_metrics = compute_t_count(circuit_transpiled)
            t_count = t_metrics['t_count']

        return {
            'qubits': qubits,
            'depth': depth,
            't_count': t_count,
        }
    finally:
        if os.path.exists(c_path):
            os.unlink(c_path)
        if json_path and os.path.exists(json_path):
            os.unlink(json_path)


def run_benchmark():
    """Run the full sparsity benchmark for 16x16 matrices."""
    total_elements = SIZE * SIZE
    results = []

    print(f"=" * 80)
    print(f"SPARSITY BENCHMARK: {SIZE}x{SIZE} matrices ({total_elements} elements)")
    print(f"Bit width: {BITS}, Backends: {BACKENDS}")
    print(f"=" * 80)
    print()

    total_tests = len(SPARSITY_LEVELS) * len(BACKENDS)
    current_test = 0

    for sparsity in SPARSITY_LEVELS:
        num_zeros = int(sparsity * total_elements)
        print(f"[Sparsity {sparsity:.0%}] {num_zeros} zeros / {total_elements} elements", flush=True)

        # Generate matrices with same seed for reproducibility
        A = generate_sparse_matrix(SIZE, num_zeros, seed=42 + num_zeros)
        B = generate_sparse_matrix(SIZE, num_zeros, seed=123 + num_zeros)
        c_code = generate_matmul_c_code(A, B, SIZE)

        for backend in BACKENDS:
            current_test += 1
            print(f"  [{current_test}/{total_tests}] Backend: {backend}...", end=" ", flush=True)

            start_time = time.time()
            try:
                metrics = compile_and_measure(c_code, BITS, backend)
                elapsed = time.time() - start_time

                results.append({
                    'sparsity': sparsity,
                    'zeros': num_zeros,
                    'backend': backend,
                    'qubits': metrics['qubits'],
                    'depth': metrics['depth'],
                    't_count': metrics['t_count'],
                })

                print(f"OK ({elapsed:.1f}s) - qubits={metrics['qubits']:,}, t_count={metrics['t_count']:,}", flush=True)

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"FAILED ({elapsed:.1f}s) - {type(e).__name__}: {e}")
                results.append({
                    'sparsity': sparsity,
                    'zeros': num_zeros,
                    'backend': backend,
                    'qubits': -1,
                    'depth': -1,
                    't_count': -1,
                })

        print()

    # Write CSV
    output_path = Path(__file__).parent / 'sparsity_results_16x16.csv'
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['sparsity', 'zeros', 'backend', 'qubits', 'depth', 't_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_path}")

    # Print summary table
    print()
    print("=" * 85)
    print(f"{'Sparsity':<10} {'Zeros':<8} {'Backend':<8} {'Qubits':<12} {'Depth':<10} {'T-count':<15}")
    print("=" * 85)
    for r in results:
        if r['qubits'] > 0:
            print(f"{r['sparsity']:<10.0%} {r['zeros']:<8} {r['backend']:<8} {r['qubits']:<12,} {r['depth']:<10,} {r['t_count']:<15,}")
        else:
            print(f"{r['sparsity']:<10.0%} {r['zeros']:<8} {r['backend']:<8} {'FAILED':<12} {'-':<10} {'-':<15}")
    print("=" * 85)


if __name__ == '__main__':
    run_benchmark()
