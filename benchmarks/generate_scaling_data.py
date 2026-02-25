#!/usr/bin/env python3
"""
Generate scaling data for vec_add, vec_dot, matmul with varying N.
Collects: Qubits, Depth, T-count for both QFT and Ripple backends.

METODOLOGIA UNIFICATA (come pipeline.py):
- Entrambi i backend vengono transpilati allo STESSO gate set Clifford+T
- Tutte le metriche (qubits, depth, T-count) sono calcolate dal circuito transpilato
- T-count = T + Tdg + Rz(π/4) + 150×Rz(arbitrari)
- Riferimento: Ross-Selinger 2014, ~150 T per rotazione arbitraria (ε=10^-15)
"""

import os
import sys
import csv
import json
import tempfile
import time
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    generate_json_ast,
    parse_ast,
    from_c_ast_to_vecmat,
    from_vecmat_to_quantum_mlir,
)
from qiskit import transpile
from step5_quantum_mlir_to_qasm.qasm_generator import generate_circuit
from step5_quantum_mlir_to_qasm.q_arithmetics import set_number_of_bits, set_arithmetic_mode

# =============================================================================
# Clifford+T Configuration (UGUALE per entrambi i backend)
# =============================================================================
CLIFFORD_T_BASIS = ['h', 's', 'sdg', 't', 'tdg', 'cx', 'rz']
T_GATES_PER_ARBITRARY_ROTATION = 150  # Solovay-Kitaev estimation

# =============================================================================
# Configuration
# =============================================================================
BITS = 8
N_VALUES_VECADD = [2, 4, 8, 16, 32, 64, 128]
N_VALUES_VECDOT = [2, 4, 8, 16, 32, 64]
N_VALUES_MATMUL = [2, 4, 8]
OUTPUT_CSV = Path(__file__).parent / "scaling_results.csv"


# =============================================================================
# Helper functions for T-count calculation (same as pipeline.py)
# =============================================================================
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


def compute_t_count_from_circuit(circuit) -> dict:
    """
    Calcola T-count dal circuito transpilato a Clifford+T.

    Returns:
        dict con t_count, t_count_exact, t_count_approx, arbitrary_rotations
    """
    counts = circuit.count_ops()

    # T e Tdg espliciti
    t_count_exact = counts.get('t', 0) + counts.get('tdg', 0)

    # Analizza Rz
    rz_t_angle = 0
    rz_arbitrary = 0

    for inst in circuit.data:
        if inst.operation.name == 'rz':
            angle = float(inst.operation.params[0])
            if _is_clifford_angle(angle):
                pass  # 0 T
            elif _is_t_angle(angle):
                rz_t_angle += 1
            else:
                rz_arbitrary += 1

    t_count_exact += rz_t_angle
    t_count_approx = rz_arbitrary * T_GATES_PER_ARBITRARY_ROTATION

    return {
        't_count': t_count_exact + t_count_approx,
        't_count_exact': t_count_exact,
        't_count_approx': t_count_approx,
        'arbitrary_rotations': rz_arbitrary,
    }


def generate_vec_add_c(n: int) -> str:
    """Generate C code for vector addition of size n."""
    init_a = ", ".join(["1"] * n)
    init_b = ", ".join(["2"] * n)
    return f"""int main() {{
    int a[{n}] = {{{init_a}}};
    int b[{n}] = {{{init_b}}};
    int c[{n}];
    for (int i = 0; i < {n}; i++) {{
        c[i] = a[i] + b[i];
    }}
    return c[0];
}}
"""


def generate_vec_dot_c(n: int) -> str:
    """Generate C code for dot product of size n."""
    init_a = ", ".join(["1"] * n)
    init_b = ", ".join(["2"] * n)
    return f"""int main() {{
    int a[{n}] = {{{init_a}}};
    int b[{n}] = {{{init_b}}};
    int s = 0;
    for (int i = 0; i < {n}; i++) {{
        s = s + a[i] * b[i];
    }}
    return s;
}}
"""


def generate_matmul_c(n: int) -> str:
    """Generate C code for NxN matrix multiplication."""
    row = ", ".join(["1"] * n)
    matrix = ", ".join([f"{{{row}}}" for _ in range(n)])
    return f"""int main() {{
    int A[{n}][{n}] = {{{matrix}}};
    int B[{n}][{n}] = {{{matrix}}};
    int C[{n}][{n}];
    for (int i = 0; i < {n}; i++) {{
        for (int j = 0; j < {n}; j++) {{
            C[i][j] = 0;
            for (int k = 0; k < {n}; k++) {{
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }}
        }}
    }}
    return C[0][0];
}}
"""


def compile_and_measure(c_code: str, bits: int, backend: str) -> dict:
    """
    Compile C code and return circuit metrics.

    METODOLOGIA UNIFICATA (uguale per QFT e Ripple):
    1. Genera circuito
    2. Transpile a Clifford+T basis (STESSO per entrambi)
    3. Calcola TUTTE le metriche dal circuito transpilato
    """
    set_number_of_bits(bits)
    set_arithmetic_mode(backend)

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

        # =====================================================================
        # TRANSPILE A CLIFFORD+T (uguale per entrambi i backend)
        # =====================================================================
        transpiled = transpile(circuit, basis_gates=CLIFFORD_T_BASIS, optimization_level=3)

        # =====================================================================
        # TUTTE LE METRICHE DAL CIRCUITO TRANSPILATO
        # =====================================================================
        qubits = transpiled.num_qubits
        depth = transpiled.depth()

        # T-count con analisi Rz
        t_metrics = compute_t_count_from_circuit(transpiled)
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


def main():
    os.chdir(Path(__file__).parent.parent)

    results = []

    generators = {
        'vec_add': generate_vec_add_c,
        'vec_dot': generate_vec_dot_c,
        'matmul': generate_matmul_c,
    }

    n_values_map = {
        'vec_add': N_VALUES_VECADD,
        'vec_dot': N_VALUES_VECDOT,
        'matmul': N_VALUES_MATMUL,
    }

    print("=" * 70)
    print("SCALING BENCHMARK - Metodologia Unificata Clifford+T")
    print(f"Bit width: {BITS}")
    print(f"Gate set: {CLIFFORD_T_BASIS}")
    print(f"T-count: T + Tdg + Rz(π/4) + {T_GATES_PER_ARBITRARY_ROTATION}×Rz(arbitrari)")
    print("=" * 70)

    for op_name, gen_func in generators.items():
        print(f"\n=== {op_name} ===")
        for n in n_values_map[op_name]:
            print(f"N={n}...", end=" ", flush=True)

            c_code = gen_func(n)

            for backend in ['qft', 'ripple']:
                start_time = time.time()
                try:
                    metrics = compile_and_measure(c_code, BITS, backend)
                    elapsed = time.time() - start_time

                    results.append({
                        'operation': op_name,
                        'N': n,
                        'backend': backend,
                        'qubits': metrics['qubits'],
                        'depth': metrics['depth'],
                        't_count': metrics['t_count'],
                    })
                    print(f"{backend}:OK({elapsed:.1f}s)", end=" ", flush=True)

                except Exception as e:
                    elapsed = time.time() - start_time
                    print(f"{backend}:FAIL({elapsed:.1f}s,{type(e).__name__})", end=" ", flush=True)
                    results.append({
                        'operation': op_name,
                        'N': n,
                        'backend': backend,
                        'qubits': 0,
                        'depth': 0,
                        't_count': 0,
                    })
            print()

    # Save to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['operation', 'N', 'backend', 'qubits', 'depth', 't_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Operation':<10} {'N':<6} {'Backend':<8} {'Qubits':<12} {'Depth':<12} {'T-count':<15}")
    print("=" * 80)
    for r in results:
        if r['qubits'] > 0:
            print(f"{r['operation']:<10} {r['N']:<6} {r['backend']:<8} {r['qubits']:<12,} {r['depth']:<12,} {r['t_count']:<15,}")
        else:
            print(f"{r['operation']:<10} {r['N']:<6} {r['backend']:<8} {'FAILED':<12}")
    print("=" * 80)


if __name__ == "__main__":
    main()
