#!/usr/bin/env python3
"""
Benchmark script for sparsity optimization on 16x16 matrices.
Measures Clifford+T metrics: qubits, depth, t_count, t_depth.

Uses subprocess approach (like generate_scaling_data.py): each benchmark
runs pipeline.py as a separate process, ensuring metrics are computed
via the unified Clifford+T flow (_compute_metrics_and_score).

Sparsity levels chosen for ML relevance:
- 0%: Dense baseline
- 25%: Light sparsity
- 50%: Moderate sparsity
- 75%: Sparse
- 90%: Typical ML pruning
"""

import os
import sys
import csv
import random
import subprocess
import tempfile
import time
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================
SIZE = 16           # Matrix dimension (16x16 = 256 elements)
BITS = 8            # Bit width for integers
BACKENDS = ['qft', 'ripple']

# Sparsity levels (percentage of zeros)
SPARSITY_LEVELS = [0.90, 0.75, 0.50, 0.25, 0.0]

OUTPUT_CSV = Path(__file__).parent / "sparsity_results_16x16.csv"


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


def compile_and_get_metrics(c_code: str, bits: int, backend: str) -> dict | None:
    """Compile C code via subprocess and extract metrics from pipeline.py output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(c_code)
        c_path = f.name

    try:
        cmd = [sys.executable, "pipeline.py", c_path, "--bits", str(bits), "--adder", backend]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=18000)
        output = result.stdout + result.stderr

        metrics = {
            'qubits': 0,
            'depth': 0,
            't_count': 0,
            't_depth': 0,
        }

        for line in output.split('\n'):
            if 'num_qubits' in line:
                metrics['qubits'] = int(line.split(':')[1].strip())
            elif line.strip().startswith('depth'):
                metrics['depth'] = int(line.split(':')[1].strip())
            elif 'T-count' in line and 'total' in line:
                val = line.split(':')[1].split('(')[0].strip()
                metrics['t_count'] = int(val)
            elif 'T-depth' in line and 'layers' in line:
                val = line.split(':')[1].split('(')[0].strip()
                metrics['t_depth'] = int(val)

        return metrics
    except subprocess.TimeoutExpired:
        print("TIMEOUT", end=" ", flush=True)
        return None
    except Exception as e:
        print(f"ERROR({type(e).__name__})", end=" ", flush=True)
        return None
    finally:
        os.unlink(c_path)


def save_results(results):
    """Save partial or final results to CSV."""
    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['sparsity', 'zeros', 'backend', 'qubits', 'depth', 't_count', 't_depth']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_benchmark():
    """Run the full sparsity benchmark for 16x16 matrices."""
    os.chdir(Path(__file__).parent.parent)

    total_elements = SIZE * SIZE
    results = []

    print("=" * 80)
    print(f"SPARSITY BENCHMARK: {SIZE}x{SIZE} matrices ({total_elements} elements)")
    print(f"Bit width: {BITS}, Backends: {BACKENDS}")
    print(f"Subprocess approach (unified Clifford+T metrics via pipeline.py)")
    print("=" * 80)
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
            metrics = compile_and_get_metrics(c_code, BITS, backend)
            elapsed = time.time() - start_time

            if metrics and metrics['qubits'] > 0:
                results.append({
                    'sparsity': sparsity,
                    'zeros': num_zeros,
                    'backend': backend,
                    **metrics,
                })
                print(f"OK ({elapsed:.1f}s) - qubits={metrics['qubits']:,}, t_count={metrics['t_count']:,}", flush=True)
            else:
                results.append({
                    'sparsity': sparsity,
                    'zeros': num_zeros,
                    'backend': backend,
                    'qubits': 0,
                    'depth': 0,
                    't_count': 0,
                    't_depth': 0,
                })
                print(f"FAIL ({elapsed:.1f}s)", flush=True)

        print()
        save_results(results)

    save_results(results)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Print summary table
    print()
    print("=" * 100)
    print(f"{'Sparsity':<10} {'Zeros':<8} {'Backend':<8} {'Qubits':<12} {'Depth':<12} {'T-count':<15} {'T-depth':<12}")
    print("=" * 100)
    for r in results:
        if r['qubits'] > 0:
            print(f"{r['sparsity']:<10.0%} {r['zeros']:<8} {r['backend']:<8} {r['qubits']:<12,} {r['depth']:<12,} {r['t_count']:<15,} {r['t_depth']:<12,}")
        else:
            print(f"{r['sparsity']:<10.0%} {r['zeros']:<8} {r['backend']:<8} {'FAILED':<12}")
    print("=" * 100)


if __name__ == '__main__':
    run_benchmark()
