#!/usr/bin/env python3
"""
Generate scaling data for vec_add, vec_dot, matmul with varying N.
Collects: Qubits, Depth, T-count, T-depth for both QFT and Ripple backends.

Approccio subprocess: ogni benchmark viene eseguito come processo separato
tramite pipeline.py, evitando accumulo di memoria.
"""

import subprocess
import sys
import os
import csv
import time
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

BITS = 8
N_VALUES_VECADD = [2, 4, 8, 16, 32, 64, 128]
N_VALUES_VECDOT = [2, 4, 8, 16, 32, 64]
N_VALUES_MATMUL = [2, 4, 8, 16]
OUTPUT_CSV = Path(__file__).parent / "scaling_results.csv"


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


def compile_and_get_metrics(c_code: str, adder: str) -> dict:
    """Compile C code via subprocess and extract metrics from pipeline.py output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(c_code)
        c_path = f.name

    try:
        cmd = [sys.executable, "pipeline.py", c_path, "--bits", str(BITS), "--adder", adder]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
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
        print(f"TIMEOUT", end=" ", flush=True)
        return None
    except Exception as e:
        print(f"ERROR({type(e).__name__})", end=" ", flush=True)
        return None
    finally:
        os.unlink(c_path)


def save_results(results):
    """Salva risultati parziali o finali su CSV."""
    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['operation', 'N', 'backend', 'qubits', 'depth', 't_count', 't_depth']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


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
    print("SCALING BENCHMARK - Subprocess approach")
    print(f"Bit width: {BITS}")
    print("=" * 70)

    for op_name, gen_func in generators.items():
        print(f"\n=== {op_name} ===")
        for n in n_values_map[op_name]:
            print(f"N={n}...", end=" ", flush=True)

            c_code = gen_func(n)

            for backend in ['qft', 'ripple']:
                start_time = time.time()
                metrics = compile_and_get_metrics(c_code, backend)
                elapsed = time.time() - start_time

                if metrics:
                    results.append({
                        'operation': op_name,
                        'N': n,
                        'backend': backend,
                        **metrics,
                    })
                    print(f"{backend}:OK({elapsed:.1f}s)", end=" ", flush=True)
                else:
                    results.append({
                        'operation': op_name,
                        'N': n,
                        'backend': backend,
                        'qubits': 0,
                        'depth': 0,
                        't_count': 0,
                        't_depth': 0,
                    })
                    print(f"{backend}:FAIL({elapsed:.1f}s)", end=" ", flush=True)
            print()
            save_results(results)

    save_results(results)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Operation':<10} {'N':<6} {'Backend':<8} {'Qubits':<12} {'Depth':<12} {'T-count':<15} {'T-depth':<12}")
    print("=" * 90)
    for r in results:
        if r['qubits'] > 0:
            print(f"{r['operation']:<10} {r['N']:<6} {r['backend']:<8} {r['qubits']:<12,} {r['depth']:<12,} {r['t_count']:<15,} {r['t_depth']:<12,}")
        else:
            print(f"{r['operation']:<10} {r['N']:<6} {r['backend']:<8} {'FAILED':<12}")
    print("=" * 90)


if __name__ == "__main__":
    main()
