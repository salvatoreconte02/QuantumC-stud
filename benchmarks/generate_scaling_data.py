#!/usr/bin/env python3
"""
Generate scaling data for vec_add, vec_dot, matmul with varying N.
Collects: Qubits, Depth, T-count for both QFT and Ripple backends.
"""

import subprocess
import sys
import os
import csv
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

BITS = 8
N_VALUES_VECADD = [2, 4, 8, 16, 32, 64, 128]
N_VALUES_VECDOT = [2, 4, 8, 16, 32, 64]
N_VALUES_MATMUL = [2, 4, 8]
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
    # Simple initialization: A and B are all 1s
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
    """Compile C code and extract metrics."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(c_code)
        c_path = f.name

    try:
        cmd = [sys.executable, "pipeline.py", c_path, "--bits", str(BITS), "--adder", adder]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        output = result.stdout + result.stderr

        # Parse metrics from output
        metrics = {
            'qubits': 0,
            'depth': 0,
            't_count': 0,
            'arb_rot': 0,
        }

        for line in output.split('\n'):
            if 'num_qubits' in line:
                metrics['qubits'] = int(line.split(':')[1].strip())
            elif line.strip().startswith('depth'):
                metrics['depth'] = int(line.split(':')[1].strip())
            elif 'T-count' in line and 'total' in line:
                # T-count     : 123 (total)
                val = line.split(':')[1].split('(')[0].strip()
                metrics['t_count'] = int(val)
            elif 'arb. rot.' in line:
                val = line.split(':')[1].split('(')[0].strip()
                metrics['arb_rot'] = int(val)

        return metrics
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for {adder}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    finally:
        os.unlink(c_path)


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

    for op_name, gen_func in generators.items():
        print(f"\n=== {op_name} ===")
        for n in n_values_map[op_name]:
            print(f"N={n}...", end=" ", flush=True)

            c_code = gen_func(n)

            for adder in ['qft', 'ripple']:
                metrics = compile_and_get_metrics(c_code, adder)
                if metrics:
                    results.append({
                        'operation': op_name,
                        'N': n,
                        'backend': adder,
                        **metrics
                    })
                    print(f"{adder}:OK", end=" ", flush=True)
                else:
                    print(f"{adder}:FAIL", end=" ", flush=True)
            print()

    # Save to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
