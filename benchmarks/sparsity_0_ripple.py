#!/usr/bin/env python3
"""Run only the 0% sparsity ripple-carry test and append to CSV."""
import os, sys, csv, random, subprocess, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SIZE = 16
BITS = 8

def generate_sparse_matrix(size, num_zeros, seed=None):
    if seed is not None:
        random.seed(seed)
    total = size * size
    if num_zeros > total:
        num_zeros = total
    values = [random.randint(1, 7) for _ in range(total - num_zeros)]
    values.extend([0] * num_zeros)
    random.shuffle(values)
    return [values[i*size:(i+1)*size] for i in range(size)]

def matrix_to_c_init(matrix):
    rows = ["{" + ", ".join(str(x) for x in row) + "}" for row in matrix]
    return "{" + ", ".join(rows) + "}"

def generate_matmul_c_code(A, B, size):
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

os.chdir(Path(__file__).parent.parent)

num_zeros = 0
A = generate_sparse_matrix(SIZE, num_zeros, seed=42 + num_zeros)
B = generate_sparse_matrix(SIZE, num_zeros, seed=123 + num_zeros)
c_code = generate_matmul_c_code(A, B, SIZE)

with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
    f.write(c_code)
    c_path = f.name

print(f"Running 0% sparsity ripple-carry (timeout=10h)...", flush=True)
start = time.time()

try:
    cmd = [sys.executable, "pipeline.py", c_path, "--bits", str(BITS), "--adder", "ripple"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
    output = result.stdout + result.stderr
    elapsed = time.time() - start

    metrics = {'qubits': 0, 'depth': 0, 't_count': 0, 't_depth': 0}
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

    print(f"OK ({elapsed:.1f}s) - qubits={metrics['qubits']:,}, t_count={metrics['t_count']:,}", flush=True)

    # Append to CSV
    csv_path = Path(__file__).parent / "sparsity_results_16x16.csv"
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([0.0, 0, 'ripple', metrics['qubits'], metrics['depth'], metrics['t_count'], metrics['t_depth']])
    print(f"Appended to {csv_path}")

except subprocess.TimeoutExpired:
    elapsed = time.time() - start
    print(f"TIMEOUT ({elapsed:.1f}s)")
finally:
    os.unlink(c_path)
