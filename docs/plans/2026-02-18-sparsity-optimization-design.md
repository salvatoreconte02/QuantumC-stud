# Sparsity Optimization Design

**Date**: 2026-02-18
**Status**: Approved
**Target**: Section 4 (Results) of HPQCI 2026 paper

## Goal

Implement compile-time sparsity optimization for matrix operations in QuantumC. When matrix elements are zero, skip the corresponding quantum multiplication and addition operations to reduce gate count and circuit depth.

## Scope

### Operations to optimize
- **matmul**: `C[i][j] += A[i][k] * B[k][j]` - skip when `A[i][k] == 0` OR `B[k][j] == 0`
- **vec_dot**: `s += a[i] * b[i]` - skip when `a[i] == 0` OR `b[i] == 0`

### Out of scope
- **vec_add**: Only has addition, less benefit from optimization
- **Qubit allocation optimization**: Registers are still allocated for zero elements

## Implementation

### File to modify
`my_extensions/vecmat_to_quantum_mlir.py`

### Changes

1. **Propagate `const_arrays`** through the call chain:
   - `from_vecmat_to_quantum_mlir` → `_emit_vecmat_ops` → `_lower_matmul` / `_lower_vec_dot`

2. **Add sparsity check in `_lower_matmul`** (inside triple loop):
   ```python
   a_vals = const_arrays.get(op.lhs)
   b_vals = const_arrays.get(op.rhs)

   if a_vals is not None and b_vals is not None:
       a_val = a_vals[idx(i, kk, k)]
       b_val = b_vals[idx(kk, j, n)]
       if a_val == 0 or b_val == 0:
           continue  # skip QMuliOp + QAddiOp
   ```

3. **Add sparsity check in `_lower_vec_dot`** (inside loop):
   ```python
   a_vals = const_arrays.get(op.lhs)
   b_vals = const_arrays.get(op.rhs)

   if a_vals is not None and b_vals is not None:
       if a_vals[i] == 0 or b_vals[i] == 0:
           continue  # skip QMuliOp + QAddiOp
   ```

## Benchmark Script

**Location**: `benchmarks/sparsity_benchmark.py`

**Parameters**:
- Matrix size: 4x4 (fixed)
- Bit width: 8 bits
- Backends: QFT and Ripple-carry
- Sparsity levels: 0%, 25%, 50%, 75%, 87.5%, 93.75%

**Output**: CSV file with columns:
```
sparsity,zeros,backend,gates,depth,qubits
```

**Generates 3 graphs** for the paper:
1. Gates vs Sparsity (two lines: QFT, Ripple)
2. Depth vs Sparsity (two lines: QFT, Ripple)
3. Qubits vs Sparsity (two lines: QFT, Ripple)

## Expected Results

- **Gates**: Decrease proportionally with sparsity
- **Depth**: Decrease proportionally with sparsity
- **Qubits**: Remain constant (allocation not optimized)

## Notes

- All values are known at compile-time (tool requirement)
- Optimization applies to all array-initialized matrices/vectors
- Correctness preserved: 0 * x = 0, acc + 0 = acc
