# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is an **extension** of [QuantumC](https://github.com/leonardopagliochini/QuantumC), an MLIR-based compiler that translates C programs into quantum circuits (OpenQASM).

### Original QuantumC (upstream)
- Compiles C with **scalar integer operations only**
- Uses QFT-based quantum arithmetic
- 5-stage pipeline: C → JSON AST → Dataclasses → Classical MLIR → Quantum MLIR → QASM

### Extensions in this fork
1. **Vector/Matrix support** (`my_extensions/`): Pattern-based recognition of `vec_add`, `vec_dot`, `matmul` operations
2. **Ripple-Carry backend**: Alternative to QFT arithmetic (`--adder ripple`)
3. **Hybrid compilation**: Mixed scalar + vector/matrix programs in a single circuit
4. **Compile-time array initialization**: `int a[4] = {1,2,3,4};` and 2D matrices
5. **Metrics & comparison system**: Automatic scoring and comparison between backends (`--adder both`)

### Files added/modified
| Type | Files |
|------|-------|
| **New (extensions)** | `my_extensions/*` (vecmat_ir, vecmat_lowering, qar_ir, vecmat_to_qar, qar_to_quantum_mlir) |
| **Modified** | `pipeline.py` (hybrid path, adder selection, metrics), `step5_quantum_mlir_to_qasm/*` (ripple-carry, metrics) |
| **Original** | `step1_*`, `step2_*`, `step3_*`, `step4_*` (mostly unchanged) |

## Build & Run Commands

```bash
# Install dependencies
pip install xdsl qiskit qiskit-aer

# Basic compilation
python pipeline.py                              # Compile c_code/try.c (default)
python pipeline.py c_code/example.c             # Compile specific file

# Options
python pipeline.py c_code/example.c --bits 8    # 8-bit integers (default: 16)
python pipeline.py c_code/example.c --run       # Simulate after generation
python pipeline.py c_code/example.c --verbose   # Debug output
python pipeline.py c_code/example.c --pretty    # Print reconstructed C
python pipeline.py c_code/example.c --time      # Report timing

# Arithmetic backend selection
python pipeline.py c_code/example.c --adder qft       # QFT-based (default)
python pipeline.py c_code/example.c --adder ripple    # Ripple-carry adder
python pipeline.py c_code/example.c --adder both      # Compare both backends

# Test classical vs quantum output
python test_c_file.py c_code/example.c
```

Requires: Python 3.10+, clang with `-ast-dump=json` support.

## Compilation Pipeline Architecture

The compiler has a 5-stage pipeline with an additional hybrid path for vector/matrix operations:

```
C Source
    ↓
[Stage 1] clang -ast-dump=json → JSON AST
    ↓
[Stage 2] step2_ast_to_dataclasses/c_ast.py: parse_ast() → TranslationUnit
    ↓
[Stage 3] step3_dataclasses_to_mlir/mlir_generator.py → Classical MLIR (SSA form via xDSL)
    ↓
[Stage 4] step4_mlir_to_quantum_mlir/quantum_translate.py → Quantum Dialect MLIR
    ↓
[Stage 5] step5_quantum_mlir_to_qasm/qasm_generator.py → Qiskit circuit → OpenQASM
```

### Hybrid Vector/Matrix Path (my_extensions/)

When the AST contains recognized patterns (vec_add, vec_dot, matmul), a parallel path is activated:

1. **Pattern matching** (`vecmat_lowering.py`): Recognizes loop patterns and extracts VecMatModule
2. **VecMat → QAR** (`vecmat_to_qar.py`): Converts to Quantum Arithmetic Representation
3. **QAR → Quantum MLIR** (`qar_to_quantum_mlir.py`): Generates quantum operations with `result_map`
4. **Merge** (`pipeline.py:_merge_scalar_and_vec_quantum`): Combines scalar and vector paths

The hybrid path avoids unrolling loops for vector operations, reducing circuit size.

### Pattern Recognition

Recognized C patterns in `vecmat_lowering.py`:
- **Vector addition**: `for (i=0; i<N; i++) c[i] = a[i] + b[i];`
- **Dot product**: `s=0; for (i=0; i<N; i++) s += a[i] * b[i];`
- **Matrix multiply**: Triple nested loop `i-j-k` with `C[i][j] += A[i][k] * B[k][j]`
- **Constant arrays**: `int a[4] = {1,2,3,4};` and `int A[2][2] = {{1,2},{3,4}};`

## Key Design Concepts

### SSA Value Tracking
The quantum translator (`quantum_translate.py`) maintains SSA semantics:
- Each operation produces a fresh quantum register
- `ValueInfo(reg, version, expr)` tracks register versions
- Overwritten values can be recomputed from cached expressions

### Quantum Arithmetic
Two backends available (`step5_quantum_mlir_to_qasm/`):
- **QFT-based** (`q_arithmetics.py`): Uses Quantum Fourier Transform for ancilla-efficient arithmetic
- **Ripple-carry** (`q_arithmetics.py`): Classical-style carry propagation with Toffoli gates

### Control Flow
- `if/else` → `cf.cond_br` + basic blocks with controlled quantum operations
- `for` loops → Static unrolling (max 10 iterations)
- Conditional operations use control qubits from branch conditions

## Output Artifacts

| Stage | Output Path |
|-------|------------|
| 1 | `json_out/<name>.json` |
| 3 | `mlir_out/<name>_classical.mlir` |
| 4 scalar | `quantum_mlir_out/<name>_quantum_scalar.mlir` |
| 4 vector | `quantum_mlir_out/<name>_quantum_vec.mlir` |
| 4 merged | `quantum_mlir_out/<name>_quantum_final.mlir` |
| 5 | `output/<name>_<adder>.qasm` |

## Supported C Subset

**Supported:**
- Integer variables (signed, configurable bitwidth)
- Arithmetic: `+`, `-`, `*`, `/`
- Logical: `&&`, `||`, `!`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Unary: `+`, `-`, `~`, `++`, `--`
- Control flow: `if`, `else`, `for` (static unroll)
- Arrays (1D vectors, 2D matrices) with compile-time initialization
- Return statements

**Not supported:** Pointers, floats, function calls, structs, dynamic allocation, while loops.

## Key Files Reference

| File | Purpose |
|------|---------|
| `pipeline.py` | Main entry point, orchestrates all stages |
| `step2_ast_to_dataclasses/c_ast.py` | AST node definitions and parsing |
| `step3_dataclasses_to_mlir/mlir_generator.py` | Classical MLIR generation |
| `step4_mlir_to_quantum_mlir/quantum_translate.py` | Quantum dialect translation |
| `step4_mlir_to_quantum_mlir/quantum_dialect.py` | Quantum operation definitions |
| `step5_quantum_mlir_to_qasm/qasm_generator.py` | Circuit generation |
| `step5_quantum_mlir_to_qasm/q_arithmetics.py` | QFT-based quantum arithmetic |
| `my_extensions/vecmat_lowering.py` | C AST → VecMat pattern matching |
| `my_extensions/qar_to_quantum_mlir.py` | QAR → Quantum MLIR with result_map |

## Test Programs

- `c_code/try.c` - Basic scalar example
- `c_code/vect_and_matrix/vec_add.c` - Pure vector addition
- `c_code/vect_and_matrix/vec_dot.c` - Dot product
- `c_code/vect_and_matrix/matmul.c` - Matrix multiplication
- `c_code/vect_and_matrix/hybrid_*.c` - Mixed scalar + vector programs

## Git Commit Guidelines

- **NEVER reference Claude in commit messages** (no Co-Authored-By, no mentions of AI assistance)
- Write commit messages in Italian or English, describing what was changed and why
