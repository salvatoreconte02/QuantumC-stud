# QuantumC Extended

**QuantumC Extended** is an extended version of [QuantumC](https://github.com/leonardopagliochini/QuantumC), a modular high-level synthesis framework that compiles classical C programs into quantum circuits (OpenQASM).

This fork adds support for **vector/matrix operations**, **multiple arithmetic backends**, and **hybrid scalar+vector compilation**.

---

## Extensions over Original QuantumC

| Extension | Description |
|-----------|-------------|
| **Vector Operations** | Pattern-based recognition of `vec_add`, `vec_dot` loops |
| **Matrix Operations** | Pattern-based recognition of `matmul` triple-nested loops |
| **Ripple-Carry Backend** | Alternative to QFT arithmetic (`--adder ripple`) |
| **Hybrid Compilation** | Mixed scalar + vector/matrix programs in a single circuit |
| **Compile-time Arrays** | Support for `int a[4] = {1,2,3,4};` and 2D matrices |
| **Metrics & Comparison** | Automatic scoring and comparison between backends |

---

## Quick Start

```bash
# Install dependencies
pip install xdsl qiskit qiskit-aer

# Basic compilation
python pipeline.py tests/scalar/add.c

# Compile and simulate
python pipeline.py tests/scalar/add.c --run

# Use ripple-carry arithmetic
python pipeline.py tests/scalar/add.c --adder ripple --run

# Compare QFT vs Ripple-carry
python pipeline.py tests/scalar/add.c --adder both
```

---

## Test Suite

The test suite is organized into 5 categories with 18 tests total:

| Category | Tests | Description |
|----------|-------|-------------|
| `scalar` | 3 | Basic scalar integer operations |
| `vector` | 3 | Vector addition and dot product |
| `matrix` | 3 | Matrix multiplication |
| `hybrid` | 3 | Mixed scalar + vector/matrix programs |
| `comparison` | 6 | Backend comparison tests |

### Running Tests

```bash
# Run all tests
python test_c_file.py --all

# Run a specific category
python test_c_file.py --category scalar
python test_c_file.py --category vector
python test_c_file.py --category matrix

# Run with different backends
python test_c_file.py --all --adder ripple
python test_c_file.py --all --adder both

# Run a single test
python test_c_file.py tests/scalar/add.c
```

---

## Pipeline Architecture

The compilation pipeline follows these stages:

```
C Source
    |
[Stage 1] clang -ast-dump=json -> JSON AST
    |
[Stage 2] AST to Python dataclasses
    |
[Stage 3] Dataclasses to Classical MLIR (SSA form via xDSL)
    |
[Stage 4] Classical MLIR to Quantum MLIR
    |     + Hybrid path for vector/matrix operations
    |
[Stage 5] Quantum MLIR to QASM (via Qiskit)
```

### Hybrid Vector/Matrix Path

When the AST contains recognized patterns (vec_add, vec_dot, matmul), a parallel path is activated:

1. **Pattern matching**: Recognizes loop patterns and extracts VecMatModule
2. **VecMat to QAR**: Converts to Quantum Arithmetic Representation
3. **QAR to Quantum MLIR**: Generates quantum operations
4. **Merge**: Combines scalar and vector paths

---

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

---

## Command Line Options

| Flag | Description |
|------|-------------|
| `--bits N` | Set bits per quantum integer (default: 16) |
| `--run` | Simulate the circuit after generation |
| `--adder {qft,ripple,both}` | Arithmetic backend selection |
| `--verbose` | Print detailed debug information |
| `--pretty` | Pretty-print reconstructed C code |
| `--time` | Report compilation timing |

---

## Output Artifacts

| Stage | Output Path |
|-------|-------------|
| JSON AST | `json_out/<name>.json` |
| Classical MLIR | `mlir_out/<name>_classical.mlir` |
| Quantum MLIR | `quantum_mlir_out/<name>_quantum_*.mlir` |
| OpenQASM | `output/<name>_<adder>.qasm` |

---

## Requirements

- Python 3.10+
- Clang with `-ast-dump=json` support
- xDSL, Qiskit, Qiskit-Aer

```bash
pip install xdsl qiskit qiskit-aer
```

---

## Repository Structure

| Folder | Purpose |
|--------|---------|
| `tests/` | Organized test suite (scalar, vector, matrix, hybrid, comparison) |
| `c_code/others/` | Legacy tests from upstream |
| `my_extensions/` | Vector/matrix pattern recognition and lowering |
| `step1_c_to_ast/` | AST generation with Clang |
| `step2_ast_to_dataclasses/` | AST parsing to Python dataclasses |
| `step3_dataclasses_to_mlir/` | Classical MLIR generation |
| `step4_mlir_to_quantum_mlir/` | Quantum dialect translation |
| `step5_quantum_mlir_to_qasm/` | QASM generation via Qiskit |

---

## Credits

This project extends the original [QuantumC](https://github.com/leonardopagliochini/QuantumC) by Leonardo Pagliochini and Francesco Rosnati.

### Original QuantumC Authors
- Leonardo Ignazio Pagliochini - [GitHub](https://github.com/leonardopagliochini)
- Francesco Rosnati - [GitHub](https://github.com/RosNaviGator)

### Extensions
- Salvatore Conte
- Simone Cosenza
