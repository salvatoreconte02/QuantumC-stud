# Paper Guide: Extending QuantumC with Vector/Matrix Support

Documento guida per la stesura del paper. Struttura basata sul paper QuantumC originale (SAC '26).

---

## Note dal Tutor

1. **File separati per sezione** → usare `\input{sezione}` nel main
2. **Tabelle e grafici in LaTeX** (no screenshot)
3. **Immagini in LaTeX** se possibile

### Struttura file LaTeX suggerita:
```
docs/latex/
├── main.tex              # File principale con \input
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex  # include related work + contributions
│   ├── background.tex
│   ├── experimental.tex  # o "pipeline.tex"
│   ├── results.tex
│   └── conclusion.tex
├── figures/
│   └── pipeline.pdf      # diagrammi
└── references.bib
```

---

## 1. Abstract

**STATUS: COMPLETATO**

Modern quantum workloads are increasingly driven by linear algebra–centric applications, such as machine learning and data processing, which naturally rely on vector and matrix computations. However, existing high-level quantum compilers primarily target scalar arithmetic, limiting their expressiveness and applicability to these emerging domains. In this work, we extend QuantumC to support the compilation of C programs featuring vector and matrix operations into quantum circuits. Our approach introduces pattern-based recognition of common vector and matrix computations and integrates a dedicated intermediate representation into the existing QuantumC pipeline, while preserving compatibility with scalar code and hybrid programs. The extended compiler supports compile-time initialization of vector and matrix data and produces a unified quantum circuit representation. As an analysis enabled by this extension, we compare two alternative arithmetic backends—QFT-based and ripple-carry adders—using structural circuit metrics such as qubit count, circuit depth, and gate composition. Experimental results on vector and matrix benchmarks highlight clear trade-offs between the two approaches.

---

## 2. Introduction

**STATUS: COMPLETATO**

Quantum computing is rapidly evolving from theoretical exploration toward practical applications. Among the most promising use cases are machine learning and data processing workloads, which fundamentally rely on vector and matrix computations—the core building blocks of linear algebra. Algorithms such as HHL for linear systems and quantum machine learning models leverage these operations to achieve potential speedups over classical approaches. As quantum hardware matures and error-correction techniques improve, the ability to express and compile such computations efficiently becomes increasingly important.

However, current high-level quantum compilers primarily support scalar integer arithmetic, leaving a significant gap in expressiveness for developers who wish to leverage quantum computing for linear algebra workloads. While these compilers successfully demonstrate the feasibility of translating classical code into quantum circuits, they cannot directly handle the vector and matrix operations that dominate modern computational workloads. This limitation forces developers to either manually decompose their algorithms into scalar operations or abandon high-level compilation altogether.

**Related Work.** QuantumC introduced an MLIR-based compilation pipeline that translates C programs with scalar integer operations into quantum circuits expressed in OpenQASM. While this approach successfully demonstrates the feasibility of high-level quantum compilation from standard C, it is limited to scalar arithmetic and does not support array, vector, or matrix operations. QHLS explores a similar C-to-quantum synthesis approach but relies on ad-hoc transformations instead of MLIR, which limits its modularity and optimization potential. In contrast, our work extends QuantumC to support vector and matrix operations while leveraging the MLIR infrastructure for modular dialect design.

**This Work.** We extend QuantumC to support vector and matrix operations while maintaining full compatibility with existing scalar code. Our extension introduces pattern-based recognition for common linear algebra operations—vector addition, dot product, and matrix multiplication—directly from the C source AST. These patterns are identified through structural analysis of loop constructs and array access patterns, enabling the compiler to capture high-level semantics that would otherwise be lost through naive loop unrolling.

To represent and transform these operations, we introduce two dedicated intermediate representations integrated into the existing MLIR pipeline. The *VecMat IR* captures vector and matrix operations at a high level of abstraction, while the *QAR (Quantum Arithmetic Representation)* expresses arithmetic operations in a form suitable for quantum circuit generation. The extended compiler supports hybrid programs that combine scalar and vector/matrix operations, automatically merging the separate compilation paths into a unified quantum circuit. Additionally, compile-time initialization of vector and matrix data is supported, allowing constant arrays to be directly encoded in the generated circuit. As an analysis enabled by this extension, we compare two alternative arithmetic backends—QFT-based and ripple-carry adders—using structural circuit metrics.

**Contributions.** This work makes several contributions to the field of high-level quantum compilation. We introduce pattern-based recognition of vector and matrix operations directly from C source code, enabling the compiler to identify common linear algebra computations such as vector addition, dot product, and matrix multiplication. To represent these operations throughout the compilation process, we design two dedicated intermediate representations—VecMat IR and QAR—that integrate seamlessly into the existing MLIR pipeline. Our extended compiler supports hybrid programs that combine scalar and vector/matrix operations, with compile-time initialization of array data. Finally, we present a comparative analysis of two arithmetic backends, QFT-based and ripple-carry adders, using structural circuit metrics to highlight their respective trade-offs.

**Structure of the paper.** The remainder of this paper is organized as follows. Section 2 provides background on QuantumC and quantum arithmetic implementations. Section 3 describes our extended compilation pipeline, including pattern recognition and the VecMat/QAR intermediate representations. Section 4 presents experimental results on vector and matrix benchmarks and the arithmetic backend comparison. Section 5 concludes and discusses future work.

---

## 3. Background

**STATUS: COMPLETATO**

This section introduces the foundational concepts of quantum computation relevant to our work, provides an overview of the QuantumC compiler that we extend, and describes the quantum arithmetic implementations that we compare.

**2.1 Quantum Computing Preliminaries.** In quantum computing, the fundamental unit of information is the qubit, whose state is a superposition of two basis states: |ψ⟩ = α|0⟩ + β|1⟩, where α and β are complex amplitudes satisfying |α|² + |β|² = 1. Multiple qubits can be grouped into quantum registers to represent integer values, analogous to how classical variables are stored in processor registers.

Quantum computation proceeds by applying quantum gates, which are unitary operators that evolve the system state. In the circuit model, qubits are depicted as horizontal lines, gates as operations applied to one or more qubits, and computation flows from left to right. Common gates include single-qubit operations (X, H, phase rotations) and multi-qubit operations (CNOT, Toffoli).

Two fundamental constraints distinguish quantum from classical computation. First, quantum operations are inherently reversible: since gates are unitary, applying the adjoint U† recovers the original state. This implies that variables cannot be directly overwritten; intermediate values must be preserved or explicitly uncomputed. Second, the no-cloning theorem forbids duplicating arbitrary quantum states, eliminating the classical notion of fan-out for variable copying.

These constraints have direct implications for compiling classical programs to quantum circuits. The reversibility requirement naturally aligns with Static Single Assignment (SSA) form, where each variable is assigned exactly once. The quality and cost of a quantum circuit are commonly evaluated using metrics such as qubit count (width), circuit depth (critical path length), and gate count.

**2.2 QuantumC Overview.** QuantumC is an MLIR-based compiler that translates C programs into quantum circuits expressed in OpenQASM format. The compilation proceeds through five stages: (1) the C source is parsed using Clang to produce a JSON abstract syntax tree; (2) the AST is converted into Python dataclass representations; (3) a classical MLIR module in SSA form is generated using xDSL; (4) classical operations are translated into quantum operations; and (5) the quantum MLIR is lowered to a Qiskit circuit and exported as OpenQASM.

The use of MLIR provides a modular infrastructure with well-defined dialects and progressive lowering, enabling optimizations at multiple abstraction levels. The SSA form adopted in the classical MLIR stage ensures that each value is defined exactly once, which naturally preserves reversibility by preventing variable overwriting.

QuantumC supports scalar integer variables with configurable bitwidth, arithmetic operations (addition, subtraction, multiplication, division), comparison operators, and control flow constructs including conditional branches and for-loops with static unrolling. However, the current implementation focuses on scalar integer operations and does not directly support array declarations, vector operations, or matrix computations, limiting its applicability to linear algebra workloads that are central to quantum machine learning and algorithms such as HHL.

**2.3 Quantum Arithmetic.** Arithmetic operations are fundamental building blocks in the compilation of classical programs to quantum circuits. Addition is the most basic operation, and two main approaches exist for its quantum implementation.

QFT-based addition: The approach introduced by Draper performs addition in the Fourier basis. The target register is first transformed via the Quantum Fourier Transform (QFT), then the addend is incorporated through controlled phase rotations, and finally the inverse QFT is applied to return to the computational basis. This method does not require ancilla qubits for carry propagation, as the carry information is encoded in the phase of the quantum state.

Ripple-carry addition: The approach proposed by Cuccaro et al. follows the classical model of carry propagation. It uses a sequence of majority and unmajority gates, implemented with Toffoli (CCX) and CNOT gates, to propagate the carry bit through the register. This method requires ancilla qubits to store intermediate carry values but operates entirely in the computational basis without requiring the QFT.

Other arithmetic operations in our compilation flow are built upon addition. Subtraction is performed via two's complement negation followed by addition. Multiplication can be implemented through repeated additions (shift-and-add) or using QFT-based techniques with controlled-controlled phase rotations.

The two addition approaches present different trade-offs in terms of qubit count, circuit depth, and gate composition. These trade-offs depend on the target hardware characteristics, such as native gate sets and qubit connectivity. Our extended compiler supports both backends, and we present a comparative analysis in Section 4.

**Citazioni:** nielsen-chuang, quantumc, mlir, xdsl, draper-qft, cuccaro-ripple, hhl

---

## 4. Extended Compilation Pipeline

**STATUS: COMPLETATO**

### Contenuto scritto:

La sezione descrive l'estensione della pipeline QuantumC per supportare operazioni vettoriali e matriciali. È strutturata in 5 sottosezioni:

**3.1 (intro senza titolo):** Overview dell'architettura estesa. Spiega come i pattern riconosciuti vengono estratti dall'AST e processati attraverso un path dedicato (VecMat IR → QAR → Quantum Ops), mentre il codice scalare rimanente segue la pipeline standard. I due path convergono in una fase di merge.

**3.1 Pattern Recognition:** Descrive come l'algoritmo analizza i loop `for` nell'AST cercando strutture specifiche. Condizioni di matching: header canonico (init=0, comparison vs bound, unit increment), array access con loop variable, struttura aritmetica attesa. Tabella con i 3 pattern supportati (vec_add, vec_dot, matmul).

**3.2 Intermediate Representations:**
- **VecMat IR:** Operazioni VecAdd, VecDot, MatMul con attributi (dest, lhs, rhs, dimensions, elem_bits). Mantiene constant table per inizializzazioni compile-time.
- **QAR:** Bridge tra VecMat e quantum ops. Mapping 1:1 (QarMapAdd, QarDot, QarMatMul). Propaga constant table.
- **Example:** Figura che mostra C Source → VecMat IR → QAR → Quantum Ops per vector addition.

**3.3 Lowering to Quantum Operations:** Register allocation (N registri per vettore, M×N per matrice), inizializzazione con valori costanti o zero. Lowering delle 3 operazioni QAR a quantum.addi/muli. Result map per tracciare SSA values finali.

**3.4 Hybrid Compilation:** Path separation, merge phase (inserimento init all'inizio, arithmetic prima del return, SSA remapping). Return value resolution tramite return hints e result map lookup.

### Figure e Tabelle:
- **Figura 1 (fig:pipeline):** Pipeline estesa - PLACEHOLDER da creare
- **Figura 2 (fig:lowering-example):** Lowering example C→VecMat→QAR→Quantum - COMPLETATA (TikZ)
- **Tabella 1 (tab:patterns):** Pattern supportati - COMPLETATA

### File LaTeX:
- `docs/latex/sections/experimental.tex`
- `docs/latex/figures/lowering-example.tex`

---

## 5. Results and Discussion

**STATUS: DA FARE**

### Struttura suggerita:

#### 5.1 Experimental Setup
- Tools: Clang, xDSL, Qiskit (con versioni)
- Hardware: specs della macchina
- Benchmark suite: lista dei file di test usati

**Test files disponibili:**
- `tests/scalar/` - test scalari
- `tests/vector/` - vec_add, vec_dot
- `tests/matrix/` - matmul
- `tests/hybrid/` - programmi misti

#### 5.2 Vector/Matrix Compilation Results
Mostrare che la compilazione funziona:
- Tabella con programmi di test
- Metriche: qubit count, depth, gate count
- Confronto con unrolling manuale (se applicabile)

#### 5.3 Arithmetic Backend Comparison

**IMPORTANTE - Cosa dire nel paper:**
- Il confronto usa metriche strutturali a livello di gate logici
- Metriche: qubit count, circuit depth, gate composition (cx, t, h, p, etc.)
- Entrambi i backend passano per la stessa pipeline
- NON menzionare Clifford+T

**Dati da mostrare:**
- Tabella comparativa QFT vs Ripple per diversi benchmark
- Metriche: num_qubits, depth, size, count_cx, count_t, count_h, count_p
- Score composito (opzionale)

**Interpretazione risultati:**
- QFT: meno qubit, meno depth, usa rotazioni di fase (cp, p)
- Ripple: più qubit (carry), più depth, usa cx e t gates
- Trade-off dipende dal target hardware

**Comando per generare dati:**
```bash
python3 pipeline.py tests/scalar/add.c --adder both --bits 4
```

#### 5.4 Discussion
- Vantaggi dell'approccio pattern-based
- Limitazioni attuali
- Quando usare QFT vs Ripple

---

## 6. Conclusions

**STATUS: DA FARE**

### Struttura suggerita (1 pagina max):

#### Riassunto contributi
- Estensione QuantumC con supporto vector/matrix
- Pattern recognition + IR dedicate
- Compilazione ibrida
- Confronto backend aritmetici

#### Limitazioni
- Pattern limitati (solo vec_add, vec_dot, matmul)
- No supporto per dimensioni runtime
- Confronto backend a livello gate logici (no fault-tolerant analysis)

#### Future Work
- Più pattern (convolution, transpose, etc.)
- Ottimizzazioni circuit-level
- Supporto per altri backend
- Analisi fault-tolerant (Clifford+T decomposition)

---

## Note e Decisioni

| Argomento | Decisione |
|-----------|-----------|
| Confronto QFT vs Ripple | Confronto a livello di gate logici (metriche strutturali) |
| Terminologia | NON usare "Clifford+T" - il confronto usa gate logici pre-sintesi |
| Core del paper | Estensione vector/matrix, il confronto backend è secondario |
| Coerenza | Usare sempre "structural circuit metrics" o "logical gate level" |
| **Riferimenti** | Inserire `\cite{}` man mano durante la scrittura, non alla fine. Verificare sempre `references.bib` prima di aggiungere nuove entry. |
| **Stile introduzione** | Stile "flat" come paper QuantumC originale: "Related Work.", "This Work.", "Contributions." come titoli in grassetto inline, senza `\subsection{}` |

---

## Riferimenti Chiave da Citare

### Fondamentali
- **QuantumC originale**: Lancellotti, Rosnati, Pagliochini, Agosta. "Quantum Circuit Synthesis from C via Multi-Level Intermediate Representation." SAC '26.
- **MLIR**: Lattner et al. "MLIR: Scaling compiler infrastructure for domain specific computation." CGO '21.
- **xDSL**: Fehr et al. "xDSL: Sidekick compilation for SSA-based compilers." CGO '25.

### Quantum Arithmetic
- **QFT adder**: Draper. "Addition on a Quantum Computer." 2000.
- **QFT arithmetic**: Şahin. "Quantum arithmetic operations based on quantum fourier transform on signed integers." 2020.
- **Ripple-carry**: Cuccaro et al. "A new quantum ripple-carry addition circuit." 2004.

### Related Work
- **QHLS**: Lu, Pilato, Basu. "QHLS: An HLS framework to convert high-level descriptions to quantum circuits." 2024.
- **OpenQASM**: Cross et al. "OpenQASM 3: A broader and deeper quantum assembly language." 2022.

### Background Quantum Computing
- **Nielsen & Chuang**: "Quantum Computation and Quantum Information." 2011.
- **Qiskit**: Qiskit Development Team. 2025.

---

## Checklist Pre-Submission

- [ ] Abstract coerente con contenuto
- [ ] Nessuna menzione di Clifford+T nel confronto backend
- [ ] Figure della pipeline chiare
- [ ] Tabelle risultati complete
- [ ] Riferimenti completi
- [ ] Codice disponibile su GitHub
