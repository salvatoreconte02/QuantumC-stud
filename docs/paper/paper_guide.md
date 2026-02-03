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

**STATUS: DA FARE**

### Struttura suggerita:

#### 3.1 QuantumC Overview
Riassunto breve del paper originale:
- Pipeline 5 stadi: C → AST → Dataclasses → Classical MLIR → Quantum MLIR → QASM
- Supporto: variabili scalari, aritmetica, if/else, for loops
- Limitazioni: solo operazioni scalari, no array/vector/matrix

#### 3.2 Vector and Matrix Operations
Operazioni target della tua estensione:
- Vector addition: `c[i] = a[i] + b[i]`
- Dot product: `s += a[i] * b[i]`
- Matrix multiplication: `C[i][j] += A[i][k] * B[k][j]`
- Perché sono importanti per quantum computing (quantum ML, HHL, etc.)

#### 3.3 Quantum Arithmetic Implementations
Due approcci per aritmetica quantistica:
- **QFT-based**: usa Quantum Fourier Transform, meno qubit, rotazioni di fase
- **Ripple-carry**: stile classico con carry propagation, più gate ma diversa struttura
- Trade-off tra i due (anticipare senza dettagli, approfondire in Results)

### Riferimenti utili per Background:
- Paper QuantumC (Section 2)
- Draper QFT adder
- Cuccaro ripple-carry adder

---

## 4. Extended Compilation Pipeline

**STATUS: DA FARE**

### Struttura suggerita:

#### 4.1 Architecture Overview
Figura della pipeline estesa:
```
C Source (con vector/matrix)
    ↓
[Stages 1-3] Pipeline QuantumC standard
    ↓
[Pattern Recognition] vecmat_lowering.py
    ↓
    ├── Scalar path (standard)
    └── Vector/Matrix path (nuovo)
            ↓
        VecMat IR
            ↓
        QAR (Quantum Arithmetic Representation)
            ↓
        Quantum MLIR
    ↓
[Merge] Combinazione scalar + vector
    ↓
[Stage 5] QASM generation (con scelta backend)
```

#### 4.2 Pattern Recognition
Come vengono riconosciuti i pattern in `vecmat_lowering.py`:
- Analisi AST per identificare loop structure
- Pattern matching per vec_add, vec_dot, matmul
- Estrazione dimensioni e operandi

**Pattern supportati:**
| Pattern | C Code | Riconoscimento |
|---------|--------|----------------|
| vec_add | `for(i) c[i] = a[i] + b[i]` | Loop singolo, accesso array indicizzato |
| vec_dot | `for(i) s += a[i] * b[i]` | Loop singolo, accumulo scalare |
| matmul | `for(i,j,k) C[i][j] += A[i][k] * B[k][j]` | Triple loop annidato |

#### 4.3 VecMat Intermediate Representation
Descrivere `vecmat_ir.py`:
- Operazioni: VecAdd, VecDot, MatMul
- Attributi: dimensioni, nomi registri, costanti
- Come si integra con la pipeline

#### 4.4 QAR (Quantum Arithmetic Representation)
Descrivere `qar_ir.py` e `vecmat_to_qar.py`:
- Rappresentazione intermedia per operazioni aritmetiche quantum
- Lowering da VecMat a QAR
- Mapping su operazioni quantum (add, mul, etc.)

#### 4.5 Hybrid Compilation
Descrivere `_merge_scalar_and_vec_quantum` in `pipeline.py`:
- Come vengono combinati i path scalare e vector
- result_map per tracciare i registri
- Gestione delle dipendenze

#### 4.6 Compile-time Array Initialization
Supporto per:
- `int a[4] = {1, 2, 3, 4};`
- `int A[2][2] = {{1,2}, {3,4}};`
- Come vengono gestiti a livello IR

### File di riferimento per questa sezione:
- `my_extensions/vecmat_lowering.py`
- `my_extensions/vecmat_ir.py`
- `my_extensions/qar_ir.py`
- `my_extensions/vecmat_to_qar.py`
- `my_extensions/qar_to_quantum_mlir.py`
- `pipeline.py` (funzione `_merge_scalar_and_vec_quantum`)

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
