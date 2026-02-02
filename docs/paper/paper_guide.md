# Paper Guide: Extending QuantumC with Vector/Matrix Support

Documento guida per la stesura del paper. Struttura basata sul paper QuantumC originale (SAC '26).

---

## 1. Abstract

**STATUS: COMPLETATO**

Modern quantum workloads are increasingly driven by linear algebra–centric applications, such as machine learning and data processing, which naturally rely on vector and matrix computations. However, existing high-level quantum compilers primarily target scalar arithmetic, limiting their expressiveness and applicability to these emerging domains. In this work, we extend QuantumC to support the compilation of C programs featuring vector and matrix operations into quantum circuits. Our approach introduces pattern-based recognition of common vector and matrix computations and integrates a dedicated intermediate representation into the existing QuantumC pipeline, while preserving compatibility with scalar code and hybrid programs. The extended compiler supports compile-time initialization of vector and matrix data and produces a unified quantum circuit representation. As an analysis enabled by this extension, we compare two alternative arithmetic backends—QFT-based and ripple-carry adders—using structural circuit metrics such as qubit count, circuit depth, and gate composition. Experimental results on vector and matrix benchmarks highlight clear trade-offs between the two approaches.

---

## 2. Introduction

**STATUS: DA FARE**

### Struttura suggerita (seguendo QuantumC paper):

#### 2.1 Motivazione (1-2 paragrafi)
- Quantum computing si sta muovendo verso applicazioni reali
- ML e data processing sono workload chiave → richiedono operazioni vector/matrix
- I compilatori quantum attuali supportano solo aritmetica scalare
- Gap da colmare: permettere agli sviluppatori di scrivere codice C con vector/matrix

#### 2.2 Related Work (1 paragrafo)
Citare e differenziarsi da:
- QuantumC originale (Lancellotti et al.) - solo scalare
- QHLS (Lu et al.) - approccio diverso, no MLIR
- Altri lavori su quantum linear algebra (se esistono)

#### 2.3 This Work (1-2 paragrafi)
Descrivere cosa fa la tua estensione:
- Estende QuantumC con supporto vector/matrix
- Pattern recognition per operazioni comuni (vec_add, vec_dot, matmul)
- IR dedicata (VecMat) integrata nella pipeline esistente
- Supporto per programmi ibridi (scalare + vector/matrix)
- Inizializzazione compile-time di array

#### 2.4 Contributions (lista puntata)
Esempio:
- Pattern-based recognition di operazioni vector/matrix in C
- VecMat IR e QAR (Quantum Arithmetic Representation)
- Compilazione ibrida scalare + vector/matrix
- Supporto inizializzazione array compile-time
- Confronto tra backend aritmetici (QFT vs ripple-carry)

#### 2.5 Structure of the paper (1 paragrafo breve)
"Section 2 presents... Section 3 describes... Section 4 reports... Section 5 concludes..."

### Riferimenti utili per Introduction:
- Paper QuantumC per lo stile
- Citazioni su quantum ML (per motivare vector/matrix)

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
