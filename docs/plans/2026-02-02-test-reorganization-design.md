# Design Document: Riorganizzazione Test Suite QuantumC

**Data:** 2026-02-02
**Obiettivo:** Riorganizzare i test per la presentazione ufficiale e pubblicazione

---

## 1. Panoramica

### Situazione Attuale
- 36 file C di test sparsi in varie cartelle
- 1 script Python (`test_c_file.py`) per testing manuale
- ~8.5MB di output accumulati (QASM, JSON, MLIR)
- Nessuna organizzazione sistematica per categorie

### Obiettivo
- **18 test** ben organizzati in 5 categorie
- Script di test migliorato con supporto batch e report
- Pulizia completa degli output
- README aggiornato per pubblicazione

---

## 2. Nuova Struttura Directory

```
tests/
├── scalar/           # 3 test - operazioni base su interi
│   ├── add.c
│   ├── unary.c
│   └── arithmetic.c
├── vector/           # 3 test - vec_add e dot product
│   ├── add.c
│   ├── dot.c
│   └── add_4elem.c
├── matrix/           # 3 test - moltiplicazione matriciale
│   ├── mul_2x2.c
│   ├── mul_2x2_alt.c
│   └── mul_observable.c
├── hybrid/           # 3 test - mix scalare + vettoriale
│   ├── scalar_vecadd_dot.c
│   ├── scalar_vecadd.c
│   └── scalar_matmul.c
└── comparison/       # 6 test - confronto QFT vs Ripple
    ├── scalar_add_chain.c
    ├── scalar_sub_neg.c
    ├── scalar_mul.c
    ├── vector_add.c
    ├── vector_dot.c
    └── matrix_mul.c
```

---

## 3. Mapping File: Origine → Destinazione

### 3.1 Categoria `scalar/`

| Origine | Destinazione | Risultato Atteso |
|---------|--------------|------------------|
| `c_code/try.c` | `tests/scalar/add.c` | 7 |
| `c_code/unary_ops.c` | `tests/scalar/unary.c` | (calcolo complesso) |
| *nuovo* | `tests/scalar/arithmetic.c` | 53 |

### 3.2 Categoria `vector/`

| Origine | Destinazione | Risultato Atteso |
|---------|--------------|------------------|
| `c_code/vect_and_matrix/vec_add_init.c` | `tests/vector/add.c` | c = {4, 6} |
| `c_code/vect_and_matrix/vec_dot_observable.c` | `tests/vector/dot.c` | 11 |
| `c_code/test_confronto/t_vec_add.c` | `tests/vector/add_4elem.c` | 5 |

### 3.3 Categoria `matrix/`

| Origine | Destinazione | Risultato Atteso |
|---------|--------------|------------------|
| `c_code/vect_and_matrix/matmul_2x2.c` | `tests/matrix/mul_2x2.c` | C[0][0] = 19 |
| `c_code/test_confronto/t_matmul_2x2.c` | `tests/matrix/mul_2x2_alt.c` | C[1][0] = 10 |
| `c_code/vect_and_matrix/matmul_observable.c` | `tests/matrix/mul_observable.c` | 7 |

### 3.4 Categoria `hybrid/`

| Origine | Destinazione | Risultato Atteso |
|---------|--------------|------------------|
| `c_code/vect_and_matrix/hybrid_small.c` | `tests/hybrid/scalar_vecadd_dot.c` | 23 |
| `c_code/vect_and_matrix/hybrid_vecadd_observable.c` | `tests/hybrid/scalar_vecadd.c` | 16 |
| `c_code/vect_and_matrix/hybrid_matmul_return_z.c` | `tests/hybrid/scalar_matmul.c` | (z + C[1][1]) |

### 3.5 Categoria `comparison/`

| Origine | Destinazione | Risultato Atteso |
|---------|--------------|------------------|
| `c_code/test_confronto/t_add_chain.c` | `tests/comparison/scalar_add_chain.c` | 6 |
| `c_code/test_confronto/t_sub_mix.c` | `tests/comparison/scalar_sub_neg.c` | -2 |
| `c_code/test_confronto/t_mul_small.c` | `tests/comparison/scalar_mul.c` | 6 |
| `c_code/test_confronto/t_vec_add.c` | `tests/comparison/vector_add.c` | 5 |
| `c_code/test_confronto/t_vec_dot.c` | `tests/comparison/vector_dot.c` | 20 |
| `c_code/test_confronto/t_matmul_2x2.c` | `tests/comparison/matrix_mul.c` | 10 |

---

## 4. File da Eliminare

### 4.1 File C Ridondanti o Problematici

| File | Motivo |
|------|--------|
| `c_code/smoke_add.c` | Ridondante con try.c |
| `c_code/vect_and_matrix/vec_add.c` | Non inizializzato |
| `c_code/vect_and_matrix/vec_add2.c` | Bug buffer overflow |
| `c_code/vect_and_matrix/vec_add_small.c` | Non inizializzato |
| `c_code/vect_and_matrix/vec_add_pure.c` | Ridondante |
| `c_code/vect_and_matrix/vec_dot.c` | Non inizializzato |
| `c_code/vect_and_matrix/matmul.c` | 4x4 non inizializzato |
| `c_code/vect_and_matrix/mat_init_only.c` | Nessuna computazione |
| `c_code/vect_and_matrix/scalar_add.c` | Ridondante |
| `c_code/vect_and_matrix/vec_add_dot.c` | Non inizializzato |
| `c_code/vect_and_matrix/hybrid_vecadd_return_z.c` | Meno utile |
| `c_code/vect_and_matrix/matmul_2x2_hybrid.c` | Ridondante |

### 4.2 Directory Output (pulizia completa)

```
output/*.qasm
json_out/*.json
mlir_out/*.mlir
quantum_mlir_out/*.mlir
```

### 4.3 Vecchie Directory (dopo migrazione)

```
c_code/vect_and_matrix/  → eliminare dopo migrazione
c_code/test_confronto/   → eliminare dopo migrazione
c_code/*.c (tranne quelli migrati) → eliminare
```

---

## 5. File da Creare

### 5.1 `tests/scalar/arithmetic.c`

```c
// Test completo operazioni aritmetiche: +, -, *, /
int main() {
    int a = 10;
    int b = 3;
    int add = a + b;    // 13
    int sub = a - b;    // 7
    int mul = a * b;    // 30
    int div = a / b;    // 3
    int result = add + sub + mul + div;  // 53
    return result;
}
```

### 5.2 `c_code/others/README.md`

```markdown
# Legacy Tests (Upstream QuantumC)

Questa cartella contiene i test originali del progetto upstream
[QuantumC](https://github.com/leonardopagliochini/QuantumC).

Questi file sono mantenuti per riferimento storico ma **non fanno parte
della test suite ufficiale** di questa estensione.

Per i test ufficiali, vedere la cartella `tests/`.
```

---

## 6. Miglioramenti `test_c_file.py`

### 6.1 Nuove Funzionalità

| Feature | Comando | Descrizione |
|---------|---------|-------------|
| Singolo file | `python test_c_file.py tests/scalar/add.c` | Comportamento attuale |
| Cartella | `python test_c_file.py tests/scalar/` | Esegue tutti i .c |
| Categoria | `python test_c_file.py --category scalar` | Esegue categoria |
| Tutti | `python test_c_file.py --all` | Esegue tutte le categorie |
| Backend | `python test_c_file.py --adder qft` | Seleziona backend |
| Verbose | `python test_c_file.py --verbose` | Output dettagliato |

### 6.2 Output Esempio

```
=== QuantumC Test Suite ===
Backend: qft | Bits: 16

[scalar] 3/3 passed
  ✓ add.c              expected: 7, got: 7
  ✓ unary.c            expected: 42, got: 42
  ✓ arithmetic.c       expected: 53, got: 53

[vector] 3/3 passed
  ✓ add.c              expected: 6, got: 6
  ✓ dot.c              expected: 11, got: 11
  ✓ add_4elem.c        expected: 5, got: 5

[matrix] 3/3 passed
  ✓ mul_2x2.c          expected: 19, got: 19
  ✓ mul_2x2_alt.c      expected: 10, got: 10
  ✓ mul_observable.c   expected: 7, got: 7

[hybrid] 3/3 passed
  ✓ scalar_vecadd_dot.c    expected: 23, got: 23
  ✓ scalar_vecadd.c        expected: 16, got: 16
  ✓ scalar_matmul.c        expected: 34, got: 34

[comparison] 6/6 passed
  ✓ scalar_add_chain.c     expected: 6, got: 6
  ✓ scalar_sub_neg.c       expected: -2, got: -2
  ✓ scalar_mul.c           expected: 6, got: 6
  ✓ vector_add.c           expected: 5, got: 5
  ✓ vector_dot.c           expected: 20, got: 20
  ✓ matrix_mul.c           expected: 10, got: 10

========================================
Summary: 18/18 passed (100%)
Backend: qft | Time: 45.2s
```

---

## 7. Aggiornamento README.md Principale

### 7.1 Sezioni da Aggiungere/Modificare

Il README attuale è dell'upstream. Aggiornare con:

1. **Titolo e descrizione** - Chiarire che è un'estensione con supporto vector/matrix
2. **Features aggiunte** - Lista delle estensioni:
   - Vector/Matrix operations (vec_add, vec_dot, matmul)
   - Ripple-carry backend alternativo
   - Hybrid compilation (scalar + vector/matrix)
   - Compile-time array initialization
   - Backend comparison mode (`--adder both`)
3. **Quick start** - Esempi di compilazione
4. **Test suite** - Come eseguire i test
5. **Architettura** - Diagramma pipeline con path ibrido
6. **Credits** - Riferimento all'upstream originale

### 7.2 Struttura Proposta

```markdown
# QuantumC Extended

MLIR-based compiler that translates C programs into quantum circuits,
extended with vector/matrix operations support.

## Features

**Original QuantumC:**
- Scalar integer arithmetic → QFT-based quantum circuits
- 5-stage pipeline: C → AST → MLIR → Quantum MLIR → QASM

**Extensions (this fork):**
- Vector operations: `vec_add`, `vec_dot`
- Matrix multiplication: `matmul` (2x2, 4x4)
- Hybrid compilation: mixed scalar + vector/matrix
- Ripple-carry adder backend (`--adder ripple`)
- Backend comparison mode (`--adder both`)
- Compile-time array initialization

## Quick Start

[... esempi ...]

## Running Tests

python test_c_file.py --all              # Run all tests
python test_c_file.py --category scalar  # Run scalar tests
python test_c_file.py --adder both       # Compare backends

## Credits

Based on [QuantumC](https://github.com/leonardopagliochini/QuantumC)
by Leonardo Pagliochini.
```

---

## 8. Piano di Implementazione

### Fase 1: Setup
1. Creare struttura directory `tests/`
2. Creare `tests/scalar/arithmetic.c`

### Fase 2: Migrazione File
3. Copiare e rinominare file secondo mapping (Sezione 3)
4. Creare `c_code/others/README.md`

### Fase 3: Pulizia
5. Eliminare file ridondanti (Sezione 4.1)
6. Pulire directory output (Sezione 4.2)
7. Eliminare vecchie directory c_code/* (Sezione 4.3)

### Fase 4: Test Script
8. Modificare `test_c_file.py` con nuove funzionalità

### Fase 5: Documentazione
9. Aggiornare README.md principale

### Fase 6: Verifica
10. Eseguire `python test_c_file.py --all` e verificare 18/18 passed

---

## 9. Rischi e Mitigazioni

| Rischio | Mitigazione |
|---------|-------------|
| Test falliscono dopo migrazione | Verificare ogni test prima di eliminare originale |
| Path hardcoded in pipeline.py | Verificare che pipeline.py accetti path arbitrari |
| Output comparison cambia | Mantenere backup prima di pulizia |

---

## 10. Checklist Finale

- [ ] Struttura `tests/` creata
- [ ] 18 file C migrati e rinominati
- [ ] `arithmetic.c` creato e funzionante
- [ ] File ridondanti eliminati
- [ ] Output directories pulite
- [ ] `c_code/others/README.md` creato
- [ ] `test_c_file.py` migliorato
- [ ] README.md principale aggiornato
- [ ] `python test_c_file.py --all` → 18/18 passed
- [ ] Commit finale
