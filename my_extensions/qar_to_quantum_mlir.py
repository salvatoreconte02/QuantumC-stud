# my_extensions/qar_to_quantum_mlir.py

"""
Lowering da QAR (QarModule) al quantum MLIR di QuantumC.

ATTENZIONE: questo pass non "esegue" le operazioni numeriche su dati reali.
Come QuantumC, costruisce SOLO la STRUTTURA del circuito quantistico:

  - quante operazioni servono,
  - come sono composte (pattern di add/mul),
  - come cresce la complessità al variare di length, m, n, k.

I registri inizializzati con QuantumInitOp(0) sono PLACEHOLDER:
non rappresentano il valore 0 del programma C, ma slot quantistici
su cui agiscono le primitive (addi, muli, ecc.).
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region, SSAValue

from my_extensions.qar_ir import (
    QarModule,
    QarFunction,
    QarMapAdd,
    QarDot,
    QarMatMul,
)

from step4_mlir_to_quantum_mlir.quantum_dialect import (
    QuantumInitOp,
    QAddiOp,
    QMuliOp,
)


class QarToQuantumLowering:
    """
    Traduttore schematico da QAR al quantum dialect.

    - Si costruisce un ModuleOp con una func.func per ogni QarFunction.
    - Ogni macro-op QAR viene espansa in una sequenza di operazioni
      quantum.* (init, addi, muli).
    - Il numero di operazioni generate dipende DIRETTAMENTE da:
        * QarMapAdd  -> length
        * QarDot     -> length
        * QarMatMul  -> m, n, k

    Questo rende visibile la complessità del circuito in funzione delle
    dimensioni dei vettori/matrici, che è l’aspetto rilevante per il progetto.
    """

    def __init__(self, qar_module: QarModule, num_bits: int = 16):
        self.qar_module = qar_module
        self.num_bits = num_bits

        self.q_module: ModuleOp = ModuleOp([])
        self.next_reg: int = 0
        self.current_block: Block | None = None

    # --- utilità base -------------------------------------------------

    def _allocate_reg(self) -> int:
        """
        Alloca un identificatore di "registro logico" (solo per name_hint).

        Non esiste un modello dettagliato di layout dei registri: il contatore
        serve solo per generare nomi coerenti (q0_0, q1_0, ...), come fa
        QuantumTranslator.
        """
        r = self.next_reg
        self.next_reg += 1
        return r

    def _emit_init_const(self, value: int) -> SSAValue:
        """
        Crea un nuovo registro quantistico inizializzato con 'value'.

        NOTA IMPORTANTE:
        Nel contesto QAR -> quantum questo valore è un PLACEHOLDER.
        Il circuito rappresenta la struttura dell'operazione, non i dati reali
        del programma C.
        """
        reg = self._allocate_reg()
        op = QuantumInitOp(value)
        self.current_block.add_op(op)
        op.results[0].name_hint = f"q{reg}_0"
        return op.results[0]

    def _emit_add(self, lhs: SSAValue, rhs: SSAValue) -> SSAValue:
        """
        Emette quantum.addi(lhs, rhs) su un nuovo registro.

        Ogni chiamata aggiunge UNA operazione di somma quantistica.
        In questo modo il numero di addizioni è proporzionale a length / m,n,k.
        """
        reg = self._allocate_reg()
        op = QAddiOp(lhs, rhs)
        self.current_block.add_op(op)
        op.results[0].name_hint = f"q{reg}_0"
        return op.results[0]

    def _emit_mul(self, lhs: SSAValue, rhs: SSAValue) -> SSAValue:
        """
        Emette quantum.muli(lhs, rhs) su un nuovo registro.

        Ogni chiamata aggiunge UNA operazione di moltiplicazione quantistica.
        """
        reg = self._allocate_reg()
        op = QMuliOp(lhs, rhs)
        self.current_block.add_op(op)
        op.results[0].name_hint = f"q{reg}_0"
        return op.results[0]

    # --- lowering delle singole macro-op ------------------------------

    def _lower_map_add(self, op: QarMapAdd) -> None:
        """
        Lowering schematico di:

            c = a + b   (vettori di lunghezza op.length)

        Per ogni elemento si simula:

            c[i] = a[i] + b[i]

        SENZA modellare direttamente gli indici o i contenuti:
        si emettono 'length' addizioni quantistiche, quindi il costo
        del circuito cresce O(length).
        """

        for i in range(op.length):
            # Placeholder per "a[i]" e "b[i]": in un vero modello sarebbero
            # registri derivati dallo stato di input; qui servono solo a
            # contare e rappresentare le operazioni.
            qa_i = self._emit_init_const(0)
            qb_i = self._emit_init_const(0)

            # Una addizione per elemento del vettore.
            _ = self._emit_add(qa_i, qb_i)

    def _lower_dot(self, op: QarDot) -> SSAValue:
        """
        Lowering schematico di:

            dest = sum_{i=0}^{length-1} a[i] * b[i]

        Viene costruito un accumulatore quantistico e, per ciascun i,
        si aggiungono UNA moltiplicazione e UNA addizione:

            prod_i = a[i] * b[i]
            acc    = acc + prod_i

        Totale: O(length) moltiplicazioni + O(length) addizioni.
        """
        # Accumulatore iniziale (placeholder).
        acc = self._emit_init_const(0)

        for i in range(op.length):
            qa_i = self._emit_init_const(0)    # placeholder a[i]
            qb_i = self._emit_init_const(0)    # placeholder b[i]
            prod = self._emit_mul(qa_i, qb_i)  # a[i] * b[i]
            acc = self._emit_add(acc, prod)    # acc = acc + prod

        return acc

    def _lower_matmul(self, op: QarMatMul) -> None:
        """
        Lowering schematico di:

            C = A * B
        con dimensioni (m x k) * (k x n) = (m x n).

        Ogni elemento C[i,j] è un prodotto scalare tra:
            - riga i-esima di A,
            - colonna j-esima di B.

        Qui si costruiscono m*n prodotti scalari, ognuno di lunghezza k:

            per ogni (i,j):
                acc = 0
                per kk in [0..k-1]:
                    acc += A[i,kk] * B[kk,j]

        Complessità: O(m * n * k) moltiplicazioni + O(m * n * k) addizioni.
        """
        m, n, k = op.m, op.n, op.k

        for i in range(m):
            for j in range(n):
                # Placeholder per C[i,j] come accumulatore.
                acc = self._emit_init_const(0)

                for kk in range(k):
                    # Placeholder per A[i,kk] e B[kk,j].
                    qa = self._emit_init_const(0)
                    qb = self._emit_init_const(0)

                    prod = self._emit_mul(qa, qb)
                    acc = self._emit_add(acc, prod)

                # 'acc' rappresenta il valore di C[i,j] in forma quantistica.
                # Non viene usato oltre: l'obiettivo è mostrare il pattern
                # e la complessità del circuito, non i dati.

    # --- lowering di una funzione QAR ---------------------------------

    def _lower_function(self, func: QarFunction) -> FuncOp:
        """
        Crea una func.func nel quantum dialect a partire da una QarFunction.

        Scelta semplificata:
        - nessun argomento esplicito,
        - ritorno i32 (come nelle funzioni generate da QuantumTranslator),
        - il valore ritornato, se esiste, è l'ultimo scalare prodotto (es. dot).
        """

        self.current_block = Block()
        self.next_reg = 0  # reset per ogni funzione

        last_scalar: SSAValue | None = None

        for op in func.ops:
            if isinstance(op, QarMapAdd):
                self._lower_map_add(op)
            elif isinstance(op, QarDot):
                last_scalar = self._lower_dot(op)
            elif isinstance(op, QarMatMul):
                self._lower_matmul(op)
            else:
                # Operazione QAR non riconosciuta: ignorata per ora.
                pass

        # Se non è stato prodotto alcuno scalare (es. solo VecAdd/MatMul),
        # si ritorna un placeholder costante.
        if last_scalar is None:
            last_scalar = self._emit_init_const(0)

        ret = ReturnOp(last_scalar)
        self.current_block.add_op(ret)

        func_type = ([], [i32])
        return FuncOp(func.name, func_type, Region([self.current_block]))

    # --- entry point --------------------------------------------------

    def lower(self) -> ModuleOp:
        """
        Traduzione completa del QarModule in un ModuleOp quantum.

        Il modulo risultante è compatibile con il backend di QuantumC
        (generate_circuit + qasm_generator).
        """
        top_block = self.q_module.body.blocks[0]

        for func in self.qar_module.functions:
            q_func = self._lower_function(func)
            top_block.add_op(q_func)

        return self.q_module


def from_qar_to_quantum_mlir(qar_module: QarModule, num_bits: int = 16) -> ModuleOp:
    """
    Entry point usato dalla pipeline:

        QarModule  ->  ModuleOp (quantum dialect)
    """
    lowering = QarToQuantumLowering(qar_module, num_bits=num_bits)
    return lowering.lower()