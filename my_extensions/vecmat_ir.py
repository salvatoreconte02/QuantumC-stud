# my_extensions/vecmat_ir.py

from dataclasses import dataclass, field
from typing import List, Optional, Union


# =========================
# Tipi del dialetto
# =========================

@dataclass
class ScalarType:
    """Tipo scalare classico (es. int a 8 o 16 bit)."""
    bits: int = 16


@dataclass
class VectorType:
    """Tipo vettore: N elementi scalari di bitwidth fissato."""
    elem_bits: int
    length: int


@dataclass
class MatrixType:
    """Tipo matrice: rows x cols elementi scalari."""
    elem_bits: int
    rows: int
    cols: int


VecMatType = Union[ScalarType, VectorType, MatrixType]


# =========================
# Base per le operazioni
# =========================

@dataclass
class VecMatOp:
    """Classe base astratta per tutte le operazioni del dialetto."""
    pass


# =========================
# Operazioni scalari di supporto
# =========================

@dataclass
class ScalarAddOp(VecMatOp):
    """Operazione di somma scalare: dest = lhs + rhs."""
    dest: str
    lhs: str
    rhs: str
    ty: ScalarType


@dataclass
class ScalarMulOp(VecMatOp):
    """Operazione di prodotto scalare: dest = lhs * rhs."""
    dest: str
    lhs: str
    rhs: str
    ty: ScalarType


# =========================
# Macro-operazioni vettoriali/matriciali
# =========================

@dataclass
class VecAddOp(VecMatOp):
    """
    Somma di vettori: c[i] = a[i] + b[i] per i = 0..length-1.

    a, b, c sono nomi di variabili/buffer logici.
    """
    dest: str         # nome vettore risultato (es. "c")
    lhs: str          # nome vettore a
    rhs: str          # nome vettore b
    length: int       # dimensione del vettore
    elem_bits: int    # bitwidth dell'elemento


@dataclass
class VecDotOp(VecMatOp):
    """
    Prodotto scalare: acc = Σ_{i=0..length-1} a[i] * b[i].

    a, b sono vettori; acc è la variabile scalare risultato.
    """
    dest: str         # nome scalare accumulatore (es. "s")
    lhs: str          # nome vettore a
    rhs: str          # nome vettore b
    length: int       # dimensione del vettore
    elem_bits: int    # bitwidth dell'elemento


@dataclass
class MatMulOp(VecMatOp):
    """
    Moltiplicazione di matrici: C = A * B.

    A: M x K, B: K x N, C: M x N.
    Le dimensioni sono esplicite per facilitare il mapping successivo.
    """
    dest: str         # nome matrice risultato (es. "C")
    lhs: str          # nome matrice A
    rhs: str          # nome matrice B
    m: int            # numero di righe di A e C
    n: int            # numero di colonne di B e C
    k: int            # dimensione interna (colonne A, righe B)
    elem_bits: int    # bitwidth dell'elemento


# =========================
# Funzioni e modulo
# =========================

@dataclass
class VecMatFunction:
    """
    Funzione nel dialetto vettoriale/matriciale.

    Contiene il nome, la firma (parametri) e la lista di operazioni.
    """
    name: str
    params: List[str] = field(default_factory=list)
    ops: List[VecMatOp] = field(default_factory=list)


@dataclass
class VecMatModule:
    """
    Modulo di alto livello per il dialetto intermedio.

    Contiene una lista di funzioni.
    """
    functions: List[VecMatFunction] = field(default_factory=list)