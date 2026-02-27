# my_extensions/vecmat_ir.py

from dataclasses import dataclass, field
from typing import List, Union, Tuple, Dict

# Tipi del dialetto

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


# Base per le operazioni

@dataclass
class VecMatOp:
    """Classe base astratta per tutte le operazioni del dialetto."""
    pass


# Operazioni scalari di supporto

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


# Macro-operazioni vettoriali/matriciali

@dataclass
class VecAddOp(VecMatOp):
    """
    Somma di vettori: c[i] = a[i] + b[i] per i = 0..length-1.
    """
    dest: str
    lhs: str
    rhs: str
    length: int
    elem_bits: int


@dataclass
class VecDotOp(VecMatOp):
    """
    Prodotto scalare: acc = Σ_{i=0..length-1} a[i] * b[i].
    """
    dest: str
    lhs: str
    rhs: str
    length: int
    elem_bits: int


@dataclass
class MatMulOp(VecMatOp):
    """
    Moltiplicazione di matrici: C = A * B.

    A: M x K, B: K x N, C: M x N.
    """
    dest: str
    lhs: str
    rhs: str
    m: int
    n: int
    k: int
    elem_bits: int


# Funzioni e modulo

@dataclass
class VecMatFunction:
    """
    Funzione nel dialetto vettoriale/matriciale.
    """
    name: str
    params: List[str] = field(default_factory=list)
    ops: List[VecMatOp] = field(default_factory=list)


@dataclass
class VecMatModule:
    """
    Modulo di alto livello VecMat.

    Oltre alle funzioni, mantiene anche eventuali inizializzazioni costanti
    lette dal C tramite InitList (vettori/matrici).

    Convenzioni:
      - const_shapes[name] == (L,)         -> vettore di lunghezza L
      - const_shapes[name] == (R, C)       -> matrice RxC (row-major)
      - const_arrays[name] è sempre "flat":
            * vettore: L elementi
            * matrice: R*C elementi in ordine riga (row-major)

    Nota: per ora si assume supporto fino a 2D (vettori e matrici).
    """
    functions: List[VecMatFunction] = field(default_factory=list)

    const_arrays: Dict[str, List[int]] = field(default_factory=dict)
    const_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)