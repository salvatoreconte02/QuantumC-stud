from dataclasses import dataclass, field
from typing import List

@dataclass
class QarOp:
    """Classe base astratta per operazioni QAR."""
    pass


@dataclass
class QarMapAdd(QarOp):
    """Addizione vettoriale c = a + b eseguita in dominio QFT."""
    dest: str
    lhs: str
    rhs: str
    length: int
    elem_bits: int


@dataclass
class QarDot(QarOp):
    """Prodotto scalare s = <a, b> implementato come MAC quantistico QFT-based."""
    dest: str
    lhs: str
    rhs: str
    length: int
    elem_bits: int


@dataclass
class QarMatMul(QarOp):
    """Moltiplicazione matriciale C = A * B in dominio QFT."""
    dest: str
    lhs: str
    rhs: str
    m: int
    n: int
    k: int
    elem_bits: int


@dataclass
class QarFunction:
    name: str
    ops: List[QarOp] = field(default_factory=list)


@dataclass
class QarModule:
    functions: list[QarFunction] = field(default_factory=list)
    const_arrays: dict[str, list[int]] = field(default_factory=dict)      # NUOVO
    const_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)   # NUOVO
