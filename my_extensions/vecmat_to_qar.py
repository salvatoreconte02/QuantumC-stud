# my_extensions/vecmat_to_qar.py

from my_extensions.vecmat_ir import VecMatModule, VecAddOp, VecDotOp, MatMulOp
from my_extensions.qar_ir import QarModule, QarFunction, QarMapAdd, QarDot, QarMatMul


def from_vecmat_to_qar(vm_module: VecMatModule) -> QarModule:
    """
    Converte un VecMatModule nel dialetto QAR, mappando
    1:1 le macro-operazioni vettoriali/matriciali in
    macro-operazioni di aritmetica quantistica.

    Inoltre PROPAGA:
      - const_arrays (valori costanti estratti da InitList)
      - const_shapes (shape: (L,) o (rows, cols))
    """
    qar_mod = QarModule()

    # Copia inizializzazioni costanti (difensivo)
    vm_const_arrays = getattr(vm_module, "const_arrays", {}) or {}
    vm_const_shapes = getattr(vm_module, "const_shapes", {}) or {}
    qar_mod.const_arrays = dict(vm_const_arrays)
    qar_mod.const_shapes = dict(vm_const_shapes)

    for func in vm_module.functions:
        qar_func = QarFunction(name=func.name)

        for op in func.ops:
            if isinstance(op, VecAddOp):
                qar_func.ops.append(
                    QarMapAdd(
                        dest=op.dest,
                        lhs=op.lhs,
                        rhs=op.rhs,
                        length=op.length,
                        elem_bits=op.elem_bits,
                    )
                )
            elif isinstance(op, VecDotOp):
                qar_func.ops.append(
                    QarDot(
                        dest=op.dest,
                        lhs=op.lhs,
                        rhs=op.rhs,
                        length=op.length,
                        elem_bits=op.elem_bits,
                    )
                )
            elif isinstance(op, MatMulOp):
                qar_func.ops.append(
                    QarMatMul(
                        dest=op.dest,
                        lhs=op.lhs,
                        rhs=op.rhs,
                        m=op.m,
                        n=op.n,
                        k=op.k,
                        elem_bits=op.elem_bits,
                    )
                )

        qar_mod.functions.append(qar_func)

    return qar_mod