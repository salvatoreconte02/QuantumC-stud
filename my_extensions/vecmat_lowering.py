# my_extensions/vecmat_lowering.py

from typing import Optional

from step2_ast_to_dataclasses.c_ast import (
    TranslationUnit,
    FunctionDecl,
    CompoundStmt,
    VarDecl,
    ForStmt,
    AssignStmt,
    IntegerLiteral,
    DeclRef,
    BinaryOperator,
    BinaryOperatorWithImmediate,
    ArrayAccess,
)
from my_extensions.vecmat_ir import VecMatModule, VecMatFunction, VecAddOp, VecDotOp


def _match_vec_add_for(for_stmt: ForStmt, elem_bits: int) -> Optional[VecAddOp]:
    """
    Riconosce:

        for (int i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
    """

    # 1) init: VarDecl(name=idx_name, init=IntegerLiteral(0))
    init = for_stmt.init
    if not isinstance(init, VarDecl):
        return None
    if not isinstance(init.init, IntegerLiteral):
        return None
    if init.init.value != 0:
        return None
    idx_name = init.name

    # 2) condition: BinaryOperatorWithImmediate("<" o "<=", DeclRef(idx_name), IntegerLiteral(N))
    cond = for_stmt.condition
    if not isinstance(cond, BinaryOperatorWithImmediate):
        return None
    if cond.opcode not in ("<", "<="):
        return None
    if not isinstance(cond.lhs, DeclRef) or cond.lhs.name != idx_name:
        return None
    if not isinstance(cond.rhs, IntegerLiteral):
        return None
    length = cond.rhs.value

    # 3) increment: i = i + 1
    incr = for_stmt.increment
    if not isinstance(incr, AssignStmt):
        return None
    if not isinstance(incr.name, str) or incr.name != idx_name:
        return None
    incr_expr = incr.value
    if not isinstance(incr_expr, BinaryOperatorWithImmediate):
        return None
    if incr_expr.opcode != "+":
        return None
    if not isinstance(incr_expr.lhs, DeclRef) or incr_expr.lhs.name != idx_name:
        return None
    if not isinstance(incr_expr.rhs, IntegerLiteral) or incr_expr.rhs.value != 1:
        return None

    # 4) body: un solo AssignStmt: c[i] = a[i] + b[i];
    body = for_stmt.body
    if not isinstance(body, CompoundStmt):
        return None
    if len(body.stmts) != 1:
        return None
    stmt = body.stmts[0]
    if not isinstance(stmt, AssignStmt):
        return None

    lhs = stmt.name
    if not isinstance(lhs, ArrayAccess):
        return None
    if not isinstance(lhs.array, DeclRef):
        return None
    dest_name = lhs.array.name
    if not isinstance(lhs.index, DeclRef) or lhs.index.name != idx_name:
        return None

    rhs = stmt.value
    if not isinstance(rhs, BinaryOperator) or rhs.opcode != "+":
        return None

    lhs_term = rhs.lhs
    if not isinstance(lhs_term, ArrayAccess):
        return None
    if not isinstance(lhs_term.array, DeclRef):
        return None
    vec_a = lhs_term.array.name
    if not isinstance(lhs_term.index, DeclRef) or lhs_term.index.name != idx_name:
        return None

    rhs_term = rhs.rhs
    if not isinstance(rhs_term, ArrayAccess):
        return None
    if not isinstance(rhs_term.array, DeclRef):
        return None
    vec_b = rhs_term.array.name
    if not isinstance(rhs_term.index, DeclRef) or rhs_term.index.name != idx_name:
        return None

    return VecAddOp(
        dest=dest_name,
        lhs=vec_a,
        rhs=vec_b,
        length=length,
        elem_bits=elem_bits,
    )


def _match_vec_dot_for(for_stmt: ForStmt, elem_bits: int) -> Optional[VecDotOp]:
    """
    Riconosce:

        for (int i = 0; i < N; i++) {
            s = s + a[i] * b[i];
        }
    """

    # 1) init: int idx = 0;
    init = for_stmt.init
    if not isinstance(init, VarDecl):
        return None
    if not isinstance(init.init, IntegerLiteral):
        return None
    if init.init.value != 0:
        return None
    idx_name = init.name

    # 2) condition: idx < N
    cond = for_stmt.condition
    if not isinstance(cond, BinaryOperatorWithImmediate):
        return None
    if cond.opcode not in ("<", "<="):
        return None
    if not isinstance(cond.lhs, DeclRef) or cond.lhs.name != idx_name:
        return None
    if not isinstance(cond.rhs, IntegerLiteral):
        return None
    length = cond.rhs.value

    # 3) increment: idx = idx + 1
    incr = for_stmt.increment
    if not isinstance(incr, AssignStmt):
        return None
    if not isinstance(incr.name, str) or incr.name != idx_name:
        return None
    incr_expr = incr.value
    if not isinstance(incr_expr, BinaryOperatorWithImmediate):
        return None
    if incr_expr.opcode != "+":
        return None
    if not isinstance(incr_expr.lhs, DeclRef) or incr_expr.lhs.name != idx_name:
        return None
    if not isinstance(incr_expr.rhs, IntegerLiteral) or incr_expr.rhs.value != 1:
        return None

    # 4) body: un solo AssignStmt: s = s + a[i] * b[i];
    body = for_stmt.body
    if not isinstance(body, CompoundStmt):
        return None
    if len(body.stmts) != 1:
        return None
    stmt = body.stmts[0]
    if not isinstance(stmt, AssignStmt):
        return None

    # LHS: variabile scalare s
    if not isinstance(stmt.name, str):
        return None
    dest_name = stmt.name

    rhs = stmt.value
    if not isinstance(rhs, BinaryOperator) or rhs.opcode != "+":
        return None

    # lhs del '+' deve essere s
    if not isinstance(rhs.lhs, DeclRef) or rhs.lhs.name != dest_name:
        return None

    # rhs del '+' deve essere un '*'
    mul = rhs.rhs
    if not isinstance(mul, BinaryOperator) or mul.opcode != "*":
        return None

    # primo fattore: a[i]
    lhs_term = mul.lhs
    if not isinstance(lhs_term, ArrayAccess):
        return None
    if not isinstance(lhs_term.array, DeclRef):
        return None
    vec_a = lhs_term.array.name
    if not isinstance(lhs_term.index, DeclRef) or lhs_term.index.name != idx_name:
        return None

    # secondo fattore: b[i]
    rhs_term = mul.rhs
    if not isinstance(rhs_term, ArrayAccess):
        return None
    if not isinstance(rhs_term.array, DeclRef):
        return None
    vec_b = rhs_term.array.name
    if not isinstance(rhs_term.index, DeclRef) or rhs_term.index.name != idx_name:
        return None

    return VecDotOp(
        dest=dest_name,
        lhs=vec_a,
        rhs=vec_b,
        length=length,
        elem_bits=elem_bits,
    )


def from_c_ast_to_vecmat(tu: TranslationUnit, elem_bits: int) -> VecMatModule:
    """
    Converte un TranslationUnit (dataclass C) in un VecMatModule
    del dialetto vettoriale/matriciale, riconoscendo solo i pattern
    base vec_add e vec_dot.
    """

    module = VecMatModule()

    for func in tu.decls:
        if not isinstance(func, FunctionDecl):
            continue

        vec_func = VecMatFunction(name=func.name, params=list(func.params))

        for stmt in func.body.stmts:
            if isinstance(stmt, ForStmt):
                # Prova prima vec_add
                vec_add_op = _match_vec_add_for(stmt, elem_bits)
                if vec_add_op is not None:
                    vec_func.ops.append(vec_add_op)
                    continue

                # Poi prova vec_dot
                vec_dot_op = _match_vec_dot_for(stmt, elem_bits)
                if vec_dot_op is not None:
                    vec_func.ops.append(vec_dot_op)
                    continue

                # Altri pattern di for: non supportati per ora

        module.functions.append(vec_func)

    return module