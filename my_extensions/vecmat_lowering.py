# my_extensions/vecmat_lowering.py

from typing import Optional, Dict, List, Tuple

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
    InitList,          # NUOVO
)

from my_extensions.vecmat_ir import (
    VecMatModule,
    VecMatFunction,
    VecAddOp,
    VecDotOp,
    MatMulOp,
)


# =========================
# vec_add
# =========================

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


# =========================
# vec_dot
# =========================

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


# =========================
# matmul (triplo loop)
# =========================

def _match_matmul_for(for_i: ForStmt, elem_bits: int) -> Optional[MatMulOp]:
    """
    Riconosce il pattern:

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                c[i][j] = 0;
                for (int k = 0; k < K; k++) {
                    c[i][j] = c[i][j] + a[i][k] * b[k][j];
                }
            }
        }
    """

    # ----- for i -----
    init_i = for_i.init
    if not isinstance(init_i, VarDecl):
        return None
    if not isinstance(init_i.init, IntegerLiteral) or init_i.init.value != 0:
        return None
    i_name = init_i.name

    cond_i = for_i.condition
    if not isinstance(cond_i, BinaryOperatorWithImmediate):
        return None
    if cond_i.opcode not in ("<", "<="):
        return None
    if not isinstance(cond_i.lhs, DeclRef) or cond_i.lhs.name != i_name:
        return None
    if not isinstance(cond_i.rhs, IntegerLiteral):
        return None
    m = cond_i.rhs.value  # numero di righe

    incr_i = for_i.increment
    if not isinstance(incr_i, AssignStmt):
        return None
    if incr_i.name != i_name:
        return None
    incr_i_expr = incr_i.value
    if not isinstance(incr_i_expr, BinaryOperatorWithImmediate):
        return None
    if incr_i_expr.opcode != "+":
        return None
    if not isinstance(incr_i_expr.lhs, DeclRef) or incr_i_expr.lhs.name != i_name:
        return None
    if not isinstance(incr_i_expr.rhs, IntegerLiteral) or incr_i_expr.rhs.value != 1:
        return None

    # body i: ci aspettiamo un solo ForStmt (for j)
    body_i = for_i.body
    if not isinstance(body_i, CompoundStmt):
        return None
    if len(body_i.stmts) != 1:
        return None
    for_j = body_i.stmts[0]
    if not isinstance(for_j, ForStmt):
        return None

    # ----- for j -----
    init_j = for_j.init
    if not isinstance(init_j, VarDecl):
        return None
    if not isinstance(init_j.init, IntegerLiteral) or init_j.init.value != 0:
        return None
    j_name = init_j.name

    cond_j = for_j.condition
    if not isinstance(cond_j, BinaryOperatorWithImmediate):
        return None
    if cond_j.opcode not in ("<", "<="):
        return None
    if not isinstance(cond_j.lhs, DeclRef) or cond_j.lhs.name != j_name:
        return None
    if not isinstance(cond_j.rhs, IntegerLiteral):
        return None
    n = cond_j.rhs.value  # numero di colonne

    incr_j = for_j.increment
    if not isinstance(incr_j, AssignStmt):
        return None
    if incr_j.name != j_name:
        return None
    incr_j_expr = incr_j.value
    if not isinstance(incr_j_expr, BinaryOperatorWithImmediate):
        return None
    if incr_j_expr.opcode != "+":
        return None
    if not isinstance(incr_j_expr.lhs, DeclRef) or incr_j_expr.lhs.name != j_name:
        return None
    if not isinstance(incr_j_expr.rhs, IntegerLiteral) or incr_j_expr.rhs.value != 1:
        return None

    # body j: ci aspettiamo [ AssignStmt(c[i][j] = 0), ForStmt(k) ]
    body_j = for_j.body
    if not isinstance(body_j, CompoundStmt):
        return None
    if len(body_j.stmts) != 2:
        return None

    zero_stmt = body_j.stmts[0]
    for_k = body_j.stmts[1]
    if not isinstance(zero_stmt, AssignStmt):
        return None
    if not isinstance(for_k, ForStmt):
        return None

    # c[i][j] = 0;
    lhs_zero = zero_stmt.name
    if not isinstance(lhs_zero, ArrayAccess):
        return None
    if not isinstance(lhs_zero.array, ArrayAccess):
        return None
    c_outer = lhs_zero.array
    if not isinstance(c_outer.array, DeclRef):
        return None
    dest_name = c_outer.array.name  # "c"
    if not isinstance(c_outer.index, DeclRef) or c_outer.index.name != i_name:
        return None
    if not isinstance(lhs_zero.index, DeclRef) or lhs_zero.index.name != j_name:
        return None
    if not isinstance(zero_stmt.value, IntegerLiteral) or zero_stmt.value.value != 0:
        return None

    # ----- for k -----
    init_k = for_k.init
    if not isinstance(init_k, VarDecl):
        return None
    if not isinstance(init_k.init, IntegerLiteral) or init_k.init.value != 0:
        return None
    k_name = init_k.name

    cond_k = for_k.condition
    if not isinstance(cond_k, BinaryOperatorWithImmediate):
        return None
    if cond_k.opcode not in ("<", "<="):
        return None
    if not isinstance(cond_k.lhs, DeclRef) or cond_k.lhs.name != k_name:
        return None
    if not isinstance(cond_k.rhs, IntegerLiteral):
        return None
    k_dim = cond_k.rhs.value  # dimensione interna

    incr_k = for_k.increment
    if not isinstance(incr_k, AssignStmt):
        return None
    if incr_k.name != k_name:
        return None
    incr_k_expr = incr_k.value
    if not isinstance(incr_k_expr, BinaryOperatorWithImmediate):
        return None
    if incr_k_expr.opcode != "+":
        return None
    if not isinstance(incr_k_expr.lhs, DeclRef) or incr_k_expr.lhs.name != k_name:
        return None
    if not isinstance(incr_k_expr.rhs, IntegerLiteral) or incr_k_expr.rhs.value != 1:
        return None

    # body k: un solo AssignStmt con accumulo:
    # c[i][j] = c[i][j] + a[i][k] * b[k][j];
    body_k = for_k.body
    if not isinstance(body_k, CompoundStmt):
        return None
    if len(body_k.stmts) != 1:
        return None
    update_stmt = body_k.stmts[0]
    if not isinstance(update_stmt, AssignStmt):
        return None

    lhs_update = update_stmt.name
    if not isinstance(lhs_update, ArrayAccess):
        return None
    if not isinstance(lhs_update.array, ArrayAccess):
        return None
    c_outer2 = lhs_update.array
    if not isinstance(c_outer2.array, DeclRef) or c_outer2.array.name != dest_name:
        return None
    if not isinstance(c_outer2.index, DeclRef) or c_outer2.index.name != i_name:
        return None
    if not isinstance(lhs_update.index, DeclRef) or lhs_update.index.name != j_name:
        return None

    rhs_update = update_stmt.value
    if not isinstance(rhs_update, BinaryOperator) or rhs_update.opcode != "+":
        return None

    # lhs del '+' deve essere ancora c[i][j]
    lhs_sum = rhs_update.lhs
    if not isinstance(lhs_sum, ArrayAccess):
        return None
    if not isinstance(lhs_sum.array, ArrayAccess):
        return None
    c_outer3 = lhs_sum.array
    if not isinstance(c_outer3.array, DeclRef) or c_outer3.array.name != dest_name:
        return None
    if not isinstance(c_outer3.index, DeclRef) or c_outer3.index.name != i_name:
        return None
    if not isinstance(lhs_sum.index, DeclRef) or lhs_sum.index.name != j_name:
        return None

    # rhs del '+' deve essere un prodotto: a[i][k] * b[k][j]
    mul = rhs_update.rhs
    if not isinstance(mul, BinaryOperator) or mul.opcode != "*":
        return None

    # a[i][k]
    a_term = mul.lhs
    if not isinstance(a_term, ArrayAccess):
        return None
    if not isinstance(a_term.array, ArrayAccess):
        return None
    a_outer = a_term.array
    if not isinstance(a_outer.array, DeclRef):
        return None
    lhs_name = a_outer.array.name  # "a"
    if not isinstance(a_outer.index, DeclRef) or a_outer.index.name != i_name:
        return None
    if not isinstance(a_term.index, DeclRef) or a_term.index.name != k_name:
        return None

    # b[k][j]
    b_term = mul.rhs
    if not isinstance(b_term, ArrayAccess):
        return None
    if not isinstance(b_term.array, ArrayAccess):
        return None
    b_outer = b_term.array
    if not isinstance(b_outer.array, DeclRef):
        return None
    rhs_name = b_outer.array.name  # "b"
    if not isinstance(b_outer.index, DeclRef) or b_outer.index.name != k_name:
        return None
    if not isinstance(b_term.index, DeclRef) or b_term.index.name != j_name:
        return None

    return MatMulOp(
        dest=dest_name,
        lhs=lhs_name,
        rhs=rhs_name,
        m=m,
        n=n,
        k=k_dim,
        elem_bits=elem_bits,
    )

# =========================
# Helper
# =========================

def _extract_initlist_flat_and_shape(init: InitList) -> Tuple[List[int], Tuple[int, ...]]:
    """
    Converte InitList in:
      - flat: lista piatta di int (row-major per matrici)
      - shape: (L,) per vettori oppure (rows, cols) per matrici

    Supporta:
      - {1,2,3}
      - {{1,2},{3,4}}
    """
    elems = init.elements

    # Caso vuoto: {} (raro)
    if not elems:
        return [], (0,)

    # Se il primo elemento è un InitList => assumiamo matrice (lista di righe)
    if isinstance(elems[0], InitList):
        rows = len(elems)
        # tutte le righe devono essere InitList e avere stessa lunghezza
        row_lengths = []
        flat: List[int] = []
        for row in elems:
            if not isinstance(row, InitList):
                raise ValueError("InitList misto: atteso tutte righe InitList per matrice.")
            row_vals = []
            for cell in row.elements:
                if not isinstance(cell, IntegerLiteral):
                    raise ValueError("InitList matrice: ammessi solo IntegerLiteral per ora.")
                row_vals.append(cell.value)
            row_lengths.append(len(row_vals))
            flat.extend(row_vals)

        if len(set(row_lengths)) != 1:
            raise ValueError(f"InitList matrice non rettangolare: row lengths = {row_lengths}")

        cols = row_lengths[0]
        return flat, (rows, cols)

    # Altrimenti assumiamo vettore {1,2,3}
    flat: List[int] = []
    for e in elems:
        if not isinstance(e, IntegerLiteral):
            raise ValueError("InitList vettore: ammessi solo IntegerLiteral per ora.")
        flat.append(e.value)

    return flat, (len(flat),)


# =========================
# Conversione generale
# =========================

def from_c_ast_to_vecmat(tu: TranslationUnit, elem_bits: int) -> VecMatModule:
    """
    Converte un TranslationUnit (dataclass C) in un VecMatModule
    riconoscendo i pattern:
    - matmul (triplo loop i,j,k)
    - vec_add
    - vec_dot

    In più:
    - raccoglie inizializzazioni costanti tipo:
        int a = {1,2};
        int A = {{1,2},{3,4}};
      salvandole in module.const_arrays / module.const_shapes
    """

    module = VecMatModule()

    for func in tu.decls:
        if not isinstance(func, FunctionDecl):
            continue

        vec_func = VecMatFunction(name=func.name, params=list(func.params))

        # --- PASS 0: raccogli InitList dai VarDecl ---
        for stmt in func.body.stmts:
            if isinstance(stmt, VarDecl) and isinstance(stmt.init, InitList):
                flat, shape = _extract_initlist_flat_and_shape(stmt.init)
                module.const_arrays[stmt.name] = flat
                module.const_shapes[stmt.name] = shape

        # --- PASS 1: matching dei loop (come prima) ---
        for stmt in func.body.stmts:
            if isinstance(stmt, ForStmt):
                # 1) prova matmul sul for esterno (i)
                matmul_op = _match_matmul_for(stmt, elem_bits)
                if matmul_op is not None:
                    vec_func.ops.append(matmul_op)
                    continue

                # 2) prova vec_add
                vec_add_op = _match_vec_add_for(stmt, elem_bits)
                if vec_add_op is not None:
                    vec_func.ops.append(vec_add_op)
                    continue

                # 3) prova vec_dot
                vec_dot_op = _match_vec_dot_for(stmt, elem_bits)
                if vec_dot_op is not None:
                    vec_func.ops.append(vec_dot_op)
                    continue

                # altri for: non gestiti

        module.functions.append(vec_func)

    return module
