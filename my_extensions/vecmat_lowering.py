# my_extensions/vecmat_lowering.py

from typing import Optional, List, Tuple, Iterable, Any

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
    InitList,
)

from my_extensions.vecmat_ir import (
    VecMatModule,
    VecMatFunction,
    VecAddOp,
    VecDotOp,
    MatMulOp,
)


# =========================
# Normalizzazione stmt list
# =========================

def _flatten_stmts(stmts: Iterable[Any]) -> List[Any]:
    """
    Alcuni AST possono contenere liste annidate dentro CompoundStmt.stmts, es:
        stmts=[[VarDecl(...)], ForStmt(...), AssignStmt(...)]
    Questa funzione appiattisce ricorsivamente a:
        [VarDecl(...), ForStmt(...), AssignStmt(...)]
    """
    out: List[Any] = []
    for s in stmts:
        if isinstance(s, list):
            out.extend(_flatten_stmts(s))
        else:
            out.append(s)
    return out


def _compound_stmts(body: CompoundStmt) -> List[Any]:
    """Ritorna la lista stmt normalizzata (flatten) di un CompoundStmt."""
    return _flatten_stmts(getattr(body, "stmts", []))


def _is_simple_for_header(for_stmt: ForStmt) -> tuple[bool, Optional[str], Optional[int]]:
    """
    Heuristica: riconosce header for standard:
        for (int i = 0; i < N; i++)
    Ritorna (ok, idx_name, bound).
    """
    init = getattr(for_stmt, "init", None)
    if not isinstance(init, VarDecl):
        return False, None, None
    if not isinstance(init.init, IntegerLiteral):
        return False, None, None
    if init.init.value != 0:
        return False, None, None
    idx_name = init.name

    cond = getattr(for_stmt, "condition", None)
    if not isinstance(cond, BinaryOperatorWithImmediate):
        return False, None, None
    if cond.opcode not in ("<", "<="):
        return False, None, None
    if not isinstance(cond.lhs, DeclRef) or cond.lhs.name != idx_name:
        return False, None, None
    if not isinstance(cond.rhs, IntegerLiteral):
        return False, None, None
    bound = cond.rhs.value

    incr = getattr(for_stmt, "increment", None)
    if not isinstance(incr, AssignStmt):
        return False, None, None
    if not isinstance(incr.name, str) or incr.name != idx_name:
        return False, None, None
    incr_expr = incr.value
    if not isinstance(incr_expr, BinaryOperatorWithImmediate):
        return False, None, None
    if incr_expr.opcode != "+":
        return False, None, None
    if not isinstance(incr_expr.lhs, DeclRef) or incr_expr.lhs.name != idx_name:
        return False, None, None
    if not isinstance(incr_expr.rhs, IntegerLiteral) or incr_expr.rhs.value != 1:
        return False, None, None

    return True, idx_name, bound


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
    init = for_stmt.init
    if not isinstance(init, VarDecl):
        return None
    if not isinstance(init.init, IntegerLiteral):
        return None
    if init.init.value != 0:
        return None
    idx_name = init.name

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

    body = for_stmt.body
    if not isinstance(body, CompoundStmt):
        return None
    stmts = _compound_stmts(body)
    if len(stmts) != 1:
        return None
    stmt = stmts[0]
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
    init = for_stmt.init
    if not isinstance(init, VarDecl):
        return None
    if not isinstance(init.init, IntegerLiteral):
        return None
    if init.init.value != 0:
        return None
    idx_name = init.name

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

    body = for_stmt.body
    if not isinstance(body, CompoundStmt):
        return None
    stmts = _compound_stmts(body)
    if len(stmts) != 1:
        return None
    stmt = stmts[0]
    if not isinstance(stmt, AssignStmt):
        return None

    if not isinstance(stmt.name, str):
        return None
    dest_name = stmt.name

    rhs = stmt.value
    if not isinstance(rhs, BinaryOperator) or rhs.opcode != "+":
        return None

    if not isinstance(rhs.lhs, DeclRef) or rhs.lhs.name != dest_name:
        return None

    mul = rhs.rhs
    if not isinstance(mul, BinaryOperator) or mul.opcode != "*":
        return None

    lhs_term = mul.lhs
    if not isinstance(lhs_term, ArrayAccess):
        return None
    if not isinstance(lhs_term.array, DeclRef):
        return None
    vec_a = lhs_term.array.name
    if not isinstance(lhs_term.index, DeclRef) or lhs_term.index.name != idx_name:
        return None

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
    Riconosce (forma “matmul classica” con accumulatore locale s):
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int s = 0;
                for (int k = 0; k < K; k++) {
                    s = s + A[i][k] * B[k][j];
                }
                C[i][j] = s;
            }
        }

    Nota: la versione precedente assumeva body_j.stmts esattamente di lunghezza 2:
        zero_stmt (Assign c[i][j]=0) + for_k
    Ma l’AST osservato nel test contiene:
        [VarDecl s=0, for_k(... update su s ...), Assign C[i][j]=s]
    e, in più, può contenere liste annidate (stmts=[[VarDecl(...)], ...]).
    """
    ok_i, i_name, m = _is_simple_for_header(for_i)
    if not ok_i or i_name is None or m is None:
        return None

    body_i = for_i.body
    if not isinstance(body_i, CompoundStmt):
        return None
    stmts_i = _compound_stmts(body_i)
    if len(stmts_i) != 1 or not isinstance(stmts_i[0], ForStmt):
        return None
    for_j = stmts_i[0]

    ok_j, j_name, n = _is_simple_for_header(for_j)
    if not ok_j or j_name is None or n is None:
        return None

    body_j = for_j.body
    if not isinstance(body_j, CompoundStmt):
        return None
    stmts_j = _compound_stmts(body_j)

    # Ci aspettiamo: VarDecl(s=0), ForStmt(k-loop), Assign(C[i][j]=s)
    if len(stmts_j) != 3:
        return None
    s_decl, for_k, store_c = stmts_j

    if not isinstance(s_decl, VarDecl):
        return None
    if not isinstance(s_decl.init, IntegerLiteral) or s_decl.init.value != 0:
        return None
    s_name = s_decl.name

    if not isinstance(for_k, ForStmt):
        return None

    if not isinstance(store_c, AssignStmt):
        return None

    # store_c: C[i][j] = s
    lhs_store = store_c.name
    if not isinstance(lhs_store, ArrayAccess):
        return None
    if not isinstance(lhs_store.array, ArrayAccess):
        return None
    c_outer = lhs_store.array
    if not isinstance(c_outer.array, DeclRef):
        return None
    dest_name = c_outer.array.name
    if not isinstance(c_outer.index, DeclRef) or c_outer.index.name != i_name:
        return None
    if not isinstance(lhs_store.index, DeclRef) or lhs_store.index.name != j_name:
        return None
    if not isinstance(store_c.value, DeclRef) or store_c.value.name != s_name:
        return None

    # k-loop header
    ok_k, k_name, k_dim = _is_simple_for_header(for_k)
    if not ok_k or k_name is None or k_dim is None:
        return None

    body_k = for_k.body
    if not isinstance(body_k, CompoundStmt):
        return None
    stmts_k = _compound_stmts(body_k)
    if len(stmts_k) != 1:
        return None
    update_stmt = stmts_k[0]
    if not isinstance(update_stmt, AssignStmt):
        return None

    # update: s = s + A[i][k] * B[k][j]
    if not isinstance(update_stmt.name, str) or update_stmt.name != s_name:
        return None

    rhs_update = update_stmt.value
    if not isinstance(rhs_update, BinaryOperator) or rhs_update.opcode != "+":
        return None
    if not isinstance(rhs_update.lhs, DeclRef) or rhs_update.lhs.name != s_name:
        return None

    mul = rhs_update.rhs
    if not isinstance(mul, BinaryOperator) or mul.opcode != "*":
        return None

    # A[i][k]
    a_term = mul.lhs
    if not isinstance(a_term, ArrayAccess):
        return None
    if not isinstance(a_term.array, ArrayAccess):
        return None
    a_outer = a_term.array
    if not isinstance(a_outer.array, DeclRef):
        return None
    lhs_name = a_outer.array.name
    if not isinstance(a_outer.index, DeclRef) or a_outer.index.name != i_name:
        return None
    if not isinstance(a_term.index, DeclRef) or a_term.index.name != k_name:
        return None

    # B[k][j]
    b_term = mul.rhs
    if not isinstance(b_term, ArrayAccess):
        return None
    if not isinstance(b_term.array, ArrayAccess):
        return None
    b_outer = b_term.array
    if not isinstance(b_outer.array, DeclRef):
        return None
    rhs_name = b_outer.array.name
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

    if not elems:
        return [], (0,)

    if isinstance(elems[0], InitList):
        rows = len(elems)
        row_lengths: List[int] = []
        flat: List[int] = []

        for row in elems:
            if not isinstance(row, InitList):
                raise ValueError("InitList misto: atteso tutte righe InitList per matrice.")
            row_vals: List[int] = []
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
    Converte TranslationUnit in VecMatModule:
    - riconosce vec_add, vec_dot, matmul
    - raccoglie InitList dei VarDecl in:
        module.const_arrays / module.const_shapes
    """
    module = VecMatModule()

    for func in tu.decls:
        if not isinstance(func, FunctionDecl):
            continue

        vec_func = VecMatFunction(name=func.name, params=list(func.params))

        # PASS 0: raccogli InitList
        for stmt in func.body.stmts:
            if isinstance(stmt, VarDecl) and isinstance(stmt.init, InitList):
                flat, shape = _extract_initlist_flat_and_shape(stmt.init)

                # Se la stessa variabile è inizializzata più volte (caso raro), si sovrascrive.
                module.const_arrays[stmt.name] = flat
                module.const_shapes[stmt.name] = shape

        # PASS 1: matching dei loop
        for stmt in func.body.stmts:
            if isinstance(stmt, ForStmt):
                matmul_op = _match_matmul_for(stmt, elem_bits)
                if matmul_op is not None:
                    vec_func.ops.append(matmul_op)
                    continue

                vec_add_op = _match_vec_add_for(stmt, elem_bits)
                if vec_add_op is not None:
                    vec_func.ops.append(vec_add_op)
                    continue

                vec_dot_op = _match_vec_dot_for(stmt, elem_bits)
                if vec_dot_op is not None:
                    vec_func.ops.append(vec_dot_op)
                    continue

        # Diagnostica minima: se ci sono matrici 2D costanti ma nessun MatMulOp matchato,
        # e nel body compare almeno un ForStmt (potenziale triplo loop), segnala.
        try:
            has_2d_const = any(
                isinstance(sh, tuple) and len(sh) == 2
                for sh in module.const_shapes.values()
            )
            has_any_for = any(isinstance(s, ForStmt) for s in getattr(func.body, "stmts", []))
            has_matmul_op = any(isinstance(op, MatMulOp) for op in vec_func.ops)
            if has_2d_const and has_any_for and not has_matmul_op:
                print(
                    f"[WARN] Possibile matmul presente in '{func.name}' ma pattern non riconosciuto. "
                    f"Controllare la forma dell'AST (CompoundStmt.stmts annidati) o la struttura del triplo loop."
                )
        except Exception:
            pass

        module.functions.append(vec_func)

    return module