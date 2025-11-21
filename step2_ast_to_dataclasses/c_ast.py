# =============================================================================
# AST Node Definitions and Parsing Utilities
# =============================================================================

"""Dataclasses representing a tiny C-like language and parsing helpers.

This module defines a very small set of dataclasses capturing the subset of C
that is handled by the rest of the repository.  It also contains utilities to
convert the JSON output produced by ``clang -ast-dump=json`` into these
dataclasses and back into a pretty-printed C-like source.  The goal is to keep
the front-end logic separate from the MLIR generation and quantum translation
stages.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union


# -----------------------------------------------------------------------------
# AST Classes
# -----------------------------------------------------------------------------

class Expression:
    """Abstract base class for all expressions."""
    pass


@dataclass
class IntegerLiteral(Expression):
    """Integer constant."""

    value: int


@dataclass
class DeclRef(Expression):
    """Reference to a previously declared variable."""

    name: str


@dataclass
class BinaryOperator(Expression):
    """Binary operation between ``lhs`` and ``rhs``."""

    opcode: str
    lhs: Expression
    rhs: Expression


@dataclass
class BinaryOperatorWithImmediate(Expression):
    """Binary operation where one side is an immediate."""

    opcode: str
    lhs: Expression
    rhs: Expression

@dataclass
class UnaryOperator(Expression):
    """Unary operation like ``-x`` or ``x++``.

    ``is_postfix`` distinguishes prefix from postfix forms for ``++`` and ``--``.
    For all other operators the flag is ignored.
    """
    opcode: str
    operand: Expression
    is_postfix: bool = False
    
@dataclass
class ArrayAccess(Expression):
    """Accesso a un elemento di array/vettore, es. c[i]."""

    array: Expression   # es. DeclRef("c")
    index: Expression   # es. DeclRef("i") oppure BinaryOperator(...)


@dataclass
class VarDecl:
    """Variable declaration optionally with initialization."""

    name: str
    init: Optional[Expression] = None


@dataclass
class ReturnStmt:
    """Return statement with optional value."""

    value: Optional[Expression] = None


@dataclass
class AssignStmt:
    """Assignment of ``value`` to variable ``name``."""

    name: Union[str, Expression]
    value: Expression


@dataclass
class CompoundStmt:
    stmts: List[Union[VarDecl, ReturnStmt, AssignStmt, 'IfStmt']] = field(default_factory=list)

@dataclass
class FunctionDecl:
    """Function definition."""

    name: str
    body: CompoundStmt
    params: List[str] = field(default_factory=list)


@dataclass
class TranslationUnit:
    """Top-level container of all functions."""

    decls: List[FunctionDecl] = field(default_factory=list)

@dataclass
class IfStmt:
    condition: Expression
    then_body: CompoundStmt
    else_body: Optional[CompoundStmt] = None

@dataclass
class ForStmt:
    init: Optional[Union[VarDecl, AssignStmt]]
    condition: Optional[Expression]
    increment: Optional[AssignStmt]
    body: CompoundStmt
    



# -----------------------------------------------------------------------------
# Expression Parsing
# -----------------------------------------------------------------------------

def parse_expression(expr: Dict) -> Expression:
    kind = expr.get("kind")

    if kind == "IntegerLiteral":
        # letterale intero, es. 5
        value = int(expr["value"])
        return IntegerLiteral(value)

    elif kind == "DeclRefExpr":
        # riferimento a variabile, es. acc, i, a, c
        name = expr.get("name") or expr.get("referencedDecl", {}).get("name")
        return DeclRef(name)

    elif kind == "BinaryOperator":
        # operazione binaria, es. a + b, i < N, ...
        opcode = expr["opcode"]
        inner = expr.get("inner", [])
        lhs = parse_expression(inner[0])
        rhs = parse_expression(inner[1])

        # se uno dei due è una costante intera, possiamo usare BinaryOperatorWithImmediate
        if isinstance(lhs, IntegerLiteral) or isinstance(rhs, IntegerLiteral):
            return BinaryOperatorWithImmediate(opcode, lhs, rhs)
        return BinaryOperator(opcode, lhs, rhs)

    elif kind == "UnaryOperator":
        # operatore unario, es. -x, ++i, i++
        opcode = expr["opcode"]
        inner = expr.get("inner", [])
        operand = parse_expression(inner[0])
        # se clang fornisse info su prefix/postfix, qui si potrebbe usarla;
        # per ora impostiamo is_postfix=False di default
        return UnaryOperator(opcode, operand, is_postfix=False)

    elif kind == "ParenExpr":
        # parentesi, es. (a + b)
        inner = expr.get("inner", [])
        return parse_expression(inner[0])

    elif kind == "ImplicitCastExpr":
        # cast implicito: in genere basta propagare l'espressione interna
        inner = expr.get("inner", [])
        return parse_expression(inner[0])

    elif kind == "ArraySubscriptExpr":
        # accesso a vettore/array, es. c[i], a[i+1]
        inner = expr.get("inner", [])
        if len(inner) != 2:
            raise ValueError("ArraySubscriptExpr with unexpected inner length")

        base_expr = parse_expression(inner[0])   # es. DeclRef("c")
        index_expr = parse_expression(inner[1])  # es. DeclRef("i") o BinaryOperator(...)
        return ArrayAccess(base_expr, index_expr)

    else:
        raise ValueError(f"Unsupported expression node: {kind}")



# -----------------------------------------------------------------------------
# AST Parser (JSON -> Dataclasses)
# -----------------------------------------------------------------------------

def parse_ast(ast_json: Dict) -> TranslationUnit:
    """Convert a complete Clang JSON AST into a :class:`TranslationUnit`.

    Parameters
    ----------
    ast_json:
        Parsed JSON object produced by ``clang -ast-dump=json``.

    Returns
    -------
    TranslationUnit
        The root of the dataclass representation.
    """
    tu = TranslationUnit()

    # Walk over the top-level declarations in the JSON AST.
    for decl in ast_json.get("inner", []):
        if decl.get("kind") != "FunctionDecl":
            continue  # Skip anything that isn't a function.

        func_name = decl["name"]
        compound_stmt = None

        # Find the body of the function which is represented as a CompoundStmt.
        for inner in decl.get("inner", []):
            if inner.get("kind") != "CompoundStmt":
                continue

            compound_stmt = CompoundStmt()

            # Iterate over all statements inside the compound statement.
            for stmt in inner.get("inner", []):
                parsed = parse_statement(stmt)
                if parsed:
                    if isinstance(parsed, list):
                        compound_stmt.stmts.extend(parsed)
                    else:
                        compound_stmt.stmts.append(parsed)


        # If we successfully built a function body, add the function to the TU.
        if compound_stmt:
            tu.decls.append(FunctionDecl(func_name, compound_stmt))

    return tu

def parse_statement(stmt: Dict) -> Optional[Union[VarDecl, AssignStmt, ReturnStmt, IfStmt]]:
    kind = stmt.get("kind")

    if kind == "DeclStmt":
        decls: list[VarDecl] = []
        for var_decl in stmt.get("inner", []):
            if var_decl.get("kind") == "VarDecl":
                init_expr = None
                if "inner" in var_decl and var_decl["inner"]:
                    init_expr = parse_expression(var_decl["inner"][0])
                decls.append(VarDecl(var_decl["name"], init_expr))
        return decls  # restituisce lista di VarDecl

    elif kind == "BinaryOperator" and stmt["opcode"] == "=":
        lhs = stmt["inner"][0]
        rhs = stmt["inner"][1]

        # Gestione LHS:
        # - se è una variabile semplice (DeclRefExpr), usiamo il nome
        # - altrimenti (es. ArraySubscriptExpr) lo trattiamo come espressione completa
        if lhs.get("kind") == "DeclRefExpr":
            var_name = lhs.get("name") or lhs.get("referencedDecl", {}).get("name")
            target = var_name
        else:
            # Permette LHS come c[i], a[i+1], ecc. usando la stessa logica di parse_expression
            target = parse_expression(lhs)

        rhs_expr = parse_expression(rhs)
        return AssignStmt(target, rhs_expr)

    elif kind == "UnaryOperator" and stmt.get("opcode") == "++":
        # Gestione di i++ come statement di incremento: lo normalizziamo a i = i + 1
        inner = stmt.get("inner", [])
        if not inner:
            raise ValueError("UnaryOperator ++ senza operando")

        target_expr = parse_expression(inner[0])
        if not isinstance(target_expr, DeclRef):
            raise ValueError("Unsupported ++ on non-variable expression")

        # i = i + 1  → BinaryOperatorWithImmediate("+", i, 1)
        inc_expr = BinaryOperatorWithImmediate(
            opcode="+",
            lhs=target_expr,
            rhs=IntegerLiteral(1),
        )
        # AssignStmt.name: usiamo il nome della variabile, non l'espressione
        return AssignStmt(target_expr.name, inc_expr)

    elif kind == "ReturnStmt":
        if "inner" in stmt and stmt["inner"]:
            return_expr = parse_expression(stmt["inner"][0])
            return ReturnStmt(return_expr)
        else:
            return ReturnStmt()

    elif kind == "IfStmt":
        condition = parse_expression(stmt["inner"][0])

        then_raw = stmt["inner"][1]
        then_block = CompoundStmt()
        if then_raw["kind"] == "CompoundStmt":
            for s in then_raw.get("inner", []):
                parsed = parse_statement(s)
                if parsed:
                    then_block.stmts.append(parsed)

        else_block = None
        if len(stmt["inner"]) > 2:
            else_raw = stmt["inner"][2]
            if else_raw["kind"] == "IfStmt":
                # this is an else if
                nested_if = parse_statement(else_raw)
                else_block = CompoundStmt(stmts=[nested_if]) if nested_if else None
            elif else_raw["kind"] == "CompoundStmt":
                else_block = CompoundStmt()
                for s in else_raw.get("inner", []):
                    parsed = parse_statement(s)
                    if parsed:
                        else_block.stmts.append(parsed)

        return IfStmt(condition, then_block, else_block)
    
    elif kind == "ForStmt":
        inner = stmt.get("inner", [])
        real_inner = [x for x in inner if isinstance(x, dict) and 'kind' in x]

        init_stmt = parse_statement(real_inner[0]) if len(real_inner) > 0 else None
        if isinstance(init_stmt, list):  # fix qui
            init_stmt = init_stmt[0] if init_stmt else None

        condition_expr = parse_expression(real_inner[1]) if len(real_inner) > 1 else None
        increment_stmt = parse_statement(real_inner[2]) if len(real_inner) > 2 else None

        body = CompoundStmt()
        if len(real_inner) > 3 and real_inner[3].get("kind") == "CompoundStmt":
            for s in real_inner[3].get("inner", []):
                parsed = parse_statement(s)
                if parsed:
                    body.stmts.append(parsed)

        return ForStmt(init_stmt, condition_expr, increment_stmt, body)

    return None





# -----------------------------------------------------------------------------
# Pretty-Printing Utilities
# -----------------------------------------------------------------------------

def pretty_print_statement(stmt, indent=1) -> List[str]:
    """Pretty-print any statement with correct indentation."""
    indent_str = "    " * indent
    lines: List[str] = []

    if isinstance(stmt, VarDecl):
        if stmt.init:
            expr_str = pretty_print_expression(stmt.init)
            lines.append(f"{indent_str}int {stmt.name} = {expr_str};")
        else:
            lines.append(f"{indent_str}int {stmt.name};")

    elif isinstance(stmt, AssignStmt):
        expr_str = pretty_print_expression(stmt.value)
        lines.append(f"{indent_str}{stmt.name} = {expr_str};")

    elif isinstance(stmt, ReturnStmt):
        if stmt.value:
            expr_str = pretty_print_expression(stmt.value)
            lines.append(f"{indent_str}return {expr_str};")
        else:
            lines.append(f"{indent_str}return;")

    elif isinstance(stmt, IfStmt):
        cond_str = pretty_print_expression(stmt.condition)
        lines.append(f"{indent_str}if ({cond_str}) {{")
        for inner in stmt.then_body.stmts:
            lines.extend(pretty_print_statement(inner, indent + 1))
        lines.append(f"{indent_str}}}")
        if stmt.else_body:
            lines.append(f"{indent_str}else {{")
            for inner in stmt.else_body.stmts:
                lines.extend(pretty_print_statement(inner, indent + 1))
            lines.append(f"{indent_str}}}")
    
    elif isinstance(stmt, ForStmt):
        # Print init (either VarDecl or AssignStmt)
        if isinstance(stmt.init, VarDecl):
            init_str = f"int {stmt.init.name} = {pretty_print_expression(stmt.init.init)}" if stmt.init.init else f"int {stmt.init.name}"
        elif isinstance(stmt.init, AssignStmt):
            init_str = f"{stmt.init.name} = {pretty_print_expression(stmt.init.value)}"
        else:
            init_str = ''

        # Print condition
        cond_str = pretty_print_expression(stmt.condition) if stmt.condition else ''

        # Print increment (must be an AssignStmt)
        if isinstance(stmt.increment, AssignStmt):
            incr_str = f"{stmt.increment.name} = {pretty_print_expression(stmt.increment.value)}"
        else:
            incr_str = ''

        # Emit the for loop
        lines.append(f"{indent_str}for ({init_str}; {cond_str}; {incr_str}) {{")
        for inner in stmt.body.stmts:
            lines.extend(pretty_print_statement(inner, indent + 1))
        lines.append(f"{indent_str}}}")


    

    else:
        lines.append(f"{indent_str}// Unsupported statement: {type(stmt).__name__}")

    return lines

def pretty_print_translation_unit(tu: TranslationUnit) -> str:
    lines: List[str] = []

    for func in tu.decls:
        params = ", ".join(f"int {p}" for p in func.params)
        lines.append(f"int {func.name}({params}) {{")

        for stmt in func.body.stmts:
            lines.extend(pretty_print_statement(stmt, indent=1))

        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def pretty_print_expression(expr: Expression) -> str:
    """Convert an :class:`Expression` into a C-style string.

    Parameters
    ----------
    expr:
        Expression node to print.

    Returns
    -------
    str
        A textual representation of ``expr``.
    """
    # Integer constants appear as-is.
    if isinstance(expr, IntegerLiteral):
        return str(expr.value)
    # Variables are referenced by name.
    if isinstance(expr, DeclRef):
        return expr.name
    # Unary operations
    if isinstance(expr, UnaryOperator):
        operand = pretty_print_expression(expr.operand)
        if expr.is_postfix:
            return f"({operand}{expr.opcode})"
        else:
            return f"({expr.opcode}{operand})"
    # Standard binary operator using infix notation.
    if isinstance(expr, BinaryOperator):
        lhs = pretty_print_expression(expr.lhs)
        rhs = pretty_print_expression(expr.rhs)
        return f"({lhs} {expr.opcode} {rhs})"
    # Binary operator where one side is an immediate.
    if isinstance(expr, BinaryOperatorWithImmediate):
        lhs = pretty_print_expression(expr.lhs)
        rhs = pretty_print_expression(expr.rhs)
        return f"({lhs} {expr.opcode} {rhs})"
    return "<unsupported_expr>"

