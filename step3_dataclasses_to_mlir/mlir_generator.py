from xdsl.ir import Block, Region, SSAValue
from xdsl.dialects.builtin import i32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.arith import ConstantOp, AddiOp, SubiOp, MuliOp, DivSIOp, CmpiOp, ExtUIOp
from .dialect_ops import AddiImmOp, SubiImmOp, MuliImmOp, DivSImmOp
from xdsl.dialects.cf import ConditionalBranchOp as CondBranchOp
from step2_ast_to_dataclasses.c_ast import (
    Expression,
    IntegerLiteral,
    DeclRef,
    BinaryOperator,
    BinaryOperatorWithImmediate,
    UnaryOperator,
    VarDecl,
    AssignStmt,
    ReturnStmt,
    FunctionDecl,
    CompoundStmt,
    IfStmt,
    ForStmt,
    ArrayAccess,
)

MAX_UNROLL = 10


class MLIRGenerator:
    def __init__(self) -> None:
        self.symbol_table: dict[str, SSAValue | None] = {}
        self.current_block: Block | None = None
        self.function_region: Region | None = None

    # ------------------------------------------------------------------
    # Espressioni
    # ------------------------------------------------------------------
    def process_expression(self, expr: Expression) -> SSAValue:
        if isinstance(expr, IntegerLiteral):
            op = ConstantOp.from_int_and_width(expr.value, 32)
            self.current_block.add_op(op)
            return op.results[0]
        
        if isinstance(expr, DeclRef):
            if expr.name not in self.symbol_table or self.symbol_table[expr.name] is None:
                raise ValueError(f"Use of undeclared or uninitialized variable '{expr.name}'")
            return self.symbol_table[expr.name]

        if isinstance(expr, UnaryOperator):
            operand_val = self.process_expression(expr.operand)
            if expr.opcode == '+':
                return operand_val
            if expr.opcode == '-':
                zero = ConstantOp.from_int_and_width(0, 32)
                self.current_block.add_op(zero)
                op = SubiOp(zero.results[0], operand_val)
                self.current_block.add_op(op)
                return op.results[0]
            if expr.opcode == '!':
                zero = ConstantOp.from_int_and_width(0, 32)
                self.current_block.add_op(zero)
                cmp = CmpiOp(operand_val, zero.results[0], "eq")
                self.current_block.add_op(cmp)
                return cmp.results[0]
            if expr.opcode == '~':
                zero = ConstantOp.from_int_and_width(0, 32)
                one = ConstantOp.from_int_and_width(1, 32)
                self.current_block.add_op(zero)
                self.current_block.add_op(one)
                neg = SubiOp(zero.results[0], operand_val)
                self.current_block.add_op(neg)
                res = SubiOp(neg.results[0], one.results[0])
                self.current_block.add_op(res)
                return res.results[0]
            if expr.opcode in ('++', '--'):
                if not isinstance(expr.operand, DeclRef):
                    raise ValueError("Increment/decrement requires variable reference")
                var_name = expr.operand.name
                imm = 1
                if expr.opcode == '++':
                    op = AddiImmOp(operand_val, imm)
                else:
                    op = SubiImmOp(operand_val, imm)
                self.current_block.add_op(op)
                self.symbol_table[var_name] = op.results[0]
                return operand_val if expr.is_postfix else op.results[0]
            raise ValueError(f"Unsupported unary operator: {expr.opcode}")

        if isinstance(expr, BinaryOperator):
            lhs_val = self.process_expression(expr.lhs)
            rhs_val = self.process_expression(expr.rhs)
            arith_map = {'+': AddiOp, '-': SubiOp, '*': MuliOp, '/': DivSIOp}
            cmp_map = {'==': "eq", '!=': "ne", '<': "slt", '<=': "sle", '>': "sgt", '>=': "sge"}

            if expr.opcode in arith_map:
                op = arith_map[expr.opcode](lhs_val, rhs_val)
                self.current_block.add_op(op)
                return op.results[0]
            elif expr.opcode in cmp_map:
                op = CmpiOp(lhs_val, rhs_val, cmp_map[expr.opcode])
                self.current_block.add_op(op)
                return op.results[0]
            raise ValueError(f"Unsupported binary operator: {expr.opcode}")

        if isinstance(expr, BinaryOperatorWithImmediate):
            arith_map = {
                '+': AddiImmOp,
                '-': SubiImmOp,
                '*': MuliImmOp,
                '/': DivSImmOp,
            }

            cmp_map = {
                '==': "eq",
                '!=': "ne",
                '<': "slt",
                '<=': "sle",
                '>': "sgt",
                '>=': "sge",
            }

            # caso: immediato a sinistra
            if isinstance(expr.lhs, IntegerLiteral):
                imm_val = expr.lhs.value
                rhs_val = self.process_expression(expr.rhs)

                if expr.opcode in ('+', '*'):  # commutativi
                    op = arith_map[expr.opcode](rhs_val, imm_val)
                    self.current_block.add_op(op)
                    return op.results[0]
                else:
                    raise ValueError(f"Unsupported lhs-immediate for non-commutative op: {expr.opcode}")

            # caso: immediato a destra
            elif isinstance(expr.rhs, IntegerLiteral):
                lhs_val = self.process_expression(expr.lhs)
                imm_val = expr.rhs.value

                if expr.opcode in arith_map:
                    op = arith_map[expr.opcode](lhs_val, imm_val)
                    self.current_block.add_op(op)
                    return op.results[0]
                
                elif expr.opcode in cmp_map:
                    const_op = ConstantOp.from_int_and_width(imm_val, 32)
                    self.current_block.add_op(const_op)
                    rhs_val = const_op.results[0]
                    cmp_op = CmpiOp(lhs_val, rhs_val, cmp_map[expr.opcode])
                    self.current_block.add_op(cmp_op)
                    return cmp_op.results[0]

            raise ValueError("BinaryOperatorWithImmediate must contain an IntegerLiteral on one side")

        raise TypeError(f"Unsupported expression type: {type(expr)}")

    # ------------------------------------------------------------------
    # Helper per gli statement “semplici”
    # ------------------------------------------------------------------
    def lower_vardecl(self, stmt: VarDecl) -> None:
        """
        Abbassa una dichiarazione di variabile scalare.
        Se non ha init, per semplicità si inizializza a 0.
        """
        if stmt.init is not None:
            val = self.process_expression(stmt.init)
        else:
            const0 = ConstantOp.from_int_and_width(0, 32)
            self.current_block.add_op(const0)
            val = const0.results[0]
        self.symbol_table[stmt.name] = val

    def lower_assign(self, stmt: AssignStmt) -> None:
        """
        Abbassa un assegnamento scalare:
            x = expr;
        Se il LHS non è un nome scalare (es. è ArrayAccess), per il backend
        scalare lo si ignora: quei casi sono gestiti dal dialetto vettoriale.
        """
        # LHS scalare: nome o DeclRef
        if isinstance(stmt.name, str):
            var_name = stmt.name
        elif isinstance(stmt.name, DeclRef):
            var_name = stmt.name.name
        else:
            # Esempio: ArrayAccess per vettori/matrici -> ignorato qui
            return

        rhs_val = self.process_expression(stmt.value)
        self.symbol_table[var_name] = rhs_val

    def lower_return(self, stmt: ReturnStmt) -> None:
        """
        Abbassa un return; se senza valore, ritorna 0.
        """
        if stmt.value is None:
            const0 = ConstantOp.from_int_and_width(0, 32)
            self.current_block.add_op(const0)
            ret = ReturnOp(const0.results[0])
        else:
            val = self.process_expression(stmt.value)
            ret = ReturnOp(val)
        self.current_block.add_op(ret)

    # ------------------------------------------------------------------
    # Abbassamento di blocchi (con skip dei for vettoriali)
    # ------------------------------------------------------------------
    def _lower_block(self, stmts: list) -> None:
        """
        Abbassa una lista di statement del nostro AST in MLIR scalare.

        Nota importante:
        - i loop che implementano kernel vettoriali/matriciali (vec_add, vec_dot, ecc.)
          vengono SALTATI qui, perché sono gestiti dal dialetto VecMat/QAR.
        """
        i = 0
        while i < len(stmts):
            stmt = stmts[i]

            # -------------------------
            # Dichiarazioni di variabili
            # -------------------------
            if isinstance(stmt, VarDecl):
                self.lower_vardecl(stmt)

            # ---------------
            # Assegnamenti
            # ---------------
            elif isinstance(stmt, AssignStmt):
                self.lower_assign(stmt)

            # ---------------
            # If
            # ---------------
            elif isinstance(stmt, IfStmt):
                self.lower_if(stmt, stmts[i + 1:])

            # ---------------
            # For
            # ---------------
            elif isinstance(stmt, ForStmt):
                is_vec_loop = False
                body = stmt.body

                if isinstance(body, CompoundStmt) and len(body.stmts) == 1:
                    inner = body.stmts[0]

                    # Caso 1: c[i] = a[i] + b[i];  (vec_add)
                    if isinstance(inner, AssignStmt):
                        # LHS array -> tipico vec_add
                        if isinstance(inner.name, ArrayAccess):
                            is_vec_loop = True
                        else:
                            # Caso 2: s = s + a[i] * b[i];  (vec_dot)
                            rhs = inner.value
                            if (
                                isinstance(rhs, BinaryOperator)
                                and rhs.opcode == "+"
                                and isinstance(rhs.rhs, BinaryOperator)
                                and rhs.rhs.opcode == "*"
                                and isinstance(rhs.rhs.lhs, ArrayAccess)
                                and isinstance(rhs.rhs.rhs, ArrayAccess)
                            ):
                                # pattern del prodotto scalare
                                is_vec_loop = True

                if is_vec_loop:
                    # Questo for è un kernel vettoriale/matriciale:
                    # viene gestito dal dialetto VecMat/QAR, quindi
                    # NON lo abbassiamo in MLIR scalare.
                    i += 1
                    continue

                # Caso normale: for davvero scalare → usa la logica originale
                self.lower_for(stmt, stmts[i + 1:])

            # ---------------
            # Altri statement (se presenti)
            # ---------------

            i += 1

    # ------------------------------------------------------------------
    # If / For (come prima, ma usando _lower_block aggiornato)
    # ------------------------------------------------------------------
    def lower_if(self, stmt: IfStmt, tail: list) -> None:
        if isinstance(stmt.condition, BinaryOperator) and stmt.condition.opcode in ("&&", "||"):
            lhs = stmt.condition.lhs
            rhs = stmt.condition.rhs

            if stmt.condition.opcode == "&&":
                nested_if = IfStmt(
                    condition=rhs,
                    then_body=stmt.then_body,
                    else_body=None
                )
                outer_if = IfStmt(
                    condition=lhs,
                    then_body=CompoundStmt([nested_if]),
                    else_body=stmt.else_body
                )
                self.lower_if(outer_if, tail)

            elif stmt.condition.opcode == "||":
                then_copy_1 = stmt.then_body
                then_copy_2 = stmt.then_body
                self.lower_if(IfStmt(condition=lhs, then_body=then_copy_1), tail)
                self.lower_if(IfStmt(condition=rhs, then_body=then_copy_2), tail)
            return

        cond_val = self.process_expression(stmt.condition)
        then_block = Block()
        else_block = Block()
        self.function_region.add_block(then_block)
        self.function_region.add_block(else_block)

        self.current_block.add_op(CondBranchOp(cond_val, then_block, [], else_block, []))

        original_symtable = dict(self.symbol_table)

        self.symbol_table = dict(original_symtable)
        self.current_block = then_block
        self._lower_block(stmt.then_body.stmts + tail)

        self.symbol_table = dict(original_symtable)
        self.current_block = else_block
        if stmt.else_body:
            self._lower_block(stmt.else_body.stmts + tail)
        else:
            self._lower_block(tail)

    def lower_for(self, stmt: ForStmt, tail: list) -> None:
        if isinstance(stmt.init, VarDecl):
            self.symbol_table[stmt.init.name] = self.process_expression(stmt.init.init)
        elif isinstance(stmt.init, AssignStmt):
            self.symbol_table[stmt.init.name] = self.process_expression(stmt.init.value)

        cond_val = self.process_expression(stmt.condition)
        then_block = Block()
        else_block = Block()
        self.function_region.add_block(then_block)
        self.function_region.add_block(else_block)
        self.current_block.add_op(CondBranchOp(cond_val, then_block, [], else_block, []))

        # ramo "else": esci dal loop, continua col tail
        self.current_block = else_block
        self._lower_block(tail)

        # unrolling statico fino a MAX_UNROLL
        for _ in range(MAX_UNROLL):
            self.current_block = then_block
            self._lower_block(stmt.body.stmts)

            if stmt.increment:
                self.symbol_table[stmt.increment.name] = self.process_expression(stmt.increment.value)

            cond_val = self.process_expression(stmt.condition)
            next_then = Block()
            next_else = Block()
            self.function_region.add_block(next_then)
            self.function_region.add_block(next_else)
            self.current_block.add_op(CondBranchOp(cond_val, next_then, [], next_else, []))

            self.current_block = next_else
            self._lower_block(tail)

            then_block = next_then

    # ------------------------------------------------------------------
    # Generazione funzione
    # ------------------------------------------------------------------
    def generate_function(self, func: FunctionDecl) -> FuncOp:
        self.symbol_table.clear()
        entry_block = Block()
        self.function_region = Region()
        self.function_region.add_block(entry_block)
        self.current_block = entry_block

        self._lower_block(func.body.stmts)

        func_type = ([i32] * len(func.params), [i32])
        return FuncOp(func.name, func_type, self.function_region)