"""Utilities to build quantum circuits from quantum MLIR and emit QASM."""
from __future__ import annotations

import os
from typing import Dict

from qiskit import QuantumCircuit
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.builtin import ModuleOp

from step4_mlir_to_quantum_mlir.quantum_dialect import (
    QuantumInitOp,
    QuantumCInitOp,
    QAddiOp,
    QSubiOp,
    QMuliOp,
    QDivSOp,
    QAddiImmOp,
    QSubiImmOp,
    QMuliImmOp,
    QDivSImmOp,
    CQAddiOp,
    CQSubiOp,
    CQMuliOp,
    CQDivSOp,
    CQAddiImmOp,
    CQSubiImmOp,
    CQMuliImmOp,
    CQDivSImmOp,
    QCmpiOp,
    QAndOp,
    QNotOp,
)

from . import q_arithmetics as qa
from . import q_arithmetics_controlled as qac


def generate_circuit(module: ModuleOp, num_bits: int = 16, verbose: bool = False) -> QuantumCircuit:
    """Convert ``module`` using the quantum dialect to a ``QuantumCircuit``."""

    # ------------------------------------------------------------
    # Sync robusto del numero di bit su tutti i moduli coinvolti
    # ------------------------------------------------------------
    qa.set_number_of_bits(num_bits)
    if hasattr(qa, "NUMBER_OF_BITS"):
        qa.NUMBER_OF_BITS = num_bits

    # q_arithmetics_controlled può mantenere uno stato proprio
    if hasattr(qac, "set_number_of_bits"):
        try:
            qac.set_number_of_bits(num_bits)
        except Exception:
            pass
    qac.__dict__["NUMBER_OF_BITS"] = num_bits

    # se qac mantiene un riferimento interno a qa, sincronizza anche quello
    if hasattr(qac, "qa"):
        try:
            qac.qa.set_number_of_bits(num_bits)
        except Exception:
            pass
        if hasattr(qac.qa, "NUMBER_OF_BITS"):
            qac.qa.NUMBER_OF_BITS = num_bits

    qc = QuantumCircuit()
    reg_map: Dict[object, object] = {}

    def log_op(op, msg=None):
        if verbose:
            result = op.results[0] if getattr(op, "results", None) else "?"
            op_type = op.__class__.__name__
            operands = ", ".join(str(a) for a in getattr(op, "operands", []))
            tail = f" -> {msg}" if msg else ""
            print(f"[{op_type}] {result} = {op.name}({operands}){tail}")

    def _get_reg(val):
        """Restituisce il registro associato a `val`, oppure ne crea uno fittizio."""
        if val not in reg_map:
            if verbose:
                print(f"[WARN] Operand {val} non presente in reg_map, inizializzo un registro a 0.")
            reg = qa.initialize_variable(qc, 0)
            reg_map[val] = reg
        return reg_map[val]

    # ------------------------------------------------------------
    # FIX CRITICO: iterazione corretta delle funzioni nel modulo xdsl
    # ------------------------------------------------------------
    top = module.body.blocks[0]

    for top_op in top.ops:
        if not isinstance(top_op, FuncOp):
            # ignora eventuali simboli/operazioni top-level non-funzione
            continue

        block = top_op.body.blocks[0]

        # --------------------------------------------------------
        # (Robustezza) Prima passata: materializza TUTTI gli init
        # così reg_map è popolata prima di qualsiasi uso.
        # --------------------------------------------------------
        for op in block.ops:
            if isinstance(op, QuantumInitOp):
                val = int(op.value.value.data)
                log_op(op, f"init {val}")
                reg = qa.initialize_variable(qc, val)
                reg_map[op.results[0]] = reg

            elif isinstance(op, QuantumCInitOp):
                val = int(op.value.value.data)
                ctrl = _get_reg(op.ctrl)
                log_op(op, f"c_init {val} controlled by {op.ctrl}")
                reg = qac.initialize_variable_controlled(qc, val, ctrl)
                reg_map[op.results[0]] = reg

        # --------------------------------------------------------
        # Seconda passata: tutte le altre operazioni + return
        # --------------------------------------------------------
        for op in block.ops:
            # init già gestiti sopra
            if isinstance(op, (QuantumInitOp, QuantumCInitOp)):
                continue

            if isinstance(op, QAddiOp):
                log_op(op, "add")
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                reg_map[op.results[0]] = qa.add(qc, lhs, rhs)

            elif isinstance(op, QSubiOp):
                log_op(op, "sub")
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                reg_map[op.results[0]] = qa.sub(qc, lhs, rhs)

            elif isinstance(op, QMuliOp):
                log_op(op, "mul")
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                reg_map[op.results[0]] = qa.mul(qc, lhs, rhs)

            elif isinstance(op, QDivSOp):
                log_op(op, "div")
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                reg_map[op.results[0]], _ = qa.div(qc, lhs, rhs)

            elif isinstance(op, QAddiImmOp):
                imm = int(op.imm.value.data)
                log_op(op, f"addi_imm {imm}")
                lhs = _get_reg(op.lhs)
                reg_map[op.results[0]] = qa.addi(qc, lhs, imm)

            elif isinstance(op, QSubiImmOp):
                imm = int(op.imm.value.data)
                log_op(op, f"subi_imm {imm}")
                lhs = _get_reg(op.lhs)
                reg_map[op.results[0]] = qa.subi(qc, lhs, imm)

            elif isinstance(op, QMuliImmOp):
                imm = int(op.imm.value.data)
                log_op(op, f"muli_imm {imm}")
                lhs = _get_reg(op.lhs)
                reg_map[op.results[0]] = qa.muli(qc, lhs, imm)

            elif isinstance(op, QDivSImmOp):
                imm = int(op.imm.value.data)
                log_op(op, f"divi_imm {imm}")
                lhs = _get_reg(op.lhs)
                reg_map[op.results[0]], _ = qa.divi(qc, lhs, imm)

            elif isinstance(op, CQAddiOp):
                log_op(op, "c_add")
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                ctrl = _get_reg(op.ctrl)
                reg_map[op.results[0]] = qac.add_controlled(qc, lhs, rhs, ctrl)

            elif isinstance(op, CQSubiOp):
                log_op(op, "c_sub")
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                ctrl = _get_reg(op.ctrl)
                reg_map[op.results[0]] = qac.sub_controlled(qc, lhs, rhs, ctrl)

            elif isinstance(op, CQMuliOp):
                log_op(op, "c_mul")
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                ctrl = _get_reg(op.ctrl)
                reg_map[op.results[0]] = qac.mul_controlled(qc, lhs, rhs, ctrl)

            elif isinstance(op, CQDivSOp):
                log_op(op, "c_div")
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                ctrl = _get_reg(op.ctrl)
                reg_map[op.results[0]], _ = qac.div_controlled(qc, lhs, rhs, ctrl)

            elif isinstance(op, CQAddiImmOp):
                imm = int(op.imm.value.data)
                log_op(op, f"c_addi_imm {imm}")
                lhs = _get_reg(op.lhs)
                ctrl = _get_reg(op.ctrl)
                reg_map[op.results[0]] = qac.addi_controlled(qc, lhs, imm, ctrl)

            elif isinstance(op, CQSubiImmOp):
                imm = int(op.imm.value.data)
                log_op(op, f"c_subi_imm {imm}")
                lhs = _get_reg(op.lhs)
                ctrl = _get_reg(op.ctrl)
                reg_map[op.results[0]] = qac.subi_controlled(qc, lhs, imm, ctrl)

            elif isinstance(op, CQMuliImmOp):
                imm = int(op.imm.value.data)
                log_op(op, f"c_muli_imm {imm}")
                lhs = _get_reg(op.lhs)
                ctrl = _get_reg(op.ctrl)
                reg_map[op.results[0]] = qac.muli_controlled(qc, lhs, imm, ctrl)

            elif isinstance(op, CQDivSImmOp):
                imm = int(op.imm.value.data)
                log_op(op, f"c_divi_imm {imm}")
                lhs = _get_reg(op.lhs)
                ctrl = _get_reg(op.ctrl)
                reg_map[op.results[0]], _ = qac.divi_controlled(qc, lhs, imm, ctrl)

            elif isinstance(op, QCmpiOp):
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                predicate = int(op.predicate.value.data)
                msg = ["eq", "neq", "lt", "le", "gt", "ge"][predicate]
                log_op(op, f"cmpi.{msg}")

                if predicate == 0:
                    reg_map[op.results[0]] = qa.equal(qc, lhs, rhs)
                elif predicate == 1:
                    reg_map[op.results[0]] = qa.not_equal(qc, lhs, rhs)
                elif predicate == 2:
                    reg_map[op.results[0]] = qa.less_than(qc, lhs, rhs)
                elif predicate == 3:
                    reg_map[op.results[0]] = qa.less_equal(qc, lhs, rhs)
                elif predicate == 4:
                    reg_map[op.results[0]] = qa.greater_than(qc, lhs, rhs)
                elif predicate == 5:
                    reg_map[op.results[0]] = qa.greater_equal(qc, lhs, rhs)
                else:
                    raise NotImplementedError(f"Unsupported cmp predicate: {predicate}")

            elif isinstance(op, QAndOp):
                log_op(op, "and")
                lhs = _get_reg(op.lhs)
                rhs = _get_reg(op.rhs)
                reg_map[op.results[0]] = qa.logical_and(qc, lhs, rhs)

            elif isinstance(op, QNotOp):
                operand = _get_reg(op.operand)
                existing_names = {reg.name for reg in qc.qregs}
                idx = 0
                while f"not{idx}" in existing_names:
                    idx += 1
                out = qa.initialize_bit(qc, 1, f"not{idx}")
                log_op(op, "not")
                qc.cx(operand, out)
                reg_map[op.results[0]] = out

            elif isinstance(op, ReturnOp):
                if not op.operands:
                    log_op(op, "return (void)")
                    continue

                ret_val = op.operands[0]
                log_op(op, f"return {ret_val}")

                if ret_val not in reg_map:
                    if verbose:
                        print(
                            f"[WARN] Return value {ret_val} non presente in reg_map, "
                            f"nessuna misura eseguita (caso ibrido/vettoriale)."
                        )
                    continue

                try:
                    qa.measure(qc, reg_map[ret_val])
                except Exception as e:
                    if "already exists" in str(e):
                        if verbose:
                            print(f"Skipping duplicate measurement for {reg_map[ret_val].name}")
                    else:
                        raise

            else:
                raise NotImplementedError(f"Unsupported op {op.name}")

    return qc


def export_qasm(circuit: QuantumCircuit, path: str) -> str:
    """Write ``circuit`` to ``path`` in QASM format and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from qiskit import qasm2, transpile

    basis_gates = ["u1", "u2", "u3", "cx", "id", "measure", "reset"]
    transpiled = transpile(circuit, basis_gates=basis_gates)

    with open(path, "w") as f:
        f.write(qasm2.dumps(transpiled))
    print(f"QASM circuit written to {path}")
    return path


from qiskit import transpile
from qiskit.qasm2 import dumps


def export_qasm_clifford_t(circuit: QuantumCircuit, path: str) -> str:
    """Export the circuit to QASM with Clifford+T-only basis."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    clifford_t_basis = ["h", "t", "tdg", "s", "sdg", "cx", "x", "measure", "rz", "p", "cp", "crz"]
    transpiled = transpile(circuit, basis_gates=clifford_t_basis, optimization_level=3)

    with open(path, "w") as f:
        f.write(dumps(transpiled))
    print(f"✅ QASM written to: {path}")
    return path