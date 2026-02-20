"""Utilities to build quantum circuits from quantum MLIR and emit QASM."""
from __future__ import annotations

import os
from typing import Dict, Optional

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


def generate_circuit(
    module: ModuleOp,
    num_bits: int = 16,
    verbose: bool = False,
    arithmetic_mode: str = "qft",  # "qft" | "ripple"
) -> QuantumCircuit:
    """Convert ``module`` using the quantum dialect to a ``QuantumCircuit``."""

    # ------------------------------------------------------------
    # Selezione backend aritmetico (QFT vs Ripple Carry)
    # ------------------------------------------------------------
    # Default: "qft" (comportamento pre-esistente).
    if hasattr(qa, "set_arithmetic_mode"):
        qa.set_arithmetic_mode(arithmetic_mode)

    # Propagazione anche al modulo controlled (per evitare confronto “misto”)
    if hasattr(qac, "set_arithmetic_mode"):
        try:
            qac.set_arithmetic_mode(arithmetic_mode)
        except Exception:
            pass

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
        if hasattr(qac.qa, "set_arithmetic_mode") and hasattr(qa, "set_arithmetic_mode"):
            try:
                qac.qa.set_arithmetic_mode(arithmetic_mode)
            except Exception:
                pass

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
import numpy as np

# =============================================================================
# Clifford+T Decomposition Constants
# =============================================================================
# Numero di T-gates per approssimare una rotazione arbitraria (Solovay-Kitaev/gridsynth)
# con precisione epsilon ~= 10^-15. Formula: T-count ≈ 3 * log2(1/epsilon)
# Per epsilon = 10^-15: log2(10^15) ≈ 50, quindi 3*50 = 150
# Riferimento: Ross-Selinger 2014, "Optimal ancilla-free Clifford+T approximation"
T_GATES_PER_ARBITRARY_ROTATION = 150

# T-gates per decomporre un Toffoli (CCX) - decomposizione esatta standard
# Riferimento: Nielsen & Chuang, decomposizione con 7 T-gates
T_GATES_PER_TOFFOLI = 7

# Angoli che sono multipli esatti di pi/4 (nativi in Clifford+T)
# T = Rz(pi/4), S = Rz(pi/2), Z = Rz(pi)
CLIFFORD_T_ANGLES = {
    0: 0,                    # Identity
    np.pi / 4: 1,            # T gate (1 T)
    np.pi / 2: 0,            # S gate (Clifford)
    3 * np.pi / 4: 1,        # T + S (1 T)
    np.pi: 0,                # Z gate (Clifford)
    5 * np.pi / 4: 1,        # Tdg + S (1 T)
    3 * np.pi / 2: 0,        # Sdg (Clifford)
    7 * np.pi / 4: 1,        # Tdg (1 T)
    -np.pi / 4: 1,           # Tdg
    -np.pi / 2: 0,           # Sdg
    -3 * np.pi / 4: 1,       # Tdg + Sdg
    -np.pi: 0,               # Z
}


def _is_clifford_angle(angle: float, tolerance: float = 1e-10) -> bool:
    """Check if angle is a multiple of pi/2 (Clifford gate)."""
    # Normalizza l'angolo in [0, 2*pi)
    normalized = angle % (2 * np.pi)
    # Controlla se è multiplo di pi/2
    remainder = normalized % (np.pi / 2)
    return remainder < tolerance or (np.pi / 2 - remainder) < tolerance


def _is_t_angle(angle: float, tolerance: float = 1e-10) -> bool:
    """Check if angle is a multiple of pi/4 (T or Clifford gate)."""
    normalized = angle % (2 * np.pi)
    remainder = normalized % (np.pi / 4)
    return remainder < tolerance or (np.pi / 4 - remainder) < tolerance


def _t_count_for_angle(angle: float, tolerance: float = 1e-10) -> int:
    """
    Calcola il T-count per approssimare Rz(angle).

    - Se angle è multiplo di pi/2 → 0 T (è Clifford: S, Z, I)
    - Se angle è multiplo di pi/4 → 1 T (è T o Tdg)
    - Altrimenti → T_GATES_PER_ARBITRARY_ROTATION (approssimazione)
    """
    if _is_clifford_angle(angle, tolerance):
        return 0
    if _is_t_angle(angle, tolerance):
        return 1
    return T_GATES_PER_ARBITRARY_ROTATION


def compute_t_count(circuit: QuantumCircuit) -> dict:
    """
    Calcola il T-count totale per un circuito, considerando:
    - T/Tdg gates: 1 T ciascuno
    - CCX (Toffoli): 7 T ciascuno
    - Rz/P/CP con angoli arbitrari: ~150 T ciascuno (approssimazione Solovay-Kitaev)
    - Rz/P/CP con angoli multipli di pi/4: 0-1 T (esatto)

    Returns:
        dict con:
        - t_count: numero totale di T-gates
        - t_count_exact: T-gates da decomposizioni esatte (T, Tdg, Toffoli)
        - t_count_approx: T-gates da approssimazioni (rotazioni arbitrarie)
        - arbitrary_rotations: numero di rotazioni arbitrarie
    """
    t_count_exact = 0
    t_count_approx = 0
    arbitrary_rotations = 0

    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()

        # T e Tdg: esattamente 1 T
        if gate_name in ('t', 'tdg'):
            t_count_exact += 1

        # Toffoli (CCX): esattamente 7 T
        elif gate_name == 'ccx':
            t_count_exact += T_GATES_PER_TOFFOLI

        # MCX (multi-controlled X): approssimazione basata su numero di controlli
        elif gate_name == 'mcx':
            num_controls = len(instruction.qubits) - 1
            # Decomposizione in Toffoli: O(n) Toffoli per n controlli
            t_count_exact += (2 * num_controls - 3) * T_GATES_PER_TOFFOLI

        # Gate con parametri (rotazioni)
        elif gate_name in ('rz', 'p', 'rx', 'ry', 'u1', 'u', 'u3'):
            if hasattr(instruction.operation, 'params') and instruction.operation.params:
                # Prendi il primo parametro (l'angolo principale)
                angle = float(instruction.operation.params[0])
                t_for_angle = _t_count_for_angle(angle)
                if t_for_angle == T_GATES_PER_ARBITRARY_ROTATION:
                    t_count_approx += t_for_angle
                    arbitrary_rotations += 1
                else:
                    t_count_exact += t_for_angle

        # Controlled-phase gates
        elif gate_name in ('cp', 'crz', 'cu1', 'cphase'):
            if hasattr(instruction.operation, 'params') and instruction.operation.params:
                angle = float(instruction.operation.params[0])
                t_for_angle = _t_count_for_angle(angle)
                if t_for_angle == T_GATES_PER_ARBITRARY_ROTATION:
                    t_count_approx += t_for_angle
                    arbitrary_rotations += 1
                else:
                    # Controlled version richiede ~2x per decomposizione
                    t_count_exact += t_for_angle * 2

        # CCPhase (doubly-controlled phase): ancora più costoso
        elif gate_name in ('ccphase', 'ccp'):
            if hasattr(instruction.operation, 'params') and instruction.operation.params:
                angle = float(instruction.operation.params[0])
                t_for_angle = _t_count_for_angle(angle)
                if t_for_angle == T_GATES_PER_ARBITRARY_ROTATION:
                    # CCPhase con angolo arbitrario
                    t_count_approx += t_for_angle * 2  # ~2x overhead per doppio controllo
                    arbitrary_rotations += 1
                else:
                    t_count_exact += t_for_angle * 4

    return {
        't_count': t_count_exact + t_count_approx,
        't_count_exact': t_count_exact,
        't_count_approx': t_count_approx,
        'arbitrary_rotations': arbitrary_rotations,
    }


def export_qasm_clifford_t(circuit: QuantumCircuit, path: str) -> str:
    """
    Export the circuit to QASM, decomposed to Clifford+T gate set.

    Gate set finale: {H, S, Sdg, T, Tdg, X, Y, Z, CX}

    Nota: le rotazioni arbitrarie vengono lasciate come RZ nel QASM per simulazione,
    ma il T-count viene calcolato separatamente assumendo decomposizione Solovay-Kitaev.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Prima trasportiamo verso un gate set che includa T
    # Nota: Qiskit non fa automaticamente Solovay-Kitaev, quindi usiamo
    # un gate set intermedio e calcoliamo il T-count separatamente
    clifford_t_basis = ["h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx", "rz", "measure"]
    transpiled = transpile(circuit, basis_gates=clifford_t_basis, optimization_level=3)

    with open(path, "w") as f:
        f.write(dumps(transpiled))
    print(f"✅ QASM (Clifford+T basis) written to: {path}")
    return path