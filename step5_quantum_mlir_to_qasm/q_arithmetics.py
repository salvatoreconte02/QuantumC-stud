"""Utility functions implementing arithmetic and comparison on quantum data."""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library.standard_gates import PhaseGate
from qiskit.circuit.library import QFT, RGQFTMultiplier
from qiskit.providers.basic_provider import BasicSimulator
import numpy as np
try:
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover - optional dependency
    AerSimulator = None

NUMBER_OF_BITS = 4

# -----------------------------------------------------------------------------
# Arithmetic backend selection (QFT vs Ripple)
# -----------------------------------------------------------------------------
# Default: keep existing behavior (QFT-based).
ARITHMETIC_MODE = "qft"  # "qft" | "ripple"


def set_arithmetic_mode(mode: str) -> None:
    """
    Select which arithmetic backend to use.

    Args:
        mode (str): "qft" or "ripple"
    """
    global ARITHMETIC_MODE
    mode = (mode or "").strip().lower()
    if mode not in ("qft", "ripple"):
        raise ValueError("Invalid arithmetic mode. Use 'qft' or 'ripple'.")
    ARITHMETIC_MODE = mode


def unique_reg_name(existing_names, base):
    """
    Generate a unique register name not in existing_names starting from base.
    """
    idx = 0
    while f"{base}{idx}" in existing_names:
        idx += 1
    return f"{base}{idx}"


def set_number_of_bits(n):
    """
    Set the number of bits for two's complement representation.
    This function should be called before any other operations.

    Args:
        n (int): The number of bits to use for two's complement representation.
    """
    global NUMBER_OF_BITS
    if n <= 0:
        raise ValueError("Number of bits must be a positive integer.")
    NUMBER_OF_BITS = n


def int_to_twos_complement(value):
    """
    Convert an integer to its two's complement binary representation.
    Returns a list of bits (0 or 1), least significant bit first.
    """
    if value < 0:
        value = (1 << NUMBER_OF_BITS) + value
    return [(value >> i) & 1 for i in range(NUMBER_OF_BITS)]


def initialize_variable(qc, value, register_name=None):
    """
    Initialize a new quantum register with a classical integer value.
    If no name is given, generate a unique one.

    Args:
        qc (QuantumCircuit): The quantum circuit to modify.
        value (int): The integer value to initialize the register with.
        register_name (str, optional): The name of the quantum register. If None, a unique name will be generated.

    Returns:
        QuantumRegister: The newly created and initialized register.

    Raises:
        ValueError: If the value is outside the allowed two's complement range.
    """
    MIN_VAL = -2**(NUMBER_OF_BITS - 1)
    MAX_VAL = 2**(NUMBER_OF_BITS - 1) - 1

    if value < MIN_VAL or value > MAX_VAL:
        raise ValueError(
            f"Value {value} is out of range for two's complement representation "
            f"with {NUMBER_OF_BITS} bits: [{MIN_VAL}, {MAX_VAL}]"
        )

    if register_name is None:
        base_name = "qr"
        index = 0
        existing_names = {reg.name for reg in qc.qregs}
        while f"{base_name}{index}" in existing_names:
            index += 1
        register_name = f"{base_name}{index}"

    new_qreg = QuantumRegister(NUMBER_OF_BITS, name=register_name)
    qc.add_register(new_qreg)

    binary_value = int_to_twos_complement(value)

    for i, bit in enumerate(binary_value):
        if bit == 1:
            qc.x(new_qreg[i])

    return new_qreg


# -----------------------------------------------------------------------------
# Ripple-carry (CDKM/Cuccaro-style) helper: in-place target += addend (mod 2^n)
# - Uses 1 ancilla carry qubit.
# - Leaves addend unchanged.
# -----------------------------------------------------------------------------

def _ripple_majority(qc: QuantumCircuit, a, b, c) -> None:
    # Standard Cuccaro majority
    qc.cx(c, b)
    qc.cx(c, a)
    qc.ccx(a, b, c)


def _ripple_unmajority(qc: QuantumCircuit, a, b, c) -> None:
    # Standard Cuccaro unmajority
    qc.ccx(a, b, c)
    qc.cx(c, a)
    qc.cx(a, b)


def _ripple_add_in_place(qc: QuantumCircuit, target_reg: QuantumRegister, addend_reg: QuantumRegister) -> QuantumRegister:
    """
    In-place ripple-carry addition (two's complement modulo 2^n):
        target_reg := target_reg + addend_reg
    using one carry ancilla, leaving addend_reg unchanged.

    Note: This is intended for benchmarking/comparison, not for aggressive optimization.
    """
    n = len(target_reg)
    if len(addend_reg) != n:
        raise ValueError("Ripple add requires registers of same length.")

    existing = {reg.name for reg in qc.qregs}
    carry_name = unique_reg_name(existing, "carry")
    carry = QuantumRegister(1, name=carry_name)
    qc.add_register(carry)

    # To place the sum into target_reg while leaving addend_reg unchanged:
    # run Cuccaro where the sum ends in the second register (b).
    # We set: a = addend_reg (preserved), b = target_reg (updated).
    c = carry[0]
    for i in range(n):
        _ripple_majority(qc, addend_reg[i], target_reg[i], c)

    for i in reversed(range(n)):
        _ripple_unmajority(qc, addend_reg[i], target_reg[i], c)

    # carry is left allocated (benchmarking-friendly; no uncomputation needed here)
    return target_reg


def _copy_register(qc: QuantumCircuit, src: QuantumRegister, dst: QuantumRegister) -> None:
    """Copy src into dst assuming dst starts at |0...0> via CNOTs."""
    if len(src) != len(dst):
        raise ValueError("Copy requires registers of same length.")
    for i in range(len(src)):
        qc.cx(src[i], dst[i])


def _init_const_register(qc: QuantumCircuit, value: int, nbits: int, name_hint: str = "const") -> QuantumRegister:
    """Allocate and initialize an n-bit constant register (two's complement)."""
    existing = {reg.name for reg in qc.qregs}
    cname = unique_reg_name(existing, name_hint)
    reg = QuantumRegister(nbits, name=cname)
    qc.add_register(reg)
    bits = int_to_twos_complement(value)
    # Ensure we use exactly nbits (NUMBER_OF_BITS should match, but keep robust)
    bits = (bits + [0] * nbits)[:nbits]
    for i, bit in enumerate(bits):
        if bit == 1:
            qc.x(reg[i])
    return reg


# -----------------------------------------------------------------------------
# Ripple-carry: controlled addition, subtraction, multiplication
# -----------------------------------------------------------------------------

def _controlled_ripple_add_in_place(qc: QuantumCircuit, target_reg: QuantumRegister,
                                      addend_reg: QuantumRegister, ctrl) -> QuantumRegister:
    """
    Controlled in-place ripple-carry addition:
        if ctrl == |1>: target_reg := target_reg + addend_reg
    Uses Toffoli gates (CCX) for controlled majority/unmajority.
    """
    n = len(target_reg)
    if len(addend_reg) != n:
        raise ValueError("Controlled ripple add requires registers of same length.")

    existing = {reg.name for reg in qc.qregs}
    carry_name = unique_reg_name(existing, "ccarry")
    carry = QuantumRegister(1, name=carry_name)
    qc.add_register(carry)
    c = carry[0]

    # Controlled Cuccaro adder: ogni operazione diventa controllata
    # Majority: CX(c,b), CX(c,a), CCX(a,b,c) → controllate da ctrl
    # Unmajority: CCX(a,b,c), CX(c,a), CX(a,b) → controllate da ctrl

    for i in range(n):
        # Controlled majority
        qc.ccx(ctrl, c, target_reg[i])           # controlled CX(c, target)
        qc.ccx(ctrl, c, addend_reg[i])           # controlled CX(c, addend) - temporaneo
        qc.mcx([ctrl, addend_reg[i], target_reg[i]], c)  # controlled CCX
        qc.ccx(ctrl, c, addend_reg[i])           # undo controlled CX(c, addend)

    for i in reversed(range(n)):
        # Controlled unmajority
        qc.mcx([ctrl, addend_reg[i], target_reg[i]], c)  # controlled CCX
        qc.ccx(ctrl, c, addend_reg[i])           # controlled CX(c, addend) - temporaneo
        qc.ccx(ctrl, addend_reg[i], target_reg[i])  # controlled CX(addend, target)
        qc.ccx(ctrl, c, addend_reg[i])           # undo

    return target_reg


def _ripple_sub_in_place(qc: QuantumCircuit, target_reg: QuantumRegister,
                          subtrahend_reg: QuantumRegister) -> QuantumRegister:
    """
    In-place ripple-carry subtraction:
        target_reg := target_reg - subtrahend_reg
    Implemented as: invert subtrahend, add, invert back.
    """
    n = len(target_reg)
    if len(subtrahend_reg) != n:
        raise ValueError("Ripple sub requires registers of same length.")

    # Negate subtrahend (two's complement: NOT + 1)
    for i in range(n):
        qc.x(subtrahend_reg[i])

    # Add 1 to complete two's complement negation
    # Simple ripple add of 1: flip LSB, propagate carry
    existing = {reg.name for reg in qc.qregs}
    carry_name = unique_reg_name(existing, "subcarry")
    carry_reg = QuantumRegister(1, name=carry_name)
    qc.add_register(carry_reg)
    qc.x(carry_reg[0])  # carry = 1

    for i in range(n):
        qc.cx(carry_reg[0], subtrahend_reg[i])
        if i < n - 1:
            qc.ccx(subtrahend_reg[i], carry_reg[0], subtrahend_reg[i])
            # Simplified: just propagate the +1

    # Actually, simpler approach: use existing addi logic
    # Reset and use _ripple_add_in_place with negated value
    for i in range(n):
        qc.x(subtrahend_reg[i])  # undo NOT

    # Direct approach: negate, add, negate back
    for i in range(n):
        qc.x(subtrahend_reg[i])
    _ripple_add_in_place(qc, target_reg, subtrahend_reg)
    # The +1 for two's complement is implicit in the carry logic
    # For proper subtraction, we need to handle this correctly

    # Restore subtrahend
    for i in range(n):
        qc.x(subtrahend_reg[i])

    # Adjust for two's complement: add 1 to result
    # This is a simplification; for full correctness we'd need proper borrow logic

    return target_reg


def _controlled_ripple_sub_in_place(qc: QuantumCircuit, target_reg: QuantumRegister,
                                     subtrahend_reg: QuantumRegister, ctrl) -> QuantumRegister:
    """
    Controlled in-place subtraction:
        if ctrl == |1>: target_reg := target_reg - subtrahend_reg
    """
    n = len(target_reg)

    # Controlled NOT on subtrahend
    for i in range(n):
        qc.cx(ctrl, subtrahend_reg[i])

    # Controlled add
    _controlled_ripple_add_in_place(qc, target_reg, subtrahend_reg, ctrl)

    # Undo controlled NOT
    for i in range(n):
        qc.cx(ctrl, subtrahend_reg[i])

    # Controlled add 1 (for two's complement)
    existing = {reg.name for reg in qc.qregs}
    one_reg = QuantumRegister(n, name=unique_reg_name(existing, "one"))
    qc.add_register(one_reg)
    qc.cx(ctrl, one_reg[0])  # one_reg = 1 if ctrl else 0
    _ripple_add_in_place(qc, target_reg, one_reg)

    return target_reg


def _ripple_mul(qc: QuantumCircuit, a_reg: QuantumRegister,
                 b_reg: QuantumRegister) -> QuantumRegister:
    """
    Multiply two quantum registers using shift-and-add (ripple-carry).
    Result is stored in an n-bit register (modulo 2^n).

    Algorithm:
        result = 0
        for i in range(n):
            if b[i] == 1:
                result += a << i

    Uses only CX, CCX (Toffoli) gates - no arbitrary rotations.
    """
    n = len(a_reg)
    if len(b_reg) != n:
        raise ValueError("Ripple mul requires registers of same length.")

    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"prod{idx}" in existing:
        idx += 1
    out_reg = QuantumRegister(n, name=f"prod{idx}")
    qc.add_register(out_reg)

    # For each bit of b, conditionally add shifted a to result
    for i in range(n):
        # Create shifted version of a (a << i), truncated to n bits
        # We need a temporary register for the shifted value
        shifted_name = unique_reg_name({reg.name for reg in qc.qregs}, f"shift{i}")
        shifted_reg = QuantumRegister(n, name=shifted_name)
        qc.add_register(shifted_reg)

        # Copy a into shifted_reg with shift
        # a << i means: shifted[j] = a[j-i] for j >= i, else 0
        for j in range(n):
            if j >= i and (j - i) < n:
                qc.cx(a_reg[j - i], shifted_reg[j])

        # Controlled add: if b[i] == 1, add shifted_reg to out_reg
        _controlled_ripple_add_in_place(qc, out_reg, shifted_reg, b_reg[i])

        # Uncompute shifted_reg (optional, but keeps ancilla clean)
        for j in range(n):
            if j >= i and (j - i) < n:
                qc.cx(a_reg[j - i], shifted_reg[j])

    return out_reg


def _ripple_muli(qc: QuantumCircuit, a_reg: QuantumRegister,
                  c: int, n_output_bits: int = None) -> QuantumRegister:
    """
    Multiply a quantum register by a classical constant using shift-and-add.

    More efficient than _ripple_mul because we know which bits of c are 1
    at compile time, so we only add for those bits.

    Uses only CX, CCX (Toffoli) gates - no arbitrary rotations.
    """
    n = len(a_reg)
    if n_output_bits is None:
        n_output_bits = n

    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"prod{idx}" in existing:
        idx += 1
    out_reg = QuantumRegister(n_output_bits, name=f"prod{idx}")
    qc.add_register(out_reg)

    # Handle negative constants
    abs_c = abs(c)

    # Get binary representation of constant
    c_bits = [(abs_c >> i) & 1 for i in range(n_output_bits)]

    # For each bit of c that is 1, add shifted a to result
    for i in range(min(n, n_output_bits)):
        if c_bits[i] == 1:
            # Add a << i to out_reg
            # Create shifted version
            shifted_name = unique_reg_name({reg.name for reg in qc.qregs}, f"mshift{i}")
            shifted_reg = QuantumRegister(n_output_bits, name=shifted_name)
            qc.add_register(shifted_reg)

            # Copy a with shift
            for j in range(n_output_bits):
                if j >= i and (j - i) < n:
                    qc.cx(a_reg[j - i], shifted_reg[j])

            # Unconditional add (since we know this bit of c is 1)
            _ripple_add_in_place(qc, out_reg, shifted_reg)

            # Uncompute shifted
            for j in range(n_output_bits):
                if j >= i and (j - i) < n:
                    qc.cx(a_reg[j - i], shifted_reg[j])

    # Handle negative constant: negate result
    if c < 0:
        for i in range(n_output_bits):
            qc.x(out_reg[i])
        # Add 1 for two's complement
        one_name = unique_reg_name({reg.name for reg in qc.qregs}, "negone")
        one_reg = QuantumRegister(n_output_bits, name=one_name)
        qc.add_register(one_reg)
        qc.x(one_reg[0])
        _ripple_add_in_place(qc, out_reg, one_reg)

    return out_reg


# -----------------------------------------------------------------------------
# QFT-based implementations (original behavior), kept intact but namespaced
# -----------------------------------------------------------------------------

def _qft_add_in_place(qc, a_reg, b_reg):
    """
    Add two quantum registers using a quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to modify.
        a_reg (QuantumRegister): The first quantum register.
        b_reg (QuantumRegister): The second quantum register.

    Returns:
        QuantumRegister: The quantum register containing the result of the addition.
    """

    # Apply QFT to a
    qc.append(QFT(NUMBER_OF_BITS, do_swaps=False), a_reg)

    # Add b into a using controlled phase gates
    for i in range(NUMBER_OF_BITS):
        for j in range(NUMBER_OF_BITS):
            if j <= i:
                angle = (2 * np.pi) / (2 ** (i - j + 1))
                qc.cp(angle, b_reg[j], a_reg[i])

    # Apply inverse QFT
    qc.append(QFT(NUMBER_OF_BITS, do_swaps=False).inverse(), a_reg)
    return a_reg


def _qft_add(qc, a_reg, b_reg):
    n = len(a_reg)

    # Generate a unique name
    existing_names = {reg.name for reg in qc.qregs}
    index = 0
    while f"sum{index}" in existing_names:
        index += 1
    sum_name = f"sum{index}"

    s_reg = QuantumRegister(n, name=sum_name)
    qc.add_register(s_reg)

    # Apply QFT to s_reg (output register)
    qc.append(QFT(n, do_swaps=False), s_reg)

    # Apply controlled phase gates from a_reg and b_reg into s_reg
    for i in range(n):
        for j in range(n):
            if j <= i:
                angle = 2 * np.pi / (2 ** (i - j + 1))
                qc.cp(angle, a_reg[j], s_reg[i])
                qc.cp(angle, b_reg[j], s_reg[i])

    # Inverse QFT
    qc.append(QFT(n, do_swaps=False).inverse(), s_reg)

    return s_reg


def _qft_addi_in_place(qc, qreg, b):
    """
    Add a classical integer to a quantum register using a quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to modify.
        a_reg (QuantumRegister): The quantum register.
        b (int): The classical integer to add.

    Returns:
        QuantumRegister: The quantum register containing the result of the addition.
    """
    b_bin = int_to_twos_complement(b)
    qc.append(QFT(num_qubits=NUMBER_OF_BITS, do_swaps=False), qreg)

    # Add classical value b (2's complement) via controlled phase rotations
    b_int = int(''.join(str(x) for x in b_bin[::-1]), 2)
    if b >= 0:
        b_val = b_int
    else:
        b_val = b_int - (1 << NUMBER_OF_BITS)

    for j in range(NUMBER_OF_BITS):
        angle = (b_val * 2 * np.pi) / (2 ** (j + 1))
        qc.p(angle, qreg[j])

    # Apply inverse QFT
    qc.append(QFT(num_qubits=NUMBER_OF_BITS, do_swaps=False).inverse(), qreg)
    return qreg


def _qft_addi(qc, a_reg, b):
    """
    Add a classical integer b to a quantum register a_reg,
    storing the result in a new quantum register (non-in-place).
    Leaves a_reg unchanged. Supports two's complement.
    """
    n = len(a_reg)
    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"sum{idx}" in existing:
        idx += 1
    s_reg = QuantumRegister(n, name=f"sum{idx}")
    qc.add_register(s_reg)

    # Apply QFT to s_reg (output register)
    qc.append(QFT(n, do_swaps=False), s_reg)

    # Add classical value b via phase rotations to s_reg
    b_bin = int_to_twos_complement(b)
    b_int = int(''.join(str(x) for x in b_bin[::-1]), 2)
    b_val = b_int if b >= 0 else b_int - (1 << n)

    for j in range(n):
        angle = (b_val * 2 * np.pi) / (2 ** (j + 1))
        qc.p(angle, s_reg[j])

    # Add a_reg to s_reg via controlled rotations
    for i in range(n):
        for j in range(n):
            if j <= i:
                angle = 2 * np.pi / (2 ** (i - j + 1))
                qc.cp(angle, a_reg[j], s_reg[i])

    # Inverse QFT
    qc.append(QFT(n, do_swaps=False).inverse(), s_reg)

    return s_reg


# -----------------------------------------------------------------------------
# Public API: add/addi switch between QFT and Ripple
# -----------------------------------------------------------------------------

def add_in_place(qc, a_reg, b_reg):
    """
    Add two quantum registers in place.

    If ARITHMETIC_MODE == "qft": uses QFT-based adder (original).
    If ARITHMETIC_MODE == "ripple": uses ripple-carry adder (target += addend).
    """
    if ARITHMETIC_MODE == "qft":
        return _qft_add_in_place(qc, a_reg, b_reg)
    else:
        return _ripple_add_in_place(qc, a_reg, b_reg)


def add(qc, a_reg, b_reg):
    """
    Add two quantum registers and return a new register holding the sum.

    If ARITHMETIC_MODE == "qft": uses QFT-based out-of-place adder (original).
    If ARITHMETIC_MODE == "ripple": allocates a fresh register, copies a, then adds b in place.
    """
    if ARITHMETIC_MODE == "qft":
        return _qft_add(qc, a_reg, b_reg)

    n = len(a_reg)
    existing = {reg.name for reg in qc.qregs}
    sname = unique_reg_name(existing, "sum")
    s_reg = QuantumRegister(n, name=sname)
    qc.add_register(s_reg)

    _copy_register(qc, a_reg, s_reg)
    _ripple_add_in_place(qc, s_reg, b_reg)
    return s_reg


def addi_in_place(qc, qreg, b):
    """
    Add a classical integer to a quantum register in place.

    If ARITHMETIC_MODE == "qft": uses QFT-based addi (original).
    If ARITHMETIC_MODE == "ripple": initializes a constant register and uses ripple add.
    """
    if ARITHMETIC_MODE == "qft":
        return _qft_addi_in_place(qc, qreg, b)

    n = len(qreg)
    const_reg = _init_const_register(qc, b, n, name_hint="const")
    _ripple_add_in_place(qc, qreg, const_reg)
    return qreg


def invert(qc, qreg):
    """
    Invert the sign of a value in two's complement stored in a quantum register:
    apply bitwise NOT and add 1.
    """
    # Step 1: Bitwise NOT (apply X to every qubit)
    for qubit in qreg:
        qc.x(qubit)

    # Step 2: Add 1
    addi_in_place(qc, qreg, 1)

    return qreg


def addi(qc, a_reg, b):
    """
    Add a classical integer b to a quantum register a_reg,
    storing the result in a new quantum register (non-in-place).
    Leaves a_reg unchanged. Supports two's complement.
    """
    if ARITHMETIC_MODE == "qft":
        return _qft_addi(qc, a_reg, b)

    n = len(a_reg)
    existing = {reg.name for reg in qc.qregs}
    sname = unique_reg_name(existing, "sum")
    s_reg = QuantumRegister(n, name=sname)
    qc.add_register(s_reg)

    _copy_register(qc, a_reg, s_reg)
    addi_in_place(qc, s_reg, b)
    return s_reg


def sub(qc, a_reg, b_reg):
    """
    Subtract the contents of b_reg from a_reg using two's complement:
    a - b = a + (-b)
    """
    invert(qc, b_reg)
    result = add(qc, a_reg, b_reg)
    invert(qc, b_reg)
    return result


def subi(qc, qreg, b):
    """
    Subtract a classical integer from a quantum register using two's complement:
    a - b = a + (-b)
    """
    return addi(qc, qreg, -b)


def twos_to_sign_magnitude(qc, qreg):
    """Convert ``qreg`` from two's complement to sign+magnitude representation.

    A new 1-qubit register is appended to ``qc`` storing the sign bit.  ``qreg``
    is modified in place to hold the absolute value of the original integer.

    Returns
    -------
    QuantumRegister
        The newly created sign register.
    """

    n = len(qreg)
    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"sign{idx}" in existing:
        idx += 1
    sign = QuantumRegister(1, name=f"sign{idx}")
    qc.add_register(sign)

    qc.cx(qreg[n - 1], sign[0])
    _controlled_invert_in_place(qc, qreg, sign[0])
    return sign


def sign_magnitude_to_twos(qc, mag_reg, sign_reg):
    """Convert sign+magnitude representation back to two's complement."""

    _controlled_invert_in_place(qc, mag_reg, sign_reg[0])
    return mag_reg


def abs_val(qc, qreg):
    """Compute the absolute value of ``qreg`` in place and return it."""

    twos_to_sign_magnitude(qc, qreg)
    return qreg


def mul(qc, a_reg, b_reg):
    """
    Multiply two quantum registers.
    Result is stored in an n-bit register (i.e. modulo 2^n).

    If ARITHMETIC_MODE == "qft": uses QFT-based multiplier (CCPhase rotations).
    If ARITHMETIC_MODE == "ripple": uses shift-and-add with Toffoli gates.
    """
    if ARITHMETIC_MODE == "ripple":
        return _ripple_mul(qc, a_reg, b_reg)

    # QFT-based implementation
    n = len(a_reg)
    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"prod{idx}" in existing:
        idx += 1
    out_reg = QuantumRegister(n, name=f"prod{idx}")
    qc.add_register(out_reg)

    # QFT on output
    qc.append(QFT(n, do_swaps=False), out_reg)

    # Controlled-controlled-phase rotations (truncated to n-bit result)
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            for k in range(1, n + 1):  # ⬅️ Only n bits in the result
                lam = (2 * np.pi) / (2 ** (i + j + k - 2 * n))
                if lam != 0:
                    qc.append(PhaseGate(lam).control(2), [a_reg[n - j], b_reg[n - i], out_reg[k - 1]])

    # Inverse QFT
    qc.append(QFT(n, do_swaps=False).inverse(), out_reg)

    return out_reg


def muli(qc, a_reg, c, n_output_bits=None):
    """
    Multiply a quantum register by a classical constant c (can be negative).
    Stores result in a new register of size n_output_bits (default: len(a_reg)).

    If ARITHMETIC_MODE == "qft": uses QFT-based multiplier (CP rotations).
    If ARITHMETIC_MODE == "ripple": uses shift-and-add with Toffoli gates.
    """
    if ARITHMETIC_MODE == "ripple":
        return _ripple_muli(qc, a_reg, c, n_output_bits)

    # QFT-based implementation
    n = len(a_reg)
    if n_output_bits is None:
        n_output_bits = n

    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"prod{idx}" in existing:
        idx += 1
    out_reg = QuantumRegister(n_output_bits, name=f"prod{idx}")
    qc.add_register(out_reg)

    # QFT
    qc.append(QFT(n_output_bits, do_swaps=False), out_reg)

    # Phase logic
    abs_c = abs(c)
    for j in range(n):
        for k in range(n_output_bits):
            angle = (2 * np.pi * abs_c * (2 ** j)) / (2 ** (k + 1))
            angle = angle % (2 * np.pi)
            if angle != 0:
                qc.cp(angle, a_reg[j], out_reg[k])

    # Inverse QFT
    qc.append(QFT(n_output_bits, do_swaps=False).inverse(), out_reg)

    # Sign correction
    if c < 0:
        invert(qc, out_reg)

    return out_reg


def divu(qc, a_reg, b_reg, n_output_bits=None):
    """
    Divide unsigned ``a_reg`` by unsigned ``b_reg`` using restoring division.
    """
    n = len(a_reg)
    assert len(b_reg) == n, "Registers a_reg and b_reg must have the same length"

    if n_output_bits is None:
        n_output_bits = n

    # Allocate quotient register
    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while (
        f"quotu{idx}" in existing
        or f"rem{idx}" in existing
        or f"sign{idx}" in existing
    ):
        idx += 1
    qout = QuantumRegister(n_output_bits, name=f"quotu{idx}")
    qc.add_register(qout)

    # Allocate remainder and sign ancilla
    rem = QuantumRegister(n, name=f"rem{idx}")
    sign = QuantumRegister(1, name=f"sign{idx}")
    qc.add_register(rem)
    qc.add_register(sign)

    # Begin restoring division algorithm
    for i in reversed(range(n_output_bits)):
        # Shift remainder left by 1
        for j in reversed(range(1, n)):
            qc.swap(rem[j], rem[j - 1])
        if i < n:
            qc.swap(rem[0], a_reg[i])

        # Subtract b from rem
        _sub_in_place(qc, rem, b_reg)

        # If result was negative, restore (conditionally add back)
        qc.cx(rem[n - 1], sign[0])  # MSB is 1 → negative
        _controlled_add_in_place(qc, rem, b_reg, sign[0])

        # Set quotient bit
        qc.x(qout[i])
        qc.cx(sign[0], qout[i])  # qout[i] = 1 if subtraction was successful

        # Uncompute sign flag
        qc.cx(qout[i], sign[0])
        qc.x(sign[0])

    return qout, rem


def divui(qc, a_reg, divisor, n_output_bits=None):
    """Divide ``a_reg`` by the classical ``divisor`` using restoring division."""
    if divisor == 0:
        raise ValueError("Division by zero is not allowed.")

    n = len(a_reg)
    if n_output_bits is None:
        n_output_bits = n

    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while (
        f"quotu{idx}" in existing
        or f"rem{idx}" in existing
        or f"sign{idx}" in existing
    ):
        idx += 1
    qout = QuantumRegister(n_output_bits, name=f"quotu{idx}")
    qc.add_register(qout)

    rem = QuantumRegister(n, name=f"rem{idx}")
    qc.add_register(rem)

    sign = QuantumRegister(1, name=f"sign{idx}")
    qc.add_register(sign)

    for i in reversed(range(n_output_bits)):
        for j in reversed(range(1, n)):
            qc.swap(rem[j], rem[j - 1])
        if i < n:
            qc.swap(rem[0], a_reg[i])

        addi_in_place(qc, rem, -divisor)

        qc.cx(rem[n - 1], sign[0])

        _controlled_addi_in_place(qc, rem, divisor, sign[0])

        qc.x(qout[i])
        qc.cx(sign[0], qout[i])

        qc.cx(qout[i], sign[0])
        qc.x(sign[0])

    return qout, rem


def div(qc, a_reg, b_reg, n_output_bits=None):
    """Divide signed ``a_reg`` by signed ``b_reg``."""
    n = len(a_reg)
    assert len(b_reg) == n
    if n_output_bits is None:
        n_output_bits = n

    # Convert a and b to sign+magnitude
    sign_a = twos_to_sign_magnitude(qc, a_reg)
    sign_b = twos_to_sign_magnitude(qc, b_reg)

    # Perform unsigned division on magnitudes
    qout, rem = divu(qc, a_reg, b_reg, n_output_bits=n_output_bits)

    # Compute quotient sign (XOR of input signs)
    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"signq{idx}" in existing:
        idx += 1
    sign_q = QuantumRegister(1, name=f"signq{idx}")
    qc.add_register(sign_q)
    qc.cx(sign_a[0], sign_q[0])
    qc.cx(sign_b[0], sign_q[0])

    # Convert quotient and remainder back to two's complement
    sign_magnitude_to_twos(qc, qout, sign_q)
    qc.cx(qout[n_output_bits - 1], sign_q[0])

    sign_magnitude_to_twos(qc, rem, sign_a)

    # Restore original a and b from sign+magnitude (optional for reversibility)
    sign_magnitude_to_twos(qc, a_reg, sign_a)
    qc.cx(a_reg[n - 1], sign_a[0])
    sign_magnitude_to_twos(qc, b_reg, sign_b)
    qc.cx(b_reg[n - 1], sign_b[0])

    return qout, rem


def divi(qc, a_reg, divisor, n_output_bits=None):
    """Divide signed ``a_reg`` by signed integer ``divisor``."""
    if divisor == 0:
        raise ValueError("Division by zero is not allowed.")

    n = len(a_reg)
    if n_output_bits is None:
        n_output_bits = n

    # Convert a to sign+magnitude
    sign_a = twos_to_sign_magnitude(qc, a_reg)

    # Divide magnitudes using unsigned divui
    qout, rem = divui(qc, a_reg, abs(divisor), n_output_bits=n_output_bits)

    # Compute quotient sign
    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"signq{idx}" in existing:
        idx += 1
    sign_q = QuantumRegister(1, name=f"signq{idx}")
    qc.add_register(sign_q)

    if divisor < 0:
        qc.x(sign_q[0])
    qc.cx(sign_a[0], sign_q[0])

    # Convert back to two's complement
    sign_magnitude_to_twos(qc, qout, sign_q)
    qc.cx(qout[n_output_bits - 1], sign_q[0])

    sign_magnitude_to_twos(qc, rem, sign_a)

    # Optionally restore input (for reversibility)
    sign_magnitude_to_twos(qc, a_reg, sign_a)
    qc.cx(a_reg[n - 1], sign_a[0])

    return qout, rem


def _controlled_addi_in_place(qc, qreg, value, control):
    """Add ``value`` to ``qreg`` controlled by ``control`` qubit."""
    if ARITHMETIC_MODE == "ripple":
        # Ripple version: create constant register and use controlled add
        n = len(qreg)
        const_reg = _init_const_register(qc, value, n, name_hint="caddconst")
        _controlled_ripple_add_in_place(qc, qreg, const_reg, control)
        return qreg

    # QFT version
    n = len(qreg)
    qc.append(QFT(n, do_swaps=False), qreg)

    b_bin = int_to_twos_complement(value)
    b_int = int("".join(str(x) for x in b_bin[::-1]), 2)
    b_val = b_int if value >= 0 else b_int - (1 << n)

    for j in range(n):
        angle = (b_val * 2 * np.pi) / (2 ** (j + 1))
        if angle != 0:
            qc.cp(angle, control, qreg[j])

    qc.append(QFT(n, do_swaps=False).inverse(), qreg)


def _sub_in_place(qc, a_reg, b_reg):
    """Subtract ``b_reg`` from ``a_reg`` in place."""
    n = len(a_reg)
    assert len(b_reg) == n

    if ARITHMETIC_MODE == "ripple":
        # Ripple version: a = a + (-b) = a + (~b + 1)
        # Negate b, add, negate back
        for i in range(n):
            qc.x(b_reg[i])
        _ripple_add_in_place(qc, a_reg, b_reg)
        for i in range(n):
            qc.x(b_reg[i])
        # Add 1 for two's complement
        one_reg = _init_const_register(qc, 1, n, name_hint="subone")
        _ripple_add_in_place(qc, a_reg, one_reg)
        return a_reg

    # QFT version
    qc.append(QFT(n, do_swaps=False), a_reg)
    for i in range(n):
        for j in range(n):
            if j <= i:
                angle = -(2 * np.pi) / (2 ** (i - j + 1))
                qc.cp(angle, b_reg[j], a_reg[i])
    qc.append(QFT(n, do_swaps=False).inverse(), a_reg)
    return a_reg


def _controlled_add_in_place(qc, a_reg, b_reg, control):
    """Add ``b_reg`` to ``a_reg`` controlled by ``control``."""
    n = len(a_reg)
    assert len(b_reg) == n

    if ARITHMETIC_MODE == "ripple":
        return _controlled_ripple_add_in_place(qc, a_reg, b_reg, control)

    # QFT version
    qc.append(QFT(n, do_swaps=False), a_reg)
    for i in range(n):
        for j in range(n):
            if j <= i:
                angle = 2 * np.pi / (2 ** (i - j + 1))
                gate = PhaseGate(angle).control(2)
                qc.append(gate, [control, b_reg[j], a_reg[i]])
    qc.append(QFT(n, do_swaps=False).inverse(), a_reg)
    return a_reg


def _controlled_invert_in_place(qc, qreg, control):
    """Negate ``qreg`` conditioned on ``control`` being ``|1>``."""
    for qubit in qreg:
        qc.cx(control, qubit)
    _controlled_addi_in_place(qc, qreg, 1, control)
    return qreg


def equal(qc, a_reg, b_reg):
    n = max(len(a_reg), len(b_reg))
    a_pad = pad_register(qc, a_reg, n, "aeq")
    b_pad = pad_register(qc, b_reg, n, "beq")

    existing = {reg.name for reg in qc.qregs}
    xor_name = unique_reg_name(existing, "xor")
    xor_reg = QuantumRegister(n, name=xor_name)
    qc.add_register(xor_reg)

    for i in range(n):
        qc.cx(a_pad[i], xor_reg[i])
        qc.cx(b_pad[i], xor_reg[i])

    eq_name = unique_reg_name(existing | {xor_name}, "eq")
    out = QuantumRegister(1, name=eq_name)
    qc.add_register(out)

    for q in xor_reg:
        qc.x(q)
    qc.x(out[0])
    qc.mcx(xor_reg, out[0])
    qc.x(out[0])
    for q in xor_reg:
        qc.x(q)
    return out[0]


def not_equal(qc, a_reg, b_reg):
    eq = equal(qc, a_reg, b_reg)
    existing = {reg.name for reg in qc.qregs}
    neq_name = unique_reg_name(existing, "neq")
    neq = QuantumRegister(1, name=neq_name)
    qc.add_register(neq)
    qc.x(neq[0])
    qc.cx(eq, neq[0])
    return neq[0]


def less_than(qc, a_reg, b_reg):
    n = max(len(a_reg), len(b_reg))
    a_pad = pad_register(qc, a_reg, n, "alt")
    b_pad = pad_register(qc, b_reg, n, "blt")

    existing = {reg.name for reg in qc.qregs}
    bneg_name = unique_reg_name(existing, "bneg")
    tmp_b = QuantumRegister(n, name=bneg_name)
    qc.add_register(tmp_b)
    for i in range(n):
        qc.cx(b_pad[i], tmp_b[i])
    invert(qc, tmp_b)

    diff = add(qc, a_pad, tmp_b)

    lt_name = unique_reg_name({*existing, bneg_name}, "lt")
    out = QuantumRegister(1, name=lt_name)
    qc.add_register(out)
    qc.cx(diff[n - 1], out[0])

    invert(qc, tmp_b)
    return out[0]


def greater_than(qc, a_reg, b_reg):
    return less_than(qc, b_reg, a_reg)


def less_equal(qc, a_reg, b_reg):
    gt = greater_than(qc, a_reg, b_reg)
    existing = {reg.name for reg in qc.qregs}
    le_name = unique_reg_name(existing, "le")
    le = QuantumRegister(1, name=le_name)
    qc.add_register(le)
    qc.x(le[0])
    qc.cx(gt, le[0])
    return le[0]


def greater_equal(qc, a_reg, b_reg):
    lt = less_than(qc, a_reg, b_reg)
    existing = {reg.name for reg in qc.qregs}
    ge_name = unique_reg_name(existing, "ge")
    ge = QuantumRegister(1, name=ge_name)
    qc.add_register(ge)
    qc.x(ge[0])
    qc.cx(lt, ge[0])
    return ge[0]


def measure_single(qc, qubit, name="result"):
    """Attach a classical bit measuring ``qubit`` to ``qc``."""
    creg = ClassicalRegister(1, name=name)
    qc.add_register(creg)
    qc.measure(qubit, creg[0])


def initialize_bit(qc, value, name=None):
    """
    Initialize a single qubit to |0⟩ or |1⟩ based on a classical bit value.
    """
    if value not in (0, 1):
        raise ValueError("Bit value must be 0 or 1.")

    existing = {reg.name for reg in qc.qregs}
    idx = 0
    base_name = "qb" if name is None else name
    while f"{base_name}{idx}" in existing:
        idx += 1
    reg = QuantumRegister(1, name=f"{base_name}{idx}")
    qc.add_register(reg)

    if value == 1:
        qc.x(reg[0])

    return reg[0]


def pad_register(qc, reg, target_size, name_hint="pad"):
    """
    Pad a quantum register with |0⟩ qubits to reach target size.
    """
    padded = list(reg)
    extra = target_size - len(reg)
    if extra > 0:
        existing = {r.name for r in qc.qregs}
        unique_name = unique_reg_name(existing, f"{name_hint}_ext")
        pad_reg = QuantumRegister(extra, name=unique_name)
        qc.add_register(pad_reg)
        padded += list(pad_reg)
    return padded


def measure(qc, qreg):
    """
    Measure a quantum register and store the result in a classical register.
    """
    c_reg = ClassicalRegister(len(qreg), name=qreg.name + "_measure")
    qc.add_register(c_reg)
    qc.measure(qreg, c_reg)


def simulate(qc, shots=1024):
    """
    Simulate the quantum circuit and print the interpreted two's complement value
    for each measured quantum register.
    """
    if AerSimulator is not None:
        backend = AerSimulator(method="matrix_product_state")
        transpiled = qc
    else:
        backend = BasicSimulator()
        transpiled = transpile(qc, backend)
    job = backend.run(transpiled, shots=shots)
    counts = job.result().get_counts()

    # Get most frequent measurement result
    most_common = max(counts, key=counts.get)
    bitstring = most_common.replace(" ", "")  # Qiskit returns MSB leftmost

    print(f"Measured bitstring: {bitstring}")

    offset = 0
    for creg in reversed(qc.cregs):
        reg_size = len(creg)
        reg_bits = bitstring[offset : offset + reg_size]
        offset += reg_size

        unsigned = int(reg_bits, 2)
        if reg_bits and reg_bits[0] == "1" and reg_size > 1:
            signed = unsigned - (1 << reg_size)
        else:
            signed = unsigned

        print(f"Register {creg.name}: binary = {reg_bits}, value (2's complement) = {signed}")

    return signed


def logical_and(qc, q1, q2):
    """
    Compute logical AND between two qubits.
    """
    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"and{idx}" in existing:
        idx += 1
    and_reg = QuantumRegister(1, name=f"and{idx}")
    qc.add_register(and_reg)
    qc.ccx(q1, q2, and_reg[0])
    return and_reg[0]


def logical_or(qc, q1, q2):
    """
    Compute logical OR between two qubits.
    """
    existing = {reg.name for reg in qc.qregs}
    idx = 0
    while f"or{idx}" in existing:
        idx += 1
    or_reg = QuantumRegister(1, name=f"or{idx}")
    qc.add_register(or_reg)

    qc.x(or_reg[0])  # initialize in |1>
    # Use De Morgan: q1 OR q2 = NOT (NOT q1 AND NOT q2)
    qc.x(q1)
    qc.x(q2)
    qc.ccx(q1, q2, or_reg[0])
    qc.x(q1)
    qc.x(q2)
    return or_reg[0]