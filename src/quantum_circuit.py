"""
Statevector Quantum Circuit Simulator in Pure NumPy.
Simulates pure quantum states in Hilbert space H = (C^2)^N (dimension 2^N).
Supports single-qubit rotations (Ry, Rz) and two-qubit entangling gates
(CNOT, SWAP, sqrtSWAP, iSWAP) across arbitrary qubit pairs.
"""

import numpy as np

# Single-qubit operators
I2 = np.eye(2, dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)

def Ry(theta):
    """Single-qubit rotation around Y-axis."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)

def Rz(theta):
    """Single-qubit rotation around Z-axis."""
    return np.array([[np.exp(-1j * theta / 2.0), 0],
                     [0, np.exp(1j * theta / 2.0)]], dtype=complex)

def embed_single_qubit_gate(gate, target, n_qubits):
    """Embeds single-qubit gate into n-qubit Hilbert space via Kronecker products."""
    ops = [I2] * n_qubits
    ops[target] = gate
    U = ops[0]
    for k in range(1, n_qubits):
        U = np.kron(U, ops[k])
    return U

def cnot_gate(control, target, n_qubits=3):
    """Constructs CNOT unitary for n-qubit system."""
    dim = 2 ** n_qubits
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(format(i, f"0{n_qubits}b"))
        if bits[control] == "1":
            bits[target] = "0" if bits[target] == "1" else "1"
        j = int("".join(bits), 2)
        U[j, i] = 1.0
    return U

def two_qubit_gate_full(gate_4x4, q0, q1, n_qubits=2):
    """
    Embeds a 4x4 two-qubit gate acting on (q0, q1) into an n-qubit unitary.
    Basis ordering: q0 is higher-significance or specified order.
    """
    if n_qubits == 2:
        if q0 == 0 and q1 == 1:
            return gate_4x4
        elif q0 == 1 and q1 == 0:
            # Swap basis
            perm = [0, 2, 1, 3]
            return gate_4x4[perm, :][:, perm]
        else:
            raise ValueError("Invalid qubit indices for 2-qubit system.")
            
    dim = 2 ** n_qubits
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits_i = list(format(i, f"0{n_qubits}b"))
        b0_in, b1_in = int(bits_i[q0]), int(bits_i[q1])
        in_idx_4 = (b0_in << 1) | b1_in
        
        for out_idx_4 in range(4):
            val = gate_4x4[out_idx_4, in_idx_4]
            if np.abs(val) > 1e-12:
                b0_out = (out_idx_4 >> 1) & 1
                b1_out = out_idx_4 & 1
                bits_j = bits_i.copy()
                bits_j[q0] = str(b0_out)
                bits_j[q1] = str(b1_out)
                j = int("".join(bits_j), 2)
                U[j, i] += val
    return U

# Canonical 4x4 two-qubit matrices in computational basis {|00>, |01>, |10>, |11>}
SWAP_4x4 = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

SQRT_SWAP_4x4 = np.array([
    [1, 0, 0, 0],
    [0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0],
    [0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0],
    [0, 0, 0, 1]
], dtype=complex)

ISWAP_4x4 = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

def get_z_operator(qubit, n_qubits):
    """Constructs Pauli-Z operator acting on designated qubit."""
    return embed_single_qubit_gate(Z_GATE, qubit, n_qubits)

def expectation_value(state, operator):
    """Computes <psi| operator |psi>."""
    return np.real(np.vdot(state, operator @ state))
