"""
Quantum-Inspired Variational Classifier (QIVC).
High-speed statevector simulation in Hilbert space H = (C^2)^N (N=2 or N=3) using tensor contractions:
- Trainable classical feature projection layers: phi_q = pi * tanh(W_q^T x_q / sqrt(|x_q|))
- Parameterized quantum rotation encoding: Ry(phi_q) and Rz(e_q)
- Interaction topology layer (CNOT-Star, CNOT-Chain, CNOT-Ring, SWAP, sqrtSWAP, iSWAP, etc.)
- Trainable variational rotation layer: Ry(v_q), Rz(v_q)
- Pauli-Z expectation measurements: <Z_q>
- Readout layer: logit = sum_q (beta_q * <Z_q>) + bias, prob = sigmoid(logit)
- Optimization via Superpositional Gradient Descent (SuperGD), Parameter Shift Rule (PSR),
  and Finite Difference / Analytical projection gradients.
"""

import numpy as np
from src.quantum_circuit import (
    Ry, Rz, cnot_gate, two_qubit_gate_full,
    SWAP_4x4, SQRT_SWAP_4x4, ISWAP_4x4, get_z_operator, expectation_value
)
from src.optimizers import supergd_update

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def bce_loss(y_true, y_prob, eps=1e-7):
    p = np.clip(y_prob, eps, 1.0 - eps)
    return - (y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

class QuantumInspiredClassifier:
    def __init__(self, n_qubits=3, topology="cnot_star", feature_indices=None,
                 lr=0.01, lam=0.5, eps_fd=1e-4, seed=42):
        self.n_qubits = n_qubits
        self.topology = topology.lower()
        self.feature_indices = feature_indices
        self.lr = lr
        self.lam = lam
        self.eps_fd = eps_fd
        self.seed = seed
        
        # Build basis state |0...0>
        self.dim = 2 ** n_qubits
        self.psi0 = np.zeros(self.dim, dtype=complex)
        self.psi0[0] = 1.0
        
        # Precompute Z operators
        self.Z_ops = [get_z_operator(q, n_qubits) for q in range(n_qubits)]
        
        # Build topology interaction operators
        self.interaction_matrices = self._build_interaction_operators()
        
        # Initialize parameters
        self.reset_parameters(seed=seed)
        
    def _build_interaction_operators(self):
        ops = []
        n = self.n_qubits
        top = self.topology
        
        if n == 2:
            if top == "swap":
                ops.append(SWAP_4x4)
            elif top in ["sqrtswap", "sqrt_swap"]:
                ops.append(SQRT_SWAP_4x4)
            elif top == "iswap":
                ops.append(ISWAP_4x4)
            elif top in ["cnot", "cnot01"]:
                ops.append(cnot_gate(0, 1, 2))
            elif top == "cnot10":
                ops.append(cnot_gate(1, 0, 2))
            elif top in ["none", "independent"]:
                pass
            else:
                raise ValueError(f"Unknown 2-qubit topology: {top}")
        elif n == 3:
            if top == "cnot_star":
                ops.append(cnot_gate(0, 1, 3))
                ops.append(cnot_gate(0, 2, 3))
            elif top == "cnot_chain":
                ops.append(cnot_gate(0, 1, 3))
                ops.append(cnot_gate(1, 2, 3))
            elif top == "cnot_ring":
                ops.append(cnot_gate(0, 1, 3))
                ops.append(cnot_gate(1, 2, 3))
                ops.append(cnot_gate(2, 0, 3))
            elif top == "swap_chain":
                ops.append(two_qubit_gate_full(SWAP_4x4, 0, 1, 3))
                ops.append(two_qubit_gate_full(SWAP_4x4, 1, 2, 3))
            elif top == "swap_star":
                ops.append(two_qubit_gate_full(SWAP_4x4, 0, 1, 3))
                ops.append(two_qubit_gate_full(SWAP_4x4, 0, 2, 3))
            elif top in ["full_mesh", "all_to_all"]:
                ops.append(cnot_gate(0, 1, 3))
                ops.append(cnot_gate(1, 2, 3))
                ops.append(cnot_gate(0, 2, 3))
            elif top in ["none", "independent"]:
                pass
            else:
                raise ValueError(f"Unknown 3-qubit topology: {top}")
        else:
            raise ValueError(f"Unsupported qubit count: {n}")
            
        return ops

    def reset_parameters(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        n = self.n_qubits
        if self.feature_indices is None:
            k = 30 // n
            self.feature_indices = [np.arange(q * k, (q + 1) * k) for q in range(n)]
            
        # Projection weights W_q
        self.W = []
        for q in range(n):
            n_feat = len(self.feature_indices[q])
            w = np.random.randn(n_feat) / np.sqrt(max(n_feat, 1))
            self.W.append(w)
            
        # Encoding angles e_q and variational angles v_q
        self.enc = np.random.randn(n) * 0.1
        self.var = np.random.randn(n) * 0.1
        
        # Readout weights and bias
        self.beta = np.random.randn(n) * 0.1
        self.bias = 0.0

    def compute_phi(self, x, W_override=None):
        W_list = self.W if W_override is None else W_override
        phi = np.zeros(self.n_qubits)
        for q in range(self.n_qubits):
            idx = self.feature_indices[q]
            u = np.dot(W_list[q], x[idx]) / np.sqrt(len(idx))
            phi[q] = np.pi * np.tanh(u)
        return phi

    def circuit_forward(self, phi, enc_angles, var_angles):
        """
        High-speed tensor-contraction statevector evolution.
        """
        n = self.n_qubits
        
        if n == 2:
            s = np.zeros((2, 2), dtype=complex)
            s[0, 0] = 1.0
            
            # Encoding: Rz(enc) @ Ry(phi)
            U0 = Rz(enc_angles[0]) @ Ry(phi[0])
            U1 = Rz(enc_angles[1]) @ Ry(phi[1])
            s = (U0 @ s) @ U1.T
            
            # Interaction layer
            vec = s.ravel()
            for U in self.interaction_matrices:
                vec = U @ vec
            s = vec.reshape(2, 2)
            
            # Variational: Rz(var) @ Ry(var)
            V0 = Rz(var_angles[0]) @ Ry(var_angles[0])
            V1 = Rz(var_angles[1]) @ Ry(var_angles[1])
            s = (V0 @ s) @ V1.T
            
            # Expectations of Z0 and Z1:
            # Z0 = diag(1, -1) on axis 0: sum(|s[0, :]|^2) - sum(|s[1, :]|^2)
            # Z1 = diag(1, -1) on axis 1: sum(|s[:, 0]|^2) - sum(|s[:, 1]|^2)
            p0 = np.sum(np.abs(s[0, :]) ** 2)
            p1 = np.sum(np.abs(s[1, :]) ** 2)
            z0 = p0 - p1
            
            q0 = np.sum(np.abs(s[:, 0]) ** 2)
            q1 = np.sum(np.abs(s[:, 1]) ** 2)
            z1 = q0 - q1
            return np.array([z0, z1])
            
        elif n == 3:
            s = np.zeros((2, 2, 2), dtype=complex)
            s[0, 0, 0] = 1.0
            
            # Encoding: Rz(enc) @ Ry(phi)
            U0 = Rz(enc_angles[0]) @ Ry(phi[0])
            U1 = Rz(enc_angles[1]) @ Ry(phi[1])
            U2 = Rz(enc_angles[2]) @ Ry(phi[2])
            
            s = (U0 @ s.reshape(2, 4)).reshape(2, 2, 2)
            s = (U1 @ s.transpose(1, 0, 2).reshape(2, 4)).reshape(2, 2, 2).transpose(1, 0, 2)
            s = (s.reshape(4, 2) @ U2.T).reshape(2, 2, 2)
            
            # Interaction layer
            vec = s.ravel()
            for U in self.interaction_matrices:
                vec = U @ vec
            s = vec.reshape(2, 2, 2)
            
            # Variational: Rz(var) @ Ry(var)
            V0 = Rz(var_angles[0]) @ Ry(var_angles[0])
            V1 = Rz(var_angles[1]) @ Ry(var_angles[1])
            V2 = Rz(var_angles[2]) @ Ry(var_angles[2])
            
            s = (V0 @ s.reshape(2, 4)).reshape(2, 2, 2)
            s = (V1 @ s.transpose(1, 0, 2).reshape(2, 4)).reshape(2, 2, 2).transpose(1, 0, 2)
            s = (s.reshape(4, 2) @ V2.T).reshape(2, 2, 2)
            
            # Expectations of Z0, Z1, Z2
            p0 = np.sum(np.abs(s[0, :, :]) ** 2)
            p1 = np.sum(np.abs(s[1, :, :]) ** 2)
            z0 = p0 - p1
            
            q0 = np.sum(np.abs(s[:, 0, :]) ** 2)
            q1 = np.sum(np.abs(s[:, 1, :]) ** 2)
            z1 = q0 - q1
            
            r0 = np.sum(np.abs(s[:, :, 0]) ** 2)
            r1 = np.sum(np.abs(s[:, :, 1]) ** 2)
            z2 = r0 - r1
            return np.array([z0, z1, z2])
            
        else:
            raise ValueError(f"Unsupported qubit count: {n}")

    def forward(self, x, W_override=None, enc_override=None, var_override=None,
                beta_override=None, bias_override=None):
        enc = self.enc if enc_override is None else enc_override
        var = self.var if var_override is None else var_override
        beta = self.beta if beta_override is None else beta_override
        bias = self.bias if bias_override is None else bias_override
        
        phi = self.compute_phi(x, W_override=W_override)
        z = self.circuit_forward(phi, enc, var)
        
        logit = np.dot(beta, z) + bias
        prob = sigmoid(logit)
        return prob, z

    def fit_sample(self, x, y, fast_grad=True):
        """
        Online SuperGD step with PSR and backprop.
        """
        phi = self.compute_phi(x)
        z = self.circuit_forward(phi, self.enc, self.var)
        logit = np.dot(self.beta, z) + self.bias
        p = sigmoid(logit)
        dL = p - y
        
        # Readout gradients
        grad_beta = dL * z
        grad_bias = dL
        self.beta = supergd_update(self.beta, grad_beta, self.lr, self.lam)
        self.bias = supergd_update(self.bias, grad_bias, self.lr, self.lam)
        
        n = self.n_qubits
        
        # Parameter Shift Rule for encoding angles
        for q in range(n):
            plus, minus = self.enc.copy(), self.enc.copy()
            plus[q] += np.pi / 2.0
            minus[q] -= np.pi / 2.0
            zp = self.circuit_forward(phi, plus, self.var)
            zm = self.circuit_forward(phi, minus, self.var)
            pp = sigmoid(np.dot(self.beta, zp) + self.bias)
            pm = sigmoid(np.dot(self.beta, zm) + self.bias)
            g_enc = dL * (pp - pm) / 2.0
            self.enc[q] = supergd_update(self.enc[q], g_enc, self.lr, self.lam)
            
        # Parameter Shift Rule for variational angles
        for q in range(n):
            plus, minus = self.var.copy(), self.var.copy()
            plus[q] += np.pi / 2.0
            minus[q] -= np.pi / 2.0
            zp = self.circuit_forward(phi, self.enc, plus)
            zm = self.circuit_forward(phi, self.enc, minus)
            pp = sigmoid(np.dot(self.beta, zp) + self.bias)
            pm = sigmoid(np.dot(self.beta, zm) + self.bias)
            g_var = dL * (pp - pm) / 2.0
            self.var[q] = supergd_update(self.var[q], g_var, self.lr, self.lam)
            
        # Projection weights W_q via PSR on phi_q + chain rule
        for q in range(n):
            idx = self.feature_indices[q]
            u = np.dot(self.W[q], x[idx]) / np.sqrt(len(idx))
            dphi_du = np.pi * (1.0 - np.tanh(u) ** 2)
            du_dW = x[idx] / np.sqrt(len(idx))
            
            phi_p, phi_m = phi.copy(), phi.copy()
            phi_p[q] += np.pi / 2.0
            phi_m[q] -= np.pi / 2.0
            zp = self.circuit_forward(phi_p, self.enc, self.var)
            zm = self.circuit_forward(phi_m, self.enc, self.var)
            pp = sigmoid(np.dot(self.beta, zp) + self.bias)
            pm = sigmoid(np.dot(self.beta, zm) + self.bias)
            df_dphi = (pp - pm) / 2.0
            
            grad_Wq = dL * df_dphi * dphi_du * du_dW
            self.W[q] = supergd_update(self.W[q], grad_Wq, self.lr, self.lam)
            
        return p

    def fit(self, X_train, y_train, epochs=50, fast_grad=True, verbose=False):
        loss_history = []
        acc_history = []
        n_samples = len(X_train)
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            
            for i in range(n_samples):
                x = X_train[i]
                y = y_train[i]
                p = self.fit_sample(x, y, fast_grad=fast_grad)
                
                total_loss += bce_loss(y, p)
                pred = 1 if p >= 0.5 else 0
                if pred == y:
                    correct += 1
                    
            epoch_loss = total_loss / n_samples
            epoch_acc = (correct / n_samples) * 100.0
            loss_history.append(epoch_loss)
            acc_history.append(epoch_acc)
            
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch {epoch+1:2d}/{epochs:2d} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
                
        return loss_history, acc_history

    def predict_proba(self, X):
        probs = np.zeros(len(X))
        for i in range(len(X)):
            phi = self.compute_phi(X[i])
            z = self.circuit_forward(phi, self.enc, self.var)
            probs[i] = sigmoid(np.dot(self.beta, z) + self.bias)
        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
