"""
Diagnostic script to verify and debug the iSWAP result (Reviewer 3 & Editorial Comment 5).
Investigates possible bugs in original implementation:
1. Standard iSWAP: diag(1, [[0, 1j], [1j, 0]], 1)
2. Incorrect matrix formulations (e.g. [[0, i], [-i, 0]], or missing complex dtype)
3. Dropping imaginary part (e.g., np.real() on statevector before measurement or gate)
4. Gradient estimation bugs with complex amplitudes.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.dataset import load_wdbc_data, get_train_test_data
from src.ranking import compute_xgboost_permutation_ranking, get_feature_allocation
from src.quantum_circuit import Ry, Rz, SWAP_4x4, SQRT_SWAP_4x4, ISWAP_4x4, expectation_value
from sklearn.metrics import accuracy_score, roc_auc_score

X, y, feature_names = load_wdbc_data()
X_tr, X_te, y_tr, y_te, scaler = get_train_test_data(X, y, test_size=0.2, random_state=42)

ranked, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=42, n_repeats=10)
q_alloc = get_feature_allocation(30, 2, 'ibip', ranked)
q0_idx, q1_idx = q_alloc[0], q_alloc[1]

I2 = np.eye(2, dtype=complex)
Z0 = np.kron(np.array([[1,0],[0,-1]], dtype=complex), I2)
Z1 = np.kron(I2, np.array([[1,0],[0,-1]], dtype=complex))
psi0 = np.array([1, 0, 0, 0], dtype=complex)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def run_iswap_variant(variant_name, gate_mat, drop_imag=False):
    np.random.seed(42)
    W0 = np.random.randn(len(q0_idx)) / np.sqrt(len(q0_idx))
    W1 = np.random.randn(len(q1_idx)) / np.sqrt(len(q1_idx))
    enc0, enc1 = np.random.randn() * 0.1, np.random.randn() * 0.1
    var0, var1 = np.random.randn() * 0.1, np.random.randn() * 0.1
    beta = np.random.randn(2) * 0.1
    bias = 0.0
    
    def circuit_forward(phi0, phi1, e0, e1, v0, v1):
        s = psi0.copy()
        s = np.kron(Ry(phi0), Ry(phi1)) @ s
        s = np.kron(Rz(e0), Rz(e1)) @ s
        s = gate_mat @ s
        if drop_imag:
            s = np.real(s).astype(complex) # Truncates phase
        s = np.kron(Ry(v0), Ry(v1)) @ s
        s = np.kron(Rz(v0), Rz(v1)) @ s
        
        z0 = expectation_value(s, Z0)
        z1 = expectation_value(s, Z1)
        return z0, z1

    lr, lam, epochs = 0.01, 0.5, 50
    for epoch in range(epochs):
        for x, yt in zip(X_tr, y_tr):
            u0 = np.dot(W0, x[q0_idx]) / np.sqrt(len(q0_idx))
            u1 = np.dot(W1, x[q1_idx]) / np.sqrt(len(q1_idx))
            phi0 = np.pi * np.tanh(u0)
            phi1 = np.pi * np.tanh(u1)
            
            z0, z1 = circuit_forward(phi0, phi1, enc0, enc1, var0, var1)
            logit = beta[0] * z0 + beta[1] * z1 + bias
            p = sigmoid(logit)
            dL = p - yt
            
            beta[0] -= lr * dL * z0 * (1 + lam * np.sin(np.pi * beta[0]))
            beta[1] -= lr * dL * z1 * (1 + lam * np.sin(np.pi * beta[1]))
            bias -= lr * dL * (1 + lam * np.sin(np.pi * bias))
            
            # PSR for enc0
            zp0, zp1 = circuit_forward(phi0, phi1, enc0 + np.pi/2, enc1, var0, var1)
            zm0, zm1 = circuit_forward(phi0, phi1, enc0 - np.pi/2, enc1, var0, var1)
            pp = sigmoid(beta[0]*zp0 + beta[1]*zp1 + bias)
            pm = sigmoid(beta[0]*zm0 + beta[1]*zm1 + bias)
            g_enc0 = dL * (pp - pm) / 2.0
            enc0 -= lr * g_enc0 * (1 + lam * np.sin(np.pi * enc0))

            # PSR for enc1
            zp0, zp1 = circuit_forward(phi0, phi1, enc0, enc1 + np.pi/2, var0, var1)
            zm0, zm1 = circuit_forward(phi0, phi1, enc0, enc1 - np.pi/2, var0, var1)
            pp = sigmoid(beta[0]*zp0 + beta[1]*zp1 + bias)
            pm = sigmoid(beta[0]*zm0 + beta[1]*zm1 + bias)
            g_enc1 = dL * (pp - pm) / 2.0
            enc1 -= lr * g_enc1 * (1 + lam * np.sin(np.pi * enc1))

            # PSR for var0
            zp0, zp1 = circuit_forward(phi0, phi1, enc0, enc1, var0 + np.pi/2, var1)
            zm0, zm1 = circuit_forward(phi0, phi1, enc0, enc1, var0 - np.pi/2, var1)
            pp = sigmoid(beta[0]*zp0 + beta[1]*zp1 + bias)
            pm = sigmoid(beta[0]*zm0 + beta[1]*zm1 + bias)
            g_var0 = dL * (pp - pm) / 2.0
            var0 -= lr * g_var0 * (1 + lam * np.sin(np.pi * var0))

            # PSR for var1
            zp0, zp1 = circuit_forward(phi0, phi1, enc0, enc1, var0, var1 + np.pi/2)
            zm0, zm1 = circuit_forward(phi0, phi1, enc0, enc1, var0, var1 - np.pi/2)
            pp = sigmoid(beta[0]*zp0 + beta[1]*zp1 + bias)
            pm = sigmoid(beta[0]*zm0 + beta[1]*zm1 + bias)
            g_var1 = dL * (pp - pm) / 2.0
            var1 -= lr * g_var1 * (1 + lam * np.sin(np.pi * var1))

            # Gradients for W0 via phi0 PSR
            zp0, zp1 = circuit_forward(phi0 + np.pi/2, phi1, enc0, enc1, var0, var1)
            zm0, zm1 = circuit_forward(phi0 - np.pi/2, phi1, enc0, enc1, var0, var1)
            pp = sigmoid(beta[0]*zp0 + beta[1]*zp1 + bias)
            pm = sigmoid(beta[0]*zm0 + beta[1]*zm1 + bias)
            df_dphi0 = (pp - pm) / 2.0
            grad_W0 = dL * df_dphi0 * (np.pi * (1.0 - np.tanh(u0)**2)) * (x[q0_idx] / np.sqrt(len(q0_idx)))
            W0 -= lr * grad_W0 * (1 + lam * np.sin(np.pi * W0))

            # Gradients for W1 via phi1 PSR
            zp0, zp1 = circuit_forward(phi0, phi1 + np.pi/2, enc0, enc1, var0, var1)
            zm0, zm1 = circuit_forward(phi0, phi1 - np.pi/2, enc0, enc1, var0, var1)
            pp = sigmoid(beta[0]*zp0 + beta[1]*zp1 + bias)
            pm = sigmoid(beta[0]*zm0 + beta[1]*zm1 + bias)
            df_dphi1 = (pp - pm) / 2.0
            grad_W1 = dL * df_dphi1 * (np.pi * (1.0 - np.tanh(u1)**2)) * (x[q1_idx] / np.sqrt(len(q1_idx)))
            W1 -= lr * grad_W1 * (1 + lam * np.sin(np.pi * W1))

    preds, probs = [], []
    for x in X_te:
        u0 = np.dot(W0, x[q0_idx]) / np.sqrt(len(q0_idx))
        u1 = np.dot(W1, x[q1_idx]) / np.sqrt(len(q1_idx))
        z0, z1 = circuit_forward(np.pi * np.tanh(u0), np.pi * np.tanh(u1), enc0, enc1, var0, var1)
        p = sigmoid(beta[0]*z0 + beta[1]*z1 + bias)
        probs.append(p)
        preds.append(1 if p >= 0.5 else 0)
    acc = accuracy_score(y_te, preds)
    auc = roc_auc_score(y_te, probs)
    print(f"[{variant_name:40s}] Acc: {acc*100:6.2f}%, AUC: {auc:.4f}, Benign preds: {sum(preds):2d}/114")

if __name__ == "__main__":
    print("Testing iSWAP Bug Hypotheses:")
    # Standard iSWAP
    run_iswap_variant("Standard iSWAP", ISWAP_4x4)
    
    # Hypothesis A: Real dtype coercion (silently dropping imaginary part: i -> 0)
    # If someone defined iSWAP without dtype=complex, or cast to float:
    real_cast_iswap = np.real(ISWAP_4x4) # Has zeros at (0,1) and (1,0) - NOT unitary!
    run_iswap_variant("Cast to float (dropping imaginary entries)", real_cast_iswap)
    
    # Hypothesis B: Dropping imaginary part of statevector
    run_iswap_variant("Dropping imag part of statevector", ISWAP_4x4, drop_imag=True)
    
    # Hypothesis C: Transposed or sign bug: [[1,0,0,0],[0,0,-1j,0],[0,1j,0,0],[0,0,0,1]]
    asym_iswap = np.array([[1,0,0,0],[0,0,-1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
    run_iswap_variant("Asymmetric iSWAP", asym_iswap)
    
    # Hypothesis D: Wrong basis matrix / missing swap
    diag_phase = np.array([[1,0,0,0],[0,1j,0,0],[0,0,1j,0],[0,0,0,1]], dtype=complex)
    run_iswap_variant("Diagonal phase without swap", diag_phase)
