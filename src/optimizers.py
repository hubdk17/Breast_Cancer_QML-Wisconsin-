"""
Optimization Algorithms for Quantum-Inspired Variational Classifiers.
Implements:
1. Superpositional Gradient Descent (SuperGD):
   theta_{t+1} = theta_t - eta * g * (1 + lambda * sin(pi * theta_t))
2. Standard SGD baseline.
3. Adam optimizer baseline.
"""

import numpy as np

def supergd_update(param, grad, lr=0.01, lam=0.5):
    """
    Superpositional Gradient Descent parameter update.
    Args:
        param (float or np.ndarray): Parameter value(s).
        grad (float or np.ndarray): Gradient of loss w.r.t. parameter.
        lr (float): Learning rate (eta).
        lam (float): Interference modulation strength (lambda).
    Returns:
        updated_param: New parameter value(s).
    """
    modulation = 1.0 + lam * np.sin(np.pi * param)
    return param - lr * grad * modulation

def standard_sgd_update(param, grad, lr=0.01):
    """Standard SGD parameter update."""
    return param - lr * grad

class AdamOptimizer:
    """Standard Adam optimizer for baseline comparisons."""
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0
        
    def update(self, key, param, grad):
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
        self.t += 1
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
