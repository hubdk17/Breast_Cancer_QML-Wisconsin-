"""
Feature ranking and partition module for Quantum-Inspired Variational Classifiers.
Supports 4 Comprehensive Allocation Strategies for Ablation Analysis:
1. 'sequential': Contiguous block partitioning based on dataset column order.
2. 'random': Random distribution of features across qubits.
3. 'importance_blocked': Importance-based allocation WITHOUT interleaving
   (Top-ranked features concentrated in Q0, mid in Q1, bottom in Q2).
4. 'ibip': Importance-Balanced Interleaved Projection (cyclical round-robin allocation).

Handles arbitrary number of features D and qubit registers n (with robust support for
uneven divisibility, e.g. 13 features across 3 qubits -> 5, 4, 4).
"""

import numpy as np
import xgboost as xgb
from sklearn.inspection import permutation_importance

def compute_xgboost_permutation_ranking(X_train, y_train, random_state=42, n_repeats=20):
    """
    Fits an XGBoost classifier on training data and computes permutation importance.
    Returns:
        ranked_indices (np.ndarray): Indices of features sorted by descending importance.
        importances_mean (np.ndarray): Mean importance scores for all features.
        importances_std (np.ndarray): Std dev of importance scores.
        xgb_model: The fitted XGBoost model.
    """
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    perm = permutation_importance(
        xgb_model,
        X_train,
        y_train,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    ranked_indices = np.argsort(perm.importances_mean)[::-1]
    return ranked_indices, perm.importances_mean, perm.importances_std, xgb_model

def get_feature_allocation(n_features=30, n_qubits=3, method="ibip", ranked_indices=None, seed=42):
    """
    Allocates features to qubits using one of the four ablation strategies.
    
    Parameters:
        n_features (int): Total number of features (e.g., 30 for WDBC, 22 for Parkinsons, 13 for Heart Disease).
        n_qubits (int): Number of qubits (e.g., 2 or 3).
        method (str): 'ibip', 'sequential', 'random', or 'importance_blocked'.
        ranked_indices (np.ndarray): Feature indices ranked by descending importance (required for IBIP and importance_blocked).
        seed (int): Random seed for 'random' allocation.
        
    Returns:
        qubit_indices (list of np.ndarray): List of length n_qubits containing feature indices assigned to each register.
    """
    method = method.lower()
    
    # Calculate partition block sizes (distributing remainder across first registers)
    base_size = n_features // n_qubits
    rem = n_features % n_qubits
    sizes = [base_size + (1 if q < rem else 0) for q in range(n_qubits)]
    
    if method == "ibip":
        if ranked_indices is None:
            raise ValueError("ranked_indices must be provided for IBIP allocation.")
        # Interleaved round-robin distribution:
        # Qubit q gets ranked_indices[q::n_qubits]
        qubit_indices = [ranked_indices[q::n_qubits] for q in range(n_qubits)]
        
    elif method == "sequential":
        # Contiguous block partitioning based on original column ordering
        qubit_indices = []
        start = 0
        for sz in sizes:
            qubit_indices.append(np.arange(start, start + sz))
            start += sz
            
    elif method in ["importance_blocked", "importance_no_interleave", "concentrated"]:
        if ranked_indices is None:
            raise ValueError("ranked_indices must be provided for importance_blocked allocation.")
        # Packs top features into Qubit 0, next into Qubit 1, least important into Qubit 2
        qubit_indices = []
        start = 0
        for sz in sizes:
            qubit_indices.append(ranked_indices[start:start + sz])
            start += sz
            
    elif method == "random":
        # Random distribution of features across qubits
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(n_features)
        qubit_indices = []
        start = 0
        for sz in sizes:
            qubit_indices.append(np.sort(shuffled[start:start + sz]))
            start += sz
            
    else:
        raise ValueError(f"Unknown allocation method: '{method}'. Supported: 'ibip', 'sequential', 'random', 'importance_blocked'")
        
    return qubit_indices
