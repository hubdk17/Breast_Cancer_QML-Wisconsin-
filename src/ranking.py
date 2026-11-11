"""
Feature ranking and partition module for Quantum-Inspired Variational Classifiers.
Implements:
1. XGBoost Permutation Importance Ranking (computed strictly on training data).
2. Importance-Balanced Interleaved Projection (IBIP).
3. Sequential Partitioning baseline.
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
        n_estimators=300,
        max_depth=4,
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

def get_feature_allocation(n_features=30, n_qubits=3, method="ibip", ranked_indices=None):
    """
    Allocates features to qubits using either IBIP (interleaved) or sequential strategy.
    
    Parameters:
        n_features (int): Total number of features (e.g., 30 for WDBC).
        n_qubits (int): Number of qubits (e.g., 2 or 3).
        method (str): 'ibip' or 'sequential'.
        ranked_indices (np.ndarray): Feature indices ranked by importance (required for IBIP).
        
    Returns:
        qubit_feature_indices (list of np.ndarray): List of length n_qubits, where each
            element is an array of feature indices assigned to that qubit.
    """
    if method == "ibip":
        if ranked_indices is None:
            raise ValueError("ranked_indices must be provided for IBIP allocation.")
        # Interleaved round-robin distribution:
        # Qubit q gets ranked_indices[q::n_qubits]
        qubit_indices = [ranked_indices[q::n_qubits] for q in range(n_qubits)]
    elif method == "sequential":
        # Sequential block partitioning:
        # e.g., for 30 features and 3 qubits -> 10, 10, 10 sequential blocks
        # [0..9], [10..19], [20..29]
        features_per_qubit = n_features // n_qubits
        qubit_indices = []
        for q in range(n_qubits):
            start = q * features_per_qubit
            end = (q + 1) * features_per_qubit if q < n_qubits - 1 else n_features
            qubit_indices.append(np.arange(start, end))
    else:
        raise ValueError(f"Unknown allocation method: {method}")
        
    return qubit_indices
