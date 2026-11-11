"""
Dataset loading and preprocessing module for Multiple Biomedical Benchmarks:
1. WDBC (Wisconsin Diagnostic Breast Cancer) - 569 samples, 30 features
2. Parkinson's Disease (UCI 174) - 195 samples, 22 features
3. Cleveland Heart Disease (UCI 45) - 303 samples, 13 features
4. Breast Cancer Coimbra (UCI 451) - 116 samples, 9 features

Ensures strict prevention of data leakage: scalers and feature selectors are fitted
strictly on training folds/splits.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_biomedical_dataset(dataset_name="wdbc"):
    """
    Loads specified biomedical dataset.
    Args:
        dataset_name (str): 'wdbc', 'parkinsons', 'heart_disease', or 'coimbra'.
    Returns:
        X (np.ndarray): N x D feature matrix.
        y (np.ndarray): Binary labels (1 = primary condition / positive, 0 = control/negative).
        feature_names (np.ndarray): Array of feature names.
    """
    dataset_name = dataset_name.lower().replace(" ", "_")
    csv_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        from data.download_all_biomedical_data import prepare_wdbc, prepare_parkinsons, prepare_heart_disease, prepare_coimbra
        if dataset_name == "wdbc": prepare_wdbc()
        elif dataset_name == "parkinsons": prepare_parkinsons()
        elif dataset_name == "heart_disease": prepare_heart_disease()
        elif dataset_name == "coimbra": prepare_coimbra()
        else: raise ValueError(f"Unknown dataset name: {dataset_name}")
        
    df = pd.read_csv(csv_path)
    ignore_cols = ["target", "diagnosis", "id", "name"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X = df[feature_cols].values.astype(np.float64)
    y = df["target"].values.astype(np.int64)
    feature_names = np.array(feature_cols)
    
    # In WDBC: 0=Malignant, 1=Benign. In clinical evaluation, Malignant is the primary condition.
    # For consistent cross-dataset clinical evaluations, we provide positive label semantics.
    return X, y, feature_names

def load_wdbc_data(csv_path=None):
    """Backward compatibility wrapper for WDBC."""
    return load_biomedical_dataset("wdbc")

def get_train_test_data(X, y, test_size=0.2, random_state=42):
    """
    Performs stratified train-test split and fits StandardScaler strictly on training set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_kfold_splits(X, y, n_splits=5, random_state=42):
    """Returns stratified K-fold splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))
