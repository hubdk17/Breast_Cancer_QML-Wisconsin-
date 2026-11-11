"""
Dataset loading and preprocessing module for Wisconsin Diagnostic Breast Cancer (WDBC).
Ensures strict prevention of data leakage: scalers and feature selectors are fitted
strictly on training folds/splits.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wdbc.csv")

def load_wdbc_data(csv_path=DATA_PATH):
    """
    Loads WDBC dataset.
    Returns:
        X (np.ndarray): 569 x 30 feature matrix.
        y (np.ndarray): 569 binary targets (1 = Benign [n=357], 0 = Malignant [n=212]).
        feature_names (np.ndarray): array of 30 feature names.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ["target", "diagnosis", "id"]]
        X = df[feature_cols].values.astype(np.float64)
        y = df["target"].values.astype(np.int64)
        feature_names = np.array(feature_cols)
    else:
        bunch = load_breast_cancer(as_frame=True)
        X = bunch.data.values.astype(np.float64)
        y = bunch.target.values.astype(np.int64)
        feature_names = np.array(bunch.feature_names)
    
    return X, y, feature_names

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
    """
    Returns stratified K-fold splits.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))
