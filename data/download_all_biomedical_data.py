"""
Download and prepare all 4 biomedical benchmark datasets:
1. WDBC (Wisconsin Diagnostic Breast Cancer) - 569 samples, 30 features
2. Parkinson's Disease (UCI 174) - 195 samples, 22 features
3. Cleveland Heart Disease (UCI 45) - 303 samples, 13 features
4. Breast Cancer Coimbra (UCI 451) - 116 samples, 9 features
"""

import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_breast_cancer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def prepare_wdbc():
    csv_path = os.path.join(DATA_DIR, "wdbc.csv")
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # 0 = Malignant (212), 1 = Benign (357)
    df["target"] = bunch.target
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
    df.to_csv(csv_path, index=False)
    print(f"[WDBC] Saved: {csv_path} | Shape: {df.shape} | Target counts: {np.bincount(df['target'])}")

def prepare_parkinsons():
    csv_path = os.path.join(DATA_DIR, "parkinsons.csv")
    p = fetch_ucirepo(id=174)
    X = p.data.features.copy()
    y = p.data.targets["status"].values.astype(int) # 1 = PD (147), 0 = Healthy (48)
    df = X.copy()
    df["target"] = y
    df.to_csv(csv_path, index=False)
    print(f"[Parkinsons] Saved: {csv_path} | Shape: {df.shape} | Target counts: {np.bincount(y)}")

def prepare_heart_disease():
    csv_path = os.path.join(DATA_DIR, "heart_disease.csv")
    h = fetch_ucirepo(id=45)
    X = h.data.features.copy()
    # Impute missing values with median
    X = X.fillna(X.median())
    y = (h.data.targets["num"].values > 0).astype(int) # 1 = Heart Disease (139), 0 = Healthy (164)
    df = X.copy()
    df["target"] = y
    df.to_csv(csv_path, index=False)
    print(f"[Heart Disease] Saved: {csv_path} | Shape: {df.shape} | Target counts: {np.bincount(y)}")

def prepare_coimbra():
    csv_path = os.path.join(DATA_DIR, "coimbra.csv")
    c = fetch_ucirepo(id=451)
    X = c.data.features.copy()
    y = (c.data.targets.values.ravel() == 2).astype(int) # 1 = Breast Cancer (64), 0 = Healthy (52)
    df = X.copy()
    df["target"] = y
    df.to_csv(csv_path, index=False)
    print(f"[Coimbra] Saved: {csv_path} | Shape: {df.shape} | Target counts: {np.bincount(y)}")

if __name__ == "__main__":
    prepare_wdbc()
    prepare_parkinsons()
    prepare_heart_disease()
    prepare_coimbra()

