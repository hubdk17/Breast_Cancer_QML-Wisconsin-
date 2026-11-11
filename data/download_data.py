"""
Script to download and verify the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
Source: UCI Machine Learning Repository (ID: 17) and Scikit-Learn mirror.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

def download_and_save_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "wdbc.csv")
    
    # Load from scikit-learn (standard validated mirror of UCI WDBC)
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    
    # Rename target column to diagnosis and add explicit class mapping
    # Note: in scikit-learn, 0 = malignant, 1 = benign
    df["target"] = bunch.target
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
    
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully saved to: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Class counts:\n{df['diagnosis'].value_counts()}")
    print(f"Target (0=Malignant, 1=Benign) counts:\n{df['target'].value_counts()}")
    
    return df

if __name__ == "__main__":
    download_and_save_data()

