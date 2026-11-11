"""
Comprehensive Classical Baseline Evaluation.
Evaluates SVM (RBF & Linear), XGBoost, Ridge Classifier, Logistic Regression, and Random Forest
under identical conditions:
1. 5 Random Seeds (0, 42, 123, 777, 2025) with 80:20 stratified split.
2. 5-Fold Stratified Cross-Validation.
All preprocessing (StandardScaler) is strictly fitted on training splits to prevent leakage.
Reports: Accuracy, Sensitivity (Malignant Recall), Specificity (Benign), Precision, F1, ROC-AUC,
Confusion Matrix, Mean, SD, and 95% Confidence Intervals.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.dataset import load_wdbc_data, get_train_test_data, get_kfold_splits
from src.classical_baselines import get_classical_models, evaluate_predictions

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_classical_multiseed(seeds=[0, 42, 123, 777, 2025]):
    print("=" * 80)
    print("RUNNING CLASSICAL BASELINES: 5-SEED EVALUATION (80:20 STRATIFIED SPLIT)")
    print("=" * 80)
    
    X, y, _ = load_wdbc_data()
    all_records = []
    
    for seed in seeds:
        X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=seed)
        models = get_classical_models(random_state=seed)
        
        for name, clf in models.items():
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            
            # Predict probabilities if available
            y_prob = None
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_te)
            elif hasattr(clf, "decision_function"):
                df = clf.decision_function(X_te)
                # Map decision function to pseudo-probability via sigmoid
                y_prob = 1.0 / (1.0 + np.exp(-df))
                
            metrics = evaluate_predictions(y_te, y_pred, y_prob=y_prob, positive_label=0) # 0 = Malignant
            
            record = {
                "Model": name,
                "Seed": seed,
                "Accuracy": metrics["accuracy"] * 100.0,
                "Sensitivity": metrics["sensitivity"] * 100.0, # Malignant recall
                "Specificity": metrics["specificity"] * 100.0, # Benign recall
                "Precision": metrics["precision"] * 100.0,
                "F1_Score": metrics["f1_score"] * 100.0,
                "ROC_AUC": metrics["roc_auc"],
                "TP": metrics["tp"], "TN": metrics["tn"],
                "FP": metrics["fp"], "FN": metrics["fn"]
            }
            all_records.append(record)
            
    df_runs = pd.DataFrame(all_records)
    df_runs.to_csv(os.path.join(OUTPUT_DIR, "classical_baselines_per_seed.csv"), index=False)
    
    # Summary Table across seeds
    summary = df_runs.groupby("Model").agg({
        "Accuracy": ["mean", "std"],
        "Sensitivity": ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Precision": ["mean", "std"],
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"]
    }).round(4)
    
    summary.to_csv(os.path.join(OUTPUT_DIR, "classical_baselines_multiseed_summary.csv"))
    print("\n--- 5-Seed Summary Table (Mean ± SD) ---")
    print(summary)
    return df_runs, summary

def run_classical_kfold(n_splits=5, random_state=42):
    print("\n" + "=" * 80)
    print(f"RUNNING CLASSICAL BASELINES: {n_splits}-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 80)
    
    X, y, _ = load_wdbc_data()
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_records = []
    
    models = get_classical_models(random_state=random_state)
    
    for name, clf_template in models.items():
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            y_tr, y_te = y[tr_idx], y[te_idx]
            
            clf = get_classical_models(random_state=random_state)[name]
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            
            y_prob = None
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_te)
            elif hasattr(clf, "decision_function"):
                df = clf.decision_function(X_te)
                y_prob = 1.0 / (1.0 + np.exp(-df))
                
            metrics = evaluate_predictions(y_te, y_pred, y_prob=y_prob, positive_label=0)
            record = {
                "Model": name,
                "Fold": fold,
                "Accuracy": metrics["accuracy"] * 100.0,
                "Sensitivity": metrics["sensitivity"] * 100.0,
                "Specificity": metrics["specificity"] * 100.0,
                "Precision": metrics["precision"] * 100.0,
                "F1_Score": metrics["f1_score"] * 100.0,
                "ROC_AUC": metrics["roc_auc"],
                "TP": metrics["tp"], "TN": metrics["tn"],
                "FP": metrics["fp"], "FN": metrics["fn"]
            }
            all_records.append(record)
            
    df_cv = pd.DataFrame(all_records)
    df_cv.to_csv(os.path.join(OUTPUT_DIR, "classical_baselines_per_fold.csv"), index=False)
    
    summary_cv = df_cv.groupby("Model").agg({
        "Accuracy": ["mean", "std"],
        "Sensitivity": ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Precision": ["mean", "std"],
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"]
    }).round(4)
    
    summary_cv.to_csv(os.path.join(OUTPUT_DIR, "classical_baselines_kfold_summary.csv"))
    print(f"\n--- {n_splits}-Fold CV Summary Table (Mean ± SD) ---")
    print(summary_cv)
    return df_cv, summary_cv

if __name__ == "__main__":
    run_classical_multiseed()
    run_classical_kfold()
