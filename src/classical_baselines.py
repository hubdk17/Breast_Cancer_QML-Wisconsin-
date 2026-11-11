"""
Classical Machine Learning Baselines for Breast Cancer Diagnosis.
Evaluates SVM (RBF and Linear), XGBoost, Ridge Classifier, Logistic Regression,
and Random Forest under identical preprocessing, data splits, and seeds as the
quantum-inspired models.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from src.dataset import load_wdbc_data, get_train_test_data

def get_classical_models(random_state=42):
    """
    Returns dictionary of instantiated classical models with explicit configurations.
    """
    models = {
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", probability=True, random_state=random_state),
        "SVM (Linear)": SVC(C=1.0, kernel="linear", probability=True, random_state=random_state),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=random_state, n_jobs=-1
        ),
        "Ridge Classifier": RidgeClassifier(alpha=1.0, random_state=random_state),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1)
    }
    return models

def evaluate_predictions(y_true, y_pred, y_prob=None, positive_label=0):
    """
    Computes comprehensive evaluation metrics.
    In clinical breast cancer diagnostics:
        positive_label = 0 corresponds to Malignant (disease detection)
        negative_label = 1 corresponds to Benign
    
    Args:
        y_true: True binary labels (0=Malignant, 1=Benign)
        y_pred: Predicted binary labels
        y_prob: Predicted probability for positive class (or probability array)
        positive_label: Which label is considered positive (default 0 = Malignant)
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Binary classification metrics treating positive_label as positive
    # If positive_label == 0 (Malignant):
    y_true_binary = (y_true == positive_label).astype(int)
    y_pred_binary = (y_pred == positive_label).astype(int)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary) # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall of positive class (Malignant)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # True negative rate (Benign)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    auc = None
    if y_prob is not None:
        try:
            # If y_prob is 1D probability of Benign (1):
            if np.ndim(y_prob) == 1:
                # If positive_label is 0, prob of 0 is 1 - prob of 1
                prob_pos = (1.0 - y_prob) if positive_label == 0 else y_prob
                auc = roc_auc_score(y_true_binary, prob_pos)
            elif np.ndim(y_prob) == 2:
                auc = roc_auc_score(y_true_binary, y_prob[:, positive_label])
        except Exception:
            auc = np.nan
            
    return {
        "accuracy": acc,
        "sensitivity": sensitivity, # Recall (Malignant)
        "specificity": specificity, # Specificity (Benign)
        "precision": precision,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }
