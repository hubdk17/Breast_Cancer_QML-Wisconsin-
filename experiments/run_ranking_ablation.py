"""
Evaluation of Alternative Feature Ranking Methods for IBIP.
Addresses Reviewer Comment 4:
Evaluates whether IBIP's balanced interleaving advantages persist across distinct
feature selection and ranking criteria:
1. XGBoost Permutation Importance (Default non-linear gradient-boosted criterion)
2. Random Forest Permutation Importance (Ensemble tree criterion)
3. Mutual Information (Non-parametric information-theoretic criterion)
4. ANOVA F-statistic (Parametric linear variance-ratio criterion)

Outputs:
- results/tables/ranking_method_ablation.csv
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.dataset import load_wdbc_data, get_train_test_data
from src.ranking import compute_feature_ranking, get_feature_allocation
from src.models import QuantumInspiredClassifier
from src.classical_baselines import evaluate_predictions

OUTPUT_TABLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "tables")
os.makedirs(OUTPUT_TABLES, exist_ok=True)

SEEDS = [0, 42, 123, 777, 2025, 1, 2, 3, 4, 5]
RANKERS = ["xgboost", "random_forest", "mutual_info", "anova_f"]

def ranking_eval_worker(task):
    (ranker_name, seed, X_tr, y_tr, X_te, y_te, alloc) = task
    
    model = QuantumInspiredClassifier(
        n_qubits=3,
        topology="cnot_star",
        feature_indices=alloc,
        lr=0.01,
        lam=0.5,
        seed=seed
    )
    model.fit(X_tr, y_tr, epochs=50, fast_grad=True, verbose=False)
    
    probs = model.predict_proba(X_te)
    preds = model.predict(X_te)
    
    metrics = evaluate_predictions(y_te, preds, y_prob=probs, positive_label=0)
    
    return {
        "Ranker": ranker_name,
        "Seed": seed,
        "Accuracy": metrics["accuracy"] * 100.0,
        "Sensitivity": metrics["sensitivity"] * 100.0,
        "Specificity": metrics["specificity"] * 100.0,
        "Precision": metrics["precision"] * 100.0,
        "F1_Score": metrics["f1_score"] * 100.0,
        "ROC_AUC": metrics["roc_auc"]
    }

def run_ranking_ablation(max_workers=10):
    print("=" * 80)
    print("RUNNING FEATURE RANKING ABLATION ACROSS 10 SEEDS")
    print("=" * 80)
    
    X, y, feat_names = load_wdbc_data()
    n_features = X.shape[1]
    
    tasks = []
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=seed)
        
        for ranker in RANKERS:
            ranked_indices, _ = compute_feature_ranking(X_tr, y_tr, method=ranker, random_state=seed, n_repeats=10)
            alloc = get_feature_allocation(n_features, n_qubits=3, method="ibip", ranked_indices=ranked_indices, seed=seed)
            tasks.append((ranker, seed, X_tr, y_tr, X_te, y_te, alloc))
            
    print(f"Queued {len(tasks)} ranking ablation tasks across {max_workers} processes...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(ranking_eval_worker, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
            
    df_runs = pd.DataFrame(results)
    
    summary = df_runs.groupby("Ranker").agg({
        "Accuracy": ["mean", "std"],
        "Sensitivity": ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Precision": ["mean", "std"],
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"]
    }).round(4)
    
    print("\n--- Feature Ranking Method Ablation Summary (10 Seeds) ---")
    print(summary)
    
    summary.to_csv(os.path.join(OUTPUT_TABLES, "ranking_method_ablation.csv"))
    df_runs.to_csv(os.path.join(OUTPUT_TABLES, "ranking_method_runs.csv"), index=False)
    print(f"\nSaved results to {os.path.join(OUTPUT_TABLES, 'ranking_method_ablation.csv')}")
    return summary

if __name__ == "__main__":
    run_ranking_ablation()
