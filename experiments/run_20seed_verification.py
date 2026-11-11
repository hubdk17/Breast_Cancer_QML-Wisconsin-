"""
20-Seed Replication and Matched Paired Statistical Verification Suite.
Addresses Reviewer Comment 3:
Evaluates IBIP CNOT-Star vs. Sequential CNOT-Star across 20 distinct random seeds (0 to 19):
- Computes per-seed matched differences (IBIP minus Sequential).
- Computes paired Student's t-test, Wilcoxon signed-rank test, Cohen's d_z effect size, and 95% CIs.
- Demonstrates statistical robustness beyond the initial 5-seed sample.

Outputs:
- results/tables/cnot_star_vs_seq_20seeds.csv
- results/tables/paired_stats_20seeds.csv
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.dataset import load_wdbc_data, get_train_test_data
from src.ranking import compute_xgboost_permutation_ranking, get_feature_allocation
from src.models import QuantumInspiredClassifier
from src.classical_baselines import evaluate_predictions
from src.evaluation import compute_paired_statistics

OUTPUT_TABLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "tables")
os.makedirs(OUTPUT_TABLES, exist_ok=True)

SEEDS = list(range(20)) # 20 seeds: 0, 1, 2, ..., 19

def seed_eval_worker(task):
    (strategy, seed, X_tr, y_tr, X_te, y_te, alloc) = task
    
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
        "Strategy": strategy,
        "Seed": seed,
        "Accuracy": metrics["accuracy"] * 100.0,
        "Sensitivity": metrics["sensitivity"] * 100.0,
        "Specificity": metrics["specificity"] * 100.0,
        "Precision": metrics["precision"] * 100.0,
        "F1_Score": metrics["f1_score"] * 100.0,
        "ROC_AUC": metrics["roc_auc"]
    }

def run_20seed_verification(max_workers=10):
    print("=" * 80)
    print("RUNNING 20-SEED VERIFICATION SUITE: IBIP VS. SEQUENTIAL (CNOT-STAR)")
    print("=" * 80)
    
    X, y, _ = load_wdbc_data()
    n_features = X.shape[1]
    
    tasks = []
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=seed)
        ranked_indices, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=seed, n_repeats=10)
        
        alloc_ibip = get_feature_allocation(n_features, 3, method="ibip", ranked_indices=ranked_indices, seed=seed)
        alloc_seq = get_feature_allocation(n_features, 3, method="sequential")
        
        tasks.append(("IBIP", seed, X_tr, y_tr, X_te, y_te, alloc_ibip))
        tasks.append(("Sequential", seed, X_tr, y_tr, X_te, y_te, alloc_seq))
        
    print(f"Queued {len(tasks)} runs across {max_workers} processes...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(seed_eval_worker, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
            
    df_runs = pd.DataFrame(results)
    df_runs.to_csv(os.path.join(OUTPUT_TABLES, "cnot_star_vs_seq_20seeds.csv"), index=False)
    
    # Summary
    summary = df_runs.groupby("Strategy").agg({
        "Accuracy": ["mean", "std"],
        "Sensitivity": ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Precision": ["mean", "std"],
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"]
    }).round(4)
    print("\n--- 20-Seed Performance Summary (Mean ± SD) ---")
    print(summary)
    
    # Paired Statistics across 20 matched seeds
    df_ibip = df_runs[df_runs["Strategy"] == "IBIP"].sort_values("Seed")
    df_seq = df_runs[df_runs["Strategy"] == "Sequential"].sort_values("Seed")
    
    stat = compute_paired_statistics(df_ibip["Accuracy"].values, df_seq["Accuracy"].values,
                                     name_a="IBIP", name_b="Sequential")
                                     
    df_stat = pd.DataFrame([{
        "Comparison": "IBIP vs. Sequential (20 Seeds)",
        "N_Seeds": 20,
        "Mean_IBIP": stat["mean_a"],
        "Mean_Seq": stat["mean_b"],
        "Mean_Diff": stat["mean_diff"],
        "CI95_Diff": f"[{stat['ci_95_diff'][0]:.2f}, {stat['ci_95_diff'][1]:.2f}]",
        "t_paired": stat["t_paired"],
        "p_paired": stat["p_paired"],
        "wilcoxon_W": stat["wilcoxon_stat"],
        "p_wilcoxon": stat["p_wilcoxon"],
        "cohen_d": stat["cohen_d"]
    }])
    df_stat.to_csv(os.path.join(OUTPUT_TABLES, "paired_stats_20seeds.csv"), index=False)
    print("\n--- 20-Seed Matched Paired Statistics ---")
    print(df_stat.to_string(index=False))
    
    return df_runs, summary, df_stat

if __name__ == "__main__":
    run_20seed_verification()
