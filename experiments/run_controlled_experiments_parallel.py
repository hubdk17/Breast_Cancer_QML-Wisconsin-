"""
Parallelized Controlled Like-for-Like Experiments for Quantum-Inspired Variational Classifiers.
Utilizes Python's multiprocessing (ProcessPoolExecutor) to evaluate all architectures concurrently.

Addresses Reviewer 2, Reviewer 3, and Editorial Comments 1, 2, 3, 5, 7:
1. Controlled comparison: 2-qubit and 3-qubit suites evaluated independently.
2. Every topology evaluated under BOTH IBIP and Sequential allocations.
3. Every architecture replicated across the SAME 5 seeds (0, 42, 123, 777, 2025).
4. Full 5-Fold Stratified Cross-Validation on all topologies.
5. Full clinical metrics: Accuracy, Sensitivity (Malignant Recall), Specificity (Benign Recall),
   Precision, F1-Score, ROC-AUC, and full confusion matrix.
6. Paired statistical analysis: Paired t-test, Wilcoxon signed-rank test, Cohen's d_z, 95% CI.
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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [0, 42, 123, 777, 2025]

def train_and_eval_worker(task):
    """
    Worker function executed in parallel worker processes.
    """
    (suite, n_qubits, topology, alloc_name, alloc_indices,
     run_type, run_id, X_tr, y_tr, X_te, y_te, epochs, lr, lam, seed) = task
     
    model = QuantumInspiredClassifier(
        n_qubits=n_qubits,
        topology=topology,
        feature_indices=alloc_indices,
        lr=lr,
        lam=lam,
        seed=seed
    )
    model.fit(X_tr, y_tr, epochs=epochs, fast_grad=True, verbose=False)
    
    probs = model.predict_proba(X_te)
    preds = model.predict(X_te)
    
    metrics = evaluate_predictions(y_te, preds, y_prob=probs, positive_label=0) # 0 = Malignant
    
    res = {
        "Suite": suite,
        "N_Qubits": n_qubits,
        "Topology": topology,
        "Allocation": alloc_name,
        "Run_Type": run_type,
        "Run_ID": run_id,
        "Accuracy": metrics["accuracy"] * 100.0,
        "Sensitivity": metrics["sensitivity"] * 100.0,
        "Specificity": metrics["specificity"] * 100.0,
        "Precision": metrics["precision"] * 100.0,
        "F1_Score": metrics["f1_score"] * 100.0,
        "ROC_AUC": metrics["roc_auc"],
        "TP": metrics["tp"], "TN": metrics["tn"],
        "FP": metrics["fp"], "FN": metrics["fn"]
    }
    return res

def run_controlled_multiseed_parallel(n_qubits, topologies, suite_name, max_workers=10):
    print("=" * 80)
    print(f"STARTING PARALLEL MULTI-SEED SUITE: {suite_name} (N_QUBITS={n_qubits})")
    print("=" * 80)
    
    X, y, _ = load_wdbc_data()
    tasks = []
    
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=seed)
        ranked_idx, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=seed, n_repeats=10)
        
        alloc_ibip = get_feature_allocation(30, n_qubits, "ibip", ranked_idx)
        alloc_seq = get_feature_allocation(30, n_qubits, "sequential")
        
        for top in topologies:
            tasks.append((suite_name, n_qubits, top, "IBIP", alloc_ibip, "Seed", seed,
                          X_tr, y_tr, X_te, y_te, 50, 0.01, 0.5, seed))
            tasks.append((suite_name, n_qubits, top, "Sequential", alloc_seq, "Seed", seed,
                          X_tr, y_tr, X_te, y_te, 50, 0.01, 0.5, seed))
                          
    print(f"Queued {len(tasks)} model training tasks. Executing across {max_workers} processes...")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_and_eval_worker, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
            
    df_runs = pd.DataFrame(results)
    df_runs.to_csv(os.path.join(OUTPUT_DIR, f"{suite_name.lower()}_per_seed.csv"), index=False)
    
    summary = df_runs.groupby(["Topology", "Allocation"]).agg({
        "Accuracy": ["mean", "std"],
        "Sensitivity": ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Precision": ["mean", "std"],
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"]
    }).round(4)
    summary.to_csv(os.path.join(OUTPUT_DIR, f"{suite_name.lower()}_summary.csv"))
    print("\n--- Seed-Averaged Summary ---")
    print(summary)
    
    # Paired stats
    paired_rows = []
    for top in topologies:
        sub_ibip = df_runs[(df_runs["Topology"] == top) & (df_runs["Allocation"] == "IBIP")].sort_values("Run_ID")
        sub_seq = df_runs[(df_runs["Topology"] == top) & (df_runs["Allocation"] == "Sequential")].sort_values("Run_ID")
        
        stat = compute_paired_statistics(sub_ibip["Accuracy"].values, sub_seq["Accuracy"].values,
                                         name_a="IBIP", name_b="Sequential")
        paired_rows.append({
            "Topology": top,
            "Mean_IBIP": stat["mean_a"],
            "Std_IBIP": stat["std_a"],
            "Mean_Seq": stat["mean_b"],
            "Std_Seq": stat["std_b"],
            "Mean_Diff": stat["mean_diff"],
            "CI95_Diff": f"[{stat['ci_95_diff'][0]:.2f}, {stat['ci_95_diff'][1]:.2f}]",
            "t_paired": stat["t_paired"],
            "p_paired": stat["p_paired"],
            "wilcoxon_stat": stat["wilcoxon_stat"],
            "p_wilcoxon": stat["p_wilcoxon"],
            "cohen_d": stat["cohen_d"],
            "t_welch": stat["t_welch"],
            "p_welch": stat["p_welch"]
        })
    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(os.path.join(OUTPUT_DIR, f"{suite_name.lower()}_paired_stats.csv"), index=False)
    print("\n--- Paired Stats (IBIP vs Sequential) ---")
    print(df_paired[["Topology", "Mean_IBIP", "Mean_Seq", "Mean_Diff", "CI95_Diff", "t_paired", "p_paired", "cohen_d"]])
    return df_runs, summary, df_paired

def run_controlled_kfold_parallel(n_qubits, topologies, suite_name, n_splits=5, max_workers=10):
    print("=" * 80)
    print(f"STARTING PARALLEL {n_splits}-FOLD CV SUITE: {suite_name} (N_QUBITS={n_qubits})")
    print("=" * 80)
    
    X, y, _ = load_wdbc_data()
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tasks = []
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        ranked_idx, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=42, n_repeats=10)
        alloc_ibip = get_feature_allocation(30, n_qubits, "ibip", ranked_idx)
        alloc_seq = get_feature_allocation(30, n_qubits, "sequential")
        
        for top in topologies:
            tasks.append((suite_name, n_qubits, top, "IBIP", alloc_ibip, "Fold", fold,
                          X_tr, y_tr, X_te, y_te, 50, 0.01, 0.5, 42 + fold))
            tasks.append((suite_name, n_qubits, top, "Sequential", alloc_seq, "Fold", fold,
                          X_tr, y_tr, X_te, y_te, 50, 0.01, 0.5, 42 + fold))
                          
    print(f"Queued {len(tasks)} CV fold tasks. Executing across {max_workers} processes...")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_and_eval_worker, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
            
    df_cv = pd.DataFrame(results)
    df_cv.to_csv(os.path.join(OUTPUT_DIR, f"{suite_name.lower()}_cv_per_fold.csv"), index=False)
    
    summary_cv = df_cv.groupby(["Topology", "Allocation"]).agg({
        "Accuracy": ["mean", "std"],
        "Sensitivity": ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Precision": ["mean", "std"],
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"]
    }).round(4)
    summary_cv.to_csv(os.path.join(OUTPUT_DIR, f"{suite_name.lower()}_cv_summary.csv"))
    print("\n--- Cross-Validation Summary ---")
    print(summary_cv)
    return df_cv, summary_cv

if __name__ == "__main__":
    topologies_2q = ["swap", "sqrtswap", "iswap", "cnot01", "none"]
    run_controlled_multiseed_parallel(2, topologies_2q, "Two_Qubit_Suite")
    run_controlled_kfold_parallel(2, topologies_2q, "Two_Qubit_Suite")
    
    topologies_3q = ["cnot_star", "cnot_chain", "cnot_ring", "swap_chain", "swap_star", "full_mesh", "none"]
    run_controlled_multiseed_parallel(3, topologies_3q, "Three_Qubit_Suite")
    run_controlled_kfold_parallel(3, topologies_3q, "Three_Qubit_Suite")

