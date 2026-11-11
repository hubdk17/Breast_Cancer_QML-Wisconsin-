"""
Controlled, Like-for-Like Experiments for Quantum-Inspired Variational Classifiers.
Addresses Editorial Comments 1, 2, 3, 5, 7 and Reviewers 2 & 3:
1. Controlled comparison: 2-Qubit and 3-Qubit suites evaluated separately under identical protocols.
2. Every topology evaluated across BOTH IBIP and Sequential allocations.
3. Every architecture replicated across the SAME 5 seeds (0, 42, 123, 777, 2025).
4. Full 5-Fold Stratified Cross-Validation on all topologies.
5. Full clinical metrics: Accuracy, Sensitivity (Malignant Recall), Specificity (Benign),
   Precision, F1-Score, ROC-AUC, Confusion Matrix.
6. Paired statistical testing (Paired t-test, Wilcoxon signed-rank, Cohen's d_z, 95% CI).
"""

import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.dataset import load_wdbc_data, get_train_test_data, get_kfold_splits
from src.ranking import compute_xgboost_permutation_ranking, get_feature_allocation
from src.models import QuantumInspiredClassifier
from src.classical_baselines import evaluate_predictions
from src.evaluation import compute_paired_statistics

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [0, 42, 123, 777, 2025]

def run_suite(n_qubits, topologies, suite_name, epochs=50, lr=0.01, lam=0.5):
    print("=" * 90)
    print(f"RUNNING CONTROLLED SUITE: {suite_name} (N_QUBITS={n_qubits})")
    print("=" * 90)
    
    X, y, feature_names = load_wdbc_data()
    all_seed_records = []
    
    # 1. 5-Seed Evaluation
    for seed in SEEDS:
        print(f"\n>>> Running Seed {seed}...")
        X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=seed)
        
        # Rank features strictly on training split
        ranked_idx, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=seed, n_repeats=10)
        
        alloc_ibip = get_feature_allocation(30, n_qubits, "ibip", ranked_idx)
        alloc_seq = get_feature_allocation(30, n_qubits, "sequential")
        
        for top in topologies:
            for alloc_name, alloc in [("IBIP", alloc_ibip), ("Sequential", alloc_seq)]:
                model = QuantumInspiredClassifier(
                    n_qubits=n_qubits,
                    topology=top,
                    feature_indices=alloc,
                    lr=lr,
                    lam=lam,
                    seed=seed
                )
                model.fit(X_tr, y_tr, epochs=epochs, fast_grad=True, verbose=False)
                
                probs = model.predict_proba(X_te)
                preds = model.predict(X_te)
                
                metrics = evaluate_predictions(y_te, preds, y_prob=probs, positive_label=0) # 0 = Malignant
                
                all_seed_records.append({
                    "Suite": suite_name,
                    "N_Qubits": n_qubits,
                    "Topology": top,
                    "Allocation": alloc_name,
                    "Seed": seed,
                    "Accuracy": metrics["accuracy"] * 100.0,
                    "Sensitivity": metrics["sensitivity"] * 100.0,
                    "Specificity": metrics["specificity"] * 100.0,
                    "Precision": metrics["precision"] * 100.0,
                    "F1_Score": metrics["f1_score"] * 100.0,
                    "ROC_AUC": metrics["roc_auc"],
                    "TP": metrics["tp"], "TN": metrics["tn"],
                    "FP": metrics["fp"], "FN": metrics["fn"]
                })
                
    df_seeds = pd.DataFrame(all_seed_records)
    csv_seed_path = os.path.join(OUTPUT_DIR, f"{suite_name.lower()}_per_seed.csv")
    df_seeds.to_csv(csv_seed_path, index=False)
    
    # Aggregate Summary across seeds
    summary = df_seeds.groupby(["Topology", "Allocation"]).agg({
        "Accuracy": ["mean", "std"],
        "Sensitivity": ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Precision": ["mean", "std"],
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"]
    }).round(4)
    summary.to_csv(os.path.join(OUTPUT_DIR, f"{suite_name.lower()}_summary.csv"))
    print("\n--- Seed-Averaged Summary (Mean ± SD) ---")
    print(summary)
    
    # Paired Statistical Tests between IBIP and Sequential for each topology
    paired_rows = []
    for top in topologies:
        sub_ibip = df_seeds[(df_seeds["Topology"] == top) & (df_seeds["Allocation"] == "IBIP")]
        sub_seq = df_seeds[(df_seeds["Topology"] == top) & (df_seeds["Allocation"] == "Sequential")]
        
        # Sort by seed to ensure matching
        sub_ibip = sub_ibip.sort_values("Seed")
        sub_seq = sub_seq.sort_values("Seed")
        
        stat_res = compute_paired_statistics(
            sub_ibip["Accuracy"].values,
            sub_seq["Accuracy"].values,
            name_a="IBIP", name_b="Sequential"
        )
        paired_rows.append({
            "Topology": top,
            "Mean_IBIP": stat_res["mean_a"],
            "Std_IBIP": stat_res["std_a"],
            "Mean_Seq": stat_res["mean_b"],
            "Std_Seq": stat_res["std_b"],
            "Mean_Diff": stat_res["mean_diff"],
            "CI95_Diff": f"[{stat_res['ci_95_diff'][0]:.2f}, {stat_res['ci_95_diff'][1]:.2f}]",
            "t_paired": stat_res["t_paired"],
            "p_paired": stat_res["p_paired"],
            "wilcoxon_stat": stat_res["wilcoxon_stat"],
            "p_wilcoxon": stat_res["p_wilcoxon"],
            "cohen_d": stat_res["cohen_d"],
            "t_welch": stat_res["t_welch"],
            "p_welch": stat_res["p_welch"],
            "Per_Seed_Diffs": stat_res["differences"]
        })
    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(os.path.join(OUTPUT_DIR, f"{suite_name.lower()}_paired_stats.csv"), index=False)
    print("\n--- Paired Statistical Comparisons (IBIP vs Sequential) ---")
    print(df_paired[["Topology", "Mean_IBIP", "Mean_Seq", "Mean_Diff", "CI95_Diff", "t_paired", "p_paired", "cohen_d"]])
    
    return df_seeds, summary, df_paired

def run_cross_validation_suite(n_qubits, topologies, suite_name, n_splits=5, epochs=50, lr=0.01, lam=0.5):
    print("\n" + "=" * 90)
    print(f"RUNNING {n_splits}-FOLD CROSS-VALIDATION SUITE: {suite_name}")
    print("=" * 90)
    
    X, y, _ = load_wdbc_data()
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_fold_records = []
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n>>> Fold {fold}/{n_splits}...")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        ranked_idx, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=42, n_repeats=10)
        alloc_ibip = get_feature_allocation(30, n_qubits, "ibip", ranked_idx)
        alloc_seq = get_feature_allocation(30, n_qubits, "sequential")
        
        for top in topologies:
            for alloc_name, alloc in [("IBIP", alloc_ibip), ("Sequential", alloc_seq)]:
                model = QuantumInspiredClassifier(
                    n_qubits=n_qubits,
                    topology=top,
                    feature_indices=alloc,
                    lr=lr,
                    lam=lam,
                    seed=42 + fold
                )
                model.fit(X_tr, y_tr, epochs=epochs, fast_grad=True, verbose=False)
                
                probs = model.predict_proba(X_te)
                preds = model.predict(X_te)
                
                metrics = evaluate_predictions(y_te, preds, y_prob=probs, positive_label=0)
                
                all_fold_records.append({
                    "Suite": suite_name,
                    "N_Qubits": n_qubits,
                    "Topology": top,
                    "Allocation": alloc_name,
                    "Fold": fold,
                    "Accuracy": metrics["accuracy"] * 100.0,
                    "Sensitivity": metrics["sensitivity"] * 100.0,
                    "Specificity": metrics["specificity"] * 100.0,
                    "Precision": metrics["precision"] * 100.0,
                    "F1_Score": metrics["f1_score"] * 100.0,
                    "ROC_AUC": metrics["roc_auc"],
                    "TP": metrics["tp"], "TN": metrics["tn"],
                    "FP": metrics["fp"], "FN": metrics["fn"]
                })
                
    df_cv = pd.DataFrame(all_fold_records)
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
    print("\n--- Cross-Validation Summary (Mean ± SD) ---")
    print(summary_cv)
    return df_cv, summary_cv

if __name__ == "__main__":
    # 2-Qubit Topologies: SWAP, sqrtSWAP, iSWAP, CNOT, Independent
    topologies_2q = ["swap", "sqrtswap", "iswap", "cnot01", "none"]
    run_suite(n_qubits=2, topologies=topologies_2q, suite_name="Two_Qubit_Suite")
    run_cross_validation_suite(n_qubits=2, topologies=topologies_2q, suite_name="Two_Qubit_Suite")
    
    # 3-Qubit Topologies: CNOT-Star, CNOT-Chain, CNOT-Ring, SWAP-Chain, SWAP-Star, Full-Mesh, Independent
    topologies_3q = ["cnot_star", "cnot_chain", "cnot_ring", "swap_chain", "swap_star", "full_mesh", "none"]
    run_suite(n_qubits=3, topologies=topologies_3q, suite_name="Three_Qubit_Suite")
    run_cross_validation_suite(n_qubits=3, topologies=topologies_3q, suite_name="Three_Qubit_Suite")
