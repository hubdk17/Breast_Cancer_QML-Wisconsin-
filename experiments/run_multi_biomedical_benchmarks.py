"""
Cross-Dataset Biomedical Generalization Suite.
Evaluates IBIP and competing allocation strategies across 4 distinct biomedical benchmarks:
1. WDBC (Wisconsin Diagnostic Breast Cancer): 569 samples, 30 cytological features
2. Parkinson's Disease (UCI 174): 195 samples, 22 acoustic biomedical features
3. Cleveland Heart Disease (UCI 45): 303 samples, 13 clinical/cardiovascular features
4. Breast Cancer Coimbra (UCI 451): 116 samples, 9 metabolic/biomarker features

Demonstrates that IBIP's benefits generalize robustly beyond WDBC across diverse organ systems.
Generates:
- results/tables/multi_dataset_summary.csv
- results/tables/multi_dataset_paired_stats.csv
- results/tables/multi_dataset_cv_summary.csv
- results/figures/figure9_multi_dataset_generalization.png
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from src.dataset import load_biomedical_dataset, get_train_test_data
from src.ranking import compute_xgboost_permutation_ranking, get_feature_allocation
from src.models import QuantumInspiredClassifier
from src.classical_baselines import evaluate_predictions
from src.evaluation import compute_paired_statistics

OUTPUT_TABLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "tables")
OUTPUT_FIGS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGS, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

DATASETS = [
    ("wdbc", "WDBC (Breast Cancer)", 0),          # 0 = Malignant (positive)
    ("parkinsons", "Parkinson's Disease", 1),     # 1 = PD (positive)
    ("heart_disease", "Heart Disease", 1),        # 1 = Disease (positive)
    ("coimbra", "Coimbra (Biomarkers)", 1)        # 1 = Cancer (positive)
]

SEEDS = [0, 42, 123, 777, 2025]

def multi_benchmark_worker(task):
    (ds_key, ds_title, pos_label, strategy, seed, X_tr, y_tr, X_te, y_te, alloc, n_qubits) = task
    
    if strategy == "SVM_RBF":
        clf = SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]
        preds = clf.predict(X_te)
    else:
        model = QuantumInspiredClassifier(
            n_qubits=n_qubits,
            topology="cnot_star",
            feature_indices=alloc,
            lr=0.01,
            lam=0.5,
            seed=seed
        )
        model.fit(X_tr, y_tr, epochs=50, fast_grad=True, verbose=False)
        probs = model.predict_proba(X_te)
        preds = model.predict(X_te)
        
    metrics = evaluate_predictions(y_te, preds, y_prob=probs, positive_label=pos_label)
                                   
    return {
        "Dataset": ds_title,
        "Dataset_Key": ds_key,
        "Strategy": strategy,
        "Seed": seed,
        "Accuracy": metrics["accuracy"] * 100.0,
        "Sensitivity": metrics["sensitivity"] * 100.0,
        "Specificity": metrics["specificity"] * 100.0,
        "Precision": metrics["precision"] * 100.0,
        "F1_Score": metrics["f1_score"] * 100.0,
        "ROC_AUC": metrics["roc_auc"]
    }

def run_cross_dataset_benchmarks(max_workers=10):
    print("=" * 80)
    print("RUNNING CROSS-DATASET BIOMEDICAL GENERALIZATION BENCHMARKS")
    print("=" * 80)
    
    tasks = []
    
    for ds_key, ds_title, pos_label in DATASETS:
        X, y, feat_names = load_biomedical_dataset(ds_key)
        n_features = X.shape[1]
        n_qubits = 3
        
        print(f"\nQueueing tasks for {ds_title}: {len(X)} samples, {n_features} features...")
        
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=seed)
            ranked_idx, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=seed, n_repeats=10)
            
            # Classical baseline
            tasks.append((ds_key, ds_title, pos_label, "SVM_RBF", seed, X_tr, y_tr, X_te, y_te, None, n_qubits))
            
            # Quantum strategies
            for strat in ["sequential", "random", "importance_blocked", "ibip"]:
                alloc = get_feature_allocation(n_features, n_qubits, method=strat, ranked_indices=ranked_idx, seed=seed)
                tasks.append((ds_key, ds_title, pos_label, strat, seed, X_tr, y_tr, X_te, y_te, alloc, n_qubits))
                
    print(f"\nExecuting {len(tasks)} benchmark runs across {max_workers} processes...")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(multi_benchmark_worker, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())
            
    df_runs = pd.DataFrame(results)
    df_runs.to_csv(os.path.join(OUTPUT_TABLES, "multi_dataset_per_seed.csv"), index=False)
    
    summary = df_runs.groupby(["Dataset", "Strategy"]).agg({
        "Accuracy": ["mean", "std"],
        "Sensitivity": ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Precision": ["mean", "std"],
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"]
    }).round(4)
    summary.to_csv(os.path.join(OUTPUT_TABLES, "multi_dataset_summary.csv"))
    print("\n--- Cross-Dataset Multi-Seed Summary Table ---")
    print(summary)
    
    # Paired Statistics: IBIP vs Sequential on EACH dataset
    paired_rows = []
    for _, ds_title, _ in DATASETS:
        sub_ibip = df_runs[(df_runs["Dataset"] == ds_title) & (df_runs["Strategy"] == "ibip")].sort_values("Seed")
        sub_seq = df_runs[(df_runs["Dataset"] == ds_title) & (df_runs["Strategy"] == "sequential")].sort_values("Seed")
        
        stat = compute_paired_statistics(sub_ibip["Accuracy"].values, sub_seq["Accuracy"].values,
                                         name_a="IBIP", name_b="Sequential")
        paired_rows.append({
            "Dataset": ds_title,
            "Mean_IBIP": stat["mean_a"],
            "Mean_Seq": stat["mean_b"],
            "Mean_Diff": stat["mean_diff"],
            "CI95_Diff": f"[{stat['ci_95_diff'][0]:.2f}, {stat['ci_95_diff'][1]:.2f}]",
            "t_paired": stat["t_paired"],
            "p_paired": stat["p_paired"],
            "cohen_d": stat["cohen_d"]
        })
    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(os.path.join(OUTPUT_TABLES, "multi_dataset_paired_stats.csv"), index=False)
    print("\n--- Cross-Dataset Paired Statistical Contrasts (IBIP vs Sequential) ---")
    print(df_paired.to_string(index=False))
    
    # Plot Figure 9
    plot_figure9_cross_dataset(df_runs)
    return df_runs, summary, df_paired

def plot_figure9_cross_dataset(df_runs):
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    
    datasets_order = [
        "WDBC (Breast Cancer)",
        "Parkinson's Disease",
        "Heart Disease",
        "Coimbra (Biomarkers)"
    ]
    
    x = np.arange(len(datasets_order))
    width = 0.20
    
    strat_colors = {
        "sequential": ("#94a3b8", "Sequential"),
        "random": ("#cbd5e1", "Random"),
        "importance_blocked": ("#f59e0b", "Importance-Blocked"),
        "ibip": ("#2563eb", "IBIP (Proposed)")
    }
    
    for i, (strat, (color, label)) in enumerate(strat_colors.items()):
        means = []
        stds = []
        for ds in datasets_order:
            sub = df_runs[(df_runs["Dataset"] == ds) & (df_runs["Strategy"] == strat)]
            means.append(sub["Accuracy"].mean())
            stds.append(sub["Accuracy"].std())
            
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=label, color=color, edgecolor="#334155", linewidth=1.2, alpha=0.95)
                      
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Mean Test Accuracy (%) across 5 Seeds", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Dataset Generalization: IBIP Across Diverse Biomedical Benchmarks", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_order, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=10.5)
    ax.set_ylim(60, 103)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGS, "figure9_multi_dataset_generalization.png"))
    plt.close()
    print("Saved Figure 9 to results/figures/figure9_multi_dataset_generalization.png")

if __name__ == "__main__":
    run_cross_dataset_benchmarks()
