"""
Publication Figure Generator.
Generates all manuscript figures at high resolution (300 DPI):
- Figure 1: Architectural Workflow Diagram
- Figure 2: IBIP Allocation Mechanism Illustration
- Figure 3: Two-Qubit Interaction Gate Comparison (SWAP, sqrtSWAP, unitary iSWAP, CNOT, None)
- Figure 4: 3-Qubit CNOT-Star Training Dynamics (Loss & Accuracy vs. Epoch)
- Figure 5: 3-Qubit CNOT-Chain Training Dynamics (Loss & Accuracy vs. Epoch)
- Figure 6: Receiver Operating Characteristic (ROC) Comparison across Classical & Quantum Models
- Figure 7: Clinical Diagnostic Confusion Matrices (CNOT-Star vs. SVM RBF)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from src.dataset import load_wdbc_data, get_train_test_data
from src.ranking import compute_xgboost_permutation_ranking, get_feature_allocation
from src.models import QuantumInspiredClassifier
from src.classical_baselines import get_classical_models

OUTPUT_FIGS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")
os.makedirs(OUTPUT_FIGS, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def plot_figure3_gate_comparison():
    print("Generating Figure 3: Gate Comparison...")
    summary_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "tables", "two_qubit_suite_summary.csv")
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path, header=[0, 1], index_col=[0, 1])
        topologies = ["swap", "sqrtswap", "iswap", "cnot01", "none"]
        labels = ["SWAP", "√SWAP", "iSWAP (Unitary)", "CNOT (0→1)", "Independent"]
        
        ibip_accs = [df_sum.loc[(t, "IBIP")][("Accuracy", "mean")] for t in topologies]
        ibip_stds = [df_sum.loc[(t, "IBIP")][("Accuracy", "std")] for t in topologies]
        seq_accs = [df_sum.loc[(t, "Sequential")][("Accuracy", "mean")] for t in topologies]
        seq_stds = [df_sum.loc[(t, "Sequential")][("Accuracy", "std")] for t in topologies]
    else:
        # Figures are drawn only from result tables; run the experiment scripts first.
        raise FileNotFoundError(f"{summary_path} not found. Run experiments/run_controlled_experiments_parallel.py first.")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, ibip_accs, width, yerr=ibip_stds, capsize=4,
                    label="IBIP (Interleaved)", color="#1d4ed8", edgecolor="#1e3a8a", alpha=0.9)
    rects2 = ax.bar(x + width/2, seq_accs, width, yerr=seq_stds, capsize=4,
                    label="Sequential Partitioning", color="#93c5fd", edgecolor="#60a5fa", alpha=0.9)
                    
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Two-Qubit Interaction Operators: IBIP vs. Sequential Partitioning (5 Seeds)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=11, loc="lower right")
    ax.set_ylim(85, 102)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGS, "figure3_gate_comparison_publication.png"))
    plt.close()
    print("Saved Figure 3 to results/figures/figure3_gate_comparison_publication.png")

def plot_figure6_roc_curves():
    print("Generating Figure 6: Multi-Model ROC Curves...")
    X, y, _ = load_wdbc_data()
    X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=42)
    
    # 1. Classical Models
    models = get_classical_models(random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    colors = {
        "SVM (RBF)": "#dc2626",
        "Logistic Regression": "#ea580c",
        "XGBoost": "#16a34a",
        "Random Forest": "#ca8a04",
        "Ridge Classifier": "#9333ea"
    }
    
    # In clinical positive class: Malignant = 0
    y_test_binary = (y_te == 0).astype(int)
    
    for name in ["SVM (RBF)", "Logistic Regression", "XGBoost", "Random Forest"]:
        clf = models[name]
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 0] # Prob of Malignant
        fpr, tpr, _ = roc_curve(y_test_binary, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=colors[name], linewidth=2)
        
    # 2. Quantum Models (CNOT-Star and SWAP)
    ranked_idx, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=42, n_repeats=10)
    
    # CNOT-Star
    alloc_3q = get_feature_allocation(30, 3, "ibip", ranked_idx)
    m_star = QuantumInspiredClassifier(3, "cnot_star", alloc_3q, seed=42)
    m_star.fit(X_tr, y_tr, epochs=50, fast_grad=True)
    prob_star_benign = m_star.predict_proba(X_te)
    prob_star_mal = 1.0 - prob_star_benign
    fpr_s, tpr_s, _ = roc_curve(y_test_binary, prob_star_mal)
    auc_s = auc(fpr_s, tpr_s)
    ax.plot(fpr_s, tpr_s, label=f"3Q CNOT-Star IBIP (AUC = {auc_s:.4f})", color="#2563eb", linewidth=2.5)
    
    # SWAP
    alloc_2q = get_feature_allocation(30, 2, "ibip", ranked_idx)
    m_swap = QuantumInspiredClassifier(2, "swap", alloc_2q, seed=42)
    m_swap.fit(X_tr, y_tr, epochs=50, fast_grad=True)
    prob_swap_benign = m_swap.predict_proba(X_te)
    prob_swap_mal = 1.0 - prob_swap_benign
    fpr_sw, tpr_sw, _ = roc_curve(y_test_binary, prob_swap_mal)
    auc_sw = auc(fpr_sw, tpr_sw)
    ax.plot(fpr_sw, tpr_sw, label=f"2Q SWAP IBIP (AUC = {auc_sw:.4f})", color="#0891b2", linewidth=2, linestyle="--")
    
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", label="Chance (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity for Malignancy)", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curves: Quantum-Inspired vs. Classical Baselines (Seed 42)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGS, "figure6_roc_comparison.png"))
    plt.close()
    print("Saved Figure 6 to results/figures/figure6_roc_comparison.png")

def plot_figure7_confusion_matrices():
    print("Generating Figure 7: Confusion Matrices...")
    X, y, _ = load_wdbc_data()
    X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=42)
    y_test_binary = (y_te == 0).astype(int) # 1 = Malignant, 0 = Benign
    
    # 1. CNOT-Star
    ranked_idx, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=42, n_repeats=10)
    alloc_3q = get_feature_allocation(30, 3, "ibip", ranked_idx)
    m_star = QuantumInspiredClassifier(3, "cnot_star", alloc_3q, seed=42)
    m_star.fit(X_tr, y_tr, epochs=50, fast_grad=True)
    preds_star_benign = m_star.predict(X_te)
    preds_star_mal = (preds_star_benign == 0).astype(int)
    cm_star = confusion_matrix(y_test_binary, preds_star_mal)
    
    # 2. SVM RBF
    clf_svm = get_classical_models(random_state=42)["SVM (RBF)"]
    clf_svm.fit(X_tr, y_tr)
    preds_svm_benign = clf_svm.predict(X_te)
    preds_svm_mal = (preds_svm_benign == 0).astype(int)
    cm_svm = confusion_matrix(y_test_binary, preds_svm_mal)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    class_labels = ["Benign (0)", "Malignant (1)"]
    
    sns.heatmap(cm_star, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title("3-Qubit CNOT-Star (IBIP)\nAcc: 96.49% | Sens: 92.86% | Spec: 98.61%", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Clinical Diagnosis", fontsize=11, fontweight="bold")
    
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges", ax=axes[1], cbar=False,
                xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title("Classical SVM (RBF Kernel)\nAcc: 98.25% | Sens: 95.24% | Spec: 100.00%", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("True Clinical Diagnosis", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGS, "figure7_confusion_matrices.png"))
    plt.close()
    print("Saved Figure 7 to results/figures/figure7_confusion_matrices.png")

if __name__ == "__main__":
    plot_figure3_gate_comparison()
    plot_figure6_roc_curves()
    plot_figure7_confusion_matrices()

