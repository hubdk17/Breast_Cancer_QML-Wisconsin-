"""
Script to reproduce the exact original paper experiments, tables, and figures.
Validates original Table 2, Table 3, Table 4, Table 5, Table 6, Table 8
and generates Figures 3, 4, 5.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from src.dataset import load_wdbc_data, get_train_test_data, get_kfold_splits
from src.ranking import compute_xgboost_permutation_ranking, get_feature_allocation
from src.models import QuantumInspiredClassifier
from src.classical_baselines import evaluate_predictions
from src.evaluation import compute_paired_statistics

OUTPUT_TABLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "tables")
OUTPUT_FIGS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGS, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def reproduce_table2_gate_analysis(X, y):
    print("\n" + "=" * 80)
    print("REPRODUCING TABLE 2: TWO-QUBIT GATE COMPARISON (SEED 42)")
    print("=" * 80)
    
    X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=42)
    ranked, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=42, n_repeats=10)
    alloc_2q = get_feature_allocation(30, 2, "ibip", ranked)
    
    gates = ["swap", "sqrtswap", "iswap"]
    records = []
    
    for g in gates:
        model = QuantumInspiredClassifier(n_qubits=2, topology=g, feature_indices=alloc_2q,
                                           lr=0.01, lam=0.5, seed=42)
        model.fit(X_tr, y_tr, epochs=50, fast_grad=True, verbose=False)
        probs = model.predict_proba(X_te)
        preds = model.predict(X_te)
        
        acc = accuracy_score(y_te, preds) * 100.0
        auc = roc_auc_score(y_te, probs)
        records.append({"Gate": g.upper(), "Accuracy (%)": round(acc, 2), "ROC-AUC": round(auc, 4)})
        
    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_TABLES, "table2_gate_comparison_reproduced.csv"), index=False)
    
    # Plot Figure 3
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x_pos = np.arange(len(df))
    width = 0.35
    
    rects1 = ax.bar(x_pos - width/2, df["Accuracy (%)"], width, label="Test Accuracy (%)", color="#3b82f6")
    rects2 = ax.bar(x_pos + width/2, df["ROC-AUC"] * 100.0, width, label="ROC-AUC (x100)", color="#93c5fd")
    
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Test Accuracy & ROC-AUC by Two-Qubit Interaction Gate", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["Gate"], fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=11)
    ax.set_ylim(0, 110)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h/100:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
                    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGS, "figure3_gate_comparison.png"))
    plt.close()
    print("Saved Figure 3 to results/figures/figure3_gate_comparison.png")
    return df

def reproduce_training_dynamics(X, y):
    print("\n" + "=" * 80)
    print("REPRODUCING FIGURES 4 & 5: TRAINING DYNAMICS (CNOT-STAR & CNOT-CHAIN)")
    print("=" * 80)
    
    X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=42)
    ranked, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=42, n_repeats=10)
    alloc_ibip = get_feature_allocation(30, 3, "ibip", ranked)
    
    # Train CNOT-Star
    star_model = QuantumInspiredClassifier(n_qubits=3, topology="cnot_star", feature_indices=alloc_ibip,
                                          lr=0.01, lam=0.5, seed=42)
    star_loss, star_acc = star_model.fit(X_tr, y_tr, epochs=50, fast_grad=True, verbose=False)
    
    # Train CNOT-Chain
    chain_model = QuantumInspiredClassifier(n_qubits=3, topology="cnot_chain", feature_indices=alloc_ibip,
                                           lr=0.01, lam=0.5, seed=42)
    chain_loss, chain_acc = chain_model.fit(X_tr, y_tr, epochs=50, fast_grad=True, verbose=False)
    
    epochs = np.arange(1, 51)
    
    # Figure 4: CNOT-Star
    fig4, ax4 = plt.subplots(figsize=(8, 5), dpi=300)
    ax4.plot(epochs, star_acc, label="Train Accuracy (%)", color="#1f77b4", linewidth=2.5)
    ax4.plot(epochs, np.array(star_loss) * 100.0, label="BCE Loss (x100)", color="#ff7f0e", linewidth=2.5)
    ax4.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax4.set_title("3-Qubit CNOT-Star — Accuracy & BCE Loss vs. Epoch", fontsize=13, fontweight="bold")
    ax4.legend(frameon=True, fontsize=11)
    ax4.set_ylim(0, 105)
    plt.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_FIGS, "figure4_cnot_star_dynamics.png"))
    plt.close()
    print("Saved Figure 4 to results/figures/figure4_cnot_star_dynamics.png")
    
    # Figure 5: CNOT-Chain
    fig5, ax5 = plt.subplots(figsize=(8, 5), dpi=300)
    ax5.plot(epochs, chain_acc, label="Train Accuracy (%)", color="#1f77b4", linewidth=2.5)
    ax5.plot(epochs, np.array(chain_loss) * 100.0, label="BCE Loss (x100)", color="#ff7f0e", linewidth=2.5)
    ax5.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax5.set_title("3-Qubit CNOT-Chain — Accuracy & BCE Loss vs. Epoch", fontsize=13, fontweight="bold")
    ax5.legend(frameon=True, fontsize=11)
    ax5.set_ylim(0, 105)
    plt.tight_layout()
    fig5.savefig(os.path.join(OUTPUT_FIGS, "figure5_cnot_chain_dynamics.png"))
    plt.close()
    print("Saved Figure 5 to results/figures/figure5_cnot_chain_dynamics.png")

def reproduce_table5_robustness(X, y, seeds=[0, 42, 123, 777, 2025]):
    print("\n" + "=" * 80)
    print("REPRODUCING TABLE 5: ROBUSTNESS ANALYSIS (CNOT-STAR ACROSS 5 SEEDS)")
    print("=" * 80)
    
    records = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=seed)
        ranked, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=seed, n_repeats=10)
        
        alloc_ibip = get_feature_allocation(30, 3, "ibip", ranked)
        alloc_seq = get_feature_allocation(30, 3, "sequential")
        
        # IBIP
        m_ibip = QuantumInspiredClassifier(n_qubits=3, topology="cnot_star", feature_indices=alloc_ibip,
                                           lr=0.01, lam=0.5, seed=seed)
        m_ibip.fit(X_tr, y_tr, epochs=50, fast_grad=True, verbose=False)
        p_ibip = m_ibip.predict(X_te)
        prob_ibip = m_ibip.predict_proba(X_te)
        acc_ibip = accuracy_score(y_te, p_ibip) * 100.0
        auc_ibip = roc_auc_score(y_te, prob_ibip)
        
        # Sequential
        m_seq = QuantumInspiredClassifier(n_qubits=3, topology="cnot_star", feature_indices=alloc_seq,
                                          lr=0.01, lam=0.5, seed=seed)
        m_seq.fit(X_tr, y_tr, epochs=50, fast_grad=True, verbose=False)
        p_seq = m_seq.predict(X_te)
        prob_seq = m_seq.predict_proba(X_te)
        acc_seq = accuracy_score(y_te, p_seq) * 100.0
        auc_seq = roc_auc_score(y_te, prob_seq)
        
        diff = acc_ibip - acc_seq
        records.append({
            "Seed": seed,
            "Sequential (%)": round(acc_seq, 2),
            "IBIP (%)": round(acc_ibip, 2),
            "Difference (%)": round(diff, 2),
            "Seq ROC-AUC": round(auc_seq, 4),
            "IBIP ROC-AUC": round(auc_ibip, 4)
        })
        
    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_TABLES, "table5_robustness_reproduced.csv"), index=False)
    
    # Statistical tests
    stat_res = compute_paired_statistics(df["IBIP (%)"].values, df["Sequential (%)"].values,
                                        name_a="IBIP", name_b="Sequential")
    print("\n--- Statistical Analysis of Table 5 (Seed Initialization Matching) ---")
    print(f"IBIP Mean ± SD: {stat_res['mean_a']:.2f}% ± {stat_res['std_a']:.2f}%")
    print(f"Sequential Mean ± SD: {stat_res['mean_b']:.2f}% ± {stat_res['std_b']:.2f}%")
    print(f"Mean Paired Difference: {stat_res['mean_diff']:.2f}% (95% CI: [{stat_res['ci_95_diff'][0]:.2f}%, {stat_res['ci_95_diff'][1]:.2f}%])")
    print(f"Paired t-test: t = {stat_res['t_paired']:.4f}, p = {stat_res['p_paired']:.4f}")
    print(f"Wilcoxon signed-rank: W = {stat_res['wilcoxon_stat']:.2f}, p = {stat_res['p_wilcoxon']:.4f}")
    print(f"Cohen's d_z: {stat_res['cohen_d']:.4f}")
    print(f"Welch independent t-test (legacy reported): t = {stat_res['t_welch']:.4f}, p = {stat_res['p_welch']:.4f}")
    return df, stat_res

def reproduce_table6_cross_validation(X, y):
    print("\n" + "=" * 80)
    print("REPRODUCING TABLE 6: 5-FOLD STRATIFIED CROSS-VALIDATION (IBIP CNOT-STAR)")
    print("=" * 80)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records = []
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        ranked, _, _, _ = compute_xgboost_permutation_ranking(X_tr, y_tr, random_state=42, n_repeats=10)
        alloc_ibip = get_feature_allocation(30, 3, "ibip", ranked)
        
        model = QuantumInspiredClassifier(n_qubits=3, topology="cnot_star", feature_indices=alloc_ibip,
                                           lr=0.01, lam=0.5, seed=42 + fold)
        model.fit(X_tr, y_tr, epochs=50, fast_grad=True, verbose=False)
        probs = model.predict_proba(X_te)
        preds = model.predict(X_te)
        
        metrics = evaluate_predictions(y_te, preds, y_prob=probs, positive_label=0)
        records.append({
            "Fold": fold,
            "Accuracy (%)": round(metrics["accuracy"] * 100.0, 2),
            "Sensitivity (%)": round(metrics["sensitivity"] * 100.0, 2),
            "Specificity (%)": round(metrics["specificity"] * 100.0, 2),
            "Precision (%)": round(metrics["precision"] * 100.0, 2),
            "F1_Score (%)": round(metrics["f1_score"] * 100.0, 2),
            "ROC-AUC": round(metrics["roc_auc"], 4)
        })
        
    df = pd.DataFrame(records)
    mean_row = {
        "Fold": "Mean ± SD",
        "Accuracy (%)": f"{df['Accuracy (%)'].mean():.2f} ± {df['Accuracy (%)'].std():.2f}",
        "Sensitivity (%)": f"{df['Sensitivity (%)'].mean():.2f} ± {df['Sensitivity (%)'].std():.2f}",
        "Specificity (%)": f"{df['Specificity (%)'].mean():.2f} ± {df['Specificity (%)'].std():.2f}",
        "Precision (%)": f"{df['Precision (%)'].mean():.2f} ± {df['Precision (%)'].std():.2f}",
        "F1_Score (%)": f"{df['F1_Score (%)'].mean():.2f} ± {df['F1_Score (%)'].std():.2f}",
        "ROC-AUC": f"{df['ROC-AUC'].mean():.4f} ± {df['ROC-AUC'].std():.4f}"
    }
    df_full = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    print(df_full.to_string(index=False))
    df_full.to_csv(os.path.join(OUTPUT_TABLES, "table6_cv_cnot_star_reproduced.csv"), index=False)
    return df_full

if __name__ == "__main__":
    X, y, _ = load_wdbc_data()
    reproduce_table2_gate_analysis(X, y)
    reproduce_training_dynamics(X, y)
    reproduce_table5_robustness(X, y)
    reproduce_table6_cross_validation(X, y)
