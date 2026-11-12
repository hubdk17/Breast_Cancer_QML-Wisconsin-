"""
Operating-threshold analysis (Table 8 of the revised manuscript).

Trains the 3-qubit QIVC (CNOT-Star and CNOT-Chain, Sequential allocation) and an RBF-SVM
on the same five stratified 80:20 WDBC splits used throughout (seeds 0, 42, 123, 777, 2025),
then classifies a sample as malignant when its malignancy score exceeds tau.

The QIVC outputs P(benign) (label 1), so the malignancy score is 1 - P(benign).
Malignant (label 0) is the positive class.

Outputs:
- results/tables/threshold_analysis_per_seed.csv
- results/tables/threshold_analysis_summary.csv
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.svm import SVC

from src.dataset import load_wdbc_data, get_train_test_data
from src.ranking import get_feature_allocation
from src.models import QuantumInspiredClassifier

OUTPUT_TABLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables")
SEEDS = [0, 42, 123, 777, 2025]
TAUS = [0.5, 0.4, 0.3, 0.2]


def malignancy_scores(task):
    model_name, seed = task
    X, y, _ = load_wdbc_data()
    X_tr, X_te, y_tr, y_te, _ = get_train_test_data(X, y, test_size=0.2, random_state=seed)
    if model_name == "SVM (RBF)":
        clf = SVC(C=1.0, kernel="rbf", gamma=1.0 / X.shape[1], probability=True, random_state=seed).fit(X_tr, y_tr)
        p_mal = clf.predict_proba(X_te)[:, list(clf.classes_).index(0)]
    else:
        topology = "cnot_star" if "Star" in model_name else "cnot_chain"
        alloc = get_feature_allocation(X.shape[1], 3, "sequential")
        model = QuantumInspiredClassifier(n_qubits=3, topology=topology, feature_indices=alloc,
                                          lr=0.01, lam=0.5, seed=seed)
        model.fit(X_tr, y_tr, epochs=50)
        p_mal = 1.0 - model.predict_proba(X_te)
    return model_name, seed, p_mal, (y_te == 0).astype(int)


def main(max_workers=4):
    models = ["QIVC CNOT-Star (Sequential)", "QIVC CNOT-Chain (Sequential)", "SVM (RBF)"]
    tasks = [(m, s) for m in models for s in SEEDS]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        outs = list(ex.map(malignancy_scores, tasks))

    rows = []
    for model_name, seed, p_mal, y_mal in outs:
        for tau in TAUS:
            pred = (p_mal >= tau).astype(int)
            tp = int(((pred == 1) & (y_mal == 1)).sum()); fn = int(((pred == 0) & (y_mal == 1)).sum())
            tn = int(((pred == 0) & (y_mal == 0)).sum()); fp = int(((pred == 1) & (y_mal == 0)).sum())
            rows.append({"Model": model_name, "Seed": seed, "Tau": tau,
                         "Sensitivity": 100.0 * tp / (tp + fn), "Specificity": 100.0 * tn / (tn + fp),
                         "Missed_Malignant": fn, "TP": tp, "TN": tn, "FP": fp, "FN": fn})
    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_TABLES, "threshold_analysis_per_seed.csv"), index=False)
    summary = df.groupby(["Model", "Tau"]).agg(
        Sensitivity_mean=("Sensitivity", "mean"), Sensitivity_sd=("Sensitivity", "std"),
        Specificity_mean=("Specificity", "mean"), Specificity_sd=("Specificity", "std"),
        Missed_Malignant_mean=("Missed_Malignant", "mean")).round(2).reset_index()
    summary = summary.sort_values(["Model", "Tau"], ascending=[True, False])
    summary.to_csv(os.path.join(OUTPUT_TABLES, "threshold_analysis_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
