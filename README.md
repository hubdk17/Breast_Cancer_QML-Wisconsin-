# Importance-Balanced Interleaved Feature Projection for Quantum-Inspired Variational Classifiers

Reproducible code and results for the manuscript:

> **"Importance-Balanced Interleaved Feature Projection for Quantum-Inspired Variational Classifiers: A Controlled Study of Feature Allocation and Interaction Topology in Breast Cancer Diagnosis"**
> Daksh Kaila, Monika Aggarwal, Samya Muhuri. Revised submission to *Academia Quantum*.

The revised manuscript is in [`paper/manuscript_revised.pdf`](paper/manuscript_revised.pdf) (Word version: `paper/manuscript_revised.docx`). Every number and figure in it is generated from the CSV files in `results/tables/` by the scripts in `experiments/`.

---

## Overview

Quantum-inspired variational classifiers (QIVCs) compress many continuous features into a few simulated qubit registers. This repository studies two design choices under matched seeds, splits and paired statistics:

1. **Feature allocation:** which features feed which qubit. Four strategies are compared:
   - **Sequential:** original column order.
   - **Random:** random equal-sized groups.
   - **Importance-Blocked:** top-ranked features packed together on the first qubit.
   - **IBIP (Importance-Balanced Interleaved Projection):** ranked features dealt round-robin across qubits.
2. **Interaction topology:**
   - Two-qubit gates: SWAP, √SWAP, iSWAP, CNOT, Independent.
   - Three-qubit graphs: CNOT-Star, CNOT-Chain, CNOT-Ring, Full-Mesh, SWAP-Star, SWAP-Chain, Independent.

The model applies a classical non-linear projection φ_q = π·tanh(W_qᵀx_q/√d_q), then exact complex statevector simulation (encoding R_y(φ_q)R_z(e_q), the interaction layer, variational R_y(v_q)R_z(v_q)), Pauli-Z expectations and a linear–sigmoid readout. It is trained with BCE loss, shift-based gradient estimates and Superpositional Gradient Descent. The 3-qubit model has 40 trainable parameters. At this scale the model is classically computable; no quantum advantage is claimed.

---

## Key results (WDBC, from `results/tables/`)

**Primary comparison: 3-qubit CNOT-Star, IBIP vs Sequential**

| Protocol | IBIP accuracy (%) | Sequential accuracy (%) | Paired difference (IBIP − Seq) |
| :--- | :---: | :---: | :--- |
| 5 seeds | 95.26 ± 2.02 | 97.37 ± 2.24 | −2.11, p = 0.009 |
| 5-fold CV | 96.14 ± 2.28 | 97.54 ± 1.69 | — |
| 20 seeds (0–19) | 96.62 ± 1.69 | 97.68 ± 1.15 | −1.05 [−1.86, −0.25], p = 0.013; ROC-AUC difference n.s. (p = 0.455) |

On WDBC, sequential allocation, which puts the mean, standard-error and worst-value descriptors on separate qubits, is more accurate than IBIP.

**Four-way allocation ablation (CNOT-Star, 5 seeds)**

| Strategy | Sequential | Random | Importance-Blocked | IBIP |
| :--- | :---: | :---: | :---: | :---: |
| Accuracy (%) | 97.37 ± 2.24 | 94.74 ± 2.24 | 92.81 ± 2.51 | 95.26 ± 2.95 |

IBIP beats importance-blocking on WDBC (+2.46, p = 0.108), Parkinson's disease (+1.03) and heart disease (+8.20, p = 0.009). It beats sequential allocation on Parkinson's (+5.64, p = 0.063) but not on WDBC or heart disease.

**Topologies.**
- Under IBIP, no entangling topology outperforms the uncoupled circuit (Independent 95.44%).
- CNOT-Ring degrades accuracy to about 81%.
- With exact complex simulation, iSWAP performs like the other exchange gates (95.96%).

**Classical baselines (same splits).**
- RBF-SVM 97.72 ± 1.59% and logistic regression 97.54 ± 1.69%.
- The best QIVC (Sequential CNOT-Chain, 97.72 ± 1.33%) matches but does not exceed the SVM.

---

## Directory structure

```
data/                     WDBC, Parkinson's, Cleveland heart disease, Coimbra (+ download scripts)
src/
  dataset.py              preprocessing, stratified splits and folds (scaler fitted on training data only)
  ranking.py              feature rankers and the four allocation strategies
  quantum_circuit.py      gate matrices and statevector utilities
  models.py               QIVC: projection, statevector simulation, shift-based gradients, SuperGD
  optimizers.py           SuperGD, SGD, Adam
  classical_baselines.py  SVM (RBF/linear), logistic regression, XGBoost, random forest, ridge
  evaluation.py           paired t-test, Wilcoxon, Cohen's d_z, 95% CI
experiments/
  run_controlled_experiments_parallel.py   2- and 3-qubit suites, IBIP vs Sequential (5 seeds + 5-fold CV)
  run_20seed_verification.py               20-seed replication of the primary comparison
  run_4way_ablation.py                     Sequential / Random / Importance-Blocked / IBIP
  run_multi_biomedical_benchmarks.py       allocation strategies on the external datasets
  run_ranking_ablation.py                  four ranking criteria (10 seeds)
  run_classical_baselines.py               six classical models (5 seeds + 5-fold CV)
  run_threshold_analysis.py                operating-threshold analysis (Table 8)
  analyze_topology_quantitatively.py       path length, entanglement entropy, gradient norms
  debug_iswap.py                           diagnostic for the version-1 iSWAP error
  make_revision_figures.py                 Figures 1, 3, 4, 5 of the revised manuscript
  generate_publication_figures.py          supplementary figures (ROC, confusion matrices)
  reproduce_original_paper.py              reproduction of version-1 exploratory tables
results/tables/           all per-seed, per-fold, summary and paired-statistics CSVs
results/figures/revision/ figures used in the revised manuscript
paper/                    revised manuscript (.docx, .pdf)
```

---

## Reproducing the results

Tested with Python 3.11.

```bash
git clone https://github.com/hubdk17/Breast_Cancer_QML-Wisconsin-.git
cd Breast_Cancer_QML-Wisconsin-
python -m venv venv && source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

python data/download_all_biomedical_data.py
python experiments/run_classical_baselines.py
python experiments/run_controlled_experiments_parallel.py
python experiments/run_20seed_verification.py
python experiments/run_4way_ablation.py
python experiments/run_multi_biomedical_benchmarks.py
python experiments/run_ranking_ablation.py
python experiments/run_threshold_analysis.py
python experiments/analyze_topology_quantitatively.py
python experiments/make_revision_figures.py
```

Permutation importance is stochastic, so IBIP per-seed accuracies can vary slightly between independently ranked runs. Sequential allocation does not depend on a ranking and reproduces exactly.

---

## Notes on version 1

- **iSWAP:** the version-1 accuracy of 58.80% was an implementation error. The prototype code was not preserved. `experiments/debug_iswap.py` shows that a real-typed iSWAP matrix, whose *i* entries are zeroed so the gate is not unitary, produces a comparable collapse, whereas discarding the statevector's imaginary part does not.
- **IBIP vs sequential:** version 1 reported IBIP 97.02% vs Sequential 94.91% on WDBC. The rebuilt, matched pipeline does not reproduce that result; see the tables above.

## Citation

Kaila D, Aggarwal M, Muhuri S. Importance-Balanced Interleaved Feature Projection for Quantum-Inspired Variational Classifiers. *Academia Quantum* (under revision). Code: https://github.com/hubdk17/Breast_Cancer_QML-Wisconsin-
