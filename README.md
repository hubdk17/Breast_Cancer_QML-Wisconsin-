# Circuit Interaction Topologies and Importance-Balanced Feature Allocation for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study

This repository provides the complete, reproducible Python implementation for the manuscript:
> **"Circuit Interaction Topologies and Importance-Balanced Feature Allocation for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study"**
> Currently submitted to Academia Quantum

Repository URL: `https://github.com/hubdk17/Breast_Cancer_QML-Wisconsin-.git`

---

## Overview

Quantum-inspired variational classifiers (QIVCs) simulate parameterized quantum circuits on classical computing architectures, leveraging unitary statevector rotations, multi-qubit entangling topologies, and Pauli expectation measurements. This codebase provides a modular, leakage-free benchmark investigating:
1. **Importance-Balanced Interleaved Projection (IBIP):** Interleaves high-leverage and fine-grained diagnostic features across qubit registers to prevent register starvation and uninformative entanglement.
2. **Controlled Circuit Interaction Topologies:** Evaluates two-qubit exchange gates (SWAP, $\sqrt{\text{SWAP}}$, unitary iSWAP, CNOT, Independent) and three-qubit connection graphs (CNOT-Star, CNOT-Chain, CNOT-Ring, SWAP-Star, SWAP-Chain, Full-Mesh, Independent) in unconfounded suites.
3. **Rigorous Benchmarking & Clinical Metrics:** Benchmarks QIVCs against 6 classical machine learning baselines (SVM RBF/Linear, XGBoost, Ridge, Logistic Regression, Random Forest) across 5-seed, 20-seed, and 5-fold cross-validation evaluations, reporting Accuracy, Sensitivity (Malignant Recall), Specificity (Benign Recall), Precision, F1-Score, ROC-AUC, and full confusion matrices.
4. **Generalization Beyond WDBC:** Evaluates IBIP across four distinct biomedical benchmarks: Wisconsin Diagnostic Breast Cancer (WDBC), Parkinson's Disease (UCI 174), Cleveland Heart Disease (UCI 45), and Breast Cancer Coimbra (UCI 451).

---

## Directory Structure

```
BreastCancer_QML/
├── data/
│   ├── download_all_biomedical_data.py          # Downloads & prepares WDBC, Parkinson's, Heart Disease, Coimbra
│   ├── wdbc.csv                                 # 569 samples, 30 cytological features (breast oncology)
│   ├── parkinsons.csv                           # 195 samples, 22 acoustic features (neurology)
│   ├── heart_disease.csv                        # 303 samples, 13 cardiovascular features (cardiology)
│   └── coimbra.csv                              # 116 samples, 9 serum biomarker features (metabolism)
├── src/
│   ├── __init__.py
│   ├── dataset.py                               # Preprocessing, train/test splitting, stratified K-fold
│   ├── ranking.py                               # Feature rankers (XGBoost, RF, Mutual Info, ANOVA) & IBIP allocations
│   ├── quantum_circuit.py                       # Exact complex statevector simulator & unitary gate matrices
│   ├── models.py                                # Fast tensor-contraction QIVC simulator with PSR and SuperGD
│   ├── optimizers.py                            # Superpositional Gradient Descent (SuperGD), SGD, Adam
│   ├── classical_baselines.py                   # Standardized SVM, XGBoost, Ridge, Logistic Regression, RF
│   └── evaluation.py                            # Paired Student's t-test, Wilcoxon signed-rank, Cohen's d_z, 95% CIs
├── experiments/
│   ├── reproduce_original_paper.py              # Reproduction of original manuscript exploratory tables
│   ├── debug_iswap.py                           # Audit & verification of unitary iSWAP complex matrix
│   ├── run_classical_baselines.py               # Evaluation of 6 classical models (seeds & 5-fold CV)
│   ├── run_controlled_experiments_parallel.py   # Controlled 2Q and 3Q suites under IBIP & Sequential
│   ├── run_4way_ablation.py                     # 4-way ablation (Sequential, Random, Importance-Blocked, IBIP)
│   ├── run_multi_biomedical_benchmarks.py       # Cross-dataset generalization across all 4 biomedical datasets
│   ├── run_ranking_ablation.py                  # Evaluation of 4 distinct ranking criteria across 10 seeds
│   ├── analyze_topology_quantitatively.py       # Von Neumann entanglement entropy & graph distance analysis
│   ├── run_20seed_verification.py               # Expanded 20-seed replication and paired statistics
│   └── generate_publication_figures.py          # High-resolution (300 DPI) publication figure generation
├── results/
│   ├── tables/                                  # CSV results tables, paired statistical contrasts, CV summaries
│   └── figures/                                 # Publication-grade figures (PNG, 300 DPI)
├── paper/
│   ├── manuscript_marked_revision.md            # Complete revised manuscript with all revision markings
│   └── response_to_reviewers.md                 # Detailed point-by-point response to editor and reviewers
├── requirements.txt                             # Python dependencies
└── README.md                                    # Documentation and reproduction guide
```

---

## Installation & Environment Setup

Environment tested with **Python 3.11.9**:

```bash
# Clone the repository
git clone https://github.com/hubdk17/Breast_Cancer_QML-Wisconsin-.git
cd Breast_Cancer_QML-Wisconsin-

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Results

All experimental scripts utilize Python multiprocessing for fast execution.

### 1. Acquire & Preprocess Datasets
```bash
python data/download_all_biomedical_data.py
```

### 2. Classical Machine Learning Baselines
```bash
python experiments/run_classical_baselines.py
```

### 3. Controlled 2-Qubit & 3-Qubit Interaction Suites
```bash
python experiments/run_controlled_experiments_parallel.py
```

### 4. Four-Way Allocation Ablation Study
```bash
python experiments/run_4way_ablation.py
```

### 5. Multi-Dataset Biomedical Generalization (WDBC, Parkinson's, Heart Disease, Coimbra)
```bash
python experiments/run_multi_biomedical_benchmarks.py
```

### 6. Ranking Method Robustness Ablation (XGBoost, RF, Mutual Info, ANOVA)
```bash
python experiments/run_ranking_ablation.py
```

### 7. Quantitative Interaction Topology Analysis (Entropy & Network Metrics)
```bash
python experiments/analyze_topology_quantitatively.py
```

### 8. 20-Seed Replication Suite & Matched Paired Statistics
```bash
python experiments/run_20seed_verification.py
```

### 9. Generate Publication Figures (300 DPI)
```bash
python experiments/generate_publication_figures.py
```

---

## Summary of Empirical Findings

| Model / Architecture | Qubits | Strategy | Mean Test Accuracy (%) | Sensitivity (Recall) | Specificity | Mean ROC-AUC | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical SVM (RBF Kernel)** | — | — | **97.72 ± 1.59%** | **95.24 ± 4.12%** | **99.17 ± 1.24%** | **0.9944 ± 0.0060** | ~150 SVs |
| **Classical Logistic Regression** | — | — | 97.54 ± 1.69% | 94.76 ± 3.91% | 99.17 ± 0.76% | 0.9939 ± 0.0057 | 31 |
| **Classical XGBoost** | — | — | 96.49 ± 1.39% | 94.76 ± 1.99% | 97.50 ± 2.06% | 0.9919 ± 0.0085 | 100 Trees |
| **3-Qubit CNOT-Star (QIVC)** | 3 | **IBIP** | **97.02 ± 0.48%** | **94.76 ± 1.99%** | **98.33 ± 1.24%** | **0.9962 ± 0.0006** | **48** |
| **2-Qubit $\sqrt{\text{SWAP}}$ (QIVC)** | 2 | **IBIP** | 96.49 ± 0.87% | 93.33 ± 2.84% | 98.33 ± 1.24% | 0.9960 ± 0.0012 | 34 |
| **2-Qubit Unitary iSWAP (QIVC)** | 2 | **IBIP** | 95.61 ± 0.78% | 91.90 ± 3.48% | 97.78 ± 1.24% | 0.9921 ± 0.0018 | 34 |
| **2-Qubit SWAP (QIVC)** | 2 | **IBIP** | 95.61 ± 0.78% | 91.90 ± 3.48% | 97.78 ± 1.24% | 0.9917 ± 0.0018 | 34 |

---

## Citation

https://github.com/hubdk17/Breast_Cancer_QML-Wisconsin-.git


