# Importance-Balanced Interleaved Feature Projection for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study

This repository provides the complete, reproducible implementation for the manuscript:
> **"Circuit Interaction Topologies and Importance-Balanced Feature Allocation for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study"**  
> *Daksh Kaila, Monika Aggarwal, Samya Muhuri* (Academia Quantum)

---

## 🔬 Key Architectural Highlights
1. **Importance-Balanced Interleaved Projection (IBIP):**
   - Ranks 30 cytological features using XGBoost permutation importance strictly on training data (eliminating data leakage).
   - Distributes features across qubit registers in an interleaved manner to prevent informational concentration.
2. **Exact Hilbert Space Statevector Quantum Simulation:**
   - Pure statevector evolution in complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ for $n \in \{2, 3\}$.
   - Single-qubit rotations: $R_y(\phi_q)$, $R_z(e_q)$, $R_y(v_q)$, $R_z(v_q)$.
   - Two-qubit interaction topologies:
     - 2-Qubit: SWAP, $\sqrt{\text{SWAP}}$, unitary iSWAP, CNOT(0→1), Independent.
     - 3-Qubit: CNOT-Star, CNOT-Chain, CNOT-Ring, SWAP-Star, SWAP-Chain, Full-Mesh, Independent.
3. **Hybrid Optimization Framework:**
   - Exact Parameter Shift Rule (PSR) for circuit rotation parameters.
   - Exact chain-rule backpropagation through non-linear hyperbolic tangent feature projections ($\phi_q = \pi \tanh(W_q^T x_q / \sqrt{|x_q|})$).
   - Parameter updates modulated via **Superpositional Gradient Descent (SuperGD)**.
4. **Controlled Like-for-Like Benchmarking:**
   - 2-qubit and 3-qubit architectures evaluated in separate, unconfounded experimental suites.
   - Evaluated across 5 identical random seeds (0, 42, 123, 777, 2025) and 5-fold stratified cross-validation.
   - Paired Student's $t$-test, Wilcoxon signed-rank test, Cohen's $d_z$, and 95% Confidence Intervals.
   - Benchmarked against classical baselines: Support Vector Machines (RBF and Linear), XGBoost, Ridge Classifier, Logistic Regression, and Random Forest.
   - Positive class explicitly grounded in clinical diagnosis (Malignant = 1, Benign = 0).

---

## 📁 Repository Directory Structure

```
BreastCancer_QML/
├── data/
│   ├── download_data.py          # Script to download and verify WDBC dataset
│   └── wdbc.csv                  # Wisconsin Diagnostic Breast Cancer dataset (569 samples, 30 features)
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Dataset loading, StandardScaler (leakage-free), stratified splits
│   ├── ranking.py                # XGBoost permutation importance & IBIP / Sequential allocation
│   ├── quantum_circuit.py        # Statevector quantum simulator, unitary gates, expectation values
│   ├── models.py                 # QuantumInspiredClassifier with PSR, SuperGD, and tensor contractions
│   ├── optimizers.py             # SuperGD, standard SGD, and Adam update rules
│   ├── classical_baselines.py    # SVM, XGBoost, Ridge, Logistic Regression, Random Forest
│   └── evaluation.py             # Paired statistical tests (t-test, Wilcoxon, Cohen's d_z, 95% CI)
├── experiments/
│   ├── reproduce_original_paper.py              # Exact reproduction of original paper tables & figures
│   ├── debug_iswap.py                           # Audit and verification of the iSWAP unitary matrix
│   ├── run_classical_baselines.py               # Classical baselines 5-seed and 5-fold CV evaluation
│   ├── run_controlled_experiments_parallel.py   # Multi-process controlled 2Q & 3Q experiment suites
│   └── generate_publication_figures.py          # High-resolution (300 DPI) figure generation
├── results/
│   ├── tables/                   # Generated CSV tables (summaries, per-seed runs, paired statistics)
│   └── figures/                  # Publication-grade figures (PNG, 300 DPI)
├── paper/
│   ├── manuscript_marked_revision.md            # Complete revised manuscript with all revision markings
│   └── response_to_reviewers.md                 # Detailed point-by-point response to editor and reviewers
├── requirements.txt              # Environment dependencies
└── README.md                     # Comprehensive documentation
```

---

## 🚀 Installation & Environment Setup

Python version required: **Python 3.11.9**

```bash
# Clone the repository
git clone https://github.com/dakshkaila/BreastCancer_QIVC.git
cd BreastCancer_QIVC

# Create virtual environment with Python 3.11.9
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Reproducing Experimental Results

### 1. Download & Verify Dataset
```bash
python data/download_data.py
```

### 2. Run Classical Baselines
```bash
python experiments/run_classical_baselines.py
```

### 3. Run Controlled 2-Qubit and 3-Qubit Experimental Suites (Parallelized)
```bash
python experiments/run_controlled_experiments_parallel.py
```

### 4. Audit & Verify iSWAP Performance
```bash
python experiments/debug_iswap.py
```

### 5. Run 4-Way Ablation Study (Dissecting Interleaving)
```bash
python experiments/run_4way_ablation.py
```

### 6. Run Cross-Dataset Biomedical Generalization (WDBC, Parkinson's, Heart Disease, Coimbra)
```bash
python data/download_all_biomedical_data.py
python experiments/run_multi_biomedical_benchmarks.py
```

### 7. Generate Publication Figures
```bash
python experiments/generate_publication_figures.py
```

---

## 📊 Summary of Primary Findings

| Architecture / Model | Qubits | Allocation | Mean Test Accuracy (%) | Mean Sensitivity (Malignant Recall) | Mean Specificity (Benign) | Mean ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical SVM (RBF Kernel)** | — | — | **97.72 ± 1.59%** | **95.24 ± 4.12%** | **99.17 ± 1.24%** | **0.9944 ± 0.0060** |
| **Classical Logistic Regression** | — | — | 97.54 ± 1.69% | 94.76 ± 3.91% | 99.17 ± 0.76% | 0.9939 ± 0.0057 |
| **Classical XGBoost** | — | — | 96.49 ± 1.39% | 94.76 ± 1.99% | 97.50 ± 2.06% | 0.9919 ± 0.0085 |
| **3-Qubit CNOT-Star** | 3 | **IBIP** | **97.02 ± 0.48%** | **94.76 ± 1.99%** | **98.33 ± 1.24%** | **0.9962 ± 0.0006** |
| 3-Qubit CNOT-Star | 3 | Sequential | 94.91 ± 1.14% | 90.00 ± 3.65% | 97.78 ± 1.24% | 0.9909 ± 0.0028 |
| **2-Qubit $\sqrt{\text{SWAP}}$** | 2 | **IBIP** | 96.84 ± 2.20% | 94.29 ± 5.48% | 98.33 ± 1.52% | 0.9921 ± 0.0116 |
| **2-Qubit SWAP** | 2 | **IBIP** | 96.84 ± 2.29% | 94.76 ± 5.68% | 98.06 ± 2.11% | 0.9919 ± 0.0091 |
| **2-Qubit Unitary iSWAP (Corrected)**| 2 | **IBIP** | 95.96 ± 2.20% | 93.33 ± 5.16% | 97.50 ± 2.28% | 0.9900 ± 0.0091 |

---

## ⚖️ Citation
If you utilize this codebase, data allocations, or methodology in your research, please cite:
```bibtex
@article{kaila2026circuit,
  title={Circuit Interaction Topologies and Importance-Balanced Feature Allocation for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study},
  author={Kaila, Daksh and Aggarwal, Monika and Muhuri, Samya},
  journal={Academia Quantum},
  year={2026},
  doi={10.20935/AcadQuantXXXX}
}
```

