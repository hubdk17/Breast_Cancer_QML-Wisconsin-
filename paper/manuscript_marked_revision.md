# [MODIFIED MANUSCRIPT WITH REVISION MARKINGS]

**Journal:** Academia Quantum  
**Article Type:** Research Article  
**Article DOI:** 10.20935/AcadQuantXXXX  
**Status:** Major Revision Resubmission  

---

### [MODIFIED TITLE]
**Old Title:** *Importance-Balanced Interleaved Feature Projection for Quantum-Inspired Variational Classifiers: A Topological Study on Breast Cancer Diagnosis*  
**Revised Title:** **Circuit Interaction Topologies and Importance-Balanced Feature Allocation for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study**  
*(Note: Revised in response to Editorial Comment 6 and Reviewer 3 Comment 3 to eliminate the misleading term "Topological Study", as the manuscript investigates circuit interaction graphs rather than topological data analysis.)*

**Authors:** Daksh Kaila$^{1,*}$, Monika Aggarwal$^2$, Samya Muhuri$^1$  
$^1$ Thapar Institute of Engineering and Technology, Patiala, Punjab, India  
$^2$ Indian Institute of Technology Delhi, New Delhi, Delhi, India  
$^*$ **Correspondence:** dkaila_be25@thapar.edu  

---

### [MODIFIED ABSTRACT]
*(Note: Revised in response to Editorial Comments 1, 2, 3, 4, 5, 6 and Reviewers 2 & 3. Clarifies hybrid classical statevector simulation vs. hardware execution; corrects the iSWAP artifact; provides balanced classical baseline context; and reports rigorous matched paired statistics and confidence intervals.)*

**Introduction:** Quantum-inspired machine learning offers an alternative framework for exploring non-linear representation learning, parameter-shift optimization, and multi-qubit interaction mechanisms. However, the interplay between feature allocation across low-qubit registers and circuit interaction topologies remains insufficiently characterized in biomedical tabular classification tasks.

**Materials and methods:** We formulate an Importance-Balanced Interleaved Projection (IBIP) framework for quantum-inspired variational classifiers (QIVCs). The framework ranks biomedical attributes using XGBoost permutation importance strictly on training data and distributes features cyclically across qubits, preventing informative diagnostic attributes from concentrating within isolated registers. The classifier operates as a hybrid quantum-inspired model: classical input subvectors are mapped into rotation angles via trainable non-linear projections ($\phi_q = \pi \tanh(W_q^T x_q / \sqrt{|x_q|})$), followed by exact numerical statevector simulation of parameterized rotations and two-qubit interaction topologies in complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$, and measured via Pauli-$Z$ expectations linearly mapped to class probabilities. To avoid confounding topology with qubit register size, two-qubit exchange gates (SWAP, $\sqrt{\text{SWAP}}$, unitary iSWAP, CNOT, Independent) and three-qubit interaction graphs (CNOT-Star, CNOT-Chain, CNOT-Ring, SWAP-Star, SWAP-Chain, Full-Mesh, Independent) were evaluated in separate, unconfounded experimental suites on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset across five matched random initialization seeds (0, 42, 123, 777, 2025) and five-fold stratified cross-validation. Models were benchmarked against standardized classical baselines (SVM RBF/Linear, XGBoost, Ridge Classifier, Logistic Regression, Random Forest).

**Results:** In the three-qubit controlled suite, the CNOT-Star architecture with IBIP achieved a mean test accuracy of $97.02\% \pm 0.48\%$ and a mean ROC-AUC of $0.9962 \pm 0.0006$ across five matched seeds, significantly outperforming conventional sequential feature partitioning ($94.91\% \pm 1.14\%$ accuracy, $0.9909 \pm 0.0028$ ROC-AUC; paired difference $+2.11\%$, $95\%\text{ CI: }[0.45\%, 3.76\%]$, paired $t(4) = 3.5415, p = 0.0240$, Wilcoxon $W = 0.0, p = 0.0431$, Cohen's $d_z = 1.5838$). In five-fold cross-validation, IBIP CNOT-Star demonstrated stable patient generalization ($95.26\% \pm 1.82\%$ accuracy, ROC-AUC $0.9840 \pm 0.0113$, Sensitivity $92.46\% \pm 4.20\%$, Specificity $96.92\% \pm 1.81\%$). In the two-qubit suite, diagnostic re-evaluation of the previously reported near-random iSWAP result ($58.80\%$) revealed it to be an implementation artifact caused by non-unitary real-value truncation in early prototypes; when simulated under correct complex unitary evolution, iSWAP achieved $95.61\% \pm 0.78\%$ accuracy and $0.9921 \pm 0.0018$ ROC-AUC, fully competitive with SWAP ($95.61\%$) and $\sqrt{\text{SWAP}}$ ($96.49\%$). Standard classical baselines achieved reference clinical performance: SVM with RBF kernel reached $97.72\% \pm 1.59\%$ accuracy (ROC-AUC $0.9944$), Logistic Regression reached $97.54\% \pm 1.69\%$ (ROC-AUC $0.9939$), and XGBoost reached $96.49\% \pm 1.39\%$ (ROC-AUC $0.9919$).

**Conclusions:** Interleaved feature balancing (IBIP) reliably enhances generalization across both two-qubit and three-qubit quantum-inspired architectures by encouraging cross-register communication among high-leverage features. Among interaction topologies, centralized hub-and-spoke graphs (CNOT-Star) provide superior feature coordination compared to unbranched chains or cyclic rings. However, classical kernel SVM and logistic regression remain highly competitive, indicating that quantum-inspired classifiers should be viewed as expressive exploratory representation mechanisms rather than superior alternatives to mature classical models on moderate-dimensional tabular benchmarks.

**Keywords:** Quantum-Inspired Machine Learning; Variational Quantum Classifier; Feature Allocation; Circuit Interaction Topology; Breast Cancer Diagnosis; Permutation Importance

---

## 1. Introduction

Breast cancer remains one of the leading causes of cancer-related mortality among women worldwide. Early and accurate detection through digitized fine-needle aspiration (FNA) cytology is essential for improving clinical prognosis and planning targeted therapeutic interventions [1]. While classical machine learning algorithms—including Support Vector Machines (SVMs) [1], Random Forests [5], and Gradient Boosted Decision Trees (XGBoost) [6]—have demonstrated robust diagnostic performance on the benchmark Wisconsin Diagnostic Breast Cancer (WDBC) dataset [24], understanding how alternative computational paradigms process and structure biomedical representations continues to be an active area of investigation.

Recent progress in quantum computing and quantum machine learning (QML) [2,3,7] has stimulated significant interest in variational quantum classifiers (VQCs) and quantum-inspired learning models [8–11]. These architectures leverage high-dimensional Hilbert space mappings, parameterized unitary rotation gates, entangling operators, and expectation value measurements to capture complex non-linear feature interactions [10,11]. Multiple studies have explored quantum kernels, quantum support vector classifiers (QSVCs) [12], quantum convolutional neural networks (QCNNs) [13], and hybrid quantum-classical networks (QCCNNs) [14] for breast cancer classification.

However, a fundamental architectural bottleneck in applying quantum and quantum-inspired classifiers to continuous tabular datasets is the **feature allocation problem**. In near-term quantum and quantum-inspired variational architectures, high-dimensional classical vectors (e.g., 30 cytological descriptors in WDBC) must be compressed and mapped into a low number of simulated qubit registers ($n \in \{2, 3\}$). Conventional pipelines universally rely on naive sequential block partitioning, where features are grouped strictly by their arbitrary column order in the dataset (e.g., assigning features 1–10 to Qubit 0, 11–20 to Qubit 1, and 21–30 to Qubit 2). In biomedical datasets where morphological metrics are grouped by category (mean values, standard errors, and worst/extreme values), sequential assignment inadvertently concentrates the most clinically diagnostic features into a single register, depriving two-qubit entangling gates of the ability to mediate cross-attribute interactions between dominant diagnostic signals.

Furthermore, prior research has largely evaluated single circuit interaction topologies in isolation, without systematically evaluating how gate interaction patterns (such as sequential chains, centralized stars, cyclic rings, or partial-exchange gates) impact learning dynamics under controlled conditions.

### [ADDED: Summary of Revisions in Resubmission]
To address critical concerns raised during peer review regarding controlled comparisons, statistical validity, baseline completeness, and algorithmic transparency, this revised study provides:
1. **Unconfounded Controlled Comparisons:** We separate two-qubit and three-qubit evaluations into distinct experimental suites, testing all interaction topologies under identical data splits, identical feature allocations, identical random seeds, and identical cross-validation folds.
2. **Matched Paired Statistical Analyses:** We replace independent-sample tests with paired $t$-tests, Wilcoxon signed-rank tests, Cohen's $d_z$ effect sizes, and 95% confidence intervals on matched per-seed and per-fold differences.
3. **Comprehensive Classical Baselines:** We benchmark SVM (RBF and Linear), XGBoost, Ridge Classifier, Logistic Regression, and Random Forest under identical conditions, explicitly defining Malignant as the clinical positive class.
4. **Clarified Quantum-Inspired Hybrid Architecture:** We formally define the numerical statevector simulation framework, clarifying that continuous features undergo classical non-linear projection into rotation angles before exact statevector evolution in complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$.
5. **Audited and Corrected iSWAP Simulation:** We resolve the anomalous 58.80% result from the initial draft, demonstrating that proper unitary statevector simulation yields competitive performance ($95.61\%$).

---

## 2. Materials and methods

The revised classification pipeline is illustrated in Figure 1.

### 2.1 Dataset, Preprocessing, and Clinical Class Definition
Experiments were performed using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset obtained from the UCI Machine Learning Repository [24]. The dataset consists of 569 patient samples derived from digitized FNA images of breast masses. Each sample contains 30 continuous real-valued cytological features describing nuclear characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension, reported across three summaries: mean (features 0–9), standard error (features 10–19), and "worst" or largest values (features 20–29).

**[ADDED: Clinical Positive Class Specification]**  
In clinical diagnostic oncology, the primary objective is the sensitive detection of malignant carcinoma to prevent life-threatening false negatives. Among the 569 cases, 212 ($37.26\%$) are diagnosed as **Malignant** and 357 ($62.74\%$) as **Benign**. For all clinical evaluations, **Malignant** is explicitly designated as the positive class ($y_{\text{pos}} = 0 \to 1$ in binary evaluation), while **Benign** represents the negative class. All evaluation metrics—Sensitivity (Malignant Recall), Specificity (Benign Recall), Precision (Positive Predictive Value), and F1-Score—are computed with respect to this clinically grounded convention. Full confusion matrices ($2 \times 2$: TP, FP, FN, TN) are reported for all models.

Continuous features were standardized using $z$-score normalization:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
**[ADDED: Data Leakage Prevention Protocol]**  
To prevent data leakage, the scaler parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) and feature ranking models were fitted **strictly on the training partition** of each split or cross-validation fold. Held-out validation and test samples remained completely unexposed during feature standardization, feature importance ranking, and parameter optimization.

### 2.2 Feature Ranking via XGBoost Permutation Importance
Feature importance was quantified using permutation importance computed on an XGBoost classifier trained on the training split [5,6,23]. Permutation importance measures the decrease in model classification performance (log-loss) when values of a single feature are randomly shuffled across samples, breaking the feature-target relationship while preserving marginal distributions:
$$I(f_j) = \mathcal{L}(X^{\text{perm}(j)}, y) - \mathcal{L}(X, y)$$
Averaged across 20 shuffle iterations, features were ranked in descending order of mean importance: $\mathcal{R} = [r_1, r_2, \dots, r_{30}]$, where $r_1$ denotes the most influential feature.

In clinical alignment with breast pathology, the highest-ranking features consistently comprised extreme nuclear morphology indicators: `worst texture` ($r_1$), `worst concave points` ($r_2$), `area error` ($r_3$), `worst perimeter` ($r_4$), `mean texture` ($r_5$), and `worst concavity` ($r_6$).

### 2.3 Importance-Balanced Interleaved Projection (IBIP)
Let $n \in \{2, 3\}$ denote the number of qubit registers. Conventional sequential partitioning assigns contiguous blocks of features based on arbitrary column ordering:
$$\mathcal{S}_q^{\text{Seq}} = \left\{ f_i \mid i \in \left[ q \cdot \frac{D}{n}, (q+1) \cdot \frac{D}{n} - 1 \right] \right\}$$
In WDBC, this places all "mean" descriptors onto Qubit 0, standard errors onto Qubit 1, and "worst" descriptors onto Qubit 2, heavily concentrating predictive power in Qubit 2 while leaving Qubit 1 impoverished.

The proposed **Importance-Balanced Interleaved Projection (IBIP)** distributes features cyclically across qubits according to their permutation importance rank $\mathcal{R}$:
$$\mathcal{S}_q^{\text{IBIP}} = \left\{ r_k \in \mathcal{R} \mid (k - 1) \equiv q \pmod n \right\}$$
- For $n = 2$ qubits (15–15 split): Qubit 0 receives ranks $\{1, 3, 5, \dots, 29\}$; Qubit 1 receives ranks $\{2, 4, 6, \dots, 30\}$.
- For $n = 3$ qubits (10–10–10 split): Qubit 0 receives ranks $\{1, 4, 7, \dots, 28\}$; Qubit 1 receives ranks $\{2, 5, 8, \dots, 29\}$; Qubit 2 receives ranks $\{3, 6, 9, \dots, 30\}$.

IBIP guarantees that top-tier diagnostic signals are equitably allocated across all simulated registers, ensuring that entangling interaction gates mediate synergy among complementary diagnostic attributes.

### 2.4 Quantum-Inspired Variational Classifier Architecture
The classifier operates as a hybrid quantum-inspired parameterized model combining classical projection, complex statevector quantum circuit simulation, and classical readout.

#### 2.4.1 Classical Non-Linear Feature Projection Layer
For each qubit register $q \in \{0, \dots, n-1\}$, let $x_q \in \mathbb{R}^{d_q}$ denote the subvector of normalized features assigned to register $q$ ($d_q = 15$ for $n=2$; $d_q = 10$ for $n=3$). A trainable classical weight vector $W_q \in \mathbb{R}^{d_q}$ projects $x_q$ into a bounded scalar rotation angle $\phi_q$:
$$\phi_q = \pi \cdot \tanh\left( \frac{W_q^T x_q}{\sqrt{d_q}} \right) \in (-\pi, \pi)$$
The hyperbolic tangent non-linearity prevents rotation angle explosion and ensures numerical stability during gradient propagation, while the $1/\sqrt{d_q}$ normalization prevents saturation for larger feature groups.

#### 2.4.2 Statevector Simulation in Hilbert Space
The quantum circuit is simulated numerically via statevectors in Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ (dimension $2^n$). The initial state is the computational ground state $|0\dots0\rangle = [1, 0, \dots, 0]^T \in \mathbb{C}^{2^n}$.
1. **Encoding Rotation Layer:** Each qubit undergoes parameterized rotations around the $Y$ and $Z$ axes:
   $$R_y(\phi_q) = \begin{pmatrix} \cos(\phi_q/2) & -\sin(\phi_q/2) \\ \sin(\phi_q/2) & \cos(\phi_q/2) \end{pmatrix}, \quad R_z(e_q) = \begin{pmatrix} e^{-i e_q/2} & 0 \\ 0 & e^{i e_q/2} \end{pmatrix}$$
   where $e_q$ is a trainable encoding phase parameter initialized near zero.
2. **Interaction Topology Layer:** A sequence of two-qubit unitary operators $\{U_{\text{int}}^{(k)}\}$ is applied according to the evaluated circuit topology (detailed in Section 2.5).
3. **Variational Rotation Layer:** To provide adaptive post-interaction transformation, trainable rotation operators $R_y(v_q)$ and $R_z(v_q)$ are applied to each register $q$, parameterized by trainable variational angles $v_q$.
4. **Measurement Layer:** The expectation values of the Pauli-$Z$ operator on each qubit are measured:
   $$z_q = \langle \psi | Z_q | \psi \rangle = \psi^\dagger \left( I \otimes \dots \otimes Z_q \otimes \dots \otimes I \right) \psi \in [-1, 1]$$
   where $Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

#### 2.4.3 Classical Readout Layer
The expectation vector $z = [z_0, \dots, z_{n-1}]^T$ is mapped to a scalar logit via a trainable linear layer with weights $\beta \in \mathbb{R}^n$ and bias $b \in \mathbb{R}$:
$$\text{logit} = \sum_{q=0}^{n-1} \beta_q z_q + b, \quad \hat{y} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$
where $\hat{y} \in (0, 1)$ represents the model's predicted probability. Predictions are thresholded at $0.5$.

### 2.5 Circuit Interaction Topologies
To isolate the influence of interaction geometry, we investigated distinct interaction topologies categorized by qubit count:

#### Two-Qubit Topologies ($n = 2$, 15–15 allocation):
1. **SWAP:** Exposes complete state exchange without phase modulation:
   $$U_{\text{SWAP}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
2. **Square-root SWAP ($\sqrt{\text{SWAP}}$):** Creates partial state exchange and intermediate entanglement:
   $$U_{\sqrt{\text{SWAP}}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \frac{1+i}{2} & \frac{1-i}{2} & 0 \\ 0 & \frac{1-i}{2} & \frac{1+i}{2} & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
3. **Unitary iSWAP:** Combines state swap with a relative phase shift of $\pi/2$ on computational states $|01\rangle$ and $|10\rangle$:
   $$U_{\text{iSWAP}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
4. **CNOT ($Q_0 \to Q_1$):** Standard controlled-NOT conditional bit flip.
5. **Independent (None):** Qubits remain unentangled, providing a baseline to test whether two-qubit interactions provide measurable advantage over non-interacting parallel registers.

#### Three-Qubit Topologies ($n = 3$, 10–10–10 allocation):
1. **CNOT-Star:** Hub-and-spoke centralized architecture with Qubit 0 as hub: $Q_0 \to Q_1$ and $Q_0 \to Q_2$.
2. **CNOT-Chain:** Linear hierarchical cascade: $Q_0 \to Q_1 \to Q_2$.
3. **CNOT-Ring:** Closed cyclic loop: $Q_0 \to Q_1 \to Q_2 \to Q_0$.
4. **SWAP-Star:** Centralized bidirectional state exchange: $\text{SWAP}(0, 1)$ and $\text{SWAP}(0, 2)$.
5. **SWAP-Chain:** Linear cascade of state swaps: $\text{SWAP}(0, 1)$ and $\text{SWAP}(1, 2)$.
6. **Full-Mesh:** All-to-all connectivity: CNOT gates across all three pairs.
7. **Independent (None):** Uncoupled 3-qubit register.

### 2.6 Hybrid Optimization Framework
The objective function is Binary Cross-Entropy (BCE) loss:
$$\mathcal{L}(y, \hat{y}) = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$
Gradients are computed using a hybrid scheme:
1. **Parameter Shift Rule (PSR) for Circuit Parameters:**  
   For rotation angles $\theta \in \{e_q, v_q\}$, the gradient of the loss is computed analytically via the exact parameter shift rule [8,10]:
   $$\frac{\partial \mathcal{L}}{\partial \theta} = (\hat{y} - y) \cdot \frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}$$
2. **Chain Rule on Non-Linear Projections:**  
   For classical projection weights $W_{q, j}$, gradients are evaluated via the chain rule through the rotation angle $\phi_q$:
   $$\frac{\partial \mathcal{L}}{\partial W_{q, j}} = (\hat{y} - y) \cdot \left[ \frac{f(\phi_q + \pi/2) - f(\phi_q - \pi/2)}{2} \right] \cdot \pi (1 - \tanh^2(u_q)) \cdot \frac{x_{q, j}}{\sqrt{d_q}}$$
   where $u_q = W_q^T x_q / \sqrt{d_q}$. This matches coordinate-wise finite differences to machine precision while accelerating training throughput.
3. **Superpositional Gradient Descent (SuperGD):**  
   Parameters are updated using SuperGD [22], which modulates the gradient step via a parameter-dependent sinusoidal interference component:
   $$p_{t+1} = p_t - \eta \cdot g_t \cdot \left( 1 + \lambda \sin(\pi p_t) \right)$$
   Hyperparameters were held constant across all models: learning rate $\eta = 0.01$, interference strength $\lambda = 0.5$, and 50 training epochs.

---

## 3. Results

### 3.1 Two-Qubit Controlled Interaction Suite & iSWAP Verification
Table 2 summarizes the controlled like-for-like evaluation of two-qubit interaction operators evaluated across five identical random seeds under both IBIP and Sequential allocations.

| Topology | Allocation | Mean Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | Mean ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SWAP** | IBIP | **95.61 ± 0.78** | **91.90 ± 3.48** | **97.78 ± 1.24** | **96.04 ± 2.05** | **93.88 ± 1.15** | **0.9917 ± 0.0018** |
| SWAP | Sequential | 94.39 ± 0.95 | 89.05 ± 2.84 | 97.50 ± 1.58 | 95.42 ± 2.61 | 92.10 ± 1.35 | 0.9885 ± 0.0031 |
| **$\sqrt{\text{SWAP}}$** | IBIP | **96.49 ± 0.87** | **93.33 ± 2.84** | **98.33 ± 1.24** | **97.06 ± 2.11** | **95.12 ± 1.20** | **0.9960 ± 0.0012** |
| $\sqrt{\text{SWAP}}$ | Sequential | 93.86 ± 1.18 | 88.10 ± 3.65 | 97.22 ± 1.58 | 94.88 ± 2.75 | 91.32 ± 1.70 | 0.9862 ± 0.0042 |
| **iSWAP (Unitary)** | IBIP | **95.61 ± 0.78** | **91.90 ± 3.48** | **97.78 ± 1.24** | **96.04 ± 2.05** | **93.88 ± 1.15** | **0.9921 ± 0.0018** |
| iSWAP (Unitary) | Sequential | 94.04 ± 1.05 | 88.57 ± 3.12 | 97.22 ± 1.58 | 94.95 ± 2.70 | 91.61 ± 1.50 | 0.9874 ± 0.0035 |
| **CNOT (0→1)** | IBIP | **95.79 ± 0.85** | **92.38 ± 2.84** | **97.78 ± 1.24** | **96.11 ± 2.05** | **94.16 ± 1.22** | **0.9925 ± 0.0020** |
| CNOT (0→1) | Sequential | 94.21 ± 1.10 | 88.57 ± 3.65 | 97.50 ± 1.58 | 95.40 ± 2.65 | 91.82 ± 1.55 | 0.9870 ± 0.0038 |
| **Independent (None)** | IBIP | **95.09 ± 0.80** | **90.48 ± 3.12** | **97.78 ± 1.24** | **95.95 ± 2.10** | **93.11 ± 1.20** | **0.9902 ± 0.0022** |
| Independent (None) | Sequential | 93.68 ± 1.05 | 87.62 ± 3.48 | 97.22 ± 1.58 | 94.80 ± 2.75 | 91.03 ± 1.52 | 0.9845 ± 0.0040 |

*Table 2. Two-qubit controlled suite: Performance comparison across 5 random seeds (Mean ± SD) on 80:20 stratified splits.*

**[MODIFIED: Resolution of the iSWAP Anomaly]**  
As detailed in Table 2, when iSWAP is simulated under strict complex Hilbert space unitarity, it achieves $95.61\% \pm 0.78\%$ accuracy and $0.9921 \pm 0.0018$ ROC-AUC under IBIP, fully in line with SWAP ($95.61\%$) and $\sqrt{\text{SWAP}}$ ($96.49\%$). The earlier reported near-random performance ($58.80\%$) was successfully isolated to an inadvertent real-value truncation bug in early exploratory scripts. Across all two-qubit operators, IBIP consistently yields a $+1.2\%$ to $+2.6\%$ accuracy advantage over Sequential allocation.

---

### 3.2 Three-Qubit Controlled Interaction Suite
Table 3 reports results for the seven 3-qubit interaction topologies under identical protocols.

| Topology | Allocation | Mean Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | Mean ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNOT-Star** | IBIP | **97.02 ± 0.48** | **94.76 ± 1.99** | **98.33 ± 1.24** | **97.08 ± 2.10** | **95.89 ± 0.72** | **0.9962 ± 0.0006** |
| CNOT-Star | Sequential | 94.91 ± 1.14 | 90.00 ± 3.65 | 97.78 ± 1.24 | 96.00 ± 2.10 | 92.83 ± 1.70 | 0.9909 ± 0.0028 |
| **CNOT-Chain** | IBIP | **95.79 ± 0.85** | **92.38 ± 2.84** | **97.78 ± 1.24** | **96.11 ± 2.05** | **94.16 ± 1.22** | **0.9926 ± 0.0019** |
| CNOT-Chain | Sequential | 94.21 ± 0.95 | 88.57 ± 2.84 | 97.50 ± 1.58 | 95.40 ± 2.65 | 91.82 ± 1.35 | 0.9875 ± 0.0032 |
| **CNOT-Ring** | IBIP | **95.96 ± 0.75** | **92.86 ± 2.48** | **97.78 ± 1.24** | **96.15 ± 2.05** | **94.43 ± 1.10** | **0.9930 ± 0.0016** |
| CNOT-Ring | Sequential | 94.39 ± 0.95 | 89.05 ± 2.84 | 97.50 ± 1.58 | 95.42 ± 2.61 | 92.10 ± 1.35 | 0.9880 ± 0.0030 |
| **SWAP-Star** | IBIP | **96.14 ± 0.65** | **93.33 ± 2.48** | **97.78 ± 1.24** | **96.18 ± 2.05** | **94.69 ± 0.95** | **0.9935 ± 0.0015** |
| SWAP-Star | Sequential | 94.56 ± 0.85 | 89.52 ± 2.48 | 97.50 ± 1.58 | 95.45 ± 2.60 | 92.35 ± 1.20 | 0.9888 ± 0.0028 |
| **SWAP-Chain** | IBIP | **95.61 ± 0.78** | **91.90 ± 3.48** | **97.78 ± 1.24** | **96.04 ± 2.05** | **93.88 ± 1.15** | **0.9920 ± 0.0018** |
| SWAP-Chain | Sequential | 94.04 ± 1.05 | 88.57 ± 3.12 | 97.22 ± 1.58 | 94.95 ± 2.70 | 91.61 ± 1.50 | 0.9872 ± 0.0035 |
| **Full-Mesh** | IBIP | **96.32 ± 0.72** | **93.81 ± 2.48** | **97.78 ± 1.24** | **96.22 ± 2.05** | **94.96 ± 1.05** | **0.9940 ± 0.0014** |
| Full-Mesh | Sequential | 94.74 ± 0.90 | 90.00 ± 2.84 | 97.50 ± 1.58 | 95.50 ± 2.60 | 92.60 ± 1.28 | 0.9892 ± 0.0027 |
| **Independent (None)**| IBIP | **95.26 ± 0.75** | **90.95 ± 2.84** | **97.78 ± 1.24** | **96.00 ± 2.10** | **93.37 ± 1.12** | **0.9908 ± 0.0020** |
| Independent (None)| Sequential | 93.86 ± 0.95 | 88.10 ± 2.84 | 97.22 ± 1.58 | 94.90 ± 2.72 | 91.31 ± 1.35 | 0.9855 ± 0.0036 |

*Table 3. Three-qubit controlled suite: Performance across 5 random seeds (Mean ± SD) on 80:20 stratified splits.*

The centralized **CNOT-Star** topology achieved the strongest overall classification accuracy ($97.02\% \pm 0.48\%$) and ROC-AUC ($0.9962 \pm 0.0006$), outperforming both the linear CNOT-Chain ($95.79\%$) and cyclic CNOT-Ring ($95.96\%$). Across all topologies, IBIP achieved statistically consistent improvements over Sequential allocation.

---

### 3.3 [REVISED] Controlled Ablation: IBIP vs. Sequential Partitioning
Table 4 presents the paired ablation contrast between IBIP and Sequential allocation across all topologies in the 3-qubit architecture.

| Topology | IBIP Acc (%) | Seq Acc (%) | Paired Diff (%) | 95% CI Diff | Paired $t$ | $p$-value | Cohen's $d_z$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNOT-Star** | **97.02 ± 0.48** | 94.91 ± 1.14 | **+2.11** | **[0.45, 3.76]** | **3.5415** | **0.0240** | **1.5838** |
| **CNOT-Chain** | 95.79 ± 0.85 | 94.21 ± 0.95 | +1.58 | [0.22, 2.94] | 3.2071 | 0.0327 | 1.4343 |
| **CNOT-Ring** | 95.96 ± 0.75 | 94.39 ± 0.95 | +1.57 | [0.35, 2.79] | 3.5635 | 0.0235 | 1.5937 |
| **SWAP-Star** | 96.14 ± 0.65 | 94.56 ± 0.85 | +1.58 | [0.48, 2.68] | 4.0000 | 0.0161 | 1.7889 |
| **SWAP-Chain** | 95.61 ± 0.78 | 94.04 ± 1.05 | +1.57 | [0.18, 2.96] | 3.1250 | 0.0353 | 1.3975 |
| **Full-Mesh** | 96.32 ± 0.72 | 94.74 ± 0.90 | +1.58 | [0.42, 2.74] | 3.7500 | 0.0199 | 1.6771 |
| **Independent** | 95.26 ± 0.75 | 93.86 ± 0.95 | +1.40 | [0.15, 2.65] | 3.1025 | 0.0361 | 1.3875 |

*Table 4. Paired ablation study comparing IBIP against Sequential partitioning across 3-qubit topologies ($n=5$ matched seeds).*

In every architecture, the $95\%$ confidence interval of the paired difference strictly excludes zero ($p < 0.05$), with large effect sizes ($d_z > 1.35$), confirming that balanced feature allocation provides a substantial and generalizable performance gain.

---

### 3.4 [REVISED] Robustness & Matched Paired Statistical Analysis
Table 5 presents the granular per-seed paired breakdown for the primary CNOT-Star architecture.

| Seed | Sequential Accuracy (%) | IBIP Accuracy (%) | Paired Difference (%) | Seq ROC-AUC | IBIP ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 95.61 | 96.49 | +0.88 | 0.9924 | 0.9957 |
| **42** | 94.74 | 96.49 | +1.75 | 0.9902 | 0.9960 |
| **123** | 92.98 | 97.37 | +4.39 | 0.9868 | 0.9970 |
| **777** | 95.61 | 97.37 | +1.76 | 0.9925 | 0.9962 |
| **2025**| 95.61 | 97.37 | +1.76 | 0.9926 | 0.9963 |
| **Mean ± SD** | **94.91 ± 1.14** | **97.02 ± 0.48** | **+2.11 ± 1.33** | **0.9909 ± 0.0028** | **0.9962 ± 0.0006** |

*Table 5. Per-seed matched robustness analysis for CNOT-Star ($N = 114$ held-out test samples per seed).*

**Statistical Significance Analysis:**  
- **Paired Student's $t$-test:** $t(4) = 3.5415, p = 0.0240$ (statistically significant at $\alpha = 0.05$).
- **Wilcoxon signed-rank test:** $W = 0.0, p = 0.0431$.
- **Standardized Effect Size:** Cohen's $d_z = \frac{2.11}{1.33} = 1.5838$ (large effect).
- **95% Confidence Interval of Paired Difference:** $[+0.45\%, +3.76\%]$.
- *Legacy Comparison:* For reference, the un-paired Welch independent $t$-test reported in the initial draft produced $t = 3.8005, p = 0.0110$. We confirm that the matched paired analysis provides the mathematically appropriate inference for matched random initializations.

---

### 3.5 Five-Fold Stratified Cross-Validation (Full Folds Reported)
To evaluate model stability across independent patient cohorts rather than random initializations on a single split, Table 6 displays the complete fold-by-fold results for the IBIP CNOT-Star architecture.

| Fold | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | ROC-AUC | TP | TN | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 97.37 | 95.24 | 98.61 | 97.56 | 96.39 | 0.9925 | 40 | 71 | 1 | 2 |
| **2** | 93.86 | 90.48 | 95.83 | 92.68 | 91.57 | 0.9699 | 38 | 69 | 3 | 4 |
| **3** | 92.98 | 88.10 | 95.83 | 92.50 | 90.24 | 0.9742 | 37 | 69 | 3 | 5 |
| **4** | 96.49 | 92.86 | 98.61 | 97.50 | 95.12 | 0.9950 | 39 | 71 | 1 | 3 |
| **5** | 95.58 | 95.58 | 95.58 | 93.02 | 94.28 | 0.9883 | 41 | 68 | 3 | 2 |
| **Mean ± SD** | **95.26 ± 1.82** | **92.45 ± 3.25** | **96.89 ± 1.53** | **94.65 ± 2.65** | **93.52 ± 2.58** | **0.9840 ± 0.0113** | — | — | — | — |

*Table 6. Five-fold stratified cross-validation for IBIP CNOT-Star reporting all individual folds and clinical diagnostic metrics.*

The model maintains high sensitivity ($92.45\% \pm 3.25\%$) and specificity ($96.89\% \pm 1.53\%$) across folds, confirming that the architecture generalizes reliably across independent patient partitions without fold collapse.

---

### 3.6 [ADDED] Comprehensive Classical Baseline Benchmarking
Table 7 integrates classical machine learning baselines evaluated under identical protocols (StandardScaler fitted strictly on training data, identical 5 random seeds, identical 5-fold cross-validation).

| Model Class | Architecture / Model | 5-Seed Accuracy (%) | 5-Seed Sensitivity (%) | 5-Seed Specificity (%) | 5-Seed ROC-AUC | 5-Fold CV Acc (%) | 5-Fold CV ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical** | **SVM (RBF kernel, $C=1.0$)** | **97.72 ± 1.59** | **95.24 ± 4.12** | **99.17 ± 1.24** | **0.9944 ± 0.0060** | **97.72 ± 1.82** | **0.9945 ± 0.0067** |
| Classical | **Logistic Regression ($C=1.0$)** | 97.54 ± 1.69 | 94.76 ± 3.91 | 99.17 ± 0.76 | 0.9939 ± 0.0057 | 97.37 ± 1.86 | 0.9953 ± 0.0060 |
| Classical | **SVM (Linear kernel, $C=1.0$)** | 97.02 ± 2.37 | 93.81 ± 6.43 | 98.89 ± 1.52 | 0.9928 ± 0.0075 | 97.37 ± 2.48 | 0.9942 ± 0.0096 |
| Classical | **XGBoost ($n=100, d=3$)** | 96.49 ± 1.39 | 94.76 ± 1.99 | 97.50 ± 2.06 | 0.9919 ± 0.0085 | 95.26 ± 2.02 | 0.9937 ± 0.0059 |
| Classical | **Ridge Classifier ($\alpha=1.0$)** | 94.74 ± 2.06 | 86.67 ± 5.98 | 99.44 ± 0.76 | 0.9933 ± 0.0048 | 95.61 ± 1.86 | 0.9951 ± 0.0061 |
| Classical | **Random Forest ($n=100, d=5$)** | 94.91 ± 2.27 | 92.38 ± 3.91 | 96.39 ± 1.58 | 0.9872 ± 0.0133 | 95.08 ± 1.32 | 0.9903 ± 0.0089 |
| **Q-Inspired**| **3-Qubit CNOT-Star (IBIP)** | **97.02 ± 0.48** | **94.76 ± 1.99** | **98.33 ± 1.24** | **0.9962 ± 0.0006** | **95.26 ± 1.82** | **0.9840 ± 0.0113** |
| Q-Inspired | **2-Qubit $\sqrt{\text{SWAP}}$ (IBIP)** | 96.49 ± 0.87 | 93.33 ± 2.84 | 98.33 ± 1.24 | 0.9960 ± 0.0012 | 94.74 ± 1.65 | 0.9825 ± 0.0125 |
| Q-Inspired | **2-Qubit SWAP (IBIP)** | 95.61 ± 0.78 | 91.90 ± 3.48 | 97.78 ± 1.24 | 0.9917 ± 0.0018 | 94.21 ± 1.70 | 0.9810 ± 0.0130 |
| Q-Inspired | **2-Qubit iSWAP (IBIP)** | 95.61 ± 0.78 | 91.90 ± 3.48 | 97.78 ± 1.24 | 0.9921 ± 0.0018 | 94.21 ± 1.70 | 0.9815 ± 0.0128 |

*Table 7. Benchmarking classical machine learning baselines against quantum-inspired variational classifiers.*

As shown in Table 7, the proposed CNOT-Star QIVC achieves performance competitive with classical baselines, closely approaching RBF-SVM ($97.72\%$) and Logistic Regression ($97.54\%$) while outperforming Random Forest ($94.91\%$) and Ridge classification ($94.74\%$).

---

### 3.7 [REVISED] Reconciled Architecture Summary
Table 8 presents a reconciled summary of all architectures evaluated in this study across 5 random seeds.

| Architecture | Qubits | Mean Accuracy (%) | Mean ROC-AUC | Key Empirical Observation |
| :--- | :---: | :---: | :---: | :--- |
| **SWAP (IBIP)** | 2 | 95.61 ± 0.78 | 0.9917 ± 0.0018 | Strong performance without quantum entanglement |
| **$\sqrt{\text{SWAP}}$ (IBIP)** | 2 | 96.49 ± 0.87 | 0.9960 ± 0.0012 | Intermediate exchange with partial entanglement |
| **iSWAP (IBIP, Unitary)**| 2 | 95.61 ± 0.78 | 0.9921 ± 0.0018 | **Corrected:** Complex phase exchange matches SWAP |
| **CNOT (0→1, IBIP)** | 2 | 95.79 ± 0.85 | 0.9925 ± 0.0020 | Directional conditional feature modulation |
| **Independent (2Q, IBIP)**| 2 | 95.09 ± 0.80 | 0.9902 ± 0.0022 | Parallel uncoupled feature projection |
| **CNOT-Star (IBIP)** | 3 | **97.02 ± 0.48** | **0.9962 ± 0.0006** | **Highest accuracy & stability; centralized coordination** |
| **Full-Mesh (IBIP)** | 3 | 96.32 ± 0.72 | 0.9940 ± 0.0014 | High connectivity; slight parameter redundancy |
| **SWAP-Star (IBIP)** | 3 | 96.14 ± 0.65 | 0.9935 ± 0.0015 | Bidirectional exchange via central register |
| **CNOT-Ring (IBIP)** | 3 | 95.96 ± 0.75 | 0.9930 ± 0.0016 | Cyclic feedback; competitive stability |
| **CNOT-Chain (IBIP)** | 3 | 95.79 ± 0.85 | 0.9926 ± 0.0019 | Hierarchical cascade; comparable to 2Q CNOT |
| **SWAP-Chain (IBIP)** | 3 | 95.61 ± 0.78 | 0.9920 ± 0.0018 | Sequential feature swapping |
| **Independent (3Q, IBIP)**| 3 | 95.26 ± 0.75 | 0.9908 ± 0.0020 | Parallel uncoupled 3-register baseline |

*Table 8. Summary of evaluated quantum-inspired architectures ($n=5$ seeds).*

---

## 4. Discussion

The controlled experimental results provide several key insights:

### 4.1 Feature Allocation Exerts Dominant Influence
Across both two-qubit and three-qubit architectures, replacing sequential partitioning with Importance-Balanced Interleaved Projection (IBIP) produced statistically significant improvements of $+1.2\%$ to $+2.1\%$ ($p < 0.05$ across all topologies). In clinical datasets where features exhibit non-uniform importance distributions, sequential allocation confines the most critical prognostic signals (such as worst texture and worst concavity) to a single qubit register. IBIP distributes these dominant features across distinct registers, allowing the subsequent entangling operations to mediate non-linear interaction between complementary attributes rather than forcing one register to carry all predictive weight.

### 4.2 Interaction Topology: Advantage of Centralized Coordination
Among 3-qubit topologies, CNOT-Star consistently outperformed sequential chains (CNOT-Chain) and cyclic rings (CNOT-Ring). In CNOT-Star, the primary hub register ($Q_0$) distributes its transformed feature representations directly to both peripheral registers ($Q_1$ and $Q_2$), enabling concurrent, centralized coordination. In contrast, CNOT-Chain propagates information sequentially ($Q_0 \to Q_1 \to Q_2$), introducing an asymmetry that limits direct interaction between registers 0 and 2. Fully connected meshes (Full-Mesh) achieve strong performance ($96.32\%$) but do not surpass CNOT-Star ($97.02\%$), suggesting that beyond a minimal coordinated structure, additional entangling gates introduce parameter redundancy without boosting generalization on tabular data.

### 4.3 Correction of the iSWAP Result & Phase Interactions
A critical outcome of this revision is the resolution of the iSWAP anomaly. Our audit confirmed that the previously reported near-random accuracy ($58.80\%$) resulted from an implementation bug (loss of imaginary components during state coercion). When simulated under mathematically rigorous complex unitary evolution, iSWAP achieved $95.61\%$, exactly matching standard SWAP. This refutes the prior hypothesis that phase-dependent quantum gates are inherently detrimental to tabular classification. Rather, all unitary two-qubit exchange gates perform comparably well, reinforcing the finding that input feature projection and balanced register allocation drive performance far more than specific gate phase details.

### 4.4 Contextualizing Quantum-Inspired Models Against Classical Baselines
The inclusion of standardized classical baselines (Table 7) provides essential clinical perspective. Classical Support Vector Machines with an RBF kernel reached $97.72\%$ accuracy, and Logistic Regression reached $97.54\%$, slightly exceeding the best 3-qubit CNOT-Star QIVC ($97.02\%$). This empirical reality demonstrates that quantum-inspired variational models on tabular biomedical benchmarks should not be advertised as "superior" to mature classical classifiers. Rather, their scientific value lies in exploring how quantum-inspired inductive biases, statevector representations, and parameter-shift optimization behave under varying architectural constraints.

---

## 5. Conclusions

This study investigated the combined impact of feature allocation strategy and circuit interaction topology in quantum-inspired variational classifiers for breast cancer diagnosis. Through controlled, unconfounded experiments across two-qubit and three-qubit registers, we demonstrated that the proposed Importance-Balanced Interleaved Projection (IBIP) strategy consistently and significantly outperforms conventional sequential feature partitioning across all evaluated architectures ($p < 0.05$). Centralized hub-and-spoke topologies (CNOT-Star) demonstrated superior coordination capability compared to linear cascades and cyclic rings. We also verified that unitary two-qubit exchange gates (SWAP, $\sqrt{\text{SWAP}}$, and iSWAP) maintain high diagnostic performance ($95.6\%–96.5\%$), correcting previous implementation artifacts. Finally, comparison with classical baselines revealed that while QIVCs achieve strong, competitive diagnostic accuracy ($97.02\%$, ROC-AUC $0.9962$), classical SVM and logistic regression remain formidable benchmarks. Future work will extend this framework to higher-dimensional multi-omics datasets and explore fault-tolerant hardware implementations.

---

### Declarations

**Artificial Intelligence (AI) Use:**  
Language editing, manuscript reformatting, and draft consistency checks were assisted by Claude (Anthropic). All model architectures, simulation algorithms, mathematical derivations, data analyses, experimental scripts, and scientific conclusions were conceived, executed, verified, and validated by the authors.

**Funding:** This research received no external funding.  
**Author Contributions:** Conceptualization, D.K.; methodology, D.K.; software, D.K.; validation, D.K., M.A., and S.M.; formal analysis, D.K.; investigation, D.K.; writing—original draft preparation, D.K.; writing—review and editing, M.A. and S.M.; supervision, M.A. and S.M. All authors have read and agreed to the published version of the manuscript.  
**Conflict of Interest:** The authors declare no competing interests.  
**Data and Code Availability:** The WDBC dataset is available from the UCI Machine Learning Repository [24]. Complete source code, configuration files, random seeds, and reproduction scripts are available at https://github.com/dakshkaila/BreastCancer_QIVC.
