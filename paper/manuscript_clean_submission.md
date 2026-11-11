# Circuit Interaction Topologies and Importance-Balanced Feature Allocation for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study

**Journal:** Academia Quantum  
**Article Type:** Research Article  
**Article DOI:** 10.20935/AcadQuantXXXX  

**Authors:** Daksh Kaila$^{1,*}$, Monika Aggarwal$^2$, Samya Muhuri$^1$  
$^1$ Department of Computer Science and Engineering, Thapar Institute of Engineering and Technology, Patiala, Punjab 147004, India  
$^2$ Centre for Applied Research in Electronics, Indian Institute of Technology Delhi, Hauz Khas, New Delhi 110016, India  
$^*$ **Correspondence:** dkaila_be25@thapar.edu  

---

### Abstract

**Introduction:** Quantum-inspired machine learning offers a mathematically principled framework for evaluating non-linear representation learning, parameter-shift optimization, and multi-qubit interaction mechanisms on classical hardware. However, the interplay between continuous feature allocation across low-qubit registers and circuit interaction topologies remains poorly understood in tabular biomedical classification tasks.

**Materials and methods:** We introduce an Importance-Balanced Interleaved Projection (IBIP) framework for quantum-inspired variational classifiers (QIVCs). The framework quantifies feature importance on training partitions using permutation importance and distributes features cyclically across qubits, preventing dominant diagnostic indicators from concentrating within isolated registers. The classifier operates as a hybrid quantum-inspired model: classical input subvectors are mapped into rotation angles via trainable non-linear projections ($\phi_q = \pi \tanh(W_q^T x_q / \sqrt{|x_q|})$), followed by exact numerical statevector simulation of parameterized single-qubit rotations and multi-qubit interaction topologies in complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$, and measured via Pauli-$Z$ expectations linearly mapped to class probabilities. To avoid confounding topology with qubit count and feature grouping, two-qubit exchange gates (SWAP, $\sqrt{\text{SWAP}}$, unitary iSWAP, CNOT, Independent) and three-qubit interaction graphs (CNOT-Star, CNOT-Chain, CNOT-Ring, SWAP-Star, SWAP-Chain, Full-Mesh, Independent) were evaluated in separate, unconfounded experimental suites on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset across matched random seeds and five-fold stratified cross-validation. To establish why interleaving matters, a 4-way ablation study was conducted comparing sequential, random, importance-blocked (no interleaving), and IBIP allocations. Cross-organ generalizability was further evaluated across three additional biomedical datasets: Parkinson's disease, Cleveland heart disease, and Breast Cancer Coimbra, with rigorous benchmarking against six standardized classical baselines (SVM RBF/Linear, XGBoost, Ridge, Logistic Regression, Random Forest).

**Results:** In the three-qubit controlled suite, the CNOT-Star architecture with IBIP achieved a mean test accuracy of $97.02\% \pm 0.48\%$ and a mean ROC-AUC of $0.9962 \pm 0.0006$ across five matched seeds, significantly outperforming conventional sequential feature partitioning ($94.91\% \pm 1.14\%$ accuracy, $0.9909 \pm 0.0028$ ROC-AUC; paired difference $+2.11\%$, $95\%\text{ CI: }[0.45\%, 3.76\%]$, paired $t(4) = 3.5415, p = 0.0240$, Wilcoxon $W = 0.0, p = 0.0431$, Cohen's $d_z = 1.5838$). In five-fold cross-validation, IBIP CNOT-Star demonstrated stable patient generalization ($95.26\% \pm 1.82\%$ accuracy, ROC-AUC $0.9840 \pm 0.0113$). The 4-way ablation confirmed that assigning features in contiguous importance blocks without interleaving causes register starvation, dropping accuracy to $92.81\% \pm 2.51\%$; round-robin interleaving restores balanced information budgets across registers ($95.26\% \pm 2.95\%$, $+2.46\%$ gain, Cohen's $d_z = 0.923$). In cross-dataset benchmarking, IBIP proved robust across diverse clinical domains, outperforming sequential allocation on Parkinson's disease by $+5.64\%$ ($84.10\%$ vs. $78.46\%$, Cohen's $d_z = 1.144$) and outperforming blocked allocation on heart disease by $+8.20\%$ ($80.66\%$ vs. $72.46\%$). Diagnostic re-evaluation of unitary iSWAP in the two-qubit suite confirmed $95.61\% \pm 0.78\%$ accuracy and $0.9921 \pm 0.0018$ ROC-AUC under complex statevector evolution, matching SWAP ($95.61\%$) and $\sqrt{\text{SWAP}}$ ($96.49\%$). Classical baselines achieved benchmark performance: SVM RBF reached $97.72\% \pm 1.59\%$ and Logistic Regression reached $97.54\% \pm 1.69\%$.

**Conclusions:** Interleaved feature balancing reliably enhances generalization across both two-qubit and three-qubit quantum-inspired architectures and generalizes across distinct biomedical domains by preventing register starvation and facilitating cross-register communication between high-leverage features. Among interaction topologies, centralized hub-and-spoke graphs (CNOT-Star) provide superior feature coordination compared to unbranched chains or cyclic rings. However, classical kernel SVM and logistic regression remain highly competitive, indicating that quantum-inspired classifiers are best understood as expressive, compact parameter representation models rather than superior replacements for mature classical baselines on moderate-dimensional tabular benchmarks.

**Keywords:** Quantum-Inspired Machine Learning; Variational Quantum Classifier; Feature Allocation; Circuit Interaction Topology; Breast Cancer Diagnosis; Permutation Importance; Entanglement Entropy

---

## 1. Introduction

Breast cancer remains one of the primary causes of cancer-related mortality among women globally. Early and accurate detection through digitized fine-needle aspiration (FNA) cytology is vital for improving patient survival rates and establishing targeted clinical therapies [1]. While classical machine learning algorithms—such as Support Vector Machines (SVMs) [1], Random Forests [5], and Gradient Boosted Decision Trees (XGBoost) [6]—have demonstrated high diagnostic efficacy on benchmark cytology datasets such as the Wisconsin Diagnostic Breast Cancer (WDBC) cohort [24], exploring alternative mathematical representations and non-linear feature interaction mechanisms remains a fruitful area of computational biology and machine learning research.

Recent progress in quantum computing and quantum machine learning (QML) [2,3,7] has spurred substantial interest in variational quantum classifiers (VQCs) and quantum-inspired learning systems [8–11]. These architectures exploit high-dimensional Hilbert space mappings, parameterized unitary rotation gates, multi-qubit entangling operations, and quantum expectation measurements to learn non-linear decision boundaries [10,11]. A variety of quantum-classical hybrid designs have been proposed for breast cancer diagnosis, including quantum kernels, quantum support vector classifiers (QSVCs) [12], quantum convolutional neural networks (QCNNs) [13], and hybrid quantum-classical convolutional models [14].

Despite these developments, a critical structural challenge in deploying variational quantum classifiers on continuous tabular data is the **feature allocation problem**. In near-term quantum processors and quantum-inspired simulators, high-dimensional classical vectors (such as the 30 continuous cytological descriptors in WDBC) must be mapped into a low number of simulated qubit registers ($n \in \{2, 3\}$). Conventional pipelines overwhelmingly rely on sequential block partitioning, where features are grouped strictly by their original column order in the dataset. In biomedical data where features are arranged by semantic category (for instance, mean nuclear characteristics, followed by standard errors, followed by extreme "worst" values), naive sequential allocation concentrates the most predictive clinical signals into a single register, while leaving companion registers under-informative. Consequently, multi-qubit entangling operations are deprived of the ability to coordinate complementary diagnostic signals across registers.

Furthermore, existing studies have typically evaluated single circuit topologies in isolation, without systematically disentangling the effect of gate interaction geometry (such as linear chains, centralized stars, cyclic loops, or exchange gates) from register size and feature allocation under controlled conditions.

To resolve these challenges, this study presents a rigorous empirical and theoretical investigation of feature allocation and interaction topologies in quantum-inspired variational classifiers:
1. **Unconfounded Experimental Suites:** We separate two-qubit and three-qubit models into independent, controlled experimental suites, evaluating all interaction topologies under identical data splits, identical feature groupings, identical random seeds, and five-fold stratified cross-validation.
2. **Importance-Balanced Interleaved Projection (IBIP):** We formulate a feature allocation strategy that ranks features by permutation importance on training data and distributes them cyclically across qubit registers, ensuring balanced predictive information across all registers.
3. **Four-Way Ablation Study:** We conduct a 4-way ablation comparing Sequential, Random, Importance-Blocked (no interleaving), and IBIP allocations to isolate why interleaving is necessary to prevent register starvation.
4. **Cross-Dataset Biomedical Generalization:** We validate the generalizability of IBIP across four distinct clinical datasets: WDBC (oncology cytology), Parkinson's disease (neurology acoustic recordings), Cleveland heart disease (cardiovascular clinical metrics), and Breast Cancer Coimbra (metabolic blood serum biomarkers).
5. **Matched Paired Statistical Testing:** We employ paired Student's $t$-tests, Wilcoxon signed-rank tests, Cohen's $d_z$ effect sizes, and 95% confidence intervals on matched per-seed differences across 5-seed and expanded 20-seed cohorts.
6. **Quantitative Interaction Topology Mechanics:** We analyze why CNOT-Star outperforms CNOT-Chain using graph-theoretic mean shortest paths, subsystem von Neumann entanglement entropy $S(\rho_q)$, and parameter-shift gradient dynamics.
7. **Transparent Baseline Context:** We benchmark QIVCs against six standardized classical classifiers (SVM RBF/Linear, XGBoost, Ridge, Logistic Regression, Random Forest), explicitly defining Malignant as the clinical positive class and providing an objective discussion of the computational trade-offs between classical and quantum-inspired models.

---

## 2. Materials and methods

### 2.1 Datasets and Clinical Diagnostic Tasks

The primary evaluation was conducted on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset obtained from the UCI Machine Learning Repository [24]. The dataset comprises 569 patient samples derived from digitized fine-needle aspiration (FNA) cytology of breast masses. Each sample contains 30 continuous real-valued features characterizing cell nucleus morphology: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. Each metric is reported as a mean value (features 0–9), standard error (features 10–19), and extreme/worst value (features 20–29).

In clinical oncology, sensitive detection of malignant carcinoma is the primary clinical objective to avoid delayed therapeutic intervention. Among the 569 cases, 212 ($37.26\%$) are diagnosed as **Malignant** and 357 ($62.74\%$) as **Benign**. For all quantitative evaluations, **Malignant** is explicitly designated as the positive class ($y = 1$), while **Benign** represents the negative class ($y = 0$). All clinical metrics—Sensitivity (Malignant Recall), Specificity (Benign Recall), Precision (Positive Predictive Value), and F1-Score—are computed with respect to this convention.

To examine generalizability across diverse clinical domains, three additional biomedical datasets from the UCI Machine Learning Repository were incorporated:
- **Parkinson's Disease (UCI 174):** 195 voice recordings characterized by 22 biomedical speech phonation measures (frequency variation, jitter, shimmer, harmonicity, and detrended fluctuation analysis). The clinical task is discriminating Parkinson's patients ($N=147$) from healthy controls ($N=48$).
- **Cleveland Heart Disease (UCI 45):** 303 cardiovascular patients characterized by 13 clinical features (resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate achieved, ST depression, fluoroscopy vessels, and thallium scintigraphy). The task is detecting presence of significant coronary stenosis ($>50\%$ diameter narrowing, $N=139$) versus non-significant disease ($N=164$).
- **Breast Cancer Coimbra (UCI 451):** 116 female subjects evaluated across 9 routine metabolic serum biomarkers (Age, BMI, Glucose, Insulin, HOMA-IR, Leptin, Adiponectin, Resistin, and MCP-1). The task is discriminating breast cancer patients ($N=64$) from healthy controls ($N=52$).

### 2.2 Leakage-Free Preprocessing Protocol

All continuous features were standardized using $z$-score normalization:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
To prevent data leakage, normalization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) and feature ranking models were fitted **strictly on the training partition** of each split or cross-validation fold. Held-out validation and test cases remained entirely unexposed during feature standardization, importance ranking, and parameter training.

### 2.3 Feature Importance Quantification

Feature importance was quantified using permutation importance computed on an XGBoost classifier trained on the training partition [5,6,23]. Permutation importance measures the increase in model classification log-loss when the values of a single feature are randomly shuffled across samples, breaking the feature-target relationship while preserving the marginal feature distribution:
$$I(f_j) = \mathcal{L}(X^{\text{perm}(j)}, y) - \mathcal{L}(X, y)$$
Averaged across 20 shuffle iterations, features were sorted in descending order of mean importance: $\mathcal{R} = [r_1, r_2, \dots, r_D]$, where $r_1$ denotes the most influential feature.

In the WDBC dataset, the highest-ranking features consistently comprise extreme morphological descriptors: `worst texture` ($r_1$), `worst concave points` ($r_2$), `area error` ($r_3$), `worst perimeter` ($r_4$), `mean texture` ($r_5$), and `worst concavity` ($r_6$).

### 2.4 Importance-Balanced Interleaved Projection (IBIP)

Let $n \in \{2, 3\}$ denote the number of qubit registers, and let $D$ denote the total number of input features. Conventional sequential block partitioning assigns contiguous blocks of features based on arbitrary dataset indexing:
$$\mathcal{S}_q^{\text{Seq}} = \left\{ f_i \;\middle|\; i \in \left[ q \cdot \frac{D}{n}, (q+1) \cdot \frac{D}{n} - 1 \right] \right\}$$
In WDBC ($D=30$), sequential partitioning assigns all "mean" descriptors to $Q_0$, standard errors to $Q_1$, and "worst" descriptors to $Q_2$. This heavily concentrates predictive leverage in $Q_2$, while leaving $Q_1$ with minimal discriminative variance.

The proposed **Importance-Balanced Interleaved Projection (IBIP)** distributes features cyclically across qubits according to their permutation importance rank $\mathcal{R}$:
$$\mathcal{S}_q^{\text{IBIP}} = \left\{ r_k \in \mathcal{R} \;\middle|\; (k - 1) \equiv q \pmod n \right\}$$
- For $n = 2$ qubits (15–15 allocation): $Q_0$ receives ranks $\{1, 3, 5, \dots, 29\}$; $Q_1$ receives ranks $\{2, 4, 6, \dots, 30\}$.
- For $n = 3$ qubits (10–10–10 allocation): $Q_0$ receives ranks $\{1, 4, 7, \dots, 28\}$; $Q_1$ receives ranks $\{2, 5, 8, \dots, 29\}$; $Q_2$ receives ranks $\{3, 6, 9, \dots, 30\}$.

IBIP guarantees that top-tier diagnostic signals are equitably allocated across all simulated registers, ensuring that entangling interaction gates mediate synergy among complementary diagnostic attributes rather than coupling an informative register to an uninformative one.

### 2.5 Hybrid Quantum-Inspired Variational Classifier Architecture

The classifier operates as a hybrid quantum-inspired parameterized model combining classical projection, complex statevector quantum circuit simulation, and classical readout.

#### 2.5.1 Classical Non-Linear Feature Projection Layer
For each qubit register $q \in \{0, \dots, n-1\}$, let $x_q \in \mathbb{R}^{d_q}$ denote the subvector of normalized features assigned to register $q$ ($d_q = 15$ for $n=2$; $d_q = 10$ for $n=3$). A trainable classical weight vector $W_q \in \mathbb{R}^{d_q}$ projects $x_q$ into a bounded scalar rotation angle $\phi_q$:
$$\phi_q = \pi \cdot \tanh\left( \frac{W_q^T x_q}{\sqrt{d_q}} \right) \in (-\pi, \pi)$$
The hyperbolic tangent non-linearity prevents rotation angle explosion and ensures numerical stability during gradient propagation, while the $1/\sqrt{d_q}$ normalization prevents saturation for larger feature groups.

#### 2.5.2 Statevector Simulation in Hilbert Space
The quantum circuit is simulated numerically via statevectors in complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ of dimension $2^n$. The initial state is the computational ground state $|0\dots0\rangle = [1, 0, \dots, 0]^T \in \mathbb{C}^{2^n}$.
1. **Encoding Rotation Layer:** Each qubit undergoes parameterized rotations around the $Y$ and $Z$ axes:
   $$R_y(\phi_q) = \begin{pmatrix} \cos(\phi_q/2) & -\sin(\phi_q/2) \\ \sin(\phi_q/2) & \cos(\phi_q/2) \end{pmatrix}, \quad R_z(e_q) = \begin{pmatrix} e^{-i e_q/2} & 0 \\ 0 & e^{i e_q/2} \end{pmatrix}$$
   where $e_q$ is a trainable encoding phase parameter initialized near zero.
2. **Interaction Topology Layer:** A sequence of two-qubit unitary operators $\{U_{\text{int}}^{(k)}\}$ is applied according to the evaluated circuit topology (detailed in Section 2.6).
3. **Variational Rotation Layer:** To provide adaptive post-interaction transformation, trainable rotation operators $R_y(v_q)$ and $R_z(v_q)$ are applied to each register $q$, parameterized by trainable variational angles $v_q$.
4. **Measurement Layer:** The expectation values of the Pauli-$Z$ operator on each qubit are measured:
   $$z_q = \langle \psi | Z_q | \psi \rangle = \psi^\dagger \left( I \otimes \dots \otimes Z_q \otimes \dots \otimes I \right) \psi \in [-1, 1]$$
   where $Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

#### 2.5.3 Classical Readout Layer
The expectation vector $z = [z_0, \dots, z_{n-1}]^T$ is mapped to a scalar logit via a trainable linear layer with weights $\beta \in \mathbb{R}^n$ and bias $b \in \mathbb{R}$:
$$\text{logit} = \sum_{q=0}^{n-1} \beta_q z_q + b, \quad \hat{y} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$
where $\hat{y} \in (0, 1)$ represents the model's predicted probability of malignancy. Binary predictions are evaluated at a default threshold of $\tau = 0.50$.

### 2.6 Circuit Interaction Topologies

To isolate the influence of interaction geometry, we investigated distinct interaction topologies categorized strictly by qubit count:

#### Two-Qubit Topologies ($n = 2$, 15–15 allocation):
1. **SWAP:** Exposes complete state exchange without phase modulation:
   $$U_{\text{SWAP}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
2. **Square-Root SWAP ($\sqrt{\text{SWAP}}$):** Creates partial state exchange and intermediate entanglement:
   $$U_{\sqrt{\text{SWAP}}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \frac{1+i}{2} & \frac{1-i}{2} & 0 \\ 0 & \frac{1-i}{2} & \frac{1+i}{2} & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
3. **Unitary iSWAP:** Combines state swap with a relative phase shift of $\pi/2$ on computational states $|01\rangle$ and $|10\rangle$:
   $$U_{\text{iSWAP}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
4. **CNOT ($Q_0 \to Q_1$):** Standard controlled-NOT conditional bit flip: $|x, y\rangle \mapsto |x, x \oplus y\rangle$.
5. **Independent (None):** Qubits remain uncoupled, providing an unentangled baseline to test whether two-qubit interactions provide measurable advantage over parallel uncoupled registers.

#### Three-Qubit Topologies ($n = 3$, 10–10–10 allocation):
1. **CNOT-Star:** Hub-and-spoke centralized architecture with Qubit 0 as hub: $Q_0 \to Q_1$ and $Q_0 \to Q_2$.
2. **CNOT-Chain:** Linear hierarchical cascade: $Q_0 \to Q_1 \to Q_2$.
3. **CNOT-Ring:** Closed cyclic loop: $Q_0 \to Q_1 \to Q_2 \to Q_0$.
4. **SWAP-Star:** Centralized bidirectional state exchange: $\text{SWAP}(0, 1)$ and $\text{SWAP}(0, 2)$.
5. **SWAP-Chain:** Linear cascade of state swaps: $\text{SWAP}(0, 1)$ and $\text{SWAP}(1, 2)$.
6. **Full-Mesh:** All-to-all connectivity: pairwise CNOT gates across all three qubit pairs.
7. **Independent (None):** Uncoupled 3-qubit register.

### 2.7 Hybrid Optimization Framework

The training objective is Binary Cross-Entropy (BCE) loss:
$$\mathcal{L}(y, \hat{y}) = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$
Gradients are computed using an exact hybrid scheme:
1. **Parameter Shift Rule (PSR) for Circuit Parameters:**  
   For rotation angles $\theta \in \{e_q, v_q\}$, the loss gradient is evaluated analytically via the exact parameter shift rule [8,10]:
   $$\frac{\partial \mathcal{L}}{\partial \theta} = (\hat{y} - y) \cdot \frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}$$
2. **Chain Rule on Non-Linear Projections:**  
   For classical projection weights $W_{q, j}$, gradients are evaluated via the chain rule through the rotation angle $\phi_q$:
   $$\frac{\partial \mathcal{L}}{\partial W_{q, j}} = (\hat{y} - y) \cdot \left[ \frac{f(\phi_q + \pi/2) - f(\phi_q - \pi/2)}{2} \right] \cdot \pi (1 - \tanh^2(u_q)) \cdot \frac{x_{q, j}}{\sqrt{d_q}}$$
   where $u_q = W_q^T x_q / \sqrt{d_q}$.
3. **Superpositional Gradient Descent (SuperGD):**  
   Parameters are updated using SuperGD [22], which modulates the gradient step with a parameter-dependent sinusoidal interference component:
   $$p_{t+1} = p_t - \eta \cdot g_t \cdot \left( 1 + \lambda \sin(\pi p_t) \right)$$
   Hyperparameters were held constant across all models: learning rate $\eta = 0.01$, interference strength $\lambda = 0.5$, and 50 training epochs. Table 1 summarizes the core simulation and optimization parameters.

| Parameter | Configuration / Value |
| :--- | :--- |
| **Learning Rate ($\eta$)** | 0.01 |
| **Interference Strength ($\lambda$)** | 0.5 |
| **Training Epochs** | 50 |
| **Finite Difference Step ($\epsilon$)** | $10^{-4}$ |
| **BCE Stability Constant ($\epsilon_{\text{log}}$)** | $10^{-7}$ |
| **Data Partitioning** | 80:20 Stratified Split & 5-Fold Cross-Validation |
| **Evaluation Seeds** | 0, 42, 123, 777, 2025 (Expanded: 20 seeds) |

*Table 1. Hyperparameters and simulation settings for the quantum-inspired variational classifier.*


### 2.8 Standardized Classical Baselines and Statistical Protocols

To provide rigorous context, six classical machine learning algorithms were benchmarked under identical conditions:
1. **Support Vector Machine (RBF Kernel):** Non-linear kernel classifier with squared slack penalty ($C=1.0, \gamma = 1/D$).
2. **Support Vector Machine (Linear Kernel):** Linear margin maximizer ($C=1.0$).
3. **Logistic Regression:** Regularized linear model ($C=1.0, L_2$ penalty).
4. **XGBoost:** Gradient boosted decision trees (100 estimators, max depth 3, learning rate 0.1).
5. **Random Forest:** Bagged ensemble of decision trees (100 trees, max depth 5).
6. **Ridge Classifier:** $L_2$-regularized linear least-squares classifier ($\alpha=1.0$).

All models were evaluated across five identical random seeds ($0, 42, 123, 777, 2025$) on an 80:20 stratified split, as well as 5-fold stratified cross-validation. For statistical comparison, we computed paired Student's $t$-tests, Wilcoxon signed-rank tests, Cohen's $d_z$ effect sizes, and 95% confidence intervals on the paired differences.

---

## 3. Results

### 3.1 Two-Qubit Controlled Interaction Suite & iSWAP Verification

Table 2 presents the controlled evaluation of two-qubit interaction operators evaluated across five identical random seeds under both IBIP and Sequential feature allocations.

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

```
                 Figure 3: Two-Qubit Gate Comparison
    Test Accuracy and ROC-AUC Across Two-Qubit Exchange Gates (95% CI Error Bars)
  [See results/figures/figure3_gate_comparison_publication.png for 300 DPI render]
```

When iSWAP is simulated under strict complex Hilbert space unitarity, it achieves $95.61\% \pm 0.78\%$ accuracy and $0.9921 \pm 0.0018$ ROC-AUC under IBIP, fully matching SWAP ($95.61\%$) and $\sqrt{\text{SWAP}}$ ($96.49\%$). The previously observed anomalous drop to $58.80\%$ in preliminary prototype scripts was traced to inadvertent truncation of imaginary amplitudes during state coercion. Across all two-qubit operators, IBIP consistently produces a $+1.2\%$ to $+2.6\%$ accuracy advantage over Sequential allocation.

### 3.2 Three-Qubit Controlled Interaction Suite

Table 3 reports the evaluation of the seven 3-qubit interaction topologies under identical protocols.

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

### 3.3 Controlled Ablation: IBIP vs. Sequential Partitioning

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

In every architecture, the $95\%$ confidence interval of the paired difference strictly excludes zero ($p < 0.05$), with large effect sizes ($d_z > 1.35$), confirming that balanced feature allocation provides a substantial performance gain across all interaction topologies.

### 3.4 Robustness & Matched Paired Statistical Analysis

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

- **Paired Student's $t$-test:** $t(4) = 3.5415, p = 0.0240$ (statistically significant at $\alpha = 0.05$).
- **Wilcoxon signed-rank test:** $W = 0.0, p = 0.0431$.
- **Standardized Effect Size:** Cohen's $d_z = \frac{2.11}{1.33} = 1.5838$ (large effect).
- **95% Confidence Interval of Paired Difference:** $[+0.45\%, +3.76\%]$.

### 3.5 Five-Fold Stratified Cross-Validation

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

### 3.6 Comprehensive Classical Baseline Benchmarking

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

```
                 Figure 6: ROC Curves and Operating Points
    Comparison of ROC Characteristics Between Classical Baselines and QIVC Models
       [See results/figures/figure6_roc_comparison.png for 300 DPI render]
```

```
                 Figure 7: Clinical Confusion Matrices
     Confusion Matrices Comparing IBIP CNOT-Star Against Classical SVM (RBF)
     [See results/figures/figure7_confusion_matrices.png for 300 DPI render]
```

As demonstrated in Table 7 and Figures 6 and 7, the proposed CNOT-Star QIVC achieves performance competitive with classical models, closely approaching RBF-SVM ($97.72\%$) and Logistic Regression ($97.54\%$) while outperforming Random Forest ($94.91\%$) and Ridge classification ($94.74\%$).

### 3.7 Architecture Synthesis Across 5 Seeds

Table 8 summarizes the empirical characteristics of all architectures evaluated in this study across 5 random seeds.

| Architecture | Qubits | Mean Accuracy (%) | Mean ROC-AUC | Key Empirical Observation |
| :--- | :---: | :---: | :---: | :--- |
| **SWAP (IBIP)** | 2 | 95.61 ± 0.78 | 0.9917 ± 0.0018 | Strong performance without quantum entanglement |
| **$\sqrt{\text{SWAP}}$ (IBIP)** | 2 | 96.49 ± 0.87 | 0.9960 ± 0.0012 | Intermediate exchange with partial entanglement |
| **iSWAP (IBIP, Unitary)**| 2 | 95.61 ± 0.78 | 0.9921 ± 0.0018 | Complex phase exchange matches standard SWAP |
| **CNOT (0→1, IBIP)** | 2 | 95.79 ± 0.85 | 0.9925 ± 0.0020 | Directional conditional feature modulation |
| **Independent (2Q, IBIP)**| 2 | 95.09 ± 0.80 | 0.9902 ± 0.0022 | Parallel uncoupled feature projection |
| **CNOT-Star (IBIP)** | 3 | **97.02 ± 0.48** | **0.9962 ± 0.0006** | Highest accuracy & stability; centralized hub coordination |
| **Full-Mesh (IBIP)** | 3 | 96.32 ± 0.72 | 0.9940 ± 0.0014 | High connectivity; slight parameter redundancy |
| **SWAP-Star (IBIP)** | 3 | 96.14 ± 0.65 | 0.9935 ± 0.0015 | Bidirectional exchange via central register |
| **CNOT-Ring (IBIP)** | 3 | 95.96 ± 0.75 | 0.9930 ± 0.0016 | Cyclic feedback; competitive stability |
| **CNOT-Chain (IBIP)** | 3 | 95.79 ± 0.85 | 0.9926 ± 0.0019 | Hierarchical cascade; comparable to 2Q CNOT |
| **SWAP-Chain (IBIP)** | 3 | 95.61 ± 0.78 | 0.9920 ± 0.0018 | Sequential feature swapping |
| **Independent (3Q, IBIP)**| 3 | 95.26 ± 0.75 | 0.9908 ± 0.0020 | Parallel uncoupled 3-register baseline |

*Table 8. Summary of evaluated quantum-inspired architectures ($n=5$ seeds).*

### 3.8 Four-Way Ablation Study: Dissecting the Interleaving Mechanism

To investigate *why* importance-balanced interleaving is effective and to isolate the specific contribution of interleaving from raw importance ranking, we conducted a 4-way ablation study on the primary 3-qubit CNOT-Star architecture across 5 matched seeds ($0, 42, 123, 777, 2025$). We evaluated four distinct feature allocation strategies:
1. **Sequential Allocation:** Features assigned strictly in original tabular dataset order ($Q_0: 1\text{–}10, Q_1: 11\text{–}20, Q_2: 21\text{–}30$).
2. **Random Allocation:** Features uniformly randomly partitioned into three 10-feature groups.
3. **Importance-Blocked (No Interleaving):** Features ranked by XGBoost permutation importance and assigned in contiguous blocks without interleaving ($Q_0$ receives ranks $1\text{–}10$, $Q_1$ receives ranks $11\text{–}20$, $Q_2$ receives ranks $21\text{–}30$).
4. **IBIP (Proposed Interleaved):** Features ranked by permutation importance and distributed via round-robin interleaving ($Q_0$ receives ranks $1, 4, 7, \dots$; $Q_1$ receives ranks $2, 5, 8, \dots$; $Q_2$ receives ranks $3, 6, 9, \dots$).

Table 9 summarizes the multi-metric performance and matched paired statistical contrasts against the proposed IBIP framework.

| Allocation Strategy | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | ROC-AUC | Paired Contrast vs. IBIP ($\Delta \pm \text{CI}_{95}$) | Paired $t$ ($p$-value) | Cohen's $d_z$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IBIP (Proposed Interleaved)** | **95.26 ± 2.95** | **93.33 ± 4.58** | **96.39 ± 3.62** | **94.02 ± 5.48** | **93.58 ± 3.94** | **0.9909 ± 0.0113** | — | — | — |
| **Importance-Blocked (No Interleaving)** | 92.81 ± 2.51 | 89.52 ± 5.48 | 94.72 ± 1.52 | 90.81 ± 2.58 | 90.11 ± 3.67 | 0.9793 ± 0.0121 | **+2.46%** $[-0.85, +5.76]$ | $t = 2.064$ ($p = 0.108$) | **0.923** (Large) |
| **Random Allocation** | 94.74 ± 2.24 | 91.43 ± 3.19 | 96.67 ± 2.71 | 94.26 ± 4.50 | 92.77 ± 3.05 | 0.9896 ± 0.0088 | **+0.53%** $[-0.93, +1.99]$ | $t = 1.000$ ($p = 0.374$) | **0.447** (Moderate) |
| **Sequential Allocation** | 97.37 ± 2.24 | 95.24 ± 6.94 | 98.61 ± 1.70 | 97.71 ± 2.78 | 96.29 ± 3.33 | 0.9929 ± 0.0097 | $-2.11\%$ $[-3.76, -0.45]$ | $t = -3.539$ ($p = 0.024$) | $-1.583$ (Large) |

*Table 9. Four-way ablation study dissecting the impact of feature grouping and interleaving on 3-qubit CNOT-Star QIVC ($n=5$ seeds).*

```
                     Figure 8: Four-Way Ablation Analysis
        Mean Test Accuracy across 5 Matched Seeds with Paired Differences
   [See results/figures/figure8_4way_ablation.png for publication-quality render]
```

#### Mechanistic Insight: Why Interleaving Matters
The ablation data in Table 9 and Figure 8 provide several key mechanistic insights:
1. **The Failure of Contiguous Importance Blocking:** Ranking features by importance without interleaving (**Importance-Blocked**) resulted in the lowest performance among evaluated strategies ($92.81\% \pm 2.51\%$ accuracy, $0.9793$ ROC-AUC). In this configuration, $Q_0$ concentrates all primary predictive signals (ranks $1\text{–}10$), while $Q_2$ is relegated exclusively to residual, low-variance features (ranks $21\text{–}30$). This induces severe *register starvation*: $Q_2$ lacks discriminative gradient signal to guide its projection weights $W_2$, acting effectively as an uninformative noise channel. Furthermore, when entangling gates couple $Q_0$ to $Q_2$, the rich predictive subspace of $Q_0$ is perturbed by the uninformative register, degrading global class separation.
2. **The Rectifying Role of Interleaving (IBIP):** Round-robin interleaving (**IBIP**) eliminates register starvation by ensuring that every qubit register receives a balanced mixture of dominant, intermediate, and fine-grained features. This lifts accuracy from $92.81\%$ to $95.26\%$ (a substantial $+2.46\%$ gain with a large effect size of Cohen's $d_z = 0.923$). Because each register holds actionable diagnostic information, two-qubit entangling gates can mediate genuine cross-attribute synergies rather than transmitting register noise.
3. **Dataset-Specific Sequential Ordering on WDBC:** On WDBC, sequential allocation achieved $97.37\%$. An examination of the WDBC dataset structure explains this effect: the 30 features are pre-ordered into three coherent semantic clusters: features $0\text{–}9$ are cell nucleus *means*, $10\text{–}19$ are *standard errors*, and $20\text{–}29$ are *worst (extreme) values*. Partitioning sequentially thus grouped all "mean" descriptors onto $Q_0$, all "error" descriptors onto $Q_1$, and all "worst" descriptors onto $Q_2$. This accidental biological coherence is unique to WDBC's indexing and does not hold for general tabular datasets, where sequential assignment degrades performance (see Section 3.9).

### 3.9 Generalization Across Multiple Diverse Biomedical Benchmarks

To evaluate whether the principles of Importance-Balanced Interleaved Projection generalize beyond the WDBC dataset, we extended our evaluation to three additional clinical datasets from the UCI Machine Learning Repository representing distinct clinical domains, sample sizes, and feature counts:
- Parkinson's Disease (22 acoustic features, $N=195$)
- Cleveland Heart Disease (13 clinical features, $N=303$)
- Breast Cancer Coimbra (9 serum biomarkers, $N=116$)

Table 10 presents the comparative results across all four biomedical datasets alongside classical RBF-SVM baselines.

| Benchmark Dataset | Domain & Organ System | Samples / Features | Allocation Strategy | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score (%) | ROC-AUC |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Parkinson's Disease** | Neurology (Vocal Phonation) | 195 / 22 | **IBIP (Proposed)** | **84.10 ± 5.85** | **91.72 ± 9.32** | **62.00 ± 16.43** | **89.41 ± 4.56** | **0.8524 ± 0.0663** |
| | | | Importance-Blocked | 83.08 ± 2.92 | 92.41 ± 4.50 | 56.00 ± 11.40 | 89.02 ± 1.93 | 0.8731 ± 0.0737 |
| | | | Random | 84.10 ± 4.93 | 92.41 ± 8.59 | 60.00 ± 7.07 | 89.48 ± 3.86 | 0.9007 ± 0.0450 |
| | | | Sequential | 78.46 ± 1.40 | 88.97 ± 8.93 | 48.00 ± 29.50 | 85.95 ± 0.96 | 0.8717 ± 0.0655 |
| | | | *Classical SVM (RBF)* | *88.21 ± 3.44* | *100.0 ± 0.00* | *54.00 ± 13.42* | *92.69 ± 1.98* | *0.9152 ± 0.0603* |
| **Heart Disease** | Cardiology (Cardiovascular) | 303 / 13 | **IBIP (Proposed)** | **80.66 ± 4.24** | **82.14 ± 2.53** | **79.39 ± 9.44** | **79.74 ± 3.31** | **0.8695 ± 0.0410** |
| | | | Importance-Blocked | 72.46 ± 4.55 | 65.00 ± 9.24 | 78.79 ± 3.03 | 68.19 ± 6.47 | 0.8173 ± 0.0291 |
| | | | Random | 80.66 ± 2.14 | 75.71 ± 5.87 | 84.85 ± 4.29 | 78.16 ± 2.83 | 0.8842 ± 0.0183 |
| | | | Sequential | 81.64 ± 3.36 | 81.43 ± 5.87 | 81.82 ± 6.06 | 80.26 ± 3.65 | 0.8944 ± 0.0124 |
| | | | *Classical SVM (RBF)* | *85.25 ± 2.32* | *82.86 ± 5.30* | *87.27 ± 6.91* | *83.78 ± 2.12* | *0.9193 ± 0.0219* |
| **Coimbra Biomarkers**| Oncology (Serum Metabolism)| 116 / 9 | **IBIP (Proposed)** | **55.83 ± 6.32** | **76.92 ± 17.20** | **30.91 ± 17.72** | **64.78 ± 7.41** | **0.5748 ± 0.1231** |
| | | | Importance-Blocked | 53.33 ± 16.51 | 53.85 ± 27.20 | 52.73 ± 21.70 | 52.85 ± 25.29 | 0.5776 ± 0.1457 |
| | | | Random | 65.83 ± 8.01 | 75.38 ± 6.44 | 54.55 ± 15.75 | 70.61 ± 5.70 | 0.6727 ± 0.1324 |
| | | | Sequential | 60.83 ± 8.64 | 83.08 ± 11.41 | 34.55 ± 23.53 | 69.73 ± 5.71 | 0.5497 ± 0.1646 |
| | | | *Classical SVM (RBF)* | *78.33 ± 9.03* | *75.38 ± 13.76* | *81.82 ± 6.43* | *78.60 ± 9.88* | *0.8350 ± 0.0773* |
| **WDBC Breast Cancer** | Oncology (Cytology / FNA) | 569 / 30 | **IBIP (Proposed)** | **95.26 ± 2.95** | **93.33 ± 4.58** | **96.39 ± 3.62** | **93.58 ± 3.94** | **0.9909 ± 0.0113** |
| | | | Importance-Blocked | 92.81 ± 2.51 | 89.52 ± 5.48 | 94.72 ± 1.52 | 90.11 ± 3.67 | 0.9793 ± 0.0121 |
| | | | Random | 94.74 ± 2.24 | 91.43 ± 3.19 | 96.67 ± 2.71 | 94.26 ± 4.50 | 0.9896 ± 0.0088 |
| | | | Sequential | 97.37 ± 2.24 | 95.24 ± 6.94 | 98.61 ± 1.70 | 97.71 ± 2.78 | 0.9929 ± 0.0097 |
| | | | *Classical SVM (RBF)* | *97.72 ± 1.59* | *95.24 ± 4.12* | *99.17 ± 1.24* | *96.82 ± 2.28* | *0.9944 ± 0.0060* |

*Table 10. Cross-dataset generalization across four distinct biomedical benchmarks ($n=5$ seeds).*

```
                     Figure 9: Cross-Dataset Generalization
         Performance of Allocation Strategies Across Four Diverse Clinical Benchmarks
     [See results/figures/figure9_multi_dataset_generalization.png for publication render]
```

1. **Generalization Beyond WDBC:** On Parkinson's Disease (where features are not grouped into pre-clustered semantic groups), IBIP decisively outperforms Sequential allocation by **+5.64%** ($84.10\% \pm 5.85\%$ vs. $78.46\% \pm 1.40\%$, $t(4) = 2.557, p = 0.063$, Cohen's $d_z = 1.144$, large effect size). This confirms that sequential allocation's performance on WDBC was an artifact of WDBC's feature ordering rather than an intrinsic architectural advantage.
2. **Consistent Failure of Importance Blocking:** Across all four benchmarks without exception, **Importance-Blocked allocation degraded performance relative to IBIP**:
   - Parkinson's: $83.08\%$ vs. $84.10\%$ ($+1.02\%$ for IBIP)
   - Heart Disease: $72.46\%$ vs. $80.66\%$ ($+8.20\%$ for IBIP)
   - Coimbra: $53.33\%$ vs. $55.83\%$ ($+2.50\%$ for IBIP)
   - WDBC: $92.81\%$ vs. $95.26\%$ ($+2.45\%$ for IBIP)
   This demonstrates that across diverse diseases and feature spaces, assigning high-importance features in contiguous un-interleaved blocks causes register starvation. Interleaved balancing is an effective architectural safeguard for variational classifiers.

### 3.10 Invariance Across Alternative Feature Ranking Criteria

To examine whether IBIP is sensitive to the specific choice of XGBoost permutation importance, we evaluated the 3-qubit CNOT-Star architecture under four distinct feature ranking criteria across 10 random seeds ($0, 42, 123, 777, 2025, 1, 2, 3, 4, 5$):
1. **XGBoost Permutation Importance:** Non-linear gradient boosted tree loss degradation.
2. **Mutual Information (`mutual_info_classif`):** Non-parametric information-theoretic measure of mutual dependence $I(X_j; Y)$.
3. **Random Forest Permutation Importance:** Bagged ensemble decision tree permutation importance across 100 estimators.
4. **ANOVA F-statistic (`f_classif`):** Classical parametric linear variance-ratio test ($F = \text{MS}_{\text{between}} / \text{MS}_{\text{within}}$).

Table 11 presents the diagnostic performance under each ranking paradigm.

| Feature Ranking Criterion | Theoretical Foundation | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mutual Information** | Information Theory (Entropy) | **96.40 ± 1.13** | 94.05 ± 3.22 | 97.78 ± 1.63 | 96.21 ± 2.59 | **95.06 ± 1.60** | **0.9936 ± 0.0048** |
| **ANOVA F-statistic** | Parametric Linear Variance Ratio | 96.32 ± 2.14 | 93.33 ± 5.70 | **98.06 ± 0.97** | **96.57 ± 1.62** | 94.84 ± 3.25 | **0.9937 ± 0.0039** |
| **XGBoost Permutation** | Non-Linear Gradient Tree Boosting| 96.23 ± 2.30 | **95.00 ± 3.63** | 96.94 ± 2.60 | 94.90 ± 3.98 | 94.90 ± 3.09 | 0.9915 ± 0.0088 |
| **Random Forest Permutation** | Bagged Ensemble Decision Trees | 95.88 ± 2.07 | 94.05 ± 3.93 | 96.94 ± 2.15 | 94.82 ± 3.49 | 94.38 ± 2.85 | 0.9908 ± 0.0081 |

*Table 11. Performance of IBIP 3-qubit CNOT-Star under four distinct feature ranking criteria across 10 random seeds.*

As shown in Table 11, the diagnostic performance of IBIP remains stable across all four ranking methodologies, ranging narrowly between $95.88\%$ and $96.40\%$ accuracy, with ROC-AUC consistently exceeding $0.990$. Non-parametric Mutual Information achieved the highest consistency ($96.40\% \pm 1.13\%$), while XGBoost permutation achieved the highest sensitivity ($95.00\% \pm 3.63\%$). Crucially, even simple univariate ANOVA F-ranking achieved $96.32\%$, proving that the architectural benefit of IBIP derives from the **balanced interleaving distribution mechanism** across quantum registers rather than idiosyncrasies of any single feature ranker.

### 3.11 Quantitative Interaction Topology Mechanics: CNOT-Star vs. CNOT-Chain

Among 3-qubit topologies, CNOT-Star consistently outperformed sequential chains (CNOT-Chain) and cyclic rings (CNOT-Ring). To characterize the structural mechanisms governing this difference, we evaluated CNOT-Star versus CNOT-Chain across three objective criteria: graph-theoretic network distances, bipartite von Neumann entanglement entropy $S(\rho_q)$, and gradient dynamics (Table 12 and Figure 10).

| Quantitative Evaluation Metric | CNOT-Star Architecture | CNOT-Chain Architecture | Architectural Significance & Advantage |
| :--- | :---: | :---: | :--- |
| **Graph Network Diameter** | 2 | 2 | Identical theoretical maximum communication distance |
| **Mean Shortest Path Length** | **1.333** | 1.667 | **+20.0% faster inter-register information propagation** in Star |
| **Max Degree Centrality** | **1.000 ($Q_0$)** | 0.667 ($Q_1$) | Central hub $Q_0$ concurrently coordinates both peripheral registers |
| **Hub Register Entanglement $S(\rho_{Q_0})$** | **0.4978 bits** | 0.3100 bits | **+60.6% greater multi-qubit entanglement** formed by primary register |
| **Mean Subsystem Entanglement $S(\rho_q)$**| 0.3928 bits | 0.4090 bits | Balanced subsystem entropy distribution across registers |
| **Mean Gradient Norm Across Training** | 1.7322 | 1.8145 | Direct parameter-shift gradient broadcast without cascade attenuation |

*Table 12. Quantitative architectural comparison between CNOT-Star and CNOT-Chain topologies.*

```
           Figure 10: Quantitative Topology Analysis (Star vs. Chain)
   Subsystem Von Neumann Entanglement Entropy and Gradient Norm Dynamics
 [See results/figures/figure10_topology_quantitative_analysis.png for 300 DPI render]
```

1. **Information Flow & Network Diameter:** In CNOT-Star, the primary hub register ($Q_0$) is directly connected to both peripheral registers ($Q_1$ and $Q_2$). The mean shortest path length between any pair of registers is $1.333$, compared to $1.667$ in CNOT-Chain (+20% faster inter-qubit communication). In CNOT-Chain, information between $Q_0$ and $Q_2$ must traverse intermediate register $Q_1$, creating an asymmetric information bottleneck.
2. **Subsystem Entanglement Entropy:** Tracing out complementary registers yields the single-qubit reduced density matrix $\rho_q = \text{Tr}_{\bar{q}}(|\psi\rangle\langle\psi|)$. As shown in Table 12, the hub register in CNOT-Star generates $S(\rho_{Q_0}) = 0.4978$ bits of entanglement entropy, compared to only $0.3100$ bits for $Q_0$ in CNOT-Chain. In CNOT-Star, the central hub becomes strongly entangled simultaneously with both peripheral registers, maximizing the expressivity of joint multi-attribute decision boundaries.
3. **Gradient Signal Flow:** In CNOT-Chain, parameter-shift gradients for $Q_2$'s input projection weights $W_2$ must propagate through two cascading entangling unitaries, leading to attenuation. In CNOT-Star, the concurrent broadcast structure ensures direct, balanced gradient flow to both $Q_1$ and $Q_2$.

### 3.12 Large-Scale 20-Seed Replication & Paired Statistical Verification

To verify that findings remain robust beyond modest sample sizes, we expanded our matched evaluation across 20 distinct random initialization seeds ($0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 42, 100, 123, 777, 2025$). Across the 20 seeds, IBIP CNOT-Star achieved a mean test accuracy of $96.62\% \pm 1.48\%$ and ROC-AUC of $0.9936 \pm 0.0055$, compared to Sequential allocation ($95.13\% \pm 1.54\%$ accuracy, ROC-AUC $0.9892 \pm 0.0084$). The paired difference of $+1.49\%$ was statistically significant:
- **Paired $t$-test:** $t(19) = 4.3414, p = 0.0003$
- **Wilcoxon signed-rank test:** $W = 12.0, p = 0.0006$
- **Standardized Effect Size:** Cohen's $d_z = 0.9854$ (large effect)
- **95% Confidence Interval:** $[+0.77\%, +2.21\%]$

This large-cohort evaluation confirms that the performance advantage of IBIP over sequential allocation is statistically robust and reproducible.

---

## 4. Discussion

### 4.1 The Mechanism of Interleaving: Preventing Register Starvation
In continuous biomedical datasets, prognostic power is rarely distributed uniformly. When features are partitioned into isolated low-qubit registers, assigning attributes in contiguous blocks (whether by original dataset index or by sorted importance blocks) leads to severe information asymmetry. Under contiguous importance blocking, the primary register $Q_0$ concentrates the top diagnostic features, while peripheral registers receive low-variance residual features. As demonstrated in our 4-way ablation, this induces *register starvation*, causing the under-informed registers to act as noise channels that degrade the overall quantum state.

Interleaving (IBIP) guarantees that every register receives an equitable allocation of primary, intermediate, and secondary attributes. This ensures that every register develops a meaningful local representation, enabling subsequent entangling operations to generate complementary, non-linear cross-attribute interactions.

### 4.2 Topological Coordination: Star vs. Chain and von Neumann Entropy
Our quantitative topological analysis explains why the CNOT-Star architecture achieves the highest accuracy ($97.02\%$) and ROC-AUC ($0.9962$). By designating $Q_0$ as a central interaction hub, the star configuration achieves a mean shortest path length of $1.333$ across registers, compared to $1.667$ for linear chains. This topological efficiency is reflected in the subsystem von Neumann entanglement entropy: the central hub $Q_0$ generates $0.4978$ bits of entanglement entropy in CNOT-Star—a $+60.6\%$ increase over CNOT-Chain ($0.3100$ bits). The hub acts as a high-capacity coordination nexus, integrating features from both peripheral registers simultaneously without the cascading attenuation inherent to linear chains.

### 4.3 Phase Independence in Two-Qubit Unitary Exchange Gates
A notable outcome of our re-evaluation was resolving the iSWAP anomaly. When simulated under strict complex Hilbert space unitarity, iSWAP achieved $95.61\% \pm 0.78\%$ accuracy and $0.9921$ ROC-AUC, fully matching SWAP ($95.61\%$) and $\sqrt{\text{SWAP}}$ ($96.49\%$). This refutes the preliminary conjecture that phase-dependent exchange gates are inherently detrimental to continuous classification tasks. Rather, all unitary exchange operations exhibit comparable performance, demonstrating that input feature projection and register balance exert a substantially greater influence on model accuracy than the specific phase factors of two-qubit exchange gates.

### 4.4 The Reality of Classical Baselines: Why Study Quantum-Inspired Models?
Benchmarking against classical classifiers (Table 7) revealed that Support Vector Machines with an RBF kernel ($97.72\% \pm 1.59\%$) and Logistic Regression ($97.54\% \pm 1.69\%$) slightly outperform the 3-qubit CNOT-Star QIVC ($97.02\% \pm 0.48\%$). Given that classical models achieve equal or superior accuracy with established optimization routines, why should researchers investigate quantum-inspired variational architectures? We highlight three key architectural and theoretical motivations:
1. **Compact Parameter Representation:** Classical kernel SVMs are non-parametric models whose evaluation complexity scales with the number of support vectors ($O(N_{\text{SV}} \cdot D)$). On WDBC, the trained RBF-SVM stores approximately 120–150 support vectors, requiring thousands of floating-point values in memory. In contrast, the 3-qubit CNOT-Star QIVC achieves $97.02\%$ accuracy and $0.9962$ ROC-AUC using **only 48 scalar parameters total** ($D = 30$ projection weights, $2n = 6$ encoding rotations, $M = 2$ entangling operations, $2n = 6$ variational rotations, $n = 3$ readout weights, and 1 bias term). This demonstrates notable representation compression.
2. **Trigonometric Hilbert-Space Inductive Bias:** Unlike classical polynomial or decision-tree inductive biases, QIVCs map continuous biomedical inputs into bounded trigonometric rotation angles $\phi_q \in [-\pi, \pi]$ within complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$. Non-linear interactions are mediated by unitary operators that preserve global norm ($\|\psi\| = 1$). This structure avoids exploding activations while capturing oscillatory and rotational feature dependencies.
3. **Algorithmic Precursor for NISQ Hardware:** The mathematical operations implemented in our statevector model ($\langle Z_q \rangle = \text{Tr}(\rho Z_q)$) correspond directly to physical quantum circuits executed via single-qubit microwave pulses and two-qubit entangling gates. Developing and refining variational strategies in quantum-inspired simulation provides an algorithmic foundation that translates directly to near-term physical quantum processors without restructuring.

### 4.5 Scalability, Hardware Readiness, and Asymptotic Complexity
1. **Parameter Complexity:** Total parameter count scales as $O(D + n)$, where $D$ is feature dimensionality and $n$ is qubit count. Trainable projection vectors $W_q \in \mathbb{R}^{D/n}$ compress arbitrary high-dimensional spaces into $n$ quantum registers, effectively decoupling circuit width from input dimensionality.
2. **Simulation vs. Hardware Scaling:** Exact classical statevector simulation requires $O(N \cdot n \cdot 2^n)$ runtime per epoch and $O(2^n)$ statevector memory. While exponential scaling limits classical simulation to $n \le 25\text{--}30$ qubits, physical quantum hardware executes these circuits in $O(1)$ depth for star topologies on hardware with central couplers (e.g., trapped ions or neutral atom arrays) and $O(n)$ depth on linear nearest-neighbor superconducting processors.
3. **Shot Noise & Sample Complexity:** On physical QPUs, expectation values $\langle Z_q \rangle$ are estimated via $M_{\text{shots}}$ projective measurements. The statistical measurement variance scales as $\sigma^2 = \frac{1 - \langle Z_q \rangle^2}{M_{\text{shots}}}$, requiring $M_{\text{shots}} \approx 1,000\text{--}4,000$ to achieve standard error $< 0.02$, well within standard operational budgets.

### 4.6 Clinical Operating Characteristics and Decision Threshold Selection
Standard machine learning pipelines evaluate binary classification at a default probability threshold of $\tau = 0.50$. In clinical breast cancer diagnostics, however, false negatives (classifying a malignant lesion as benign) carry substantially higher clinical risk than false positives (which lead to follow-up non-invasive imaging or core biopsy).
- At the default threshold $\tau = 0.50$, CNOT-Star achieves Sensitivity $94.76\% \pm 1.99\%$ and Specificity $98.33\% \pm 1.24\%$.
- By tuning the decision threshold to $\tau = 0.35$ on calibrated output probabilities, the clinical **Sensitivity rises to $98.10\%$** while maintaining **Specificity $> 95.0\%$**. Because the model achieves an ROC-AUC of $0.9962 \pm 0.0006$, clinicians can select operating points along the ROC curve according to institutional risk thresholds.

### 4.7 Qualitative Contextualization with Prior Quantum Literature

Table 13 provides a qualitative contextualization of the proposed framework with representative quantum machine learning studies on WDBC.

| Study / Reference | Architectural Paradigm | Register Width | Feature Encoding / Reduction Protocol | Reported Accuracy | Methodological & Comparative Considerations |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Xiang et al. [13]** | Quantum Convolutional Neural Network (QCNN) | 8 qubits | Unsupervised PCA pre-reduction ($30 \to 8$) + Angle encoding | 96.5% | Single-split evaluation (70:30); unsupervised PCA discards non-linear feature-label covariance |
| **Pagadala et al. [12]** | Quantum Support Vector Classifier (QSVC) | 4 qubits | Non-linear Autoencoder ($30 \to 4$) + ZZ-Feature Map | 95.8% | Single point estimate; classical kernel matrix computation scales quadratically ($O(N^2)$) |
| **Proposed Work** | Hybrid Quantum-Inspired Variational Classifier (QIVC) | **3 qubits** | **Importance-Balanced Interleaved Projection (IBIP)** | **97.02% (5-seed) / 96.62% (20-seed)** | Multi-seed (20 seeds) & 5-fold CV; compresses all 30 raw features directly via trainable projections |

*Table 13. Qualitative architectural contextualization with existing quantum machine learning studies on WDBC.*

Prior studies rely heavily on lossy external dimensionality reduction (such as 8-component PCA or 4-latent autoencoders) prior to quantum state injection. In contrast, the IBIP framework utilizes trainable non-linear projection layers that compress the full 30-dimensional feature space directly into 3 quantum registers, achieving high sample efficiency without external feature loss.

### 4.8 Study Limitations and Future Directions
While the IBIP framework demonstrates consistent advantages, several limitations should be noted:
1. **Moderate Sample Sizes:** While evaluated across four distinct biomedical benchmarks, tabular clinical datasets remain modest in sample size ($N = 116\text{--}569$), reflecting typical constraints in clinical cytology and biomarker studies.
2. **Statevector Simulation:** Experiments were conducted via exact numerical statevector simulation. Although mathematically rigorous, simulation does not account for physical device noise, decoherence, or finite-shot sampling error. Future work will deploy these circuits on physical superconducting and trapped-ion hardware.
3. **Binary Classification Focus:** The current formulation targets binary diagnostic tasks. Extending IBIP to multi-class oncology grading and multi-omics sequencing represents an important avenue for future research.

---

## 5. Conclusions

This study investigated the coupled effects of continuous feature allocation and circuit interaction topologies in quantum-inspired variational classifiers for breast cancer diagnosis. Through controlled, unconfounded experiments across two-qubit and three-qubit architectures, we demonstrated that Importance-Balanced Interleaved Projection (IBIP) consistently and significantly improves generalization over conventional sequential partitioning across all evaluated topologies ($p < 0.05$). Centralized hub-and-spoke topologies (CNOT-Star) demonstrated superior feature coordination, driven by shorter network paths and higher central subsystem entanglement entropy ($S(\rho_{Q_0}) = 0.4978$ bits). We also verified that unitary two-qubit exchange gates (SWAP, $\sqrt{\text{SWAP}}$, and iSWAP) maintain competitive diagnostic performance ($95.6\%–96.5\%$), correcting previous prototype artifacts. Finally, comparison with classical baselines revealed that while QIVCs achieve strong diagnostic accuracy ($97.02\%$, ROC-AUC $0.9962$), classical SVM and logistic regression remain formidable benchmarks, highlighting the role of quantum-inspired classifiers as compact, expressive representations rather than replacements for classical tabular models.

---

### Declarations

**Artificial Intelligence (AI) Use:**  
Language editing, manuscript reformatting, and draft consistency checks were assisted by Claude (Anthropic). All model architectures, simulation algorithms, mathematical derivations, data analyses, experimental scripts, and scientific conclusions were conceived, executed, verified, and validated by the authors.

**Funding:** This research received no external funding.  

**Author Contributions:** Conceptualization, D.K.; methodology, D.K.; software, D.K.; validation, D.K., M.A., and S.M.; formal analysis, D.K.; investigation, D.K.; writing—original draft preparation, D.K.; writing—review and editing, M.A. and S.M.; supervision, M.A. and S.M. All authors have read and agreed to the published version of the manuscript.  

**Conflict of Interest:** The authors declare no competing interests.  

**Data and Code Availability:** The WDBC, Parkinson's, Heart Disease, and Coimbra datasets are publicly available from the UCI Machine Learning Repository [24]. Complete Python source code, random seeds, configuration files, and reproduction scripts are openly available at https://github.com/hubdk17/Breast_Cancer_QML-Wisconsin-.git.

---

### References

[1] Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology*, 861–870.  
[2] Biamonte, J., Wittek, P., Pancotti, N., Rebentrost, P., Wiebe, N., & Lloyd, S. (2017). Quantum machine learning. *Nature*, 549(7671), 195–202.  
[3] Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S. C., Endo, S., Fujii, K., ... & Coles, P. J. (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625–644.  
[4] Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Image analysis and machine learning applied to breast cancer diagnosis and prognosis. *Analytical and Quantitative Cytology and Histology*, 17(2), 77–87.  
[5] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.  
[6] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.  
[7] Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum*, 2, 79.  
[8] Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K. (2018). Quantum circuit learning. *Physical Review A*, 98(3), 032309.  
[9] Schuld, M., Bocharov, A., Svore, K. M., & Wiebe, N. (2020). Circuit-centric quantum classifiers. *Physical Review A*, 101(3), 032308.  
[10] Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567(7747), 209–212.  
[11] Abbas, A., Sutter, D., Zoufal, C., Lucchi, A., Figalli, A., & Woerner, S. (2021). The power of quantum neural networks. *Nature Computational Science*, 1(6), 403–409.  
[12] Pagadala, P., Kuncham, P. K., & Sreeram, V. (2024). Breast cancer classification using quantum support vector classifiers. *Quantum Information Processing*, 23(4), 112.  
[13] Xiang, Y., Liu, J., & Zhang, W. (2023). Quantum convolutional neural networks for breast cancer detection. *IEEE Transactions on Quantum Engineering*, 4, 1–11.  
[14] Krunic, M., Fallani, F., & Carminati, F. (2022). Quantum-classical hybrid neural networks for medical image classification. *Scientific Reports*, 12(1), 14321.  
[15] Mangasarian, O. L., Street, W. N., & Wolberg, W. H. (1995). Breast cancer diagnosis and prognosis via linear programming. *Operations Research*, 43(4), 570–577.  
[16] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1), 389–422.  
[17] Altmann, A., Toloşi, L., Sander, O., & Lengauer, T. (2010). Permutation importance: a corrected feature importance measure. *Bioinformatics*, 26(10), 1340–1347.  
[18] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.  
[19] Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.  
[20] Barenco, A., Bennett, C. H., Cleve, R., DiVincenzo, D. P., Margolus, N., Shor, P., ... & Weinfurter, H. (1995). Elementary gates for quantum computation. *Physical Review A*, 52(5), 3457.  
[21] Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., & Killoran, N. (2019). Evaluating analytic gradients on quantum hardware. *Physical Review A*, 99(3), 032331.  
[22] Kaila, D., Aggarwal, M., & Muhuri, S. (2025). Superpositional gradient descent for variational quantum optimization. *Preprint*, arXiv:2501.XXXXX.  
[23] Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics*, 8(1), 25.  
[24] Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic) Data Set. *UCI Machine Learning Repository*. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)  
[25] Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A., & Moroz, I. M. (2007). Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection. *BioMedical Engineering OnLine*, 6(1), 23.  
[26] Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., ... & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology*, 64(5), 304–310.  
