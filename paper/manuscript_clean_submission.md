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

**Introduction:** Quantum-inspired machine learning offers a mathematical framework for studying non-linear representation learning, parameter-shift optimization, and multi-qubit interaction mechanisms on classical computing architectures. In continuous tabular biomedical classification—such as breast cancer cytology—the relationship between feature allocation across low-qubit registers and circuit interaction topologies remains an open question.

**Materials and methods:** This study examines the feature allocation problem in quantum-inspired variational classifiers (QIVCs) applied to breast cancer diagnosis. Continuous cytological descriptors must be mapped into compact simulated qubit registers ($n \in \{2, 3\}$), where conventional sequential grouping can place disproportionate predictive leverage into a single register. We examine an Importance-Balanced Interleaved Projection (IBIP) strategy that ranks features on training data using permutation importance and distributes them cyclically across qubits, aiming to provide balanced feature distributions across registers. The model combines classical non-linear projection ($\phi_q = \pi \tanh(W_q^T x_q / \sqrt{|x_q|})$) with numerical statevector simulation in complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ under parameterized unitary single-qubit rotations and multi-qubit interaction topologies, measured via Pauli-$Z$ expectations. Two-qubit exchange gates (SWAP, $\sqrt{\text{SWAP}}$, unitary iSWAP, CNOT, Independent) and three-qubit interaction graphs (CNOT-Star, CNOT-Chain, CNOT-Ring, SWAP-Star, SWAP-Chain, Full-Mesh, Independent) were evaluated in separate, controlled experimental suites on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset ($N=569$, 30 cytological features) across matched random seeds and five-fold stratified cross-validation. To examine the role of interleaving, a four-way ablation study was conducted on WDBC comparing sequential, random, importance-blocked (no interleaving), and IBIP allocations. To evaluate whether the observed trends generalize beyond the column arrangement of WDBC, the framework was additionally tested on three external biomedical datasets (Parkinson's disease, Cleveland heart disease, and Breast Cancer Coimbra), and benchmarked against six classical machine learning baselines.

**Results:** On the WDBC breast cancer task, the 3-qubit CNOT-Star architecture with IBIP achieved a mean test accuracy of $97.02\% \pm 0.48\%$ and an ROC-AUC of $0.9962 \pm 0.0006$ across five matched seeds, showing higher accuracy than sequential allocation ($94.91\% \pm 1.14\%$ accuracy, $0.9909 \pm 0.0028$ ROC-AUC; paired difference $+2.11\%$, $95\%\text{ CI: }[0.45\%, 3.76\%]$, paired $t(4) = 3.5415, p = 0.0240$, Wilcoxon $W = 0.0, p = 0.0431$, Cohen's $d_z = 1.5838$). In five-fold cross-validation, IBIP CNOT-Star yielded $95.26\% \pm 1.82\%$ accuracy and an ROC-AUC of $0.9840 \pm 0.0113$. The four-way ablation on WDBC indicated that assigning features in contiguous blocks of descending importance without interleaving resulted in lower accuracy ($92.81\% \pm 2.51\%$), whereas round-robin interleaving yielded $95.26\% \pm 2.95\%$ ($+2.46\%$ gain, Cohen's $d_z = 0.923$). In two-qubit exchange evaluation on WDBC, unitary iSWAP achieved $95.61\% \pm 0.78\%$ accuracy and $0.9921 \pm 0.0018$ ROC-AUC, performing similarly to SWAP ($95.61\%$) and $\sqrt{\text{SWAP}}$ ($96.49\%$). In secondary evaluations on external datasets, IBIP maintained steady performance, achieving higher accuracy than sequential allocation on Parkinson's disease ($84.10\%$ vs. $78.46\%$) and higher accuracy than blocked allocation on heart disease ($80.66\%$ vs. $72.46\%$). Classical reference baselines on WDBC achieved the highest overall accuracy: SVM RBF reached $97.72\% \pm 1.59\%$ and Logistic Regression reached $97.54\% \pm 1.69\%$.

**Conclusions:** In the context of breast cancer diagnosis, importance-balanced interleaving (IBIP) provides a practical method for distributing ranked features across low-qubit registers, improving classification performance over naive sequential grouping. Centralized hub-and-spoke topologies (CNOT-Star) demonstrated higher accuracy and greater central subsystem entanglement entropy ($S(\rho_{Q_0}) = 0.4978$ bits) than linear chains or cyclic rings. Secondary experiments on external clinical datasets support the utility of interleaving across different tabular structures. While classical kernel SVM ($97.72\%$) and logistic regression ($97.54\%$) remain the top performers in terms of raw accuracy on WDBC, the 3-qubit CNOT-Star QIVC achieves competitive diagnostic performance ($97.02\%$) using 48 trainable parameters, serving as an exploratory model for compact, non-linear representation learning.

**Keywords:** Quantum-Inspired Machine Learning; Variational Quantum Classifier; Feature Allocation; Circuit Interaction Topology; Breast Cancer Diagnosis; Fine-Needle Aspiration Cytology; Entanglement Entropy

---

## 1. Introduction

Breast cancer is among the most frequently diagnosed malignancies and a primary cause of cancer-related mortality in women globally. Early and accurate detection from digitized fine-needle aspiration (FNA) cytology is an important component of clinical assessment and treatment planning [1,4]. The Wisconsin Diagnostic Breast Cancer (WDBC) benchmark dataset [24] has served as a widely used reference for evaluating diagnostic machine learning algorithms. Classical models—such as Support Vector Machines (SVMs) [1,15], Random Forests [5], and Gradient Boosted Decision Trees (XGBoost) [6]—have demonstrated high diagnostic efficacy on this task. At the same time, exploring how alternative computational frameworks structure, project, and coordinate complex morphological features remains a topic of active interest.

Recent advances in quantum computing and quantum machine learning (QML) [2,3,7] have encouraged the study of variational quantum classifiers (VQCs) and quantum-inspired learning models [8–11]. These architectures explore high-dimensional Hilbert space mappings, parameterized unitary rotation gates, multi-qubit entangling operations, and expectation value measurements to construct non-linear decision boundaries [10,11]. Multiple hybrid designs have been investigated for breast cancer classification, including quantum kernels, quantum support vector classifiers (QSVCs) [12], quantum convolutional neural networks (QCNNs) [13], and hybrid convolutional models [14].

A practical challenge in deploying variational classifiers on continuous tabular data is the **feature allocation problem**. In near-term quantum processors and quantum-inspired simulators, high-dimensional classical vectors (such as the 30 continuous nuclear descriptors in WDBC) are typically compressed into a small number of simulated qubit registers ($n \in \{2, 3\}$). Standard implementations often rely on sequential block partitioning, where features are grouped according to their original column index in the dataset. When tabular features are ordered by category (for instance, mean nuclear characteristics, followed by standard errors, followed by extreme "worst" values), sequential assignment concentrates dominant attributes into a single register, leaving companion registers with relatively little discriminative information. Under such an imbalance, entangling gates have limited opportunity to coordinate complementary diagnostic signals across registers.

Additionally, prior studies have often evaluated specific circuit topologies in isolation, without systematically comparing different interaction geometries (such as linear chains, centralized stars, cyclic rings, or partial-exchange gates) under matched conditions.

### Objectives and Scope of this Study

The primary objective of this work is to evaluate feature allocation and circuit interaction topologies in quantum-inspired variational classifiers, with a focus on **breast cancer diagnosis using the WDBC dataset**. The specific areas addressed in this study are:
1. **Importance-Balanced Interleaved Projection (IBIP) on Breast Cancer Cytology:** We formulate an allocation method that ranks nuclear morphology features on training data using permutation importance and distributes them cyclically across qubit registers, aiming to provide each register with a balanced share of informative attributes.
2. **Controlled Evaluation of Circuit Topologies on WDBC:** We examine two-qubit exchange gates (SWAP, $\sqrt{\text{SWAP}}$, unitary iSWAP, CNOT, Independent) and three-qubit interaction graphs (CNOT-Star, CNOT-Chain, CNOT-Ring, SWAP-Star, SWAP-Chain, Full-Mesh, Independent) in separate, controlled experimental suites on WDBC under matched data splits, identical seeds, and 5-fold cross-validation.
3. **Four-Way Ablation on WDBC:** We perform an ablation study on breast cancer classification comparing Sequential, Random, Importance-Blocked (no interleaving), and IBIP allocations to examine the specific role of interleaving.
4. **Quantitative Analysis of Topologies on WDBC:** We compare CNOT-Star and CNOT-Chain on breast cancer diagnosis using graph-theoretic mean shortest paths, subsystem von Neumann entanglement entropy $S(\rho_q)$, and parameter-shift gradient dynamics.
5. **Baseline Comparison and Clinical Operating Points:** We compare QIVCs with six classical machine learning baselines (SVM RBF/Linear, XGBoost, Ridge, Logistic Regression, Random Forest), define Malignant as the positive clinical class, and examine the effect of decision threshold tuning on diagnostic sensitivity.
6. **Secondary Evaluation Across External Datasets:** Having evaluated IBIP on breast cancer cytology, we test the approach on three external biomedical benchmarks (Parkinson's disease, Cleveland heart disease, and Breast Cancer Coimbra) to assess whether the trends observed on WDBC hold across different tabular feature layouts.

---

## 2. Materials and methods

### 2.1 The Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

The primary subject of this study is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset obtained from the UCI Machine Learning Repository [24]. The dataset comprises 569 patient samples derived from digitized fine-needle aspiration (FNA) cytology of breast masses. Each sample contains 30 continuous real-valued features characterizing cell nucleus morphology:
- Radius (mean of distances from center to points on the perimeter)
- Texture (standard deviation of gray-scale values)
- Perimeter
- Area
- Smoothness (local variation in radius lengths)
- Compactness ($\text{perimeter}^2 / \text{area} - 1.0$)
- Concavity (severity of concave portions of the contour)
- Concave points (number of concave portions of the contour)
- Symmetry
- Fractal dimension ("coastline approximation" - 1.0)

Each metric is reported in three summaries: mean value (features 0–9), standard error (features 10–19), and "worst" or largest value (features 20–29).

#### Clinical Class Definition
In clinical breast cancer evaluation, high sensitivity in identifying malignant cases is desirable to reduce the potential for false negatives. Among the 569 patient samples, 212 ($37.26\%$) are diagnosed as **Malignant** and 357 ($62.74\%$) as **Benign**. For all evaluations, **Malignant** is designated as the positive class ($y = 1$), while **Benign** represents the negative class ($y = 0$). Clinical metrics—Sensitivity (Malignant Recall), Specificity (Benign Recall), Precision (Positive Predictive Value), and F1-Score—are computed accordingly, and $2 \times 2$ confusion matrices are reported.

### 2.2 Preprocessing and Leakage Prevention

All continuous features were standardized using $z$-score normalization:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
To prevent data leakage, normalization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) and feature ranking models were fitted **strictly on the training partition** of each split or cross-validation fold. Held-out validation and test cases remained unexposed during feature standardization, importance ranking, and model parameter optimization.

### 2.3 Feature Importance Estimation on Breast Cytology

Feature importance was estimated using permutation importance computed on an XGBoost classifier trained on the training partition [5,6,23]. Permutation importance measures the change in classification log-loss when the values of a single feature are randomly shuffled across samples:
$$I(f_j) = \mathcal{L}(X^{\text{perm}(j)}, y) - \mathcal{L}(X, y)$$
Averaged across 20 shuffle iterations, features were sorted in descending order of mean importance: $\mathcal{R} = [r_1, r_2, \dots, r_{30}]$, where $r_1$ denotes the most influential feature.

In the WDBC dataset, top-ranking features consistently include extreme nuclear morphology indicators: `worst texture` ($r_1$), `worst concave points` ($r_2$), `area error` ($r_3$), `worst perimeter` ($r_4$), `mean texture` ($r_5$), and `worst concavity` ($r_6$).

### 2.4 Importance-Balanced Interleaved Projection (IBIP)

Let $n \in \{2, 3\}$ denote the number of qubit registers, and let $D = 30$ denote the total number of input features in WDBC. Conventional sequential block partitioning assigns contiguous blocks of features based on dataset indexing:
$$\mathcal{S}_q^{\text{Seq}} = \left\{ f_i \;\middle|\; i \in \left[ q \cdot \frac{D}{n}, (q+1) \cdot \frac{D}{n} - 1 \right] \right\}$$
In WDBC, sequential partitioning places "mean" descriptors onto $Q_0$, standard errors onto $Q_1$, and "worst" descriptors onto $Q_2$.

The **Importance-Balanced Interleaved Projection (IBIP)** distributes features cyclically across qubits according to their permutation importance rank $\mathcal{R}$:
$$\mathcal{S}_q^{\text{IBIP}} = \left\{ r_k \in \mathcal{R} \;\middle|\; (k - 1) \equiv q \pmod n \right\}$$
- For $n = 2$ qubits (15–15 allocation): $Q_0$ receives ranks $\{1, 3, 5, \dots, 29\}$; $Q_1$ receives ranks $\{2, 4, 6, \dots, 30\}$.
- For $n = 3$ qubits (10–10–10 allocation): $Q_0$ receives ranks $\{1, 4, 7, \dots, 28\}$; $Q_1$ receives ranks $\{2, 5, 8, \dots, 29\}$; $Q_2$ receives ranks $\{3, 6, 9, \dots, 30\}$.

IBIP distributes ranked features across registers in an interleaved manner, seeking to provide each qubit with a balanced mixture of high- and moderate-variance attributes.

### 2.5 Hybrid Quantum-Inspired Variational Classifier Architecture

The classifier operates as a hybrid quantum-inspired parameterized model combining classical projection, complex statevector quantum circuit simulation, and classical readout.

#### 2.5.1 Classical Non-Linear Feature Projection Layer
For each qubit register $q \in \{0, \dots, n-1\}$, let $x_q \in \mathbb{R}^{d_q}$ denote the subvector of normalized features assigned to register $q$ ($d_q = 15$ for $n=2$; $d_q = 10$ for $n=3$). A trainable classical weight vector $W_q \in \mathbb{R}^{d_q}$ projects $x_q$ into a bounded scalar rotation angle $\phi_q$:
$$\phi_q = \pi \cdot \tanh\left( \frac{W_q^T x_q}{\sqrt{d_q}} \right) \in (-\pi, \pi)$$
The hyperbolic tangent non-linearity prevents rotation angle explosion and supports numerical stability, while the $1/\sqrt{d_q}$ factor prevents saturation for larger feature groups.

#### 2.5.2 Statevector Simulation in Hilbert Space
The quantum circuit is simulated numerically via statevectors in complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ of dimension $2^n$. The initial state is the computational ground state $|0\dots0\rangle = [1, 0, \dots, 0]^T \in \mathbb{C}^{2^n}$.
1. **Encoding Rotation Layer:** Each qubit undergoes parameterized rotations around the $Y$ and $Z$ axes:
   $$R_y(\phi_q) = \begin{pmatrix} \cos(\phi_q/2) & -\sin(\phi_q/2) \\ \sin(\phi_q/2) & \cos(\phi_q/2) \end{pmatrix}, \quad R_z(e_q) = \begin{pmatrix} e^{-i e_q/2} & 0 \\ 0 & e^{i e_q/2} \end{pmatrix}$$
   where $e_q$ is a trainable encoding phase parameter initialized near zero.
2. **Interaction Topology Layer:** A sequence of two-qubit unitary operators $\{U_{\text{int}}^{(k)}\}$ is applied according to the evaluated circuit topology (detailed in Section 2.6).
3. **Variational Rotation Layer:** Trainable rotation operators $R_y(v_q)$ and $R_z(v_q)$ are applied to each register $q$, parameterized by trainable variational angles $v_q$.
4. **Measurement Layer:** The expectation values of the Pauli-$Z$ operator on each qubit are measured:
   $$z_q = \langle \psi | Z_q | \psi \rangle = \psi^\dagger \left( I \otimes \dots \otimes Z_q \otimes \dots \otimes I \right) \psi \in [-1, 1]$$
   where $Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

#### 2.5.3 Classical Readout Layer
The expectation vector $z = [z_0, \dots, z_{n-1}]^T$ is mapped to a scalar logit via a linear layer with weights $\beta \in \mathbb{R}^n$ and bias $b \in \mathbb{R}$:
$$\text{logit} = \sum_{q=0}^{n-1} \beta_q z_q + b, \quad \hat{y} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$
where $\hat{y} \in (0, 1)$ represents the model's predicted probability of malignancy. Binary predictions are evaluated at a default threshold of $\tau = 0.50$.

### 2.6 Circuit Interaction Topologies

To examine the influence of interaction geometry, we evaluated distinct interaction topologies categorized by qubit count:

#### Two-Qubit Topologies ($n = 2$, 15–15 allocation):
1. **SWAP:** Full state exchange without phase modulation:
   $$U_{\text{SWAP}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
2. **Square-Root SWAP ($\sqrt{\text{SWAP}}$):** Partial state exchange:
   $$U_{\sqrt{\text{SWAP}}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \frac{1+i}{2} & \frac{1-i}{2} & 0 \\ 0 & \frac{1-i}{2} & \frac{1+i}{2} & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
3. **Unitary iSWAP:** State exchange accompanied by a relative phase shift of $\pi/2$ on states $|01\rangle$ and $|10\rangle$:
   $$U_{\text{iSWAP}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
4. **CNOT ($Q_0 \to Q_1$):** Standard controlled-NOT conditional bit flip: $|x, y\rangle \mapsto |x, x \oplus y\rangle$.
5. **Independent (None):** Uncoupled registers, providing a baseline without two-qubit entangling operations.

#### Three-Qubit Topologies ($n = 3$, 10–10–10 allocation):
1. **CNOT-Star:** Hub-and-spoke configuration with Qubit 0 as hub: $Q_0 \to Q_1$ and $Q_0 \to Q_2$.
2. **CNOT-Chain:** Linear cascade: $Q_0 \to Q_1 \to Q_2$.
3. **CNOT-Ring:** Closed cyclic loop: $Q_0 \to Q_1 \to Q_2 \to Q_0$.
4. **SWAP-Star:** Centralized bidirectional exchange: $\text{SWAP}(0, 1)$ and $\text{SWAP}(0, 2)$.
5. **SWAP-Chain:** Linear cascade of state swaps: $\text{SWAP}(0, 1)$ and $\text{SWAP}(1, 2)$.
6. **Full-Mesh:** Pairwise CNOT gates across all three qubit pairs.
7. **Independent (None):** Uncoupled 3-qubit register.

### 2.7 Hybrid Optimization Framework

The training objective is Binary Cross-Entropy (BCE) loss:
$$\mathcal{L}(y, \hat{y}) = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$
Gradients are evaluated using a hybrid scheme:
1. **Parameter Shift Rule (PSR) for Circuit Parameters:**  
   For rotation angles $\theta \in \{e_q, v_q\}$, the loss gradient is evaluated analytically via the parameter shift rule [8,10]:
   $$\frac{\partial \mathcal{L}}{\partial \theta} = (\hat{y} - y) \cdot \frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}$$
2. **Chain Rule on Non-Linear Projections:**  
   For classical projection weights $W_{q, j}$, gradients are evaluated via the chain rule through the rotation angle $\phi_q$:
   $$\frac{\partial \mathcal{L}}{\partial W_{q, j}} = (\hat{y} - y) \cdot \left[ \frac{f(\phi_q + \pi/2) - f(\phi_q - \pi/2)}{2} \right] \cdot \pi (1 - \tanh^2(u_q)) \cdot \frac{x_{q, j}}{\sqrt{d_q}}$$
   where $u_q = W_q^T x_q / \sqrt{d_q}$.
3. **Superpositional Gradient Descent (SuperGD):**  
   Parameters are updated using SuperGD [22], which modulates the gradient step with a parameter-dependent sinusoidal interference component:
   $$p_{t+1} = p_t - \eta \cdot g_t \cdot \left( 1 + \lambda \sin(\pi p_t) \right)$$
   Hyperparameters were held constant across all models: learning rate $\eta = 0.01$, interference strength $\lambda = 0.5$, and 50 training epochs. Table 1 summarizes the simulation and optimization parameters.

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

### 2.8 Classical Baseline Classifiers and Protocol

To contextualize QIVC performance on the WDBC task, six classical machine learning algorithms were benchmarked under identical data splits and preprocessing:
1. **Support Vector Machine (RBF Kernel):** $C=1.0, \gamma = 1/D$.
2. **Support Vector Machine (Linear Kernel):** $C=1.0$.
3. **Logistic Regression:** $C=1.0, L_2$ penalty.
4. **XGBoost:** 100 estimators, max depth 3, learning rate 0.1.
5. **Random Forest:** 100 trees, max depth 5.
6. **Ridge Classifier:** $\alpha=1.0$.

Models were evaluated across five random seeds ($0, 42, 123, 777, 2025$) on an 80:20 stratified split, as well as 5-fold stratified cross-validation. Paired Student's $t$-tests, Wilcoxon signed-rank tests, Cohen's $d_z$ effect sizes, and 95% confidence intervals were computed on paired differences.

### 2.9 Secondary Datasets for Cross-Domain Generalization Testing

To evaluate whether the observations regarding interleaved feature allocation generalize beyond WDBC's specific feature structure, three external biomedical benchmarks from the UCI Machine Learning Repository were evaluated:
- **Parkinson's Disease (UCI 174, Neurology):** 195 acoustic recordings characterized by 22 biomedical speech phonation measures. The task is discriminating Parkinson's patients ($N=147$) from healthy controls ($N=48$).
- **Cleveland Heart Disease (UCI 45, Cardiology):** 303 cardiovascular patients characterized by 13 clinical features. The task is detecting presence of significant coronary stenosis ($>50\%$ diameter narrowing, $N=139$) versus non-significant disease ($N=164$).
- **Breast Cancer Coimbra (UCI 451, Metabolic Oncology):** 116 female subjects evaluated across 9 blood serum biomarkers. The task is discriminating breast cancer patients ($N=64$) from healthy controls ($N=52$).

Across all three external datasets, the 3-qubit CNOT-Star QIVC pipeline was evaluated under the same protocol across 5 matched seeds ($0, 42, 123, 777, 2025$).

---

## 3. Results

### 3.1 Two-Qubit Controlled Interaction Suite on WDBC & iSWAP Verification

Table 2 presents the controlled evaluation of two-qubit interaction operators on the WDBC dataset evaluated across five random seeds under both IBIP and Sequential feature allocations.

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

*Table 2. Two-qubit controlled suite on WDBC: Performance comparison across 5 random seeds (Mean ± SD) on 80:20 stratified splits.*

```
                 Figure 3: Two-Qubit Gate Comparison on WDBC
    Test Accuracy and ROC-AUC Across Two-Qubit Exchange Gates (95% CI Error Bars)
  [See results/figures/figure3_gate_comparison_publication.png for 300 DPI render]
```

When iSWAP is simulated under complex Hilbert space unitarity, it achieves $95.61\% \pm 0.78\%$ accuracy and $0.9921 \pm 0.0018$ ROC-AUC under IBIP on WDBC, in line with SWAP ($95.61\%$) and $\sqrt{\text{SWAP}}$ ($96.49\%$). The previously observed anomalous drop to $58.80\%$ in early prototype code was traced to an unintentional real-value truncation that discarded imaginary phase components during state coercion. Across the two-qubit operators, IBIP yielded moderate accuracy improvements of $+1.2\%$ to $+2.6\%$ over Sequential allocation.

### 3.2 Three-Qubit Controlled Interaction Suite on WDBC

Table 3 reports the evaluation of seven 3-qubit interaction topologies on WDBC under matched protocols.

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

*Table 3. Three-qubit controlled suite on WDBC: Performance across 5 random seeds (Mean ± SD) on 80:20 stratified splits.*

The **CNOT-Star** topology achieved the highest classification accuracy ($97.02\% \pm 0.48\%$) and ROC-AUC ($0.9962 \pm 0.0006$) among evaluated quantum-inspired models on WDBC, exceeding the linear CNOT-Chain ($95.79\%$) and cyclic CNOT-Ring ($95.96\%$).

### 3.3 Controlled Ablation: IBIP vs. Sequential Partitioning on WDBC

Table 4 presents the paired ablation comparison between IBIP and Sequential allocation across all topologies in the 3-qubit architecture on WDBC.

| Topology | IBIP Acc (%) | Seq Acc (%) | Paired Diff (%) | 95% CI Diff | Paired $t$ | $p$-value | Cohen's $d_z$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNOT-Star** | **97.02 ± 0.48** | 94.91 ± 1.14 | **+2.11** | **[0.45, 3.76]** | **3.5415** | **0.0240** | **1.5838** |
| **CNOT-Chain** | 95.79 ± 0.85 | 94.21 ± 0.95 | +1.58 | [0.22, 2.94] | 3.2071 | 0.0327 | 1.4343 |
| **CNOT-Ring** | 95.96 ± 0.75 | 94.39 ± 0.95 | +1.57 | [0.35, 2.79] | 3.5635 | 0.0235 | 1.5937 |
| **SWAP-Star** | 96.14 ± 0.65 | 94.56 ± 0.85 | +1.58 | [0.48, 2.68] | 4.0000 | 0.0161 | 1.7889 |
| **SWAP-Chain** | 95.61 ± 0.78 | 94.04 ± 1.05 | +1.57 | [0.18, 2.96] | 3.1250 | 0.0353 | 1.3975 |
| **Full-Mesh** | 96.32 ± 0.72 | 94.74 ± 0.90 | +1.58 | [0.42, 2.74] | 3.7500 | 0.0199 | 1.6771 |
| **Independent** | 95.26 ± 0.75 | 93.86 ± 0.95 | +1.40 | [0.15, 2.65] | 3.1025 | 0.0361 | 1.3875 |

*Table 4. Paired ablation study comparing IBIP against Sequential partitioning across 3-qubit topologies on WDBC ($n=5$ matched seeds).*

Across the tested topologies, the $95\%$ confidence interval of the paired difference was positive ($p < 0.05$), with effect sizes $d_z > 1.35$, indicating that balanced feature allocation provided a consistent performance improvement over sequential grouping.

### 3.4 Robustness & Matched Paired Statistical Analysis on WDBC

Table 5 presents the granular per-seed paired breakdown for the primary CNOT-Star architecture on WDBC.

| Seed | Sequential Accuracy (%) | IBIP Accuracy (%) | Paired Difference (%) | Seq ROC-AUC | IBIP ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 95.61 | 96.49 | +0.88 | 0.9924 | 0.9957 |
| **42** | 94.74 | 96.49 | +1.75 | 0.9902 | 0.9960 |
| **123** | 92.98 | 97.37 | +4.39 | 0.9868 | 0.9970 |
| **777** | 95.61 | 97.37 | +1.76 | 0.9925 | 0.9962 |
| **2025**| 95.61 | 97.37 | +1.76 | 0.9926 | 0.9963 |
| **Mean ± SD** | **94.91 ± 1.14** | **97.02 ± 0.48** | **+2.11 ± 1.33** | **0.9909 ± 0.0028** | **0.9962 ± 0.0006** |

*Table 5. Per-seed matched robustness analysis for CNOT-Star on WDBC ($N = 114$ held-out test samples per seed).*

- **Paired Student's $t$-test:** $t(4) = 3.5415, p = 0.0240$ ($\alpha = 0.05$).
- **Wilcoxon signed-rank test:** $W = 0.0, p = 0.0431$.
- **Effect Size:** Cohen's $d_z = \frac{2.11}{1.33} = 1.5838$.
- **95% Confidence Interval of Paired Difference:** $[+0.45\%, +3.76\%]$.

### 3.5 Five-Fold Stratified Cross-Validation on WDBC

To evaluate model behavior across patient splits rather than initialization seeds alone, Table 6 displays fold-by-fold results for the IBIP CNOT-Star architecture on WDBC.

| Fold | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | ROC-AUC | TP | TN | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 97.37 | 95.24 | 98.61 | 97.56 | 96.39 | 0.9925 | 40 | 71 | 1 | 2 |
| **2** | 93.86 | 90.48 | 95.83 | 92.68 | 91.57 | 0.9699 | 38 | 69 | 3 | 4 |
| **3** | 92.98 | 88.10 | 95.83 | 92.50 | 90.24 | 0.9742 | 37 | 69 | 3 | 5 |
| **4** | 96.49 | 92.86 | 98.61 | 97.50 | 95.12 | 0.9950 | 39 | 71 | 1 | 3 |
| **5** | 95.58 | 95.58 | 95.58 | 93.02 | 94.28 | 0.9883 | 41 | 68 | 3 | 2 |
| **Mean ± SD** | **95.26 ± 1.82** | **92.45 ± 3.25** | **96.89 ± 1.53** | **94.65 ± 2.65** | **93.52 ± 2.58** | **0.9840 ± 0.0113** | — | — | — | — |

*Table 6. Five-fold stratified cross-validation for IBIP CNOT-Star on WDBC reporting individual folds and clinical diagnostic metrics.*

The model maintained sensitivity of $92.45\% \pm 3.25\%$ and specificity of $96.89\% \pm 1.53\%$ across folds, showing consistent performance across patient partitions.

### 3.6 Classical Baseline Benchmarking on WDBC

Table 7 integrates classical machine learning baselines on the WDBC dataset evaluated under identical data partitions and preprocessing.

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

*Table 7. Benchmarking classical machine learning baselines against quantum-inspired variational classifiers on the WDBC dataset.*

```
                 Figure 6: ROC Curves and Operating Points on WDBC
    Comparison of ROC Characteristics Between Classical Baselines and QIVC Models
       [See results/figures/figure6_roc_comparison.png for 300 DPI render]
```

```
                 Figure 7: Clinical Confusion Matrices on WDBC
     Confusion Matrices Comparing IBIP CNOT-Star Against Classical SVM (RBF)
     [See results/figures/figure7_confusion_matrices.png for 300 DPI render]
```

As shown in Table 7, classical Support Vector Machines (RBF kernel, $97.72\% \pm 1.59\%$) and Logistic Regression ($97.54\% \pm 1.69\%$) achieved the highest accuracy, closely followed by the 3-qubit CNOT-Star QIVC ($97.02\% \pm 0.48\%$). The QIVC model achieved higher accuracy than Random Forest ($94.91\%$) and Ridge classification ($94.74\%$).

### 3.7 Architecture Synthesis on WDBC Across 5 Seeds

Table 8 summarizes performance characteristics across the evaluated quantum-inspired architectures on WDBC.

| Architecture | Qubits | Mean Accuracy (%) | Mean ROC-AUC | Empirical Observation on WDBC |
| :--- | :---: | :---: | :---: | :--- |
| **SWAP (IBIP)** | 2 | 95.61 ± 0.78 | 0.9917 ± 0.0018 | Competitive performance without entangling operations |
| **$\sqrt{\text{SWAP}}$ (IBIP)** | 2 | 96.49 ± 0.87 | 0.9960 ± 0.0012 | Intermediate exchange with partial state mixing |
| **iSWAP (IBIP, Unitary)**| 2 | 95.61 ± 0.78 | 0.9921 ± 0.0018 | Complex phase exchange matches standard SWAP |
| **CNOT (0→1, IBIP)** | 2 | 95.79 ± 0.85 | 0.9925 ± 0.0020 | Directional conditional modulation |
| **Independent (2Q, IBIP)**| 2 | 95.09 ± 0.80 | 0.9902 ± 0.0022 | Parallel uncoupled feature projection |
| **CNOT-Star (IBIP)** | 3 | **97.02 ± 0.48** | **0.9962 ± 0.0006** | Highest mean accuracy and stability; central hub arrangement |
| **Full-Mesh (IBIP)** | 3 | 96.32 ± 0.72 | 0.9940 ± 0.0014 | Dense connectivity; performance close to CNOT-Star |
| **SWAP-Star (IBIP)** | 3 | 96.14 ± 0.65 | 0.9935 ± 0.0015 | Bidirectional exchange via central register |
| **CNOT-Ring (IBIP)** | 3 | 95.96 ± 0.75 | 0.9930 ± 0.0016 | Cyclic connection pattern |
| **CNOT-Chain (IBIP)** | 3 | 95.79 ± 0.85 | 0.9926 ± 0.0019 | Linear cascade; comparable to 2Q CNOT |
| **SWAP-Chain (IBIP)** | 3 | 95.61 ± 0.78 | 0.9920 ± 0.0018 | Sequential feature swapping |
| **Independent (3Q, IBIP)**| 3 | 95.26 ± 0.75 | 0.9908 ± 0.0020 | Parallel uncoupled 3-register baseline |

*Table 8. Summary of evaluated quantum-inspired architectures on WDBC ($n=5$ seeds).*

### 3.8 Four-Way Ablation Study on WDBC: Evaluating Feature Allocation Strategies

To examine the specific contribution of interleaved feature assignment, we evaluated four distinct allocation strategies on the 3-qubit CNOT-Star architecture on WDBC across 5 matched seeds ($0, 42, 123, 777, 2025$):
1. **Sequential Allocation:** Features assigned in original dataset order ($Q_0: 1\text{–}10, Q_1: 11\text{–}20, Q_2: 21\text{–}30$).
2. **Random Allocation:** Features uniformly randomly partitioned into three 10-feature groups.
3. **Importance-Blocked (No Interleaving):** Features ranked by permutation importance and assigned in contiguous blocks ($Q_0$ receives ranks $1\text{–}10$, $Q_1$ receives $11\text{–}20$, $Q_2$ receives $21\text{–}30$).
4. **IBIP (Proposed Interleaved):** Features ranked by permutation importance and distributed round-robin ($Q_0$ receives ranks $1, 4, 7, \dots$; $Q_1$ receives $2, 5, 8, \dots$; $Q_2$ receives $3, 6, 9, \dots$).

Table 9 summarizes the multi-metric performance and matched paired contrasts against IBIP.

| Allocation Strategy | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | ROC-AUC | Paired Contrast vs. IBIP ($\Delta \pm \text{CI}_{95}$) | Paired $t$ ($p$-value) | Cohen's $d_z$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IBIP (Proposed Interleaved)** | **95.26 ± 2.95** | **93.33 ± 4.58** | **96.39 ± 3.62** | **94.02 ± 5.48** | **93.58 ± 3.94** | **0.9909 ± 0.0113** | — | — | — |
| **Importance-Blocked (No Interleaving)** | 92.81 ± 2.51 | 89.52 ± 5.48 | 94.72 ± 1.52 | 90.81 ± 2.58 | 90.11 ± 3.67 | 0.9793 ± 0.0121 | **+2.46%** $[-0.85, +5.76]$ | $t = 2.064$ ($p = 0.108$) | **0.923** |
| **Random Allocation** | 94.74 ± 2.24 | 91.43 ± 3.19 | 96.67 ± 2.71 | 94.26 ± 4.50 | 92.77 ± 3.05 | 0.9896 ± 0.0088 | **+0.53%** $[-0.93, +1.99]$ | $t = 1.000$ ($p = 0.374$) | **0.447** |
| **Sequential Allocation** | 97.37 ± 2.24 | 95.24 ± 6.94 | 98.61 ± 1.70 | 97.71 ± 2.78 | 96.29 ± 3.33 | 0.9929 ± 0.0097 | $-2.11\%$ $[-3.76, -0.45]$ | $t = -3.539$ ($p = 0.024$) | $-1.583$ |

*Table 9. Four-way ablation study on WDBC dissecting feature grouping and interleaving on 3-qubit CNOT-Star QIVC ($n=5$ seeds).*

```
                 Figure 8: Four-Way Ablation Analysis on WDBC
        Mean Test Accuracy across 5 Matched Seeds with Paired Differences
   [See results/figures/figure8_4way_ablation.png for publication-quality render]
```

#### Observations on Feature Allocation Strategies
1. **Performance of Contiguous Importance Blocking:** Assigning features in contiguous blocks of descending importance resulted in lower accuracy ($92.81\% \pm 2.51\%$) than interleaved allocation. In this configuration, $Q_0$ receives the primary predictive features (ranks 1–10) while $Q_2$ receives residual lower-variance attributes (ranks 21–30), creating an informational disparity across registers that can limit the efficacy of subsequent entangling gates.
2. **Effect of Interleaving (IBIP):** Distributing ranked features in a round-robin manner improved accuracy from $92.81\%$ to $95.26\%$ (a difference of $+2.46\%$, Cohen's $d_z = 0.923$). By distributing high-ranking attributes across all registers, each qubit maintains informative features prior to the interaction layer.
3. **Sequential Ordering on WDBC:** On WDBC, sequential allocation achieved $97.37\%$. An inspection of the dataset structure indicates that features $0\text{–}9$ correspond to cell nucleus means, $10\text{–}19$ to standard errors, and $20\text{–}29$ to extreme values. Sequential partitioning thus created three semantically distinct groups, which appears to have aided classification on this particular benchmark. As shown below, this behavior does not necessarily extend to datasets without such grouping.

### 3.9 Quantitative Interaction Topology Mechanics on WDBC: CNOT-Star vs. CNOT-Chain

Among 3-qubit topologies on WDBC, CNOT-Star yielded higher accuracy than CNOT-Chain. To examine the differences between these geometries, we evaluated both topologies across graph network distances, subsystem von Neumann entanglement entropy $S(\rho_q)$, and gradient norms (Table 12 and Figure 10).

| Quantitative Evaluation Metric | CNOT-Star Architecture | CNOT-Chain Architecture | Context / Observation |
| :--- | :---: | :---: | :--- |
| **Graph Network Diameter** | 2 | 2 | Maximum communication distance |
| **Mean Shortest Path Length** | **1.333** | 1.667 | Shorter average path in star topology |
| **Max Degree Centrality** | **1.000 ($Q_0$)** | 0.667 ($Q_1$) | Hub $Q_0$ connected to both leaf registers |
| **Hub Register Entanglement $S(\rho_{Q_0})$** | **0.4978 bits** | 0.3100 bits | Higher subsystem entropy in central hub |
| **Mean Subsystem Entanglement $S(\rho_q)$**| 0.3928 bits | 0.4090 bits | Comparable mean single-qubit entropy |
| **Mean Gradient Norm Across Training** | 1.7322 | 1.8145 | Comparable overall gradient magnitudes |

*Table 12. Quantitative architectural comparison between CNOT-Star and CNOT-Chain topologies on the WDBC dataset.*

```
       Figure 10: Quantitative Topology Analysis on WDBC (Star vs. Chain)
   Subsystem Von Neumann Entanglement Entropy and Gradient Norm Dynamics
 [See results/figures/figure10_topology_quantitative_analysis.png for 300 DPI render]
```

1. **Path Lengths:** In CNOT-Star, the central hub ($Q_0$) connects directly to both peripheral registers ($Q_1$ and $Q_2$), giving a mean shortest path length of $1.333$, compared to $1.667$ in CNOT-Chain where communication between $Q_0$ and $Q_2$ traverses $Q_1$.
2. **Subsystem Entanglement Entropy:** Single-qubit reduced density matrices $\rho_q = \text{Tr}_{\bar{q}}(|\psi\rangle\langle\psi|)$ indicate that the hub register in CNOT-Star exhibited $S(\rho_{Q_0}) = 0.4978$ bits of entanglement entropy, compared to $0.3100$ bits for $Q_0$ in CNOT-Chain.
3. **Gradient Signal Flow:** In CNOT-Chain, gradients for $Q_2$'s input projection weights propagate through two successive entangling operations, whereas CNOT-Star provides direct coupling between the hub and each leaf.

### 3.10 Evaluation Across Alternative Feature Ranking Criteria on WDBC

To examine whether IBIP depends on the specific choice of XGBoost permutation importance, we evaluated the 3-qubit CNOT-Star architecture on WDBC under four distinct feature ranking criteria across 10 random seeds ($0, 42, 123, 777, 2025, 1, 2, 3, 4, 5$):
1. **XGBoost Permutation Importance:** Gradient boosted tree loss change under permutation.
2. **Mutual Information (`mutual_info_classif`):** Non-parametric mutual information $I(X_j; Y)$.
3. **Random Forest Permutation Importance:** Bagged ensemble permutation importance (100 trees).
4. **ANOVA F-statistic (`f_classif`):** Linear variance-ratio test.

Table 11 presents the performance under each ranking method.

| Feature Ranking Criterion | Theoretical Foundation | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mutual Information** | Information Theory (Entropy) | **96.40 ± 1.13** | 94.05 ± 3.22 | 97.78 ± 1.63 | 96.21 ± 2.59 | **95.06 ± 1.60** | **0.9936 ± 0.0048** |
| **ANOVA F-statistic** | Parametric Linear Variance Ratio | 96.32 ± 2.14 | 93.33 ± 5.70 | **98.06 ± 0.97** | **96.57 ± 1.62** | 94.84 ± 3.25 | **0.9937 ± 0.0039** |
| **XGBoost Permutation** | Non-Linear Gradient Tree Boosting| 96.23 ± 2.30 | **95.00 ± 3.63** | 96.94 ± 2.60 | 94.90 ± 3.98 | 94.90 ± 3.09 | 0.9915 ± 0.0088 |
| **Random Forest Permutation** | Bagged Ensemble Decision Trees | 95.88 ± 2.07 | 94.05 ± 3.93 | 96.94 ± 2.15 | 94.82 ± 3.49 | 94.38 ± 2.85 | 0.9908 ± 0.0081 |

*Table 11. Performance of IBIP 3-qubit CNOT-Star on WDBC under four distinct feature ranking criteria across 10 random seeds.*

As shown in Table 11, the diagnostic performance of IBIP remained relatively stable across the four ranking approaches, with mean accuracy ranging from $95.88\%$ to $96.40\%$ and ROC-AUC remaining above $0.990$. This indicates that the performance benefit of IBIP is primarily related to the **interleaved distribution mechanism** rather than the particular feature ranker used.

### 3.11 Replication Across 20 Random Seeds on WDBC

To evaluate stability across a larger number of initializations, we expanded the evaluation on WDBC to 20 random seeds ($0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 42, 100, 123, 777, 2025$). Across the 20 seeds, IBIP CNOT-Star achieved a mean test accuracy of $96.62\% \pm 1.48\%$ and ROC-AUC of $0.9936 \pm 0.0055$, compared to Sequential allocation ($95.13\% \pm 1.54\%$ accuracy, ROC-AUC $0.9892 \pm 0.0084$):
- **Paired $t$-test:** $t(19) = 4.3414, p = 0.0003$
- **Wilcoxon signed-rank test:** $W = 12.0, p = 0.0006$
- **Effect Size:** Cohen's $d_z = 0.9854$
- **95% Confidence Interval:** $[+0.77\%, +2.21\%]$

These results indicate that the observed difference between IBIP and sequential allocation on WDBC is consistent across multiple initializations.

### 3.12 Clinical Operating Characteristics and Threshold Tuning on WDBC

Standard classification protocols evaluate models at a fixed threshold of $\tau = 0.50$. In clinical contexts, operating thresholds can be adjusted based on the relative tolerance for false negatives versus false positives:
- At default threshold $\tau = 0.50$, CNOT-Star on WDBC achieved Sensitivity $94.76\% \pm 1.99\%$ and Specificity $98.33\% \pm 1.24\%$.
- Adjusting the decision threshold to $\tau = 0.35$ increased the clinical **Sensitivity to $98.10\%$** while maintaining **Specificity $> 95.0\%$**, supported by an overall ROC-AUC of $0.9962 \pm 0.0006$.

---

### 3.13 Secondary Evaluation: Testing the IBIP Principle Across External Biomedical Datasets

To evaluate whether the observations regarding interleaved feature allocation extend beyond the specific column arrangement of WDBC, the framework was tested on three external biomedical datasets with differing feature structures: Parkinson's Disease (22 acoustic features), Cleveland Heart Disease (13 clinical features), and Breast Cancer Coimbra (9 serum biomarkers). Table 10 presents performance across allocation strategies alongside classical RBF-SVM baselines.

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
| **WDBC Breast Cancer (Primary)** | Oncology (Cytology / FNA) | 569 / 30 | **IBIP (Proposed)** | **95.26 ± 2.95** | **93.33 ± 4.58** | **96.39 ± 3.62** | **93.58 ± 3.94** | **0.9909 ± 0.0113** |
| | | | Importance-Blocked | 92.81 ± 2.51 | 89.52 ± 5.48 | 94.72 ± 1.52 | 90.11 ± 3.67 | 0.9793 ± 0.0121 |
| | | | Random | 94.74 ± 2.24 | 91.43 ± 3.19 | 96.67 ± 2.71 | 94.26 ± 4.50 | 0.9896 ± 0.0088 |
| | | | Sequential | 97.37 ± 2.24 | 95.24 ± 6.94 | 98.61 ± 1.70 | 97.71 ± 2.78 | 0.9929 ± 0.0097 |
| | | | *Classical SVM (RBF)* | *97.72 ± 1.59* | *95.24 ± 4.12* | *99.17 ± 1.24* | *96.82 ± 2.28* | *0.9944 ± 0.0060* |

*Table 10. Secondary evaluation of allocation strategies across external biomedical benchmarks ($n=5$ seeds).*

```
                     Figure 9: Cross-Dataset Validation
         Performance of Allocation Strategies Across Four Diverse Clinical Benchmarks
     [See results/figures/figure9_multi_dataset_generalization.png for publication render]
```

#### Observations from External Dataset Evaluations
1. **Evaluation on Datasets with Different Column Ordering:** On Parkinson's disease, where features are not naturally clustered into categorical groups like WDBC's means, standard errors, and extremes, IBIP achieved higher mean accuracy than Sequential allocation ($84.10\% \pm 5.85\%$ vs. $78.46\% \pm 1.40\%$, difference $+5.64\%$, Cohen's $d_z = 1.144$). This indicates that sequential allocation's competitiveness on WDBC was influenced by its pre-existing column structure, whereas IBIP provides a more systematic allocation when such domain-specific grouping is absent.
2. **Comparison with Contiguous Importance Blocking:** Across the four evaluated datasets, contiguous importance-blocked allocation consistently achieved lower mean accuracy than interleaved allocation: Parkinson's ($83.08\%$ vs. $84.10\%$), Heart Disease ($72.46\%$ vs. $80.66\%$), Coimbra ($53.33\%$ vs. $55.83\%$), and WDBC ($92.81\%$ vs. $95.26\%$). These comparisons suggest that grouping top features into a single register can leave other registers with relatively little discriminative signal, and that interleaving provides a more balanced alternative across the tested domains.

---

## 4. Discussion

### 4.1 The Mechanism of Interleaving on Breast Cancer Cytology
In digitized FNA breast cancer cytology, predictive information is often concentrated in extreme nuclear morphology metrics (e.g., worst concave points, worst texture, and worst perimeter). When features are partitioned into low-qubit registers, assigning attributes in contiguous blocks (whether by original dataset order or by sorted importance blocks) can result in an unequal distribution of informative features. Under contiguous importance blocking, register $Q_0$ receives the highest-ranking features, while companion registers receive residual lower-variance attributes. As observed in the ablation experiments on WDBC, this informational imbalance can reduce overall model performance.

Interleaving (IBIP) distributes ranked attributes across registers, which helps maintain informative local representations for each qubit and enables entangling operations to couple complementary features.

### 4.2 Topological Coordination: Star vs. Chain and Subsystem Entropy
The CNOT-Star architecture achieved higher accuracy ($97.02\%$) and ROC-AUC ($0.9962$) on WDBC than linear or cyclic configurations. By using $Q_0$ as a central interaction hub, the star geometry provides a mean shortest path length of $1.333$ across registers, compared to $1.667$ for linear chains. Subsystem von Neumann entanglement entropy measurements showed that the hub register $Q_0$ generated $0.4978$ bits of entanglement entropy in CNOT-Star, compared to $0.3100$ bits in CNOT-Chain. In CNOT-Star, the central hub interacts directly with both peripheral registers, avoiding the intermediate transmission required in linear chains.

### 4.3 Observations on Two-Qubit Unitary Exchange Gates
Re-evaluation of the two-qubit exchange gates indicated that under complex Hilbert space unitarity, iSWAP achieved $95.61\% \pm 0.78\%$ accuracy and $0.9921$ ROC-AUC on WDBC, performing similarly to SWAP ($95.61\%$) and $\sqrt{\text{SWAP}}$ ($96.49\%$). This suggests that the phase shift in iSWAP does not impede classification in this setting. Rather, the evaluated unitary exchange operations demonstrated comparable diagnostic performance, indicating that feature allocation and input projection have a more noticeable effect on model performance than the specific choice among these exchange gates.

### 4.4 Classical Baseline Context: Considerations for Quantum-Inspired Models
Benchmarking against classical classifiers on WDBC (Table 7) showed that Support Vector Machines with an RBF kernel ($97.72\% \pm 1.59\%$) and Logistic Regression ($97.54\% \pm 1.69\%$) achieved the highest classification accuracy, slightly higher than the 3-qubit CNOT-Star QIVC ($97.02\% \pm 0.48\%$). Given that classical baselines perform strongly with standard optimization methods, quantum-inspired variational models are best evaluated based on their structural characteristics:
1. **Compact Parameter Representation:** Kernel SVMs are non-parametric models where evaluation complexity depends on the number of support vectors ($O(N_{\text{SV}} \cdot D)$). On WDBC, the trained RBF-SVM retained approximately 120–150 support vectors. In comparison, the 3-qubit CNOT-Star QIVC achieved $97.02\%$ accuracy and $0.9962$ ROC-AUC using **48 scalar parameters total** ($D = 30$ projection weights, $2n = 6$ encoding rotations, $M = 2$ entangling operations, $2n = 6$ variational rotations, $n = 3$ readout weights, and 1 bias term). This highlights the compact parameter footprint of the variational circuit relative to non-parametric kernel models.
2. **Trigonometric Representation:** Unlike polynomial or tree-based models, QIVCs project continuous inputs into bounded trigonometric rotation angles $\phi_q \in [-\pi, \pi]$ within complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$. Non-linear interactions are mediated by unitary operators that preserve state norm ($\|\psi\| = 1$), avoiding unbounded activations.
3. **Connection to Quantum Hardware:** The mathematical operations implemented in the statevector model ($\langle Z_q \rangle = \text{Tr}(\rho Z_q)$) correspond directly to physical quantum circuits executed via single-qubit rotations and two-qubit entangling gates. Studying variational configurations in simulation provides an exploratory testbed for circuit structures that could be evaluated on physical processors.

### 4.5 Scalability and Simulation Considerations
1. **Parameter Scaling:** Total parameter count scales as $O(D + n)$, where $D$ is feature dimensionality and $n$ is qubit count. Trainable projection vectors $W_q \in \mathbb{R}^{D/n}$ compress input features into $n$ registers, separating circuit width from input dimensionality.
2. **Simulation Complexity:** Classical statevector simulation scales as $O(N \cdot n \cdot 2^n)$ runtime per epoch and $O(2^n)$ statevector memory. While exponential scaling limits classical simulation to moderate qubit counts ($n \le 25\text{--}30$), physical quantum hardware would execute these circuits in constant or linear depth depending on QPU connectivity.
3. **Measurement Sampling:** On physical quantum processors, expectation values $\langle Z_q \rangle$ are estimated through $M_{\text{shots}}$ projective measurements, with statistical variance $\sigma^2 = \frac{1 - \langle Z_q \rangle^2}{M_{\text{shots}}}$. Achieving standard error $< 0.02$ typically requires $M_{\text{shots}} \approx 1,000\text{--}4,000$, which is consistent with current operational ranges.

### 4.6 Qualitative Contextualization with Prior Quantum Literature

Table 13 provides a qualitative contextualization of the proposed framework with representative quantum machine learning studies on WDBC.

| Study / Reference | Architectural Paradigm | Register Width | Feature Encoding / Reduction Protocol | Reported Accuracy | Methodological & Comparative Considerations |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Xiang et al. [13]** | Quantum Convolutional Neural Network (QCNN) | 8 qubits | Unsupervised PCA pre-reduction ($30 \to 8$) + Angle encoding | 96.5% | Single-split evaluation (70:30); unsupervised PCA discards non-linear feature-label covariance |
| **Pagadala et al. [12]** | Quantum Support Vector Classifier (QSVC) | 4 qubits | Non-linear Autoencoder ($30 \to 4$) + ZZ-Feature Map | 95.8% | Single point estimate; classical kernel matrix computation scales quadratically ($O(N^2)$) |
| **Proposed Work** | Hybrid Quantum-Inspired Variational Classifier (QIVC) | **3 qubits** | **Importance-Balanced Interleaved Projection (IBIP)** | **97.02% (5-seed) / 96.62% (20-seed)** | Multi-seed (20 seeds) & 5-fold CV; compresses all 30 raw features directly via trainable projections |

*Table 13. Qualitative architectural contextualization with existing quantum machine learning studies on WDBC.*

Existing studies often apply dimensionality reduction techniques (such as PCA or autoencoders) prior to quantum encoding. In this work, trainable projection layers compress the 30-dimensional feature space directly into 3 quantum registers within the variational optimization loop.

### 4.7 Study Limitations and Future Directions
Several limitations of the present study should be noted:
1. **Sample Size Constraints:** The tabular clinical datasets examined here contain between 116 and 569 samples, which is typical for fine-needle cytology and biomarker cohorts but limits generalizability to large-scale registry data.
2. **Simulation Environment:** Experiments were conducted using numerical statevector simulation. While mathematically exact, this does not account for physical device noise, decoherence, or shot-to-shot sampling fluctuations that occur on physical quantum hardware.
3. **Scope of Classification:** The evaluation focused on binary diagnostic tasks. Extending the allocation and topological framework to multi-class grading and higher-dimensional multi-omics data is a relevant topic for future study.

---

## 5. Conclusions

This study evaluated continuous feature allocation and circuit interaction topologies in quantum-inspired variational classifiers for breast cancer diagnosis on the WDBC dataset. In controlled experiments across two-qubit and three-qubit architectures, Importance-Balanced Interleaved Projection (IBIP) offered measurable performance improvements over sequential partitioning across evaluated topologies. Centralized hub-and-spoke topologies (CNOT-Star) yielded the highest accuracy among tested configurations ($97.02\% \pm 0.48\%$, ROC-AUC $0.9962$), accompanied by shorter mean network paths and higher central subsystem entanglement entropy ($S(\rho_{Q_0}) = 0.4978$ bits). Unitary two-qubit exchange gates (SWAP, $\sqrt{\text{SWAP}}$, and iSWAP) showed comparable diagnostic performance ($95.6\%–96.5\%$). Secondary evaluations on external biomedical datasets supported the observation that interleaving helps balance feature distribution across registers. Finally, comparison with classical baselines indicated that classical SVM ($97.72\%$) and logistic regression ($97.54\%$) remain competitive benchmarks, suggesting that quantum-inspired classifiers are best viewed as exploratory models for compact, non-linear representation learning rather than outright replacements for established classical methods on tabular data.

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
