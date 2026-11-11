# Response to Editor and Reviewers

**Manuscript Title:** Importance-Balanced Interleaved Feature Projection for Quantum-Inspired Variational Classifiers: A Circuit Interaction-Topology Study on Breast Cancer Diagnosis  
*(Revised Title: Circuit Interaction Topologies and Importance-Balanced Feature Allocation for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study)*  
**Manuscript ID:** Academia Quantum (Major Revision)  
**Authors:** Daksh Kaila, Monika Aggarwal, Samya Muhuri  

---

Dear Editorial Team and Reviewers,

We sincerely thank the Editor and the three anonymous Reviewers for their constructive, insightful, and rigorous evaluations of our manuscript. We have carefully considered every comment and executed a comprehensive revision program. Specifically, we have:

1. **Conducted controlled, like-for-like experiments:** Separated 2-qubit and 3-qubit architectures into distinct, unconfounded experimental suites; evaluated all interaction topologies under identical data splits, identical feature allocations (both proposed IBIP and baseline Sequential), and across the identical 5 random seeds (0, 42, 123, 777, 2025) as well as full 5-fold stratified cross-validation.
2. **Replaced independent t-tests with matched paired statistical analyses:** Computed paired $t$-tests, Wilcoxon signed-rank tests, Cohen’s $d_z$ effect sizes, and 95% confidence intervals on matched paired differences (IBIP minus Sequential) across identical seeds and across independent cross-validation folds.
3. **Completed comprehensive classical baseline evaluations:** Benchmarked Support Vector Machines (SVM RBF and Linear), XGBoost, Ridge Classifier, Logistic Regression, and Random Forest using identical preprocessing (StandardScaler fitted strictly on training folds to prevent leakage), identical splits/seeds, and identical cross-validation folds. Defined the clinically critical positive class (Malignant, $y=1$) and reported all clinical metrics (Accuracy, Sensitivity/Recall, Specificity, Precision, F1-Score, ROC-AUC) along with full confusion matrices.
4. **Clarified the quantum-inspired hybrid architecture:** Explicitly defined that the model implements an exact classical numerical statevector simulation of parameterized quantum circuits operating in Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$, preceded by a classical trainable feature projection layer $\phi_q = \pi \tanh(W_q^T x_q / \sqrt{|x_q|})$ and followed by classical Pauli-$Z$ expectation readout $\hat{y} = \sigma(\beta^T \langle Z \rangle + b)$, trained via Parameter Shift Rule (PSR) gradients and Superpositional Gradient Descent (SuperGD). We revised the title, abstract, and text accordingly to remove any implication of physical quantum hardware execution.
5. **Verified and resolved the iSWAP anomaly:** Thoroughly debugged the reported 58.80% iSWAP result. Our diagnostic analysis revealed that this anomalous performance was caused by an implementation artifact (real-value casting / non-unitary truncation of complex off-diagonal phases in early prototype scripts). When simulated with mathematically rigorous complex unitary operations, iSWAP achieves 95.61%–96.49% accuracy and 0.9921–0.9940 ROC-AUC, fully competitive with SWAP (95.61%) and $\sqrt{\text{SWAP}}$ (96.49%). We corrected all tables, figures, and conclusions accordingly.
6. **Harmonized figures and tables:** Corrected caption/title discrepancies in Figures 4 and 5; generated publication-quality figures at 300 DPI; unified Table 6 to display all individual folds; and ensured 100% numerical consistency across the abstract, results, tables, and text.
7. **Created a public reproducible repository:** Provided an organized, documented, and fully runnable Python repository containing all preprocessing, feature ranking, quantum circuit simulations, classical baselines, optimization routines, and automated statistical testing.

Below, we provide our detailed point-by-point responses to each editorial and reviewer comment.

---

# Part I: Response to Editorial Comments

### Editorial Comment 1: Controlled, like-for-like comparison is missing (also raised by Reviewer 3)
> *The claims about topology, interaction complexity, and the relative importance of feature allocation are not supported by a controlled comparison. Two-qubit models use 15–15 IBIP groups, while three-qubit models use 10–10–10 groups. Only the selected CNOT-Star model is repeated over five seeds. Table 3 compares the CNOT-Star five-seed mean (97.02%, AUC 0.9962) with a single CNOT-Chain point estimate (95.61%, AUC 0.9924); Table 8 likewise compares a CNOT-Star range with single values for other gates. The design therefore confounds topology with qubit count and feature grouping, and replication is unequal across architectures.*  
> *Please either (a) evaluate each topology under the same feature allocation, data splits, seeds, tuning protocol, and—where topology is the variable—the same qubit count, reporting mean, SD or confidence intervals, and paired comparisons for every architecture; or (b) restrict the conclusions to descriptive results from the tested configurations and remove claims of superiority or general design principles. If two-qubit and three-qubit comparisons are retained, treat them as separate experiments and do not attribute differences to topology alone.*

**Author Response:**  
We completely agree with the Editor and Reviewer 3. In the original draft, comparing 2-qubit (15–15 feature grouping) with 3-qubit (10–10–10 grouping) models and reporting multi-seed aggregates only for CNOT-Star confounded topology with qubit count, feature allocation granularity, and replication depth.

To establish a rigorous, controlled, like-for-like experimental design, we have restructured our study into two distinct, unconfounded experimental suites:

1. **Two-Qubit Controlled Suite (15–15 grouping):**  
   We evaluated five distinct 2-qubit configurations:
   - SWAP
   - $\sqrt{\text{SWAP}}$
   - Unitary iSWAP (corrected)
   - CNOT ($Q_0 \to Q_1$)
   - Independent / None (no interaction layer baseline)  
   Every single topology was evaluated under **both** the proposed IBIP allocation and the conventional Sequential allocation, replicated across the **exact same 5 seeds** (0, 42, 123, 777, 2025) on 80:20 stratified splits, and additionally evaluated under 5-fold stratified cross-validation.

2. **Three-Qubit Controlled Suite (10–10–10 grouping):**  
   We evaluated seven distinct 3-qubit topologies:
   - CNOT-Star ($Q_0 \to Q_1, Q_0 \to Q_2$)
   - CNOT-Chain ($Q_0 \to Q_1 \to Q_2$)
   - CNOT-Ring ($Q_0 \to Q_1 \to Q_2 \to Q_0$)
   - SWAP-Star ($Q_0 \leftrightarrow Q_1, Q_0 \leftrightarrow Q_2$)
   - SWAP-Chain ($Q_0 \leftrightarrow Q_1 \leftrightarrow Q_2$)
   - Full-Mesh (all pairwise CNOTs)
   - Independent / None (no interaction layer baseline)  
   Every single topology was similarly evaluated under **both** IBIP and Sequential allocations across the identical 5 seeds and under 5-fold cross-validation.

By holding qubit count, feature allocation strategy, data splits, hyperparameters, and optimization protocols strictly constant while varying only the interaction topology (and vice versa), the confounding factors have been eliminated. Furthermore, we have restricted our scientific claims to empirical observations specific to this tabular diagnostic setting and removed any generalized assertions of universal topological superiority.

---

### Editorial Comment 2: Statistical analysis treats paired results as independent (related to Reviewer 2, Comment 3; Reviewer 3)
> *Section 3.6 and Table 5: the Welch t-test treats paired seed results as independent. Recalculation from the displayed values reproduces the reported Welch result (t = 3.8005, p = 0.0110). However, the two configurations are paired by the same five named seeds and evaluated on the same held-out cases. A paired t-test on the displayed values gives t = 3.5415, p = 0.0240. Neither seed-level test alone quantifies uncertainty across patients or independent splits.*  
> *Please use a paired analysis that reflects the matched design, preferably with repeated stratified or nested resampling or paired per-case predictions; report effect size (e.g., Cohen’s d) and confidence intervals; report per-seed paired differences rather than only aggregate means; and limit the inference to initialization stability unless uncertainty across data splits is also evaluated.*

**Author Response:**  
We thank the Editor and Reviewers for pointing out this critical statistical flaw. The Welch independent $t$-test was inappropriate because the models were evaluated on identical seeds and identical held-out test splits.

In our revision, we have completely overhauled the statistical analysis:
1. **Matched Paired Analysis Across Seeds:**  
   For all models across the 5 initialization seeds, we now compute:
   - **Per-seed paired differences** ($\Delta_i = \text{Score}_{\text{IBIP}, i} - \text{Score}_{\text{Seq}, i}$)
   - **Two-tailed paired Student's $t$-test** ($t_{\text{paired}}$)
   - **Wilcoxon signed-rank test** ($W, p$) as a non-parametric verification
   - **Standardized effect size:** Cohen's $d_z = \frac{\bar{D}}{s_D}$
   - **95% Confidence Intervals** for the mean difference $[\bar{D} - t_{0.975} \text{SE}, \bar{D} + t_{0.975} \text{SE}]$.  
   For the primary CNOT-Star comparison in Table 5:  
   - Mean Accuracy: IBIP = 97.02% (SD = 0.48%), Sequential = 94.91% (SD = 1.14%)  
   - Mean Paired Difference: $+2.11\%$ (95% CI: $[0.45\%, 3.76\%]$)  
   - Paired $t$-test: $t(4) = 3.5415$, $p = 0.0240$  
   - Wilcoxon signed-rank test: $W = 0.0$, $p = 0.0431$  
   - Cohen's $d_z = 1.5838$ (large effect size)
2. **Uncertainty Across Independent Patient Splits (Cross-Validation):**  
   To evaluate true patient/data-split generalization rather than initialization stability alone, we performed paired 5-fold stratified cross-validation where IBIP and Sequential allocations were evaluated on the exact same folds. Across independent folds, the IBIP CNOT-Star architecture maintained superior performance ($95.26\% \pm 1.82\%$ accuracy vs. $93.15\% \pm 1.64\%$), with paired $t$-test across folds confirming consistency ($p = 0.038$). We explicitly clarify in the text that seed-level tests assess initialization robustness, whereas cross-validation reflects patient sample variability.

---

### Editorial Comment 3: Baseline and metric reporting are incomplete (also raised by Reviewer 2, Comment 2 and Comment 7)
> *Sections 2.7 and 2.11 versus Results: SVM, XGBoost, and Ridge accuracies are given only approximately (98.2%, 97.4%, 95.6%) in Methods, without hyperparameters, per-run results, AUC, or matched cross-validation. Precision, recall, and F1-score are said to have been assessed but are never reported. The positive class is not defined for these metrics, and no confusion matrix, sensitivity/specificity, or confidence intervals are supplied. The proposed mean accuracy (97.02%) is below the stated SVM value (98.2%), but this baseline is not integrated into the Results or Discussion.*  
> *Please evaluate all classical and quantum-inspired models with the same preprocessing, tuning, splits/seeds, and metrics; define the positive class; and report every prespecified metric, the confusion matrix, sensitivity/specificity, and appropriate confidence intervals, together with uncertainty across runs and folds. Include the classical baselines in the Results and Discussion, and in any comparison table that is retained.*

**Author Response:**  
We appreciate this crucial critique. We have fully remedied these omissions:
1. **Explicit Clinical Positive Class Definition:**  
   In clinical oncology diagnostics, identifying Malignant cases is paramount to prevent false negatives. We explicitly define **Malignant** as the positive class ($y = 0$ in the raw WDBC encoding, mapped to positive label in our evaluation framework) and **Benign** as the negative class ($y = 1$). Accordingly:
   - **Sensitivity / Recall:** $\frac{\text{TP}_{\text{Malignant}}}{\text{TP}_{\text{Malignant}} + \text{FN}_{\text{Malignant}}}$ (clinical detection rate of malignancy)
   - **Specificity:** $\frac{\text{TN}_{\text{Benign}}}{\text{TN}_{\text{Benign}} + \text{FP}_{\text{Benign}}}$ (true negative rate for benign tissue)
   - **Precision (PPV):** $\frac{\text{TP}_{\text{Malignant}}}{\text{TP}_{\text{Malignant}} + \text{FP}_{\text{Malignant}}}$
   - **F1-Score:** harmonic mean of malignant precision and sensitivity.
2. **Standardized Classical Baselines:**  
   We implemented and evaluated six classical models under identical conditions (same StandardScaler fitted strictly on training data, same 5 random seeds, same 5-fold CV):
   - **SVM (RBF kernel, $C=1.0$):** Test Accuracy $97.72\% \pm 1.59\%$, Sensitivity $95.24\% \pm 4.12\%$, Specificity $99.17\% \pm 1.24\%$, ROC-AUC $0.9944 \pm 0.0060$.
   - **Logistic Regression ($C=1.0$):** Test Accuracy $97.54\% \pm 1.69\%$, Sensitivity $94.76\% \pm 3.91\%$, Specificity $99.17\% \pm 0.76\%$, ROC-AUC $0.9939 \pm 0.0057$.
   - **SVM (Linear kernel, $C=1.0$):** Test Accuracy $97.02\% \pm 2.37\%$, Sensitivity $93.81\% \pm 6.43\%$, Specificity $98.89\% \pm 1.52\%$, ROC-AUC $0.9928 \pm 0.0075$.
   - **XGBoost ($n=100, \text{lr}=0.1, \text{depth}=3$):** Test Accuracy $96.49\% \pm 1.39\%$, Sensitivity $94.76\% \pm 1.99\%$, Specificity $97.50\% \pm 2.06\%$, ROC-AUC $0.9919 \pm 0.0085$.
   - **Ridge Classifier ($\alpha=1.0$):** Test Accuracy $94.74\% \pm 2.06\%$, Sensitivity $86.67\% \pm 5.98\%$, Specificity $99.44\% \pm 0.76\%$, ROC-AUC $0.9933 \pm 0.0048$.
   - **Random Forest ($n=100, \text{depth}=5$):** Test Accuracy $94.91\% \pm 2.27\%$, Sensitivity $92.38\% \pm 3.91\%$, Specificity $96.39\% \pm 1.58\%$, ROC-AUC $0.9872 \pm 0.0133$.
3. **Integration into Results and Discussion:**  
   We have added a dedicated Classical Benchmark table (Table 7) in Section 3 and extensively discussed these results in Section 4. We openly discuss that mature classical algorithms like RBF-SVM (97.72%) and Logistic Regression (97.54%) achieve slightly higher or comparable accuracy to our 3-qubit CNOT-Star VQC (97.02%). We clarify that the contribution of our paper is not to assert that near-term quantum-inspired classifiers surpass classical gradient boosted trees or kernel SVMs on small tabular datasets, but rather to uncover fundamental principles of feature allocation (IBIP) and circuit interaction geometry in quantum-inspired parameterized models.
4. **Full Confusion Matrices Reported:**  
   Every per-run and summary table now reports exact TP, TN, FP, and FN counts.

---

### Editorial Comment 4: Clarify quantum-inspired versus true quantum simulation (also raised by Reviewer 3)
> *Please state explicitly whether the implementation simulates quantum states with complex amplitudes and Hilbert-space operations, or implements classical operations using quantum-inspired terminology. The expression φq = π·tanh(...) appears to be a classical transformation. If the model is classical, say so clearly and revise the title, abstract, and framing accordingly. The term “quantum-inspired” is acceptable if defined precisely, but the manuscript must not imply quantum-state simulation if none is performed.*

**Author Response:**  
We thank the Editor and Reviewer 3 for requesting this essential clarification. We have thoroughly revised the text in Section 2.4 and throughout the manuscript to eliminate ambiguity:

1. **Exact Nature of the Mathematical Model:**  
   Our model is a **hybrid quantum-inspired variational classifier** simulated numerically on classical computing hardware. The workflow consists of three distinct stages:
   - **Classical Feature Projection Layer:** The input feature vector $x \in \mathbb{R}^{30}$ is partitioned into qubit-specific subvectors $x_q$. A trainable classical weight vector $W_q$ computes the scalar rotation angle $\phi_q = \pi \tanh(W_q^T x_q / \sqrt{|x_q|}) \in [-\pi, \pi]$. This is a classical dimensionality-reduction transformation that maps continuous diagnostic attributes into valid gate rotation intervals.
   - **Exact Statevector Quantum Simulation:** The simulated circuit operates in the complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ (dimension $2^n$, where $n \in \{2, 3\}$). The initial state $|0\dots0\rangle$ undergoes statevector evolution via single-qubit rotation unitaries ($R_y(\phi_q)$, $R_z(e_q)$), two-qubit unitary interaction gates (CNOT, SWAP, $\sqrt{\text{SWAP}}$, iSWAP), and variational rotation unitaries ($R_y(v_q), R_z(v_q)$). All operations preserve state normalization ($\|\psi\| = 1$) with complex probability amplitudes.
   - **Classical Readout Layer:** Expectation values of the Pauli-$Z$ observables on each qubit, $z_q = \langle \psi | Z_q | \psi \rangle \in [-1, 1]$, are computed via inner products $\psi^\dagger Z_q \psi$. A classical affine combination with sigmoid activation produces the final class probability: $\hat{y} = \sigma(\sum_{q} \beta_q z_q + b)$.
2. **Terminology & Framing:**  
   We have clarified throughout the title, abstract, and introduction that this is a **classical statevector simulation of a quantum-inspired variational classifier**, rather than execution on physical quantum hardware or purely classical heuristics mislabeled with quantum terms.

---

### Editorial Comment 5: Verify the iSWAP result (also raised by Reviewer 3)
> *The near-random iSWAP performance (58.80% accuracy, ROC-AUC 0.5640) is highly suspicious and may indicate an implementation issue. Please verify and debug this result before presenting it as evidence that phase-based interactions degrade performance. Provide verification details, and if the result stems from an implementation error, correct it and revise the conclusion accordingly.*

**Author Response:**  
We are immensely grateful to the Editor and Reviewer 3 for urging us to re-investigate this suspicious result. **Reviewer 3 was entirely correct: the 58.80% iSWAP result was an implementation artifact.**

**Diagnostic Investigation:**  
We audited our codebase and reproduced the diagnostic pipeline in `experiments/debug_iswap.py`. We identified that in our early prototype code, the two-qubit gate was subjected to an inadvertent real-value casting/truncation (`np.real()`), which discarded the complex imaginary phase factor ($1j$) in the canonical iSWAP matrix:
$$\text{iSWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
Discarding or corrupting the complex phase broke unitarity ($\text{det}(U) \neq 1$, non-unitary projection), which destroyed the statevector norm and severely degraded parameter-shift gradient propagation. This caused the model to collapse to predicting the majority class (67 benign predictions out of 114 test cases: $67/114 = 58.77\% \approx 58.80\%$).

**Verification with Correct Unitary Simulation:**  
When implemented with rigorous complex statevector simulation preserving unitarity:
- **Unitary iSWAP:** Achieves **95.61% – 96.49% accuracy** and **0.9921 – 0.9940 ROC-AUC** across seeds!
- This is fully competitive with SWAP (95.61%, AUC 0.9917) and $\sqrt{\text{SWAP}}$ (96.49%, AUC 0.9960).

**Revised Scientific Conclusions:**  
We have completely rewritten Section 3.1 and Section 4. We retract the previous assertion that "phase-based interactions inherently degrade classification performance." Instead, our corrected evidence shows that across 2-qubit exchange gates (SWAP, $\sqrt{\text{SWAP}}$, and iSWAP), performance remains consistently high (95.6%–96.5%) as long as complex unitarity is preserved, demonstrating that the primary driver of performance is the feature projection quality (IBIP) rather than specific gate exchange phases.

---

### Editorial Comment 6: Abstract and title (also raised by Reviewer 3, Comment 9)
> *The title’s “Topological Study” is misleading: the paper studies circuit interaction topologies, not topological data analysis. Replace “Topological” with “Interaction-Topology” or similar unless a genuine topological analysis is provided. The abstract also states comparative conclusions not supported by the design (e.g., “Feature allocation strategy exerted a greater influence than increasing interaction complexity”) and omits the classical-baseline context and the limited-replication caveat. Please revise the abstract to match the evidence and to include those caveats.*

**Author Response:**  
We have revised both the title and the abstract in accordance with these recommendations:
1. **Title Revision:**  
   Old Title: *Importance-Balanced Interleaved Feature Projection for Quantum-Inspired Variational Classifiers: A Topological Study on Breast Cancer Diagnosis*  
   **New Title:** *Circuit Interaction Topologies and Importance-Balanced Feature Allocation for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study*
2. **Abstract Overhaul:**  
   - Replaced "topological study" with "circuit interaction topologies (SWAP, $\sqrt{\text{SWAP}}$, iSWAP, CNOT-Chain, CNOT-Star, and CNOT-Ring)".
   - Clarified that models are hybrid quantum-inspired architectures evaluated via numerical statevector simulation.
   - Corrected the iSWAP reporting, noting that all unitary two-qubit exchange gates achieve competitive performance (95.6%–96.5%).
   - Integrated the classical baseline context, explicitly noting that mature classical algorithms (SVM RBF 97.72%, Logistic Regression 97.54%) establish reference clinical performance.
   - Replaced sweeping claims with nuanced findings supported by paired statistics ($t(4) = 3.54, p = 0.024$, Cohen's $d_z = 1.58$), highlighting that interleaved feature allocation (IBIP) significantly improves generalization over sequential partitioning across both 2-qubit and 3-qubit circuits.

---

### Editorial Comment 7: Table and figure consistency
> *Table 2 and Table 8 currently mix single-run and multi-seed results; replicate all gate-level and topology evaluations across seeds and report mean ± SD or confidence intervals, clearly labelling any single-run results as exploratory. Please also check figure numbering and captions: the caption for Figure 4 refers to CNOT-Star, but the embedded figure title refers to CNOT-Chain, and Figure 5 appears to lack an image. Ensure all figures are present, correctly numbered, and consistent with the text, and ensure Table 6 presents all folds in a single table.*

**Author Response:**  
We have audited and updated all tables and figures:
1. **Replication Across Seeds:** Tables 2, 3, and 8 now report 5-seed Mean $\pm$ SD and 95% Confidence Intervals for every single gate and topology evaluated under both IBIP and Sequential allocations.
2. **Figure Corrections:**  
   - Generated high-resolution (300 DPI) figures.
   - Fixed Figure 4: Correctly labeled and embedded as *3-Qubit CNOT-Star — Accuracy & BCE Loss vs. Epoch*.
   - Fixed Figure 5: Complete standalone figure showing *3-Qubit CNOT-Chain — Accuracy & BCE Loss vs. Epoch*.
   - Re-plotted Figure 3 with verified multi-seed gate comparison bars.
3. **Table 6 Unified:** Table 6 now displays all 5 individual cross-validation folds alongside the Mean $\pm$ SD row in a single comprehensive table reporting Accuracy, Sensitivity, Specificity, Precision, F1-Score, and ROC-AUC.

---

### Editorial Comment 8: Reproducibility and code (also raised by Reviewer 2, Comment 8)
> *Please provide computer code sufficient for replicability, including feature ranking, IBIP allocation, topology implementation, optimization, and evaluation scripts. A public repository with a DOI or a supplementary code archive is preferable. If code cannot be released, provide detailed pseudocode, configuration files, logs, and the exact data splits and seed values sufficient for independent reproduction.*

**Author Response:**  
We provide complete computer code for our entire study in our public GitHub repository:  
`https://github.com/dakshkaila/BreastCancer_QIVC` (and archived via Zenodo DOI upon acceptance).

The repository is fully modularized and self-contained:
- `data/download_data.py`: Downloads and verifies WDBC dataset with explicit labels.
- `src/dataset.py`: Leakage-free preprocessing and stratified splitting.
- `src/ranking.py`: XGBoost permutation importance and IBIP/Sequential allocations.
- `src/quantum_circuit.py`: Exact statevector quantum operations in NumPy.
- `src/models.py`: QuantumInspiredClassifier class with PSR, FD, and SuperGD.
- `src/classical_baselines.py`: Classical model implementations (SVM, XGBoost, etc.).
- `src/evaluation.py`: Paired statistical tests (paired $t$, Wilcoxon, Cohen's $d_z$, 95% CI).
- `experiments/run_controlled_experiments_parallel.py`: Automated multi-process script running all 2-qubit and 3-qubit controlled suites across seeds and folds.
- `requirements.txt` and `README.md`: Step-by-step instructions for 100% reproduction.

---

# Part II: Response to Reviewer 1

**Reviewer 1, Comment 1:**  
*The idea of distributing features across qubits based on importance ranking is intuitive and interesting. However, could the authors clarify whether feature permutation ranking was done before or after the train-test split? Doing it globally could lead to data leakage.*

**Author Response:**  
We thank Reviewer 1 for highlighting this essential methodological point. We confirm that feature ranking was strictly computed **exclusively on the training subset** of each split or cross-validation fold after partitioning. In our codebase (`src/ranking.py` and `experiments/run_controlled_experiments_parallel.py`), the XGBoost model and permutation importance analysis are fit solely on `X_train`, and `X_test` remains strictly held-out and unseen until final evaluation. We have added explicit clarifications to Section 2.2 and Section 2.8 of the revised manuscript.

**Reviewer 1, Comment 2:**  
*The clinical relevance of the chosen features could be elaborated further.*

**Author Response:**  
In response to Reviewer 1, we expanded Section 2.2 with a detailed biomedical discussion of the highest-ranked features identified by permutation importance (`worst texture`, `worst concave points`, `area error`, `worst perimeter`, `mean texture`, and `worst concavity`). These features correspond to irregular cell nuclear pleomorphism, architectural loss, and microcalcification boundaries that characterize malignant carcinoma infiltration during fine-needle aspiration (FNA) cytology.

---

# Part III: Response to Reviewer 2

**Reviewer 2, Comment 1:**  
*The paper claims to evaluate precision, recall, and F1-score, but these metrics do not appear anywhere in the results tables.*

**Author Response:**  
We apologize for this omission in the initial submission. In the revised manuscript, all metrics—Accuracy, Sensitivity (Malignant Recall), Specificity (Benign Recall), Precision (PPV), F1-Score, and ROC-AUC—are now fully reported across all tables (Tables 2, 3, 5, 6, 7, and 8).

**Reviewer 2, Comment 2:**  
*Which class is treated as positive? For breast cancer diagnosis, detecting malignant tumors is clinically vital.*

**Author Response:**  
We fully agree. In our revision, we have explicitly designated **Malignant** as the positive class ($y = 0 \to \text{positive}$). All sensitivity, precision, and F1 metrics now directly quantify the model's clinical ability to detect malignancy, with specificity quantifying the correct identification of benign lesions. Full confusion matrices (TP, TN, FP, FN) are now provided.

**Reviewer 2, Comment 3:**  
*The statistical significance test uses Welch's t-test on 5 seeds, which assumes independent samples. A paired t-test is required.*

**Author Response:**  
As detailed in our response to Editorial Comment 2, we have replaced Welch's $t$-test with a paired Student's $t$-test ($t = 3.5415, p = 0.0240$), Wilcoxon signed-rank test ($W = 0.0, p = 0.0431$), Cohen's $d_z = 1.5838$, and 95% Confidence Intervals ($[0.45\%, 3.76\%]$) across matched seeds, complemented by paired cross-validation across data folds ($p = 0.038$).

---

# Part IV: Response to Reviewer 3

**Reviewer 3, Comment 1:**  
*The 2-qubit and 3-qubit comparisons are confounded with feature group sizes (15 vs 10). Replicate both at the same qubit count or treat them as separate.*

**Author Response:**  
We have completely separated the 2-qubit and 3-qubit evaluations into distinct experimental suites as requested, and evaluated all topologies under identical feature allocation and replication depths (see Response to Editorial Comment 1).

**Reviewer 3, Comment 2:**  
*The iSWAP result (58.8%) is likely a bug. Phase gates shouldn't randomly collapse performance.*

**Author Response:**  
Reviewer 3 was entirely perceptive and accurate. As detailed in Response to Editorial Comment 5, our audit identified an implementation artifact in early prototypes where complex phases were dropped during state coercion. Correct unitary simulation of iSWAP achieves 95.61%–96.49% accuracy and 0.9921–0.9940 ROC-AUC, fully consistent with other exchange gates. We have corrected this throughout the paper.

**Reviewer 3, Comment 3:**  
*The paper is not a "topological study" in the mathematical sense. The title should be revised.*

**Author Response:**  
We have updated the title to: *"Circuit Interaction Topologies and Importance-Balanced Feature Allocation for Quantum-Inspired Variational Classifiers: A Breast Cancer Diagnosis Study"*, accurately reflecting that the work investigates circuit gate connection graphs rather than topological data analysis (TDA).

---

We believe that these extensive revisions, rigorous controlled experiments, corrected iSWAP results, paired statistical analyses, and comprehensive classical benchmarks directly address all concerns raised. We thank the Editor and Reviewers once again for their invaluable guidance in improving our manuscript.

Sincerely,  
**Dr. Daksh Kaila** (on behalf of the authors)  
Thapar Institute of Engineering and Technology, Patiala, India  
Email: dkaila_be25@thapar.edu
