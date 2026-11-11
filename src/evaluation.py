"""
Evaluation and Statistical Analysis Module.
Implements:
1. Paired t-test, Wilcoxon signed-rank test, and Welch t-test.
2. Effect size calculation: Cohen's d_z for paired samples.
3. 95% Confidence Intervals for means and paired differences.
4. Comprehensive multi-metric clinical evaluation (Malignant positive class).
5. Publication-grade plotting for figures.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def compute_paired_statistics(scores_a, scores_b, name_a="IBIP", name_b="Sequential"):
    """
    Computes rigorous paired statistical tests between two matched configurations.
    
    Args:
        scores_a (list or np.ndarray): Scores of method A across matched seeds/folds.
        scores_b (list or np.ndarray): Scores of method B across matched seeds/folds.
        
    Returns:
        dict containing:
            mean_a, std_a, mean_b, std_b
            mean_diff, std_diff, ci_diff (95% CI)
            t_paired, p_paired
            wilcoxon_stat, p_wilcoxon
            cohen_d
            t_welch, p_welch (for comparison with legacy un-paired reporting)
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    n = len(a)
    
    diff = a - b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1) if n > 1 else 0.0
    sem_diff = std_diff / np.sqrt(n) if n > 1 else 0.0
    
    # 95% Confidence Interval for paired difference
    if n > 1 and std_diff > 1e-12:
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lower = mean_diff - t_crit * sem_diff
        ci_upper = mean_diff + t_crit * sem_diff
    else:
        ci_lower, ci_upper = mean_diff, mean_diff
        
    # Paired t-test
    if n > 1 and std_diff > 1e-12:
        t_paired, p_paired = stats.ttest_rel(a, b)
    else:
        t_paired, p_paired = 0.0, 1.0
        
    # Wilcoxon signed-rank test
    try:
        if np.all(diff == 0):
            w_stat, p_wilcoxon = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(a, b, zero_method="wilcox")
            w_stat, p_wilcoxon = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, p_wilcoxon = np.nan, np.nan
        
    # Cohen's d_z for paired samples: mean_diff / std_diff
    cohen_d = (mean_diff / std_diff) if std_diff > 1e-12 else 0.0
    
    # Independent Welch's t-test (reproducing original paper's calculation)
    t_welch, p_welch = stats.ttest_ind(a, b, equal_var=False)
    
    return {
        "n": n,
        "mean_a": float(np.mean(a)),
        "std_a": float(np.std(a, ddof=1)),
        "mean_b": float(np.mean(b)),
        "std_b": float(np.std(b, ddof=1)),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "ci_95_diff": (float(ci_lower), float(ci_upper)),
        "t_paired": float(t_paired),
        "p_paired": float(p_paired),
        "wilcoxon_stat": float(w_stat),
        "p_wilcoxon": float(p_wilcoxon),
        "cohen_d": float(cohen_d),
        "t_welch": float(t_welch),
        "p_welch": float(p_welch),
        "differences": diff.tolist()
    }

