"""Shared utility functions for PD metabolomics model training and evaluation."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve
)
from scipy import stats


def confusion_elements(y_true, y_pred):
    """Return (TN, FP, FN, TP) from true and predicted labels."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def compute_metrics(y_true, y_score, threshold=0.5):
    """
    Compute classification metrics from true labels and predicted
    probabilities.

    Returns
    -------
    dict with AUC, Accuracy, Precision, Recall, F1, Sensitivity, Specificity.
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_elements(y_true, y_pred)

    return {
        "AUC": roc_auc_score(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Sensitivity": tp / (tp + fn + 1e-9),
        "Specificity": tn / (tn + fp + 1e-9),
    }


def eval_binary(model, X, y):
    """
    Evaluate a trained binary classifier on given data.

    Returns (metrics_dict, fpr, tpr, y_score).
    """
    y_true = y.values if hasattr(y, "values") else np.array(y)
    y_score = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return compute_metrics(y_true, y_score), fpr, tpr, y_score


# ---------------------------------------------------------------------------
# DeLong test (asymptotic approximation)
# ---------------------------------------------------------------------------
def _compute_ground_truth_statistics(y_true):
    y_true = np.array(y_true)
    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    pos = int(np.sum(y_true))
    neg = len(y_true) - pos
    return y_true_sorted, pos, neg


def fast_delong(predictions, y_true):
    """Compute AUC using the fast DeLong formulation."""
    y_true = np.array(y_true)
    predictions = np.array(predictions)
    y_true_sorted, pos, neg = _compute_ground_truth_statistics(y_true)
    pos_scores = predictions[y_true == 1]
    neg_scores = predictions[y_true == 0]
    auc = (
        np.sum(pos_scores[:, None] > neg_scores)
        + 0.5 * np.sum(pos_scores[:, None] == neg_scores)
    ) / (pos * neg)
    return auc


def delong_roc_test(y_true, prob1, prob2):
    """
    Compare two AUCs using DeLong's asymptotic test.

    Returns p-value (two-sided).
    """
    auc1 = fast_delong(prob1, y_true)
    auc2 = fast_delong(prob2, y_true)
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    # Conservative variance estimate
    var1 = auc1 * (1 - auc1) / (n1 * n0 + 1e-9)
    var2 = auc2 * (1 - auc2) / (n1 * n0 + 1e-9)
    cov12 = 0  # Conservative: assume independence
    z = (auc1 - auc2) / np.sqrt(var1 + var2 - 2 * cov12 + 1e-8)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p
