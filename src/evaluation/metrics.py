"""Evaluation helper functions for the ML experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """Return a small dictionary of common binary classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_score: Optional numeric scores for ROC and PR metrics.

    Returns:
        Dictionary with key metrics.
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def precision_at_50(y_true: pd.Series | np.ndarray, y_score: np.ndarray) -> float:
    """Return the proportion of positives among the top 50 ranked rows.

    If the dataset has fewer than 50 rows, the function reports Precision@k
    over the available sample size instead of returning NaN.
    """
    y_values = np.asarray(y_true)
    scores = np.asarray(y_score)
    k = min(50, len(y_values))

    if k == 0:
        return 0.0

    order = np.argsort(-scores)
    top_k_true = y_values[order[:k]]
    return float(np.mean(top_k_true == 1))
