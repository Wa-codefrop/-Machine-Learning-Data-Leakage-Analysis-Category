"""Reusable evaluation module for a simple binary classification workflow."""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import compute_classification_metrics, precision_at_50


def evaluate_model(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None = None) -> dict:
    """Evaluate a binary classifier and produce a dictionary of metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_score: Probability scores for the positive class.

    Returns:
        Metrics dictionary.
    """
    if y_score is None:
        y_score = pd.Series(y_pred.astype(float))

    metrics = compute_classification_metrics(y_true, y_pred, y_score)
    metrics["precision_at_50"] = precision_at_50(y_true, y_score)
    return metrics
