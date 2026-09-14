"""Prediction utility for trained model pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_with_model(model, X: pd.DataFrame) -> np.ndarray:
    """Return class predictions from a fitted sklearn-style pipeline."""
    return model.predict(X)


def predict_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return probability scores for positive class when available."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    return model.decision_function(X)
