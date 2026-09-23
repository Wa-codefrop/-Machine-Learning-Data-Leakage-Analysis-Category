"""Helpers for defining valid SEO prediction targets from the real dataset."""

from __future__ import annotations

import pandas as pd


def make_high_performance_flag(df: pd.DataFrame, metric: str = "Organic_Traffic", q: float = 0.75) -> pd.Series:
    """Create a transparent high-performance flag from a percentile threshold.

    This deliberately avoids arbitrary hand-crafted labels. The threshold is derived
    from the actual observed distribution in the dataset.
    """
    threshold = df[metric].quantile(q)
    return (df[metric] >= threshold).astype(int)


def make_traffic_bucket(df: pd.DataFrame, metric: str = "Organic_Traffic") -> pd.Series:
    """Create a simple traffic bucket label for exploratory classification tasks."""
    thresholds = [df[metric].quantile(0.25), df[metric].quantile(0.75)]
    labels = pd.cut(df[metric], bins=[-1, thresholds[0], thresholds[1], float('inf')], labels=['low', 'mid', 'high'], right=False)
    return labels.astype(str)
