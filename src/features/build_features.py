"""Feature-building helpers for the FlyRank AI project."""

from __future__ import annotations

import pandas as pd


FEATURE_WINDOW_COLUMNS = [
    "content_type",
    "category",
    "word_count",
    "content_age_days",
    "author_type",
    "has_schema",
    "internal_link_count",
    "external_link_count",
    "organic_clicks",
    "organic_impressions",
    "ctr",
    "average_position",
    "keyword_count",
    "ranking_keywords",
    "backlinks",
    "domain_authority",
    "sessions",
    "bounce_rate",
    "avg_session_duration",
    "conversions",
    "clicks_7d",
    "clicks_30d",
    "impressions_7d",
    "impressions_30d",
    "position_7d",
    "position_30d",
]


def build_feature_matrix(df: pd.DataFrame, leaked: bool = False) -> pd.DataFrame:
    """Return the candidate feature set for a given experiment.

    Args:
        df: Input dataframe.
        leaked: When True, include the known future-derived leakage columns.

    Returns:
        A DataFrame containing the feature columns and the target.
    """
    features = list(FEATURE_WINDOW_COLUMNS)
    if leaked:
        features.extend(["trend_direction", "trend_pct", "future_clicks", "future_position"])

    return df[features + ["is_declining_label"]]
