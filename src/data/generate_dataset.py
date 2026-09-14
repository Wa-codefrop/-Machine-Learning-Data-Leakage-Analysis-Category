"""Generate a reproducible synthetic SEO content performance dataset.

The dataset is purpose-built for an internship-style data leakage analysis
project. It contains a feature window and a future/label window where the
future variables define the target label. This is the intended leakage
scenario that later notebooks will document.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import RANDOM_STATE, RAW_DATA_PATH


CATEGORICAL_TYPES = ["blog", "landing_page", "product_page", "guide", "news"]
CATEGORIES = ["technology", "finance", "health", "travel", "ecommerce", "education"]
AUTHOR_TYPES = ["brand", "freelance", "in_house", "agency"]


def generate_dataset(n_records: int = 12000, output_path: str | None = None) -> pd.DataFrame:
    """Generate the synthetic dataset and save it to a CSV file.

    Args:
        n_records: Number of synthetic records to create.
        output_path: Optional output path. Defaults to the project data/raw file.

    Returns:
        DataFrame containing the full generated dataset.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Content dimensions
    n = int(n_records)
    content_id = [f"CF-{i:06d}" for i in range(n)]
    content_type = rng.choice(CATEGORICAL_TYPES, size=n)
    category = rng.choice(CATEGORIES, size=n)
    word_count = np.clip(rng.normal(850, 300, n), 120, 2400).astype(int)
    content_age_days = np.clip(rng.normal(180, 160, n), 7, 1800).astype(int)
    author_type = rng.choice(AUTHOR_TYPES, size=n)
    has_schema = rng.binomial(1, 0.58, n)
    internal_link_count = np.clip(rng.poisson(20, n), 0, 120)
    external_link_count = np.clip(rng.poisson(5, n), 0, 50)

    # SEO base metrics
    domain_authority = np.clip(rng.normal(43, 15, n), 5, 98)
    keyword_count = np.clip(rng.poisson(8, n) + rng.normal(0, 2, n), 1, 40).astype(int)
    ranking_keywords = np.clip(rng.poisson(4, n) + keyword_count // 2, 1, 60)
    backlinks = np.clip(rng.poisson(30, n) + domain_authority * 0.8, 0, 300).astype(int)

    organic_impressions = np.clip(rng.lognormal(mean=6.0, sigma=1.2, size=n), 50, 200000).astype(int)
    organic_clicks = np.clip(organic_impressions * rng.uniform(0.01, 0.25, n), 0, 30000).astype(int)
    ctr = np.clip(organic_clicks / np.maximum(organic_impressions, 1), 0.0001, 0.50)
    average_position = np.clip(rng.normal(18, 8, n) + (1 / np.maximum(ctr, 0.001)) * 2, 1, 80)

    sessions = np.clip(organic_clicks * rng.uniform(2.0, 8.0, n), 0, 50000).astype(int)
    bounce_rate = np.clip(rng.normal(0.43, 0.12, n), 0.08, 0.82)
    avg_session_duration = np.clip(rng.normal(130, 60, n), 25, 500)
    conversions = np.clip(organic_clicks * rng.uniform(0.005, 0.08, n), 0, 750).astype(int)

    # Historical windows
    clicks_7d = np.clip(np.round(organic_clicks * rng.uniform(0.50, 1.10, n)), 0, None).astype(int)
    clicks_30d = np.clip(np.round(clicks_7d * rng.uniform(3.0, 5.5, n)), 0, None).astype(int)
    impressions_7d = np.clip(np.round(organic_impressions * rng.uniform(0.50, 1.10, n)), 0, None).astype(int)
    impressions_30d = np.clip(np.round(impressions_7d * rng.uniform(3.0, 5.5, n)), 0, None).astype(int)
    position_7d = np.clip(np.round(average_position + rng.normal(0, 4, n)), 1, 80)
    position_30d = np.clip(np.round(position_7d + rng.normal(0, 5, n)), 1, 80)

    # Future variables that will be used as leakage columns
    future_clicks = np.clip(np.round(clicks_30d * rng.uniform(0.70, 1.30, n)), 0, None).astype(int)
    future_position = np.clip(np.round(position_30d + rng.normal(-3, 5, n)), 1, 80)
    trend_direction = np.where(rng.random(n) < 0.50, "down", "up")

    # trend_pct should carry a down/up signal that is realistic and consistent
    # with the target generation rule. This remains synthetic and stable.
    trend_pct = np.zeros(n)
    for i in range(n):
        if trend_direction[i] == "down":
            trend_pct[i] = rng.normal(-0.14, 0.08)
        else:
            trend_pct[i] = rng.normal(0.11, 0.08)

    trend_pct = np.clip(trend_pct, -0.70, 0.70)

    # Encode decline label by future performance derivative signal.
    # The target is intentionally derived from future information to create a leakage design.
    # Add some noise, so labels are not perfectly predictable.
    decline_signal = (
        (future_clicks < clicks_30d * 1.05)
        | (future_position > position_30d + 5)
        | (trend_direction == "down")
        | (trend_pct < -0.03)
    )
    decline_noise = rng.binomial(1, 0.12, n).astype(bool)
    is_declining_label = np.where(decline_signal | decline_noise, 1, 0).astype(int)

    # Distort some relationship to avoid perfect derivation from future vars
    # but keep enough signal for leakage demonstration.
    is_declining_label = np.where(
        rng.random(n) < 0.03,
        1 - is_declining_label,
        is_declining_label,
    )

    df = pd.DataFrame(
        {
            "content_id": content_id,
            "content_type": content_type,
            "category": category,
            "word_count": word_count,
            "content_age_days": content_age_days,
            "author_type": author_type,
            "has_schema": has_schema.astype(bool),
            "internal_link_count": internal_link_count,
            "external_link_count": external_link_count,
            "organic_clicks": organic_clicks,
            "organic_impressions": organic_impressions,
            "ctr": ctr,
            "average_position": average_position,
            "keyword_count": keyword_count,
            "ranking_keywords": ranking_keywords,
            "backlinks": backlinks,
            "domain_authority": domain_authority,
            "sessions": sessions,
            "bounce_rate": bounce_rate,
            "avg_session_duration": avg_session_duration,
            "conversions": conversions,
            "clicks_7d": clicks_7d,
            "clicks_30d": clicks_30d,
            "impressions_7d": impressions_7d,
            "impressions_30d": impressions_30d,
            "position_7d": position_7d,
            "position_30d": position_30d,
            "trend_direction": trend_direction,
            "trend_pct": trend_pct,
            "future_clicks": future_clicks,
            "future_position": future_position,
            "is_declining_label": is_declining_label,
        }
    )

    # Make data path and create raw directory
    output = Path(output_path) if output_path else RAW_DATA_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    return df
