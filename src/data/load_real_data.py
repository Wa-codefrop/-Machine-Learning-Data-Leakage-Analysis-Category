"""Utilities for loading the real UrbanScape SEO dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "UrbanScape_Apparel_SEO_Performance_Final_Dataset.xlsx"


def load_real_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the real UrbanScape SEO dataset from the raw data directory.

    This helper is intentionally separate from the legacy synthetic project loader
    so the project can pivot to the real dataset without overwriting the original raw files.
    """
    df = pd.read_excel(path)
    return df
