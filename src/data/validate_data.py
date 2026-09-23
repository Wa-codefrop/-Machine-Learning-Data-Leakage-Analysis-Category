"""Validation helpers for the real UrbanScape SEO dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact dataset-quality summary for the real SEO dataset."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().sort_values(ascending=False).to_dict(),
        "date_range": {
            "min": pd.to_datetime(df['Date'], errors='coerce').min().isoformat() if 'Date' in df.columns else None,
            "max": pd.to_datetime(df['Date'], errors='coerce').max().isoformat() if 'Date' in df.columns else None,
        },
        "column_types": df.dtypes.astype(str).to_dict(),
    }
