"""Time-aware train/validation/test split utilities.

The project intentionally keeps this lightweight and portfolio-friendly.
"""

from __future__ import annotations

import pandas as pd


def time_aware_split(df: pd.DataFrame, date_col: str = "published_date", train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train/validation/test splits.

    Args:
        df: Input dataset requiring a date column.
        date_col: Temporal column used for ordering.
        train_frac: Fraction of the ordered data for training.
        val_frac: Fraction of the ordered data for validation.

    Returns:
        A tuple of train, validation, and test dataframes.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column {date_col} is required for temporal sampling.")

    ordered = df.sort_values(date_col).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = ordered.iloc[:train_end]
    val = ordered.iloc[train_end:val_end]
    test = ordered.iloc[val_end:]
    return train, val, test
