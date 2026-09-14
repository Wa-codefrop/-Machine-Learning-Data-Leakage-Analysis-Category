"""Small data loading helper for the FlyRank AI project."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/seo_content_performance.csv")


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the generated synthetic SEO performance dataset.

    Args:
        path: Path to the CSV file.

    Returns:
        The dataset as a pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df
