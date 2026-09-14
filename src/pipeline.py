"""Top-level reproducible workflow for the FlyRank AI project.

Usage:
    python -m src.pipeline
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.generate_dataset import generate_dataset
from src.data.load_data import load_data
from src.evaluation.evaluate import evaluate_model
from src.features.build_features import build_feature_matrix
from src.models.train import make_pipelines
from src.utils.config import PROJECT_ROOT, RAW_DATA_PATH


def generate_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create an experiment results table using the requested columns.

    Returns:
        A DataFrame with columns for experiment_name, model, feature_set,
        precision, recall, f1, roc_auc, pr_auc, precision_at_50, notes.
    """
    results = []

    # Deterministic placeholder demonstration: logistic regression and random forest.
    for model_name in ["logistic_regression", "random_forest"]:
        row = {
            "experiment_name": f"{model_name}_baseline",
            "model": model_name,
            "feature_set": "prediction_time_features_only",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "precision_at_50": 0.0,
            "notes": "Synthetic baseline placeholder workflow",
        }
        results.append(row)

    return pd.DataFrame(results)


def main() -> None:
    """Generate data and create an experiment table in a reproducible way.

    This project keeps the implemented pipeline intentionally lightweight and
    clear enough for an internship-level portfolio demonstration.
    """
    parser = argparse.ArgumentParser(description="FlyRank AI synthetic pipeline")
    parser.add_argument("--generate", action="store_true", help="Regenerate the dataset")
    args = parser.parse_args()

    if args.generate or not RAW_DATA_PATH.exists():
        generate_dataset()

    df = load_data(RAW_DATA_PATH)
    feature_matrix = build_feature_matrix(df, leaked=False)

    # Exercise the training module and save a simple experiment table.
    X = feature_matrix.drop(columns=["is_declining_label"])
    y = feature_matrix["is_declining_label"]
    pipelines = make_pipelines(X)
    for model_name, model in pipelines.items():
        model.fit(X, y)

    experiment_table = generate_results_table(feature_matrix)
    experiments_dir = PROJECT_ROOT / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    experiment_table.to_csv(experiments_dir / "results.csv", index=False)

    print(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print("Saved experiment results to experiments/results.csv")


if __name__ == "__main__":
    main()
