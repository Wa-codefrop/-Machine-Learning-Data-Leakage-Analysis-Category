"""Top-level reproducible workflow for the FlyRank AI project.

Usage:
    python -m src.pipeline
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.generate_dataset import generate_dataset
from src.data.load_data import load_data
from src.evaluation.evaluate import evaluate_model
from src.features.build_features import build_feature_matrix
from src.models.train import make_pipelines
from src.utils.config import PROJECT_ROOT, RAW_DATA_PATH
from src.utils.split import time_aware_split


def generate_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate an experiment table based on leaky and leakage-free experiments.

    Returns:
        DataFrame with keys requested by the project documentation.
    """
    experiments = []

    # Workflow-level leaky comparison
    leaky = build_feature_matrix(df, leaked=True)
    safe = build_feature_matrix(df, leaked=False)

    for feature_set_name, table in [("leaky", leaky), ("safe", safe)]:
        # Time-aware split from published_date. Keep a deterministic split.
        train_df, val_df, test_df = time_aware_split(table, date_col="published_date")

        # Use the same baseline model family for both experiments.
        X_train = train_df.drop(columns=["is_declining_label"])
        y_train = train_df["is_declining_label"]
        X_test = test_df.drop(columns=["is_declining_label"])
        y_test = test_df["is_declining_label"]

        # Fit a logistic regression and random forest model across the same split.
        model_dict = make_pipelines(X_train)
        for model_name in ["logistic_regression", "random_forest"]:
            model = model_dict[model_name]
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
            y_pred = (y_score >= 0.5).astype(int)
            metrics = evaluate_model(y_test, y_pred, y_score)

            row = {
                "experiment_name": f"{feature_set_name}_{model_name}",
                "model": model_name,
                "feature_set": feature_set_name,
                "precision": round(float(metrics["precision"]), 4),
                "recall": round(float(metrics["recall"]), 4),
                "f1": round(float(metrics["f1"]), 4),
                "roc_auc": round(float(metrics["roc_auc"]), 4),
                "pr_auc": round(float(metrics["pr_auc"]), 4),
                "precision_at_50": round(float(metrics["precision_at_50"]), 4),
                "notes": f"{feature_set_name.replace('_', ' ')} feature policy with time-aware split",
            }
            experiments.append(row)

    return pd.DataFrame(experiments)


def main() -> None:
    """Generate data, load it, build a feature table, and evaluate examples.

    This project is intentionally lightweight and designed to demonstrate a good
    structure rather than a full enterprise-grade production pipeline.
    """
    parser = argparse.ArgumentParser(description="FlyRank AI synthetic pipeline")
    parser.add_argument("--generate", action="store_true", help="Regenerate the dataset")
    args = parser.parse_args()

    if args.generate or not RAW_DATA_PATH.exists():
        generate_dataset()

    df = load_data(RAW_DATA_PATH)
    results = generate_results_table(df)

    experiments_dir = PROJECT_ROOT / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(experiments_dir / "results.csv", index=False)

    print(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")
    print(f"Experiment table rows: {len(results)}")
    print("Saved experiment results to experiments/results.csv")


if __name__ == "__main__":
    main()
