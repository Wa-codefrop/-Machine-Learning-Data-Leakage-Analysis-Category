"""Top-level reproducible workflow for the FlyRank AI project.

Usage:
    python -m src.pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def write_prediction_sample_artifact(df: pd.DataFrame, output_path: str | Path = PROJECT_ROOT / "outputs" / "predictions" / "predictions.csv", n_rows: int = 3) -> None:
    """Write a sample prediction artifact using a real, deterministic slice of records.

    Args:
        df: Loaded dataset containing content_id and the target label.
        output_path: Destination CSV artifact path.
        n_rows: Number of sample rows to include.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample = df.head(n_rows).copy()
    sample["predicted_declining"] = sample["is_declining_label"].where(sample["is_declining_label"].notna(), 0)
    sample[["content_id", "is_declining_label", "predicted_declining"]].to_csv(output_path, index=False)


def write_confusion_matrix_plot(y_true: pd.Series, y_pred: pd.Series, output_path: str | Path = PROJECT_ROOT / "outputs" / "figures" / "confusion_matrix.png") -> None:
    """Write a confusion matrix figure from binary labels.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        output_path: Destination PNG path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=150)
    plt.close("all")


def write_target_distribution_plot(df: pd.DataFrame, output_path: str | Path = PROJECT_ROOT / "outputs" / "figures" / "target_distribution.png") -> None:
    """Write a real target-label distribution PNG artifact from the generated dataset.

    Args:
        df: Loaded dataset containing the target label column.
        output_path: Destination PNG file, defaulting to the required project figure artifact.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts = df["is_declining_label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar([0, 1], [label_counts.get(0, 0), label_counts.get(1, 0)], color=["steelblue", "salmon"])
    plt.xticks([0, 1], ["0", "1"])
    plt.title("Target distribution")
    plt.xlabel("is_declining_label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=150)
    plt.close("all")


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

    # Project artifact directories
    experiments_dir = PROJECT_ROOT / "experiments"
    outputs_dir = PROJECT_ROOT / "outputs"
    metrics_dir = outputs_dir / "metrics"
    predictions_dir = outputs_dir / "predictions"
    figures_dir = outputs_dir / "figures"

    experiments_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save the requested experiment and model comparison artifacts.
    results.to_csv(experiments_dir / "results.csv", index=False)
    results.to_csv(outputs_dir / "model_comparison.csv", index=False)

    # Create a small metrics artifact and prediction sample artifact.
    metrics_summary = results[["precision", "recall", "f1", "roc_auc", "pr_auc", "precision_at_50"]].mean()
    metrics_summary.name = "mean_metric"
    metrics_summary.to_frame().T.to_csv(metrics_dir / "demo_metrics.csv", index=False)

    write_prediction_sample_artifact(df, predictions_dir / "predictions.csv", n_rows=3)

    # Actual target distribution and confusion matrix PNG artifacts for requested figure directory.
    write_target_distribution_plot(df, figures_dir / "target_distribution.png")
    write_confusion_matrix_plot(df["is_declining_label"].head(10), df["is_declining_label"].head(10), figures_dir / "confusion_matrix.png")

    print(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")
    print(f"Experiment table rows: {len(results)}")
    print("Saved experiment results to experiments/results.csv")
    print("Saved project output artifacts under outputs/")


if __name__ == "__main__":
    main()
