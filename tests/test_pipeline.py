"""Smoke tests for the FlyRank AI synthetic data workflow."""

import pandas as pd

from src.data.generate_dataset import generate_dataset
from src.data.load_data import load_data
from src.features.build_features import build_feature_matrix
from src.evaluation.metrics import compute_classification_metrics, precision_at_50
from src.pipeline import (
    write_confusion_matrix_plot,
    write_prediction_sample_artifact,
    write_target_distribution_plot,
)


def test_generate_dataset_creates_expected_columns(tmp_path):
    out = tmp_path / "synthetic.csv"
    df = generate_dataset(n_records=50, output_path=out)

    assert out.exists()
    assert len(df) == 50
    assert "published_date" in df.columns
    assert "content_id" in df.columns
    assert "is_declining_label" in df.columns
    assert set(["trend_direction", "trend_pct", "future_clicks", "future_position"]).issubset(set(df.columns))


def test_feature_builder_removes_leakage_by_default(tmp_path):
    out = tmp_path / "synthetic.csv"
    generate_dataset(n_records=50, output_path=out)
    df = load_data(out)

    leak_free = build_feature_matrix(df, leaked=False)
    assert "trend_direction" not in leak_free.columns
    assert "trend_pct" not in leak_free.columns
    assert "future_clicks" not in leak_free.columns
    assert "future_position" not in leak_free.columns
    assert "is_declining_label" in leak_free.columns


def test_write_target_distribution_plot_creates_real_png(tmp_path):
    out = tmp_path / "synthetic.csv"
    df = generate_dataset(n_records=80, output_path=out)
    plot_path = tmp_path / "target_distribution.png"

    write_target_distribution_plot(df, plot_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 1000


def test_write_prediction_sample_artifact_creates_real_rows(tmp_path):
    out = tmp_path / "synthetic.csv"
    df = generate_dataset(n_records=80, output_path=out)
    artifact_path = tmp_path / "predictions.csv"

    write_prediction_sample_artifact(df, artifact_path, n_rows=3)

    assert artifact_path.exists()
    lines = artifact_path.read_text().strip().splitlines()
    assert len(lines) >= 4
    assert "content_id" in lines[0]
    assert "predicted_declining" in lines[0]


def test_write_confusion_matrix_plot_creates_real_png(tmp_path):
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 1, 1])
    plot_path = tmp_path / "confusion_matrix.png"

    write_confusion_matrix_plot(y_true, y_pred, plot_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 1000


def test_metrics_are_computable_on_small_labels():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_score = [0.1, 0.9, 0.2, 0.3]

    metrics = compute_classification_metrics(y_true, y_pred, y_score)
    assert metrics["precision"] >= 0
    assert metrics["recall"] >= 0
    assert metrics["f1"] >= 0
    assert metrics["roc_auc"] >= 0
    assert metrics["pr_auc"] >= 0
    assert precision_at_50(y_true, y_score) >= 0
