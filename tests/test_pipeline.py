"""Smoke tests for the FlyRank AI synthetic data workflow."""

from pathlib import Path

from src.data.generate_dataset import generate_dataset
from src.data.load_data import load_data
from src.features.build_features import build_feature_matrix
from src.evaluation.metrics import compute_classification_metrics, precision_at_50


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
