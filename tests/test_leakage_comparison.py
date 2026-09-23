import pandas as pd

from src.data.load_real_data import load_real_data
from src.models.compare_real_baselines import compare_safe_vs_leaky_models, prepare_feature_matrix


def test_prepare_feature_matrix_uses_time_aware_target():
    df = load_real_data()
    model_df = prepare_feature_matrix(df)

    assert "published_date" in model_df.columns
    assert "high_performance_flag" in model_df.columns
    assert model_df["high_performance_flag"].isin([0, 1]).all()
    assert model_df["published_date"].is_monotonic_increasing


def test_compare_safe_vs_leaky_models_returns_metrics_table():
    df = load_real_data()
    results = compare_safe_vs_leaky_models(df)

    assert not results.empty
    assert {"safe", "leaky"}.issubset(set(results["feature_set"]))
    assert {"model", "f1", "roc_auc"}.issubset(set(results.columns))
    assert results["f1"].between(0, 1).all()
    assert results["roc_auc"].between(0, 1).all()
