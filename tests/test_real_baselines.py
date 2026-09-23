import pandas as pd

from src.data.load_real_data import load_real_data
from src.models.real_baselines import prepare_model_frame, train_real_baselines


def test_prepare_model_frame_creates_time_aware_target():
    df = load_real_data()
    model_df = prepare_model_frame(df)

    assert "published_date" in model_df.columns
    assert "high_performance_flag" in model_df.columns
    assert model_df["high_performance_flag"].isin([0, 1]).all()
    assert model_df["published_date"].is_monotonic_increasing


def test_train_real_baselines_returns_metrics():
    df = load_real_data()
    model_df = prepare_model_frame(df)
    metrics = train_real_baselines(model_df, target_col="high_performance_flag")

    assert set(metrics).issuperset({"logistic_regression", "random_forest"})
    assert metrics["logistic_regression"]["f1"] >= 0
    assert metrics["logistic_regression"]["roc_auc"] >= 0
    assert metrics["random_forest"]["f1"] >= 0
    assert metrics["random_forest"]["roc_auc"] >= 0
