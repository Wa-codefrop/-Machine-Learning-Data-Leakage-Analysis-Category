"""Real-data baseline models for the UrbanScape SEO dataset."""

from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_model_frame(df: pd.DataFrame, target_col: str = "Organic_Traffic") -> pd.DataFrame:
    """Create a leakage-safe, prediction-time feature frame for SEO modeling.

    The dataset is not a panel with repeated entities over time, so the honest target
    is a binary high-performance flag derived from the true traffic distribution. We
    also create a chronological date field to enable a time-aware split.
    """
    out = df.copy()
    out["published_date"] = pd.to_datetime(out["Date"])
    threshold = out[target_col].quantile(0.75)
    out["high_performance_flag"] = (out[target_col] >= threshold).astype(int)
    return out.sort_values("published_date").reset_index(drop=True)


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def train_real_baselines(df: pd.DataFrame, target_col: str = "high_performance_flag") -> dict[str, dict[str, float]]:
    """Train leakage-safe baseline models using a time-aware split."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data.")

    ordered = df.sort_values("published_date").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()

    feature_cols = [
        col for col in ordered.columns
        if col not in {"Date", "Month", "Year", "Quarter", "Time Of Day", "high_performance_flag", "published_date", target_col}
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    base_models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"),
    }

    results: dict[str, dict[str, float]] = {}
    for name, model in base_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", _make_preprocessor(X_train)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
            "pr_auc": float(average_precision_score(y_test, y_score)),
            "val_f1": float(f1_score(y_val, pipeline.predict(X_val), zero_division=0)),
        }

    return results
