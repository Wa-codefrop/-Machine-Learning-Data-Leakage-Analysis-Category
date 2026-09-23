"""Compare a leakage-safe real-data baseline with a leakage-prone variant."""

from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create a valid time-aware frame using a high-performance target."""
    out = df.copy()
    out["published_date"] = pd.to_datetime(out["Date"])
    threshold = out["Organic_Traffic"].quantile(0.75)
    out["high_performance_flag"] = (out["Organic_Traffic"] >= threshold).astype(int)
    return out.sort_values("published_date").reset_index(drop=True)


def _baseline_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def _train_and_score(features: pd.DataFrame, label_col: str, leakage: bool) -> dict[str, float]:
    ordered = features.sort_values("published_date").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.7)
    test_end = int(n * 0.85)

    train = ordered.iloc[:train_end].copy()
    test = ordered.iloc[test_end:].copy()

    model_cols = [col for col in ordered.columns if col not in {"Date", "Month", "Year", "Quarter", "Time Of Day", "published_date", label_col}]

    if leakage:
        suspicious = [
            "Clicks",
            "Impressions",
            "CTR (%)",
            "Average_Position",
            "Organic_Traffic",
            "Conversion_Rate (%)",
            "Goal_Completions",
            "Organic_Revenue ($)",
        ]
        model_cols = [c for c in model_cols if c not in suspicious]

    X_train = train[model_cols]
    y_train = train[label_col]
    X_test = test[model_cols]
    y_test = test[label_col]

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample"),
    }

    metrics = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", _baseline_preprocessor(X_train)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        score = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        metrics[name] = {
            "model": name,
            "feature_set": "leaky" if leakage else "safe",
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, score)),
        }

    return metrics


def compare_safe_vs_leaky_models(df: pd.DataFrame) -> pd.DataFrame:
    """Return a comparison table between safe and intentionally leaky baselines."""
    model_df = prepare_feature_matrix(df)
    rows = []
    for leakage in [False, True]:
        metrics = _train_and_score(model_df, label_col="high_performance_flag", leakage=leakage)
        for model_name, values in metrics.items():
            rows.append({
                "feature_set": values["feature_set"],
                "model": values["model"],
                "f1": values["f1"],
                "roc_auc": values["roc_auc"],
            })
    return pd.DataFrame(rows)
