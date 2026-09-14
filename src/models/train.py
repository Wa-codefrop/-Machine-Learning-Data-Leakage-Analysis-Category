"""Training helper functions for experiments."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a standard preprocessing block for numeric and categorical features."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def make_pipelines(X_train: pd.DataFrame) -> dict:
    """Create logistic regression and random forest pipelines.

    Args:
        X_train: Training feature matrix.

    Returns:
        A simple dictionary of sklearn-compatible pipelines.
    """
    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=50, random_state=42, min_samples_leaf=2)),
            ]
        ),
    }

    return models
