"""
Unit tests for models benchmark module.
"""

import numpy as np
import polars as pl
import pytest
from src.models.benchmark import train_and_evaluate_model, DEFAULT_MODEL_PARAMS


def test_train_isolation_forest():
    np.random.seed(42)
    # 100 mẫu, 5 đặc trưng
    X_train = np.random.randn(100, 5)
    df_eval = pl.DataFrame({"idx": range(100)})

    res = train_and_evaluate_model(
        model_name="isolation_forest",
        params={"n_estimators": 20, "contamination": 0.05, "random_state": 42},
        X_train=X_train,
        df_eval=df_eval,
        feature_names=["f1", "f2", "f3", "f4", "f5"],
    )

    assert res["model_name"] == "isolation_forest"
    assert res["n_samples"] == 100
    assert len(res["anomaly_scores"]) == 100
    assert res["n_anomalies"] > 0


def test_train_local_outlier_factor():
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    df_eval = pl.DataFrame({"idx": range(100)})

    res = train_and_evaluate_model(
        model_name="local_outlier_factor",
        params={"n_neighbors": 10, "contamination": 0.05, "novelty": True},
        X_train=X_train,
        df_eval=df_eval,
        feature_names=["f1", "f2", "f3", "f4", "f5"],
    )

    assert res["model_name"] == "local_outlier_factor"
    assert res["n_samples"] == 100
    assert len(res["anomaly_scores"]) == 100
