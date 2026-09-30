"""
Unit tests for models benchmark module.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from src.features.schema import FeatureSchema
from src.models.base import as_float_matrix
from src.models.baselines import RuleThresholdBaseline, ZScoreBaseline
from src.models.benchmark import DEFAULT_MODEL_PARAMS, train_and_evaluate_model
from src.models.detectors import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
)
from src.models.pipeline import AnomalyPipeline
from src.models.registry import (
    DEFAULT_MODEL_NAMES,
    available_models,
    create_model,
    create_pipeline,
    load_params,
    resolve_model,
)

DETECTOR_CLASSES = (IsolationForestDetector, LocalOutlierFactorDetector, OneClassSVMDetector)
PARAMS_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_params.yaml"
CORE_FEATURES = list(FeatureSchema().core_features)
N_ROWS, N_TRAIN, OUTLIER_INDEX = 300, 200, 250
OUTLIER_POS_IN_TEST = OUTLIER_INDEX - N_TRAIN


def synthetic_matrix(n_samples: int = 500, n_features: int = 16, outlier_index: int = 7, seed: int = 0):
    """Ma trận tổng hợp: nhiễu chuẩn + 1 dòng dị biệt rõ ràng tại ``outlier_index``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    X[outlier_index] = 12.0
    return X


def synthetic_feature_frame(n_rows: int = N_ROWS, outlier_index: int = OUTLIER_INDEX, seed: int = 1) -> pl.DataFrame:
    """
    DataFrame mô phỏng parquet thật: đủ 22 đặc trưng core + vài cột KHÔNG core (bẫy),
    cố tình để 3 NULL hợp lệ, và 1 dòng dị biệt (mọi đặc trưng = 99.0) nằm ngoài tập train.
    ``outlier_index=None`` để tạo dữ liệu "sạch" (không tiêm dị biệt) phục vụ đo drift.
    """
    rng = np.random.default_rng(seed)
    data: dict = {
        "DomainName": ["DOM"] * n_rows,
        "UserName": [f"User{i}" for i in range(n_rows)],
        "day": [1] * n_rows,
        "entity_type": ["User"] * n_rows,
        "total_logons": rng.integers(1, 50, size=n_rows).astype(float).tolist(),
        "distinct_hosts": rng.integers(1, 5, size=n_rows).astype(float).tolist(),
        "rare_logon_type_count": np.zeros(n_rows).tolist(),
        "user_key": [f"key{i}" for i in range(n_rows)],
    }
    for name in CORE_FEATURES:
        if name in ("interarrival_dt_mean", "log_total_logons", "log_distinct_hosts", "rare_logon_type_count_log"):
            values = (rng.random(n_rows) * 50.0).tolist()
        elif name == "delta_t_cv":
            values = (rng.random(n_rows) * 3.0).tolist()
        elif name == "is_single_event":
            values = rng.integers(0, 2, size=n_rows).astype(float).tolist()
        else:
            values = (rng.random(n_rows) * 0.5).tolist()
        if outlier_index is not None:
            values[outlier_index] = 99.0
        data[name] = values

    # NULL hợp lệ theo schema: imputer phải dùng median của tập train, KHÔNG được fill 0.
    data["interarrival_dt_mean"][7] = None
    data["delta_t_cv"][8] = None
    data["failure_locked_out_share"][9] = None
    return pl.DataFrame(data)


def _pipeline_benchmark(model, df_train, df_test, outlier_pos):
    """Fit pipeline trên train, chấm điểm test và kiểm tra các bất biến chung của mọi mô hình."""
    pipeline = AnomalyPipeline(model, random_state=42).fit(df_train)
    scores = pipeline.score(df_test)
    labels = pipeline.predict(df_test)

    assert scores.shape == (df_test.height,)
    assert labels.shape == (df_test.height,)
    # predict() phải đúng là so sánh với ngưỡng đã fit trên TRAIN (không nhìn tập đánh giá).
    assert np.array_equal(labels, (scores >= pipeline.model.threshold_).astype(np.int8))

    # Cơ chế ngân sách cảnh báo: phân vị hạng TRÊN CHÍNH TẬP ĐÁNH GIÁ -> đúng ~contamination.
    rank_pct = pipeline.score_rank_pct(df_test, reference_scores=scores)
    budget_flags = rank_pct >= 1.0 - model.contamination
    assert abs(float(budget_flags.mean()) - model.contamination) <= 0.02

    # Ngưỡng fit trên train nên alert rate của predict() có thể trôi khi tập train quá nhỏ:
    # đo bằng scripts/diagnostics/alert_rate_calibration.py:
    #   train 200 dòng   -> IF 19.1%, LOF 10.2%, OCSVM 31.7%
    #   train 5.000 dòng -> IF  5.3%, LOF  6.9%, OCSVM  6.1%
    # Tức đây là hiện tượng mẫu nhỏ (không phải lỗi giao diện), và được báo cáo trong benchmark
    # thay vì bị ép về đúng 5% bằng cách nhìn trước tập đánh giá.
    assert rank_pct[outlier_pos] >= 0.98, f"{model.name} không xếp dòng dị biệt vào nhóm điểm cao nhất"
    assert labels[outlier_pos] == 1
    assert pipeline.model.n_fit_ == df_train.height
    return pipeline, scores




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


# --------------------------------------------------------------------------- #
# Giao diện chung BaseAnomalyModel (B1 + B2)
# --------------------------------------------------------------------------- #
def test_model_interfaces():
    """Cả 3 detector phơi ra đúng một API chung và trả về mảng 1 chiều cùng độ dài đầu vào."""
    X = synthetic_matrix()
    for cls in DETECTOR_CLASSES:
        model = cls(contamination=0.05, max_train_samples=200)
        assert model.name and model.aliases and model.is_baseline is False
        assert model.requires_scaling is True
        for method in ("fit", "score", "predict", "score_rank_pct", "get_metadata", "save", "load"):
            assert callable(getattr(model, method)), f"{cls.__name__} thiếu phương thức {method}"

        model.fit(X)
        scores = model.score(X)
        labels = model.predict(X)
        assert scores.shape == (X.shape[0],)
        assert labels.shape == (X.shape[0],)
        assert set(np.unique(labels)).issubset({0, 1})

        rank_pct = model.score_rank_pct(X, reference_scores=scores)
        assert rank_pct.min() >= 0.0 and rank_pct.max() <= 1.0
        # Thứ hạng percentile phải đơn điệu với điểm thô (cùng một thứ tự xếp hạng).
        assert np.array_equal(np.argsort(scores), np.argsort(rank_pct))

        metadata = model.get_metadata()
        assert metadata["model_name"] == model.name
        assert metadata["is_fitted"] is True
        assert metadata["threshold"] is not None


def test_score_direction():
    """Bất biến số 1: CAO = DỊ BIỆT — argmax điểm phải là dòng dị biệt được tiêm."""
    X = synthetic_matrix(outlier_index=7)
    for cls in DETECTOR_CLASSES:
        model = cls(contamination=0.05, max_train_samples=200).fit(X)
        assert int(np.argmax(model.score(X))) == 7, f"{cls.__name__} bị đảo dấu điểm dị biệt"


def test_model_save_and_load(temp_artifact_dir):
    """Roundtrip joblib: điểm số và nhãn phải y hệt sau khi nạp lại."""
    X = synthetic_matrix()
    model = IsolationForestDetector(contamination=0.05, max_train_samples=200).fit(X)
    path = model.save(temp_artifact_dir / "isolation_forest.joblib")
    assert path.is_file()

    loaded = IsolationForestDetector.load(path)
    assert isinstance(loaded, IsolationForestDetector)
    assert loaded.threshold_ == model.threshold_
    assert loaded.n_fit_ == model.n_fit_
    assert np.allclose(loaded.score(X), model.score(X))
    assert np.array_equal(loaded.predict(X), model.predict(X))

    with pytest.raises(FileNotFoundError):
        IsolationForestDetector.load(temp_artifact_dir / "khong_ton_tai.joblib")


def test_max_train_samples_subsampling():
    """Ngân sách fit được log tách bạch: n_fit_ vs số dòng tập train; 0 = không giới hạn."""
    X = synthetic_matrix(n_samples=500)
    capped = OneClassSVMDetector(contamination=0.05, max_train_samples=100).fit(X)
    assert capped.n_fit_ == 100
    assert capped.subsample_indices_ is not None
    assert len(capped.subsample_indices_) == 100

    uncapped = OneClassSVMDetector(contamination=0.05, max_train_samples=0).fit(X)
    assert uncapped.n_fit_ == 500
    assert uncapped.subsample_indices_ is None

    assert LocalOutlierFactorDetector.default_max_train_samples == 20_000
    assert IsolationForestDetector.default_max_train_samples is None


def test_input_sanitization():
    """NaN/Inf, sai số đặc trưng, tham số sai và gọi khi chưa fit đều phải báo lỗi rõ ràng."""
    model = IsolationForestDetector(contamination=0.05).fit(synthetic_matrix(n_samples=100))

    X_nan = synthetic_matrix(n_samples=100)
    X_nan[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        model.score(X_nan)

    with pytest.raises(ValueError, match="sai số đặc trưng"):
        model.score(synthetic_matrix(n_samples=10, n_features=4))

    with pytest.raises(ValueError, match="2 chiều"):
        as_float_matrix(np.zeros(5))

    with pytest.raises(ValueError, match="contamination"):
        IsolationForestDetector(contamination=1.5)

    with pytest.raises(ValueError, match="Tham số không hợp lệ"):
        IsolationForestDetector(n_estimators=10, khong_ton_tai=1)

    with pytest.raises(RuntimeError, match="chưa được fit"):
        IsolationForestDetector().score(synthetic_matrix(n_samples=10))


def test_params_whitelist():
    """Mọi key trong configs/model_params.yaml phải là tham số hợp lệ của sklearn (chống key chết)."""
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mapping = {
        "isolation_forest": IsolationForestDetector,
        "local_outlier_factor": LocalOutlierFactorDetector,
        "one_class_svm": OneClassSVMDetector,
    }
    for section, cls in mapping.items():
        allowed = set(cls.native_param_names())
        unknown = sorted(set(raw.get(section, {})) - allowed)
        assert not unknown, f"{section}: tham số không tồn tại trong sklearn {sorted(unknown)}"
        assert allowed, f"{cls.__name__} chưa khai báo estimator_class"


# --------------------------------------------------------------------------- #
# Baseline label-free (B3)
# --------------------------------------------------------------------------- #
def test_baselines():
    """Z-score và luật ngưỡng: cùng API, bắt được dòng bất thường, alert rate ~ contamination."""
    X = synthetic_matrix(n_samples=500, outlier_index=7)

    z_model = ZScoreBaseline(method="robust", agg="max").fit(X)
    assert z_model.is_baseline is True
    assert z_model.requires_scaling is False
    assert int(np.argmax(z_model.score(X))) == 7
    assert abs(float(z_model.predict(X).mean()) - z_model.contamination) < 0.02

    fixed = ZScoreBaseline(method="classic", agg="mean", z_threshold=3.0, threshold_mode="fixed").fit(X)
    assert fixed.threshold_ == 3.0

    names = [f"f{i}" for i in range(16)]
    rules = [
        {"feature": "f0", "op": ">=", "value": 5.0, "rationale": "kiểm thử"},
        {"feature": "f1", "op": ">=", "value": 5.0, "rationale": "kiểm thử"},
        {"feature": "f2", "op": ">=", "value": 5.0, "rationale": "kiểm thử"},
    ]
    rule_model = RuleThresholdBaseline(rules=rules, feature_names=names).fit(X)
    scores = rule_model.score(X)
    assert 0.0 <= scores.min() and scores.max() <= 1.0
    assert scores[7] == 1.0
    assert float(np.median(scores)) == 0.0
    assert rule_model.get_metadata()["n_rules"] == 3

    rule_fixed = RuleThresholdBaseline(
        rules=rules, feature_names=names, threshold_mode="fixed", min_rules=2
    ).fit(X)
    assert rule_fixed.threshold_ == pytest.approx(2.0 / 3.0)

    with pytest.raises(ValueError, match="không thuộc tập đặc trưng"):
        RuleThresholdBaseline(rules=[{"feature": "khong_co", "op": ">=", "value": 1.0}], feature_names=names).fit(X)
    with pytest.raises(ValueError, match="toán tử"):
        RuleThresholdBaseline(rules=[{"feature": "f0", "op": "~=", "value": 1.0}], feature_names=names).fit(X)
    with pytest.raises(ValueError, match="cần TÊN đặc trưng"):
        RuleThresholdBaseline(rules=rules).fit(X)


def test_baselines_config_integration():
    """Cấu hình YAML phải nạp được vào 2 baseline (6 luật, tham số z-score hợp lệ)."""
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    assert {"zscore_baseline", "rule_threshold_baseline"} <= set(raw)
    assert len(raw["rule_threshold_baseline"]["rules"]) == 6

    X = synthetic_matrix(n_samples=300, outlier_index=3)
    names = [f"f{i}" for i in range(16)]
    z_model = ZScoreBaseline(**raw["zscore_baseline"]).fit(X)
    assert int(np.argmax(z_model.score(X))) == 3

    # Đặc trưng luật thật không có trong ma trận giả lập -> phải báo lỗi rõ ràng, không im lặng bỏ luật.
    with pytest.raises(ValueError, match="không thuộc tập đặc trưng"):
        RuleThresholdBaseline(feature_names=names, **raw["rule_threshold_baseline"]).fit(X)


# --------------------------------------------------------------------------- #
# Pipeline 16 đặc trưng core (B4)
# --------------------------------------------------------------------------- #
def test_core_features_only():
    """Pipeline chỉ dùng 20 đặc trưng core; cột không core không được ảnh hưởng điểm; NULL phải impute có căn cứ."""
    df = synthetic_feature_frame()
    pipeline = AnomalyPipeline(IsolationForestDetector(contamination=0.05, random_state=42), random_state=42)

    assert pipeline.feature_names == CORE_FEATURES
    assert len(pipeline.feature_names) == 20
    assert pipeline.get_metadata()["scaler"] == "RobustScaler"

    pipeline.fit(df.head(N_TRAIN))
    assert pipeline.n_nulls_imputed_ == 3

    # Median của tập train cho 3 cột có NULL hợp lệ: khác 0 -> chứng minh không fill 0 mù quáng.
    for feature in ("interarrival_dt_mean", "delta_t_cv", "failure_locked_out_share"):
        stat = pipeline.imputer_.statistics_[pipeline.feature_names.index(feature)]
        assert stat > 0.0

    df_test = df.tail(N_ROWS - N_TRAIN)
    base_scores = pipeline.score(df_test)
    df_perturbed = df_test.with_columns(
        [
            (pl.col("total_logons") * 1000).alias("total_logons"),
            (pl.col("distinct_hosts") + 999).alias("distinct_hosts"),
            pl.lit(12345.0).alias("cot_moi_khong_core"),
        ]
    )
    assert np.allclose(pipeline.score(df_perturbed), base_scores)

    with pytest.raises(ValueError, match="thiếu"):
        pipeline.score(df.drop("failure_ratio"))
    with pytest.raises(RuntimeError, match="chưa fit"):
        AnomalyPipeline(IsolationForestDetector()).transform(df)


def test_isolation_forest_pipeline(temp_artifact_dir):
    """Pipeline Isolation Forest: alert rate theo ngân sách, bắt được dòng dị biệt, roundtrip joblib."""
    df = synthetic_feature_frame()
    pipeline, scores = _pipeline_benchmark(
        IsolationForestDetector(contamination=0.05, random_state=42),
        df.head(N_TRAIN),
        df.tail(N_ROWS - N_TRAIN),
        OUTLIER_POS_IN_TEST,
    )

    path = pipeline.save(temp_artifact_dir / "isolation_forest_pipeline.joblib")
    reloaded = AnomalyPipeline.load(path)
    assert reloaded.model.name == "isolation_forest"
    assert reloaded.get_metadata()["n_features"] == 20
    assert np.allclose(reloaded.score(df.tail(N_ROWS - N_TRAIN)), scores)


def test_local_outlier_factor_pipeline():
    """Pipeline LOF: ngân sách fit mặc định 20.000 dòng, điểm CAO = DỊ BIỆT."""
    df = synthetic_feature_frame()
    model = LocalOutlierFactorDetector(contamination=0.05, random_state=42)
    assert model.max_train_samples == 20_000

    pipeline, _ = _pipeline_benchmark(model, df.head(N_TRAIN), df.tail(N_ROWS - N_TRAIN), OUTLIER_POS_IN_TEST)
    assert pipeline.model.estimator_.novelty is True


def test_one_class_svm_pipeline():
    """Pipeline One-Class SVM: dùng ngưỡng phân vị thay cho predict() nội bộ của sklearn."""
    df = synthetic_feature_frame()
    model = OneClassSVMDetector(contamination=0.05, random_state=42, nu=0.05)
    assert model.contamination == 0.05  # 'nu' được đồng bộ thành ngân sách cảnh báo

    pipeline, scores = _pipeline_benchmark(model, df.head(N_TRAIN), df.tail(N_ROWS - N_TRAIN), OUTLIER_POS_IN_TEST)

    # Ngưỡng phải là phân vị trên tập train, không phải ngưỡng offset nội bộ của sklearn.
    expected = np.quantile(model.fit_scores_, 1.0 - model.contamination)
    assert model.threshold_ == pytest.approx(expected)
    assert int(np.argmax(scores)) == OUTLIER_POS_IN_TEST


# --------------------------------------------------------------------------- #
# Registry & factory (B5)
# --------------------------------------------------------------------------- #
def test_model_factory():
    """Factory: 5 tên canonical + alias CamelCase của log cũ tạo đúng lớp; tên lạ bị chặn."""
    params = load_params(PARAMS_PATH)

    assert DEFAULT_MODEL_NAMES == [
        "isolation_forest",
        "local_outlier_factor",
        "one_class_svm",
        "zscore_baseline",
        "rule_threshold_baseline",
    ]
    assert available_models() == DEFAULT_MODEL_NAMES

    for name in available_models():
        model = create_model(name, params=params)
        assert isinstance(model, resolve_model(name))
        assert model.contamination == params["defaults"]["contamination"]
        assert model.random_state == params["defaults"]["random_state"]

        pipeline = create_pipeline(name, params=params)
        assert pipeline.model.name == name
        assert pipeline.get_metadata()["imputer"] == "SimpleImputer"
        expected_scaler = "RobustScaler" if model.requires_scaling else None
        assert pipeline.get_metadata()["scaler"] == expected_scaler

    # Alias tương thích ngược: experiment_log.csv cũ ghi tên theo CamelCase.
    assert isinstance(create_model("IsolationForest", params=params), IsolationForestDetector)
    assert isinstance(create_model("LocalOutlierFactor", params=params), LocalOutlierFactorDetector)
    assert isinstance(create_model("OneClassSVM", params=params), OneClassSVMDetector)
    assert isinstance(create_model("zscore", params=params), ZScoreBaseline)

    # Tham số truyền trực tiếp phải thắng cấu hình (để CLI ghi đè được khi tái lập).
    overridden = create_model("isolation_forest", params=params, contamination=0.10, random_state=7)
    assert overridden.contamination == 0.10
    assert overridden.random_state == 7

    with pytest.raises(KeyError, match="không tồn tại"):
        create_model("khong_co_model", params=params)
    with pytest.raises(ValueError, match="Tham số không hợp lệ"):
        create_model("isolation_forest", params={"isolation_forest": {"sai_key": 1}})
    with pytest.raises(ValueError, match="contamination"):
        create_model("isolation_forest", params=params, contamination=2.0)


