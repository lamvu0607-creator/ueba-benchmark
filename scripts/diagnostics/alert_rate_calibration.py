"""
Chẩn đoán: ``predict()`` có giữ đúng ngân sách cảnh báo (contamination) hay không?

Chạy: ``python scripts/diagnostics/alert_rate_calibration.py``

Bối cảnh thiết kế: ngưỡng cảnh báo được fit trên **tập train**:
``threshold_ = quantile(score_train, 1 - contamination)``. Khi tập train quá nhỏ trong không
gian 16 chiều, các điểm MỚI của tập đánh giá rơi ra ngoài "biên" nhiều hơn mức ngân sách
(hiệu ứng novelty trong không gian nhiều chiều), nên alert rate của ``predict()`` cao hơn 5%.

Script in ra 3 kịch bản để định lượng hiện tượng này, và cho thấy:
  * ``alert_rate(predict)`` hội tụ về contamination khi tập train đủ lớn,
  * ``budget_rate`` (xếp hạng theo phân vị trên chính tập đánh giá) luôn đúng bằng ngân sách
    theo thiết kế — đây là chỉ số so sánh liên mô hình công bằng của benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.schema import FeatureSchema  # noqa: E402
from src.models.detectors import (  # noqa: E402
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
)
from src.models.pipeline import AnomalyPipeline  # noqa: E402

CORE_FEATURES = list(FeatureSchema().core_features)
CONTAMINATION = 0.05

if sys.platform == "win32":
    # Console Windows mặc định cp1252 -> không in được tiếng Việt có dấu.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - chỉ ảnh hưởng hiển thị console
        pass

SCENARIOS = [
    ("sạch, train 200 dòng", 5_000, 200, None),
    ("sạch, train 5.000 dòng", 12_000, 5_000, None),
    ("tiêm 1 dòng dị biệt, train 200 dòng", 5_000, 200, 2_500),
]


def synthetic_frame(n_rows: int, outlier_index: int | None = None, seed: int = 1) -> pl.DataFrame:
    """Dữ liệu tổng hợp: 16 đặc trưng core dạng uniform + 3 NULL hợp lệ + (tuỳ chọn) 1 dòng dị biệt."""
    rng = np.random.default_rng(seed)
    data: dict = {"UserName": [f"User{i}" for i in range(n_rows)], "day": [1] * n_rows}
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
    data["interarrival_dt_mean"][7] = None
    data["delta_t_cv"][8] = None
    data["failure_locked_out_share"][9] = None
    return pl.DataFrame(data)


def measure(label: str, n_rows: int, n_train: int, outlier_index: int | None) -> None:
    df = synthetic_frame(n_rows=n_rows, outlier_index=outlier_index)
    train, test = df.head(n_train), df.tail(n_rows - n_train)
    print("=" * 86)
    print(f"{label} (train={n_train:,}, test={n_rows - n_train:,})")
    for cls in (IsolationForestDetector, LocalOutlierFactorDetector, OneClassSVMDetector):
        pipeline = AnomalyPipeline(cls(contamination=CONTAMINATION, random_state=42), random_state=42).fit(train)
        scores = pipeline.score(test)
        rank_pct = pipeline.score_rank_pct(test, reference_scores=scores)
        alert_rate = float(pipeline.predict(test).mean())
        budget_rate = float((rank_pct >= 1.0 - CONTAMINATION).mean())
        print(
            f"  {cls.name:24s} n_fit={pipeline.model.n_fit_:>6,}  "
            f"alert_rate(predict)={alert_rate:6.2%}  budget_rate(5%)={budget_rate:6.2%}  "
            f"score_p50={float(np.median(scores)):9.4f}"
        )


if __name__ == "__main__":
    for scenario in SCENARIOS:
        measure(*scenario)
