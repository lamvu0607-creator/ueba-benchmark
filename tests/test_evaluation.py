"""
Unit tests for the evaluation package: time-based split, label-free metrics, experiment log, manifest.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from src.evaluation.metrics import (
    alert_rate,
    average_precision,
    budget_flags,
    budget_size,
    pairwise_stability,
    precision_at_k,
    rank_correlation,
    recall_at_budget,
    roc_auc,
    score_quantiles,
    timed,
    top_k_indices,
    topk_overlap,
)
from src.evaluation.split import SplitInfo, resolve_split_day, time_split
from src.models.baselines import ZScoreBaseline

PROCESSED_MATRIX = Path(__file__).resolve().parents[1] / "data" / "processed" / "feature_matrix_processed.parquet"


def synthetic_days_frame(n_days: int = 6, rows_per_day: int = 10, day_col: str = "day") -> pl.DataFrame:
    """Frame nhiều ngày, mỗi ngày đúng ``rows_per_day`` dòng (dễ kiểm tra số học của split)."""
    days = np.repeat(np.arange(1, n_days + 1), rows_per_day)
    return pl.DataFrame({day_col: days, "value": np.arange(days.size, dtype=np.float64)})


def test_time_based_split():
    """Chia theo ngày: đúng ranh giới, không giao nhau, không mất dòng, metadata đầy đủ."""
    df = synthetic_days_frame(n_days=6, rows_per_day=10)

    train, test, info = time_split(df, day_col="day", split_day=3)

    assert train.height == 30 and test.height == 30
    assert train["day"].max() == 3
    assert test["day"].min() == 4
    assert set(train["day"].unique().to_list()).isdisjoint(set(test["day"].unique().to_list()))
    assert train.height + test.height == df.height
    assert set(train.columns) == set(df.columns)

    assert isinstance(info, SplitInfo)
    assert (info.split_day, info.min_day, info.max_day) == (3, 1, 6)
    assert (info.n_train, info.n_eval) == (30, 30)
    assert (info.n_train_days, info.n_eval_days) == (3, 3)
    assert info.eval_ratio == pytest.approx(0.5)
    assert info.strategy == "time"

    payload = info.to_dict()
    assert payload["n_total"] == 60
    assert payload["eval_ratio"] == pytest.approx(0.5)
    assert payload["split_day"] == 3

    # Suy ra split_day từ tỷ lệ: 50% -> cắt sau ngày 3 (đúng 30/60 dòng);
    # 30% -> ngày 4 (20/60 = 33,3% gần 18 dòng mục tiêu hơn mức 30/60 = 50% của ngày 3).
    assert resolve_split_day(df, test_split_ratio=0.5) == 3
    assert resolve_split_day(df, test_split_ratio=0.30) == 4

    with pytest.raises(ValueError, match="phải nằm trong"):
        resolve_split_day(df, test_split_ratio=1.5)
    with pytest.raises(ValueError, match="cột ngày"):
        time_split(df, day_col="khong_co")
    with pytest.raises(ValueError, match="tập rỗng"):
        time_split(df, split_day=0)
    with pytest.raises(ValueError, match="ít nhất 2 ngày"):
        resolve_split_day(synthetic_days_frame(n_days=1, rows_per_day=5))


def test_time_based_split_on_real_matrix():
    """Nếu đã có parquet processed: xác nhận đúng ranh giới ngày 42 = 721.612 / 333.671 dòng."""
    if not PROCESSED_MATRIX.is_file():
        pytest.skip("Chưa có data/processed/feature_matrix_processed.parquet")

    df = pl.read_parquet(PROCESSED_MATRIX, columns=["day"])
    train, test, info = time_split(df, day_col="day", split_day=42)

    assert train["day"].max() == 42
    assert test["day"].min() == 43
    assert (info.n_train, info.n_eval) == (721_612, 333_671)
    assert info.eval_ratio == pytest.approx(0.3162, abs=1e-4)


# --------------------------------------------------------------------------- #
# Chỉ số đánh giá (B7)
# --------------------------------------------------------------------------- #
def test_evaluation_metrics():
    """Chỉ số label-free + label-aware: ngân sách, Top-K, overlap, phân vị, ổn định đa seed."""
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    labels = np.array([0, 1, 0, 1, 0])

    assert top_k_indices(scores, 2).tolist() == [1, 3]
    assert budget_size(1_055_283, 0.05) == 52_764
    flags = budget_flags(scores, 0.4)  # 2/5 dòng cao nhất
    assert flags.tolist() == [0, 1, 0, 1, 0]
    assert alert_rate(flags) == pytest.approx(0.4)

    assert topk_overlap(scores, scores, 2) == 1.0
    assert topk_overlap(scores, -scores, 2) == 0.0
    assert rank_correlation(scores, scores) == pytest.approx(1.0)
    assert rank_correlation(scores, -scores) == pytest.approx(-1.0)

    quantiles = score_quantiles(scores)
    assert len(quantiles) == 11
    assert set(quantiles) == {f"p{i:02d}" for i in range(0, 101, 10)}
    assert quantiles["p00"] == 0.1
    assert quantiles["p100"] == 0.9

    # Chỉ số cần nhãn (dùng chính thức từ Tuần 4) đã đúng trên nhãn tổng hợp.
    assert precision_at_k(labels, scores, 2) == 1.0
    assert recall_at_budget(labels, scores, 0.4) == 1.0
    assert roc_auc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0

    stability = pairwise_stability([scores, scores, scores], k=2)
    assert stability["n_runs"] == 3.0
    assert stability["spearman_mean"] == pytest.approx(1.0)
    assert stability["topk_overlap_mean"] == pytest.approx(1.0)
    reversed_stability = pairwise_stability([scores, -scores], k=2)
    assert reversed_stability["spearman_mean"] == pytest.approx(-1.0)

    # Nhóm đồng điểm: tie-break ổn định theo thứ tự dòng (tái lập được 100%).
    assert top_k_indices(np.array([1.0, 1.0, 1.0, 0.0, 0.0]), 2).tolist() == [0, 1]

    _, elapsed = timed(lambda: sum(range(1000)))
    assert elapsed >= 0.0

    with pytest.raises(ValueError, match="lớn hơn số dòng"):
        top_k_indices(scores, 10)
    with pytest.raises(ValueError, match="budget_ratio"):
        budget_size(100, 0.0)
    with pytest.raises(ValueError, match="0/1"):
        precision_at_k(np.array([0, 2, 0, 1, 0]), scores, 2)
    with pytest.raises(ValueError, match="nhãn dương"):
        recall_at_budget(np.zeros(5, dtype=int), scores, 0.4)
    with pytest.raises(ValueError, match="ít nhất 2"):
        pairwise_stability([scores], k=2)


def test_score_rank_pct_matches_score_order():
    """score_rank_pct: Spearman với điểm thô = 1.0, giá trị trong [0, 1] — điều kiện so sánh liên mô hình."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    X[5] = 8.0  # dòng dị biệt (điểm cao nhất)

    model = ZScoreBaseline(contamination=0.05).fit(X)
    scores = model.score(X)
    pct = model.score_rank_pct(X, reference_scores=scores)

    assert pct.min() >= 0.0 and pct.max() <= 1.0
    assert spearmanr(scores, pct).statistic == pytest.approx(1.0)
    assert pct[5] == pytest.approx(1.0)
    assert abs(float(pct.mean()) - 0.5) < 0.05  # xấp xỉ phân bố đều trên [0, 1]

    # Mặc định (không truyền reference) dùng điểm tập fit -> giá trị cũng phải nằm trong [0, 1].
    pct_fit = model.score_rank_pct(X)
    assert pct_fit.min() >= 0.0 and pct_fit.max() <= 1.0
