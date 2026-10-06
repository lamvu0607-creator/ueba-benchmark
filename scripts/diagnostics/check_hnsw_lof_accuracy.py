"""
Kiểm chứng độ chính xác của ``HNSWLOF`` (láng giềng xấp xỉ) so với LOF chính xác của sklearn.

Chạy: ``python scripts/diagnostics/check_hnsw_lof_accuracy.py [--n-train 100000 --n-test 50000 --ef 200]``

Lấy mẫu ngẫu nhiên (seed 42) ``n_train`` dòng từ tập train (ngày ≤ split_day) và ``n_test`` dòng
từ tập test, tiền xử lý ĐÚNG như ``AnomalyPipeline`` (median imputer + RobustScaler fit trên mẫu
train), rồi so ``HNSWLOF`` với ``sklearn.neighbors.LocalOutlierFactor(novelty=True)`` cùng
``n_neighbors``: tương quan Spearman và % trùng Top-5% trên tập test, kèm thời gian.
Mục tiêu: Spearman ≥ 0,99 — nếu thấp hơn, tăng ``--ef`` rồi chạy lại.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features.schema import FeatureSchema  # noqa: E402
from src.models.pyod_detectors import HNSWLOF  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass


def top_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.05) -> float:
    k = max(1, int(round(len(a) * frac)))
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-test", type=int, default=50_000)
    ap.add_argument("--ef", type=int, default=None, help="mặc định lấy từ configs/model_params.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    params = yaml.safe_load(open(REPO_ROOT / "configs" / "model_params.yaml", encoding="utf-8"))
    lof_cfg = dict(params["local_outlier_factor"])
    lof_cfg.pop("contamination", None)
    if args.ef is not None:
        lof_cfg["ef"] = args.ef
    split_day = yaml.safe_load(open(REPO_ROOT / "configs" / "system_config.yaml", encoding="utf-8"))[
        "evaluation"]["split_day"]

    feats = FeatureSchema().core_features
    df = pl.read_parquet(REPO_ROOT / "data" / "processed" / "feature_matrix_processed.parquet",
                         columns=feats + ["day"])
    train = df.filter(pl.col("day") <= split_day).sample(args.n_train, seed=args.seed)
    test = df.filter(pl.col("day") > split_day).sample(args.n_test, seed=args.seed)
    X_tr = train.select(feats).cast(pl.Float64).to_numpy()
    X_te = test.select(feats).cast(pl.Float64).to_numpy()
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_tr)
    sc = RobustScaler().fit(imp.transform(X_tr))
    X_tr, X_te = sc.transform(imp.transform(X_tr)), sc.transform(imp.transform(X_te))
    print(f"Mẫu: train {X_tr.shape}, test {X_te.shape}, {len(feats)} đặc trưng; HNSW {lof_cfg}")

    t0 = time.time()
    approx_model = HNSWLOF(random_state=42, **lof_cfg).fit(X_tr)
    t_fit_a = time.time() - t0
    t0 = time.time()
    approx = approx_model.decision_function(X_te)
    t_score_a = time.time() - t0
    print(f"HNSWLOF: fit {t_fit_a:.1f}s, score {t_score_a:.1f}s; trùng lặp: {approx_model.duplicate_stats_}")

    t0 = time.time()
    exact_model = LocalOutlierFactor(n_neighbors=lof_cfg["n_neighbors"], novelty=True, n_jobs=-1).fit(X_tr)
    t_fit_e = time.time() - t0
    t0 = time.time()
    exact = -exact_model.score_samples(X_te)
    t_score_e = time.time() - t0
    print(f"sklearn LOF: fit {t_fit_e:.1f}s, score {t_score_e:.1f}s")

    rho = spearmanr(approx, exact).statistic
    print(f"\nSpearman = {rho:.4f}  (mục tiêu ≥ 0,99)")
    print(f"Trùng Top-5% = {top_overlap(approx, exact):.2%}  ·  Top-1% = {top_overlap(approx, exact, 0.01):.2%}")
    print(f"Nhanh hơn: fit+score ×{(t_fit_e + t_score_e) / (t_fit_a + t_score_a):.2f}")
    return 0 if rho >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
