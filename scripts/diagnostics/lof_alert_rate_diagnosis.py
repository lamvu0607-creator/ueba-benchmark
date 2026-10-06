"""
Chẩn đoán vì sao alert rate test của LOF ≈ 15,8% thay vì ngân sách 5% (xem
``docs/reports/bao_cao_lof_trung_lap.md``, mục 6).

Chạy: ``python scripts/diagnostics/lof_alert_rate_diagnosis.py [--out lof_alert_rate_diagnosis.csv]``

Mỗi thí nghiệm chọn tập FIT và tập ĐÁNH GIÁ, tiền xử lý như ``AnomalyPipeline`` (median imputer +
RobustScaler fit trên tập fit), ngưỡng = phân vị 95% điểm tập fit, rồi đo alert rate trên tập đánh
giá cho LOF (HNSWLOF, cấu hình YAML) và Isolation Forest (đối chứng):

* ``random_holdout``   — fit 80% dòng ngẫu nhiên của ngày 1–42, đánh giá 20% còn lại (không trôi
  thời gian) ⇒ đo riêng độ lệch "leave-self-out trên train vs out-of-sample".
* ``time_holdout``     — fit ngày 1–35, đánh giá ngày 36–42 (trôi thời gian ngắn, vẫn trong train).
* ``baseline``         — fit ngày 1–42, đánh giá ngày 43–60 (đúng benchmark).
* ``no_warmup``        — fit ngày 8–42 (bỏ 7 ngày đầu có đặc trưng *_7d NULL), đánh giá ngày 43–60.
* ``no_warmup_time``   — fit ngày 8–35, đánh giá ngày 36–42.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from pyod.models.iforest import IForest
from sklearn.impute import SimpleImputer
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

CONTAMINATION = 0.05


def experiments(df: pl.DataFrame, seed: int):
    day = pl.col("day")
    train = df.filter(day <= 42)
    mask = np.random.default_rng(seed).random(train.height) < 0.8
    yield "random_holdout", train.filter(pl.Series(mask)), train.filter(pl.Series(~mask))
    yield "time_holdout", df.filter(day <= 35), df.filter((day > 35) & (day <= 42))
    yield "baseline", train, df.filter(day > 42)
    yield "no_warmup", df.filter((day >= 8) & (day <= 42)), df.filter(day > 42)
    yield "no_warmup_time", df.filter((day >= 8) & (day <= 35)), df.filter((day > 35) & (day <= 42))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None, help="Ghi bảng kết quả ra CSV")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    params = yaml.safe_load(open(REPO_ROOT / "configs" / "model_params.yaml", encoding="utf-8"))
    lof_cfg = {k: v for k, v in params["local_outlier_factor"].items() if k != "contamination"}
    if_cfg = {k: v for k, v in params["isolation_forest"].items() if k != "contamination"}

    feats = FeatureSchema().core_features
    df = pl.read_parquet(REPO_ROOT / "data" / "processed" / "feature_matrix_processed.parquet",
                         columns=feats + ["day"])

    rows = []
    for name, fit_df, eval_df in experiments(df, args.seed):
        X_fit = fit_df.select(feats).cast(pl.Float64).to_numpy()
        X_ev = eval_df.select(feats).cast(pl.Float64).to_numpy()
        imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_fit)
        sc = RobustScaler().fit(imp.transform(X_fit))
        X_fit, X_ev = sc.transform(imp.transform(X_fit)), sc.transform(imp.transform(X_ev))
        null_fit = float(fit_df.select(feats).null_count().sum_horizontal()[0]) / fit_df.height

        for model_name, model in (
            ("local_outlier_factor", HNSWLOF(contamination=CONTAMINATION, **lof_cfg)),
            ("isolation_forest", IForest(contamination=CONTAMINATION, **if_cfg)),
        ):
            t0 = time.time()
            model.fit(X_fit)
            fit_scores, ev_scores = model.decision_scores_, model.decision_function(X_ev)
            thr = float(np.quantile(fit_scores, 1 - CONTAMINATION))
            ev_days = eval_df["day"].to_numpy()
            per_day = [float((ev_scores[ev_days == d] > thr).mean()) for d in np.unique(ev_days)]
            rows.append({
                "experiment": name,
                "model": model_name,
                "fit_days": f"{fit_df['day'].min()}-{fit_df['day'].max()}",
                "eval_days": f"{eval_df['day'].min()}-{eval_df['day'].max()}",
                "n_fit": fit_df.height,
                "n_eval": eval_df.height,
                "nulls_per_fit_row": null_fit,
                "threshold": thr,
                "alert_eval_pct": 100 * float((ev_scores > thr).mean()),
                "alert_eval_day_min_pct": 100 * min(per_day),
                "alert_eval_day_max_pct": 100 * max(per_day),
                "seconds": time.time() - t0,
            })
            print(rows[-1], flush=True)

    out = pl.DataFrame(rows)
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=300, float_precision=3):
        print(out)
    if args.out:
        out.write_csv(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
