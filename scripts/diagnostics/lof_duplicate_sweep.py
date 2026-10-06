"""
So sánh các cách xử lý dòng trùng lặp của ``HNSWLOF`` trên TOÀN BỘ train/test (xem
``docs/reports/bao_cao_lof_trung_lap.md``).

Chạy: ``python scripts/diagnostics/lof_duplicate_sweep.py [--out lof_duplicate_sweep.csv]``

Tiền xử lý đúng như ``AnomalyPipeline`` (median imputer + RobustScaler fit trên train), rồi fit
``HNSWLOF`` theo từng biến thể: không xử lý, thêm nhiễu Gauss σ vào train, sàn ``min_k_distance``,
``dedup``. In ngưỡng (p95 điểm train), alert rate test, các phân vị và Spearman / trùng Top-5%
của điểm test so với biến thể "raw".
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

#: (tên, σ nhiễu Gauss thêm vào train, tham số HNSWLOF)
VARIANTS = [
    ("raw", 0.0, {"dedup": False}),
    *[(f"jitter {s:g}", s, {"dedup": False}) for s in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)],
    ("floor 1e-3", 0.0, {"dedup": False, "min_k_distance": 1e-3}),
    ("floor 1e-2", 0.0, {"dedup": False, "min_k_distance": 1e-2}),
    ("dedup", 0.0, {"dedup": True}),
    ("dedup + floor 1e-2", 0.0, {"dedup": True, "min_k_distance": 1e-2}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None, help="Ghi bảng kết quả ra CSV")
    args = ap.parse_args()

    params = yaml.safe_load(open(REPO_ROOT / "configs" / "model_params.yaml", encoding="utf-8"))
    cfg = dict(params["local_outlier_factor"])
    for key in ("contamination", "dedup", "min_k_distance"):
        cfg.pop(key, None)
    split_day = yaml.safe_load(open(REPO_ROOT / "configs" / "system_config.yaml", encoding="utf-8"))[
        "evaluation"]["split_day"]

    feats = FeatureSchema().core_features
    df = pl.read_parquet(REPO_ROOT / "data" / "processed" / "feature_matrix_processed.parquet",
                         columns=feats + ["day"])
    X_tr = df.filter(pl.col("day") <= split_day).select(feats).cast(pl.Float64).to_numpy()
    X_te = df.filter(pl.col("day") > split_day).select(feats).cast(pl.Float64).to_numpy()
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_tr)
    sc = RobustScaler().fit(imp.transform(X_tr))
    X_tr, X_te = sc.transform(imp.transform(X_tr)), sc.transform(imp.transform(X_te))
    print(f"train {X_tr.shape}, test {X_te.shape}; HNSW {cfg}", flush=True)

    ref, rows = None, []
    for name, sigma, kw in VARIANTS:
        X_fit = X_tr + np.random.default_rng(42).normal(0.0, sigma, X_tr.shape) if sigma > 0 else X_tr
        t0 = time.time()
        model = HNSWLOF(random_state=42, **cfg, **kw).fit(X_fit)
        tr, te = model.decision_scores_, model.decision_function(X_te)
        secs = time.time() - t0
        thr = float(np.quantile(tr, 0.95))
        ref = te if ref is None else ref
        k = int(len(te) * 0.05)
        rows.append({
            "variant": name,
            "n_index": model.n_index_points_,
            "k_distance_zero": int((model._k_distance == 0).sum()),
            "threshold": thr,
            "alert_test_pct": 100 * float((te > thr).mean()),
            "train_p100": float(tr.max()),
            **{f"test_p{p:g}": float(np.percentile(te, p)) for p in (50, 95, 99, 99.9, 100)},
            "n_test_gt_1e3": int((te > 1e3).sum()),
            "spearman_vs_raw": float(spearmanr(te, ref).statistic),
            "top5_overlap_vs_raw": len(set(np.argsort(-te)[:k]) & set(np.argsort(-ref)[:k])) / k,
            "seconds": secs,
        })
        print(rows[-1], flush=True)

    out = pl.DataFrame(rows)
    with pl.Config(tbl_cols=-1, tbl_width_chars=300, float_precision=4):
        print(out)
    if args.out:
        out.write_csv(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
