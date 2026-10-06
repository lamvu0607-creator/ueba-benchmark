"""
So sánh hai lần chạy benchmark: thời gian fit/chấm điểm, alert rate trên test và % trùng Top-5% cảnh báo.

Chạy:
    python scripts/diagnostics/compare_benchmark_runs.py \
        --old experiments/results_baseline_libsvm_20261004 --new experiments/results

Top-5% của mỗi mô hình = 5% dòng test có điểm cao nhất (xếp theo ``<model>_score`` trong
``anomaly_scores.parquet``), ghép theo khoá (DomainName, UserName, day).

Run có phân khúc (cột ``segment``) được so theo cặp (segment, model); run cũ không có cột này được
coi là ``segment = "all"``, nên so một run phân khúc với một run gộp chỉ ghép được các dòng có cùng nhãn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import polars as pl

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

KEYS = ["DomainName", "UserName", "day"]


def with_segment(frame):
    """Run cũ (trước khi có phân khúc) không có cột ``segment`` ⇒ coi là "all"."""
    if isinstance(frame, pl.DataFrame):
        return frame if "segment" in frame.columns else frame.with_columns(pl.lit("all").alias("segment"))
    return frame if "segment" in frame.columns else frame.assign(segment="all")


def top_keys(scores: pl.DataFrame, model: str, frac: float) -> set:
    k = int(round(scores.height * frac))
    top = scores.select(KEYS + [f"{model}_score"]).sort(f"{model}_score", descending=True).head(k)
    return set(top.select(KEYS).iter_rows())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    args = ap.parse_args()

    cols = ["model", "n_fit", "fit_seconds", "score_seconds", "alert_rate_pct"]
    index = ["segment", "model"]
    old = with_segment(pd.read_csv(args.old / "benchmark_summary.csv"))[index + cols[1:]].set_index(index)
    new = with_segment(pd.read_csv(args.new / "benchmark_summary.csv"))[index + cols[1:]].set_index(index)
    s_old = with_segment(pl.read_parquet(args.old / "anomaly_scores.parquet"))
    s_new = with_segment(pl.read_parquet(args.new / "anomaly_scores.parquet"))

    rows = []
    for seg, m in new.index:
        rec = {"segment": seg, "model": m}
        for c in cols[1:]:
            rec[f"{c}_old"] = old.loc[(seg, m), c] if (seg, m) in old.index else None
            rec[f"{c}_new"] = new.loc[(seg, m), c]
        if f"{m}_score" in s_old.columns and f"{m}_score" in s_new.columns:
            seg_old = s_old.filter(pl.col("segment") == seg)
            if seg_old.height:
                a = top_keys(seg_old, m, args.frac)
                b = top_keys(s_new.filter(pl.col("segment") == seg), m, args.frac)
                rec[f"top{int(args.frac * 100)}pct_overlap"] = len(a & b) / max(len(a), 1)
        rows.append(rec)
    out = pd.DataFrame(rows).set_index(index)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
