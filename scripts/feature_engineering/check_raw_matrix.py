"""Kiểm tra ma trận đặc trưng RAW (bước 1) — bất biến + phân phối.

Cách dùng:
    python scripts/feature_engineering/check_raw_matrix.py
    python scripts/feature_engineering/check_raw_matrix.py --matrix data/features/raw/feature_matrix_raw.parquet

Exit code 0 nếu PASS, 1 nếu có mục FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]

META_COLS = ["DomainName", "UserName", "day", "entity_type"]
EXPECTED_FEATS = [
    "total_logons",
    "failure_ratio", "failure_locked_out_share",
    "off_hours_ratio",
    "interarrival_dt_mean", "delta_t_cv", "same_second_share", "is_single_event",
    "interactive_ratio", "rare_logon_type_count",
    "ntlm_ratio",
    "distinct_hosts", "distinct_sources_count",
    "missing_source_ratio", "remote_logon_ratio", "custom_proc_share",
]
TOP_N = 4


def main() -> int:
    parser = argparse.ArgumentParser(description="Nghiệm thu ma trận đặc trưng RAW (không log)")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "raw" / "feature_matrix_raw.parquet",
    )
    args = parser.parse_args()

    fails = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal fails
        if not ok:
            fails += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    if not args.matrix.exists():
        print(f"[LỖI] Không tìm thấy ma trận: {args.matrix}")
        return 1

    df = pl.read_parquet(args.matrix)
    print("=" * 100)
    print("NGHIỆM THU MA TRẬN ĐẶC TRƯNG THÔ (RAW) — CHƯA CHUẨN HÓA")
    print("=" * 100)
    print(f"File : {args.matrix}  ({args.matrix.stat().st_size / 1e6:.2f} MB)")
    print(f"Quy mô: {df.height:,} dòng × {df.width} cột | ngày {df['day'].min()}..{df['day'].max()} "
          f"({df['day'].n_unique()} ngày)")
    print(f"Cột  : {df.columns}")
    print()

    check("1. Đúng hợp đồng 20 cột", list(df.columns) == META_COLS + EXPECTED_FEATS,
          f"{len(df.columns)} cột")
    log_cols = [c for c in df.columns if c.startswith("log_")]
    check("2. KHÔNG có cột log (bất biến của bước RAW)", len(log_cols) == 0,
          f"cột vi phạm: {log_cols or 'không có'}")
    const = [c for c in EXPECTED_FEATS if df.select(pl.col(c).drop_nulls().n_unique()).item() <= 1]
    check("3. Không có cột hằng số", len(const) == 0, f"cột vi phạm: {const or 'không có'}")

    dup = df.select(META_COLS[:3]).is_duplicated().sum()
    check("4. Khoá (DomainName, UserName, day) duy nhất", dup == 0, f"{dup} dòng trùng")
    print(f"        -> {df.select(['DomainName', 'UserName']).unique().height:,} danh tính")
    et = (df.group_by("entity_type").agg(pl.len().alias("n"), pl.col("UserName").n_unique().alias("acc"))
            .sort("n", descending=True))
    print("        entity_type: " + " | ".join(f"{r[0]}={r[1]:,}" for r in et.iter_rows()))

    bad_lock = df.filter((pl.col("failure_ratio") == 0) & pl.col("failure_locked_out_share").is_not_null()).height
    bad_dt = df.filter(pl.col("interarrival_dt_mean").is_null() & (pl.col("is_single_event") == 0)).height
    check("5. NULL đúng chỗ", bad_lock == 0 and bad_dt == 0,
          f"failure_locked_out_share sai {bad_lock} | interarrival sai {bad_dt}")

    oob = []
    for c in EXPECTED_FEATS:
        if c.endswith("_ratio") or c.endswith("_share"):
            s = df[c].drop_nulls()
            if s.len() and (float(s.min()) < -1e-9 or float(s.max()) > 1 + 1e-9):
                oob.append(c)
    check("6. Ratio/share ∈ [0,1]", len(oob) == 0, f"cột vi phạm: {oob or 'không có'}")

    # Bảng phân phối (số đo thật)
    print()
    print("BẢNG ĐỘ LỆCH PHÂN PHỐI (RAW — không log)")
    rows = []
    for c in EXPECTED_FEATS:
        s = df[c]
        nn = s.drop_nulls()
        nn_pd = nn.to_pandas()
        sk = float(nn_pd.skew())
        can_log = bool(nn_pd.min() >= 0)
        sk_log = float(np.log1p(nn_pd).skew()) if can_log else float("nan")
        rows.append({
            "feature": c,
            "null_%": round(s.null_count() / df.height * 100, 2),
            "zero_%": round(float((nn == 0).sum()) / nn.len() * 100, 2),
            "p50": round(float(nn.median()), 3),
            "p99": round(float(nn.quantile(0.99)), 3),
            "max": round(float(nn.max()), 1),
            "skew": round(sk, 2),
            "kurtosis": round(float(nn_pd.kurtosis()), 1),
            "skew_log1p": round(sk_log, 2) if not np.isnan(sk_log) else None,
            "reduction_pct": round((abs(sk) - abs(sk_log)) / abs(sk) * 100, 1)
            if (not np.isnan(sk_log) and abs(sk) > 1e-9) else None,
        })
    tbl = pd.DataFrame(rows).sort_values("skew", ascending=False)
    print(tbl.to_string(index=False))

    recs = tbl.to_dict("records")
    print()
    print(f"{TOP_N} cột lệch nặng nhất: "
          + ", ".join(f"{r['feature']} ({r['skew']})" for r in recs[:TOP_N]))
    print("Nếu chuẩn hóa log1p, mức giảm skew tương ứng: "
          + ", ".join(f"{r['feature']}: {r['reduction_pct']}%" for r in recs[:TOP_N]))

    print()
    print("=" * 100)
    print(f"KẾT LUẬN: {'✅ TẤT CẢ PASS' if fails == 0 else f'❌ {fails} mục FAIL'} (6 kiểm tra cấu trúc)")
    print("=" * 100)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
