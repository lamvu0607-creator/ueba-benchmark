"""
check_feature_matrix.py

Kiểm tra một ma trận đặc trưng (Tài khoản × Ngày) có đúng "hợp đồng schema" bản 2.0 hay không.
Dùng được ở bất kỳ máy nào, không phụ thuộc cấu trúc repo.

Cách dùng:
    python check_feature_matrix.py --matrix data/features/model/feature_matrix_model.parquet
    python check_feature_matrix.py --matrix <path> --schema configs/feature_schema.yaml

LƯU Ý: các cột log chỉ tồn tại ở ma trận ĐÃ CHUẨN HÓA. Riêng ma trận THÔ (bước 1) hãy dùng
       `check_raw_matrix.py` (kiểm tra 20 cột, KHÔNG có cột log, khoá duy nhất, NULL đúng chỗ).

Exit code:
    0 = tất cả kiểm tra PASS
    1 = có kiểm tra FAIL (dùng được trong CI)

Căn cứ: docs/feature_engineering/feature_correlation_review.md §11.4 (Definition of Done).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

if sys.platform == "win32":  # console Windows mặc định cp1252 -> ép UTF-8 để in được tiếng Việt
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

RHO_THRESHOLD = 0.85
VIF_THRESHOLD = 10.0

META_COLS = ["UserName", "DomainName", "day", "entity_type"]
DISPLAY_COLS = ["total_logons"]
NULLABLE_WHEN_NO_FAILURE = ["failure_locked_out_share"]
NULLABLE_WHEN_SINGLE_EVENT = ["interarrival_dt_mean", "delta_t_cv"]


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, no: str, title: str, passed: bool, detail: str) -> None:
        if not passed:
            self.failed += 1
        print(f"[{'PASS' if passed else 'FAIL'}] {no} {title}: {detail}")

    def info(self, msg: str = "") -> None:
        print(msg)

    def section(self, title: str) -> None:
        print()
        print("=" * 90)
        print(title)
        print("=" * 90)


def vif_table(frame: pd.DataFrame) -> pd.Series:
    corr = frame.corr(method="pearson").to_numpy()
    corr = corr + np.eye(corr.shape[0]) * 1e-8
    return pd.Series(np.diag(np.linalg.inv(corr)), index=frame.columns).sort_values(ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra ma trận đặc trưng UEBA (schema v2.0)")
    parser.add_argument("--matrix", type=Path, default=Path("data/features/account_day_matrix.parquet"))
    parser.add_argument("--schema", type=Path, default=Path("configs/feature_schema.yaml"))
    args = parser.parse_args()

    rep = Report()
    rep.section("KIỂM TRA MA TRẬN ĐẶC TRƯNG — SCHEMA v2.0")
    if not args.matrix.exists():
        print(f"[LỖI] Không tìm thấy ma trận: {args.matrix.resolve()}")
        return 1

    df = pl.read_parquet(args.matrix)
    features = [c for c in df.columns if c not in META_COLS + DISPLAY_COLS]
    rep.info(f"Ma trận:   {args.matrix.resolve()}")
    rep.info(f"Kích thước: {df.height:,} dòng × {df.width} cột | {len(features)} đặc trưng")
    rep.info(f"Ngày:       {sorted(df['day'].unique().to_list())}")

    # 1. Khoá danh tính duy nhất
    dup = df.select(["DomainName", "UserName", "day"]).is_duplicated().sum()
    n_id = df.select(["DomainName", "UserName"]).unique().height
    rep.check("1", "Khoá (DomainName, UserName, day) duy nhất", dup == 0,
              f"{dup} dòng trùng khoá | {n_id:,} danh tính")

    # 2. Không còn cột hằng số / toàn null
    constant = [c for c in features
                if df.select(pl.col(c).drop_nulls().n_unique()).item() <= 1]
    rep.check("2", "Không còn cột hằng số/toàn null", len(constant) == 0,
              f"cột vi phạm: {constant or 'không có'}")

    # 3. Đa cộng tuyến
    pdf = df.select(features).to_pandas().replace([np.inf, -np.inf], np.nan)
    pdf_c = pdf.fillna(0.0)
    corr_p, corr_s = pdf_c.corr(method="pearson"), pdf_c.corr(method="spearman")
    both = pd.concat([corr_p.abs().stack(), corr_s.abs().stack()], axis=1).max(axis=1)
    pairs = both[(both >= RHO_THRESHOLD)
                 & (both.index.get_level_values(0) < both.index.get_level_values(1))]
    vif = vif_table(pdf_c)
    rep.check("3", f"Không có cặp |rho| >= {RHO_THRESHOLD}", len(pairs) == 0,
              f"{len(pairs)} cặp vi phạm")
    for (a, b), v in pairs.sort_values(ascending=False).items():
        rep.info(f"        - {a} <-> {b}: {v:.4f}")
    rep.check("3b", f"VIF lớn nhất < {VIF_THRESHOLD}", float(vif.max()) < VIF_THRESHOLD,
              f"VIF max = {vif.max():.2f} ({vif.index[0]})")
    rep.info("        Top VIF:")
    rep.info("        " + vif.head(5).round(2).to_frame("VIF").to_string().replace("\n", "\n        "))

    # 4. Không còn artefact fill 86400
    if "interarrival_dt_mean" in df.columns:
        filled = df.filter(pl.col("interarrival_dt_mean") >= 86399).height
        bad = df.filter(pl.col("interarrival_dt_mean").is_null()
                        & (pl.col("is_single_event") == 0)).height
        n_null = df["interarrival_dt_mean"].null_count()
        rep.check("4", "Không còn fill_null(86400) & NULL đúng chỗ", filled == 0 and bad == 0,
                  f"{filled} dòng >= 86399 | {n_null:,} NULL (sai chỗ: {bad})")

    # 5. Tỷ lệ nằm trong [0, 1]
    ratio_bad = []
    for c in features:
        if c.endswith("_ratio") or c.endswith("_share"):
            col = df[c].drop_nulls()
            if len(col) and (float(col.min()) < -1e-9 or float(col.max()) > 1 + 1e-9):
                ratio_bad.append(f"{c} ∈ [{col.min():.4f}, {col.max():.4f}]")
    rep.check("5", "Các cột ratio/share nằm trong [0, 1]", len(ratio_bad) == 0,
              "; ".join(ratio_bad) or "tất cả hợp lệ")

    # 6. failure_*_share chỉ xác định khi có thất bại
    bad_share = 0
    for c in NULLABLE_WHEN_NO_FAILURE:
        if c in df.columns:
            bad_share += df.filter((pl.col("failure_ratio") == 0) & pl.col(c).is_not_null()).height
            bad_share += df.filter((pl.col("failure_ratio") > 0) & pl.col(c).is_null()).height
    rep.check("6", "failure_*_share chỉ xác định khi có thất bại", bad_share == 0,
              f"{bad_share} dòng vi phạm")

    # 7. entity_type tường minh (không còn nhánh 'Other' chưa phân loại)
    et = df.group_by("entity_type").agg(
        pl.len().alias("rows"), pl.col("UserName").n_unique().alias("accounts")
    ).sort("rows", descending=True)
    labels = set(et["entity_type"].to_list())
    rep.check("7", "entity_type đã phân loại hết", "Other" not in labels,
              f"nhãn: {sorted(labels)}")
    rep.info("        " + et.to_pandas().to_string(index=False).replace("\n", "\n        "))

    # 8. Đối chiếu với hợp đồng schema (nếu có file)
    if args.schema.exists() and yaml is not None:
        schema = yaml.safe_load(args.schema.read_text(encoding="utf-8")) or {}
        declared = [f["name"] for f in schema.get("features", [])]
        removed = [f["name"] for f in schema.get("removed", [])]
        miss = sorted(set(declared) - set(features))
        extra = sorted(set(features) - set(declared))
        leaked = sorted(set(removed) & set(features))
        rep.check("8", "Ma trận khớp hợp đồng schema", not (miss or extra or leaked),
                  f"thiếu={miss} | thừa={extra} | đã loại nhưng vẫn có={leaked}")
    else:
        rep.info(f"[SKIP] Không đọc được schema ({args.schema}) hoặc thiếu PyYAML -> bỏ qua mục 8.")

    # Tổng quan phân phối
    rep.section("TỔNG QUAN PHÂN PHỐI")
    rows = []
    for c in features:
        s = df[c]
        nn = s.null_count() < df.height
        rows.append({
            "feature": c,
            "null_%": round(s.null_count() / df.height * 100, 3),
            "zero_%": round((s == 0).sum() / df.height * 100, 2),
            "min": round(float(s.min()), 4) if nn else None,
            "median": round(float(s.median()), 4) if nn else None,
            "max": round(float(s.max()), 3) if nn else None,
            "skew": round(float(s.drop_nulls().to_pandas().skew()), 2) if nn else None,
        })
    rep.info(pd.DataFrame(rows).to_string(index=False))

    rep.section("KẾT LUẬN")
    total = 9
    print(f"{'✅ TẤT CẢ KIỂM TRA PASS' if rep.failed == 0 else f'❌ CÓ {rep.failed} KIỂM TRA FAIL'}"
          f" ({total - rep.failed}/{total})")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
