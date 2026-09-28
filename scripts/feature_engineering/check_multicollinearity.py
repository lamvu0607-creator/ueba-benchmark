"""Kiểm tra đa cộng tuyến (multicollinearity) cho ma trận đặc trưng UEBA.

Cách dùng:
    python scripts/feature_engineering/check_multicollinearity.py
    python scripts/feature_engineering/check_multicollinearity.py --matrix data/features/raw/feature_matrix_raw.parquet --threshold 0.85

Kiểm tra 3 lớp:
    1. Cặp Spearman |rho| >= threshold  (bền với đuôi dài, bất biến khi log1p)
    2. Cặp Pearson  |r|   >= threshold  (đối chiếu, dễ bị thổi phồng bởi outlier)
    3. VIF - Variance Inflation Factor  (ngưỡng cảnh báo > 5, nghiêm trọng > 10)

Đầu ra:
    - docs/feature_engineering/tables/multicollinearity_vif.csv
    - In bảng tóm tắt ra console.
Exit code 0 nếu PASS (không cặp nào vượt ngưỡng tương quan và VIF max <= 10), 1 nếu FAIL.
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
META_COLS = {"DomainName", "UserName", "day", "entity_type", "Day", "Time"}
VIF_WARN = 5.0
VIF_SEVERE = 10.0


def load_features(matrix_path: Path, encode_type: bool = False) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Đọc ma trận, tách cột đặc trưng số và loại cột hằng số."""
    df = pl.read_parquet(matrix_path).to_pandas()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in META_COLS]
    std = df[num_cols].std()
    const_cols = std[std == 0].index.tolist()
    feats = [c for c in num_cols if c not in const_cols]
    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, feats, const_cols


def upper_triangle_pairs(corr: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Trải ma trận tương quan thành danh sách cặp (chỉ nửa trên để tránh trùng lặp)."""
    cols = list(corr.columns)
    records = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            records.append(
                {"Feature_1": cols[i], "Feature_2": cols[j], value_name: corr.iloc[j, i]}
            )
    out = pd.DataFrame(records)
    out["Abs"] = out[value_name].abs()
    return out.sort_values("Abs", ascending=False, na_position="last").reset_index(drop=True)


def compute_vif(corr_pearson: pd.DataFrame) -> pd.Series:
    """VIF = đường chéo của nghịch đảo ma trận tương quan Pearson (dùng pinv cho an toàn số học)."""
    c = corr_pearson.to_numpy().copy()
    np.fill_diagonal(c, 1.0)
    inv = np.linalg.pinv(c)
    return pd.Series(np.diag(inv), index=corr_pearson.columns).sort_values(ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra đa cộng tuyến cho ma trận đặc trưng UEBA")
    parser.add_argument(
        "--matrix",
        "--data-path",
        dest="matrix",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "raw" / "feature_matrix_raw.parquet",
        help="Đường dẫn ma trận parquet (alias --data-path dùng cho run_all_feature_plots.py)",
    )
    parser.add_argument("--threshold", type=float, default=0.85, help="Ngưỡng |rho| coi là đa cộng tuyến")
    parser.add_argument("--top", type=int, default=10, help="Số cặp mạnh nhất in ra console")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO_ROOT / "docs" / "feature_engineering" / "tables" / "multicollinearity_vif.csv",
    )
    args = parser.parse_args()

    if not args.matrix.exists():
        print(f"[FAIL] Không tìm thấy ma trận: {args.matrix}")
        return 1

    print("=" * 96)
    print("KIỂM TRA ĐA CỘNG TUYẾN (MULTICOLLINEARITY) CHO MA TRẬN ĐẶC TRƯNG")
    print("=" * 96)
    print(f"[INFO] Ma trận: {args.matrix}")

    df_shape = pl.read_parquet(args.matrix, columns=["day"]).height
    n_days = pl.read_parquet(args.matrix, columns=["day"])["day"].n_unique()
    X, feats, const_cols = load_features(args.matrix)
    print(f"[INFO] Kích thước: {df_shape:,} dòng × {n_days} ngày | đặc trưng số: {len(feats)}")
    if const_cols:
        print(f"[WARN] Cột hằng số (loại bỏ, không tính tương quan): {const_cols}")

    corr_s = X.corr(method="spearman")
    corr_p = X.corr(method="pearson")
    pairs_s = upper_triangle_pairs(corr_s, "Spearman_rho")
    pairs_p = upper_triangle_pairs(corr_p, "Pearson_r")

    # --- Lớp 1 & 2: cặp vượt ngưỡng -------------------------------------------------
    hi_s = pairs_s[pairs_s["Abs"] >= args.threshold].drop(columns=["Abs"])
    hi_p = pairs_p[pairs_p["Abs"] >= args.threshold].drop(columns=["Abs"])
    n_pairs = len(pairs_s)

    print("\n" + "-" * 96)
    print(f"[1] CẶP |SPEARMAN rho| >= {args.threshold}: {len(hi_s)}/{n_pairs} cặp")
    print("-" * 96)
    if hi_s.empty:
        print("    (không có cặp nào -> dữ liệu không bị đa cộng tuyến theo Spearman)")
    else:
        print(hi_s.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "-" * 96)
    print(f"[2] CẶP |PEARSON r| >= {args.threshold}: {len(hi_p)}/{n_pairs} cặp (đối chiếu)")
    print("-" * 96)
    if hi_p.empty:
        print("    (không có cặp nào)")
    else:
        print(hi_p.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # --- Lớp 3: VIF -----------------------------------------------------------------
    vif = compute_vif(corr_p)
    vif_df = pd.DataFrame({"Feature": vif.index, "VIF": vif.to_numpy()})
    vif_df["Muc_do"] = np.where(
        vif_df["VIF"] > VIF_SEVERE, "NGHIEM_TRONG",
        np.where(vif_df["VIF"] > VIF_WARN, "CANH_BAO", "BINH_THUONG"),
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    vif_df.to_csv(args.out_csv, index=False, float_format="%.4f")

    print("\n" + "-" * 96)
    print("[3] VIF (Variance Inflation Factor) — tính trên ma trận Pearson")
    print("-" * 96)
    print(vif_df.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("-" * 96)
    print(f"[INFO] VIF max = {vif.max():.2f} ({vif.index[0]}) | VIF > {VIF_WARN:.0f}: "
          f"{(vif > VIF_WARN).sum()}/{len(vif)} | VIF > {VIF_SEVERE:.0f}: {(vif > VIF_SEVERE).sum()}/{len(vif)}")
    print(f"[✓] Đã lưu bảng VIF: {args.out_csv.resolve()}")

    # --- Thông tin tham khảo --------------------------------------------------------
    mean_abs = (
        corr_s.abs().where(~np.eye(len(feats), dtype=bool)).mean().sort_values(ascending=False)
    )
    print("\n" + "-" * 96)
    print(f"[4] TOP {args.top} CẶP |SPEARMAN| MẠNH NHẤT")
    print("-" * 96)
    print(pairs_s.head(args.top)[["Feature_1", "Feature_2", "Spearman_rho"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n" + "-" * 96)
    print("[5] ĐỘ LIÊN KẾT TRUNG BÌNH mean |rho| (cao = dư thừa thông tin)")
    print("-" * 96)
    print(mean_abs.head(args.top).to_string(float_format=lambda v: f"{v:.3f}"))

    n_nan = int(pairs_s["Spearman_rho"].isna().sum())
    if n_nan:
        nan_pairs = pairs_s[pairs_s["Spearman_rho"].isna()][["Feature_1", "Feature_2"]]
        print(f"\n[WARN] {n_nan} cặp không tính được Spearman (một biến hằng số trên tập hợp lệ):")
        print(nan_pairs.to_string(index=False))

    # --- Kết luận -------------------------------------------------------------------
    ok_pairs = len(hi_s) == 0 and len(hi_p) == 0
    ok_vif = vif.max() <= VIF_SEVERE
    print("\n" + "=" * 96)
    if ok_pairs and ok_vif:
        print("[PASS] Không phát hiện đa cộng tuyến nghiêm trọng."
              f" Không cặp nào |rho| >= {args.threshold}; VIF max = {vif.max():.2f} <= {VIF_SEVERE:.0f}.")
    else:
        print("[FAIL] Có dấu hiệu đa cộng tuyến cần xử lý "
              f"(cặp vượt ngưỡng: {len(hi_s)} Spearman / {len(hi_p)} Pearson; VIF max = {vif.max():.2f}).")
    print("=" * 96)
    return 0 if (ok_pairs and ok_vif) else 1


if __name__ == "__main__":
    raise SystemExit(main())
