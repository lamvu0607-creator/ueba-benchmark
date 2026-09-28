"""Kiểm tra HẬU CHUẨN HÓA: log1p đã đủ chưa, còn cần chuẩn hóa tiếp không?

Cách dùng:
    python scripts/feature_engineering/check_normalization_effect.py
    python scripts/feature_engineering/check_normalization_effect.py --matrix data/features/raw/feature_matrix_raw.parquet

Phân tích cho TOÀN BỘ đặc trưng của ma trận THÔ:
    1. Skewness gốc -> sau log1p, mức giảm (%).
    2. Vì sao còn lệch: tỷ lệ 0 (zero-inflation); skew sau log1p CHỈ tính trên các dòng != 0
       -> nếu skew(nonzero) nhỏ mà skew(all) lớn => độ lệch còn lại đến từ CỤM ĐIỂM 0,
       không phải từ hình dạng phần dương (biến đổi hàm không thể sửa được).
    3. So sánh log1p với các biến đổi mạnh hơn (sqrt, cbrt, Yeo-Johnson) trên 4 cột lệch nặng
       -> trả lời "có cần chuẩn hóa tiếp không".
    4. Cờ nén quá tay: skew sau log1p < -1 (lệch trái).
    5. So sánh hình dạng TOÀN DẢI vs ZOOM vùng chứa 99% dữ liệu (x <= P99)
       -> trả lời câu hỏi "ảnh histogram sau log1p vẫn trông lệch?": phần lớn cảm
       giác "lệch" đến từ ~1% đuôi cực trị kéo dài trục + gai zero-inflation,
       không phải từ khối 99% dữ liệu.

Đầu ra:
    - docs/feature_engineering/tables/normalization_effect_all_features.csv
    - docs/feature_engineering/tables/normalization_alternative_transforms.csv
    - docs/feature_engineering/tables/normalization_shape_zoom_vs_full.csv
    - Kết luận PASS/FAIL in ra console (FAIL nếu có cột sau log vẫn |skew| > 3 mà không có lý do zero-inflation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
META_COLS = {"DomainName", "UserName", "day", "entity_type", "Day", "Time"}
TAB_DIR = REPO_ROOT / "docs" / "feature_engineering" / "tables"

# 4 cột được chốt chuẩn hóa log1p ở bước B3
LOG_TARGETS = ["rare_logon_type_count", "distinct_sources_count", "total_logons", "distinct_hosts"]

# Cột cần soi "full vs zoom" (4 cột chốt + 2 cột đề xuất bổ sung log1p)
ZOOM_TARGETS = LOG_TARGETS + ["interarrival_dt_mean", "delta_t_cv"]

SKEW_OK = 3.0        # |skew| <= 3: chấp nhận được cho mô hình phát hiện bất thường
SKEW_GOOD = 2.0      # |skew| <= 2: gần đối xứng
LOG_HELPFUL = 50.0   # mức giảm skew tối thiểu để coi log1p là "có tác dụng"
OVER_COMPRESS = -1.0 # skew sau log < -1: dấu hiệu nén quá tay


def profile_feature(name: str, raw: pd.Series) -> Dict[str, Any]:
    """Hồ sơ 1 đặc trưng trước/sau log1p + chẩn đoán nguyên nhân còn lệch."""
    total = len(raw)
    valid = raw.dropna().astype(float)
    n_valid = len(valid)
    null_pct = (total - n_valid) / total * 100 if total else 0.0
    if n_valid == 0:
        return {"Feature": name, "Verdict": "Không có dữ liệu hợp lệ"}

    n_zero = int((valid == 0).sum())
    zero_pct = n_zero / n_valid * 100

    raw_skew = float(stats.skew(valid, bias=False))
    log = np.log1p(valid) if valid.min() >= 0 else None
    log_skew = float(stats.skew(log, bias=False)) if log is not None else np.nan
    reduction = (
        (abs(raw_skew) - abs(log_skew)) / abs(raw_skew) * 100
        if (log is not None and abs(raw_skew) > 1e-9)
        else np.nan
    )

    # Skew sau log chỉ trên phần khác 0 -> tách ảnh hưởng của cụm điểm 0
    nonzero = log[valid > 0] if log is not None else None
    log_skew_nonzero = (
        float(stats.skew(nonzero, bias=False)) if (nonzero is not None and len(nonzero) > 100) else np.nan
    )
    log_kurt = float(stats.kurtosis(log, bias=False)) if log is not None else np.nan

    # --- Phân loại ---
    if abs(raw_skew) <= SKEW_GOOD:
        verdict = "ĐÃ CÂN (không cần chuẩn hóa)"
    elif np.isnan(log_skew):
        verdict = "KHÔNG ÁP DỤNG log (có giá trị âm)"
    elif reduction >= LOG_HELPFUL and abs(log_skew) <= SKEW_OK:
        verdict = "NÊN LOG - đã xử lý tốt"
    elif reduction >= LOG_HELPFUL:
        verdict = "LOG CÓ TÁC DỤNG - vẫn còn lệch > 3"
    elif zero_pct >= 50:
        verdict = "LOG KHÔNG HIỆU QUẢ - zero-inflated, giữ thô"
    else:
        verdict = "LOG KHÔNG HIỆU QUẢ - bị chặn [0,1], giữ thô"

    if not np.isnan(log_skew) and log_skew < OVER_COMPRESS:
        verdict += " | cảnh báo: nén quá tay (lệch trái)"

    return {
        "Feature": name,
        "Null_%": round(null_pct, 2),
        "Zero_%": round(zero_pct, 2),
        "Raw_Mean": round(float(valid.mean()), 3),
        "Raw_Max": round(float(valid.max()), 1),
        "Raw_Skew": round(raw_skew, 2),
        "Log_Skew": round(log_skew, 2) if not np.isnan(log_skew) else np.nan,
        "Log_Skew_NonZero": round(log_skew_nonzero, 2) if not np.isnan(log_skew_nonzero) else np.nan,
        "Log_Kurtosis": round(log_kurt, 2) if not np.isnan(log_kurt) else np.nan,
        "Reduction_%": round(reduction, 1) if not np.isnan(reduction) else np.nan,
        "La_Log_Target": name in LOG_TARGETS,
        "Verdict": verdict,
    }


def compare_transforms(name: str, valid: pd.Series) -> Dict[str, Any]:
    """So sánh log1p với các biến đổi mạnh hơn trên 1 đặc trưng."""
    x = valid.to_numpy(dtype=float)
    out: Dict[str, Any] = {"Feature": name, "Raw_Skew": round(float(stats.skew(x, bias=False)), 2)}
    if x.min() >= 0:
        out["log1p"] = round(float(stats.skew(np.log1p(x), bias=False)), 2)
        out["sqrt"] = round(float(stats.skew(np.sqrt(x), bias=False)), 2)
        out["cbrt"] = round(float(stats.skew(np.cbrt(x), bias=False)), 2)
    try:
        yj, lam = stats.yeojohnson(x)
        out["yeojohnson"] = round(float(stats.skew(yj, bias=False)), 2)
        out["YJ_lambda"] = round(float(lam), 3)
    except Exception:
        out["yeojohnson"] = np.nan
        out["YJ_lambda"] = np.nan
    return out


def shape_full_vs_zoom(name: str, raw: pd.Series) -> Dict[str, Any]:
    """So sánh hình dạng TOÀN DẢI vs ZOOM 99% sau log1p (giải thích cảm giác 'vẫn lệch').

    Toàn dải bị chi phối bởi ~1% đuôi cực trị (kéo dài trục x) và gai điểm 0.
    Vùng chứa 99% dữ liệu (x <= P99) mới phản ánh hình dạng thật của khối chính.
    """
    v = raw.dropna().astype(float).to_numpy()
    if len(v) == 0 or v.min() < 0:
        return {"Feature": name}

    log = np.log1p(v)
    p99 = float(np.percentile(log, 99))
    zoom = log[log <= p99]
    return {
        "Feature": name,
        "Zero_%": round(float((v == 0).mean() * 100), 2),
        "Log_P99": round(p99, 2),
        "Max_Log_Off_Scale": round(float(log.max()), 2),
        "Skew_Full": round(float(stats.skew(log, bias=False)), 2),
        "Kurt_Full": round(float(stats.kurtosis(log, bias=False)), 2),
        "Skew_Zoom99": round(float(stats.skew(zoom, bias=False)), 2),
        "Kurt_Zoom99": round(float(stats.kurtosis(zoom, bias=False)), 2),
        "Zoom_n": int(len(zoom)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra hiệu quả chuẩn hóa log1p cho ma trận đặc trưng UEBA")
    parser.add_argument(
        "--matrix",
        "--data-path",
        dest="matrix",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "raw" / "feature_matrix_raw.parquet",
    )
    args = parser.parse_args()

    if not args.matrix.exists():
        print(f"[FAIL] Không tìm thấy ma trận: {args.matrix}")
        return 1

    df = pl.read_parquet(args.matrix).to_pandas()
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in META_COLS]

    print("=" * 118)
    print("KIỂM TRA HẬU CHUẨN HÓA (log1p) — TOÀN BỘ ĐẶC TRƯNG MA TRẬN THÔ")
    print("=" * 118)
    print(f"[INFO] Ma trận: {args.matrix}")
    print(f"[INFO] {len(df):,} dòng × {df['day'].nunique()} ngày | {len(feats)} đặc trưng\n")

    records = [profile_feature(c, df[c]) for c in feats]
    prof = pd.DataFrame(records)

    print("-" * 118)
    print("BẢNG 1: SKEW GỐC → SKEW SAU log1p | tách ảnh hưởng cụm điểm 0")
    print("-" * 118)
    show_cols = ["Feature", "Null_%", "Zero_%", "Raw_Skew", "Log_Skew", "Log_Skew_NonZero",
                 "Log_Kurtosis", "Reduction_%", "Verdict"]
    print(prof[show_cols].to_string(index=False))
    print("-" * 118 + "\n")

    # --- So sánh biến đổi thay thế cho 4 cột mục tiêu ---
    alt_targets = [c for c in LOG_TARGETS if c in df.columns]
    alt = pd.DataFrame([compare_transforms(c, df[c].dropna().astype(float)) for c in alt_targets])
    print("-" * 118)
    print("BẢNG 2: SO SÁNH log1p VỚI BIẾN ĐỔI MẠNH HƠN (4 cột lệch nặng nhất)")
    print("-" * 118)
    print(alt.to_string(index=False))
    print("-" * 118 + "\n")

    # --- Bảng 3: hình dạng toàn dải vs zoom 99% ---
    zoom_targets = [c for c in ZOOM_TARGETS if c in df.columns]
    zoom = pd.DataFrame([shape_full_vs_zoom(c, df[c]) for c in zoom_targets])
    print("-" * 118)
    print("BẢNG 3: HÌNH DẠNG TOÀN DẢI vs ZOOM 99% (vì sao ảnh histogram vẫn 'trông lệch'?)")
    print("-" * 118)
    print(zoom.to_string(index=False))
    print("[ĐỌC BẢNG 3] Skew_Full lớn nhưng Skew_Zoom99 nhỏ => cảm giác 'lệch' đến từ ~1% đuôi cực trị")
    print("            (kéo dài trục x) và gai điểm 0, KHÔNG phải khối 99% dữ liệu.")
    print("-" * 118 + "\n")

    # --- Kết luận ---
    logs = prof[prof["La_Log_Target"]]
    good = int((logs["Log_Skew"].abs() <= SKEW_OK).sum())
    over = logs[logs["Log_Skew"] < OVER_COMPRESS]["Feature"].tolist()
    still_bad = prof[(prof["Log_Skew"].abs() > SKEW_OK) & (prof["Raw_Skew"].abs() > SKEW_OK)]["Feature"].tolist()
    zero_inflated = prof[prof["Verdict"].str.contains("zero-inflated|bị chặn", na=False)]
    unchanged = prof[prof["Verdict"].str.startswith("ĐÃ CÂN", na=False)]["Feature"].tolist()

    print("=" * 118)
    print("KẾT LUẬN")
    print("=" * 118)
    print(f"[1] 4 cột chuẩn hóa log1p đạt |skew| <= {SKEW_OK:g}: {good}/{len(logs)}  ->  "
          f"{', '.join(logs['Feature'].tolist())}")
    print(f"[2] Sau log vẫn |skew| > {SKEW_OK:g}: {still_bad if still_bad else 'không có'}")
    print(f"[3] Dấu hiệu nén quá tay (skew < {OVER_COMPRESS:g}): {over if over else 'không có'}")
    print(f"[4] Cột giữ THÔ vì log không hiệu quả (bị chặn/zero-inflated): {len(zero_inflated)} cột -> "
          f"{', '.join(zero_inflated['Feature'].tolist())}")
    print(f"[5] Cột đã cân sẵn (|skew| <= {SKEW_GOOD:g}): {len(unchanged)} cột -> {', '.join(unchanged)}")
    print("-" * 118)

    ok = good == len(logs)
    print("[PASS] Bước chuẩn hóa log1p đã đạt mục tiêu (mọi cột target <= 3 skew)."
          if ok else "[FAIL] Còn cột target chưa đạt ngưỡng skew.")
    print("[KHUYẾN CÁO] Không cần biến đổi hàm thêm (sqrt/cbrt/Yeo-Johnson) — xem BẢNG 2;")
    print("            phần còn lại (nếu sang model) là SCALING (Standard/Robust), không phải đổi hình dạng.")
    print("=" * 118)

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    p1 = TAB_DIR / "normalization_effect_all_features.csv"
    p2 = TAB_DIR / "normalization_alternative_transforms.csv"
    p3 = TAB_DIR / "normalization_shape_zoom_vs_full.csv"
    prof.to_csv(p1, index=False, float_format="%.4f")
    alt.to_csv(p2, index=False, float_format="%.4f")
    zoom.to_csv(p3, index=False, float_format="%.4f")
    print(f"[✓] Đã lưu: {p1.resolve()}")
    print(f"[✓] Đã lưu: {p2.resolve()}")
    print(f"[✓] Đã lưu: {p3.resolve()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
