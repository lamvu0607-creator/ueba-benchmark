"""
plot_feature_correlation.py

Tính toán, kiểm tra đa cộng tuyến và vẽ biểu đồ nhiệt (Heatmap)
ma trận tương quan thứ hạng Spearman cho dữ liệu UEBA (Tài khoản × Ngày).
Đáp ứng yêu cầu Feature Engineering & Validation Tuần 2.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple

# Thiết lập encoding UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "features" / "raw" / "feature_matrix_raw.parquet"  # ma trận THÔ (bước 1)

OUTPUT_FIG_DIR = BASE_DIR / "docs" / "feature_engineering" / "figures"
OUTPUT_TAB_DIR = BASE_DIR / "docs" / "feature_engineering" / "tables"
ARTIFACT_DIR = BASE_DIR / "artifacts"

OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TAB_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Ngưỡng cảnh báo đa cộng tuyến nghiêm trọng
MULTICOLLINEARITY_THRESHOLD = 0.85


# ==============================================================================
# 2. HÀM PHÂN CỤM ĐẶC TRƯNG THEO TRỤC NGHIỆP VỤ (DOMAIN SORTING)
# ==============================================================================
def sort_features_by_domain_group(cols: List[str]) -> List[str]:
    """
    Sắp xếp các cột đặc trưng theo 5 khối nghiệp vụ UEBA chuẩn.
    Việc này giúp Heatmap hiển thị thành các cụm khối (blocks) tự nhiên,
    chứng minh tính độc lập tương đối giữa các chiều hành vi.
    """
    groups = {
        "1_volume": ["total", "count", "cnt", "volume", "freq"],
        "2_failure": ["fail", "error", "bad", "reject"],
        "3_temporal": ["hour", "night", "off_hour", "work", "time", "inter_arrival", "std_dt", "skew_time"],
        "4_spatial": ["source", "dest", "host", "comp", "ip", "entropy", "distinct", "unique"],
        "5_logon_type": ["logon_type", "type_2", "type_3", "type_10", "auth_ntlm", "package", "proto"],
        "6_entity": ["machine", "is_admin", "user_role", "priv"]
    }

    assigned = {g: [] for g in groups}
    remaining = []

    for col in cols:
        matched = False
        col_lower = col.lower()
        for g_key, keywords in groups.items():
            if any(kw in col_lower for kw in keywords):
                assigned[g_key].append(col)
                matched = True
                break
        if not matched:
            remaining.append(col)

    sorted_cols = []
    for g_key in sorted(assigned.keys()):
        sorted_cols.extend(sorted(assigned[g_key]))
    sorted_cols.extend(sorted(remaining))
    return sorted_cols


# ==============================================================================
# 3. QUÉT VÀ BÁO CÁO CÁC CẶP ĐA CỘNG TUYẾN (|ρ| >= 0.85)
# ==============================================================================
def analyze_multicollinearity(corr_matrix: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Trích xuất danh sách các cặp đặc trưng có hệ số tương quan vượt ngưỡng đa cộng tuyến.
    """
    mask_upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    unstacked = corr_matrix.where(mask_upper).stack().reset_index()
    unstacked.columns = ["Feature_1", "Feature_2", "Spearman_rho"]
    unstacked["Abs_rho"] = unstacked["Spearman_rho"].abs()

    high_corr = unstacked[unstacked["Abs_rho"] >= threshold].sort_values(
        by="Abs_rho", ascending=False
    ).reset_index(drop=True)

    return high_corr


# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN WORKFLOW)
# ==============================================================================
def main():
    if not DATA_PATH.exists():
        print(f"[ERROR] Không tìm thấy file dữ liệu ma trận đặc trưng tại: {DATA_PATH.resolve()}")
        sys.exit(1)

    print(f"[INFO] Đọc dữ liệu từ: {DATA_PATH.resolve()}")
    df_raw = pl.read_parquet(DATA_PATH).to_pandas()
    print(f"[INFO] Kích thước dữ liệu: {df_raw.shape[0]:,} dòng × {df_raw.shape[1]} cột")

    # Loại bỏ các cột định danh không phải feature số
    exclude_cols = {"UserName", "day", "Day", "Time", "entity_type", "AccountName"}
    num_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    # Loại bỏ các cột có phương sai = 0 (hằng số)
    stds = df_raw[num_cols].std()
    zero_std_cols = stds[stds == 0].index.tolist()
    if zero_std_cols:
        print(f"[WARN] Bỏ qua các đặc trưng có phương sai = 0 (hằng số): {zero_std_cols}")

    active_cols = [c for c in num_cols if c not in zero_std_cols]
    
    # Sắp xếp các cột theo khối nghiệp vụ
    sorted_cols = sort_features_by_domain_group(active_cols)
    print(f"[INFO] Số lượng đặc trưng đưa vào phân tích: {len(sorted_cols)}")

    df_features = df_raw[sorted_cols]

    # --- 1. TÍNH MA TRẬN SPEARMAN RANK CORRELATION ---
    print("[INFO] Đang tính toán ma trận tương quan thứ hạng Spearman...")
    corr_spearman = df_features.corr(method="spearman")

    # Lưu bảng tương quan đầy đủ ra CSV
    spearman_csv = OUTPUT_TAB_DIR / "correlation_spearman.csv"
    corr_spearman.to_csv(spearman_csv, float_format="%.4f")
    print(f"[✓] Đã lưu bảng Spearman CSV: {spearman_csv.resolve()}")

    # --- 2. RÀ SOÁT ĐA CỘNG TUYẾN ---
    high_corr_df = analyze_multicollinearity(corr_spearman, threshold=MULTICOLLINEARITY_THRESHOLD)
    report_csv = OUTPUT_TAB_DIR / "high_multicollinearity_pairs.csv"
    high_corr_df.to_csv(report_csv, index=False, float_format="%.4f")

    print("\n" + "=" * 80)
    print(f"[RÀ SOÁT ĐA CỘNG TUYẾN] CÁC CẶP BIẾN CÓ |Spearman ρ| >= {MULTICOLLINEARITY_THRESHOLD}")
    print("=" * 80)
    if not high_corr_df.empty:
        for idx, row in high_corr_df.iterrows():
            print(f"  {idx + 1:02d}. {row['Feature_1']} <---> {row['Feature_2']}: ρ = {row['Spearman_rho']:+.4f}")
        print("-" * 80)
        print(f"[KHUYẾN CÁO] Đã xuất {len(high_corr_df)} cặp đa cộng tuyến ra: {report_csv.name}")
        print("[HÀNH ĐỘNG] Cần loại bớt 1 biến hoặc chuyển thành dạng tỉ lệ trước khi đưa vào LOF/OCSVM!")
    else:
        print(f"[✓] TUYỆT VỜI: Không có cặp đặc trưng nào vượt ngưỡng |ρ| >= {MULTICOLLINEARITY_THRESHOLD}.")
        print("    Các đặc trưng đảm bảo tính độc lập thông tin tốt.")
    print("=" * 80 + "\n")

    # --- 3. VẼ BIỂU ĐỒ HEATMAP CHUẨN BÁO CÁO KỸ THUẬT ---
    print("[INFO] Đang vẽ biểu đồ Heatmap Spearman...")
    
    # Thiết lập giao diện biểu đồ
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    # Dynamic figsize dựa trên số lượng feature
    fig_size = max(14, int(len(sorted_cols) * 0.65))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

    # Mặt nạ che nửa trên (Upper Triangle Mask)
    mask = np.triu(np.ones_like(corr_spearman, dtype=bool), k=1)

    # Dải màu phân kỳ Diverging Palette chuẩn an toàn thông tin
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Vẽ Heatmap
    sns.heatmap(
        corr_spearman,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": max(8.5, int(150 / len(sorted_cols)))},
        square=True,
        linewidths=0.6,
        linecolor="#F2F4F7",
        cbar_kws={
            "shrink": 0.75,
            "label": "Hệ số tương quan thứ hạng Spearman (ρ)",
            "pad": 0.03
        },
        ax=ax,
    )

    # Kiểm chứng: đếm số ô đã được ghi số (annot) để chắc chắn biểu đồ có số ở các ô
    n_annot = len([t for t in ax.texts if t.get_text()])
    expected = len(sorted_cols) * (len(sorted_cols) + 1) // 2  # nửa dưới + đường chéo
    print(f"[✓] Đã ghi số vào {n_annot} ô (kỳ vọng {expected} ô = nửa dưới + đường chéo)")

    # Đặt tiêu đề và căn chỉnh nhãn trục
    ax.set_title(
        "Ma Trận Tương Quan Thứ Hạng Spearman (Spearman Rank Correlation Matrix)\n"
        f"UEBA Windows Event Log (Tài khoản × Ngày, N={len(df_raw):,}, {len(sorted_cols)} Đặc trưng)",
        fontsize=15,
        fontweight="bold",
        pad=20,
        color="#173F5F"
    )

    plt.xticks(rotation=45, ha="right", fontsize=9, fontweight="medium")
    plt.yticks(rotation=0, fontsize=9, fontweight="medium")
    plt.tight_layout()

    # --- 4. XUẤT ẢNH CHẤT LƯỢNG CAO ---
    heatmap_fig_path = OUTPUT_FIG_DIR / "spearman_correlation_matrix.png"
    artifact_fig_path = ARTIFACT_DIR / "spearman_correlation_matrix.png"

    plt.savefig(heatmap_fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(artifact_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[✓] Đã lưu biểu đồ Heatmap tại: {heatmap_fig_path.resolve()}")
    print(f"[✓] Đã đồng bộ sang thư mục Artifacts: {artifact_fig_path.resolve()}")
    print("\n[SUCCESS] Hoàn thành phân tích và vẽ biểu đồ tương quan Spearman!")


if __name__ == "__main__":
    main()