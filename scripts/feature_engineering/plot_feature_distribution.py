"""
plot_feature_distribution.py

Trực quan hóa và xuất các hình ảnh riêng biệt cho **4 đặc trưng lệch nặng nhất**
(số liệu skew thật trên 60 ngày, `data/features/raw/distribution_stats_raw.csv`):
1. rare_logon_type_count   (skew 151,66)
2. distinct_sources_count  (skew 147,39)
3. total_logons            (skew  94,52)
4. distinct_hosts          (skew  91,12)

Mục đích: chứng minh dữ liệu bị LỆCH PHÂN PHỐI và việc CHUẨN HÓA LOG(1+x) có tác dụng.
- Input là bản THÔ (raw matrix — KHÔNG chứa cột log); bản log được suy ra bằng log1p.
- Nếu input có sẵn cột log thì dùng luôn, và tái lập bản thô bằng expm1.

Bố cục mỗi ảnh (3 panel):
- Panel 1: Thẻ thống kê (Mean, Median, P99, Max, tỷ lệ = 0) + bằng chứng giảm lệch
           (Skew gốc vs sau log) và Kurtosis sau log — giải thích vì sao ảnh vẫn
           "trông lệch" dù skew đã giảm 98-99% (đuôi dày / gai zero-inflation).
- Panel 2: Toàn bộ dải sau log(1 + x) (Histogram + KDE) kèm chú thích gai 0 và đuôi ngoài khung.
- Panel 3: Zoom vùng chứa 99% dữ liệu (x <= P99) — nơi thấy rõ hình dạng khối chính.

Đầu vào (mặc định):
  - data/features/raw/feature_matrix_raw.parquet   (ma trận THÔ, 16 đặc trưng, không có cột log)

Đầu ra (giữ nguyên):
  - docs/feature_engineering/figures/distribution_<feature_name>.png (các ảnh độc lập)
  - artifacts/distribution_<feature_name>.png (các ảnh độc lập)
  - docs/feature_engineering/tables/distribution_skewness_comparison.csv
  - Bảng tự kiểm ở cuối log: mỗi ảnh phải tồn tại ở CẢ 2 thư mục; thiếu ảnh -> exit code 1
    (chạy KHÔNG tham số = đủ 4 ảnh; nếu dùng `--features`, chỉ các cột được liệt kê mới sinh ảnh)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Đảm bảo in UTF-8 trên Windows console
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
from scipy import stats
import seaborn as sns

# ==============================================================================
# 1. THIẾT LẬP ĐƯỜNG DẪN THƯ MỤC
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts"
OUTPUT_FIG_DIR = BASE_DIR / "docs" / "feature_engineering" / "figures"
OUTPUT_TAB_DIR = BASE_DIR / "docs" / "feature_engineering" / "tables"

OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TAB_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# ĐẦU VÀO: 4 ĐẶC TRƯNG LỆCH NẶNG NHẤT (xếp theo skew giảm dần)
#   Nguồn số liệu: data/features/raw/distribution_stats_raw.csv (60 ngày, |skew| giảm dần)
#   Đây là 4 cột sẽ được chuẩn hóa log(1+x); script chứng minh dữ liệu bị lệch
#   và việc nén log có tác dụng. Tên cột là tên trong ma trận THÔ (không có cột log).
# ==============================================================================
DEFAULT_TARGET_FEATURES = [
    "rare_logon_type_count",    # skew 151,66  (hạng 1)
    "distinct_sources_count",   # skew 147,39  (hạng 2)
    "total_logons",             # skew  94,52  (hạng 3)
    "distinct_hosts",           # skew  91,12  (hạng 4)
]

# Định nghĩa ánh xạ cặp (bản thô <-> bản log) và các tên gọi dự phòng.
#   - `output_id`: tên dùng đặt file ảnh & cột `Feature` trong CSV (GIỮ NGUYÊN như trước).
#   - `raw_candidates` / `log_candidates`: tên cột có thể có trong dữ liệu đầu vào.
FEATURE_PAIR_MAPPINGS = [
    {
        "canonical_name": "rare_logon_type_count",
        "output_id": "rare_logon_type_count_log",
        "raw_candidates": ["rare_logon_type_count", "rare_logon_count"],
        "log_candidates": ["rare_logon_type_count_log", "log_rare_logon_type_count", "rare_logons_log"],
        "display_title": "Số lần đăng nhập loại hiếm (Rare Logon Types)",
    },
    {
        "canonical_name": "distinct_sources_count",
        "output_id": "distinct_sources_count",
        "raw_candidates": ["distinct_sources_count", "distinct_sources", "source_count"],
        "log_candidates": ["log_distinct_sources_count", "log_distinct_sources"],
        "display_title": "Số lượng máy nguồn phân biệt (Distinct Sources)",
    },
    {
        "canonical_name": "total_logons",          # skew 94,52 (hạng 3)
        "output_id": "log_total_logons",
        "raw_candidates": ["total_logons", "total_events", "event_count"],
        "log_candidates": ["log_total_logons", "log1p_total_logons"],
        "display_title": "Tổng số sự kiện đăng nhập (Total Logons)",
    },
    {
        "canonical_name": "distinct_hosts",        # skew 91,12 (hạng 4)
        "output_id": "log_distinct_hosts",
        "raw_candidates": ["distinct_hosts", "distinct_hosts_count", "distinct_destinations"],
        "log_candidates": ["log_distinct_hosts", "distinct_hosts_log"],
        "display_title": "Số lượng máy đích phân biệt (Distinct Hosts)",
    },
]


def resolve_series_pair(
    df: pd.DataFrame, mapping: Dict[str, Any]
) -> Optional[Tuple[str, pd.Series, pd.Series, str]]:
    """
    Xác định cặp (s_raw, s_log) từ DataFrame.
    Nếu chỉ có bản log: Tái lập bản thô bằng expm1(log_val).
    Nếu chỉ có bản thô: Tạo bản log bằng log1p(raw_val).
    """
    raw_col = next((c for c in mapping["raw_candidates"] if c in df.columns), None)
    log_col = next((c for c in mapping["log_candidates"] if c in df.columns), None)

    if not raw_col and not log_col:
        return None

    if raw_col and log_col:
        s_raw = df[raw_col].fillna(0).astype(float)
        s_log = df[log_col].fillna(0).astype(float)
    elif log_col and not raw_col:
        s_log = df[log_col].fillna(0).astype(float)
        # Tái lập giá trị thô từ log: raw = exp(log) - 1
        s_raw = np.expm1(np.maximum(0.0, s_log))
    else:  # raw_col and not log_col
        s_raw = df[raw_col].fillna(0).astype(float)
        # Nén log từ bản thô: log = log(1 + raw)
        s_log = np.log1p(np.maximum(0.0, s_raw))

    # Tên dùng đặt file ảnh & cột `Feature` trong CSV: ưu tiên `output_id`
    # (giữ nguyên tên đầu ra như trước khi đổi input).
    feature_id = str(mapping.get("output_id") or log_col or raw_col)

    return feature_id, s_raw, s_log, mapping["display_title"]


def compute_metrics(series: pd.Series) -> Dict[str, float]:
    """Tính các chỉ số phân vị và mô tả hình thái phân phối (skew + kurtosis + tỷ lệ 0)."""
    s_clean = series.dropna().astype(float)
    if len(s_clean) == 0:
        return {"mean": 0.0, "median": 0.0, "p99": 0.0, "max": 0.0, "skew": 0.0,
                "kurtosis": 0.0, "zero_pct": 0.0, "n": 0}

    return {
        "mean": float(s_clean.mean()),
        "median": float(s_clean.median()),
        "p99": float(np.percentile(s_clean, 99)),
        "max": float(s_clean.max()),
        "skew": float(stats.skew(s_clean, bias=False)),
        "kurtosis": float(stats.kurtosis(s_clean, bias=False)),
        "zero_pct": float((s_clean == 0).mean() * 100),
        "n": int(len(s_clean)),
    }


def generate_single_feature_plot(
    feature_id: str,
    title_text: str,
    s_raw: pd.Series,
    s_log: pd.Series,
    output_fig_dir: Path,
    artifact_dir: Path,
    bins: int = 35,
) -> Dict[str, Any]:
    """Tạo và lưu 1 biểu đồ độc lập gồm thẻ thống kê bên trái và đồ thị log bên phải."""
    m_raw = compute_metrics(s_raw)
    m_log = compute_metrics(s_log)

    # Tính tỷ lệ giảm độ lệch Skewness
    skew_reduction = 0.0
    if abs(m_raw["skew"]) > 1e-4:
        skew_reduction = (abs(m_raw["skew"]) - abs(m_log["skew"])) / abs(m_raw["skew"]) * 100

    # Cấu hình Figure: 1 hàng × 3 cột = [thẻ thống kê | toàn bộ dải | zoom 99%]
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18.5, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.5, 1.5]},
    )

    ax_left, ax_full, ax_zoom = axes[0], axes[1], axes[2]

    # =========================================================================
    # BÊN TRÁI: THỐNG KÊ GỌN GÀNG (MEAN, MEDIAN, P99, MAX & BẰNG CHỨNG GIẢM LỆCH)
    # =========================================================================
    ax_left.axis("off")

    stats_box_text = (
        f"ĐẶC TRƯNG: {feature_id}\n"
        f"Mô tả: {title_text}\n"
        f"─────────────────────────────────────\n"
        f"• Mean        : {m_raw['mean']:>14,.2f}\n"
        f"• Median (P50): {m_raw['median']:>14,.2f}\n"
        f"• P99         : {m_raw['p99']:>14,.2f}\n"
        f"• Max         : {m_raw['max']:>14,.2f}\n"
        f"• Tỷ lệ = 0    : {m_log['zero_pct']:>13.2f}%\n"
        f"─────────────────────────────────────\n"
        f"• Skewness gốc : {m_raw['skew']:>13.2f} (Lệch nặng)\n"
        f"• Skewness log : {m_log['skew']:>13.2f} (Gần đối xứng)\n"
        f"• Kurtosis log : {m_log['kurtosis']:>13.2f} (Đuôi dày)\n"
        f"• Mức giảm lệch: {skew_reduction:>12.1f}%"
    )

    ax_left.text(
        0.05, 0.50,
        stats_box_text,
        transform=ax_left.transAxes,
        fontsize=10.5,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=1.2,
        ),
    )

    # =========================================================================
    # CHUẨN BỊ DỮ LIỆU LOG & CÁC MỐC PHÂN VỊ
    # =========================================================================
    v_log = s_log.dropna().astype(float).to_numpy()
    p99_log = float(np.percentile(v_log, 99))
    max_log = float(v_log.max())
    v_zoom = v_log[v_log <= p99_log]
    skew_zoom = float(stats.skew(v_zoom, bias=False))
    kurt_zoom = float(stats.kurtosis(v_zoom, bias=False))

    # =========================================================================
    # PANEL 2: TOÀN BỘ DẢI (kèm chú thích gai 0 và đuôi ngoài khung)
    # =========================================================================
    sns.histplot(x=v_log, bins=bins, stat="density", color="#1B7F53",
                 edgecolor="white", linewidth=0.4, alpha=0.70, ax=ax_full)
    sns.kdeplot(x=v_log, color="#0E5C3A", linewidth=1.5, ax=ax_full)

    ax_full.set_title(
        f"Toàn bộ dải sau log(1 + x) — N = {len(v_log):,}\n"
        f"skew = {m_log['skew']:.2f} | kurtosis = {m_log['kurtosis']:.2f}",
        fontsize=11.5, fontweight="bold", pad=10, color="#173F5F",
    )
    ax_full.set_xlabel(f"log(1 + x) của {feature_id}", fontsize=10.5)
    ax_full.set_ylabel("Mật độ xác suất (Density)", fontsize=10.5)

    if m_log["zero_pct"] >= 1.0:
        ax_full.annotate(
            f"{m_log['zero_pct']:.1f}% giá trị = 0\n(gai zero-inflation)",
            xy=(0.0, 0.0), xycoords="data",
            xytext=(0.06, 0.72), textcoords="axes fraction",
            fontsize=9.5, color="#8A2B2B", ha="left",
            arrowprops=dict(arrowstyle="->", color="#8A2B2B", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FDECEA", edgecolor="#E8A9A2", linewidth=0.9),
        )
    if max_log > p99_log * 1.35:
        ax_full.annotate(
            f"đuôi dài tới Max log = {max_log:.2f}\n(ngoài khung zoom bên cạnh)",
            xy=(max_log, 0.0), xycoords="data",
            xytext=(0.52, 0.52), textcoords="axes fraction",
            fontsize=9.5, color="#173F5F", ha="left",
            arrowprops=dict(arrowstyle="->", color="#173F5F", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#EAF2F8", edgecolor="#A9C4DA", linewidth=0.9),
        )

    # =========================================================================
    # PANEL 3: ZOOM 99% DỮ LIỆU (nhìn đúng "hình dạng thật" của khối chính)
    # =========================================================================
    sns.histplot(x=v_zoom, bins=max(12, bins - 12), stat="density", color="#2E86AB",
                 edgecolor="white", linewidth=0.4, alpha=0.70, ax=ax_zoom)
    sns.kdeplot(x=v_zoom, color="#1B4F72", linewidth=1.6, ax=ax_zoom)
    med_zoom = float(np.median(v_zoom))
    ax_zoom.axvline(med_zoom, color="#D9534F", linestyle="--", linewidth=1.5,
                    label=f"Median = {med_zoom:.2f}")
    ax_zoom.axvline(float(v_zoom.mean()), color="#F0AD4E", linestyle=":", linewidth=1.6,
                    label=f"Mean = {v_zoom.mean():.2f}")
    ax_zoom.set_title(
        f"Zoom vùng chứa 99% dữ liệu (x ≤ P99 = {p99_log:.2f})\n"
        f"skew = {skew_zoom:.2f} | kurtosis = {kurt_zoom:.2f}",
        fontsize=11.5, fontweight="bold", pad=10, color="#173F5F",
    )
    ax_zoom.set_xlabel(f"log(1 + x) của {feature_id} (đã zoom)", fontsize=10.5)
    ax_zoom.set_ylabel("Mật độ xác suất (Density)", fontsize=10.5)
    ax_zoom.legend(loc="upper right", fontsize=9, framealpha=0.85)

    ax_full.grid(alpha=0.35)

    ax_zoom.grid(alpha=0.35)

    plt.tight_layout()

    # Tên file ảnh cho từng đặc trưng
    clean_id = feature_id.replace("/", "_")
    fig_filename = f"distribution_{clean_id}.png"
    fig_path = output_fig_dir / fig_filename
    artifact_path = artifact_dir / fig_filename

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(artifact_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] Đã tạo ảnh: {fig_path.name}")

    return {
        "Feature": feature_id,
        "Description": title_text,
        "Raw_Mean": round(m_raw["mean"], 2),
        "Raw_Median": round(m_raw["median"], 2),
        "Raw_P99": round(m_raw["p99"], 2),
        "Raw_Max": round(m_raw["max"], 2),
        "Zero_%": round(m_log["zero_pct"], 2),
        "Raw_Skewness": round(m_raw["skew"], 2),
        "Raw_Kurtosis": round(m_raw["kurtosis"], 2),
        "Log_Skewness": round(m_log["skew"], 2),
        "Log_Kurtosis": round(m_log["kurtosis"], 2),
        "Skewness_Reduction_Pct": round(skew_reduction, 1),
    }


def generate_distribution_report(
    data_path: Path,
    output_fig_dir: Path,
    output_tab_dir: Path,
    features: Optional[List[str]] = None,
    bins: int = 35,
) -> None:
    """Tạo các file ảnh riêng biệt và xuất bảng thống kê CSV."""
    if not data_path.exists():
        print(f"[ERROR] Không tìm thấy file dữ liệu tại: {data_path.resolve()}")
        sys.exit(1)

    print(f"[INFO] Nạp dữ liệu từ: {data_path.resolve()}")
    df_raw = pl.read_parquet(data_path).to_pandas()
    print(f"[INFO] Kích thước dữ liệu: {len(df_raw):,} dòng")

    # Cấu hình giao diện chuẩn
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    summary_records: List[Dict[str, Any]] = []

    # 1. Chạy trên các cặp đặc trưng trọng yếu mặc định
    for mapping in FEATURE_PAIR_MAPPINGS:
        if features and not any(
            f in mapping["raw_candidates"] or f in mapping["log_candidates"]
            for f in features
        ):
            continue

        pair_result = resolve_series_pair(df_raw, mapping)
        if pair_result is None:
            continue

        feat_id, s_raw, s_log, display_title = pair_result
        res = generate_single_feature_plot(
            feature_id=feat_id,
            title_text=display_title,
            s_raw=s_raw,
            s_log=s_log,
            output_fig_dir=output_fig_dir,
            artifact_dir=ARTIFACT_DIR,
            bins=bins,
        )
        summary_records.append(res)

    # 2. Xử lý các cột tùy biến nếu người dùng truyền danh sách cột riêng qua --features
    #    (bỏ qua các cột đã được xử lý ở phần ánh xạ cặp bên trên để KHÔNG tạo ảnh trùng)
    if features:
        handled_names = {r["Feature"] for r in summary_records}
        for m in FEATURE_PAIR_MAPPINGS:
            handled_names.update(m["raw_candidates"])
            handled_names.update(m["log_candidates"])
        for custom_feat in features:
            if custom_feat in df_raw.columns and custom_feat not in handled_names:
                s_col = df_raw[custom_feat].fillna(0).astype(float)
                # Tự suy luận bản log hoặc thô
                if "log" in custom_feat.lower():
                    s_log = s_col
                    s_raw = np.expm1(np.maximum(0.0, s_log))
                else:
                    s_raw = s_col
                    s_log = np.log1p(np.maximum(0.0, s_raw))

                res = generate_single_feature_plot(
                    feature_id=custom_feat,
                    title_text=f"Tùy biến: {custom_feat}",
                    s_raw=s_raw,
                    s_log=s_log,
                    output_fig_dir=output_fig_dir,
                    artifact_dir=ARTIFACT_DIR,
                    bins=bins,
                )
                summary_records.append(res)

    if not summary_records:
        print("[ERROR] Không tìm thấy đặc trưng nào khớp để xử lý.")
        return

    # Xuất bảng số liệu CSV
    df_metrics = pd.DataFrame(summary_records)
    csv_path = output_tab_dir / "distribution_skewness_comparison.csv"
    df_metrics.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"\n[✓] Đã lưu bảng thống kê Skewness tại: {csv_path.resolve()}\n")

    print("=" * 95)
    print("BẢNG TỔNG HỢP HIỆU QUẢ GIẢM ĐỘ LỆCH SKEWNESS TRƯỚC VÀ SAU KHI NÉN LOG:")
    print("=" * 95)
    print(
        df_metrics[
            [
                "Feature",
                "Raw_Median",
                "Raw_P99",
                "Raw_Max",
                "Raw_Skewness",
                "Log_Skewness",
                "Skewness_Reduction_Pct",
            ]
        ].to_string(index=False)
    )
    print("=" * 95 + "\n")

    # ==========================================================================
    # TỰ KIỂM ĐẾM ẢNH: bảo đảm MỌI đặc trưng đều có ảnh ở CẢ 2 thư mục
    # (chống lỗi "chỉ thấy 3 ảnh" khi 1 file bị thiếu/ghi hụt ở docs/.../figures)
    # ==========================================================================
    print("[KIỂM TRA SẢN PHẨM ẢNH ĐÃ GHI RA ĐĨA]")
    print("-" * 95)
    missing_imgs: List[str] = []
    for rec in summary_records:
        fname = f"distribution_{str(rec['Feature']).replace('/', '_')}.png"
        ok_fig = (output_fig_dir / fname).exists()
        ok_art = (ARTIFACT_DIR / fname).exists()
        if not (ok_fig and ok_art):
            missing_imgs.append(fname)
        print(
            f"  [{'✓' if (ok_fig and ok_art) else '✗'}] {fname:<48}"
            f" | figures: {'CÓ' if ok_fig else 'THIẾU':<5} | artifacts: {'CÓ' if ok_art else 'THIẾU'}"
        )
    print("-" * 95)
    if missing_imgs:
        print(f"[FAIL] Thiếu {len(missing_imgs)} ảnh: {missing_imgs}")
        print(f"       figures = {output_fig_dir.resolve()}")
        print(f"       artifacts = {ARTIFACT_DIR.resolve()}")
        sys.exit(1)
    print(
        f"[✓] Đã sinh đủ {len(summary_records)}/{len(summary_records)} ảnh ở cả 2 thư mục:\n"
        f"    figures   = {output_fig_dir.resolve()}\n"
        f"    artifacts = {ARTIFACT_DIR.resolve()}"
    )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vẽ và xuất các ảnh phân phối riêng biệt cho từng đặc trưng đếm")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=BASE_DIR / "data" / "features" / "raw" / "feature_matrix_raw.parquet",
        help="Đường dẫn file ma trận đặc trưng THÔ (mặc định: data/features/raw/feature_matrix_raw.parquet)",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=OUTPUT_FIG_DIR,
        help="Thư mục lưu ảnh",
    )
    parser.add_argument(
        "--output-tab",
        type=Path,
        default=OUTPUT_TAB_DIR,
        help="Thư mục lưu bảng CSV",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=35,
        help="Số bin của histogram (mặc định: 35)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Danh sách cột đặc trưng muốn chạy tùy chọn. Bỏ trống = chạy đúng 4 đặc trưng "
             "lệch nặng nhất: rare_logon_type_count, distinct_sources_count, distinct_hosts, total_logons",
    )
    args = parser.parse_args()

    generate_distribution_report(
        data_path=args.data_path,
        output_fig_dir=args.output_fig,
        output_tab_dir=args.output_tab,
        features=args.features,
        bins=args.bins,
    )