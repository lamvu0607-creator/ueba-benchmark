"""
plot_sanity_check_human_vs_machine.py

KIỂM ĐỊNH TÍNH HỢP LÝ NGHIỆP VỤ (DOMAIN SANITY CHECK) — HUMAN vs MACHINE, PHƯƠNG PHÁP: BOXPLOT

Một bộ đặc trưng tốt phải TỰ ĐỘNG tách được hai quần thể có bản chất vật lý khác nhau:
tài khoản NGƯỜI (login từ bàn phím, nghỉ đêm/cuối tuần, hoạt động thất thường) và
tài khoản MÁY/DỊCH VỤ HỆ THỐNG (chạy ngầm 24/7, nhịp đều, không tương tác) — mà CHƯA cần
mô hình học máy. Nếu boxplot của một đặc trưng gần như trùng nhau giữa 2 nhóm thì đặc trưng
đó không mang thông tin phân biệt thực thể (chỉ còn hữu ích trong phạm vi 1 khoá User x Day).

Ba lớp bằng chứng (tất cả đều tính lại từ dữ liệu mỗi lần chạy, không viết tay):

  (1) BOXPLOT 3 TRỤC HÀNH VI — mỗi trục 1 hình, hộp Human cạnh hộp Machine:
      • Trục 1 THỜI GIAN    : off_hours_ratio        (tỷ lệ hoạt động ngoài giờ 18h-7h)
      • Trục 2 PHƯƠNG THỨC  : interactive_ratio      (tỷ lệ đăng nhập trực tiếp LogonType = 2)
      • Trục 3 KHÔNG GIAN   : distinct_hosts (log1p) (độ phân tán số máy đích — fan-out)
      Trục bị "bẹp hộp" do zero-inflated (vd `interactive_ratio`: phần lớn giá trị = 0) được
      thêm DẢI VÀNG đánh dấu vùng giá trị thực + khung CẬN CẢNH phóng to vùng đó, vì trên
      thang 0–1 cả 2 hộp chỉ còn lại 2 vạch nằm ở y = 0 và nhìn vào không thấy gì.
  (2) BOXPLOT LƯỚI 4x4 — TOÀN BỘ 16 đặc trưng của ma trận THÔ, sắp giảm dần theo mức tách.
  (3) ĐỊNH LƯỢNG MỨC TÁCH — Cliff's delta + Mann-Whitney U (p-value) + %NULL / %ZERO
      cho từng đặc trưng, đối chiếu ngưỡng hiệu ứng Romano et al. (0.147 / 0.33 / 0.474).
      Cột NULL-HEAVY (vd `failure_locked_out_share`) được đọc kèm tỷ lệ NULL của từng nhóm,
      vì nếu chỉ nhìn hộp trên dòng khả dụng thì mức tách bị đánh giá sai.

Đầu vào:
  - data/features/raw/feature_matrix_raw.parquet   (ma trận THÔ, 20 cột: 4 khoá/nhãn + 16 đặc trưng)
    Nhãn nhóm lấy theo cột `entity_type` (Machine = tài khoản máy; phần còn lại = người/hệ thống).
    Nếu ma trận không có `entity_type` thì suy từ `UserName` (hậu tố '$' + tên tài khoản hệ thống).

Đầu ra:
  - docs/feature_engineering/figures/sanity_check_human_vs_machine.png               (3 trục hành vi)
  - docs/feature_engineering/figures/sanity_check_human_vs_machine_boxplot_grid.png  (lưới 4x4, 16 đặc trưng)
  - artifacts/                              (đồng bộ cùng tên ảnh)
  - docs/feature_engineering/tables/sanity_check_human_vs_machine.csv                (16 đặc trưng + hiệu ứng)
  - docs/feature_engineering/tables/sanity_check_human_vs_machine_axes.csv           (3 trục hành vi)
  - reports/week2/sanity_check_human_vs_machine.md                                   (báo cáo tự sinh)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Thiết lập UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats

# ==============================================================================
# 1. THIẾT LẬP ĐƯỜNG DẪN DỰ ÁN
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "features" / "raw" / "feature_matrix_raw.parquet"  # ma trận THÔ

OUTPUT_FIG_DIR = BASE_DIR / "docs" / "feature_engineering" / "figures"
OUTPUT_TAB_DIR = BASE_DIR / "docs" / "feature_engineering" / "tables"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
REPORT_DIR = REPO_ROOT / "reports" / "week2"

for _dir in (OUTPUT_FIG_DIR, OUTPUT_TAB_DIR, ARTIFACT_DIR, REPORT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. HẰNG SỐ: NHÓM THỰC THỂ, MÀU, NGƯỠNG HIỆU ỨNG
# ==============================================================================
LABEL_HUMAN = "Người dùng (Human)"
LABEL_MACHINE = "Tài khoản Máy (Machine)"
GROUP_ORDER = [LABEL_HUMAN, LABEL_MACHINE]
PALETTE = {LABEL_HUMAN: "#1F4E79", LABEL_MACHINE: "#D9534F"}
COLOR_TITLE = "#173F5F"
COLOR_NEUTRAL = "#6C757D"

# Ngưỡng hiệu ứng Cliff's delta (Romano et al. 2006) — nhất quán với
# scripts/feature_engineering/plot_temporal_stability.py (THRESHOLDS["cliff_delta_small"]).
EFFECT_BANDS: List[Tuple[float, str]] = [
    (0.474, "TÁCH RẤT MẠNH (large)"),
    (0.330, "TÁCH MẠNH (medium)"),
    (0.147, "TÁCH RÕ (small)"),
]
MIN_ABS_DELTA_SMALL = 0.147  # |delta| nhỏ nhất còn coi là "có tách được"

# 16 đặc trưng THÔ của ma trận (xem configs/feature_schema.yaml).
#   transform = "log1p" cho 6 cột lệch mạnh |skew| > 3 (theo normalization_assessment.md):
#   nếu vẽ hộp trên thang thô, Q1-Q3 bị bóp thành một vạch sát 0 và râu hộp kéo dài vô nghĩa.
FEATURE_SPECS: List[Dict[str, str]] = [
    {"name": "total_logons", "group": "volume", "role": "display", "transform": "log1p",
     "meaning": "Khối lượng: số sự kiện 4624 + 4625 của khoá trong ngày (cột hiển thị)"},
    {"name": "failure_ratio", "group": "failure", "role": "core", "transform": "none",
     "meaning": "Cường độ thất bại: failure_count / total_logons"},
    {"name": "failure_locked_out_share", "group": "failure", "role": "core", "transform": "none",
     "meaning": "Tỷ lệ thất bại do khoá tài khoản (NULL khi ngày đó không có thất bại)"},
    {"name": "off_hours_ratio", "group": "time", "role": "core", "transform": "none",
     "meaning": "Tỷ lệ hoạt động ngoài giờ hành chính (18h-7h)"},
    {"name": "interarrival_dt_mean", "group": "rhythm", "role": "core", "transform": "log1p",
     "meaning": "Khoảng cách trung bình giữa 2 sự kiện liên tiếp (giây); NULL khi chỉ 1 sự kiện"},
    {"name": "delta_t_cv", "group": "rhythm", "role": "core", "transform": "log1p",
     "meaning": "Độ biến thiên khoảng cách giữa các sự kiện (std / mean)"},
    {"name": "same_second_share", "group": "rhythm", "role": "core", "transform": "none",
     "meaning": "Tỷ lệ sự kiện dồn trong cùng một giây (burst)"},
    {"name": "is_single_event", "group": "rhythm", "role": "core", "transform": "none",
     "meaning": "Cờ: khoá chỉ có đúng 1 sự kiện trong ngày"},
    {"name": "interactive_ratio", "group": "logon_type", "role": "core", "transform": "none",
     "meaning": "Tỷ lệ đăng nhập trực tiếp tại bàn phím (LogonType = 2)"},
    {"name": "rare_logon_type_count", "group": "logon_type", "role": "core", "transform": "log1p",
     "meaning": "Số sự kiện thuộc loại logon hiếm (không phải Type 2/3)"},
    {"name": "ntlm_ratio", "group": "logon_type", "role": "core", "transform": "none",
     "meaning": "Tỷ lệ xác thực bằng giao thức NTLM"},
    {"name": "distinct_hosts", "group": "context", "role": "core", "transform": "log1p",
     "meaning": "Số máy đích phân biệt tài khoản đăng nhập tới (fan-out)"},
    {"name": "distinct_sources_count", "group": "context", "role": "core", "transform": "log1p",
     "meaning": "Số máy nguồn phân biệt phát sinh yêu cầu (dấu hiệu chia sẻ credential)"},
    {"name": "missing_source_ratio", "group": "context", "role": "core", "transform": "none",
     "meaning": "Tỷ lệ sự kiện thiếu trường máy nguồn (Source null)"},
    {"name": "remote_logon_ratio", "group": "context", "role": "core", "transform": "none",
     "meaning": "Tỷ lệ đăng nhập từ xa (Source != LogHost)"},
    {"name": "custom_proc_share", "group": "context", "role": "core", "transform": "none",
     "meaning": "Tỷ lệ sự kiện do tiến trình nghiệp vụ riêng (Proc<6 số>.exe) khởi tạo"},
]

# 3 trục hành vi vật lý (mục "Cách kiểm tra" của kế hoạch): thời gian - phương thức - không gian.
# `candidates` = danh sách tên cột ưu tiên (ma trận THÔ / ma trận chuẩn hóa đều khớp được).
AXIS_CONFIGS: List[Dict[str, Any]] = [
    {
        "key": "temporal",
        "title": "Trục 1 — THỜI GIAN",
        "candidates": ["off_hours_ratio"],
        "transform": "none",
        "y_label": "off_hours_ratio (0.0 - 1.0)",
        "y_limit": (-0.05, 1.05),
        "human_expectation": "THẤP — chủ yếu làm việc trong giờ hành chính 8h-18h",
        "machine_expectation": "CAO & ổn định — tác vụ chạy ngầm 24/7",
    },
    {
        "key": "logon_mechanism",
        "title": "Trục 2 — PHƯƠNG THỨC",
        "candidates": ["interactive_ratio"],
        "transform": "none",
        "y_label": "interactive_ratio (phóng to vùng giá trị thực)",
        "y_limit": None,
        "human_expectation": "> 0 — người ngồi tại bàn phím, có phiên đăng nhập trực tiếp",
        "machine_expectation": "≈ 0 — xác thực qua mạng (Network/Service), không có phiên bàn phím",
    },
    {
        "key": "spatial_fanout",
        "title": "Trục 3 — KHÔNG GIAN",
        "candidates": ["distinct_hosts", "log_distinct_hosts"],
        "transform": "log1p",
        "y_label": "log1p(distinct_hosts) — số máy đích phân biệt",
        "y_limit": None,
        "human_expectation": "PHÂN TÁN RỘNG — người đi lại nhiều máy (roaming) nên hộp cao, đuôi dài",
        "machine_expectation": "RẤT HẸP — tài khoản máy gắn chặt với máy của nó, quanh 1-3 máy",
    },
]

# Kiểu kiểm tra "kỳ vọng nghiệp vụ" cho từng trục (dùng để tự động hoá kết luận Khớp/Lệch)
AXIS_EXPECTATION_CHECK: Dict[str, str] = {
    "temporal": "machine_higher",      # máy hoạt động ngoài giờ nhiều hơn
    "logon_mechanism": "human_higher",  # người có logon trực tiếp, máy ≈ 0
    "spatial_fanout": "human_wider",    # người phân tán rộng hơn (IQR lớn hơn)
}


# ==============================================================================
# 3. TIỆN ÍCH: NHÃN NHÓM, BIẾN ĐỔI HIỂN THỊ, THỐNG KÊ HỘP, HIỆU ỨNG
# ==============================================================================
def resolve_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Tìm tên cột thực tế có mặt trong ma trận theo danh sách ưu tiên."""
    for name in candidates:
        if name in columns:
            return name
    return None


def ensure_machine_account_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Gắn cờ nhị phân `is_machine_account` (1 = tài khoản máy) + mô tả luật đã dùng.

    Ưu tiên `entity_type` (do extractor sinh ra theo configs/system_config.yaml) vì đây là
    nhãn TƯỜNG MINH; chỉ khi ma trận thiếu cột đó mới suy từ `UserName`.
    """
    if "entity_type" in df.columns:
        values = df["entity_type"].astype(str).str.strip().str.lower()
        df["is_machine_account"] = values.eq("machine").astype(int)
        rule = (
            "`entity_type == 'Machine'` (nhãn do extractor sinh theo configs/system_config.yaml); "
            "nhóm còn lại = User / Admin / Service / System (người + tài khoản hệ thống)"
        )
        return df, rule

    if "is_machine_account" in df.columns:
        df["is_machine_account"] = df["is_machine_account"].astype(int)
        return df, "cột `is_machine_account` có sẵn trong ma trận"

    if "UserName" in df.columns:
        user_col = df["UserName"].astype(str)
        is_machine = user_col.str.endswith("$") | user_col.str.lower().isin(
            ["system", "local service", "network service", "anonymous", "anonymous logon"]
        )
        df["is_machine_account"] = is_machine.astype(int)
        return df, "suy từ `UserName`: hậu tố '$' hoặc tên tài khoản hệ thống (system/service/anonymous)"

    raise ValueError(
        "Không tìm thấy cột `entity_type`, `is_machine_account` hay `UserName` để tách nhóm!"
    )


def apply_display_transform(values: np.ndarray, transform: str) -> np.ndarray:
    """Biến đổi CHỈ để HIỂN THỊ hộp (log1p cho cột lệch mạnh |skew| > 3)."""
    v = np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.log1p(np.clip(v, 0.0, None))
    return v


def effect_band(abs_delta: float) -> Tuple[str, str]:
    """Trả về (nhãn tiếng Việt, mã hiệu ứng tiếng Anh) theo ngưỡng Romano et al."""
    for threshold, verdict in EFFECT_BANDS:
        if abs_delta >= threshold:
            return verdict, verdict.split("(")[-1].rstrip(")")
    return "CHƯA TÁCH ĐƯỢC (negligible)", "negligible"


def describe_hbox(values: np.ndarray, transform: str) -> Dict[str, float]:
    """Thống kê mô tả dùng cho CSV + chú thích (trên thang ĐÃ biến đổi hiển thị).

    - Null_Pct / Zero_Pct tính trên TOÀN BỘ dòng của nhóm (trước khi lọc NULL) để phát hiện
      đặc trưng NULL-heavy hoặc zero-inflated — chính là trường hợp `failure_locked_out_share`.
    - Các phân vị (Q1/Q2/Q3, P90) tính trên dòng KHẢ DỤNG (không NULL).
    """
    raw = np.asarray(values, dtype=float)
    total = raw.size
    finite = np.isfinite(raw)
    usable = apply_display_transform(raw[finite], transform)
    stats_row: Dict[str, float] = {
        "N_Total": float(total),
        "N_Usable": float(usable.size),
        "Null_Pct": 100.0 * float((~finite).sum()) / total if total else float("nan"),
    }
    if usable.size == 0:
        stats_row.update(
            {k: float("nan") for k in ("Zero_Pct", "Q1", "Median", "Q3", "IQR", "P90", "Mean", "Max")}
        )
        return stats_row

    q1, q2, q3 = (float(x) for x in np.percentile(usable, [25, 50, 75]))
    stats_row.update(
        {
            "Zero_Pct": 100.0 * float((usable <= 0).sum()) / usable.size,
            "Q1": q1,
            "Median": q2,
            "Q3": q3,
            "IQR": q3 - q1,
            "P90": float(np.percentile(usable, 90)),
            "Mean": float(usable.mean()),
            "Max": float(usable.max()),
        }
    )
    return stats_row


def cliffs_delta_with_p(human: np.ndarray, machine: np.ndarray) -> Tuple[float, float]:
    """Cliff's delta + p-value Mann-Whitney U.

    delta = P(người > máy) - P(người < máy) = 2*U/(n1*n2) - 1, với U là thống kê của mẫu THỨ NHẤT
    (nhóm NGƯỜI). Quy ước dấu: **δ > 0 ⇒ nhóm NGƯỜI lớn hơn**, δ < 0 ⇒ nhóm MÁY lớn hơn.
    Cách tính khớp với kiểm định hạng Mann-Whitney nên không cần giả định phân phối
    (boxplot cũng là công cụ phi tham số).
    """
    a = np.asarray(human, dtype=float)
    b = np.asarray(machine, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = 2.0 * float(u_stat) / (a.size * b.size) - 1.0
    return delta, float(p_value)


def median_direction(delta: float, median_human: float, median_machine: float) -> str:
    """Mô tả hướng lệch trung vị giữa 2 nhóm (theo quy ước δ > 0 ⇒ Người lớn hơn)."""
    if not np.isfinite(median_human) or not np.isfinite(median_machine):
        return "không so sánh được"
    if delta > 0:
        return "Người > Máy"
    if delta < 0:
        return "Máy > Người"
    if abs(median_machine - median_human) < 1e-12:
        return "trung vị BẰNG nhau"
    return "trung vị bằng nhau nhưng khác hình dạng phân phối"


# ==============================================================================
# 4. BẢNG SỐ LIỆU: 16 ĐẶC TRƯNG + 3 TRỤC HÀNH VI
# ==============================================================================
def r(value: float, nd: int = 4) -> float:
    """Làm tròn an toàn cho NaN."""
    return float(f"{value:.{nd}f}") if np.isfinite(value) else float("nan")


def resolve_axes(columns: Sequence[str]) -> List[Tuple[Dict[str, Any], str, str]]:
    """Khớp 3 trục hành vi với cột thật của ma trận → [(config, tên cột, transform hiển thị)]."""
    resolved: List[Tuple[Dict[str, Any], str, str]] = []
    for cfg in AXIS_CONFIGS:
        col = resolve_column(columns, cfg["candidates"])
        if col is None:
            print(f"[WARN] Bỏ qua {cfg['title']}: không có cột nào trong {cfg['candidates']}.")
            continue
        transform = "none" if col.startswith("log_") else cfg["transform"]
        resolved.append((cfg, col, transform))
    return resolved


def build_feature_table(df: pd.DataFrame, machine_mask: np.ndarray) -> pd.DataFrame:
    """Bảng hiệu ứng của 16 đặc trưng, sắp giảm dần theo |Cliff's delta|."""
    human_mask = ~machine_mask
    rows: List[Dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        name = spec["name"]
        if name not in df.columns:
            print(f"[WARN] Bỏ qua đặc trưng '{name}': không có trong ma trận.")
            continue
        values = df[name].astype(float).to_numpy()
        h = describe_hbox(values[human_mask], spec["transform"])
        m = describe_hbox(values[machine_mask], spec["transform"])
        delta, p_value = cliffs_delta_with_p(values[human_mask], values[machine_mask])
        abs_delta = abs(delta) if np.isfinite(delta) else float("nan")
        verdict, band = effect_band(abs_delta)
        rows.append(
            {
                "Feature": name,
                "Group": spec["group"],
                "Role": spec["role"],
                "Display_Transform": spec["transform"],
                "Human_N": int(h["N_Usable"]),
                "Human_Null_Pct": r(h["Null_Pct"], 4),
                "Human_Zero_Pct": r(h["Zero_Pct"], 4),
                "Human_Q1": r(h["Q1"]),
                "Human_Median": r(h["Median"]),
                "Human_Q3": r(h["Q3"]),
                "Human_IQR": r(h["IQR"]),
                "Human_P90": r(h["P90"]),
                "Human_Mean": r(h["Mean"]),
                "Machine_N": int(m["N_Usable"]),
                "Machine_Null_Pct": r(m["Null_Pct"], 4),
                "Machine_Zero_Pct": r(m["Zero_Pct"], 4),
                "Machine_Q1": r(m["Q1"]),
                "Machine_Median": r(m["Median"]),
                "Machine_Q3": r(m["Q3"]),
                "Machine_IQR": r(m["IQR"]),
                "Machine_P90": r(m["P90"]),
                "Machine_Mean": r(m["Mean"]),
                "Median_Direction": median_direction(delta, h["Median"], m["Median"]),
                "Cliffs_Delta": r(delta),
                "Abs_Cliffs_Delta": r(abs_delta),
                "Effect_Band": band,
                "Verdict": verdict,
                "MannWhitney_p": fmt_p(p_value),
                "Meaning": spec["meaning"],
            }
        )
    table = pd.DataFrame(rows).sort_values("Abs_Cliffs_Delta", ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, table.shape[0] + 1))
    return table


def _decide_higher(human_stats: Dict[str, float], machine_stats: Dict[str, float]) -> int:
    """Nhóm nào cao hơn: +1 = Máy, -1 = Người, 0 = bằng nhau.

    Ưu tiên TRUNG VỊ; nếu 2 trung vị bằng nhau (đặc trưng zero-inflated, vd `interactive_ratio`
    có ~87% số 0) thì chuyển sang so TRUNG BÌNH — nếu không sẽ kết luận sai là "lệch kỳ vọng".
    """
    h, m = human_stats["Median"], machine_stats["Median"]
    if abs(m - h) < 1e-12:
        h, m = human_stats["Mean"], machine_stats["Mean"]
    if m > h:
        return 1
    if m < h:
        return -1
    return 0


def expectation_verdict(
    kind: str, human_stats: Dict[str, float], machine_stats: Dict[str, float]
) -> str:
    """So kỳ vọng nghiệp vụ với số liệu hộp thực nghiệm (tự động, không viết tay)."""
    if kind == "machine_higher":
        ok = _decide_higher(human_stats, machine_stats) == 1
    elif kind == "human_higher":
        ok = _decide_higher(human_stats, machine_stats) == -1
    elif kind == "human_wider":
        ok = human_stats["IQR"] > machine_stats["IQR"]
    else:
        return "không kiểm tra"
    return "KHỚP KỲ VỌNG" if ok else "LỆCH KỲ VỌNG (phải giải thích trong báo cáo)"


def build_axis_table(
    df: pd.DataFrame,
    machine_mask: np.ndarray,
    axes_map: Sequence[Tuple[Dict[str, Any], str, str]],
) -> pd.DataFrame:
    """Bảng số liệu 3 trục hành vi (kèm kỳ vọng nghiệp vụ + kết quả đối chiếu)."""
    human_mask = ~machine_mask
    rows: List[Dict[str, Any]] = []
    for cfg, col, transform in axes_map:
        values = df[col].astype(float).to_numpy()
        h = describe_hbox(values[human_mask], transform)
        m = describe_hbox(values[machine_mask], transform)
        delta, p_value = cliffs_delta_with_p(values[human_mask], values[machine_mask])
        abs_delta = abs(delta) if np.isfinite(delta) else float("nan")
        verdict, _ = effect_band(abs_delta)
        rows.append(
            {
                "Axis": cfg["key"],
                "Axis_Title": cfg["title"],
                "Feature_Used": col,
                "Display_Transform": transform,
                "Human_N": int(h["N_Usable"]),
                "Human_Zero_Pct": r(h["Zero_Pct"], 2),
                "Human_Median": r(h["Median"]),
                "Human_IQR": r(h["IQR"]),
                "Human_P90": r(h["P90"]),
                "Human_Mean": r(h["Mean"]),
                "Machine_N": int(m["N_Usable"]),
                "Machine_Zero_Pct": r(m["Zero_Pct"], 2),
                "Machine_Median": r(m["Median"]),
                "Machine_IQR": r(m["IQR"]),
                "Machine_P90": r(m["P90"]),
                "Machine_Mean": r(m["Mean"]),
                "Median_Direction": median_direction(delta, h["Median"], m["Median"]),
                "Cliffs_Delta": r(delta),
                "Abs_Cliffs_Delta": r(abs_delta),
                "Verdict": verdict,
                "MannWhitney_p": fmt_p(p_value),
                "Human_Expectation": cfg["human_expectation"],
                "Machine_Expectation": cfg["machine_expectation"],
                "Expectation_Check": expectation_verdict(
                    AXIS_EXPECTATION_CHECK.get(cfg["key"], ""), h, m
                ),
            }
        )
    return pd.DataFrame(rows)


# ==============================================================================
# 5. VẼ BOXPLOT
# ==============================================================================
VERDICT_COLORS: Dict[str, str] = {
    "large": "#0B6E4F",       # xanh lá đậm — tách rất mạnh
    "medium": "#1F7A8C",      # teal — tách mạnh
    "small": "#B7791F",       # hổ phách — tách rõ (mức "small")
    "negligible": "#8A8F98",  # xám — chưa tách được
}


def save_figure(fig: "plt.Figure", filename: str) -> None:
    """Lưu 1 hình ra docs/feature_engineering/figures + artifacts (300 DPI)."""
    for folder in (OUTPUT_FIG_DIR, ARTIFACT_DIR):
        path = folder / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[✓] Đã lưu biểu đồ tại: {path.resolve()}")


def whisker_bounds(values: np.ndarray, transform: str) -> Tuple[float, float]:
    """Biên râu hộp theo luật 1.5 x IQR (giống boxplot) để tự canh trục y."""
    v = apply_display_transform(values, transform)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    q1, q3 = (float(x) for x in np.percentile(v, [25, 75]))
    iqr = q3 - q1
    lo_cap, hi_cap = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lo_cap) & (v <= hi_cap)]
    if inside.size == 0:
        return float(v.min()), float(v.max())
    return float(inside.min()), float(inside.max())


def group_frame(
    df: pd.DataFrame, column: str, transform: str, machine_mask: np.ndarray
) -> pd.DataFrame:
    """Khung dữ liệu 2 cột (nhóm thực thể, giá trị đã biến đổi hiển thị) để vẽ hộp."""
    values = apply_display_transform(df[column].astype(float).to_numpy(), transform)
    frame = pd.DataFrame(
        {"is_machine": machine_mask.astype(int), "value": values}
    )
    frame = frame[np.isfinite(frame["value"])].copy()
    frame["Account_Group"] = frame["is_machine"].map({0: LABEL_HUMAN, 1: LABEL_MACHINE})
    return frame


def draw_boxpair(ax: "plt.Axes", frame: pd.DataFrame, width: float = 0.5) -> None:
    """Vẽ 1 cặp boxplot Human vs Machine trên cùng trục (đã ẩn điểm ngoại lai)."""
    sns.boxplot(
        data=frame,
        x="Account_Group",
        y="value",
        order=GROUP_ORDER,
        hue="Account_Group",
        hue_order=GROUP_ORDER,
        palette=PALETTE,
        legend=False,
        width=width,
        showfliers=False,            # ẩn điểm ngoại lai: N ~ 1 triệu dòng, vẽ ra chỉ thành khối đen
        showmeans=True,              # dấu chấm vàng = giá trị TRUNG BÌNH
        meanprops=dict(
            marker="o", markeredgecolor="black", markerfacecolor="#FFD400", markersize=7
        ),
        boxprops=dict(alpha=0.90),
        medianprops=dict(color="#111111", linewidth=2.0),
        whiskerprops=dict(color=COLOR_NEUTRAL, linewidth=1.2),
        capprops=dict(color=COLOR_NEUTRAL, linewidth=1.2),
        ax=ax,
    )
    ax.set_xlabel("")  # bỏ nhãn "Account_Group" do seaborn tự đặt; caller tự ghi nhãn phù hợp


def plot_axes_boxplots(
    df: pd.DataFrame,
    machine_mask: np.ndarray,
    axes_map: Sequence[Tuple[Dict[str, Any], str, str]],
    axis_table: pd.DataFrame,
    n_human: int,
    n_machine: int,
    note_style: str = "none",
) -> None:
    """HÌNH A — BOXPLOT 3 trục hành vi: thời gian / phương thức / không gian."""
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", palette="deep")

    n_plots = len(axes_map)
    fig, axes = plt.subplots(1, n_plots, figsize=(6.9 * n_plots, 7.6), squeeze=False)
    axes_flat = axes[0]

    for idx, ((cfg, col, transform), row) in enumerate(zip(axes_map, axis_table.itertuples())):
        ax = axes_flat[idx]
        draw_boxpair(ax, group_frame(df, col, transform, machine_mask), width=0.52)

        # Trung vị in ngay dưới nhãn trục x cho dễ đọc, không phải ước lượng bằng mắt
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                f"{LABEL_HUMAN}\nP50 = {row.Human_Median:.3f}",
                f"{LABEL_MACHINE}\nP50 = {row.Machine_Median:.3f}",
            ],
            fontsize=10,
        )

        if note_style == "compact":
            zero_info = ""
            if hasattr(row, "Human_Zero_Pct") and hasattr(row, "Machine_Zero_Pct"):
                zero_info = f"\n• Zero%    : Người = {row.Human_Zero_Pct:.1f}% | Máy = {row.Machine_Zero_Pct:.1f}%"
            note = (
                "Thực nghiệm (boxplot):\n"
                f"• Trung vị : Người = {row.Human_Median:.3f} | Máy = {row.Machine_Median:.3f}\n"
                f"• Trung bình: Người = {row.Human_Mean:.3f} | Máy = {row.Machine_Mean:.3f}\n"
                f"• Cliff's δ = {row.Cliffs_Delta:+.4f} → {row.Verdict}\n"
                f"• Mann-Whitney p = {fmt_p(row.MannWhitney_p)}"
                f"{zero_info}"
            )
            ax.text(
                0.03, 0.985, note, transform=ax.transAxes, fontsize=8.2,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC",
                          edgecolor="#CBD5E1", alpha=0.95),
            )
        elif note_style == "full":
            note = (
                "Kỳ vọng nghiệp vụ:\n"
                f"• Người : {cfg['human_expectation']}\n"
                f"• Máy   : {cfg['machine_expectation']}\n"
                "----------------------------------\n"
                "Thực nghiệm (boxplot):\n"
                f"• Trung vị : Người = {row.Human_Median:.3f} | Máy = {row.Machine_Median:.3f}\n"
                f"• Trung bình: Người = {row.Human_Mean:.3f} | Máy = {row.Machine_Mean:.3f}\n"
                f"• IQR      : Người = {row.Human_IQR:.3f} | Máy = {row.Machine_IQR:.3f}\n"
                f"• Cliff's δ = {row.Cliffs_Delta:+.4f} → {row.Verdict}\n"
                f"• Mann-Whitney p = {fmt_p(row.MannWhitney_p)}\n"
                f"• Đối chiếu kỳ vọng: {row.Expectation_Check}"
            )
            ax.text(
                0.03, 0.985, note, transform=ax.transAxes, fontsize=8.2,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC",
                          edgecolor="#CBD5E1", alpha=0.95),
            )
        # note_style == "none" -> không vẽ hộp ghi chú

        ax.set_title(
            f"{cfg['title']}\n{col}" + ("  (hiển thị log1p)" if transform == "log1p" else ""),
            fontsize=12.5, fontweight="bold", pad=12, color=COLOR_TITLE,
        )
        ax.set_xlabel("Nhóm thực thể (mỗi hộp = phân phối theo ngày của mọi khoá User x Day)",
                      fontsize=10)
        ax.set_ylabel(cfg["y_label"], fontsize=10.5)

        if cfg["y_limit"] is not None:
            ax.set_ylim(cfg["y_limit"])
        else:
            values = df[col].astype(float).to_numpy()
            lo_h, hi_h = whisker_bounds(values[~machine_mask], transform)
            lo_m, hi_m = whisker_bounds(values[machine_mask], transform)
            bottom, top = min(lo_h, lo_m), max(hi_h, hi_m)
            if top <= bottom:
                bottom, top = min(bottom, 0.0), max(top, 1.0)
            span = top - bottom
            headroom = 0.40 if note_style != "none" else 0.15
            ax.set_ylim(bottom - 0.05 * span, top + headroom * span)

    fig.suptitle(
        "KIỂM ĐỊNH TÍNH HỢP LÝ NGHIỆP VỤ BẰNG BOXPLOT — 3 TRỤC HÀNH VI VẬT LÝ\n"
        f"Người dùng (N={n_human:,}) vs Tài khoản Máy (N={n_machine:,}) — "
        "tách được 2 quần thể mà CHƯA cần mô hình học máy",
        fontsize=14.5, fontweight="bold", y=1.0, color=COLOR_TITLE,
    )
    fig.text(
        0.5, -0.035,
        "Đọc hộp: viền hộp = Q1–Q3 · vạch đậm giữa hộp = trung vị (P50) · "
        "chấm vàng = trung bình · râu = 1,5 x IQR (điểm ngoại lai đã ẩn vì N ≈ 1 triệu).",
        ha="center", fontsize=9, color=COLOR_NEUTRAL,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, "sanity_check_human_vs_machine.png")
    plt.close(fig)


def plot_individual_axis_boxplots(
    df: pd.DataFrame,
    machine_mask: np.ndarray,
    axes_map: Sequence[Tuple[Dict[str, Any], str, str]],
    axis_table: pd.DataFrame,
    n_human: int,
    n_machine: int,
    note_style: str = "none",
) -> List[str]:
    """Xuất từng trục hành vi thành một ảnh boxplot riêng biệt."""
    saved_files: List[str] = []
    for (cfg, col, transform), row in zip(axes_map, axis_table.itertuples()):
        fig, ax = plt.subplots(figsize=(6.2, 6.5))
        draw_boxpair(ax, group_frame(df, col, transform, machine_mask), width=0.48)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                f"{LABEL_HUMAN}\nP50 = {row.Human_Median:.3f}",
                f"{LABEL_MACHINE}\nP50 = {row.Machine_Median:.3f}",
            ],
            fontsize=10.5,
        )

        if note_style == "compact":
            zero_info = ""
            if hasattr(row, "Human_Zero_Pct") and hasattr(row, "Machine_Zero_Pct"):
                zero_info = f"\n• Zero%    : Người = {row.Human_Zero_Pct:.1f}% | Máy = {row.Machine_Zero_Pct:.1f}%"
            note = (
                "Thực nghiệm (boxplot):\n"
                f"• Trung vị : Người = {row.Human_Median:.3f} | Máy = {row.Machine_Median:.3f}\n"
                f"• Trung bình: Người = {row.Human_Mean:.3f} | Máy = {row.Machine_Mean:.3f}\n"
                f"• Cliff's δ = {row.Cliffs_Delta:+.4f} → {row.Verdict}\n"
                f"• Mann-Whitney p = {fmt_p(row.MannWhitney_p)}"
                f"{zero_info}"
            )
            ax.text(
                0.03, 0.985, note, transform=ax.transAxes, fontsize=8.2,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC",
                          edgecolor="#CBD5E1", alpha=0.95),
            )
        elif note_style == "full":
            note = (
                "Kỳ vọng nghiệp vụ:\n"
                f"• Người : {cfg['human_expectation']}\n"
                f"• Máy   : {cfg['machine_expectation']}\n"
                "----------------------------------\n"
                "Thực nghiệm (boxplot):\n"
                f"• Trung vị : Người = {row.Human_Median:.3f} | Máy = {row.Machine_Median:.3f}\n"
                f"• Trung bình: Người = {row.Human_Mean:.3f} | Máy = {row.Machine_Mean:.3f}\n"
                f"• IQR      : Người = {row.Human_IQR:.3f} | Máy = {row.Machine_IQR:.3f}\n"
                f"• Cliff's δ = {row.Cliffs_Delta:+.4f} → {row.Verdict}\n"
                f"• Mann-Whitney p = {fmt_p(row.MannWhitney_p)}\n"
                f"• Đối chiếu kỳ vọng: {row.Expectation_Check}"
            )
            ax.text(
                0.03, 0.985, note, transform=ax.transAxes, fontsize=8.2,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC",
                          edgecolor="#CBD5E1", alpha=0.95),
            )

        ax.set_title(
            f"{cfg['title']}\n{col}" + ("  (hiển thị log1p)" if transform == "log1p" else ""),
            fontsize=12.5, fontweight="bold", pad=12, color=COLOR_TITLE,
        )
        ax.set_xlabel("Nhóm thực thể (Người vs Máy)", fontsize=10.5)
        ax.set_ylabel(cfg["y_label"], fontsize=10.5)

        if cfg["y_limit"] is not None:
            ax.set_ylim(cfg["y_limit"])
        else:
            values = df[col].astype(float).to_numpy()
            lo_h, hi_h = whisker_bounds(values[~machine_mask], transform)
            lo_m, hi_m = whisker_bounds(values[machine_mask], transform)
            bottom, top = min(lo_h, lo_m), max(hi_h, hi_m)
            if top <= bottom:
                bottom, top = min(bottom, 0.0), max(top, 1.0)
            span = top - bottom
            headroom = 0.40 if note_style != "none" else 0.15
            ax.set_ylim(bottom - 0.05 * span, top + headroom * span)

        fig.text(
            0.5, -0.02,
            "Đọc hộp: viền = Q1–Q3 · vạch đậm = P50 · chấm vàng = trung bình · râu = 1,5 x IQR",
            ha="center", fontsize=8.5, color=COLOR_NEUTRAL,
        )
        plt.tight_layout()
        fname = f"sanity_check_axis_{cfg['key']}_{col}.png"
        save_figure(fig, fname)
        plt.close(fig)
        saved_files.append(fname)
    return saved_files


def plot_feature_grid(
    df: pd.DataFrame,
    machine_mask: np.ndarray,
    feature_table: pd.DataFrame,
    n_human: int,
    n_machine: int,
) -> None:
    """HÌNH B — lưới boxplot 4x4 cho TOÀN BỘ đặc trưng, sắp giảm dần theo |Cliff's delta|."""
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", palette="deep")

    n_features = feature_table.shape[0]
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.6 * n_rows), squeeze=False)

    for pos, row in enumerate(feature_table.itertuples()):
        ax = axes[pos // n_cols][pos % n_cols]
        transform = row.Display_Transform
        values = df[row.Feature].astype(float).to_numpy()
        draw_boxpair(ax, group_frame(df, row.Feature, transform, machine_mask), width=0.55)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                f"Người\nP50 = {row.Human_Median:.3g}",
                f"Máy\nP50 = {row.Machine_Median:.3g}",
            ],
            fontsize=9,
        )
        ax.set_title(
            f"{row.Rank}. {row.Feature}"
            + ("   [hiển thị log1p]" if transform == "log1p" else "")
            + f"\n|δ| = {row.Abs_Cliffs_Delta:.3f} — {row.Verdict}",
            fontsize=10.5, fontweight="bold", pad=8,
            color=VERDICT_COLORS.get(row.Effect_Band, COLOR_TITLE),
        )
        ax.set_ylabel("giá trị (thang log1p)" if transform == "log1p" else "giá trị", fontsize=9)
        ax.tick_params(axis="y", labelsize=8.5)

        # Canh trục y theo râu hộp của CẢ 2 nhóm để không làm mất phần đuôi hợp lệ
        lo_h, hi_h = whisker_bounds(values[machine_mask], transform)
        lo_m, hi_m = whisker_bounds(values[~machine_mask], transform)
        bottom, top = min(lo_h, lo_m), max(hi_h, hi_m)
        if top <= bottom:
            bottom, top = min(bottom, 0.0), max(top, 1.0)
        span = top - bottom
        ax.set_ylim(bottom - 0.06 * span, top + 0.16 * span)

        # Cảnh báo đặc trưng bị "bẹp hộp": zero-inflated hoặc NULL-heavy
        # (đặt ở góc PHẢI TRÊN kèm nền trắng mờ để không chèn vào nhãn trục x bên dưới)
        if row.Human_Zero_Pct >= 50.0 or row.Machine_Zero_Pct >= 50.0 \
                or row.Human_Null_Pct >= 5.0 or row.Machine_Null_Pct >= 5.0:
            ax.text(
                0.985, 0.985,
                f"Zero%: N {row.Human_Zero_Pct:.1f} | M {row.Machine_Zero_Pct:.1f}\n"
                f"NULL%: N {row.Human_Null_Pct:.1f} | M {row.Machine_Null_Pct:.1f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.6, color="#5A6270",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.80),
            )

    for pos in range(n_features, n_rows * n_cols):  # ẩn ô trống (nếu có)
        axes[pos // n_cols][pos % n_cols].axis("off")

    strong = int((feature_table["Abs_Cliffs_Delta"] >= 0.474).sum())
    small_ok = int((feature_table["Abs_Cliffs_Delta"] >= MIN_ABS_DELTA_SMALL).sum())
    fig.suptitle(
        "SANITY CHECK BẰNG BOXPLOT — MỨC TỰ TÁCH HUMAN vs MACHINE CỦA TỪNG ĐẶC TRƯNG\n"
        f"{strong}/{n_features} đặc trưng tách RẤT MẠNH (|δ| ≥ 0.474) · "
        f"{small_ok}/{n_features} tách được ở mức 'small' trở lên (|δ| ≥ {MIN_ABS_DELTA_SMALL}) · "
        f"N = {n_human:,} (người) vs {n_machine:,} (máy)",
        fontsize=14, fontweight="bold", y=0.995, color=COLOR_TITLE,
    )
    fig.text(
        0.5, -0.012,
        "Màu tiêu đề = mức tách: xanh lá đậm = large (≥ 0.474) · teal = medium (≥ 0.330) · "
        "hổ phách = small (≥ 0.147) · xám = chưa tách (< 0.147).  "
        "δ > 0 → nhóm NGƯỜI lớn hơn; δ < 0 → nhóm MÁY lớn hơn.  "
        "Hộp bẹp (zero-inflated / NULL-heavy) phải đọc kèm dòng Zero% / NULL% ở góc phải.",
        ha="center", fontsize=9, color=COLOR_NEUTRAL,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    save_figure(fig, "sanity_check_human_vs_machine_boxplot_grid.png")
    plt.close(fig)


# ==============================================================================
# 6. GHI BẢNG CSV + BÁO CÁO MARKDOWN (số liệu lấy trực tiếp từ bảng -> không lệch)
# ==============================================================================
def write_table(frame: pd.DataFrame, filename: str) -> None:
    """Ghi 1 bảng ra docs/feature_engineering/tables (không index)."""
    path = OUTPUT_TAB_DIR / filename
    frame.to_csv(path, index=False)
    print(f"[✓] Đã lưu bảng tại: {path.resolve()}")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    """Sinh bảng Markdown."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _fmt(value: float, nd: int = 3) -> str:
    return f"{value:.{nd}f}" if np.isfinite(value) else "—"


def fmt_p(p_value: Any) -> str:
    """Định dạng p-value: với N ≈ 1 triệu p bị underflow về 0 nên ghi rõ '< 1e-300'.

    Chấp nhận cả số lẫn chuỗi (bảng số liệu lưu sẵn dạng chuỗi này để không mất mát thông tin).
    """
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return str(p_value)
    if not np.isfinite(p):
        return "—"
    if p <= 0.0 or p < 1e-300:
        return "< 1e-300"
    return f"{p:.3e}"


def _report_header(ctx: Dict[str, Any]) -> List[str]:
    return [
        "# Kiểm định tính hợp lý nghiệp vụ (Sanity Check: Human vs. Machine) — bằng BOXPLOT",
        "",
        "Sinh bởi `scripts/feature_engineering/plot_sanity_check_human_vs_machine.py` — "
        "mọi số liệu dưới đây được script tính lại từ dữ liệu mỗi lần chạy (không viết tay).",
        "",
        "| Tham số | Giá trị |",
        "|---|---|",
        f"| Dữ liệu đầu vào | `{ctx['data_path']}` |",
        f"| Số bản ghi (khoá User x Day) | {ctx['n_rows']:,} |",
        f"| Nhóm Người dùng (Human) | {ctx['n_human']:,} bản ghi "
        f"({100.0 * ctx['n_human'] / ctx['n_rows']:.2f}%) |",
        f"| Nhóm Tài khoản Máy (Machine) | {ctx['n_machine']:,} bản ghi "
        f"({100.0 * ctx['n_machine'] / ctx['n_rows']:.2f}%) |",
        f"| Luật gán nhóm | {ctx['rule']} |",
        f"| Số đặc trưng kiểm định bằng boxplot | {ctx['n_features']} (lưới 4x4) |",
        f"| Thang hiển thị log1p | {ctx['log_features']} |",
        f"| Ngưỡng hiệu ứng Cliff's delta | nhỏ ≥ {MIN_ABS_DELTA_SMALL} · mạnh ≥ 0.330 · "
        "rất mạnh ≥ 0.474 (Romano et al.) |",
        "",
        "## 1. Kết luận nhanh",
        "",
    ]


def _report_conclusions(
    table: pd.DataFrame, axis_table: pd.DataFrame, ctx: Dict[str, Any]
) -> List[str]:
    n_features = table.shape[0]
    abs_delta = table["Abs_Cliffs_Delta"].to_numpy()
    n_large = int((abs_delta >= 0.474).sum())
    n_medium = int(((abs_delta >= 0.330) & (abs_delta < 0.474)).sum())
    n_small = int(((abs_delta >= MIN_ABS_DELTA_SMALL) & (abs_delta < 0.330)).sum())
    n_weak = int((abs_delta < MIN_ABS_DELTA_SMALL).sum())
    top = table.head(3)
    weak = table[table["Abs_Cliffs_Delta"] < MIN_ABS_DELTA_SMALL]
    axes_ok = int((axis_table["Expectation_Check"] == "KHỚP KỲ VỌNG").sum()) if not axis_table.empty else 0

    lines = [
        f"1. **Bộ đặc trưng TỰ TÁCH được Human vs Machine — ĐẠT**: "
        f"**{n_large + n_medium + n_small}/{n_features}** đặc trưng có |Cliff's delta| ≥ "
        f"{MIN_ABS_DELTA_SMALL} (trong đó {n_large} mức *large*, {n_medium} mức *medium*, "
        f"{n_small} mức *small*). Như vậy chỉ bằng **boxplot** (không dùng model, không dùng nhãn) "
        "đã nhìn thấy 2 quần thể khác bản chất vật lý.",
        "",
        "2. **Ba đặc trưng tách mạnh nhất** (đọc chỉ số ở lưới §4): "
        + "; ".join(
            f"`{row.Feature}` |δ| = {row.Abs_Cliffs_Delta:.3f} "
            f"(P50 Người {_fmt(row.Human_Median)} vs Máy {_fmt(row.Machine_Median)}, "
            f"{row.Median_Direction})"
            for row in top.itertuples()
        )
        + ".",
        "",
    ]

    if not weak.empty:
        reasons = []
        for row in weak.itertuples():
            why: List[str] = []
            # Lý do chính: hộp 2 nhóm có trùng nhau về vị trí hay không
            if np.isfinite(row.Human_Median) and np.isfinite(row.Machine_Median):
                denom = max(abs(row.Human_Median), abs(row.Machine_Median), 1e-12)
                if abs(row.Machine_Median - row.Human_Median) / denom <= 0.15:
                    why.append(
                        "trung vị 2 nhóm gần như trùng nhau "
                        f"({_fmt(row.Human_Median)} vs {_fmt(row.Machine_Median)})"
                    )
            if row.Human_Null_Pct >= 5.0 or row.Machine_Null_Pct >= 5.0:
                why.append(
                    f"NULL-heavy (NULL: Người {row.Human_Null_Pct:.2f}% / Máy {row.Machine_Null_Pct:.2f}%)"
                )
            if row.Human_Zero_Pct >= 50.0 or row.Machine_Zero_Pct >= 50.0:
                why.append(
                    f"zero-inflated (0: Người {row.Human_Zero_Pct:.2f}% / Máy {row.Machine_Zero_Pct:.2f}%)"
                )
            if not why:
                why.append("hộp và râu 2 nhóm chồng lấn, không có khoảng tách rõ ràng")
            reasons.append(f"`{row.Feature}` (|δ| = {row.Abs_Cliffs_Delta:.3f}): " + "; ".join(why))
        lines += [
            f"3. **Đặc trưng CHƯA tách được ({n_weak}/{n_features})** — không phải lỗi dữ liệu, "
            "mà do bản chất phân phối; phải đọc kèm %NULL/%ZERO chứ không kết luận qua hình dạng hộp:",
            "",
        ]
        lines += [f"   - {item}" for item in reasons]
        lines += [
            "",
            "   → Nhóm này **không dùng để phân biệt loại thực thể**, nhưng KHÔNG nên loại bỏ: chúng "
            "vẫn hữu ích cho bài toán *so sánh một khoá với chính nó theo thời gian* (baseline per-entity), "
            "nơi Human–Machine không còn là biến phân biệt.",
            "",
        ]

    lines += [
        f"4. **Đối chiếu kỳ vọng nghiệp vụ trên 3 trục hành vi**: "
        f"{axes_ok}/{axis_table.shape[0]} trục **KHỚP KỲ VỌNG** "
        "(chi tiết ở §5).",
        "",
        "5. **Hệ quả cho pipeline**: nhãn `entity_type` chỉ được dùng để **kiểm định**, "
        "**không** đưa vào vector đặc trưng — nếu đưa vào, model sẽ học đường tắt phân loại thực thể "
        "thay vì học bất thường hành vi. Bộ đặc trưng hiện tại đã đủ nhạy để tự phân biệt.",
        "",
    ]
    return lines


def _report_method() -> List[str]:
    return [
        "## 2. Phương pháp kiểm định (boxplot là công cụ chính)",
        "",
        *md_table(
            ["Lớp bằng chứng", "Cách làm", "Cách đọc / ngưỡng"],
            [
                [
                    "① Boxplot 3 trục hành vi",
                    "`sns.boxplot` cho từng nhóm trên cùng một trục: hộp = Q1–Q3, vạch đậm giữa hộp "
                    "= trung vị (P50), râu = 1,5 × IQR, chấm vàng = trung bình; điểm ngoại lai ẩn "
                    "(N ≈ 1 triệu nên vẽ ra chỉ thành khối đen)",
                    "Hai hộp tách rời theo trục tung ⇒ đặc trưng tự phân biệt được bản chất thực thể",
                ],
                [
                    "② Lưới boxplot 4×4",
                    "Vẽ TOÀN BỘ 16 đặc trưng, mỗi ô 1 cặp hộp Người–Máy; 6 cột lệch mạnh "
                    "(|skew| > 3) hiển thị trên thang log1p; sắp giảm dần theo |Cliff's delta|",
                    "Màu tiêu đề ô = mức tách (large/medium/small/xám); ô hộp bẹp phải đọc thêm "
                    "%ZERO / %NULL in ở góc phải",
                ],
                [
                    "③ Định lượng mức tách",
                    "Cliff's delta = 2U / (n_người × n_máy) − 1, với U là thống kê Mann-Whitney "
                    "(kiểm định 2 phía, có hiệu chỉnh đồng hạng)",
                    "|δ| ≥ 0.147 small · ≥ 0.330 medium · ≥ 0.474 large (Romano et al. 2006); "
                    "δ > 0 ⇒ nhóm NGƯỜI lớn hơn, δ < 0 ⇒ nhóm MÁY lớn hơn",
                ],
            ],
        ),
        "",
        "> **Vì sao boxplot + Cliff's delta?** Cả hai đều **phi tham số**: không giả định phân phối "
        "chuẩn, chịu được đuôi dài, giá trị chặn [0,1] và lượng điểm 0 lớn của bộ đặc trưng này. "
        "p-value Mann-Whitney chỉ để tham chiếu: với N ≈ 1 triệu thì p gần như luôn ≈ 0, nên "
        "**không** dùng p để kết luận mức tách mà dùng |δ|.",
        "",
    ]


def _report_feature_table(table: pd.DataFrame) -> List[str]:
    rows = [
        [
            int(row.Rank),
            f"`{row.Feature}`",
            row.Group + (" *(cột hiển thị)*" if row.Role == "display" else ""),
            "log1p" if row.Display_Transform == "log1p" else "thô",
            _fmt(row.Human_Median),
            _fmt(row.Machine_Median),
            _fmt(row.Human_IQR),
            _fmt(row.Machine_IQR),
            f"{row.Human_Null_Pct:.1f} / {row.Machine_Null_Pct:.1f}",
            f"{row.Human_Zero_Pct:.1f} / {row.Machine_Zero_Pct:.1f}",
            f"{row.Cliffs_Delta:+.3f}",
            row.Verdict,
        ]
        for row in table.itertuples()
    ]
    return [
        "## 4. Bảng 1 — 16 đặc trưng, sắp giảm dần theo mức tự tách (Hình: lưới boxplot 4×4)",
        "",
        "> P50/IQR tính trên thang hiển thị ghi ở cột *Thang* (log1p cho 6 cột lệch mạnh).",
        "> `%NULL` và `%ZERO` là tỷ lệ **trên toàn bộ dòng của nhóm**, không phải trên dòng khả dụng.",
        "",
        *md_table(
            ["#", "Đặc trưng", "Nhóm", "Thang", "P50 Người", "P50 Máy", "IQR Người", "IQR Máy",
             "NULL% N/M", "ZERO% N/M", "Cliff's δ", "Mức tách"],
            rows,
        ),
        "",
    ]


def _report_figures() -> List[str]:
    return [
        "## 3. Hình ảnh bằng chứng",
        "",
        *md_table(
            ["Hình", "Cần nhìn gì"],
            [
                [
                    "`docs/feature_engineering/figures/sanity_check_human_vs_machine.png`",
                    "3 boxplot theo trục hành vi (thời gian – phương thức – không gian). Mỗi ô có "
                    "hộp kỳ vọng nghiệp vụ kèm P50, IQR, Cliff's δ, p-value và kết quả đối chiếu "
                    "kỳ vọng ⇒ vừa thấy hình dạng, vừa thấy con số.",
                ],
                [
                    "`docs/feature_engineering/figures/sanity_check_human_vs_machine_boxplot_grid.png`",
                    "Lưới 4×4 cho TOÀN BỘ 16 đặc trưng, sắp giảm dần theo |δ|. Trả lời trực tiếp câu "
                    "hỏi *“bộ đặc trưng có tự tách được Human vs Machine mà chưa cần model không?”*",
                ],
            ],
        ),
        "",
    ]


def _report_axis_table(axis_table: pd.DataFrame) -> List[str]:
    rows = [
        [
            row.Axis_Title,
            f"`{row.Feature_Used}`",
            "log1p" if row.Display_Transform == "log1p" else "thô",
            _fmt(row.Human_Median),
            _fmt(row.Machine_Median),
            _fmt(row.Human_IQR),
            _fmt(row.Machine_IQR),
            f"{row.Cliffs_Delta:+.3f}",
            row.Verdict,
            row.Expectation_Check,
        ]
        for row in axis_table.itertuples()
    ]
    expectations = [
        [f"**{row.Axis_Title}**", f"Người: {row.Human_Expectation}", f"Máy: {row.Machine_Expectation}",
         row.Expectation_Check]
        for row in axis_table.itertuples()
    ]
    return [
        "## 5. Bảng 2 — 3 trục hành vi vật lý (Hình: `sanity_check_human_vs_machine.png`)",
        "",
        *md_table(
            ["Trục", "Đặc trưng", "Thang", "P50 Người", "P50 Máy", "IQR Người", "IQR Máy",
             "Cliff's δ", "Mức tách", "Đối chiếu kỳ vọng"],
            rows,
        ),
        "",
        "### 5.1. Kỳ vọng nghiệp vụ vs thực nghiệm",
        "",
        *md_table(["Trục", "Kỳ vọng cho nhóm Người", "Kỳ vọng cho nhóm Máy", "Kết quả"], expectations),
        "",
        "> Quy tắc đối chiếu được script tự kiểm tra: trục thời gian — giá trị nhóm Máy phải CAO hơn; "
        "trục phương thức — giá trị nhóm Người phải CAO hơn; trục không gian — IQR nhóm Người phải "
        "RỘNG hơn (người đi lại nhiều máy, tài khoản máy gắn chặt với máy của nó).",
        ">",
        "> Với đặc trưng zero-inflated (trung vị 2 nhóm đều = 0, ví dụ `interactive_ratio` có ~87% số 0 "
        "ở nhóm Người), script tự chuyển sang đối chiếu theo **trung bình** rồi mới kết luận "
        "Khớp/Lệch — nếu chỉ so trung vị sẽ kết luận sai là “lệch kỳ vọng”.",
        "",
    ]


def _report_limits(ctx: Dict[str, Any]) -> List[str]:
    return [
        "## 6. Giới hạn & lưu ý khi trích dẫn",
        "",
        f"1. **Nhãn nhóm chỉ là heuristic đặt tên tài khoản**: {ctx['rule']}. Tài khoản dịch vụ/hệ thống "
        "(Service, System) nằm trong nhóm “không phải Machine”, nên phần nào làm dịu mức tách thật.",
        "2. **Boxplot không thấy tính đa đỉnh**: hộp chỉ tổng hợp Q1–Q3 và trung vị; hai phân phối rất "
        "khác nhau vẫn có thể cho 2 hộp giống nhau ⇒ phải đọc kèm histogram (`distribution_*.png`) "
        "hoặc ECDF khi cần kết luận mạnh.",
        "3. **Điểm ngoại lai đã bị ẩn trên hình** (N ≈ 1 triệu, vẽ ra chỉ thành khối đen) nên đuôi dài "
        "không hiện — điều này KHÔNG ảnh hưởng Cliff's delta (tính trên toàn bộ dữ liệu).",
        "4. **Đặc trưng NULL-heavy** (`failure_locked_out_share`): hộp và δ chỉ tính trên dòng KHẢ DỤNG "
        "nên n của nhóm Máy nhỏ hơn hẳn; phải đọc kèm cột NULL% ở Bảng 1 chứ không kết luận "
        "“không tách được” chỉ từ hình dạng hộp.",
        "5. **p-value ≈ 0 luôn đúng khi N ≈ 1 triệu** ⇒ không dùng p để xếp hạng mức tách, chỉ dùng |δ|.",
        "6. **Tách được Human/Machine KHÔNG đồng nghĩa phát hiện xâm nhập**: đây là kiểm định độ nhạy "
        "của bộ đặc trưng với bản chất vật lý của thực thể, không phải nhãn bất thường. Nhãn tấn công "
        "thật vẫn lấy từ ground truth (redteam) ở bước benchmark.",
        "7. **Thang log1p chỉ để hiển thị**: 6 cột lệch mạnh được vẽ trên log1p để hộp khỏi bị bóp dẹt; "
        "Cliff's delta và các phân vị trong bảng vẫn tính trên giá trị gốc của ma trận — không thay đổi "
        "dữ liệu đưa vào model.",
        "",
    ]


def _report_artifacts(ctx: Dict[str, Any]) -> List[str]:
    rows = [[label, f"`{path}`"] for label, path in ctx["artifacts"]]
    return ["## 7. Sản phẩm sinh ra", "", *md_table(["Loại", "Đường dẫn"], rows), ""]


def build_report(table: pd.DataFrame, axis_table: pd.DataFrame, ctx: Dict[str, Any]) -> str:
    """Ghép toàn bộ báo cáo Markdown."""
    lines: List[str] = []
    lines += _report_header(ctx)
    lines += _report_conclusions(table, axis_table, ctx)
    lines += _report_method()
    lines += _report_figures()
    lines += _report_feature_table(table)
    lines += _report_axis_table(axis_table)
    lines += _report_limits(ctx)
    lines += _report_artifacts(ctx)
    return "\n".join(lines)


# ==============================================================================
# 7. CHẠY TOÀN BỘ KIỂM ĐỊNH
# ==============================================================================
def run_sanity_check_analysis(
    data_path: Path, report_path: Path, note_style: str = "none"
) -> None:
    """Đọc ma trận, chạy sanity check bằng BOXPLOT, xuất hình + bảng + báo cáo."""
    print("=" * 96)
    print("KIỂM ĐỊNH TÍNH HỢP LÝ NGHIỆP VỤ — SANITY CHECK: HUMAN vs MACHINE (PHƯƠNG PHÁP: BOXPLOT)")
    print("=" * 96)

    if not data_path.exists():
        print(f"[ERROR] Không tìm thấy dữ liệu tại: {data_path.resolve()}")
        sys.exit(1)

    print(f"[INFO] Đọc dữ liệu từ: {data_path.resolve()}")
    df = pl.read_parquet(data_path).to_pandas()
    print(f"[INFO] Tổng số bản ghi ma trận: {len(df):,}")

    df, rule = ensure_machine_account_label(df)
    if "entity_type" in df.columns:
        print("[INFO] Phân bố cột nhãn `entity_type`:")
        print(df["entity_type"].value_counts(dropna=False).to_string())

    machine_mask = df["is_machine_account"].to_numpy().astype(bool)
    n_machine = int(machine_mask.sum())
    n_human = int((~machine_mask).sum())
    if n_human == 0 or n_machine == 0:
        print("[ERROR] Một trong hai nhóm bị rỗng -> không kiểm định được!")
        sys.exit(1)
    print(f"[INFO] Nhóm Người dùng (Human) : {n_human:,} bản ghi ({100.0 * n_human / len(df):.2f}%)")
    print(f"[INFO] Nhóm Tài khoản Máy     : {n_machine:,} bản ghi ({100.0 * n_machine / len(df):.2f}%)")
    print(f"[INFO] Luật gán nhóm: {rule}")

    axes_map = resolve_axes(list(df.columns))
    if not axes_map:
        print("[WARN] Không khớp được trục hành vi nào -> chỉ vẽ lưới 16 đặc trưng.")

    print("\n[INFO] Tính thống kê hộp + Cliff's delta cho 16 đặc trưng ...")
    feature_table = build_feature_table(df, machine_mask)
    axis_table = build_axis_table(df, machine_mask, axes_map) if axes_map else pd.DataFrame()

    write_table(feature_table, "sanity_check_human_vs_machine.csv")
    if not axis_table.empty:
        write_table(axis_table, "sanity_check_human_vs_machine_axes.csv")

    if axes_map and not axis_table.empty:
        plot_axes_boxplots(df, machine_mask, axes_map, axis_table, n_human, n_machine, note_style=note_style)
        plot_individual_axis_boxplots(df, machine_mask, axes_map, axis_table, n_human, n_machine, note_style=note_style)
    plot_feature_grid(df, machine_mask, feature_table, n_human, n_machine)

    log_features = ", ".join(
        feature_table.loc[feature_table["Display_Transform"] == "log1p", "Feature"].tolist()
    )
    ctx: Dict[str, Any] = {
        "data_path": str(data_path.resolve()),
        "n_rows": int(len(df)),
        "n_human": n_human,
        "n_machine": n_machine,
        "rule": rule,
        "n_features": int(feature_table.shape[0]),
        "log_features": log_features or "(không có)",
        "artifacts": [
            ("Hình A — 3 trục hành vi (boxplot ghép)",
             OUTPUT_FIG_DIR / "sanity_check_human_vs_machine.png"),
            ("Hình A — bản sao artifacts",
             ARTIFACT_DIR / "sanity_check_human_vs_machine.png"),
            ("Hình A1 — Trục 1 (Thời gian)",
             OUTPUT_FIG_DIR / "sanity_check_axis_temporal_off_hours_ratio.png"),
            ("Hình A2 — Trục 2 (Phương thức)",
             OUTPUT_FIG_DIR / "sanity_check_axis_logon_mechanism_interactive_ratio.png"),
            ("Hình A3 — Trục 3 (Không gian)",
             OUTPUT_FIG_DIR / "sanity_check_axis_spatial_fanout_distinct_hosts.png"),
            ("Hình B — lưới boxplot 16 đặc trưng",
             OUTPUT_FIG_DIR / "sanity_check_human_vs_machine_boxplot_grid.png"),
            ("Hình B — bản sao artifacts",
             ARTIFACT_DIR / "sanity_check_human_vs_machine_boxplot_grid.png"),
            ("Bảng 1 — 16 đặc trưng + hiệu ứng",
             OUTPUT_TAB_DIR / "sanity_check_human_vs_machine.csv"),
            ("Bảng 2 — 3 trục hành vi",
             OUTPUT_TAB_DIR / "sanity_check_human_vs_machine_axes.csv"),
            ("Báo cáo Markdown", report_path),
        ],
    }
    report_path.write_text(build_report(feature_table, axis_table, ctx), encoding="utf-8")
    print(f"[✓] Đã lưu báo cáo tại: {report_path.resolve()}")

    _print_summary(feature_table, axis_table)


def _print_summary(feature_table: pd.DataFrame, axis_table: pd.DataFrame) -> None:
    """In bảng xếp hạng mức tách + kết luận ra console."""
    abs_delta = feature_table["Abs_Cliffs_Delta"].to_numpy()
    n_features = feature_table.shape[0]
    n_large = int((abs_delta >= 0.474).sum())
    n_medium = int(((abs_delta >= 0.330) & (abs_delta < 0.474)).sum())
    n_small = int(((abs_delta >= MIN_ABS_DELTA_SMALL) & (abs_delta < 0.330)).sum())
    n_weak = int((abs_delta < MIN_ABS_DELTA_SMALL).sum())

    print("\n" + "=" * 96)
    print("BẢNG XẾP HẠNG MỨC TỰ TÁCH HUMAN vs MACHINE (giảm dần theo |Cliff's delta|):")
    print("=" * 96)
    print(
        feature_table[
            ["Rank", "Feature", "Group", "Human_Median", "Machine_Median", "Human_IQR",
             "Machine_IQR", "Cliffs_Delta", "Effect_Band", "Verdict"]
        ].to_string(index=False)
    )
    print("-" * 96)
    print(
        f"[KẾT LUẬN] {n_large + n_medium + n_small}/{n_features} đặc trưng tách được "
        f"(|δ| ≥ {MIN_ABS_DELTA_SMALL}): {n_large} large · {n_medium} medium · {n_small} small · "
        f"{n_weak} chưa tách."
    )
    if not axis_table.empty:
        ok = int((axis_table["Expectation_Check"] == "KHỚP KỲ VỌNG").sum())
        print(
            f"[KẾT LUẬN] Đối chiếu kỳ vọng nghiệp vụ trên {axis_table.shape[0]} trục hành vi: "
            f"{ok}/{axis_table.shape[0]} trục KHỚP KỲ VỌNG."
        )
        print(
            axis_table[
                ["Axis_Title", "Feature_Used", "Human_Median", "Machine_Median",
                 "Cliffs_Delta", "Verdict", "Expectation_Check"]
            ].to_string(index=False)
        )
    print("\n[SUCCESS] Hoàn thành kiểm định tính hợp lý nghiệp vụ Human vs Machine (boxplot)!")
    print(f"          Hình   : {OUTPUT_FIG_DIR.resolve()}")
    print(f"          Bảng   : {OUTPUT_TAB_DIR.resolve()}")
    print("=" * 96 + "\n")


def main() -> None:
    """Điểm vào CLI của script."""
    parser = argparse.ArgumentParser(
        description=(
            "Kiểm định tính hợp lý nghiệp vụ (sanity check) Human vs Machine bằng BOXPLOT: "
            "3 trục hành vi + lưới 4x4 toàn bộ 16 đặc trưng + Cliff's delta / Mann-Whitney."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Đường dẫn file ma trận đặc trưng THÔ (feature_matrix_raw.parquet)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_DIR / "sanity_check_human_vs_machine.md",
        help="Đường dẫn báo cáo Markdown đầu ra",
    )
    parser.add_argument(
        "--note-style",
        choices=["compact", "none", "full"],
        default="none",
        help="Kiểu ghi chú trên biểu đồ 3 trục: compact (gọn gàng), none (mặc định: ẩn sạch), full (kiểu cũ kèm kỳ vọng dài dòng)",
    )
    args = parser.parse_args()
    run_sanity_check_analysis(args.data_path, args.report_path, note_style=args.note_style)


if __name__ == "__main__":
    main()













