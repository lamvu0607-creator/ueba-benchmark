"""Module khảo sát chất lượng dữ liệu đa ngày bằng Polars.

Cung cấp các hàm phân tích vĩ mô trên toàn bộ các ngày (mặc định 60 ngày):
- Quét nhanh từng ngày bằng Polars (Batch Aggregation) nhằm kiểm soát RAM < 300MB.
- Tính toán các chỉ số: Volume, Failure Rate, Machine Account Ratio, Off-hours Ratio, Active Users/Hosts.
- Tự động phát hiện nhịp điệu tuần (Weekday / Weekend) và các ngày có volume dị thường.
- Xuất biểu đồ xu hướng 4 bảng (Volume, Failure Rate, Active Entities, Off-hours).
"""

from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set
import yaml

# Đảm bảo UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Tải cấu hình hệ thống từ file YAML."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_day_number(day_str: str) -> int:
    """Trích xuất số thứ tự ngày để sắp xếp đúng thứ tự tự nhiên (day-01 -> 1)."""
    match = re.search(r"day[-_]?(\d+)", day_str, re.IGNORECASE)
    return int(match.group(1)) if match else 999999


def survey_single_day(
    interim_dir: Path | str,
    day_tag: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Khảo sát chất lượng dữ liệu của 1 ngày duy nhất bằng Polars.

    Chỉ đọc các cột cần thiết (Projection Pushdown) để tối ưu tốc độ và bộ nhớ.
    """
    interim_path = Path(interim_dir)
    config = config or {}

    off_hours_start = config.get("features", {}).get("off_hours_start", 18)
    off_hours_end = config.get("features", {}).get("off_hours_end", 7)

    f4624 = interim_path / "event_4624" / f"event_4624_{day_tag}.parquet"
    f4625 = interim_path / "event_4625" / f"event_4625_{day_tag}.parquet"

    metrics: Dict[str, Any] = {
        "day": day_tag,
        "day_num": extract_day_number(day_tag),
        "count_4624": 0,
        "count_4625": 0,
        "total_events": 0,
        "failure_rate_pct": 0.0,
        "distinct_users": 0,
        "distinct_hosts": 0,
        "machine_account_pct": 0.0,
        "off_hours_pct": 0.0,
        "null_critical_count": 0,
        "exact_dups": 0,
        "exact_dup_pct": 0.0,
        "session_dups": 0,
        "session_dup_pct": 0.0,
        "bursty_dups": 0,
        "bursty_dup_pct": 0.0,
        "min_time": None,
        "max_time": None,
    }

    if not f4624.exists() and not f4625.exists():
        return metrics

    # 1. Phân tích Event 4625 (Failed Logon)
    if f4625.exists():
        df_4625 = pl.read_parquet(
            f4625,
            columns=["Time", "EventID", "UserName"],
        )
        metrics["count_4625"] = df_4625.height
        metrics["null_critical_count"] += (
            df_4625["Time"].null_count()
            + df_4625["EventID"].null_count()
            + df_4625["UserName"].null_count()
        )
        del df_4625

    # 2. Phân tích Event 4624 (Successful Logon)
    if f4624.exists():
        df_4624 = pl.read_parquet(
            f4624,
            columns=["Time", "EventID", "UserName", "LogHost", "LogonType", "LogonID"],
        )
        n_4624 = df_4624.height
        metrics["count_4624"] = n_4624

        metrics["null_critical_count"] += (
            df_4624["Time"].null_count()
            + df_4624["EventID"].null_count()
            + df_4624["UserName"].null_count()
            + df_4624["LogHost"].null_count()
        )

        if n_4624 > 0:
            # Thống kê thời gian
            t_min = df_4624["Time"].min()
            t_max = df_4624["Time"].max()
            metrics["min_time"] = int(t_min) if t_min is not None else None
            metrics["max_time"] = int(t_max) if t_max is not None else None

            # Phân tích Off-hours (Giờ ngoài hành chính)
            hours = (df_4624["Time"] // 3600) % 24
            is_off = (hours >= off_hours_start) | (hours < off_hours_end)
            metrics["off_hours_pct"] = round(float(is_off.sum() / n_4624) * 100, 2)

            # Phân tích tài khoản máy (Computer accounts có kết thúc bằng $)
            users = df_4624["UserName"].drop_nulls()
            is_machine = users.str.ends_with("$")
            metrics["machine_account_pct"] = round(
                float(is_machine.sum() / n_4624) * 100, 2
            )

            # Đếm số lượng thực thể hoạt động
            metrics["distinct_users"] = df_4624["UserName"].n_unique()
            metrics["distinct_hosts"] = df_4624["LogHost"].n_unique()

            # Phân tích Trùng lặp (Duplicates): Exact vs Logical
            # 1. Exact Duplicate (Trùng tuyệt đối trên các cột định danh chính)
            metrics["exact_dups"] = df_4624.is_duplicated().sum()
            metrics["exact_dup_pct"] = round(
                float(metrics["exact_dups"] / n_4624) * 100, 2
            )

            # 2. Logical Duplicate Type 1: Session Collision (Time, User, Host, LogonType, LogonID)
            metrics["session_dups"] = (
                df_4624.select(["Time", "UserName", "LogHost", "LogonType", "LogonID"])
                .is_duplicated()
                .sum()
            )
            metrics["session_dup_pct"] = round(
                float(metrics["session_dups"] / n_4624) * 100, 2
            )

            # 3. Logical Duplicate Type 2: Bursty Polling Loops (Delta t <= 2s cùng User + Host + LogonType)
            bursty_count = (
                df_4624.select(["Time", "UserName", "LogHost", "LogonType"])
                .sort(["UserName", "LogHost", "LogonType", "Time"])
                .with_columns(
                    pl.col("Time")
                    .diff()
                    .over(["UserName", "LogHost", "LogonType"])
                    .alias("dt")
                )["dt"]
                <= 2
            ).sum()
            metrics["bursty_dups"] = bursty_count
            metrics["bursty_dup_pct"] = round(
                float(bursty_count / n_4624) * 100, 2
            )

        del df_4624

    # 3. Tính toán tổng hợp
    total = metrics["count_4624"] + metrics["count_4625"]
    metrics["total_events"] = total
    if total > 0:
        metrics["failure_rate_pct"] = round(
            (metrics["count_4625"] / total) * 100, 3
        )

    return metrics


def survey_multiple_days(
    interim_dir: Path | str,
    days_list: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """Khảo sát hàng loạt danh sách ngày, trả về Polars DataFrame tổng hợp."""
    rows = []
    sorted_days = sorted(days_list, key=extract_day_number)

    for day in sorted_days:
        m = survey_single_day(interim_dir, day, config=config)
        rows.append(m)

    df = pl.DataFrame(rows)

    if df.height > 0:
        # Tự động phát hiện ngày cuối tuần (dựa trên volume thấp hơn 75% trung vị)
        median_vol = df["total_events"].median()
        df = df.with_columns(
            (pl.col("total_events") < (median_vol * 0.75)).alias("is_weekend_detected")
        )

    return df


def plot_survey_trends(
    df: pl.DataFrame,
    output_path: Path | str,
    title: str = "Khảo sát Chất lượng & Nhịp điệu Log Windows (4624 & 4625)",
):
    """Vẽ biểu đồ xu hướng 4 bảng theo thời gian và lưu thành file ảnh PNG."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf = df.to_pandas()
    days = pdf["day"].tolist()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)

    # 1. Total Volume (4624 vs 4625)
    ax1 = axes[0]
    ax1.plot(
        days,
        pdf["total_events"] / 1e6,
        marker="o",
        color="#1f77b4",
        linewidth=2,
        label="Tổng sự kiện (Triệu)",
    )
    # Highlight detected weekend
    if "is_weekend_detected" in pdf.columns:
        weekends = pdf[pdf["is_weekend_detected"]]
        if not weekends.empty:
            ax1.scatter(
                weekends["day"],
                weekends["total_events"] / 1e6,
                color="#e74c3c",
                zorder=5,
                label="Cuối tuần (Dự đoán)",
            )
    ax1.set_ylabel("Số lượng (Triệu)")
    ax1.set_title("1. Tổng khối lượng sự kiện hàng ngày (Volume)")
    ax1.legend(loc="upper right")

    # 2. Failure Rate (%)
    ax2 = axes[1]
    ax2.plot(
        days,
        pdf["failure_rate_pct"],
        marker="s",
        color="#d62728",
        linewidth=2,
        label="Tỷ lệ thất bại (%)",
    )
    ax2.set_ylabel("Tỷ lệ (%)")
    ax2.set_title("2. Biến động tỷ lệ đăng nhập thất bại (Failure Rate %)")
    ax2.legend(loc="upper right")

    # 3. Entities (Active Users & Hosts)
    ax3 = axes[2]
    ax3.plot(
        days,
        pdf["distinct_users"],
        marker="^",
        color="#2ca02c",
        linewidth=2,
        label="Active Users",
    )
    ax3.plot(
        days,
        pdf["distinct_hosts"],
        marker="v",
        color="#ff7f0e",
        linewidth=2,
        label="Active Hosts",
    )
    ax3.set_ylabel("Số lượng thực thể")
    ax3.set_title("3. Số lượng Users và Hosts hoạt động mỗi ngày")
    ax3.legend(loc="upper right")

    # 4. Off-hours & Machine Accounts (%)
    ax4 = axes[3]
    ax4.plot(
        days,
        pdf["off_hours_pct"],
        marker="d",
        color="#9467bd",
        linewidth=2,
        label="Ngoài giờ (Off-hours %)",
    )
    ax4.plot(
        days,
        pdf["machine_account_pct"],
        marker="x",
        color="#8c564b",
        linewidth=2,
        linestyle="--",
        label="Tài khoản máy ($ %)",
    )
    ax4.set_ylabel("Tỷ lệ (%)")
    ax4.set_xlabel("Mã ngày (Day)")
    ax4.set_title("4. Tỷ lệ hoạt động ngoài giờ & Tài khoản máy tính")
    ax4.legend(loc="upper right")

    # Rotate x labels if many days
    if len(days) > 15:
        axes[3].tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
