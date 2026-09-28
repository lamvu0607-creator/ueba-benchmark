"""
plot_temporal_stability.py

Kiểm tra TÍNH ỔN ĐỊNH CHU KỲ THEO THỜI GIAN (Temporal Stationarity) của bộ đặc trưng
User x Day trước khi đưa vào model anomaly detection.

Mục tiêu nghiệp vụ: bộ đặc trưng PHẢI mô tả được nhịp tuần hoàn tự nhiên — sự sụt giảm
hoạt động vào cuối tuần (thứ Bảy, Chủ Nhật) và các khoảng nghỉ sinh học (ban đêm) —
để model KHÔNG sinh cảnh báo rác định kỳ (periodic false alert) mỗi cuối tuần / mỗi đêm.

5 câu hỏi kiểm định (Q1..Q5):
  Q1. Nhịp tuần      : lưu lượng hệ thống có chu kỳ 7 ngày ổn định? (profile theo pha, ACF lag 1..14)
  Q2. Cuối tuần      : 2 pha nào là "weekend-like" và mức sụt so với ngày thường là bao nhiêu %?
                       (SUY LUẬN từ dữ liệu — chưa xác nhận calendar origin, theo
                        docs/plan/lanl_eda_implementation_plan.md §4.3 nên gọi là "weekend-like")
  Q3. Nghỉ sinh học  : nhịp giờ trong ngày (đêm 0h-5h so với giờ làm việc 8h-17h), đồng thời
                       đối chiếu cột off_hours_ratio của ma trận với dữ liệu THÔ event-level.
  Q4. Trôi nền       : nền hoạt động (volume / failure / off_hours) có drift theo thời gian?
  Q5. Phản ánh chu kỳ: mỗi đặc trưng tách được weekday-like vs weekend-like mạnh đến đâu
                       (Cliff's delta trên giá trị trung bình có trọng số theo ngày)?

Đầu vào:
  - data/features/raw/feature_matrix_raw.parquet              (BẮT BUỘC — ma trận THÔ)
  - data/interim/event_4624/, data/interim/event_4625/         (TÙY CHỌN — chỉ đọc cột `Time`
    cho Q3; thiếu dữ liệu này script vẫn chạy, chỉ bỏ phần nhịp giờ)
  - configs/system_config.yaml   (off_hours_start/end, work_hours_start/end, weekend_days)

Đầu ra:
  - docs/feature_engineering/figures/temporal_stability_daily.png
  - docs/feature_engineering/figures/temporal_weekly_cycle.png
  - docs/feature_engineering/figures/temporal_hourly_rhythm.png      (nếu có interim)
  - artifacts/  (đồng bộ cùng tên ảnh)
  - docs/feature_engineering/tables/daily_activity_summary.csv
  - docs/feature_engineering/tables/weekly_cycle_summary.csv
  - docs/feature_engineering/tables/hourly_rhythm_summary.csv        (nếu có interim)
  - docs/feature_engineering/tables/weekday_vs_weekend_features.csv
  - docs/feature_engineering/tables/temporal_stationarity_metrics.csv
  - reports/week2/temporal_stability_check.md
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
INTERIM_DIR = BASE_DIR / "data" / "interim"
CONFIG_PATH = BASE_DIR / "configs" / "system_config.yaml"

OUTPUT_FIG_DIR = BASE_DIR / "docs" / "feature_engineering" / "figures"
OUTPUT_TAB_DIR = BASE_DIR / "docs" / "feature_engineering" / "tables"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
REPORT_DIR = REPO_ROOT / "reports" / "week2"

for _dir in (OUTPUT_FIG_DIR, OUTPUT_TAB_DIR, ARTIFACT_DIR, REPORT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. HẰNG SỐ & NGƯỠNG DIỄN GIẢI
# ==============================================================================
CYCLE_LEN = 7                  # độ dài chu kỳ tuần (ngày)
MAX_LAG = 14                   # số lag khảo sát cho ACF (2 vòng tuần)
NIGHT_START, NIGHT_END = 0, 5  # khoảng nghỉ sinh học (đêm sâu): 0h -> 5h

# Giá trị mặc định — khớp configs/system_config.yaml (dùng khi không đọc được YAML)
DEFAULT_OFF_HOURS_START = 18
DEFAULT_OFF_HOURS_END = 7
DEFAULT_WORK_HOURS_START = 8
DEFAULT_WORK_HOURS_END = 17
DEFAULT_WEEKEND_DAY_COUNT = 2  # configs/system_config.yaml: weekend_days = [5, 6]

# 16 đặc trưng mục tiêu — TÊN CỘT THÔ trong feature_matrix_raw.parquet
# (xem configs/feature_schema.yaml; total_logons chỉ dùng làm trọng số/hiển thị)
CORE_FEATURE_CANDIDATES: List[str] = [
    "failure_ratio",
    "failure_locked_out_share",
    "off_hours_ratio",
    "interarrival_dt_mean",
    "delta_t_cv",
    "same_second_share",
    "is_single_event",
    "interactive_ratio",
    "rare_logon_type_count",
    "ntlm_ratio",
    "distinct_hosts",
    "distinct_sources_count",
    "missing_source_ratio",
    "remote_logon_ratio",
    "custom_proc_share",
]

# Ngưỡng diễn giải (ghi thẳng vào bảng metrics để tránh "đọc số rồi tự suy diễn")
THRESHOLDS: Dict[str, float] = {
    "acf_lag7_strong": 0.30,                 # ACF lag 7 -> chu kỳ tuần rõ
    "phase_r2_strong_pct": 30.0,             # % phương sai ngày giải thích bởi pha tuần
    "weekend_drop_min_pct": 10.0,            # mức sụt cuối tuần tối thiểu đáng kể
    "phase_consistency_strong_pct": 75.0,    # % số tuần xác nhận đúng 2 pha thấp nhất
    "cliff_delta_small": 0.147,              # ngưỡng hiệu ứng "small" (Romano et al.)
    "drift_rho_max": 0.30,                   # |rho| cho phép giữa ngày và nền trượt 7 ngày
    "drift_abs_slope_max_pct": 2.0,          # |% thay đổi/tuần| cho phép (Theil-Sen trên log)
    "night_dip_min_pct": 20.0,               # mức sụt tối thiểu của đêm so với giờ cao điểm
}

COLOR_MAIN = "#1F4E79"
COLOR_WEEKEND = "#D9534F"
COLOR_ACCENT = "#F0AD4E"
COLOR_TITLE = "#173F5F"
COLOR_NEUTRAL = "#6C757D"


# ==============================================================================
# 3. TIỆN ÍCH ĐỌC CẤU HÌNH & TRỌNG SỐ
# ==============================================================================
def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Đọc config YAML; PyYAML là tùy chọn — thiếu thì dùng giá trị mặc định built-in."""
    if not config_path.exists():
        print(f"[CẢNH BÁO] Không thấy config {config_path} -> dùng giá trị mặc định.")
        return {}
    try:
        import yaml  # import trễ để không bắt buộc PyYAML
    except ImportError:
        print("[CẢNH BÁO] Chưa cài PyYAML -> dùng giá trị mặc định built-in.")
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_time_windows(cfg: Dict[str, Any]) -> Dict[str, int]:
    """Lấy cửa sổ giờ + số ngày cuối tuần từ config (khớp configs/system_config.yaml)."""
    feats = dict(cfg.get("features", {}) or {})
    weekend_days = feats.get("weekend_days") or [5, 6]
    return {
        "off_hours_start": int(feats.get("off_hours_start", DEFAULT_OFF_HOURS_START)),
        "off_hours_end": int(feats.get("off_hours_end", DEFAULT_OFF_HOURS_END)),
        "work_hours_start": int(feats.get("work_hours_start", DEFAULT_WORK_HOURS_START)),
        "work_hours_end": int(feats.get("work_hours_end", DEFAULT_WORK_HOURS_END)),
        # số ngày cuối tuần / tuần = số pha cần suy luận là "weekend-like"
        "weekend_day_count": max(1, min(len(list(weekend_days)), CYCLE_LEN - 1)),
    }


def is_off_hours(hour: np.ndarray, off_start: int, off_end: int) -> np.ndarray:
    """Đúng công thức của extractor: hour >= off_start HOẶC hour <= off_end."""
    return (hour >= off_start) | (hour <= off_end)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Trung bình có trọng số, bỏ qua NULL của đặc trưng (trọng số = 0 tại NULL)."""
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w)
    if not mask.any() or w[mask].sum() == 0:
        return float("nan")
    return float((x[mask] * w[mask]).sum() / w[mask].sum())


# ==============================================================================
# 4. TỔNG HỢP NGÀY (giữ nguyên các cột cũ của daily_activity_summary.csv)
# ==============================================================================
def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp chỉ số theo ngày + trung bình có trọng số của failure/off_hours."""
    work = df.assign(
        _vol=df["total_logons"].astype(float),
        _fail=(df["failure_ratio"].astype(float) if "failure_ratio" in df.columns else 0.0),
        _off=(df["off_hours_ratio"].astype(float) if "off_hours_ratio" in df.columns else np.nan),
    )
    rows: List[Dict[str, Any]] = []
    for day, grp in work.groupby("day", sort=True):
        vol = grp["_vol"].to_numpy(dtype=float)
        fail = grp["_fail"].to_numpy(dtype=float)
        rows.append(
            {
                "day": int(day),
                "total_system_volume": round(float(vol.sum()), 4),
                "active_accounts": int(vol.size),
                "user_vol_median": round(float(np.median(vol)), 4),
                "user_vol_p25": round(float(np.percentile(vol, 25)), 4),
                "user_vol_p75": round(float(np.percentile(vol, 75)), 4),
                "user_vol_p90": round(float(np.percentile(vol, 90)), 4),
                "mean_failure_ratio": round(float(np.nanmean(fail)), 4),
                "p95_failure_ratio": round(float(np.nanpercentile(fail, 95)), 4),
                "max_failure_ratio": round(float(np.nanmax(fail)), 4),
                "weighted_failure_ratio": round(weighted_mean(fail, vol), 6),
                "weighted_off_hours_ratio": round(
                    weighted_mean(grp["_off"].to_numpy(dtype=float), vol), 6
                ),
            }
        )

    daily = pd.DataFrame.from_records(rows)
    total_all = float(daily["total_system_volume"].sum())
    daily["logons_share_pct"] = (100.0 * daily["total_system_volume"] / total_all).round(4)
    daily["phase"] = (daily["day"] - 1) % CYCLE_LEN  # pha chu kỳ: 0 = ngày đầu tiên của dữ liệu
    daily["rolling7_volume"] = (
        daily["total_system_volume"].rolling(CYCLE_LEN, min_periods=1).mean().round(4)
    )
    return daily


# ==============================================================================
# 5. PHÂN TÍCH CHU KỲ 7 NGÀY (Q1, Q2)
# ==============================================================================
def build_phase_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Profile theo pha chu kỳ (day_index % 7) — bằng chứng định lượng cho Q1/Q2."""
    rows: List[Dict[str, Any]] = []
    for phase, grp in daily.groupby("phase", sort=True):
        vol = grp["total_system_volume"].to_numpy(dtype=float)
        rows.append(
            {
                "cycle_phase": int(phase),
                "n_days": int(vol.size),
                "mean_volume": round(float(vol.mean()), 4),
                "median_volume": round(float(np.median(vol)), 4),
                "min_volume": round(float(vol.min()), 4),
                "max_volume": round(float(vol.max()), 4),
                "cv_pct": round(float(100.0 * vol.std(ddof=1) / vol.mean()), 4),
                "mean_active_accounts": round(float(grp["active_accounts"].mean()), 2),
                "mean_weighted_failure_ratio": round(
                    float(grp["weighted_failure_ratio"].mean()), 6
                ),
                "mean_weighted_off_hours_ratio": round(
                    float(grp["weighted_off_hours_ratio"].mean()), 6
                ),
            }
        )
    profile = pd.DataFrame.from_records(rows)
    profile["volume_rank"] = profile["mean_volume"].rank(method="min").astype(int)
    return profile.sort_values("cycle_phase").reset_index(drop=True)


def identify_weekend_like_phases(profile: pd.DataFrame, n_weekend: int) -> List[int]:
    """Suy luận n pha có lưu lượng TRUNG BÌNH thấp nhất = 'weekend-like' (plan §4.3)."""
    ordered = profile.sort_values("mean_volume")["cycle_phase"].tolist()
    return sorted(int(p) for p in ordered[:n_weekend])


def finalize_phase_profile(
    profile: pd.DataFrame, weekend_like: Sequence[int], daily: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Gắn cờ weekend-like + mức sụt so với ngày thường vào bảng profile."""
    mask = profile["cycle_phase"].isin(list(weekend_like))
    weekday_mean = float(profile.loc[~mask, "mean_volume"].mean())
    weekend_mean = float(profile.loc[mask, "mean_volume"].mean())
    profile = profile.copy()
    profile["is_weekend_like"] = mask.astype(int)
    profile["drop_vs_weekday_pct"] = (
        100.0 * (profile["mean_volume"] - weekday_mean) / weekday_mean
    ).round(4)

    weekend_daily = daily[daily["phase"].isin(list(weekend_like))]
    weekday_daily = daily[~daily["phase"].isin(list(weekend_like))]
    summary = {
        "weekday_mean_volume": weekday_mean,
        "weekend_mean_volume": weekend_mean,
        "weekend_drop_pct": 100.0 * (weekday_mean - weekend_mean) / weekday_mean,
        "weekday_active_accounts_mean": float(weekday_daily["active_accounts"].mean()),
        "weekend_active_accounts_mean": float(weekend_daily["active_accounts"].mean()),
        "weekday_weighted_failure_ratio": float(weekday_daily["weighted_failure_ratio"].mean()),
        "weekend_weighted_failure_ratio": float(weekend_daily["weighted_failure_ratio"].mean()),
        "weekday_weighted_off_hours_ratio": float(
            weekday_daily["weighted_off_hours_ratio"].mean()
        ),
        "weekend_weighted_off_hours_ratio": float(
            weekend_daily["weighted_off_hours_ratio"].mean()
        ),
        "n_weekday_like_days": int(weekday_daily.shape[0]),
        "n_weekend_like_days": int(weekend_daily.shape[0]),
    }
    return profile, summary


def week_block_consistency(
    daily: pd.DataFrame, weekend_like: Sequence[int], n_weekend: int
) -> Tuple[float, int, int]:
    """% tuần TRỌN VẸN mà đúng n pha lưu lượng thấp nhất trùng tập pha weekend-like toàn cục."""
    ordered = daily.sort_values("day")
    n_blocks = int(ordered.shape[0] // CYCLE_LEN)
    target = set(int(p) for p in weekend_like)
    matches = 0
    for b in range(n_blocks):
        block = ordered.iloc[b * CYCLE_LEN : (b + 1) * CYCLE_LEN]
        lows = set(int(p) for p in block.nsmallest(n_weekend, "total_system_volume")["phase"])
        if lows == target:
            matches += 1
    pct = 100.0 * matches / n_blocks if n_blocks else float("nan")
    return round(float(pct), 4), matches, n_blocks


def autocorrelation(series: Sequence[float], max_lag: int) -> List[float]:
    """ACF (Pearson) của chuỗi đã trừ trung bình — dùng để tìm chu kỳ lag 7."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom == 0:
        return [float("nan")] * max_lag
    return [float((x[:-k] * x[k:]).sum() / denom) for k in range(1, max_lag + 1)]


def phase_variance_explained(
    values: Sequence[float], phases: Sequence[int]
) -> Tuple[float, float]:
    """R² của mô hình chỉ-dùng-pha + hệ số biến thiên phần dư (nền sau khi bỏ chu kỳ tuần)."""
    v = np.asarray(values, dtype=float)
    p = np.asarray(phases, dtype=int)
    grand = float(v.mean())
    fitted = np.array([float(v[p == ph].mean()) for ph in p])
    ss_tot = float(((v - grand) ** 2).sum())
    r2 = 1.0 - float(((v - fitted) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    resid_cv = float((v - fitted).std(ddof=1) / grand) if grand else float("nan")
    return round(r2, 6), round(resid_cv, 6)


def drift_metrics(daily: pd.DataFrame) -> List[Dict[str, Any]]:
    """Q4: nền hoạt động có trôi theo thời gian? (đo trên nền trượt 7 ngày để bỏ hiệu ứng tuần)"""
    day_idx = daily["day"].to_numpy(dtype=float)
    specs = [
        ("volume (log1p)", np.log1p(daily["total_system_volume"].to_numpy(dtype=float)), "log"),
        ("failure_ratio (trọng số)", daily["weighted_failure_ratio"].to_numpy(dtype=float), "ratio"),
        (
            "off_hours_ratio (trọng số)",
            daily["weighted_off_hours_ratio"].to_numpy(dtype=float),
            "ratio",
        ),
    ]
    records: List[Dict[str, Any]] = []
    for name, series, kind in specs:
        smoothed = pd.Series(series).rolling(CYCLE_LEN, min_periods=CYCLE_LEN).mean()
        valid = smoothed.notna().to_numpy() & np.isfinite(series)
        rho, p_value = stats.spearmanr(day_idx[valid], smoothed.to_numpy()[valid])
        slope = float(stats.theilslopes(series[valid], day_idx[valid]).slope)
        # volume: hệ số log -> %/tuần theo cấp số nhân; ratio: điểm phần trăm/tuần
        slope_pct = (
            (np.exp(slope * CYCLE_LEN) - 1) * 100 if kind == "log" else slope * CYCLE_LEN * 100
        )
        half = series.size // 2
        first, second = series[:half], series[half:]
        med_first, med_second = float(np.median(first)), float(np.median(second))
        ratio = med_second / med_first if med_first else float("nan")
        try:
            mwu_p = float(stats.mannwhitneyu(first, second, alternative="two-sided").pvalue)
        except ValueError:
            mwu_p = float("nan")
        ok = (
            abs(float(rho)) <= THRESHOLDS["drift_rho_max"]
            and abs(float(slope_pct)) <= THRESHOLDS["drift_abs_slope_max_pct"]
        )
        records.append(
            {
                "metric": name,
                "rolling7_spearman_rho": round(float(rho), 4),
                "spearman_p": round(float(p_value), 6),
                "theil_sen_slope_pct_per_week": round(float(slope_pct), 4),
                "half1_median": round(med_first, 6),
                "half2_median": round(med_second, 6),
                "half2_over_half1": round(float(ratio), 4),
                "mannwhitney_p": round(mwu_p, 6),
                "status": "OK (không drift rõ)" if ok else "WARN (có dấu hiệu drift)",
            }
        )
    return records


# ==============================================================================
# 6. NHỊP GIỜ TRONG NGÀY — KHOẢNG NGHỈ SINH HỌC (Q3)
# ==============================================================================
def scan_hourly_counts(
    interim_dir: Path, days: Sequence[int], event_ids: Sequence[int] = (4624, 4625)
) -> Optional[pd.DataFrame]:
    """Quét event-level (CHỈ cột `Time`) -> đếm sự kiện theo (day, event_id, hour).

    hour = (Time % 86400) // 3600 — đúng công thức của extract_account_day_matrix.py:255.
    """
    if not any((interim_dir / f"event_{eid}").exists() for eid in event_ids):
        print(f"[CẢNH BÁO] Không thấy {interim_dir}/event_4624|4625 -> BỎ QUA phần nhịp giờ.")
        return None

    rows: List[Dict[str, int]] = []
    for day in days:
        for eid in event_ids:
            path = interim_dir / f"event_{eid}" / f"event_{eid}_day-{day:02d}.parquet"
            if not path.exists():
                continue
            agg = (
                pl.scan_parquet(path)
                .select(pl.col("Time"))
                .with_columns(((pl.col("Time") % 86400) // 3600).cast(pl.Int32).alias("hour"))
                .group_by("hour")
                .len()
                .collect()
            )
            rows.extend(
                {"day": int(day), "event_id": int(eid), "hour": int(h), "count": int(c)}
                for h, c in zip(agg["hour"].to_list(), agg["len"].to_list())
            )
    if not rows:
        print("[CẢNH BÁO] Không đọc được file interim nào -> BỎ QUA phần nhịp giờ.")
        return None
    return pd.DataFrame.from_records(rows)


def build_hourly_profile(
    hourly: pd.DataFrame, daily: pd.DataFrame, weekend_like: Sequence[int], win: Dict[str, int]
) -> pd.DataFrame:
    """Profile theo giờ: tách ngày thường / cuối tuần-like; gắn cờ off_hours theo config."""
    flags = daily.set_index("day")["phase"].isin(list(weekend_like)).to_dict()
    hourly = hourly.assign(
        is_weekend_like=hourly["day"].map(lambda d: bool(flags.get(int(d), False)))
    )
    n_weekend_days = max(1, int(hourly.loc[hourly["is_weekend_like"], "day"].nunique()))
    n_weekday_days = max(1, int(hourly.loc[~hourly["is_weekend_like"], "day"].nunique()))

    rows: List[Dict[str, Any]] = []
    for hour in range(24):
        slot = hourly[hourly["hour"] == hour]
        ok = slot[slot["event_id"] == 4624]["count"].sum()
        fail = slot[slot["event_id"] == 4625]["count"].sum()
        ok_wd = slot[(slot["event_id"] == 4624) & (~slot["is_weekend_like"])]["count"].sum()
        ok_we = slot[(slot["event_id"] == 4624) & (slot["is_weekend_like"])]["count"].sum()
        rows.append(
            {
                "hour": hour,
                "ok_4624": int(ok),
                "fail_4625": int(fail),
                "total_events": int(ok + fail),
                "ok_per_day": round(float(ok) / (n_weekday_days + n_weekend_days), 2),
                "ok_per_weekday_like_day": round(float(ok_wd) / n_weekday_days, 2),
                "ok_per_weekend_like_day": round(float(ok_we) / n_weekend_days, 2),
                "failure_ratio_pct": (
                    round(100.0 * float(fail) / float(ok + fail), 4) if (ok + fail) else float("nan")
                ),
                "is_off_hours": int(
                    bool(
                        is_off_hours(
                            np.array([hour]), win["off_hours_start"], win["off_hours_end"]
                        )[0]
                    )
                ),
            }
        )
    profile = pd.DataFrame.from_records(rows)
    total = float(profile["total_events"].sum())
    profile["share_pct"] = (100.0 * profile["total_events"] / total).round(4)
    profile["weekend_over_weekday_pct"] = (
        100.0
        * profile["ok_per_weekend_like_day"]
        / profile["ok_per_weekday_like_day"].replace(0.0, np.nan)
    ).round(4)
    return profile


def hourly_summary(profile: pd.DataFrame, win: Dict[str, int]) -> Dict[str, float]:
    """Chỉ số cho Q3: độ sâu đêm, sụt giảm ban ngày vs ban đêm, đối chiếu off_hours_ratio."""
    night = profile[(profile["hour"] >= NIGHT_START) & (profile["hour"] <= NIGHT_END)]
    work = profile[
        (profile["hour"] >= win["work_hours_start"]) & (profile["hour"] <= win["work_hours_end"])
    ]
    off = profile[profile["is_off_hours"] == 1]
    peak_hour = profile.loc[profile["ok_per_day"].idxmax()]
    night_low = night.loc[night["ok_per_day"].idxmin()]
    total_events = float(profile["total_events"].sum())
    night_dip = (
        100.0 * (1.0 - float(night_low["ok_per_day"]) / float(peak_hour["ok_per_day"]))
        if float(peak_hour["ok_per_day"])
        else float("nan")
    )
    # Cuối tuần phải sụt MẠNH HƠN ở giờ làm việc so với ban đêm -> dấu hiệu "nghỉ sinh học"
    work_ratio = float(work["ok_per_weekend_like_day"].sum()) / float(
        work["ok_per_weekday_like_day"].sum()
    )
    night_ratio = float(night["ok_per_weekend_like_day"].sum()) / float(
        night["ok_per_weekday_like_day"].sum()
    )
    return {
        "night_share_pct": round(100.0 * float(night["total_events"].sum()) / total_events, 4),
        "peak_hour": int(peak_hour["hour"]),
        "peak_hour_ok_per_day": round(float(peak_hour["ok_per_day"]), 2),
        "night_min_hour": int(night_low["hour"]),
        "night_min_ok_per_day": round(float(night_low["ok_per_day"]), 2),
        "night_dip_pct": round(night_dip, 4),
        "weekend_drop_work_hours_pct": round((1.0 - work_ratio) * 100.0, 4),
        "weekend_drop_night_hours_pct": round((1.0 - night_ratio) * 100.0, 4),
        "off_hours_share_empirical_pct": round(
            100.0 * float(off["total_events"].sum()) / total_events, 4
        ),
    }


# ==============================================================================
# 7. ĐẶC TRƯNG CÓ PHẢN ÁNH CHU KỲ TUẦN KHÔNG? (Q5)
# ==============================================================================
def mann_whitney_cliffs(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, int, int]:
    """Cliff's delta (hiệu ứng) + p-value Mann-Whitney; delta > 0 nghĩa là a > b."""
    x = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    y = np.asarray([v for v in b if np.isfinite(v)], dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan"), int(x.size), int(y.size)
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    delta = 2.0 * float(res.statistic) / (x.size * y.size) - 1.0
    return float(delta), float(res.pvalue), int(x.size), int(y.size)


def build_feature_cycle_table(
    df: pd.DataFrame, daily: pd.DataFrame, weekend_like: Sequence[int]
) -> pd.DataFrame:
    """Q5: so sánh từng đặc trưng giữa ngày thường và ngày cuối tuần-like.

    Mỗi NGÀY chỉ đóng góp 1 quan sát = trung bình đặc trưng có trọng số theo total_logons
    (tránh phóng đại ý nghĩa thống kê do lặp tài khoản giữa các ngày).
    """
    features = resolve_columns(df.columns.tolist(), CORE_FEATURE_CANDIDATES)
    is_weekend_map = daily.set_index("day")["phase"].isin(list(weekend_like))

    per_day: List[Dict[str, Any]] = []
    for day, grp in df.groupby("day", sort=True):
        w = grp["total_logons"].astype(float).to_numpy(dtype=float)
        row: Dict[str, Any] = {"day": int(day)}
        for feat in features:
            row[feat] = weighted_mean(grp[feat].to_numpy(dtype=float), w)
        per_day.append(row)

    day_values = pd.DataFrame.from_records(per_day).set_index("day")
    day_values = day_values.join(is_weekend_map.rename("is_weekend_like"))
    week_mask = day_values["is_weekend_like"].astype(bool)
    daily_week_mask = daily["phase"].isin(list(weekend_like))

    records: List[Dict[str, Any]] = [
        _feature_cycle_row(
            "total_logons",
            daily.loc[daily_week_mask, "total_system_volume"].to_numpy(dtype=float),
            daily.loc[~daily_week_mask, "total_system_volume"].to_numpy(dtype=float),
            role="volume",
        )
    ]
    for feat in features:
        col = day_values[feat]
        records.append(
            _feature_cycle_row(
                feat,
                col[week_mask].to_numpy(dtype=float),
                col[~week_mask].to_numpy(dtype=float),
                role="feature",
            )
        )
    table = pd.DataFrame.from_records(records)
    table["delta_rank"] = table["cliffs_delta"].abs().rank(ascending=False, method="min").astype(int)
    return table.sort_values("delta_rank").reset_index(drop=True)


def _feature_cycle_row(
    feature: str, weekend_vals: np.ndarray, weekday_vals: np.ndarray, role: str
) -> Dict[str, Any]:
    """1 dòng của bảng Q5 (tách riêng để tái sử dụng cho cả volume và đặc trưng)."""
    wd_mean = float(np.nanmean(weekday_vals))
    we_mean = float(np.nanmean(weekend_vals))
    delta, p_value, n_we, n_wd = mann_whitney_cliffs(weekend_vals, weekday_vals)
    gap_pct = 100.0 * (we_mean - wd_mean) / wd_mean if wd_mean else float("nan")
    return {
        "feature": feature,
        "role": role,
        "n_weekend_like_days": n_we,
        "n_weekday_like_days": n_wd,
        "weekday_like_mean": round(wd_mean, 6),
        "weekend_like_mean": round(we_mean, 6),
        "weekend_minus_weekday_pct": round(gap_pct, 4),
        "cliffs_delta": round(delta, 4),
        "mannwhitney_p": round(p_value, 6),
        "reflects_weekly_cycle": (
            "YES" if abs(delta) >= THRESHOLDS["cliff_delta_small"] else "no"
        ),
    }


def resolve_columns(df_columns: Sequence[str], candidates: Sequence[str]) -> List[str]:
    """Giữ lại các tên cột ứng viên thực sự tồn tại trong ma trận."""
    return [c for c in candidates if c in df_columns]


# ==============================================================================
# 8. BẢNG CHỈ SỐ TỔNG HỢP (metrics) — "trạm kiểm soát" của check này
# ==============================================================================
def _metric_row(
    group: str, metric: str, value: Any, unit: str, interpretation: str, status: str
) -> Dict[str, Any]:
    return {
        "check_group": group,
        "metric": metric,
        "value": value,
        "unit": unit,
        "interpretation": interpretation,
        "status": status,
    }


def assemble_metrics(  # gộp mọi kết quả Q1..Q5 vào 1 bảng để dễ đối chiếu
    acf_values: Sequence[float],
    cycle_r2: float,
    residual_cv: float,
    weekend_like: Sequence[int],
    weekend_summary: Dict[str, float],
    consistency: Tuple[float, int, int],
    daily: pd.DataFrame,
    feature_table: pd.DataFrame,
    drift_records: Sequence[Dict[str, Any]],
    hourly_stats: Optional[Dict[str, float]],
) -> pd.DataFrame:
    """Gom toàn bộ kết quả Q1..Q5 thành 1 bảng có ngưỡng diễn giải tường minh."""
    rows: List[Dict[str, Any]] = []
    lag7 = float(acf_values[CYCLE_LEN - 1])
    lag14 = float(acf_values[2 * CYCLE_LEN - 1])
    peak_lag = int(np.nanargmax(acf_values)) + 1
    thr = THRESHOLDS

    # ---------------- Q1: chu kỳ 7 ngày ----------------
    rows.append(_metric_row(
        "Q1 - Nhịp tuần", "acf_lag7", round(lag7, 4), "hệ số tương quan",
        f"Ngưỡng chu kỳ rõ ≥ {thr['acf_lag7_strong']}",
        "OK" if lag7 >= thr["acf_lag7_strong"] else "WARN",
    ))
    rows.append(_metric_row(
        "Q1 - Nhịp tuần", "acf_lag14", round(lag14, 4), "hệ số tương quan",
        "Xác nhận vòng tuần thứ hai (kỳ vọng ≈ lag 7)",
        "OK" if lag14 >= thr["acf_lag7_strong"] else "WARN",
    ))
    rows.append(_metric_row(
        "Q1 - Nhịp tuần", "acf_peak_lag", peak_lag, "lag",
        "Lag có ACF cao nhất trong 1..14 (kỳ vọng = 7)",
        "OK" if peak_lag == CYCLE_LEN else "WARN",
    ))
    rows.append(_metric_row(
        "Q1 - Nhịp tuần", "phase_variance_explained_pct", round(100.0 * cycle_r2, 4), "%",
        "Phần phương sai log(lưu lượng/ngày) giải thích chỉ bằng pha tuần",
        "OK" if 100.0 * cycle_r2 >= thr["phase_r2_strong_pct"] else "WARN",
    ))
    rows.append(_metric_row(
        "Q1 - Nhịp tuần", "residual_cv_after_phase_pct", round(100.0 * residual_cv, 4), "%",
        "Biến thiên còn lại sau khi bỏ chu kỳ tuần (nền để dò bất thường THẬT)",
        "INFO",
    ))

    # ---------------- Q2: sụt giảm cuối tuần ----------------
    drop = float(weekend_summary["weekend_drop_pct"])
    acc_drop = 100.0 * (
        weekend_summary["weekday_active_accounts_mean"]
        - weekend_summary["weekend_active_accounts_mean"]
    ) / weekend_summary["weekday_active_accounts_mean"]
    cons_pct, matches, blocks = consistency
    rows.append(_metric_row(
        "Q2 - Cuối tuần", "weekend_like_phases",
        ", ".join(str(p) for p in weekend_like), "pha (day_index % 7)",
        "2 pha lưu lượng thấp nhất — SUY LUẬN, chưa xác nhận calendar origin (plan §4.3)",
        "INFO",
    ))
    rows.append(_metric_row(
        "Q2 - Cuối tuần", "weekend_drop_pct", round(drop, 4), "%",
        f"Ngưỡng đáng kể ≥ {thr['weekend_drop_min_pct']}%",
        "OK" if drop >= thr["weekend_drop_min_pct"] else "WARN",
    ))
    rows.append(_metric_row(
        "Q2 - Cuối tuần", "weekend_active_accounts_drop_pct", round(acc_drop, 4), "%",
        "Số tài khoản hoạt động cũng giảm theo cùng nhịp (không chỉ volume)", "INFO",
    ))
    rows.append(_metric_row(
        "Q2 - Cuối tuần", "week_block_consistency_pct", round(float(cons_pct), 4), "%",
        f"{matches}/{blocks} tuần trọn vẹn có đúng 2 pha thấp nhất = tập weekend-like "
        f"(ngưỡng ≥ {thr['phase_consistency_strong_pct']}%)",
        "OK" if cons_pct >= thr["phase_consistency_strong_pct"] else "WARN",
    ))
    rows.append(_metric_row(
        "Q2 - Cuối tuần", "weekend_minus_weekday_failure_ratio_pp",
        round(
            100.0
            * (
                weekend_summary["weekend_weighted_failure_ratio"]
                - weekend_summary["weekday_weighted_failure_ratio"]
            ),
            4,
        ),
        "điểm %",
        "Chênh lệch tỷ lệ thất bại (trọng số theo lưu lượng) giữa cuối tuần và ngày thường",
        "INFO",
    ))
    rows.append(_metric_row(
        "Q2 - Cuối tuần", "weekend_minus_weekday_off_hours_ratio_pp",
        round(
            100.0
            * (
                weekend_summary["weekend_weighted_off_hours_ratio"]
                - weekend_summary["weekday_weighted_off_hours_ratio"]
            ),
            4,
        ),
        "điểm %",
        "Cuối tuần off_hours_ratio cao hơn vì hoạt động dồn ra ngoài giờ hành chính",
        "INFO",
    ))

    # ---------------- Q4: trôi nền ----------------
    for rec in drift_records:
        rows.append(_metric_row(
            "Q4 - Trôi nền", f"{rec['metric']}: rho (nền trượt 7 ngày)",
            rec["rolling7_spearman_rho"], "hệ số",
            f"Theil-Sen {rec['theil_sen_slope_pct_per_week']}%/tuần; "
            f"trung vị nửa sau/nửa đầu = {rec['half2_over_half1']}; p(MWU) = {rec['mannwhitney_p']}",
            "OK" if str(rec["status"]).startswith("OK") else "WARN",
        ))

    return _assemble_hourly_and_feature_rows(rows, daily, feature_table, hourly_stats)


def _assemble_hourly_and_feature_rows(
    rows: List[Dict[str, Any]],
    daily: pd.DataFrame,
    feature_table: pd.DataFrame,
    hourly_stats: Optional[Dict[str, float]],
) -> pd.DataFrame:
    """Phần cuối của bảng metrics: Q3 (nhịp giờ) + Q5 (đặc trưng phản ánh chu kỳ)."""
    thr = THRESHOLDS

    # ---------------- Q3: nhịp giờ / nghỉ sinh học ----------------
    if hourly_stats:
        rows.append(_metric_row(
            "Q3 - Nghỉ sinh học", "night_share_pct", hourly_stats["night_share_pct"], "%",
            f"Tỷ trọng sự kiện rơi vào đêm {NIGHT_START}h-{NIGHT_END}h (nhịp rest tự nhiên)",
            "INFO",
        ))
        rows.append(_metric_row(
            "Q3 - Nghỉ sinh học", "night_dip_pct", hourly_stats["night_dip_pct"], "%",
            f"Giờ thấp nhất trong đêm ({hourly_stats['night_min_hour']}h) so với giờ cao điểm "
            f"({hourly_stats['peak_hour']}h); ngưỡng sụt ≥ {thr['night_dip_min_pct']}%",
            "OK" if float(hourly_stats["night_dip_pct"]) >= thr["night_dip_min_pct"] else "WARN",
        ))
        rows.append(_metric_row(
            "Q3 - Nghỉ sinh học", "weekend_drop_work_hours_pct",
            hourly_stats["weekend_drop_work_hours_pct"], "%",
            "Mức sụt cuối tuần trong giờ làm việc (8h-17h) — nơi con người nghỉ", "INFO",
        ))
        rows.append(_metric_row(
            "Q3 - Nghỉ sinh học", "weekend_drop_night_hours_pct",
            hourly_stats["weekend_drop_night_hours_pct"], "%",
            "Mức sụt cuối tuần ban đêm (0h-5h) — thường thấp hơn ban ngày vì máy chạy 24/7",
            "OK"
            if float(hourly_stats["weekend_drop_night_hours_pct"])
            < float(hourly_stats["weekend_drop_work_hours_pct"])
            else "WARN",
        ))
        rows.append(_metric_row(
            "Q3 - Nghỉ sinh học", "off_hours_share_empirical_pct",
            hourly_stats["off_hours_share_empirical_pct"], "%",
            "Tỷ trọng sự kiện THÔ ngoài cửa sổ off-hours (18h-7h theo config)", "INFO",
        ))
        feature_off_pct = 100.0 * float(daily["weighted_off_hours_ratio"].mean())
        rows.append(_metric_row(
            "Q3 - Nghỉ sinh học", "off_hours_ratio_feature_pct", round(feature_off_pct, 4), "%",
            "Trung bình cột off_hours_ratio của ma trận (trọng số theo total_logons)", "INFO",
        ))
        gap = abs(float(hourly_stats["off_hours_share_empirical_pct"]) - feature_off_pct)
        rows.append(_metric_row(
            "Q3 - Nghỉ sinh học", "off_hours_definition_gap_pp", round(gap, 4), "điểm %",
            "Sai lệch giữa đặc trưng và dữ liệu thô (khác quyền số giữa tài khoản); ngưỡng ±2 điểm %",
            "OK" if gap <= 2.0 else "WARN",
        ))
    else:
        rows.append(_metric_row(
            "Q3 - Nghỉ sinh học", "hourly_check", "N/A", "-",
            "Thiếu data/interim -> bỏ qua nhịp giờ (chạy lại khi có dữ liệu thô)", "SKIP",
        ))

    # ---------------- Q5: đặc trưng phản ánh chu kỳ ----------------
    features_only = feature_table[feature_table["role"] == "feature"]
    reflecting = features_only[features_only["reflects_weekly_cycle"] == "YES"]
    rows.append(_metric_row(
        "Q5 - Đặc trưng", "n_features_tested", int(features_only.shape[0]), "đặc trưng",
        "Số đặc trưng được kiểm định (không tính total_logons)", "INFO",
    ))
    rows.append(_metric_row(
        "Q5 - Đặc trưng", "n_features_reflecting_weekly_cycle", int(reflecting.shape[0]), "đặc trưng",
        f"|Cliff's delta| ≥ {THRESHOLDS['cliff_delta_small']} giữa ngày thường và cuối tuần-like",
        "INFO",
    ))
    if not features_only.empty:
        strongest = features_only.iloc[0]
        weakest = features_only.iloc[-1]
        rows.append(_metric_row(
            "Q5 - Đặc trưng", "strongest_feature", strongest["feature"], "tên đặc trưng",
            f"Cliff's delta = {strongest['cliffs_delta']}, chênh lệch "
            f"{strongest['weekend_minus_weekday_pct']}% (cuối tuần vs ngày thường), "
            f"p = {strongest['mannwhitney_p']}",
            "INFO",
        ))
        rows.append(_metric_row(
            "Q5 - Đặc trưng", "weakest_feature", weakest["feature"], "tên đặc trưng",
            f"Cliff's delta = {weakest['cliffs_delta']}, chênh lệch "
            f"{weakest['weekend_minus_weekday_pct']}%, p = {weakest['mannwhitney_p']}",
            "INFO",
        ))
    return pd.DataFrame.from_records(rows)


# ==============================================================================
# 9. VẼ BIỂU ĐỒ
# ==============================================================================
def save_figure(fig: "plt.Figure", filename: str) -> None:
    """Lưu 1 hình ra docs/feature_engineering/figures + artifacts (300 DPI)."""
    for folder in (OUTPUT_FIG_DIR, ARTIFACT_DIR):
        path = folder / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[✓] Đã lưu biểu đồ tại: {path.resolve()}")


def build_week_phase_matrix(daily: pd.DataFrame) -> np.ndarray:
    """Ma trận [tuần x pha] = % lưu lượng của pha trong tuần đó.

    Tuần THIẾU NGÀY bị để trống (NaN) vì tỷ trọng % không so sánh được với tuần đủ 7 ngày.
    """
    ordered = daily.sort_values("day")
    n_blocks = int(np.ceil(ordered.shape[0] / CYCLE_LEN))
    matrix = np.full((n_blocks, CYCLE_LEN), np.nan)
    for b in range(n_blocks):
        block = ordered.iloc[b * CYCLE_LEN : (b + 1) * CYCLE_LEN]
        if block.shape[0] < CYCLE_LEN:
            continue
        total = float(block["total_system_volume"].sum())
        if total <= 0:
            continue
        for _, row in block.iterrows():
            matrix[b, int(row["phase"])] = 100.0 * float(row["total_system_volume"]) / total
    return matrix


def plot_main_figure(
    daily: pd.DataFrame,
    weekend_like: Sequence[int],
    weekend_summary: Dict[str, float],
    cycle_r2: float,
    show_rolling: bool = False,
) -> None:
    """Hình chính 1 panel: lưu lượng hệ thống theo ngày + mốc cuối tuần-like.

    Các panel phụ trước đây (profile theo pha chu kỳ, ACF lag 1..14, failure_ratio &
    off_hours_ratio theo ngày) đã được bỏ theo yêu cầu — số liệu tương ứng vẫn nằm
    nguyên trong docs/feature_engineering/tables/*.csv và
    reports/week2/temporal_stability_check.md.
    """
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(16, 6.0))
    days = daily["day"].to_numpy()
    vol_m = daily["total_system_volume"].to_numpy(dtype=float) / 1e6
    phases_txt = ", ".join(str(p) for p in weekend_like)

    # --- PANEL DUY NHẤT: LƯU LƯỢNG THEO NGÀY + MỐC CUỐI TUẦN ---
    ax1.bar(days, vol_m, color=COLOR_MAIN, alpha=0.18, width=0.7)
    ax1.plot(days, vol_m, color=COLOR_MAIN, linewidth=1.6, marker="o", markersize=3.5,
             label="Lưu lượng hệ thống (triệu logon)")
    if show_rolling:
        ax1.plot(days, daily["rolling7_volume"].to_numpy(dtype=float) / 1e6, color=COLOR_ACCENT,
                 linestyle="--", linewidth=2.2, label="Trung bình trượt 7 ngày (nền)")
    ax1.scatter(
        daily.loc[daily["is_weekend_like"] == 1, "day"],
        daily.loc[daily["is_weekend_like"] == 1, "total_system_volume"] / 1e6,
        color=COLOR_WEEKEND, marker="v", s=60, zorder=5,
        label=f"Cuối tuần-like (pha {phases_txt})",
    )
    ax1.set_title(
        f"Lưu lượng theo ngày — sụt {weekend_summary['weekend_drop_pct']:.1f}% vào cuối tuần-like",
        fontsize=11.5, fontweight="bold", pad=10, color=COLOR_TITLE,
    )
    ax1.set_xlabel("Chỉ số ngày (Day)", fontsize=10)
    ax1.set_ylabel("Lưu lượng (triệu logon/ngày)", fontsize=10)
    ax1.set_xlim(min(days) - 1, max(days) + 1)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        "KIỂM TRA TÍNH ỔN ĐỊNH CHU KỲ THEO THỜI GIAN (TEMPORAL STATIONARITY)\n"
        f"UEBA Windows Event Log — {len(days)} ngày | chỉ riêng pha tuần đã giải thích "
        f"{100.0 * cycle_r2:.1f}% phương sai lưu lượng",
        fontsize=13.5, fontweight="bold", y=0.99, color=COLOR_TITLE,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, "temporal_stability_daily.png")
    plt.close(fig)


def plot_weekly_cycle_figure(
    daily: pd.DataFrame, feature_table: pd.DataFrame, weekend_like: Sequence[int]
) -> None:
    """Hình 2: heatmap [tuần x pha] + hiệu ứng Cliff's delta của từng đặc trưng (Q5)."""
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # --- PANEL 1: HEATMAP ỔN ĐỊNH CHU KỲ THEO TỪNG TUẦN ---
    matrix = build_week_phase_matrix(daily)
    y_labels = [
        f"Tuần {i + 1}" + (" (thiếu ngày — không tính)" if np.isnan(matrix[i]).any() else "")
        for i in range(matrix.shape[0])
    ]
    x_labels = [
        f"Pha {p}" + (" *" if p in set(weekend_like) else "") for p in range(CYCLE_LEN)
    ]
    sns.heatmap(
        matrix, ax=axes[0], cmap="YlGnBu", annot=True, fmt=".1f", linewidths=0.6,
        linecolor="white", cbar_kws={"label": "% lưu lượng của pha trong tuần"},
        mask=~np.isfinite(matrix), xticklabels=x_labels, yticklabels=y_labels,
    )
    axes[0].set_title(
        "1. Tỷ trọng lưu lượng theo pha trong từng tuần (%)\n"
        "(cột có dấu * = pha weekend-like suy luận; các tuần phải lặp lại cùng hình dạng)",
        fontsize=11.5, fontweight="bold", pad=10, color=COLOR_TITLE,
    )
    axes[0].set_xlabel("Pha chu kỳ (day_index % 7)", fontsize=10)
    axes[0].set_ylabel("Tuần dữ liệu", fontsize=10)
    axes[0].tick_params(axis="y", rotation=0)

    # --- PANEL 2: HIỆU ỨNG CỦA TỪNG ĐẶC TRƯNG ---
    ordered = feature_table.sort_values("cliffs_delta")
    bar_colors = [
        COLOR_WEEKEND if row["role"] == "volume" else (
            COLOR_MAIN if row["reflects_weekly_cycle"] == "YES" else COLOR_NEUTRAL
        )
        for _, row in ordered.iterrows()
    ]
    axes[1].barh(ordered["feature"], ordered["cliffs_delta"], color=bar_colors, alpha=0.9)
    for sign in (-THRESHOLDS["cliff_delta_small"], THRESHOLDS["cliff_delta_small"]):
        axes[1].axvline(sign, color=COLOR_ACCENT, linestyle="--", linewidth=1.2)
    axes[1].axvline(0.0, color=COLOR_NEUTRAL, linewidth=0.9)
    axes[1].set_title(
        "2. Mức độ phản ánh chu kỳ tuần của từng đặc trưng (Cliff's delta)\n"
        "(âm = cuối tuần NHỎ hơn ngày thường; đường nét đứt = ngưỡng hiệu ứng "
        f"{THRESHOLDS['cliff_delta_small']})",
        fontsize=11.5, fontweight="bold", pad=10, color=COLOR_TITLE,
    )
    axes[1].set_xlabel("Cliff's delta (cuối tuần-like vs ngày thường)", fontsize=10)
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.45)

    plt.tight_layout()
    save_figure(fig, "temporal_weekly_cycle.png")
    plt.close(fig)


def plot_hourly_figure(
    hourly_profile: pd.DataFrame, hourly_stats: Dict[str, float], win: Dict[str, int]
) -> None:
    """Hình 3: nhịp giờ trong ngày (khoảng nghỉ sinh học) — 1 panel duy nhất.

    Panel phụ trước đây (mức sụt cuối tuần-like theo từng giờ) đã được bỏ theo yêu cầu;
    số liệu vẫn nằm trong docs/feature_engineering/tables/hourly_rhythm_summary.csv và
    reports/week2/temporal_stability_check.md.
    """
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(16, 6.0))
    hours = hourly_profile["hour"].to_numpy()

    # --- PANEL DUY NHẤT: NHỊP GIỜ NGÀY THƯỜNG vs CUỐI TUẦN-LIKE ---
    ax1.axvspan(-0.5, win["off_hours_end"] + 0.5, color="#B0BEC5", alpha=0.28,
                label=f"Ngoài giờ hành chính ({win['off_hours_start']}h-{win['off_hours_end']}h)")
    ax1.axvspan(win["off_hours_start"] - 0.5, 23.5, color="#B0BEC5", alpha=0.28)
    ax1.axvspan(NIGHT_START - 0.5, NIGHT_END + 0.5, color=COLOR_MAIN, alpha=0.12,
                label=f"Đêm {NIGHT_START}h-{NIGHT_END}h (nghỉ sinh học)")
    ax1.plot(hours, hourly_profile["ok_per_weekday_like_day"] / 1e3, color=COLOR_MAIN,
             marker="o", markersize=5, linewidth=2.0, label="Ngày thường (nghìn logon/giờ)")
    ax1.plot(hours, hourly_profile["ok_per_weekend_like_day"] / 1e3, color=COLOR_WEEKEND,
             marker="s", markersize=5, linewidth=2.0, linestyle="--",
             label="Cuối tuần-like (nghìn logon/giờ)")
    ax1.annotate(
        f"đêm thấp nhất: {hourly_stats['night_min_hour']}h "
        f"(sụt {hourly_stats['night_dip_pct']:.1f}% so với giờ cao điểm {hourly_stats['peak_hour']}h)",
        xy=(hourly_stats["night_min_hour"], hourly_stats["night_min_ok_per_day"] / 1e3),
        xytext=(0.5, float(hourly_profile["ok_per_day"].max()) / 1e3 * 1.22),
        fontsize=9, color=COLOR_MAIN, fontweight="bold",
    )
    ax1.set_title("Nhịp giờ trong ngày — ban đêm là khoảng nghỉ sinh học tự nhiên",
                  fontsize=11.5, fontweight="bold", pad=10, color=COLOR_TITLE)
    ax1.set_xlabel("Giờ trong ngày (0h-23h)", fontsize=10)
    ax1.set_ylabel("Logon 4624 (nghìn sự kiện/giờ)", fontsize=10)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_ylim(0, float(hourly_profile["ok_per_day"].max()) / 1e3 * 1.34)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    fig.suptitle(
        "NHỊP SINH HỌC THEO GIỜ & MỨC SỤT CUỐI TUẦN (BẰNG CHỨNG TỪ DỮ LIỆU THÔ EVENT-LEVEL)",
        fontsize=13, fontweight="bold", y=0.99, color=COLOR_TITLE,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    save_figure(fig, "temporal_hourly_rhythm.png")
    plt.close(fig)


# ==============================================================================
# 10. GHI BẢNG CSV
# ==============================================================================
def write_table(frame: pd.DataFrame, filename: str) -> None:
    """Ghi 1 bảng ra docs/feature_engineering/tables (không index)."""
    path = OUTPUT_TAB_DIR / filename
    frame.to_csv(path, index=False)
    print(f"[✓] Đã lưu bảng tại: {path.resolve()}")


def write_tables(
    daily: pd.DataFrame,
    phase_profile: pd.DataFrame,
    hourly_profile: Optional[pd.DataFrame],
    feature_table: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Ghi toàn bộ bảng: ngày, pha chu kỳ, nhịp giờ, hiệu ứng đặc trưng, metrics."""
    write_table(daily, "daily_activity_summary.csv")
    write_table(phase_profile, "weekly_cycle_summary.csv")
    if hourly_profile is not None:
        write_table(hourly_profile, "hourly_rhythm_summary.csv")
    write_table(feature_table, "weekday_vs_weekend_features.csv")
    write_table(metrics, "temporal_stationarity_metrics.csv")


def metric_lookup(metrics: pd.DataFrame) -> Dict[str, Any]:
    """Chuyển bảng metrics thành dict {metric: value} để tiện trích số vào báo cáo."""
    return dict(zip(metrics["metric"], metrics["value"]))


# ==============================================================================
# 11. BÁO CÁO MARKDOWN (số liệu lấy trực tiếp từ bảng metrics -> không bị lệch)
# ==============================================================================
def write_markdown_report(
    metrics: pd.DataFrame,
    daily: pd.DataFrame,
    feature_table: pd.DataFrame,
    weekend_like: Sequence[int],
    win: Dict[str, int],
    data_path: Path,
    report_path: Path,
) -> None:
    """Sinh báo cáo Markdown cho phần kiểm định temporal stationarity."""
    look = metric_lookup(metrics)
    features_only = feature_table[feature_table["role"] == "feature"].reset_index(drop=True)
    context = {
        "n_days": int(daily.shape[0]),
        "n_weekend_like_days": int((daily["is_weekend_like"] == 1).sum()),
        "weekend_like_txt": ", ".join(str(p) for p in weekend_like),
        "hourly_done": "hourly_check" not in look,
        "data_path": str(data_path),
        "win": win,
    }
    lines: List[str] = []
    lines += _report_header(context)
    lines += _report_conclusions(look, metrics, context, weekend_like)
    lines += _report_methodology()
    lines += _report_metric_tables(metrics, features_only)
    lines += _report_recommendations(look, features_only)
    lines += _report_limits()
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[✓] Đã lưu báo cáo tại: {report_path.resolve()}")


def _report_header(ctx: Dict[str, Any]) -> List[str]:
    win = ctx["win"]
    return [
        "# Kiểm tra Tính ổn định chu kỳ theo thời gian (Temporal Stationarity)",
        "",
        "Sinh bởi `scripts/feature_engineering/plot_temporal_stability.py` — mọi số liệu dưới đây "
        "đều được script tính lại từ dữ liệu mỗi lần chạy (không viết tay).",
        "",
        "| Tham số | Giá trị |",
        "|---|---|",
        f"| Dữ liệu đầu vào | `{ctx['data_path']}` |",
        f"| Số ngày | {ctx['n_days']} (trong đó {ctx['n_weekend_like_days']} ngày cuối tuần-like) |",
        f"| Pha chu kỳ | `(day - 1) % 7`, pha weekend-like = {ctx['weekend_like_txt']} |",
        f"| Cửa sổ off-hours (config) | {win['off_hours_start']}h-{win['off_hours_end']}h |",
        f"| Giờ làm việc (config) | {win['work_hours_start']}h-{win['work_hours_end']}h |",
        f"| Khoảng nghỉ sinh học (đêm) | {NIGHT_START}h-{NIGHT_END}h |",
        f"| Dữ liệu thô event-level | {'có (đã kiểm chứng nhịp giờ)' if ctx['hourly_done'] else 'THIẾU -> bỏ qua Q3'} |",
        "",
        "## 1. Kết luận nhanh",
        "",
    ]


def _verdict(condition: bool, strong: str = "RÕ", weak: str = "CHƯA RÕ") -> str:
    return strong if condition else weak


def _report_conclusions(
    look: Dict[str, Any],
    metrics: pd.DataFrame,
    ctx: Dict[str, Any],
    weekend_like: Sequence[int],
) -> List[str]:
    thr = THRESHOLDS
    lines = [
        f"1. **Chu kỳ 7 ngày: {_verdict(float(look['acf_lag7']) >= thr['acf_lag7_strong'])}** — "
        f"ACF lag 7 = {look['acf_lag7']}, lag 14 = {look['acf_lag14']}, đỉnh ACF tại lag "
        f"{look['acf_peak_lag']}. Chỉ riêng pha tuần đã giải thích "
        f"**{look['phase_variance_explained_pct']}%** phương sai log(lưu lượng/ngày); phần dư "
        f"{look['residual_cv_after_phase_pct']}% mới là nền để dò bất thường thật.",
        "",
        f"2. **Sụt giảm cuối tuần: {look['weekend_drop_pct']}%** (lưu lượng trung bình cuối tuần-like "
        f"so với ngày thường). Tập pha weekend-like = {look['weekend_like_phases']}; số tài khoản "
        f"hoạt động giảm {look['weekend_active_accounts_drop_pct']}% theo cùng nhịp; "
        f"**{look['week_block_consistency_pct']}%** số tuần trọn vẹn lặp lại đúng hình dạng này.",
        "",
    ]
    if ctx["hourly_done"]:
        lines += [
            f"3. **Khoảng nghỉ sinh học: "
            f"{_verdict(float(look['night_dip_pct']) >= thr['night_dip_min_pct'])}** — giờ thấp nhất "
            f"trong đêm sụt **{look['night_dip_pct']}%** so với giờ cao điểm; tỷ trọng sự kiện "
            f"{NIGHT_START}h-{NIGHT_END}h = {look['night_share_pct']}%. "
            f"Cuối tuần sụt {look['weekend_drop_work_hours_pct']}% trong giờ làm việc nhưng chỉ "
            f"{look['weekend_drop_night_hours_pct']}% ban đêm — đúng dấu hiệu con người nghỉ, "
            "tài khoản máy vẫn chạy 24/7.",
            "",
            f"4. **Đặc trưng `off_hours_ratio` khớp dữ liệu thô**: trung bình đặc trưng = "
            f"{look['off_hours_ratio_feature_pct']}% so với tỷ trọng thô ngoài cửa sổ "
            f"{ctx['win']['off_hours_start']}h-{ctx['win']['off_hours_end']}h = "
            f"{look['off_hours_share_empirical_pct']}% (lệch {look['off_hours_definition_gap_pp']} "
            "điểm %).",
            "",
        ]
    else:
        lines += [
            "3. **Khoảng nghỉ sinh học: CHƯA KIỂM CHỨNG** — thiếu `data/interim` (cần "
            "`event_4624/`, `event_4625/`). Chạy lại script khi có dữ liệu thô.",
            "",
        ]
    drift_rows = metrics[metrics["check_group"].str.startswith("Q4")]
    drift_names = [str(row.metric).split(":")[0] for row in drift_rows.itertuples(index=False)]
    drift_txt = ", ".join(
        f"{name}: rho = {row.value}" for name, row in zip(drift_names, drift_rows.itertuples(index=False))
    )
    warned = [
        name
        for name, row in zip(drift_names, drift_rows.itertuples(index=False))
        if str(row.status) != "OK"
    ]
    if warned:
        verdict_drift = (
            f"CÓ dấu hiệu trôi nhẹ ở {', '.join('`' + w + '`' for w in warned)} ⇒ vẫn phải dùng "
            "baseline trượt / cửa sổ giới hạn và bật cơ chế cập nhật lại nền định kỳ, không dùng "
            "baseline cố định cho toàn bộ chuỗi."
        )
    else:
        verdict_drift = (
            "Nền không trôi ⇒ biến động theo chu kỳ phải được mô hình hoá như seasonality, "
            "không phải anomaly."
        )
    lines += [
        f"5. **Trôi nền (drift)**: {drift_txt}. {verdict_drift}",
        "",
        f"6. **Đặc trưng phản ánh chu kỳ**: {look['n_features_reflecting_weekly_cycle']}/"
        f"{look['n_features_tested']} đặc trưng có |Cliff's delta| ≥ {thr['cliff_delta_small']}; "
        f"mạnh nhất `{look['strongest_feature']}`, yếu nhất `{look['weakest_feature']}`.",
        "",
    ]
    return lines


def _report_methodology() -> List[str]:
    thr = THRESHOLDS
    return [
        "## 2. Phương pháp kiểm định",
        "",
        "| Câu hỏi | Cách làm | Ngưỡng chấp nhận |",
        "|---|---|---|",
        "| Q1 — nhịp tuần | ACF lag 1..14 trên log(lưu lượng/ngày) + R² của mô hình chỉ-dùng-pha | "
        f"ACF lag 7 ≥ {thr['acf_lag7_strong']}; R² pha ≥ {thr['phase_r2_strong_pct']}% |",
        "| Q2 — cuối tuần | 2 pha lưu lượng trung bình thấp nhất = weekend-like; kiểm tra lặp lại theo từng tuần | "
        f"sụt ≥ {thr['weekend_drop_min_pct']}%; nhất quán ≥ {thr['phase_consistency_strong_pct']}% số tuần |",
        "| Q3 — nghỉ sinh học | Đếm sự kiện thô theo giờ (`Time % 86400 // 3600`), tách ngày thường / cuối tuần-like | "
        f"đêm sụt ≥ {thr['night_dip_min_pct']}% so với giờ cao điểm |",
        "| Q4 — trôi nền | Spearman(ngày, nền trượt 7 ngày) + hồi quy Theil-Sen + so sánh 2 nửa dữ liệu | "
        f"|rho| ≤ {thr['drift_rho_max']}; |%/tuần| ≤ {thr['drift_abs_slope_max_pct']} |",
        "| Q5 — phản ánh chu kỳ | Trung bình có trọng số theo ngày của từng đặc trưng, so sánh ngày thường vs cuối tuần-like | "
        f"|Cliff's delta| ≥ {thr['cliff_delta_small']} (hiệu ứng 'small') |",
        "",
    ]


def _report_metric_tables(metrics: pd.DataFrame, features_only: pd.DataFrame) -> List[str]:
    lines = [
        "## 3. Bảng chỉ số then chốt",
        "",
        "| Nhóm | Chỉ số | Giá trị | Đơn vị | Trạng thái |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row.check_group} | {row.metric} | {row.value} | {row.unit} | {row.status} |"
        for row in metrics.itertuples(index=False)
    ]
    lines += [
        "",
        "> Diễn giải chi tiết từng dòng (kèm ngưỡng): "
        "`docs/feature_engineering/tables/temporal_stationarity_metrics.csv`.",
        "",
        "## 4. Top đặc trưng phản ánh chu kỳ tuần (Q5)",
        "",
        "| # | Đặc trưng | Ngày thường | Cuối tuần-like | Chênh lệch | Cliff's delta | p (MWU) | Phản ánh chu kỳ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {idx} | `{row.feature}` | {row.weekday_like_mean} | "
        f"{row.weekend_like_mean} | {row.weekend_minus_weekday_pct}% | {row.cliffs_delta} | "
        f"{row.mannwhitney_p} | {row.reflects_weekly_cycle} |"
        for idx, row in enumerate(features_only.head(5).itertuples(index=False), start=1)
    ]
    lines.append("")
    return lines


def _report_recommendations(look: Dict[str, Any], features_only: pd.DataFrame) -> List[str]:
    thr = THRESHOLDS
    no_cycle = features_only.loc[features_only["reflects_weekly_cycle"] == "no", "feature"].tolist()
    lines = [
        "## 5. Khuyến nghị để tránh cảnh báo rác định kỳ",
        "",
        f"1. **Baseline phải theo mùa vụ**: với mức sụt cuối tuần {look['weekend_drop_pct']}% và R² pha "
        f"tuần {look['phase_variance_explained_pct']}%, dùng seasonal-naive `t-7` hoặc trung bình trượt "
        "7 ngày làm nền; **không** dùng baseline toàn cục hoặc cửa sổ < 7 ngày.",
        "2. **Cửa sổ huấn luyện tối thiểu 14 ngày** để mỗi pha có ≥ 2 quan sát trước khi tính baseline.",
        "3. **Giữ `off_hours_ratio`** làm đặc trưng mã hoá nhịp sinh học (Q3 đã kiểm chứng khớp dữ liệu "
        "thô); chỉ bật thêm cờ `is_weekend`/pha như covariate sau khi xác nhận calendar origin "
        "(`docs/plan/lanl_eda_implementation_plan.md` §4.3).",
    ]
    if no_cycle:
        listed = ", ".join("`" + name + "`" for name in no_cycle)
        lines.append(
            f"4. **Rủi ro còn lại**: các đặc trưng ít phản ánh chu kỳ ({listed}) có thể sinh alert khi "
            "lưu lượng cuối tuần thấp — cần chuẩn hoá theo baseline cá nhân hoặc thêm pha ngày khi "
            f"chấm điểm (ngưỡng hiệu ứng đang dùng: |Cliff's delta| ≥ {thr['cliff_delta_small']})."
        )
    lines.append("")
    return lines


def _report_limits() -> List[str]:
    return [
        "## 6. Giới hạn & mức độ chắc chắn",
        "",
        "- **Chưa xác nhận calendar origin**: dữ liệu LANL chỉ có `Time` tương đối nên 2 pha thấp nhất "
        "được gọi là “cuối tuần-like”. Khi có ngày lịch thật, chỉ cần đổi nhãn pha — toàn bộ số liệu "
        "không thay đổi.",
        "- p-value Mann-Whitney tính trên các quan sát ngày (có tự tương quan) chỉ mang tính tham chiếu; "
        "kết luận dựa trên **độ lớn hiệu ứng (Cliff's delta)**, ACF và độ lặp lại theo từng tuần.",
        "- Nhịp giờ (Q3) tính trên toàn bộ sự kiện 4624/4625, chưa tách theo `entity_type` ở mức "
        "event-level; khác biệt máy/người chỉ được so sánh gián tiếp qua đặc trưng.",
        "- Chưa loại ngày lễ/ngày bảo trì do thiếu lịch; nếu tồn tại, chúng là điểm ngoại lai trong "
        "profile pha và sẽ làm giảm chỉ số nhất quán theo tuần.",
        "",
    ]


# ==============================================================================
# 12. ĐIỀU PHỐI CHÍNH
# ==============================================================================
def run_checks(
    data_path: Path,
    interim_dir: Path,
    config_path: Path,
    skip_hourly: bool,
    report_path: Path,
    show_rolling: bool = False,
) -> None:
    """Chạy 5 câu hỏi kiểm định, xuất hình + bảng + báo cáo."""
    print("=" * 96)
    print("KIỂM TRA TÍNH ỔN ĐỊNH CHU KỲ THEO THỜI GIAN (TEMPORAL STATIONARITY)")
    print("=" * 96)
    if not data_path.exists():
        print(f"[LỖI] Không tìm thấy ma trận đặc trưng tại: {data_path.resolve()}")
        sys.exit(1)

    cfg = load_yaml_config(config_path)
    win = resolve_time_windows(cfg)
    print(
        f"[INFO] Cửa sổ giờ (config): off {win['off_hours_start']}h-{win['off_hours_end']}h | "
        f"work {win['work_hours_start']}h-{win['work_hours_end']}h | "
        f"số ngày cuối tuần/tuần = {win['weekend_day_count']}"
    )
    print(f"[INFO] Đọc dữ liệu từ: {data_path.resolve()}")
    df = pl.read_parquet(data_path).to_pandas()
    for required in ("day", "total_logons"):
        if required not in df.columns:
            print(f"[LỖI] Ma trận thiếu cột bắt buộc '{required}'.")
            sys.exit(1)
    print(
        f"[INFO] Ma trận: {len(df):,} dòng x {df.shape[1]} cột | "
        f"{df['day'].nunique()} ngày ({int(df['day'].min())}..{int(df['day'].max())})"
    )

    # --- BƯỚC 1: tổng hợp ngày + suy luận pha cuối tuần ---
    daily = build_daily_summary(df)
    profile = build_phase_profile(daily)
    weekend_like = identify_weekend_like_phases(profile, win["weekend_day_count"])
    profile, weekend_summary = finalize_phase_profile(profile, weekend_like, daily)
    daily["is_weekend_like"] = daily["phase"].isin(list(weekend_like)).astype(int)
    consistency = week_block_consistency(daily, weekend_like, win["weekend_day_count"])
    print(
        f"[KẾT QUẢ Q2] Pha weekend-like = {weekend_like} | sụt "
        f"{weekend_summary['weekend_drop_pct']:.2f}% | nhất quán theo tuần = "
        f"{consistency[0]:.1f}% ({consistency[1]}/{consistency[2]} tuần)"
    )

    # --- BƯỚC 2: chu kỳ 7 ngày (ACF + R² theo pha) ---
    log_volume = np.log1p(daily["total_system_volume"].to_numpy(dtype=float))
    acf_values = autocorrelation(log_volume, MAX_LAG)
    cycle_r2, residual_cv = phase_variance_explained(log_volume, daily["phase"].to_numpy())
    print(
        f"[KẾT QUẢ Q1] ACF lag 7 = {acf_values[CYCLE_LEN - 1]:.3f} (lag 14 = "
        f"{acf_values[2 * CYCLE_LEN - 1]:.3f}) | R² theo pha = {100.0 * cycle_r2:.2f}% | "
        f"phần dư = {100.0 * residual_cv:.2f}%"
    )

    # --- BƯỚC 3: trôi nền + hiệu ứng đặc trưng ---
    drift_records = drift_metrics(daily)
    feature_table = build_feature_cycle_table(df, daily, weekend_like)
    print("[KẾT QUẢ Q4] Trôi nền:")
    for rec in drift_records:
        print(
            f"           - {rec['metric']}: rho = {rec['rolling7_spearman_rho']}, "
            f"{rec['theil_sen_slope_pct_per_week']}%/tuần -> {rec['status']}"
        )
    n_reflect = int((feature_table["reflects_weekly_cycle"] == "YES").sum())
    print(
        f"[KẾT QUẢ Q5] {n_reflect}/{feature_table.shape[0]} dòng (kể cả volume) có "
        f"|Cliff's delta| ≥ {THRESHOLDS['cliff_delta_small']}"
    )

    # --- BƯỚC 4: nhịp giờ / khoảng nghỉ sinh học (cần dữ liệu thô) ---
    hourly_profile: Optional[pd.DataFrame] = None
    hourly_stats: Optional[Dict[str, float]] = None
    if skip_hourly:
        print("[INFO] Bỏ qua nhịp giờ theo yêu cầu --skip-hourly.")
    else:
        hourly = scan_hourly_counts(interim_dir, daily["day"].tolist())
        if hourly is not None:
            hourly_profile = build_hourly_profile(hourly, daily, weekend_like, win)
            hourly_stats = hourly_summary(hourly_profile, win)
            print(
                f"[KẾT QUẢ Q3] Đêm {NIGHT_START}h-{NIGHT_END}h sụt "
                f"{hourly_stats['night_dip_pct']:.2f}% so với giờ cao điểm "
                f"{hourly_stats['peak_hour']}h | sụt cuối tuần: "
                f"{hourly_stats['weekend_drop_work_hours_pct']:.2f}% (giờ làm việc) vs "
                f"{hourly_stats['weekend_drop_night_hours_pct']:.2f}% (ban đêm)"
            )

    # --- BƯỚC 5: bảng chỉ số + xuất artifacts ---
    metrics = assemble_metrics(
        acf_values, cycle_r2, residual_cv, weekend_like, weekend_summary, consistency,
        daily, feature_table, drift_records, hourly_stats,
    )
    plot_main_figure(daily, weekend_like, weekend_summary, cycle_r2, show_rolling=show_rolling)
    plot_weekly_cycle_figure(daily, feature_table, weekend_like)
    if hourly_profile is not None and hourly_stats is not None:
        plot_hourly_figure(hourly_profile, hourly_stats, win)
    write_tables(daily, profile, hourly_profile, feature_table, metrics)
    write_markdown_report(
        metrics, daily, feature_table, weekend_like, win, data_path, report_path
    )

    # --- BƯỚC 6: in tổng kết ra console ---
    print("\n" + "=" * 96)
    print("PROFILE THEO PHA CHU KỲ (day_index % 7):")
    print("=" * 96)
    print(profile.to_string(index=False))
    print("\n" + "=" * 96)
    print("BẢNG CHỈ SỐ KIỂM ĐỊNH (temporal_stationarity_metrics.csv):")
    print("=" * 96)
    print(metrics.to_string(index=False))
    print("\n" + "=" * 96)
    print("[SUCCESS] Hoàn thành kiểm tra tính ổn định chu kỳ theo thời gian!")
    print(f"          Bảng   : {OUTPUT_TAB_DIR.resolve()}")
    print(f"          Hình   : {OUTPUT_FIG_DIR.resolve()}")
    print(f"          Báo cáo: {report_path.resolve()}")
    print("=" * 96 + "\n")


def main() -> None:
    """Điểm vào CLI của script."""
    parser = argparse.ArgumentParser(
        description=(
            "Kiểm tra tính ổn định chu kỳ theo thời gian (temporal stationarity): nhịp tuần, "
            "sụt giảm cuối tuần và khoảng nghỉ sinh học."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Đường dẫn file ma trận đặc trưng THÔ (feature_matrix_raw.parquet)",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_DIR,
        help="Thư mục dữ liệu thô event_4624/ + event_4625/ (dùng cho phần nhịp giờ)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Đường dẫn config YAML (off_hours_start/end, work_hours_start/end, weekend_days)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_DIR / "temporal_stability_check.md",
        help="Đường dẫn báo cáo Markdown đầu ra",
    )
    parser.add_argument(
        "--skip-hourly",
        action="store_true",
        help="Bỏ qua phần nhịp giờ từ dữ liệu thô (chạy nhanh khi không có data/interim)",
    )
    parser.add_argument(
        "--show-rolling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hiển thị đường nét đứt trung bình trượt 7 ngày (mặc định: tắt)",
    )
    args = parser.parse_args()
    run_checks(
        args.data_path,
        args.interim_dir,
        args.config,
        args.skip_hourly,
        args.report_path,
        show_rolling=args.show_rolling,
    )


if __name__ == "__main__":
    main()
