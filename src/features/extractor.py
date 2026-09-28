"""
Account x Day Feature Extractor.
Trích xuất ma trận đặc trưng hành vi (Tài khoản × Ngày) từ log interim sự kiện 4624 & 4625.
Sử dụng Polars để tối ưu hóa tốc độ xử lý và bộ nhớ.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import polars as pl
import yaml

logger = logging.getLogger("ueba_benchmark.features.extractor")

DEFAULT_FEATURE_CFG: Dict[str, Any] = {
    "off_hours_start": 18,
    "off_hours_end": 7,
    "work_hours_start": 8,
    "work_hours_end": 17,
    "entity_type": {
        "machine_suffix": "$",
        "user_prefix": "User",
        "admin_accounts": ["administrator"],
        "system_accounts": [
            "system", "local service", "network service",
            "anonymous", "anonymous logon", "-",
        ],
        "service_accounts": ["appservice", "scanner", "winservice"],
    },
    "filter": {"drop_machine_accounts": False, "drop_system_accounts": False},
}

RAW_ORDERED_COLS: List[str] = [
    "DomainName", "UserName", "day", "entity_type",
    "total_logons",
    "failure_ratio", "failure_locked_out_share",
    "off_hours_ratio",
    "interarrival_dt_mean", "delta_t_cv", "same_second_share", "is_single_event",
    "interactive_ratio", "rare_logon_type_count",
    "ntlm_ratio",
    "distinct_hosts", "distinct_sources_count",
    "missing_source_ratio", "remote_logon_ratio", "custom_proc_share",
]


def resolve_feature_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Gộp config từ system_config.yaml với giá trị mặc định."""
    feats = dict(cfg.get("features", {}) or {})
    merged = {k: v for k, v in DEFAULT_FEATURE_CFG.items() if k != "entity_type"}
    for key in ("off_hours_start", "off_hours_end", "work_hours_start", "work_hours_end"):
        if feats.get(key) is not None:
            merged[key] = int(feats[key])

    et = dict(DEFAULT_FEATURE_CFG["entity_type"])
    et.update(feats.get("entity_type", {}) or {})
    merged["entity_type"] = et

    legacy_ignore = (cfg.get("preprocessing", {}) or {}).get("ignore_system_accounts")
    if legacy_ignore and not (feats.get("entity_type", {}) or {}).get("system_accounts"):
        merged["entity_type"]["system_accounts"] = [str(x).strip().casefold() for x in legacy_ignore]

    flt = dict(DEFAULT_FEATURE_CFG["filter"])
    flt.update(feats.get("filter", {}) or {})
    legacy_machine = (cfg.get("preprocessing", {}) or {}).get("filter_machine_accounts")
    if legacy_machine is not None and "drop_machine_accounts" not in (feats.get("filter", {}) or {}):
        flt["drop_machine_accounts"] = bool(legacy_machine)
    merged["filter"] = flt
    return merged


def entity_type_expr(fcfg: Dict[str, Any]) -> pl.Expr:
    """Luật phân loại entity_type tường minh."""
    et = fcfg["entity_type"]
    low = pl.col("UserName").str.to_lowercase().str.strip_chars()
    return (
        pl.when(low.is_in([str(a).casefold() for a in et["admin_accounts"]])).then(pl.lit("Admin"))
        .when(low.is_in([str(a).casefold() for a in et["system_accounts"]])).then(pl.lit("System"))
        .when(pl.col("UserName").str.ends_with(et["machine_suffix"])).then(pl.lit("Machine"))
        .when(pl.col("UserName").str.starts_with(et["user_prefix"])).then(pl.lit("User"))
        .when(low.is_in([str(a).casefold() for a in et["service_accounts"]])).then(pl.lit("Service"))
        .otherwise(pl.lit("Other"))
        .alias("entity_type")
    )


def extract_features_single_day(
    day: int,
    interim_dir: Path,
    output_daily_dir: Optional[Path] = None,
    feature_cfg: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """
    Trích xuất vector đặc trưng THÔ cho từng cặp (DomainName, UserName, day).
    Đầu ra chuẩn 20 cột (4 khoá/nhãn + 16 đặc trưng thô).
    """
    fcfg = feature_cfg or DEFAULT_FEATURE_CFG
    path_4624 = interim_dir / "event_4624" / f"event_4624_day-{day:02d}.parquet"
    path_4625 = interim_dir / "event_4625" / f"event_4625_day-{day:02d}.parquet"

    if not path_4624.exists() or not path_4625.exists():
        logger.warning(f"Không tìm thấy dữ liệu cho Day {day:02d} trong {interim_dir}, bỏ qua.")
        return pl.DataFrame()

    cols_4624 = [
        "Time", "EventID", "UserName", "LogHost", "LogonType",
        "AuthenticationPackage", "ProcessName", "Source", "DomainName",
    ]
    cols_4625 = cols_4624 + ["FailureReason"]

    # Quét lười
    lf_4624 = (
        pl.scan_parquet(path_4624)
        .select(cols_4624)
        .with_columns(pl.lit(None, dtype=pl.String).alias("FailureReason"))
    )
    lf_4625 = pl.scan_parquet(path_4625).select(cols_4625)
    events = pl.concat([lf_4624, lf_4625], how="diagonal")

    # Làm sạch UserName
    events = events.filter(
        pl.col("UserName").is_not_null()
        & (pl.col("UserName").str.strip_chars() != "")
        & (pl.col("UserName").str.to_lowercase().is_in(["null", "none", "nan"]).not_())
    )

    # Chuẩn hoá DomainName
    events = events.with_columns(
        pl.when(pl.col("DomainName").is_null() | (pl.col("DomainName").str.strip_chars() == ""))
        .then(pl.lit("unknown"))
        .otherwise(pl.col("DomainName").str.strip_chars().str.to_lowercase())
        .alias("DomainName")
    )

    # Giờ trong ngày [0 - 23]
    events = events.with_columns(((pl.col("Time") % 86400) // 3600).cast(pl.Int32).alias("hour"))

    # Sắp xếp và tính delta_t
    df_events = events.sort(["DomainName", "UserName", "Time"]).collect()
    df_events = df_events.with_columns(
        pl.col("Time").diff().over(["DomainName", "UserName"]).alias("delta_t")
    )

    # Tổng hợp theo khoá danh tính
    off_start, off_end = fcfg["off_hours_start"], fcfg["off_hours_end"]
    wk_start, wk_end = fcfg["work_hours_start"], fcfg["work_hours_end"]
    in_off = (pl.col("hour") >= off_start) | (pl.col("hour") <= off_end)

    df_day = df_events.group_by(["DomainName", "UserName"]).agg([
        pl.len().alias("total_logons"),
        (pl.col("EventID") == 4625).sum().alias("failure_count"),
        in_off.sum().alias("off_hours_count"),
        (((pl.col("hour") >= wk_start) & (pl.col("hour") <= wk_end)).sum()).alias("work_hours_count"),
        pl.col("delta_t").mean().alias("_dt_mean"),
        pl.col("delta_t").std().alias("_dt_std"),
        (pl.col("delta_t") == 0).fill_null(False).sum().alias("_same_second_count"),
        (pl.col("LogonType") == 2).sum().alias("_type2_count"),
        (~pl.col("LogonType").is_in([2, 3])).sum().alias("rare_logon_type_count"),
        pl.col("AuthenticationPackage").str.to_lowercase().str.contains("ntlm").fill_null(False).sum().alias("_ntlm_count"),
        pl.col("FailureReason").str.to_lowercase().str.contains("account locked out").fill_null(False).sum().alias("_locked_out_count"),
        pl.col("LogHost").n_unique().alias("distinct_hosts"),
        ((pl.col("Source").is_null()) | (pl.col("Source").str.strip_chars() == "")).sum().alias("_missing_source_count"),
        ((pl.col("Source").is_not_null()) & (pl.col("Source") != pl.col("LogHost"))).sum().alias("_remote_logon_count"),
        pl.col("Source").drop_nulls().n_unique().alias("distinct_sources_count"),
        pl.col("ProcessName").str.to_lowercase().str.starts_with("proc").fill_null(False).sum().alias("_custom_proc_count"),
    ]).with_columns([
        pl.lit(day).cast(pl.Int32).alias("day"),
        entity_type_expr(fcfg),

        (pl.col("off_hours_count") / pl.col("total_logons")).alias("off_hours_ratio"),
        (pl.col("failure_count") / pl.col("total_logons")).alias("failure_ratio"),
        (pl.col("_type2_count") / pl.col("total_logons")).alias("interactive_ratio"),
        (pl.col("_ntlm_count") / pl.col("total_logons")).alias("ntlm_ratio"),
        (pl.col("_missing_source_count") / pl.col("total_logons")).alias("missing_source_ratio"),
        (pl.col("_remote_logon_count") / pl.col("total_logons")).alias("remote_logon_ratio"),
        (pl.col("_custom_proc_count") / pl.col("total_logons")).alias("custom_proc_share"),

        pl.when(pl.col("failure_count") > 0)
        .then(pl.col("_locked_out_count") / pl.col("failure_count"))
        .otherwise(pl.lit(None, dtype=pl.Float64)).alias("failure_locked_out_share"),

        pl.when(pl.col("total_logons") > 1)
        .then(pl.col("_dt_mean"))
        .otherwise(pl.lit(None, dtype=pl.Float64)).alias("interarrival_dt_mean"),

        pl.when(
            (pl.col("total_logons") > 1)
            & pl.col("_dt_mean").is_not_null()
            & (pl.col("_dt_mean") > 0)
        )
        .then(pl.col("_dt_std") / pl.col("_dt_mean"))
        .otherwise(pl.lit(None, dtype=pl.Float64)).alias("delta_t_cv"),

        (pl.col("_same_second_count") / pl.col("total_logons")).alias("same_second_share"),
        (pl.col("total_logons") == 1).cast(pl.UInt8).alias("is_single_event"),
    ])

    # Chọn đúng các cột hợp đồng
    df_day = df_day.select(RAW_ORDERED_COLS)

    # Lọc tài khoản nếu có cấu hình
    flt = fcfg.get("filter", {}) or {}
    if flt.get("drop_machine_accounts"):
        df_day = df_day.filter(pl.col("entity_type") != "Machine")
    if flt.get("drop_system_accounts"):
        df_day = df_day.filter(pl.col("entity_type") != "System")

    if output_daily_dir is not None:
        output_daily_dir.mkdir(parents=True, exist_ok=True)
        daily_file = output_daily_dir / f"user_features_day-{day:02d}.parquet"
        df_day.write_parquet(daily_file, compression="snappy")

    return df_day


def available_days(interim_dir: Path) -> List[int]:
    """Liệt kê các ngày có đủ dữ liệu trong interim."""
    days: List[int] = []
    p_4624 = interim_dir / "event_4624"
    if not p_4624.is_dir():
        return []
    for p in p_4624.glob("event_4624_day-*.parquet"):
        try:
            d = int(p.stem.split("-")[-1])
        except ValueError:
            continue
        if (interim_dir / "event_4625" / f"event_4625_day-{d:02d}.parquet").exists():
            days.append(d)
    return sorted(days)


def build_account_day_matrix(
    start_day: int = 1,
    end_day: Optional[int] = None,
    interim_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """
    Điều phối trích xuất ma trận đặc trưng cho dải ngày [start_day, end_day].
    Ghi kết quả ra thư mục output_dir/ (data/features/raw/).
    """
    interim = Path(interim_dir) if interim_dir else Path("data/interim")
    days_avail = available_days(interim)
    if not days_avail:
        logger.warning(f"Không có dữ liệu hợp lệ trong thư mục interim: {interim}")
        return pl.DataFrame()

    if end_day is None:
        end_day = days_avail[-1]

    out_path = Path(output_dir) if output_dir else Path("data/features/raw")
    out_path.mkdir(parents=True, exist_ok=True)
    daily_dir = out_path / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    fcfg = resolve_feature_config(config or {})

    logger.info(f"Bắt đầu trích xuất đặc trưng từ Day {start_day:02d} đến Day {end_day:02d}...")
    daily_dfs: List[pl.DataFrame] = []
    for d in range(start_day, end_day + 1):
        if d not in days_avail:
            continue
        t0 = time.time()
        df_d = extract_features_single_day(d, interim, daily_dir, feature_cfg=fcfg)
        if df_d.height > 0:
            daily_dfs.append(df_d)
            logger.info(f"Day {d:02d}: {df_d.height:,} dòng ({time.time() - t0:.2f}s)")

    if not daily_dfs:
        logger.warning("Không có dữ liệu nào được trích xuất.")
        return pl.DataFrame()

    df_full = pl.concat(daily_dfs, how="vertical")
    matrix_file = out_path / "feature_matrix_raw.parquet"
    df_full.write_parquet(matrix_file, compression="snappy")
    logger.info(f"Đã lưu ma trận tổng hợp {df_full.height:,} dòng x {df_full.width} cột tại '{matrix_file}'.")

    return df_full
