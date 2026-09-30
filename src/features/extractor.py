"""
Account x Day Feature Extractor.
Trích xuất ma trận đặc trưng hành vi (Tài khoản × Ngày) từ log interim sự kiện 4624 & 4625.
Sử dụng Polars để tối ưu hóa tốc độ xử lý và bộ nhớ.
"""

from __future__ import annotations

import logging
import math
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
    "activity_peak_hour_sin", "activity_peak_hour_cos", "hour_entropy",
    "interactive_ratio", "rare_logon_type_count",
    "ntlm_ratio",
    "distinct_hosts", "distinct_sources_count", "dst_host_entropy",
    "missing_source_ratio", "remote_logon_ratio", "custom_proc_share",
]

#: 4 đặc trưng nhóm 9 của schema v3.0 đã QUA cổng kiểm định trên dữ liệu thật
#: (chỉ dùng trạng thái TRONG NGÀY ⇒ không cần Dense Panel, không rò rỉ tương lai).
INTRADAY_V3_FEATURES: List[str] = [
    "activity_peak_hour_sin",
    "activity_peak_hour_cos",
    "hour_entropy",
    "dst_host_entropy",
]

#: 2 đặc trưng nhóm 9 đã ĐO và BÁC BỎ (xem ``removed:`` trong configs/feature_schema.yaml):
#: ``max_failure_streak`` (ρ = 0,9973 với failure_ratio) và ``success_after_failure_ratio``
#: (ρ = 0,9180 với failure_ratio) — đúng cơ chế zero-inflation dùng chung "ngày có thất bại hay không"
#: đã từng loại ``failure_bad_password_share`` (ρ = 0,9220) và ``failure_count`` (ρ = 0,9969).
INTRADAY_V3_REJECTED: List[str] = [
    "max_failure_streak",
    "success_after_failure_ratio",
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


def intraday_behavior_features(
    df_events: pl.DataFrame,
    keys: Optional[List[str]] = None,
    assume_sorted: bool = False,
) -> pl.DataFrame:
    """
    Tính **6 đặc trưng nhóm 9 (schema v3.0)** chỉ từ trạng thái TRONG NGÀY.

    Vì sao tách riêng khỏi ``extract_features_single_day``: hàm này không đọc/ghi file và chỉ
    phụ thuộc các cột sự kiện thô (``Time``, ``EventID``, ``LogHost`` + khoá danh tính) nên
    kiểm thử được bằng dữ liệu tổng hợp — đúng yêu cầu "mỗi biến phải có cổng kiểm định"
    (xem ``docs/reports/bao_cao_bo_dac_trung_v3.md`` §7 bước 4).

    ===================================  ==================================================
    ``activity_peak_hour_sin``           ``sin(2π·h_peak/24)`` — h_peak = khung giờ nhiều sự
    ``activity_peak_hour_cos``           kiện nhất (hoà nhau → lấy giờ nhỏ hơn, tất định)
    ``hour_entropy``                     Pielou evenness ``H / ln(S)`` của phân bố sự kiện
                                         theo các khung giờ có mặt
    ``dst_host_entropy``                 evenness tương tự trên phân bố theo ``LogHost``
    ===================================  ==================================================

    Hai biến cùng nhóm đã ĐO trên dữ liệu thật rồi **bác bỏ** (không sinh tại đây):
    ``max_failure_streak`` (ρ = 0,9973 với ``failure_ratio``) và ``success_after_failure_ratio``
    (ρ = 0,9180) — vượt ngưỡng 0,85 của hợp đồng schema, xem ``INTRADAY_V3_REJECTED``.

    **Vì sao dùng Pielou evenness thay vì entropy thô:** entropy thô bị chặn trên bởi ``ln(n)``
    nên tài khoản ít sự kiện *luôn* có entropy nhỏ ⇒ cột sẽ trùng trục với ``log_total_logons``.
    Chia cho ``ln(S)`` (S = số khung giờ/host phân biệt) cho ra độ "trải đều" ∈ [0, 1] **độc lập
    với khối lượng**; quy ước ``= 0`` khi ``S <= 1`` (mọi sự kiện cùng 1 giờ / cùng 1 host).

    ``assume_sorted=True`` chỉ dùng khi khung vào ĐÃ sắp xếp theo ``keys + ["Time"]`` — điều kiện
    bắt buộc để ``shift(1).over(keys)`` và chuỗi thất bại đúng nghĩa thời gian.
    """
    keys = list(keys or ["DomainName", "UserName"])
    out_cols = keys + list(INTRADAY_V3_FEATURES)

    if df_events.height == 0:
        schema: Dict[str, Any] = {k: pl.String for k in keys}
        schema.update({c: pl.Float64 for c in INTRADAY_V3_FEATURES})
        return pl.DataFrame(schema=schema).select(out_cols)

    ev = df_events if assume_sorted else df_events.sort(keys + ["Time"])
    if "hour" not in ev.columns:
        ev = ev.with_columns(((pl.col("Time") % 86400) // 3600).cast(pl.Int32).alias("hour"))

    def _evenness(count_col: str, raw_alias: str, distinct_alias: str) -> List[pl.Expr]:
        """``-Σ p·ln(p)`` và ``S`` cho phân bố đếm theo một chiều phân loại."""
        p = pl.col(count_col) / pl.col(count_col).sum()
        return [
            (-(p * p.log())).sum().alias(raw_alias),
            pl.col(count_col).len().cast(pl.Float64).alias(distinct_alias),
        ]

    # (a) Khung giờ: giờ cao điểm (tất định khi hoà nhau) + evenness phân bố giờ
    hour_counts = ev.group_by(keys + ["hour"]).agg(pl.len().alias("_n"))
    peak_hour = (
        hour_counts.sort(["_n", "hour"], descending=[True, False])
        .group_by(keys, maintain_order=True)
        .agg(pl.col("hour").first().alias("_peak_hour"))
    )
    hour_entropy = hour_counts.group_by(keys).agg(
        _evenness("_n", "_hour_h", "_hour_distinct")
    )

    # (b) Máy đích: evenness phân bố theo LogHost (chỉ tính host hợp lệ)
    host_counts = (
        ev.filter(pl.col("LogHost").is_not_null() & (pl.col("LogHost").str.strip_chars() != ""))
        .with_columns(pl.col("LogHost").str.strip_chars().alias("_host"))
        .group_by(keys + ["_host"])
        .agg(pl.len().alias("_n"))
    )
    host_entropy = host_counts.group_by(keys).agg(
        _evenness("_n", "_host_h", "_host_distinct")
    )

    feats = hour_entropy.join(peak_hour, on=keys, how="left").join(
        host_entropy, on=keys, how="left"
    )

    return feats.with_columns([
        pl.when(pl.col("_hour_distinct") > 1)
        .then(pl.col("_hour_h") / pl.col("_hour_distinct").log())
        .otherwise(0.0)
        .alias("hour_entropy"),
        pl.when(pl.col("_host_distinct").fill_null(0.0) > 1)
        .then(pl.col("_host_h").fill_null(0.0) / pl.col("_host_distinct").log())
        .otherwise(0.0)
        .alias("dst_host_entropy"),
        (2 * math.pi * pl.col("_peak_hour") / 24).sin().alias("activity_peak_hour_sin"),
        (2 * math.pi * pl.col("_peak_hour") / 24).cos().alias("activity_peak_hour_cos"),
    ]).select(out_cols)


def extract_features_single_day(
    day: int,
    data_dir: Path,
    output_daily_dir: Optional[Path] = None,
    feature_cfg: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """
    Trích xuất vector đặc trưng THÔ cho từng cặp (DomainName, UserName, day).
    Ưu tiên đọc trực tiếp từ dữ liệu log sạch (data/cleaned/cleaned_day-XX.parquet).
    Nếu chưa có, tự động fallback sang đọc từ log interim.
    """
    fcfg = feature_cfg or DEFAULT_FEATURE_CFG
    
    # 1. Kiểm tra file cleaned trước (Chặng 1)
    cleaned_file = data_dir / f"cleaned_day-{day:02d}.parquet"
    if not cleaned_file.exists() and (data_dir / "cleaned").exists():
        cleaned_file = data_dir / "cleaned" / f"cleaned_day-{day:02d}.parquet"

    if cleaned_file.exists():
        # Đọc dữ liệu đã qua làm sạch
        events = pl.scan_parquet(cleaned_file)
    else:
        # Fallback đọc từ interim (nếu chưa chạy Stage Clean)
        interim_dir = data_dir if (data_dir / "event_4624").exists() else (data_dir / "interim")
        path_4624 = interim_dir / "event_4624" / f"event_4624_day-{day:02d}.parquet"
        path_4625 = interim_dir / "event_4625" / f"event_4625_day-{day:02d}.parquet"

        if not path_4624.exists() or not path_4625.exists():
            logger.warning(f"Không tìm thấy dữ liệu cho Day {day:02d} trong {data_dir}, bỏ qua.")
            return pl.DataFrame()

        cols_24 = [c for c in [
            "Time", "EventID", "UserName", "LogHost", "LogonType", "AuthenticationPackage",
            "Source", "DomainName", "ProcessName",
        ] if c in pl.scan_parquet(path_4624).collect_schema().names()]
        cols_25 = cols_24 + ["FailureReason"]

        lf_4624 = pl.scan_parquet(path_4624).select(cols_24).with_columns(pl.lit(None, dtype=pl.String).alias("FailureReason"))
        lf_4625 = pl.scan_parquet(path_4625).select([c for c in cols_25 if c in pl.scan_parquet(path_4625).collect_schema().names()])
        events = pl.concat([lf_4624, lf_4625], how="diagonal")

        # Làm sạch cơ bản nếu chạy từ interim
        events = events.filter(
            pl.col("UserName").is_not_null()
            & (pl.col("UserName").str.strip_chars() != "")
            & (pl.col("UserName").str.to_lowercase().is_in(["null", "none", "nan"]).not_())
        ).with_columns(
            pl.when(pl.col("DomainName").is_null() | (pl.col("DomainName").str.strip_chars() == ""))
            .then(pl.lit("Unknown"))
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

    # Kiểm tra cột ProcessName
    has_proc = "ProcessName" in df_events.columns
    proc_expr = (
        pl.col("ProcessName").str.to_lowercase().str.starts_with("proc").fill_null(False).sum().alias("_custom_proc_count")
        if has_proc
        else pl.lit(0).alias("_custom_proc_count")
    )

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
        ((pl.col("Source").is_null()) | (pl.col("Source") == "Unknown") | (pl.col("Source").str.strip_chars() == "")).sum().alias("_missing_source_count"),
        ((pl.col("Source").is_not_null()) & (pl.col("Source") != "Unknown") & (pl.col("Source") != pl.col("LogHost"))).sum().alias("_remote_logon_count"),
        pl.col("Source").filter((pl.col("Source").is_not_null()) & (pl.col("Source") != "Unknown")).n_unique().alias("distinct_sources_count"),
        proc_expr,
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

    # Nhóm 9 (schema v3.0): 6 đặc trưng TRONG NGÀY — không dùng lịch sử nên không rủi ro
    # rò rỉ tương lai (khác nhóm novelty/recency phải chờ Dense Panel).
    df_day = df_day.join(
        intraday_behavior_features(df_events, assume_sorted=True),
        on=["DomainName", "UserName"],
        how="left",
    )

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


def available_days(data_dir: Path) -> List[int]:
    """Liệt kê các ngày có dữ liệu (ưu tiên cleaned, fallback sang interim)."""
    days: set[int] = set()

    # 1. Tìm trong data/cleaned
    cleaned_dir = data_dir if any(data_dir.glob("cleaned_day-*.parquet")) else data_dir / "cleaned"
    if cleaned_dir.is_dir():
        for p in cleaned_dir.glob("cleaned_day-*.parquet"):
            try:
                days.add(int(p.stem.split("-")[-1]))
            except ValueError:
                pass
    if days:
        return sorted(list(days))

    # 2. Fallback sang data/interim
    interim_dir = data_dir if (data_dir / "event_4624").exists() else data_dir / "interim"
    p_4624 = interim_dir / "event_4624"
    if p_4624.is_dir():
        for p in p_4624.glob("event_4624_day-*.parquet"):
            try:
                d = int(p.stem.split("-")[-1])
                if (interim_dir / "event_4625" / f"event_4625_day-{d:02d}.parquet").exists():
                    days.add(d)
            except ValueError:
                continue
    return sorted(list(days))


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
    paths_cfg = (config or {}).get("paths", {})
    cleaned_dir = Path(paths_cfg.get("cleaned_data_dir", "data/cleaned"))
    interim_default = Path(paths_cfg.get("interim_data_dir", "data/interim"))
    
    # Ưu tiên kho log sạch (data/cleaned), fallback sang log interim
    if cleaned_dir.is_dir() and any(cleaned_dir.glob("cleaned_day-*.parquet")):
        data_source = cleaned_dir
        logger.info(f"Nguồn dữ liệu trích xuất: Kho log sạch '{data_source}'.")
    else:
        data_source = Path(interim_dir) if interim_dir else interim_default
        logger.info(f"Nguồn dữ liệu trích xuất: Kho log interim '{data_source}'.")

    days_avail = available_days(data_source)
    if not days_avail:
        logger.warning(f"Không có dữ liệu hợp lệ trong thư mục: {data_source}")
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
        df_d = extract_features_single_day(d, data_source, daily_dir, feature_cfg=fcfg)
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
