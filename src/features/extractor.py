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
from typing import Any, Dict, Iterable, List, Optional
import polars as pl
import yaml

from src.features.history import HISTORY_V3_FEATURES, add_history_features
from src.features.template_engine import DayOutput, compute_day, compute_history
from src.features.templates import candidate_by_name

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

#: Hợp đồng đầy đủ của **ma trận thô** = đặc trưng TRONG NGÀY + 8 đặc trưng LỊCH SỬ (Tier A).
#: 8 biến lịch sử không tính được trong ``extract_features_single_day`` (chỉ thấy 1 ngày) nên
#: được thêm ở ``build_account_day_matrix`` bằng ``src.features.history.add_history_features``.
#: Schema v4.0 — các biến sinh từ Combinatorial Template ĐÃ QUA PHỄU (vòng 1 lập luận +
#: 4 cổng số liệu E/B/A/C; xem docs/reports/bao_cao_bo_dac_trung_v4.md). Được tính bằng
#: ``src.features.template_engine`` — cùng một engine đã dùng để đo trong phễu.
TEMPLATE_V4_FEATURES: List[str] = [
    # brute-force / password spraying
    "delta_mean_share_fail_7d",
    "delta_mean_distinct_fail_host_7d",
    "novelty_fail_host_7d",
    "novelty_fail_failreason_7d",
    # bùng nổ máy trạm mới (chiếm đoạt tài khoản)
    "delta_mean_distinct_source_7d",
    "jaccard_source_7d",
    "peer_z_distinct_source",
    # hoạt động ngoài giờ
    "dist_shift_hour_7d",
    "jaccard_hour_7d",
    "peer_z_share_night",
    # di chuyển ngang
    "delta_mean_distinct_pair_7d",
    "novelty_share_pair_7d",
    # tài khoản ngủ đông thức dậy
    "delta_mean_count_7d",
    # đổi loại logon bất thường
    "dist_shift_logontype_7d",
    "jaccard_logontype_7d",
]

MATRIX_ORDERED_COLS: List[str] = (
    list(RAW_ORDERED_COLS) + list(HISTORY_V3_FEATURES) + list(TEMPLATE_V4_FEATURES)
)


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


def _presence_frames(df_events: pl.DataFrame, day: int, keys: List[str]) -> pl.DataFrame:
    """
    Dựng **vật liệu lịch sử** cho ngày ``day``: sự hiện diện của từng thực thể
    (``Source``/``LogHost``) dưới dạng dài ``(keys, day, kind, entity)``.

    Vì sao cần: ma trận thô chỉ có *số đếm* (``distinct_sources_count``, ``distinct_hosts``)
    chứ không giữ *danh tính* Source/LogHost, nên không thể suy ra "nguồn này đã từng thấy chưa".
    Khung này là đầu vào bắt buộc của ``src.features.history.add_history_features``.
    """
    day_expr = pl.lit(day).cast(pl.Int32).alias("day")
    valid_source = pl.col("Source").is_not_null() & (pl.col("Source").str.strip_chars() != "")
    valid_host = pl.col("LogHost").is_not_null() & (pl.col("LogHost").str.strip_chars() != "")

    return pl.concat([
        df_events.filter(valid_source)
        .select(keys + [pl.col("Source").str.strip_chars().alias("entity")])
        .unique()
        .with_columns(pl.lit("Source").alias("kind")),
        df_events.filter(valid_host)
        .select(keys + [pl.col("LogHost").str.strip_chars().alias("entity")])
        .unique()
        .with_columns(pl.lit("LogHost").alias("kind")),
    ]).with_columns(day_expr)


def load_day_events(day: int, data_dir: Path) -> Optional[pl.DataFrame]:
    """
    Đọc toàn bộ sự kiện 4624 + 4625 của một ngày (đã làm sạch cơ bản), CHƯA tổng hợp.

    Tách khỏi :func:`features_from_events` để có thể **can thiệp vào luồng sự kiện** trước
    khi tính đặc trưng — cụ thể là tiêm bất thường tổng hợp (``src/evaluation/injection.py``)
    rồi chạy đúng cùng một đường tính như dữ liệu thật. Trả về ``None`` nếu thiếu dữ liệu.
    """
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
            return None

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
    return events.collect()


def extract_features_single_day(
    day: int,
    data_dir: Path,
    output_daily_dir: Optional[Path] = None,
    feature_cfg: Optional[Dict[str, Any]] = None,
    return_presences: bool = False,
):
    """
    Trích xuất vector đặc trưng THÔ cho từng cặp (DomainName, UserName, day).
    Ưu tiên đọc trực tiếp từ dữ liệu log sạch (data/cleaned/cleaned_day-XX.parquet).
    Nếu chưa có, tự động fallback sang đọc từ log interim.

    ``return_presences=True`` trả thêm ``(entities, pairs)`` — vật liệu để tính 8 đặc trưng
    lịch sử ở :func:`src.features.history.add_history_features` (xem :func:`_presence_frames`).
    """
    events = load_day_events(day, data_dir)
    if events is None:
        return (pl.DataFrame(), None) if return_presences else pl.DataFrame()
    return features_from_events(
        events, day, output_daily_dir=output_daily_dir, feature_cfg=feature_cfg,
        return_presences=return_presences,
    )


def features_from_events(
    events: pl.DataFrame,
    day: int,
    output_daily_dir: Optional[Path] = None,
    feature_cfg: Optional[Dict[str, Any]] = None,
    return_presences: bool = False,
):
    """Tổng hợp sự kiện của MỘT ngày thành các đặc trưng trong ngày (``RAW_ORDERED_COLS``)."""
    fcfg = feature_cfg or DEFAULT_FEATURE_CFG
    events = events.lazy()

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

    if return_presences:
        return df_day, _presence_frames(df_events, day, ["DomainName", "UserName"])
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


def resolve_day_source(day: int, base_dir: Path, events_dir: Optional[Path], overlay: Iterable[int]) -> Path:
    """Thư mục đọc log của ngày ``day``: ``events_dir`` nếu ngày đó có trong ``overlay``, ngược lại ``base_dir``."""
    if events_dir is not None and int(day) in set(int(d) for d in overlay):
        for eid in (4624, 4625):
            p = Path(events_dir) / f"event_{eid}" / f"event_{eid}_day-{int(day):02d}.parquet"
            if not p.is_file():
                raise FileNotFoundError(f"Ngày {day} trong '{events_dir}' phải có đủ file 4624 và 4625 (thiếu '{p}').")
        return Path(events_dir)
    return Path(base_dir)


def build_account_day_matrix(
    start_day: int = 1,
    end_day: Optional[int] = None,
    interim_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    config: Optional[Dict[str, Any]] = None,
    events_dir: Optional[Path | str] = None,
) -> pl.DataFrame:
    """
    Điều phối trích xuất ma trận đặc trưng cho dải ngày [start_day, end_day].
    Ghi kết quả ra thư mục output_dir/ (data/features/raw/).

    ``events_dir`` (layout ``event_462x/event_462x_day-NN.parquet`` như ``data/interim``): các ngày có
    file trong thư mục này được đọc TỪ ĐÓ thay cho log gốc, các ngày còn lại đọc log gốc. Khi dùng
    ``events_dir`` nguồn gốc BẮT BUỘC là interim — tầng cleaned đã trừ 3600s (DST) cho ngày >= 42 nên
    cửa sổ ngày lệch với log đè (xem ``src/injection/layout.py``).
    """
    paths_cfg = (config or {}).get("paths", {})
    cleaned_dir = Path(paths_cfg.get("cleaned_data_dir", "data/cleaned"))
    interim_default = Path(paths_cfg.get("interim_data_dir", "data/interim"))

    has_cleaned = cleaned_dir.is_dir() and any(cleaned_dir.glob("cleaned_day-*.parquet"))
    if events_dir is not None:
        # Log đè luôn ở tầng interim -> nguồn gốc phải cùng tầng, bỏ qua kho cleaned.
        data_source = Path(interim_dir) if interim_dir else interim_default
        if has_cleaned:
            logger.info("Có events_dir -> bỏ qua kho log sạch '%s', đọc interim '%s'.", cleaned_dir, data_source)
        else:
            logger.info(f"Nguồn dữ liệu trích xuất: Kho log interim '{data_source}'.")
    elif has_cleaned:
        # Ưu tiên kho log sạch (data/cleaned), fallback sang log interim
        data_source = cleaned_dir
        logger.info(f"Nguồn dữ liệu trích xuất: Kho log sạch '{data_source}'.")
    else:
        data_source = Path(interim_dir) if interim_dir else interim_default
        logger.info(f"Nguồn dữ liệu trích xuất: Kho log interim '{data_source}'.")

    overlay: List[int] = []
    if events_dir is not None:
        from src.injection.layout import overlay_days

        overlay = overlay_days(events_dir)
        if not overlay:
            raise ValueError(f"events_dir '{events_dir}' không chứa file ngày nào.")
        logger.info("Log đè từ '%s' cho %d ngày: %s", events_dir, len(overlay), overlay)

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

    v4_specs = [candidate_by_name(n) for n in TEMPLATE_V4_FEATURES]

    logger.info(f"Bắt đầu trích xuất đặc trưng từ Day {start_day:02d} đến Day {end_day:02d}...")
    daily_dfs: List[pl.DataFrame] = []
    entity_dfs: List[pl.DataFrame] = []
    template_days: List[DayOutput] = []
    for d in range(start_day, end_day + 1):
        if d not in days_avail:
            continue
        t0 = time.time()
        events = load_day_events(
            d, resolve_day_source(d, data_source, Path(events_dir) if events_dir else None, overlay)
        )
        if events is None:
            continue
        df_d, entities_d = features_from_events(
            events, d, daily_dir, feature_cfg=fcfg, return_presences=True
        )
        if v4_specs:
            template_days.append(compute_day(events, d, v4_specs, fcfg))
        del events
        if df_d.height > 0:
            daily_dfs.append(df_d)
            entity_dfs.append(entities_d)
            logger.info(f"Day {d:02d}: {df_d.height:,} dòng ({time.time() - t0:.2f}s)")

    if not daily_dfs:
        logger.warning("Không có dữ liệu nào được trích xuất.")
        return pl.DataFrame()

    df_full = pl.concat(daily_dfs, how="vertical")

    # 8 đặc trưng LỊCH SỬ (Tier A, schema v3.0): cần thấy toàn bộ dải ngày nên tính ở đây,
    # sau khi đã ghép các ngày. Hàm bảo đảm chỉ dùng dữ liệu ≤ t−1 (xem src/features/history.py).
    t_hist = time.time()
    entities_full = pl.concat(entity_dfs, how="vertical") if entity_dfs else None
    df_full = add_history_features(df_full, entities=entities_full)
    if v4_specs:
        # Schema v4.0: tầng lịch sử/peer của engine template (chỉ dùng ngày ≤ t−1 / cắt ngang ngày t)
        intraday = pl.concat([t.intraday for t in template_days])
        profiles = {
            k: pl.concat([t.profiles[k] for t in template_days]) for k in template_days[0].profiles
        }
        v4 = compute_history(
            intraday, profiles, v4_specs,
            entity_type=df_full.select(["DomainName", "UserName", "day", "entity_type"]),
        ).select(["DomainName", "UserName", "day"] + list(TEMPLATE_V4_FEATURES))
        df_full = df_full.join(v4, on=["DomainName", "UserName", "day"], how="left")
    df_full = df_full.select(MATRIX_ORDERED_COLS)
    logger.info(
        "Đã tính %d đặc trưng lịch sử + %d đặc trưng template v4 trong %.1fs (ma trận %d cột).",
        len(HISTORY_V3_FEATURES), len(TEMPLATE_V4_FEATURES), time.time() - t_hist, df_full.width,
    )

    # Ghi lại các file ngày để bản `daily/` khớp hợp đồng đầy đủ (bản ghi trong
    # `extract_features_single_day` mới chỉ có đặc trưng TRONG NGÀY).
    for key, part in df_full.partition_by("day", as_dict=True).items():
        day_value = key[0] if isinstance(key, tuple) else key
        part.write_parquet(
            daily_dir / f"user_features_day-{int(day_value):02d}.parquet", compression="snappy"
        )

    matrix_file = out_path / "feature_matrix_raw.parquet"
    df_full.write_parquet(matrix_file, compression="snappy")
    logger.info(f"Đã lưu ma trận tổng hợp {df_full.height:,} dòng x {df_full.width} cột tại '{matrix_file}'.")

    return df_full
