"""
extract_account_day_matrix.py

⚠️ BẢN ĐỘC LẬP NÀY CHẬM HƠN PIPELINE CHÍNH 1 PHIÊN BẢN HỢP ĐỒNG.
   Nó vẫn sinh **20 cột của schema v2.0**; hợp đồng hiện hành là **v3.0 (24 cột)** với 4 đặc trưng
   nhóm 9 (`activity_peak_hour_sin/cos`, `hour_entropy`, `dst_host_entropy`).
   ⇒ Chạy file này rồi kiểm bằng `check_feature_matrix.py` sẽ **FAIL** (thiếu 4 cột).
   Đường chuẩn sinh ma trận theo hợp đồng mới: `python main.py --stage features`
   (mã nguồn: src/features/extractor.py). File này chỉ giữ để tái lập báo cáo Tuần 2.

Pipeline tổng hợp log thô (Event 4624 & 4625) thành Ma trận Đặc trưng (Tài khoản × Ngày).

QUAN TRỌNG — BỘ ĐẶC TRƯNG THÔ, KHÔNG CHUẨN HÓA:
  File này CHỈ xuất **biến bình thường (raw)**: KHÔNG áp `log1p`, KHÔNG scaling, KHÔNG clip.
  Việc chuẩn hóa log (dự kiến cho 4 cột lệch nặng nhất: total_logons, distinct_hosts,
  rare_logon_type_count, distinct_sources_count) sẽ làm ở bước sau, trên file raw này.

Bộ đặc trưng xuất ra (16 đặc trưng + 4 khoá/nhãn = 20 cột):
  Khoá/nhãn : DomainName, UserName, day, entity_type
  Volume    : total_logons
  Thất bại  : failure_ratio, failure_locked_out_share
  Thời gian : off_hours_ratio
  Nhịp      : interarrival_dt_mean, delta_t_cv, same_second_share, is_single_event
  Logon type: interactive_ratio, rare_logon_type_count
  Xác thực  : ntlm_ratio
  Fan-out   : distinct_hosts, distinct_sources_count
  Ngữ cảnh  : missing_source_ratio, remote_logon_ratio, custom_proc_share

CHẠY ĐỘC LẬP (không phụ thuộc thư mục làm việc, không cần file config):
  python extract_account_day_matrix.py                            # tự dò dữ liệu/config/nơi ghi
  python extract_account_day_matrix.py --start-day 1 --end-day 60
  python extract_account_day_matrix.py --interim-dir "D:\\data\\interim" --output-dir "D:\\out"

Đầu vào (tự dò theo thứ tự; có thể ghi đè bằng tham số):
  - data/interim/event_4624/event_4624_day-{day:02d}.parquet  (và event_4625/…)
  - configs/system_config.yaml (TUỲ CHỌN — không có thì dùng cấu hình mặc định built-in)

Đầu ra (mặc định <gốc dữ liệu>/data/features/raw/):
  - feature_matrix_raw.parquet                       (ma trận tổng hợp Tài khoản × Ngày)
  - daily/user_features_day-{day:02d}.parquet         (từng ngày)
  - pivot_account_day_logons.parquet                  (bảng 2D volume × ngày)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import polars as pl

try:  # PyYAML chỉ cần khi dùng --config
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent

# Cấu hình mặc định (dùng khi không có file config) — bám README: off-hours 18h-7h.
DEFAULT_FEATURE_CFG: dict = {
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


def load_config(config_path: Path | str | None) -> dict:
    """Nạp file YAML cấu hình; trả về {} nếu không có file hoặc thiếu PyYAML."""
    if config_path is None:
        return {}
    if yaml is None:
        print("[CẢNH BÁO] Chưa cài PyYAML -> dùng cấu hình mặc định built-in.")
        return {}
    path = Path(config_path)
    if not path.exists():
        print(f"[CẢNH BÁO] Không tìm thấy config '{path}' -> dùng mặc định built-in.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_feature_config(cfg: dict) -> dict:
    """Gộp config với giá trị mặc định (cửa sổ giờ, luật phân loại tài khoản, cờ lọc)."""
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


def entity_type_expr(fcfg: dict) -> pl.Expr:
    """Luật phân loại entity_type tường minh (không có nhánh 'còn lại -> Service')."""
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


def _interim_candidates(explicit: Path | str | None) -> list[Path]:
    """Các vị trí sẽ dò thư mục interim, theo thứ tự ưu tiên."""
    cands: list[Path] = []
    if explicit is not None:
        cands.append(Path(explicit))
    cands += [
        Path("data") / "interim",
        SCRIPT_DIR / "data" / "interim",
        SCRIPT_DIR / "interim",
        SCRIPT_DIR.parent / "data" / "interim",
        SCRIPT_DIR.parent.parent / "data" / "interim",
        SCRIPT_DIR.parent.parent.parent / "data" / "interim",
    ]
    seen: set[str] = set()
    out: list[Path] = []
    for c in cands:
        key = str(c).casefold()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def find_interim_dir(explicit: Path | str | None = None) -> Path | None:
    """Tự dò thư mục interim (phải có cả event_4624/ và event_4625/)."""
    for idx, c in enumerate(_interim_candidates(explicit)):
        if not ((c / "event_4624").is_dir() and (c / "event_4625").is_dir()):
            continue
        if explicit is not None or idx <= 3:
            return c.resolve()
        root = c.parent.parent  # <root>/data/interim -> <root>
        if (root / "scripts").is_dir() or (root / "configs").is_dir() or (root / ".git").is_dir():
            return c.resolve()
    return None


def find_config_file(explicit: Path | str | None = None) -> Path | None:
    """Tự dò file config; None nếu không có (khi đó dùng cấu hình mặc định)."""
    cands: list[Path] = []
    if explicit is not None:
        cands.append(Path(explicit))
    cands += [
        Path("configs") / "system_config.yaml",
        Path("system_config.yaml"),
        SCRIPT_DIR / "system_config.yaml",
        SCRIPT_DIR.parent / "configs" / "system_config.yaml",
        SCRIPT_DIR.parent.parent / "configs" / "system_config.yaml",
    ]
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


# Thứ tự cột xuất ra: 4 khoá/nhãn + 16 đặc trưng THÔ (không có cột log nào)
RAW_ORDERED_COLS: list[str] = [
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


def extract_features_single_day(
    day: int,
    interim_dir: Path,
    output_daily_dir: Path | None = None,
    feature_cfg: dict | None = None,
) -> pl.DataFrame:
    """
    Trích xuất vector đặc trưng THÔ cho từng cặp (DomainName, UserName, day).

    Bất biến: KHÔNG áp log1p/scaling; đầu ra đúng `RAW_ORDERED_COLS` (20 cột).
    """
    fcfg = feature_cfg or DEFAULT_FEATURE_CFG
    path_4624 = interim_dir / "event_4624" / f"event_4624_day-{day:02d}.parquet"
    path_4625 = interim_dir / "event_4625" / f"event_4625_day-{day:02d}.parquet"

    if not path_4624.exists() or not path_4625.exists():
        print(f"[CẢNH BÁO] Không tìm thấy dữ liệu cho Day {day:02d}, bỏ qua.")
        return pl.DataFrame()

    cols_4624 = [
        "Time", "EventID", "UserName", "LogHost", "LogonType",
        "AuthenticationPackage", "ProcessName", "Source", "DomainName",
    ]
    cols_4625 = cols_4624 + ["FailureReason"]

    # 1. Quét lười. `Status` không dùng (null 100%); `FailureReason` chỉ có ở 4625.
    lf_4624 = (
        pl.scan_parquet(path_4624)
        .select(cols_4624)
        .with_columns(pl.lit(None, dtype=pl.String).alias("FailureReason"))
    )
    lf_4625 = pl.scan_parquet(path_4625).select(cols_4625)
    events = pl.concat([lf_4624, lf_4625], how="diagonal")

    # 2. Làm sạch UserName
    events = events.filter(
        pl.col("UserName").is_not_null()
        & (pl.col("UserName").str.strip_chars() != "")
        & (pl.col("UserName").str.to_lowercase().is_in(["null", "none", "nan"]).not_())
    )

    # 3. Chuẩn hoá DomainName thành phần của KHOÁ DANH TÍNH (thiếu -> "unknown")
    events = events.with_columns(
        pl.when(pl.col("DomainName").is_null() | (pl.col("DomainName").str.strip_chars() == ""))
        .then(pl.lit("unknown"))
        .otherwise(pl.col("DomainName").str.strip_chars().str.to_lowercase())
        .alias("DomainName")
    )

    # 4. Giờ trong ngày [0 - 23]
    events = events.with_columns(((pl.col("Time") % 86400) // 3600).cast(pl.Int32).alias("hour"))

    # 5. Sắp xếp để tính delta_t giữa 2 sự kiện liên tiếp của cùng khoá
    df_events = events.sort(["DomainName", "UserName", "Time"]).collect()
    df_events = df_events.with_columns(
        pl.col("Time").diff().over(["DomainName", "UserName"]).alias("delta_t")
    )

    # 6. Tổng hợp theo khoá danh tính (tất cả là ĐẾM/TRUNG BÌNH thô, KHÔNG log)
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

        # --- Ratios (mẫu số = total_logons) ---
        (pl.col("off_hours_count") / pl.col("total_logons")).alias("off_hours_ratio"),
        (pl.col("failure_count") / pl.col("total_logons")).alias("failure_ratio"),
        (pl.col("_type2_count") / pl.col("total_logons")).alias("interactive_ratio"),
        (pl.col("_ntlm_count") / pl.col("total_logons")).alias("ntlm_ratio"),
        (pl.col("_missing_source_count") / pl.col("total_logons")).alias("missing_source_ratio"),
        (pl.col("_remote_logon_count") / pl.col("total_logons")).alias("remote_logon_ratio"),
        (pl.col("_custom_proc_count") / pl.col("total_logons")).alias("custom_proc_share"),

        # --- Chỉ xác định khi có >= 1 thất bại (NULL nếu không) ---
        pl.when(pl.col("failure_count") > 0)
        .then(pl.col("_locked_out_count") / pl.col("failure_count"))
        .otherwise(pl.lit(None, dtype=pl.Float64)).alias("failure_locked_out_share"),

        # --- Nhịp thời gian: NULL khi chỉ có 1 sự kiện (KHÔNG fill 86400) ---
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

    # 7. Tự kiểm tra: cửa sổ giờ phải phủ đủ 24h
    gap = df_day.select(
        ((pl.col("off_hours_count") + pl.col("work_hours_count")) != pl.col("total_logons")).sum()
    ).item()
    if gap > 0:
        print(f"    [CẢNH BÁO] Day {day:02d}: {gap:,} dòng có off_hours + work_hours != total_logons.")

    # 8. Chỉ giữ đúng hợp đồng cột (loại các cột đếm trung gian bắt đầu bằng "_")
    df_day = df_day.select(RAW_ORDERED_COLS)

    # 9. Bất biến: file RAW KHÔNG được chứa cột log
    log_cols = [c for c in df_day.columns if c.startswith("log_")]
    if log_cols:
        raise AssertionError(f"Bộ RAW không được chứa cột log: {log_cols}")

    other_rows = df_day.filter(pl.col("entity_type") == "Other")
    if other_rows.height > 0:
        print(f"    [!] Day {day:02d}: {other_rows.height:,} dòng có entity_type='Other' "
              f"({other_rows['UserName'].n_unique()} tài khoản) -> bổ sung vào config nếu cần.")

    # 10. Lọc tài khoản (mặc định TẮT)
    flt = fcfg.get("filter", {}) or {}
    if flt.get("drop_machine_accounts"):
        df_day = df_day.filter(pl.col("entity_type") != "Machine")
    if flt.get("drop_system_accounts"):
        df_day = df_day.filter(pl.col("entity_type") != "System")

    if output_daily_dir is not None:
        daily_file = output_daily_dir / f"user_features_day-{day:02d}.parquet"
        df_day.write_parquet(daily_file, compression="snappy")

    return df_day


def default_output_dir(interim_dir: Path) -> Path:
    """Mặc định ghi vào <gốc dữ liệu>/data/features/raw (tách khỏi bản đã chuẩn hóa)."""
    return interim_dir.parent / "features" / "raw"


def available_days(interim_dir: Path) -> list[int]:
    """Liệt kê các ngày có đủ cả 2 file 4624 & 4625 trong thư mục interim."""
    days: list[int] = []
    for p in (interim_dir / "event_4624").glob("event_4624_day-*.parquet"):
        try:
            d = int(p.stem.split("-")[-1])
        except ValueError:
            continue
        if (interim_dir / "event_4625" / f"event_4625_day-{d:02d}.parquet").exists():
            days.append(d)
    return sorted(days)


def skew_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Bảng đo độ lệch phân phối cho 16 đặc trưng THÔ (không log)."""
    feats = [c for c in df.columns if c not in ("DomainName", "UserName", "day", "entity_type")]
    rows = []
    for c in feats:
        s = df[c]
        nn = s.drop_nulls()
        if nn.len() == 0:
            continue
        nn_pd = nn.to_pandas()
        rows.append({
            "feature": c,
            "null_%": round(s.null_count() / df.height * 100, 2),
            "zero_%": round(float((nn == 0).sum()) / nn.len() * 100, 2),
            "p50": round(float(nn.median()), 3),
            "p99": round(float(nn.quantile(0.99)), 3),
            "max": round(float(nn.max()), 3),
            "skew": round(float(nn_pd.skew()), 2),
            "kurtosis": round(float(nn_pd.kurtosis()), 1),
        })
    return pl.DataFrame(rows).sort("skew", descending=True)


def build_account_day_matrix(
    start_day: int = 1,
    end_day: int | None = None,
    interim_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    config_path: Path | str | None = None,
) -> pl.DataFrame:
    """
    Sinh ma trận đặc trưng THÔ (Tài khoản × Ngày).

    - `interim_dir`/`output_dir`/`config_path` = None -> tự dò.
    - `end_day` = None -> tự lấy ngày lớn nhất có trong thư mục interim.
    """
    t_start = time.time()

    interim = find_interim_dir(interim_dir)
    if interim is None:
        print("[LỖI] Không tìm thấy dữ liệu interim (cần có cả event_4624/ và event_4625/).")
        print("      Đã dò các vị trí sau:")
        for c in _interim_candidates(interim_dir):
            print(f"        - {c}")
        return pl.DataFrame()

    days_avail = available_days(interim)
    if end_day is None:
        if not days_avail:
            print("[LỖI] Không có ngày nào đủ dữ liệu trong thư mục interim.")
            return pl.DataFrame()
        end_day = days_avail[-1]

    output_dir = Path(output_dir) if output_dir else default_output_dir(interim)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    cfg_file = find_config_file(config_path)
    raw_cfg = load_config(cfg_file)
    fcfg = resolve_feature_config(raw_cfg)

    print("=" * 88)
    print("TRÍCH XUẤT MA TRẬN ĐẶC TRƯNG THÔ (TÀI KHOẢN × NGÀY) — KHÔNG CHUẨN HÓA LOG")
    print("=" * 88)
    print(f"Phiên bản:        python {platform.python_version()} | polars {pl.__version__}")
    print(f"Thư mục làm việc: {Path.cwd()}")
    print(f"File script:      {Path(__file__).resolve()}")
    print(f"Phạm vi ngày:     Day {start_day:02d} -> Day {end_day:02d} "
          f"(có sẵn {len(days_avail)} ngày trong interim)")
    print(f"Dữ liệu interim:  {interim}")
    print(f"Thư mục đích:     {output_dir.resolve()}")
    print(f"Config:           {cfg_file} (ĐÃ NẠP)" if (cfg_file and raw_cfg)
          else "Config:           không có file -> dùng cấu hình mặc định built-in")
    print(f"Cửa sổ giờ:       off {fcfg['off_hours_start']}h-{fcfg['off_hours_end']}h | "
          f"work {fcfg['work_hours_start']}h-{fcfg['work_hours_end']}h (phủ đủ 24h)")
    print("Khoá danh tính:   (DomainName, UserName, day) | nhãn: entity_type")
    print("Chuẩn hóa:        KHÔNG (bộ RAW thuần — log1p để bước sau)")
    print("=" * 88)

    daily_dfs: list[pl.DataFrame] = []
    for d in range(start_day, end_day + 1):
        if d not in days_avail:
            print(f"[+] Day {d:02d}: bỏ qua (thiếu file).")
            continue
        t0 = time.time()
        print(f"\n[+] Đang xử lý Day {d:02d}...", end=" ", flush=True)
        df_d = extract_features_single_day(d, interim, daily_dir, feature_cfg=fcfg)
        if df_d.height > 0:
            daily_dfs.append(df_d)
            print(f"Hoàn thành! {df_d.height:,} dòng (tài khoản × ngày) ({time.time() - t0:.2f}s)")

    if not daily_dfs:
        print("[LỖI] Không có dữ liệu ngày nào được xử lý thành công.")
        return pl.DataFrame()

    matrix = pl.concat(daily_dfs, how="vertical")
    et_info = ", ".join(
        f"{r[0]}={r[1]:,}"
        for r in matrix.group_by("entity_type").agg(pl.len().alias("n")).sort("n", descending=True).iter_rows()
    )
    print(f"\n[+] Đã ghép {len(daily_dfs)} ngày -> ma trận {matrix.height:,} dòng × {matrix.width} cột")
    print(f"    ({matrix.width - 4} đặc trưng + 4 khoá/nhãn) | entity_type: {et_info}")

    matrix_file = output_dir / "feature_matrix_raw.parquet"
    matrix.write_parquet(matrix_file, compression="snappy")
    print(f"[✓] Đã lưu ma trận RAW tại: {matrix_file.resolve()}")
    print(f"    Dung lượng: {matrix_file.stat().st_size / (1024 * 1024):.2f} MB")

    # Pivot 2D (volume theo ngày) — tiện cho báo cáo
    try:
        pivot_src = matrix.with_columns(
            (pl.col("DomainName") + pl.lit("\\") + pl.col("UserName")).alias("user_key")
        )
        pivot_df = pivot_src.pivot(values="total_logons", index="user_key", on="day").fill_null(0)
        rename_map = {c: f"day_{int(c):02d}" for c in pivot_df.columns if c != "user_key"}
        pivot_file = output_dir / "pivot_account_day_logons.parquet"
        pivot_df.rename(rename_map).write_parquet(pivot_file, compression="snappy")
        print(f"[✓] Đã tạo bảng pivot volume × ngày: {pivot_file.resolve()}")
    except Exception as e:  # noqa: BLE001
        print(f"[!] Bỏ qua tạo pivot: {e}")

    # Bảng độ lệch phân phối trên bản RAW (phục vụ báo cáo)
    print("\n" + "=" * 88)
    print("ĐỘ LỆCH PHÂN PHỐI CỦA 16 ĐẶC TRƯNG THÔ (|skew| giảm dần)")
    print("=" * 88)
    sk = skew_summary(matrix)
    print(sk.to_pandas().to_string(index=False))
    skew_csv = output_dir / "distribution_stats_raw.csv"
    sk.to_pandas().to_csv(skew_csv, index=False, float_format="%.4f")
    print(f"[✓] Đã lưu bảng phân phối RAW: {skew_csv.resolve()}")
    print("-" * 88)
    print(f"4 cột lệch nặng nhất (dự kiến chuẩn hóa log ở bước sau): "
          f"{', '.join(sk.head(4)['feature'].to_list())}")
    print("=" * 88)

    print(f"\n[HOÀN TẤT] Tổng thời gian: {time.time() - t_start:.2f}s")
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trích xuất ma trận đặc trưng THÔ (Tài khoản × Ngày) — không chuẩn hóa log"
    )
    parser.add_argument("--start-day", type=int, default=1, help="Ngày bắt đầu (mặc định: 1)")
    parser.add_argument("--end-day", type=int, default=None,
                        help="Ngày kết thúc (bỏ trống = tự lấy ngày lớn nhất có trong interim)")
    parser.add_argument("--interim-dir", type=str, default=None,
                        help="Thư mục interim (bỏ trống = tự dò)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Thư mục lưu kết quả (bỏ trống = <gốc dữ liệu>/data/features/raw)")
    parser.add_argument("--config", type=str, default=None,
                        help="File YAML cấu hình (bỏ trống = tự dò; không có thì dùng mặc định)")

    args = parser.parse_args()

    build_account_day_matrix(
        start_day=args.start_day,
        end_day=args.end_day,
        interim_dir=Path(args.interim_dir) if args.interim_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        config_path=Path(args.config) if args.config else None,
    )
