"""
extract_account_day_matrix.py

Pipeline tổng hợp log thô (Event 4624 & 4625) thành Ma trận Đặc trưng (Tài khoản × Ngày).

BẢN 2.0 — sửa các bất thường của bộ đặc trưng theo
`docs/feature_engineering/feature_correlation_review.md` §11 (mục #1–#10):
  #1  Khoá danh tính = (DomainName, UserName) thay vì chỉ UserName.
  #2  entity_type phân loại tường minh từ config: Machine / User / Service / System / Admin / Other
      (bỏ luật "còn lại -> Service" từng gán Administrator vào nhóm Service).
  #3  Danh sách tài khoản hệ thống được ĐỌC từ config (trước đây config không được dùng).
  #4  Bỏ 2 đặc trưng chết (`wrong_password_count`, `unknown_user_count` — do `Status` null 100%);
      thay bằng tỷ trọng lý do thất bại lấy từ `FailureReason` (4625 có 100% giá trị).
  #5  Cửa sổ giờ lấy từ config (off 18h-7h / work 8h-17h) => phủ đủ 24 giờ, không còn vùng trống.
  #6  Script đọc `--config configs/system_config.yaml`.
  #7  Bỏ `burst_logon_count` (70,57% sự kiện có delta_t = 0 => thực chất là đếm sự kiện trùng giây);
      thay bằng `same_second_share` (tỷ trọng, không phải count nên không trùng với volume).
  #8  Bỏ `fill_null(86400.0)` (tạo đỉnh giả 86.400 ở 0,841% số dòng); giữ NULL cho
      `interarrival_dt_mean`, thêm `is_single_event` và `delta_t_cv`.
  #9  Nhóm logon-type: đã THỬ và BÁC BỎ phương án ratio bù trừ "other = 1 - network
      - interactive" (gây phụ thuộc tuyến tính tuyệt đối, VIF ~ 5,2e7); type hiếm được
      biểu diễn bằng log1p(count) để phản ánh sự kiện không thuộc 2 nhóm chính.
  #10 Rút gọn bộ đặc trưng theo §7 (bỏ 12 đặc trưng trùng lặp / near-constant / heuristic).

CHẠY ĐỘC LẬP (không phụ thuộc thư mục làm việc, không cần file config):
  python extract_account_day_matrix.py                          # tự dò dữ liệu + config + nơi ghi
  python extract_account_day_matrix.py --start-day 1 --end-day 30
  python extract_account_day_matrix.py --interim-dir "D:\\data\\interim" --output-dir "D:\\out"

Đầu vào (tự dò theo thứ tự; có thể ghi đè bằng tham số):
  - data/interim/event_4624/event_4624_day-{day:02d}.parquet  (và event_4625/…)
    Vị trí dò: ./data/interim -> cạnh script -> <repo>/data/interim -> các cấp trên của script
  - configs/system_config.yaml (TUỲ CHỌN — không có file thì dùng cấu hình mặc định built-in)

Đầu ra:
  - <gốc dữ liệu>/data/features/daily/user_features_day-{day:02d}.parquet (từng ngày)
  - <gốc dữ liệu>/data/features/account_day_matrix.parquet (Ma trận tổng hợp Tài khoản x Ngày)
  - <gốc dữ liệu>/data/features/pivot_account_day_logons.parquet (bảng 2D volume × ngày)

Xem thêm: `configs/feature_schema.yaml` (hợp đồng schema đặc trưng), `HUONG_DAN_CHAY.md`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

try:  # PyYAML chỉ cần khi dùng --config; nhờ vậy script vẫn chạy được chỉ với polars
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


DEFAULT_CONFIG_PATH = Path("configs/system_config.yaml")
SCRIPT_DIR = Path(__file__).resolve().parent


def _interim_candidates(explicit: Path | str | None) -> list[Path]:
    """Các vị trí sẽ dò thư mục interim, theo thứ tự ưu tiên."""
    cands: list[Path] = []
    if explicit is not None:
        cands.append(Path(explicit))
    cands += [
        Path("data") / "interim",                      # 1. thư mục làm việc hiện tại
        SCRIPT_DIR / "data" / "interim",               # 2. cạnh script
        SCRIPT_DIR / "interim",                        # 3. dữ liệu đặt ngay cạnh script
        SCRIPT_DIR.parent / "data" / "interim",        # 4. scripts/data/interim
        SCRIPT_DIR.parent.parent / "data" / "interim",  # 5. <repo>/data/interim (scripts/feature_engineering/...)
        SCRIPT_DIR.parent.parent.parent / "data" / "interim",
    ]
    # bỏ trùng nhưng giữ thứ tự
    seen: set[str] = set()
    out: list[Path] = []
    for c in cands:
        key = str(c).casefold()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def find_interim_dir(explicit: Path | str | None = None) -> Path | None:
    """
    Tự dò thư mục interim (phải có cả event_4624/ và event_4625/). None nếu không thấy.

    Với các ứng viên ở tầng trên (đi ngược từ vị trí script), chỉ nhận nếu thư mục gốc
    thật sự trông giống gốc dự án (có `scripts/` hoặc `configs/` hoặc `.git/`) — tránh
    việc vô tình nhặt dữ liệu của thư mục khác trên cùng ổ đĩa.
    """
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
    """Tự dò file config: tham số -> CWD -> cạnh script -> configs/ của repo. None nếu không có."""
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


def default_output_dir(interim_dir: Path) -> Path:
    """Mặc định ghi cạnh dữ liệu: <gốc dữ liệu>/data/features khi interim = <gốc>/data/interim."""
    return interim_dir.parent / "features"

# Giá trị mặc định (dùng khi config thiếu khoá) — bám đúng README: off-hours 18h-7h.
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
    """Nạp `configs/system_config.yaml`; trả về dict (rỗng nếu không đọc được / thiếu PyYAML)."""
    if config_path is None:
        return {}
    if yaml is None:
        print("[CẢNH BÁO] Chưa cài PyYAML -> bỏ qua config, dùng giá trị mặc định built-in "
              "(kết quả không đổi vì mặc định trùng với configs/system_config.yaml).")
        return {}
    path = Path(config_path)
    if not path.exists():
        print(f"[CẢNH BÁO] Không tìm thấy config '{path}' -> dùng giá trị mặc định built-in.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_feature_config(cfg: dict) -> dict:
    """Gộp config người dùng với giá trị mặc định (dùng cho cả cửa sổ giờ và phân loại tài khoản)."""
    feats = dict(cfg.get("features", {}) or {})
    merged = {k: v for k, v in DEFAULT_FEATURE_CFG.items() if k != "entity_type"}
    for key in ("off_hours_start", "off_hours_end", "work_hours_start", "work_hours_end"):
        if feats.get(key) is not None:
            merged[key] = int(feats[key])

    et = dict(DEFAULT_FEATURE_CFG["entity_type"])
    et.update(feats.get("entity_type", {}) or {})
    merged["entity_type"] = et

    # Tương thích khoá cũ `preprocessing.ignore_system_accounts`
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
    """Luật phân loại entity_type tường minh (đã bỏ nhánh 'còn lại -> Service')."""
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
    output_daily_dir: Path | None = None,
    feature_cfg: dict | None = None,
) -> pl.DataFrame:
    """
    Trích xuất vector đặc trưng cho từng cặp (DomainName, UserName, day).

    Bộ đặc trưng bản 2.0: xem docstring đầu file và `configs/feature_schema.yaml`.
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
    cols_4625 = [
        "Time", "EventID", "UserName", "LogHost", "LogonType",
        "AuthenticationPackage", "ProcessName", "Source", "DomainName", "FailureReason",
    ]
    # Ghi chú: `Status` KHÔNG còn được dùng (null 100% ở cả 4624 và 4625) -> 2 đặc trưng
    # wrong_password_count / unknown_user_count ở bản 1.0 luôn bằng 0 và đã bị xoá (mục #4).
    # `FailureReason` (4625 có 100% giá trị) được đưa vào để thay thế.

    # 1. Quét lười (Lazy Evaluation)
    lf_4624 = (
        pl.scan_parquet(path_4624)
        .select(cols_4624)
        .with_columns(pl.lit(None, dtype=pl.String).alias("FailureReason"))
    )
    lf_4625 = pl.scan_parquet(path_4625).select(cols_4625)

    events = pl.concat([lf_4624, lf_4625], how="diagonal")

    # 2. Làm sạch trường UserName
    events = events.filter(
        pl.col("UserName").is_not_null()
        & (pl.col("UserName").str.strip_chars() != "")
        & (pl.col("UserName").str.to_lowercase().is_in(["null", "none", "nan"]).not_())
    )

    # 3. Chuẩn hoá DomainName thành một phần của KHOÁ DANH TÍNH (mục #1).
    #    DomainName thiếu (4 dòng 4624 + 33 dòng 4625 ngày 16) -> gán "unknown" theo
    #    khuyến nghị của reports/day16_missingness.md §3, KHÔNG suy diễn từ bản ghi khác.
    events = events.with_columns(
        pl.when(pl.col("DomainName").is_null() | (pl.col("DomainName").str.strip_chars() == ""))
        .then(pl.lit("unknown"))
        .otherwise(pl.col("DomainName").str.strip_chars().str.to_lowercase())
        .alias("DomainName")
    )

    # 4. Tính giờ trong ngày [0 - 23]
    events = events.with_columns(
        ((pl.col("Time") % 86400) // 3600).cast(pl.Int32).alias("hour")
    )

    # 5. Sắp xếp để tính khoảng cách liên tiếp delta_t (theo KHOÁ danh tính mới)
    df_events = events.sort(["DomainName", "UserName", "Time"]).collect()

    df_events = df_events.with_columns(
        pl.col("Time").diff().over(["DomainName", "UserName"]).alias("delta_t")
    )

    # 6. Tổng hợp đặc trưng theo KHOÁ DANH TÍNH (DomainName, UserName) cho ngày này
    off_start, off_end = fcfg["off_hours_start"], fcfg["off_hours_end"]
    wk_start, wk_end = fcfg["work_hours_start"], fcfg["work_hours_end"]
    in_off = (pl.col("hour") >= off_start) | (pl.col("hour") <= off_end)
    in_work = (pl.col("hour") >= wk_start) & (pl.col("hour") <= wk_end)

    df_day = df_events.group_by(["DomainName", "UserName"]).agg([
        # Nhóm 1: Volume & Thất bại
        pl.len().alias("total_logons"),
        (pl.col("EventID") == 4625).sum().alias("failure_count"),

        # Nhóm 2: Thời gian & Nhịp sinh học (cửa sổ giờ đọc từ config, mục #5)
        in_off.sum().alias("off_hours_count"),
        in_work.sum().alias("work_hours_count"),
        pl.col("delta_t").mean().alias("interarrival_dt_mean"),
        pl.col("delta_t").std().alias("interarrival_dt_std"),
        (pl.col("delta_t") == 0).fill_null(False).sum().alias("same_second_count"),

        # Nhóm 3: Phương thức truy cập (LogonType).
        #   LƯU Ý QUAN TRỌNG: KHÔNG tạo ratio bù trừ "other = 1 - network - interactive".
        #   Đã thử và ĐO ĐƯỢC: 3 ratio cộng lại bằng 1 -> phụ thuộc tuyến tính tuyệt đối
        #   (VIF ~ 5,2e7, rho(network, other) = 0.9415). Thay vào đó, nhóm type hiếm
        #   (0/4/5/7/8/9/10/11) được biểu diễn bằng log1p(số sự kiện).
        (pl.col("LogonType") == 3).sum().alias("type3_network_count"),  # dùng cho self-check
        (pl.col("LogonType") == 2).sum().alias("type2_interactive_count"),
        (~pl.col("LogonType").is_in([2, 3])).sum().alias("rare_logon_type_count"),

        # Nhóm 4: Xác thực & LÝ DO THẤT BẠI (thay 2 đặc trưng chết, mục #4).
        #   `FailureReason` chỉ có ở 4625 (100% giá trị); 4624 đã được gán null ở bước 1.
        #   Chỉ giữ tỷ trọng LOCKOUT: tỷ trọng "bad password" có rho = 0.9220 với
        #   failure_ratio (vì mọi lý do thất bại đều là bằng chứng 'có thất bại') nên bị loại.
        pl.col("AuthenticationPackage").str.to_lowercase().str.contains("ntlm").fill_null(False).sum().alias("ntlm_count"),
        (pl.col("FailureReason").str.to_lowercase().str.contains("account locked out").fill_null(False)).sum().alias("failure_locked_out_count"),

        # Nhóm 5: Bậc mạng Fan-out
        pl.col("LogHost").n_unique().alias("distinct_hosts_count"),

        # Nhóm 6: Ngữ cảnh Nguồn (Source)
        ((pl.col("Source").is_null()) | (pl.col("Source").str.strip_chars() == "")).sum().alias("missing_source_count"),
        ((pl.col("Source").is_not_null()) & (pl.col("Source") != pl.col("LogHost"))).sum().alias("remote_logon_count"),
        pl.col("Source").drop_nulls().n_unique().alias("distinct_sources_count"),

        # Nhóm 7: Ngữ cảnh Tiến trình — chỉ giữ dạng TỶ TRỌNG (mục #10 bỏ 3 cờ nhị phân)
        pl.col("ProcessName").str.to_lowercase().str.starts_with("proc").fill_null(False).sum().alias("custom_proc_count"),
    ]).with_columns([
        # Khoá chiều Ngày + Phân loại thực thể (luật tường minh, mục #2)
        pl.lit(day).cast(pl.Int32).alias("day"),
        entity_type_expr(fcfg),

        # Ratios theo mẫu số = total_logons
        (pl.col("off_hours_count") / pl.col("total_logons")).alias("off_hours_ratio"),
        (pl.col("failure_count") / pl.col("total_logons")).alias("failure_ratio"),
        (pl.col("type3_network_count") / pl.col("total_logons")).alias("network_ratio"),  # chỉ dùng cho self-check, KHÔNG xuất ra ma trận
        (pl.col("type2_interactive_count") / pl.col("total_logons")).alias("interactive_ratio"),
        # Type hiếm: log1p(count) thay vì ratio bù trừ (xem ghi chú ở khối agg)
        pl.col("rare_logon_type_count").log1p().alias("rare_logon_type_count_log"),
        (pl.col("ntlm_count") / pl.col("total_logons")).alias("ntlm_ratio"),
        (pl.col("missing_source_count") / pl.col("total_logons")).alias("missing_source_ratio"),
        (pl.col("remote_logon_count") / pl.col("total_logons")).alias("remote_logon_ratio"),
        (pl.col("custom_proc_count") / pl.col("total_logons")).alias("custom_proc_share"),

        # Tỷ trọng LÝ DO THẤT BẠI (mục #4): chỉ xác định khi có >= 1 thất bại,
        # cố ý để NULL khi không có thất bại để KHÔNG tái tạo lại cặp trùng lặp
        # "failure_count ~ failure_ratio" của bản 1.0.
        pl.when(pl.col("failure_count") > 0)
        .then(pl.col("failure_locked_out_count") / pl.col("failure_count"))
        .otherwise(pl.lit(None, dtype=pl.Float64)).alias("failure_locked_out_share"),

        # Nhịp thời gian (mục #8): BỎ fill_null(86400.0); để NULL khi chỉ có 1 sự kiện
        pl.when(pl.col("total_logons") > 1)
        .then(pl.col("interarrival_dt_mean"))
        .otherwise(pl.lit(None, dtype=pl.Float64)).alias("interarrival_dt_mean"),
        pl.when(
            (pl.col("total_logons") > 1)
            & pl.col("interarrival_dt_mean").is_not_null()
            & (pl.col("interarrival_dt_mean") > 0)
        )
        .then(pl.col("interarrival_dt_std") / pl.col("interarrival_dt_mean"))
        .otherwise(pl.lit(None, dtype=pl.Float64)).alias("delta_t_cv"),

        # Thay burst_logon_count (mục #7): tỷ trọng sự kiện trùng giây
        (pl.col("same_second_count") / pl.col("total_logons")).alias("same_second_share"),
        (pl.col("total_logons") == 1).cast(pl.UInt8).alias("is_single_event"),

        # Log transform cho các count lệch nặng (giữ 1 phiên bản, mục #10)
        pl.col("total_logons").log1p().alias("log_total_logons"),
        pl.col("distinct_hosts_count").log1p().alias("log_distinct_hosts"),
    ])

    # Tự kiểm tra ràng buộc (mục #5 và #9) — chạy trên khung dữ liệu còn đủ cột count:
    #   (a) off_hours + work_hours phải phủ đủ 24 giờ
    #   (b) network_ratio + interactive_ratio <= 1 (phần còn lại là type hiếm)
    chk = df_day.select(
        ((pl.col("off_hours_count") + pl.col("work_hours_count")) != pl.col("total_logons")).sum().alias("gap_24h"),
        (
            (pl.col("network_ratio") + pl.col("interactive_ratio") - 1.0) > 1e-9
        ).sum().alias("ratio_bad"),
    ).row(0)
    if chk[0] > 0:
        print(f"    [CẢNH BÁO] Day {day:02d}: {chk[0]:,} dòng có off_hours + work_hours != total_logons.")
    if chk[1] > 0:
        print(f"    [CẢNH BÁO] Day {day:02d}: {chk[1]:,} dòng có network_ratio + interactive_ratio > 1.")

    # Sắp xếp lại thứ tự cột: [khoá danh tính + nhãn] rồi tới đặc trưng (bản 2.0)
    ordered_cols = [
        # --- Khoá danh tính & nhãn (mục #1, #2) ---
        "UserName",
        "DomainName",
        "day",
        "entity_type",
        # --- Volume: giữ count thô để hiển thị/pivot, KHÔNG dùng làm đặc trưng model (mục #10) ---
        "total_logons",
        "log_total_logons",
        # --- Thất bại: 1 đặc trưng cường độ + 1 tỷ trọng lý do (mục #4) ---
        "failure_ratio",
        "failure_locked_out_share",
        # --- Thời gian / nhịp (mục #5, #7, #8) ---
        "off_hours_ratio",
        "interarrival_dt_mean",
        "delta_t_cv",
        "same_second_share",
        "is_single_event",
        # --- Phương thức truy cập ---
        #   `network_ratio` đã bị loại vì (i) gần hằng số (median = 1.000; chỉ 2.46% dòng = 0)
        #   và (ii) rho = 0.9031 với rare_logon_type_count_log (loại trừ lẫn nhau:
        #   ngày toàn type 3 thì rare = 0). Thông tin "loại tài khoản mạng" đã có ở entity_type.
        "interactive_ratio",
        "rare_logon_type_count_log",
        # --- Xác thực ---
        "ntlm_ratio",
        # --- Đa dạng thực thể ---
        "log_distinct_hosts",
        "distinct_sources_count",
        # --- Ngữ cảnh nguồn & tiến trình ---
        "missing_source_ratio",
        "remote_logon_ratio",
        "custom_proc_share",
    ]
    df_day = df_day.select(ordered_cols)

    # Cảnh báo nếu còn tài khoản rơi vào nhánh an toàn 'Other' (mục #2)
    other_rows = df_day.filter(pl.col("entity_type") == "Other")
    if other_rows.height > 0:
        names = other_rows["UserName"].unique().to_list()[:5]
        print(
            f"    [!] Day {day:02d}: {other_rows.height:,} dòng có entity_type='Other' "
            f"({other_rows['UserName'].n_unique()} tài khoản), ví dụ: {names} "
            "-> bổ sung vào features.entity_type.* trong config nếu cần."
        )

    # Bộ lọc tài khoản (mục #3) — mặc định TẮT để không đổi quần thể phân tích.
    # Bật trong config: features.filter.drop_machine_accounts / drop_system_accounts
    flt = fcfg.get("filter", {}) or {}
    if flt.get("drop_machine_accounts"):
        df_day = df_day.filter(pl.col("entity_type") != "Machine")
    if flt.get("drop_system_accounts"):
        df_day = df_day.filter(pl.col("entity_type") != "System")

    # Lưu file lẻ từng ngày nếu có chỉ định
    if output_daily_dir is not None:
        daily_file = output_daily_dir / f"user_features_day-{day:02d}.parquet"
        df_day.write_parquet(daily_file, compression="snappy")

    return df_day


def build_account_day_matrix(
    start_day: int = 1,
    end_day: int = 3,
    interim_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    config_path: Path | str | None = None,
) -> pl.DataFrame:
    """
    Hàm thực thi pipeline trích xuất ma trận (Tài khoản x Ngày) từ start_day đến end_day.

    Chạy độc lập — không cần truyền tham số:
      - `interim_dir`: None -> tự dò `data/interim` (xem `_interim_candidates`)
      - `output_dir`:  None -> ghi cạnh dữ liệu: `<gốc dữ liệu>/data/features`
      - `config_path`: None -> tự dò `configs/system_config.yaml`; không có file thì dùng
        cấu hình mặc định built-in (off 18h–7h, work 8h–17h, luật phân loại tài khoản).
    """
    t_start = time.time()

    # --- TỰ DÒ ĐƯỜNG DẪN (để "chỉ cần chạy file là chạy được") ---
    interim = find_interim_dir(interim_dir)
    if interim is None:
        print("[LỖI] Không tìm thấy dữ liệu interim (cần có cả event_4624/ và event_4625/).")
        print("      Đã dò các vị trí sau:")
        for c in _interim_candidates(interim_dir):
            print(f"        - {c}")
        print("      Cách xử lý: đặt dữ liệu vào <gốc repo>/data/interim, hoặc đặt thư mục")
        print("      'data/interim' cạnh file script, hoặc truyền --interim-dir <đường dẫn tuyệt đối>.")
        return pl.DataFrame()

    output_dir = Path(output_dir) if output_dir else default_output_dir(interim)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    cfg_file = find_config_file(config_path if config_path is not None else DEFAULT_CONFIG_PATH)
    raw_cfg = load_config(cfg_file)
    fcfg = resolve_feature_config(raw_cfg)

    print("=" * 80)
    print("PIPELINE TỔNG HỢP LOG THÔ THÀNH MA TRẬN (TÀI KHOẢN × NGÀY) — BẢN 2.0")
    print(f"Phiên bản:       python {platform.python_version()} | polars {pl.__version__}")
    print(f"Thư mục làm việc: {Path.cwd()}")
    print(f"File script:     {Path(__file__).resolve()}")
    print(f"Phạm vi:         Day {start_day:02d} đến Day {end_day:02d}")
    print(f"Dữ liệu interim: {interim}   (tự dò)" if interim_dir is None else f"Dữ liệu interim: {interim}")
    print(f"Thư mục đích:    {output_dir.resolve()}")
    print(f"Config:          {cfg_file} (ĐÃ NẠP)" if (cfg_file and raw_cfg)
          else "Config:          không có file -> dùng cấu hình mặc định built-in")
    print(
        f"Cửa sổ giờ:     off {fcfg['off_hours_start']}h–{fcfg['off_hours_end']}h | "
        f"work {fcfg['work_hours_start']}h–{fcfg['work_hours_end']}h "
        "(phủ đủ 24h — mục #5)"
    )
    print(
        "Phân loại tài khoản: "
        f"admin={fcfg['entity_type']['admin_accounts']} | "
        f"service={fcfg['entity_type']['service_accounts']}"
    )
    print(
        f"Bộ lọc tài khoản:  drop_machine={fcfg['filter']['drop_machine_accounts']} | "
        f"drop_system={fcfg['filter']['drop_system_accounts']}"
    )
    print("=" * 80)

    daily_dfs: list[pl.DataFrame] = []

    for d in range(start_day, end_day + 1):
        t0 = time.time()
        print(f"\n[+] Đang xử lý Day {d:02d}...", end=" ", flush=True)
        df_d = extract_features_single_day(d, interim, daily_dir, feature_cfg=fcfg)
        if df_d.height > 0:
            daily_dfs.append(df_d)
            print(f"Hoàn thành! {df_d.height:,} dòng (tài khoản × ngày) ({time.time() - t0:.2f}s)")
        else:
            print("Không có dữ liệu!")

    if not daily_dfs:
        print("[LỖI] Không có dữ liệu ngày nào được xử lý thành công.")
        return pl.DataFrame()

    # Ghép toàn bộ các ngày thành Ma trận (Tài khoản x Ngày)
    t_concat = time.time()
    matrix = pl.concat(daily_dfs, how="vertical")
    print(f"\n[+] Đã ghép nối toàn bộ {len(daily_dfs)} ngày thành Ma trận (Tài khoản × Ngày) trong {time.time() - t_concat:.2f}s")
    print(f"    - Tổng số bản ghi (Tài khoản × Ngày): {matrix.height:,}")
    print(f"    - Tổng số cột:                        {matrix.width} "
          f"(4 khoá/nhãn + {matrix.width - 5} đặc trưng + 1 cột hiển thị total_logons)")

    # 1. Lưu Ma trận chuẩn (Panel Data / Long format)
    matrix_file = output_dir / "account_day_matrix.parquet"
    matrix.write_parquet(matrix_file, compression="snappy")
    print(f"[✓] Đã lưu Ma trận (Tài khoản × Ngày) vào: {matrix_file.resolve()}")
    print(f"    Dung lượng file: {matrix_file.stat().st_size / (1024 * 1024):.2f} MB")

    # 2. Tạo bổ sung Ma trận 2D trực diện (Pivot Table: Rows=Domain\Account, Cols=Day) cho volume
    try:
        pivot_src = matrix.with_columns(
            (pl.col("DomainName") + pl.lit("\\") + pl.col("UserName")).alias("user_key")
        )
        pivot_df = pivot_src.pivot(
            values="total_logons",
            index="user_key",
            on="day"
        ).fill_null(0)
        # Đổi tên cột ngày cho rõ ràng: day_1, day_2...
        rename_map = {col: f"day_{int(col):02d}" for col in pivot_df.columns if col != "user_key"}
        pivot_df = pivot_df.rename(rename_map)

        pivot_file = output_dir / "pivot_account_day_logons.parquet"
        pivot_df.write_parquet(pivot_file, compression="snappy")
        print(f"[✓] Đã tạo Ma trận 2D Pivot (Accounts × Days Volume) tại: {pivot_file.resolve()}")
    except Exception as e:
        print(f"[!] Bỏ qua tạo Pivot: {e}")

    print(f"\n[HOÀN TẤT] Toàn bộ pipeline chạy xong trong: {time.time() - t_start:.2f}s")
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline tổng hợp log thô thành Ma trận (Tài khoản × Ngày)")
    parser.add_argument("--start-day", type=int, default=1, help="Ngày bắt đầu (mặc định: 1)")
    parser.add_argument("--end-day", type=int, default=3, help="Ngày kết thúc (mặc định: 3)")
    parser.add_argument(
        "--interim-dir",
        type=str,
        default=None,
        help="Thư mục interim. Bỏ trống = tự dò: ./data/interim, cạnh script, <repo>/data/interim",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thư mục lưu ma trận. Bỏ trống = <gốc dữ liệu>/data/features",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="File YAML cấu hình. Bỏ trống = tự dò; không có file thì dùng mặc định built-in",
    )

    args = parser.parse_args()

    build_account_day_matrix(
        start_day=args.start_day,
        end_day=args.end_day,
        interim_dir=Path(args.interim_dir) if args.interim_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        config_path=Path(args.config) if args.config else None,
    )


    