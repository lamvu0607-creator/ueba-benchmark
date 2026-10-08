"""
Thống kê luật KHÔNG có sẵn trong ma trận đặc trưng -> tính từ log sự kiện 4624/4625 (theo từng ngày).

  * R2 ``r2_spray_accounts``    — với mỗi Source gây thất bại (4625) cho tài khoản trong ngày, đếm số
    tài khoản KHÁC NHAU mà Source đó gây thất bại cùng ngày; giá trị của dòng = max qua các Source của nó.
    **Đây là thông tin CHÉO giữa các tài khoản** (khác mọi đặc trưng core, vốn chỉ nhìn một tài khoản):
    dấu hiệu password spraying là một nguồn thử ít mật khẩu trên rất nhiều tài khoản, nên nhìn từng
    tài khoản riêng lẻ không thấy được. Chỉ dùng log của CÙNG ngày, không dùng nhãn.
  * R4 ``r4_new_sources``        — số Source (khác null) hôm nay chưa từng xuất hiện với tài khoản ở MỌI
    ngày trước đó (< t, kể cả ngày test trước t — thông tin có sẵn tại thời điểm vận hành).
  * R6 ``r6_new_logontype_events`` — số SỰ KIỆN hôm nay có LogonType chưa từng thấy ở tài khoản trước t.
  R4/R6 = NULL ở ngày đầu tiên tài khoản xuất hiện (chưa có lịch sử để so) — sau đó baseline impute
  bằng median train như mọi NULL khác.

Cùng một lượt đọc còn gom các "ca khoá" (tài khoản, ngày) của ngày TRAIN để ước lượng ngưỡng khoá L
cho biến thể ngưỡng ngoài của R1 (``estimate_lockout_threshold``).

Chuẩn hoá DomainName/UserName dùng đúng ``load_day_events`` của extractor để khoá khớp ma trận.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl

from src.baselines._common import ACCOUNT_KEYS, logger

__all__ = [
    "EVENT_STAT_COLUMNS",
    "EventHistory",
    "day_event_stats",
    "lockout_episodes",
    "estimate_lockout_threshold",
    "compute_event_rule_stats",
]

EVENT_STAT_COLUMNS: List[str] = ["r2_spray_accounts", "r4_new_sources", "r6_new_logontype_events"]

#: Chuỗi FailureReason của sự kiện khoá tài khoản (cùng quy ước với failure_locked_out_share).
LOCKOUT_PATTERN = "account locked out"


class EventHistory:
    """Trạng thái lịch sử tích luỹ theo ngày: tài khoản đã thấy, cặp (tài khoản, Source), (tài khoản, LogonType)."""

    def __init__(self) -> None:
        self.accounts = pl.DataFrame(schema={"DomainName": pl.String, "UserName": pl.String})
        self.sources = pl.DataFrame(schema={"DomainName": pl.String, "UserName": pl.String, "Source": pl.String})
        self.logon_types = pl.DataFrame(schema={"DomainName": pl.String, "UserName": pl.String, "LogonType": pl.Int64})


def _normalise(events: pl.DataFrame) -> pl.DataFrame:
    return events.select(
        pl.col("DomainName").cast(pl.String),
        pl.col("UserName").cast(pl.String),
        pl.col("EventID").cast(pl.Int64),
        pl.col("Source").cast(pl.String),
        pl.col("LogonType").cast(pl.Int64),
    )


def day_event_stats(events: pl.DataFrame, day: int, history: EventHistory) -> pl.DataFrame:
    """
    Thống kê R2/R4/R6 của một ngày cho mọi tài khoản có sự kiện, rồi CẬP NHẬT ``history`` bằng ngày này.

    Chỉ dùng log của ngày ``day`` và lịch sử các ngày < ``day`` (gọi theo thứ tự ngày tăng dần).
    """
    ev = _normalise(events)
    accounts = ev.select(ACCOUNT_KEYS).unique()

    fails = ev.filter((pl.col("EventID") == 4625) & pl.col("Source").is_not_null()).select(ACCOUNT_KEYS + ["Source"]).unique()
    per_source = fails.group_by("Source").agg(pl.len().alias("_n_accounts"))
    r2 = fails.join(per_source, on="Source").group_by(ACCOUNT_KEYS).agg(pl.col("_n_accounts").max().alias("r2_spray_accounts"))

    src_today = ev.filter(pl.col("Source").is_not_null()).select(ACCOUNT_KEYS + ["Source"]).unique()
    r4 = (
        src_today.join(history.sources, on=ACCOUNT_KEYS + ["Source"], how="anti")
        .group_by(ACCOUNT_KEYS).agg(pl.len().alias("r4_new_sources"))
    )
    lt_today = ev.filter(pl.col("LogonType").is_not_null())
    r6 = (
        lt_today.join(history.logon_types, on=ACCOUNT_KEYS + ["LogonType"], how="anti")
        .group_by(ACCOUNT_KEYS).agg(pl.len().alias("r6_new_logontype_events"))
    )
    has_history = pl.col("_seen").fill_null(False)
    out = (
        accounts.join(history.accounts.with_columns(pl.lit(True).alias("_seen")), on=ACCOUNT_KEYS, how="left")
        .join(r2, on=ACCOUNT_KEYS, how="left")
        .join(r4, on=ACCOUNT_KEYS, how="left")
        .join(r6, on=ACCOUNT_KEYS, how="left")
        .select(
            ACCOUNT_KEYS
            + [
                pl.lit(int(day)).cast(pl.Int32).alias("day"),
                pl.col("r2_spray_accounts").fill_null(0).cast(pl.Float64),
                pl.when(has_history).then(pl.col("r4_new_sources").fill_null(0)).otherwise(None).cast(pl.Float64).alias("r4_new_sources"),
                pl.when(has_history).then(pl.col("r6_new_logontype_events").fill_null(0)).otherwise(None).cast(pl.Float64).alias("r6_new_logontype_events"),
            ]
        )
    )

    history.accounts = pl.concat([history.accounts, accounts]).unique()
    history.sources = pl.concat([history.sources, src_today]).unique()
    history.logon_types = pl.concat([history.logon_types, lt_today.select(ACCOUNT_KEYS + ["LogonType"]).unique()]).unique()
    return out


def lockout_episodes(events: pl.DataFrame) -> np.ndarray:
    """
    Với mỗi (tài khoản, ngày) có sự kiện khoá: số thất bại KHÔNG phải khoá xảy ra trước sự kiện khoá đầu tiên.

    Ca có 0 thất bại trước khoá bị bỏ (bộ đếm khoá mang sang từ hôm trước, không đo được L trong ngày).
    """
    fails = events.filter(pl.col("EventID").cast(pl.Int64) == 4625).select(
        ACCOUNT_KEYS + [pl.col("Time").cast(pl.Int64), pl.col("FailureReason").cast(pl.String)]
    )
    is_lock = pl.col("FailureReason").str.to_lowercase().str.contains(LOCKOUT_PATTERN).fill_null(False)
    first_lock = fails.filter(is_lock).group_by(ACCOUNT_KEYS).agg(pl.col("Time").min().alias("_t_lock"))
    counts = (
        fails.filter(~is_lock)
        .join(first_lock, on=ACCOUNT_KEYS, how="inner")
        .filter(pl.col("Time") < pl.col("_t_lock"))
        .group_by(ACCOUNT_KEYS).agg(pl.len().alias("n"))
    )
    return counts["n"].to_numpy().astype(np.float64)


def estimate_lockout_threshold(episode_counts: Iterable[float]) -> Dict[str, Any]:
    """L = ceil(trung vị số thất bại trước khoá) qua các ca khoá của TRAIN."""
    arr = np.asarray(list(episode_counts), dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        raise ValueError("Không có ca khoá nào (thất bại > 0 trước khoá) trên train để ước lượng L.")
    return {
        "lockout_threshold": int(math.ceil(float(np.median(arr)))),
        "n_episodes": int(arr.size),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
    }


def compute_event_rule_stats(
    days: Iterable[int],
    interim_dir: Path | str,
    train_last_day: int,
    injected_events_dir: Optional[Path | str] = None,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Đọc log từng ngày theo thứ tự tăng dần -> ``(bảng R2/R4/R6 theo DomainName, UserName, day, thông tin L)``.

    Ngày có file trong ``injected_events_dir`` (layout ``events_injected/event_462x/...``) đọc từ đó;
    các ngày khác đọc ``interim_dir``. Ca khoá chỉ gom từ ngày <= ``train_last_day``.
    """
    from src.features.extractor import load_day_events
    from src.injection.layout import interim_day_path, overlay_days

    interim = Path(interim_dir)
    injected = Path(injected_events_dir) if injected_events_dir else None
    injected_days = set(overlay_days(injected)) if injected else set()
    if injected_days and min(injected_days) <= int(train_last_day):
        raise ValueError(f"injected_events_dir chứa ngày train {sorted(d for d in injected_days if d <= train_last_day)}.")

    history = EventHistory()
    frames: List[pl.DataFrame] = []
    episodes: List[np.ndarray] = []
    for day in sorted(int(d) for d in days):
        source = injected if day in injected_days else interim
        if source is injected and not all(interim_day_path(injected, eid, day).is_file() for eid in (4624, 4625)):
            raise FileNotFoundError(f"Ngày {day} trong '{injected}' phải có đủ file 4624 và 4625.")
        t0 = time.time()
        events = load_day_events(day, source)
        if events is None:
            continue
        frames.append(day_event_stats(events, day, history))
        if day <= int(train_last_day):
            episodes.append(lockout_episodes(events))
        logger.info("[rule_stats] ngày %02d (%s): %s sự kiện, %.1fs", day, source, f"{events.height:,}", time.time() - t0)
        del events

    if not frames:
        raise ValueError(f"Không đọc được ngày nào từ '{interim}'.")
    lockout = estimate_lockout_threshold(np.concatenate(episodes) if episodes else [])
    return pl.concat(frames), lockout
