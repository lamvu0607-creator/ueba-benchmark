"""
Hồ sơ hành vi tính TỪ TRAIN (ngày ``start_day..train_end_day``, mặc định 1..42) cho bộ tiêm log.

Mọi thống kê ở đây chỉ đọc tầng **interim** — cùng tầng bộ tiêm ghi vào — qua đúng hàm đọc của
pipeline (:func:`src.features.extractor.load_day_events`). Vì vậy:

* ``day`` = số thứ tự file ``event_462x_day-NN.parquet`` (không suy từ timestamp; xem
  ``src/injection/layout.py``). Mỗi ngày đọc xong được đối chiếu ``Time`` với :func:`day_window`
  và số sự kiện lệch được ghi vào ``network["days_outside_window"]`` để kiểm tra.
* ``UserName`` rỗng/"null" bị loại, ``DomainName`` được chuẩn hoá (lowercase, rỗng -> "Unknown")
  y hệt extractor ⇒ khoá ``(DomainName, UserName)`` của hồ sơ khớp khoá ma trận đặc trưng.
* giờ = ``(Time % 86400) // 3600``; ngoài giờ = ``hour >= off_hours_start | hour <= off_hours_end``
  (cùng biểu thức với ``off_hours_ratio`` của extractor, kể cả biên ``<=``).
* Source hợp lệ = khác null, khác rỗng, khác "Unknown" (như ``distinct_sources_count``);
  LogHost hợp lệ = khác null, khác rỗng (như ``dst_host_entropy``).

Hồ sơ TÀI KHOẢN (khoá ``DomainName, UserName``):

    accounts             1 dòng / tài khoản: entity_type, n_events, n_success, n_failure,
                         n_active_days, first_day, last_day, median_daily_events (trung vị số sự kiện
                         trên các NGÀY CÓ HOẠT ĐỘNG), off_hours_ratio (gộp mọi sự kiện train),
                         n_known_sources, n_known_loghosts, n_inactive_gaps, max_internal_gap_days
    account_sources      tập Source quen: (khoá, Source, n_events, n_success, n_days)
    account_loghosts     tập LogHost quen: (khoá, LogHost, n_events, n_success, n_days)
    account_logon_types  tỷ lệ từng LogonType: (khoá, LogonType, n_events, share) — share cộng lại = 1
    account_gaps         các khoảng ngày không hoạt động trong [start_day, train_end_day]:
                         (khoá, gap_start_day, gap_end_day, n_days, kind ∈ leading/internal/trailing)
    daily_counts         số sự kiện mỗi (khoá, day) có hoạt động — vật liệu của median và gaps

Hồ sơ TOÀN MẠNG:

    hosts                1 dòng / máy (tên xuất hiện ở LogHost hoặc Source): n_accounts (dùng máy ở
                         vai trò bất kỳ), n_accounts_as_loghost, n_accounts_as_source,
                         n_user_accounts (entity_type == "User"), n_events, n_days
    lockout_episodes     mỗi lần khoá tài khoản (sự kiện "Account locked out." đầu tiên sau ≥
                         ``lockout_episode_gap_s`` không bị khoá) + độ dài chuỗi 4625 "bad password"
                         liền trước nó (``streak``) và số bad password trong cửa sổ ``lockout_window_s``
    network              dict: ngưỡng khoá L ước lượng (mode của ``streak`` ≥ 1), phân bố streak,
                         tham số, các ngày đã đọc

Ước lượng L: chuỗi được đếm trên dòng 4625 CỦA CHÍNH tài khoản (khoá miền là toàn miền, không theo
máy), theo thứ tự ``(Time, thứ tự dòng trong file)``; chuỗi dừng khi gặp một 4625 lý do khác hoặc
khoảng cách giữa hai sự kiện liên tiếp > ``lockout_window_s``. 4624 xen giữa KHÔNG làm đứt chuỗi:
trên LANL 4624 của cùng tài khoản xen vào cả trong lúc đang bị khoá (log tổng hợp từ nhiều máy), nên
coi 4624 là tín hiệu reset sẽ cắt gần hết chuỗi. Các lần khoá không có bad password nào đứng trước
(``streak`` = 0, thường do chuỗi bắt đầu trước ngày 1 hoặc ở tài khoản cục bộ) bị loại khỏi ước lượng
nhưng vẫn được đếm trong ``network["lockout"]["n_onsets_streak0"]``.
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_left
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import polars as pl

from src.features.extractor import (
    DEFAULT_FEATURE_CFG,
    available_days,
    entity_type_expr,
    load_day_events,
)
from src.injection.layout import events_outside_day

logger = logging.getLogger("ueba_benchmark.injection.profiles")

__all__ = [
    "ACCOUNT_KEYS",
    "BAD_PASSWORD_PATTERN",
    "LOCKED_OUT_PATTERN",
    "ProfileConfig",
    "TrainProfiles",
    "build_train_profiles",
    "profiles_from_day_events",
    "lockout_episodes",
    "estimate_lockout_threshold",
    "inactive_gaps",
]

ACCOUNT_KEYS: List[str] = ["DomainName", "UserName"]
BAD_PASSWORD_PATTERN = "bad password"
LOCKED_OUT_PATTERN = "account locked out"

_TABLES = (
    "accounts", "account_sources", "account_loghosts", "account_logon_types",
    "account_gaps", "daily_counts", "hosts", "lockout_episodes",
)


@dataclass(frozen=True)
class ProfileConfig:
    """Tham số tính hồ sơ. ``train_end_day`` là ngày train CUỐI (bao gồm) — mặc định split_day = 42."""

    start_day: int = 1
    train_end_day: int = 42
    off_hours_start: int = int(DEFAULT_FEATURE_CFG["off_hours_start"])
    off_hours_end: int = int(DEFAULT_FEATURE_CFG["off_hours_end"])
    lockout_window_s: int = 1800
    lockout_episode_gap_s: int = 1800

    @classmethod
    def from_system_config(cls, cfg: Dict[str, Any], **overrides: Any) -> "ProfileConfig":
        """Lấy ``split_day`` và giờ ngoài giờ từ ``configs/system_config.yaml``."""
        feats = cfg.get("features", {}) or {}
        ev = cfg.get("evaluation", {}) or {}
        kw: Dict[str, Any] = {}
        if ev.get("split_day") is not None:
            kw["train_end_day"] = int(ev["split_day"])
        for key in ("off_hours_start", "off_hours_end"):
            if feats.get(key) is not None:
                kw[key] = int(feats[key])
        kw.update(overrides)
        return cls(**kw)

    def __post_init__(self) -> None:
        if self.start_day < 1 or self.train_end_day < self.start_day:
            raise ValueError(f"Dải ngày train không hợp lệ: {self.start_day}..{self.train_end_day}.")


# --------------------------------------------------------------------------- biểu thức dùng chung

def _hour() -> pl.Expr:
    return ((pl.col("Time") % 86400) // 3600).cast(pl.Int32)


def _off_hours(cfg: ProfileConfig) -> pl.Expr:
    return (_hour() >= cfg.off_hours_start) | (_hour() <= cfg.off_hours_end)


def _valid_source() -> pl.Expr:
    s = pl.col("Source")
    return s.is_not_null() & (s.str.strip_chars() != "") & (s != "Unknown")


def _valid_loghost() -> pl.Expr:
    h = pl.col("LogHost")
    return h.is_not_null() & (h.str.strip_chars() != "")


def _reason(pattern: str) -> pl.Expr:
    return pl.col("FailureReason").str.to_lowercase().str.contains(pattern, literal=True).fill_null(False)


# --------------------------------------------------------------------------- tổng hợp một ngày

def _aggregate_day(events: pl.DataFrame, day: int, cfg: ProfileConfig) -> Dict[str, pl.DataFrame]:
    """Các bảng tổng hợp của MỘT ngày (đầu vào = khung của ``load_day_events``)."""
    d = pl.lit(int(day), pl.Int32).alias("day")
    ev = events.with_columns(
        (pl.col("EventID") == 4624).alias("_ok"),
        (int(day) * 1_000_000_000 + pl.int_range(pl.len(), dtype=pl.Int64)).alias("_seq"),
    )
    out: Dict[str, pl.DataFrame] = {}
    out["daily"] = ev.group_by(ACCOUNT_KEYS).agg(
        pl.len().alias("n_events"),
        pl.col("_ok").sum().alias("n_success"),
        _off_hours(cfg).sum().alias("n_off_hours"),
    ).with_columns(d)
    out["logon_types"] = ev.group_by(ACCOUNT_KEYS + ["LogonType"]).agg(pl.len().alias("n_events"))
    out["sources"] = (
        ev.filter(_valid_source())
        .with_columns(pl.col("Source").str.strip_chars())
        .group_by(ACCOUNT_KEYS + ["Source"])
        .agg(pl.len().alias("n_events"), pl.col("_ok").sum().alias("n_success"))
        .with_columns(d)
    )
    out["loghosts"] = (
        ev.filter(_valid_loghost())
        .with_columns(pl.col("LogHost").str.strip_chars())
        .group_by(ACCOUNT_KEYS + ["LogHost"])
        .agg(pl.len().alias("n_events"), pl.col("_ok").sum().alias("n_success"))
        .with_columns(d)
    )
    bad, locked = _reason(BAD_PASSWORD_PATTERN), _reason(LOCKED_OUT_PATTERN)
    out["failures"] = (
        ev.filter(pl.col("EventID") == 4625)
        .select(
            *ACCOUNT_KEYS, "Time", "_seq",
            pl.when(locked).then(pl.lit("locked")).when(bad).then(pl.lit("bad_pw"))
            .otherwise(pl.lit("other")).alias("kind"),
        )
        .with_columns(d)
    )
    return out


# --------------------------------------------------------------------------- khoá tài khoản

def lockout_episodes(failures: pl.DataFrame, window_s: int = 1800, episode_gap_s: int = 1800) -> pl.DataFrame:
    """
    Mỗi lần khoá (onset) + chuỗi bad password liền trước.

    ``failures``: các dòng 4625 có ``DomainName, UserName, Time, _seq, kind`` (``kind`` ∈
    ``locked``/``bad_pw``/``other``). Ra: ``DomainName, UserName, day, Time, streak, n_bad_pw_window``.
    """
    cols = ACCOUNT_KEYS + ["day", "Time", "streak", "n_bad_pw_window"]
    if failures.is_empty():
        return pl.DataFrame(schema={**{c: pl.String for c in ACCOUNT_KEYS}, "day": pl.Int32,
                                    "Time": pl.Int64, "streak": pl.Int64, "n_bad_pw_window": pl.Int64})
    f = failures.sort(ACCOUNT_KEYS + ["Time", "_seq"])
    accounts_with_lock = f.filter(pl.col("kind") == "locked").select(ACCOUNT_KEYS).unique()
    f = f.join(accounts_with_lock, on=ACCOUNT_KEYS, how="semi")
    rows: List[Dict[str, Any]] = []
    for (dom, user), g in f.group_by(*ACCOUNT_KEYS, maintain_order=True):
        times = g["Time"].to_list()
        kinds = g["kind"].to_list()
        days = g["day"].to_list()
        n_bad_before = list(accumulate((k == "bad_pw" for k in kinds), initial=0))
        last_lock: Optional[int] = None
        for i, k in enumerate(kinds):
            if k != "locked":
                continue
            t = times[i]
            is_onset = last_lock is None or t - last_lock >= episode_gap_s
            last_lock = t
            if not is_onset:
                continue
            streak, prev_t, j = 0, t, i - 1
            while j >= 0 and kinds[j] == "bad_pw" and prev_t - times[j] <= window_s:
                streak += 1
                prev_t = times[j]
                j -= 1
            n_win = n_bad_before[i] - n_bad_before[bisect_left(times, t - window_s, 0, i)]
            rows.append({"DomainName": dom, "UserName": user, "day": days[i], "Time": t,
                         "streak": streak, "n_bad_pw_window": n_win})
    if not rows:
        return lockout_episodes(pl.DataFrame(), window_s, episode_gap_s)
    return pl.DataFrame(rows).with_columns(pl.col("day").cast(pl.Int32)).select(cols)


def estimate_lockout_threshold(episodes: pl.DataFrame) -> Dict[str, Any]:
    """L = mode của ``streak`` trên các lần khoá có ≥ 1 bad password đứng trước (hoà -> giá trị nhỏ nhất)."""
    n = episodes.height
    pos = episodes.filter(pl.col("streak") >= 1)
    res: Dict[str, Any] = {
        "n_onsets": n,
        "n_onsets_streak0": n - pos.height,
        "n_accounts": episodes.select(ACCOUNT_KEYS).unique().height if n else 0,
        "L": None,
        "L_support": 0.0,
        "streak_median": None,
        "streak_hist": {},
    }
    if pos.is_empty():
        return res
    hist = pos.group_by("streak").len().sort("streak")
    res["streak_hist"] = {str(s): int(c) for s, c in hist.iter_rows()}  # khoá chuỗi: lưu JSON không đổi kiểu
    top = hist.sort(["len", "streak"], descending=[True, False]).row(0)
    res["L"] = int(top[0])
    res["L_support"] = float(top[1]) / pos.height
    res["streak_median"] = float(pos["streak"].median())
    return res


# --------------------------------------------------------------------------- khoảng không hoạt động

def inactive_gaps(daily_counts: pl.DataFrame, days: Iterable[int]) -> pl.DataFrame:
    """
    Các chuỗi ngày liên tiếp (trong lịch ``days`` đã đọc) mà tài khoản KHÔNG có sự kiện.

    ``kind``: ``leading`` (trước ngày hoạt động đầu tiên), ``internal`` (giữa hai ngày hoạt động),
    ``trailing`` (sau ngày hoạt động cuối tới hết train). Ngày thiếu file không nằm trong ``days``
    nên không được tính là ngày không hoạt động.
    """
    cal = sorted(int(d) for d in days)
    schema = {"DomainName": pl.String, "UserName": pl.String, "gap_start_day": pl.Int32,
              "gap_end_day": pl.Int32, "n_days": pl.Int32, "kind": pl.String}
    if not cal or daily_counts.is_empty():
        return pl.DataFrame(schema=schema)
    rank = pl.DataFrame({"day": cal, "_r": list(range(len(cal)))}, schema={"day": pl.Int32, "_r": pl.Int32})
    lut = pl.DataFrame({"_r": list(range(len(cal))), "_d": cal}, schema={"_r": pl.Int32, "_d": pl.Int32})
    last = len(cal) - 1
    act = (
        daily_counts.select(ACCOUNT_KEYS + [pl.col("day").cast(pl.Int32)])
        .unique()
        .join(rank, on="day", how="inner")
        .sort(ACCOUNT_KEYS + ["_r"])
        .with_columns(pl.col("_r").shift(1).over(ACCOUNT_KEYS).alias("_prev"))
    )
    internal = act.filter(pl.col("_prev").is_not_null() & (pl.col("_r") - pl.col("_prev") > 1)).select(
        *ACCOUNT_KEYS, (pl.col("_prev") + 1).alias("_a"), (pl.col("_r") - 1).alias("_b"), pl.lit("internal").alias("kind")
    )
    bounds = act.group_by(ACCOUNT_KEYS).agg(pl.col("_r").min().alias("_lo"), pl.col("_r").max().alias("_hi"))
    leading = bounds.filter(pl.col("_lo") > 0).select(
        *ACCOUNT_KEYS, pl.lit(0, pl.Int32).alias("_a"), (pl.col("_lo") - 1).alias("_b"), pl.lit("leading").alias("kind")
    )
    trailing = bounds.filter(pl.col("_hi") < last).select(
        *ACCOUNT_KEYS, (pl.col("_hi") + 1).alias("_a"), pl.lit(last, pl.Int32).alias("_b"), pl.lit("trailing").alias("kind")
    )
    gaps = pl.concat([g.with_columns(pl.col("_a", "_b").cast(pl.Int32)) for g in (leading, internal, trailing)])
    return (
        gaps.join(lut.rename({"_r": "_a", "_d": "gap_start_day"}), on="_a", how="left")
        .join(lut.rename({"_r": "_b", "_d": "gap_end_day"}), on="_b", how="left")
        .select(
            *ACCOUNT_KEYS, "gap_start_day", "gap_end_day",
            (pl.col("_b") - pl.col("_a") + 1).cast(pl.Int32).alias("n_days"), "kind",
        )
        .sort(ACCOUNT_KEYS + ["gap_start_day"])
    )


# --------------------------------------------------------------------------- kết quả

@dataclass
class TrainProfiles:
    """Toàn bộ hồ sơ train. Bảng dạng dài; các hàm truy cập trả tập/dict cho một tài khoản."""

    accounts: pl.DataFrame
    account_sources: pl.DataFrame
    account_loghosts: pl.DataFrame
    account_logon_types: pl.DataFrame
    account_gaps: pl.DataFrame
    daily_counts: pl.DataFrame
    hosts: pl.DataFrame
    lockout_episodes: pl.DataFrame
    network: Dict[str, Any] = field(default_factory=dict)

    # ---- truy cập theo tài khoản
    def _rows(self, table: pl.DataFrame, domain: str, user: str) -> pl.DataFrame:
        return table.filter((pl.col("DomainName") == domain) & (pl.col("UserName") == user))

    def known_sources(self, domain: str, user: str) -> Set[str]:
        return set(self._rows(self.account_sources, domain, user)["Source"].to_list())

    def known_loghosts(self, domain: str, user: str) -> Set[str]:
        return set(self._rows(self.account_loghosts, domain, user)["LogHost"].to_list())

    def logon_type_shares(self, domain: str, user: str) -> Dict[int, float]:
        r = self._rows(self.account_logon_types, domain, user)
        return {int(t): float(s) for t, s in zip(r["LogonType"], r["share"]) if t is not None}

    def gaps(self, domain: str, user: str) -> pl.DataFrame:
        return self._rows(self.account_gaps, domain, user)

    def account(self, domain: str, user: str) -> Dict[str, Any]:
        r = self._rows(self.accounts, domain, user)
        if r.is_empty():
            raise KeyError(f"Không có hồ sơ train cho ({domain!r}, {user!r}).")
        return r.row(0, named=True)

    @property
    def lockout_threshold(self) -> Optional[int]:
        return (self.network.get("lockout") or {}).get("L")

    # ---- lưu / nạp
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in _TABLES:
            getattr(self, name).write_parquet(out / f"{name}.parquet")
        (out / "network.json").write_text(json.dumps(self.network, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, out_dir: Path | str) -> "TrainProfiles":
        src = Path(out_dir)
        tables = {name: pl.read_parquet(src / f"{name}.parquet") for name in _TABLES}
        network = json.loads((src / "network.json").read_text(encoding="utf-8"))
        return cls(**tables, network=network)


# --------------------------------------------------------------------------- dựng hồ sơ

def _add_day(parts: Dict[str, List[pl.DataFrame]], outside: Dict[int, int], ev: pl.DataFrame, day: int,
             cfg: ProfileConfig) -> None:
    n_out = events_outside_day(ev.select("Time"), day).height
    if n_out:
        outside[day] = n_out
        logger.warning("Ngày %d: %d sự kiện có Time ngoài cửa sổ ngày (vẫn tính theo file).", day, n_out)
    for k, v in _aggregate_day(ev, day, cfg).items():
        parts.setdefault(k, []).append(v)


def profiles_from_day_events(
    day_events: Dict[int, pl.DataFrame],
    cfg: ProfileConfig = ProfileConfig(),
    feature_cfg: Optional[Dict[str, Any]] = None,
    calendar: Optional[Iterable[int]] = None,
) -> TrainProfiles:
    """
    Dựng hồ sơ từ ``{day: events}`` (khung của ``load_day_events``). Ngày > ``train_end_day`` hoặc
    < ``start_day`` bị TỪ CHỐI — hồ sơ không bao giờ được nhìn thấy test.

    ``calendar``: các ngày được coi là "có dữ liệu" khi tìm khoảng không hoạt động (mặc định mọi
    ngày ``start_day..train_end_day``).
    """
    cal = sorted(set(int(d) for d in (calendar if calendar is not None else range(cfg.start_day, cfg.train_end_day + 1))))
    bad_days = [d for d in set(day_events) | set(cal) if not cfg.start_day <= int(d) <= cfg.train_end_day]
    if bad_days:
        raise ValueError(
            f"Ngày {sorted(bad_days)} nằm ngoài train {cfg.start_day}..{cfg.train_end_day} — hồ sơ chỉ tính từ train."
        )
    missing = sorted(set(int(d) for d in day_events) - set(cal))
    if missing:
        raise ValueError(f"Ngày {missing} có sự kiện nhưng không nằm trong calendar.")
    parts: Dict[str, List[pl.DataFrame]] = {}
    outside: Dict[int, int] = {}
    for d in sorted(day_events):
        _add_day(parts, outside, day_events[d], int(d), cfg)
    return _assemble(parts, cal, cfg, feature_cfg, outside)


def build_train_profiles(
    interim_dir: Path | str = "data/interim",
    cfg: ProfileConfig = ProfileConfig(),
    feature_cfg: Optional[Dict[str, Any]] = None,
) -> TrainProfiles:
    """Đọc từng ngày train ở tầng interim (``load_day_events``), tổng hợp dần rồi ghép hồ sơ."""
    interim = Path(interim_dir)
    days = [d for d in available_days(interim) if cfg.start_day <= d <= cfg.train_end_day]
    if not days:
        raise FileNotFoundError(f"Không có ngày train {cfg.start_day}..{cfg.train_end_day} trong '{interim}'.")
    parts: Dict[str, List[pl.DataFrame]] = {}
    outside: Dict[int, int] = {}
    read: List[int] = []
    cols = ["Time", "EventID", "UserName", "DomainName", "LogHost", "LogonType", "Source", "FailureReason"]
    for d in days:
        ev = load_day_events(d, interim)
        if ev is None:
            continue
        ev = ev.select(cols)
        _add_day(parts, outside, ev, d, cfg)
        read.append(d)
        logger.info("Hồ sơ train: đã tổng hợp ngày %02d (%d sự kiện).", d, ev.height)
        del ev
    return _assemble(parts, read, cfg, feature_cfg, outside)


def _assemble(
    parts: Dict[str, List[pl.DataFrame]],
    days_read: List[int],
    cfg: ProfileConfig,
    feature_cfg: Optional[Dict[str, Any]],
    outside: Dict[int, int],
) -> TrainProfiles:
    if not days_read:
        raise ValueError("Không có ngày train nào để dựng hồ sơ.")
    fcfg = feature_cfg or DEFAULT_FEATURE_CFG
    cat = {k: pl.concat(v, how="vertical") for k, v in parts.items()}

    daily = cat["daily"].select(ACCOUNT_KEYS + ["day", "n_events", "n_success", "n_off_hours"]).sort(ACCOUNT_KEYS + ["day"])

    def _entity_table(name: str, col: str) -> pl.DataFrame:
        return (
            cat[name].group_by(ACCOUNT_KEYS + [col]).agg(
                pl.col("n_events").sum(), pl.col("n_success").sum(), pl.col("day").n_unique().cast(pl.Int32).alias("n_days")
            ).sort(ACCOUNT_KEYS + [col])
        )

    sources = _entity_table("sources", "Source")
    loghosts = _entity_table("loghosts", "LogHost")

    logon_types = (
        cat["logon_types"].group_by(ACCOUNT_KEYS + ["LogonType"]).agg(pl.col("n_events").sum())
        .with_columns((pl.col("n_events") / pl.col("n_events").sum().over(ACCOUNT_KEYS)).alias("share"))
        .sort(ACCOUNT_KEYS + ["LogonType"])
    )

    gaps = inactive_gaps(daily, days_read)
    gap_summary = gaps.group_by(ACCOUNT_KEYS).agg(
        pl.len().cast(pl.Int32).alias("n_inactive_gaps"),
        pl.col("n_days").filter(pl.col("kind") == "internal").max().alias("max_internal_gap_days"),
    )

    accounts = (
        daily.group_by(ACCOUNT_KEYS).agg(
            pl.col("n_events").sum(),
            pl.col("n_success").sum(),
            pl.col("n_off_hours").sum(),
            pl.len().cast(pl.Int32).alias("n_active_days"),
            pl.col("day").min().alias("first_day"),
            pl.col("day").max().alias("last_day"),
            pl.col("n_events").median().alias("median_daily_events"),
        )
        .with_columns(
            entity_type_expr(fcfg),
            (pl.col("n_events") - pl.col("n_success")).alias("n_failure"),
            (pl.col("n_off_hours") / pl.col("n_events")).alias("off_hours_ratio"),
        )
        .join(sources.group_by(ACCOUNT_KEYS).len("n_known_sources"), on=ACCOUNT_KEYS, how="left")
        .join(loghosts.group_by(ACCOUNT_KEYS).len("n_known_loghosts"), on=ACCOUNT_KEYS, how="left")
        .join(gap_summary, on=ACCOUNT_KEYS, how="left")
        .with_columns(
            pl.col("n_known_sources", "n_known_loghosts").fill_null(0).cast(pl.Int32),
            pl.col("n_inactive_gaps").fill_null(0),
        )
        .select(
            ACCOUNT_KEYS + ["entity_type", "n_events", "n_success", "n_failure", "n_active_days",
                            "first_day", "last_day", "median_daily_events", "off_hours_ratio",
                            "n_known_sources", "n_known_loghosts", "n_inactive_gaps", "max_internal_gap_days"]
        )
        .sort(ACCOUNT_KEYS)
    )

    # ---- máy: hợp vai trò LogHost và Source của mọi tài khoản
    etype = accounts.select(ACCOUNT_KEYS + ["entity_type"])
    roles = pl.concat([
        cat["loghosts"].select(ACCOUNT_KEYS + [pl.col("LogHost").alias("host"), "day", "n_events", pl.lit("loghost").alias("role")]),
        cat["sources"].select(ACCOUNT_KEYS + [pl.col("Source").alias("host"), "day", "n_events", pl.lit("source").alias("role")]),
    ]).join(etype, on=ACCOUNT_KEYS, how="left")
    acct_id = pl.concat_str(*ACCOUNT_KEYS, separator="\\")
    hosts = (
        roles.with_columns(acct_id.alias("_acct"))
        .group_by("host").agg(
            pl.col("_acct").n_unique().cast(pl.Int32).alias("n_accounts"),
            pl.col("_acct").filter(pl.col("role") == "loghost").n_unique().cast(pl.Int32).alias("n_accounts_as_loghost"),
            pl.col("_acct").filter(pl.col("role") == "source").n_unique().cast(pl.Int32).alias("n_accounts_as_source"),
            pl.col("_acct").filter(pl.col("entity_type") == "User").n_unique().cast(pl.Int32).alias("n_user_accounts"),
            pl.col("n_events").filter(pl.col("role") == "loghost").sum().alias("n_events_as_loghost"),
            pl.col("n_events").filter(pl.col("role") == "source").sum().alias("n_events_as_source"),
            pl.col("day").n_unique().cast(pl.Int32).alias("n_days"),
        )
        .sort(["n_accounts", "host"], descending=[True, False])
    )

    episodes = lockout_episodes(cat["failures"], cfg.lockout_window_s, cfg.lockout_episode_gap_s)
    episodes = episodes.join(etype, on=ACCOUNT_KEYS, how="left")
    lock = estimate_lockout_threshold(episodes)
    lock["by_entity_type"] = {
        str(k[0] if isinstance(k, tuple) else k): estimate_lockout_threshold(g)["L"]
        for k, g in episodes.partition_by("entity_type", as_dict=True).items()
    } if episodes.height else {}

    network = {
        "train_days": days_read,
        "config": {
            "start_day": cfg.start_day, "train_end_day": cfg.train_end_day,
            "off_hours_start": cfg.off_hours_start, "off_hours_end": cfg.off_hours_end,
            "lockout_window_s": cfg.lockout_window_s, "lockout_episode_gap_s": cfg.lockout_episode_gap_s,
        },
        "n_accounts": accounts.height,
        "n_hosts": hosts.height,
        "days_outside_window": {str(k): v for k, v in outside.items()},
        "lockout": lock,
    }
    return TrainProfiles(
        accounts=accounts,
        account_sources=sources,
        account_loghosts=loghosts,
        account_logon_types=logon_types,
        account_gaps=gaps,
        daily_counts=daily.select(ACCOUNT_KEYS + ["day", "n_events"]),
        hosts=hosts,
        lockout_episodes=episodes,
        network=network,
    )
