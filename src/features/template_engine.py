"""
Engine tính đặc trưng TỰ ĐỘNG từ Combinatorial Template (schema v4.0).

Một ứng viên ``Candidate(measure, obj, entity, window)`` (xem ``src/features/templates.py``)
được tính bằng đúng một đường mã chung — không có hàm viết tay cho từng biến — nên bất kỳ
tổ hợp nào lọt vòng 1 đều cài đặt được ngay, và bản chạy trong pipeline (``extractor``)
dùng CHÍNH engine này cho các biến đã qua phễu.

Hai tầng tính:

1. ``compute_day``  — chỉ thấy MỘT ngày: đếm, đếm phân biệt, khung 15 phút, evenness,
   fanout, và *hồ sơ* ``(tài khoản, giá trị thực thể, số sự kiện)`` làm vật liệu lịch sử.
2. ``compute_history`` — ghép mọi ngày: novelty / jaccard / dist_shift / delta_mean /
   active_days / recency (chỉ dùng ngày ≤ t−1) và peer_z (cắt ngang cùng ngày t).

Quy ước giá trị (đã chốt trước khi đo, ghi vào schema):

* Biến dạng ĐẾM (count, distinct, fanout, novelty) trả về ``log1p(số đếm)``.
* Biến dạng TỶ LỆ (share, novelty_share, jaccard, dist_shift, evenness) ∈ [0, 1].
* ``peer_z`` cắt ở ±10 (giống ``volume_robust_z_7d``).
* NULL = *không xác định* (không có sự kiện của đối tượng đó / chưa có lịch sử), không bịa 0.

Giá trị thực thể được băm sang ``UInt64`` để hồ sơ 60 ngày vẫn vừa bộ nhớ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl

from src.features.templates import BUCKET_WINDOWS, Candidate

__all__ = ["DayOutput", "compute_day", "compute_history", "ACCOUNT_KEYS"]

ACCOUNT_KEYS: List[str] = ["DomainName", "UserName"]
_ACCT = "_acct"
_BUCKET_SECONDS = {"5m": 300, "15m": 900, "60m": 3600}
_HIST_DAYS = {"7d": 7, "14d": 14}
_PEER_CLIP = 10.0
_MIN_SCALE = 1e-9


@dataclass
class DayOutput:
    """Kết quả một ngày: khung đặc trưng trong ngày + hồ sơ thực thể cho tầng lịch sử."""
    intraday: pl.DataFrame                              # DomainName, UserName, _acct, day, cột ...
    profiles: Dict[Tuple[str, str], pl.DataFrame]       # (obj, entity) -> (_acct, day, v, n)


# ---------------------------------------------------------------------------
# Chuẩn bị cột sự kiện
# ---------------------------------------------------------------------------
def _valid_str(col: str) -> pl.Expr:
    s = pl.col(col).str.strip_chars()
    return pl.col(col).is_not_null() & (s != "") & (s != "Unknown")


def _prepare(events: pl.DataFrame, fcfg: Dict[str, Any]) -> pl.DataFrame:
    off_start, off_end = int(fcfg["off_hours_start"]), int(fcfg["off_hours_end"])
    hour = ((pl.col("Time") % 86400) // 3600).cast(pl.Int32)
    src_ok, host_ok = _valid_str("Source"), _valid_str("LogHost")
    cols = [
        pl.concat_str(ACCOUNT_KEYS, separator="\x1f").hash().alias(_ACCT),
        pl.col("Time"),
        (pl.col("EventID") == 4625).alias("_o_fail"),
        (pl.col("EventID") == 4624).alias("_o_success"),
        ((hour >= off_start) | (hour <= off_end)).alias("_o_night"),
        pl.when(src_ok).then(pl.col("Source").str.strip_chars().hash()).alias("_e_Source"),
        pl.when(host_ok).then(pl.col("LogHost").str.strip_chars().hash()).alias("_e_LogHost"),
        pl.when(src_ok & host_ok)
        .then(pl.concat_str([pl.col("Source").str.strip_chars(), pl.col("LogHost").str.strip_chars()],
                            separator="\x1f").hash())
        .alias("_e_pair"),
        hour.cast(pl.UInt64).alias("_e_hour"),
        pl.col("LogonType").cast(pl.UInt64).alias("_e_logon_type"),
        pl.col("AuthenticationPackage").hash().alias("_e_auth")
        if "AuthenticationPackage" in events.columns else pl.lit(None, pl.UInt64).alias("_e_auth"),
        pl.when(pl.col("FailureReason").is_not_null()).then(pl.col("FailureReason").hash())
        .alias("_e_fail_reason")
        if "FailureReason" in events.columns else pl.lit(None, pl.UInt64).alias("_e_fail_reason"),
    ]
    return events.select(ACCOUNT_KEYS + cols)


def _obj_filter(obj: str) -> pl.Expr:
    return pl.lit(True) if obj == "all" else pl.col(f"_o_{obj}")


def _pielou(n: pl.Expr) -> pl.Expr:
    p = n / n.sum()
    s = n.len().cast(pl.Float64)
    return pl.when(s > 1).then((-(p * p.log())).sum() / s.log()).otherwise(0.0)


# ---------------------------------------------------------------------------
# Tầng 1 — trong ngày
# ---------------------------------------------------------------------------
def _needed_profiles(specs: Sequence[Candidate]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for c in specs:
        if c.entity != "event" and (c.obj, c.entity) not in out:
            out.append((c.obj, c.entity))
    return out


def base_column(c: Candidate) -> str:
    """Tên cột thống kê nền (dùng cho delta_mean / peer_z)."""
    if c.base == "count":
        return f"_base_count_{c.obj}"
    if c.base == "share":
        return f"_base_share_{c.obj}"
    return f"_base_distinct_{c.obj}_{c.entity}"


def compute_day(
    events: pl.DataFrame,
    day: int,
    specs: Sequence[Candidate],
    fcfg: Dict[str, Any],
) -> DayOutput:
    """Tính mọi đặc trưng TRONG NGÀY của ``specs`` + hồ sơ thực thể cho tầng lịch sử."""
    ev = _prepare(events, fcfg)

    # Hàng gốc: mọi tài khoản có sự kiện trong ngày + số sự kiện của từng đối tượng
    rows = ev.group_by(ACCOUNT_KEYS).agg([
        pl.col(_ACCT).first(),
        pl.len().alias("_n_all"),
        pl.col("_o_fail").sum().alias("_n_fail"),
        pl.col("_o_success").sum().alias("_n_success"),
        pl.col("_o_night").sum().alias("_n_night"),
    ])
    for o in ("all", "fail", "night", "success"):
        rows = rows.with_columns(pl.col(f"_n_{o}").cast(pl.Float64).log1p().alias(f"_base_count_{o}"))
    for o in ("fail", "night", "success"):
        rows = rows.with_columns((pl.col(f"_n_{o}") / pl.col("_n_all")).alias(f"_base_share_{o}"))

    profiles: Dict[Tuple[str, str], pl.DataFrame] = {}
    for o, e in _needed_profiles(specs):
        prof = (
            ev.filter(_obj_filter(o) & pl.col(f"_e_{e}").is_not_null())
            .group_by([_ACCT, f"_e_{e}"])
            .agg(pl.len().cast(pl.UInt32).alias("n"))
            .rename({f"_e_{e}": "v"})
        )
        profiles[(o, e)] = prof
        agg = prof.group_by(_ACCT).agg([
            pl.len().cast(pl.Float64).log1p().alias(f"_base_distinct_{o}_{e}"),
            _pielou(pl.col("n").cast(pl.Float64)).alias(f"_even_{o}_{e}"),
        ])
        rows = rows.join(agg, on=_ACCT, how="left").with_columns(
            pl.col(f"_base_distinct_{o}_{e}").fill_null(0.0)
        )

    feats: List[pl.Expr] = []
    for c in specs:
        if c.needs_history or c.measure == "peer_z":
            continue
        if c.measure == "share":
            feats.append(pl.col(f"_base_share_{c.obj}").alias(c.name))
        elif c.measure == "count" and c.window == "1d":
            feats.append(pl.col(f"_base_count_{c.obj}").alias(c.name))
        elif c.measure == "distinct" and c.window == "1d":
            feats.append(pl.col(f"_base_distinct_{c.obj}_{c.entity}").alias(c.name))
        elif c.measure == "evenness":
            feats.append(
                pl.when(pl.col(f"_n_{c.obj}") > 0).then(pl.col(f"_even_{c.obj}_{c.entity}")).alias(c.name)
            )
        elif c.measure in ("count", "distinct") and c.window in BUCKET_WINDOWS:
            rows = rows.join(_bucket_max(ev, c), on=_ACCT, how="left")
            feats.append(pl.col(c.name).fill_null(0.0))
        elif c.measure == "fanout":
            rows = rows.join(_fanout(profiles[(c.obj, c.entity)], c.name), on=_ACCT, how="left")
            feats.append(pl.col(c.name).fill_null(0.0))
        else:  # pragma: no cover - ngữ pháp đã chặn
            raise ValueError(f"compute_day không hỗ trợ {c.name}")

    keep_base = [c for c in rows.columns if c.startswith("_base_")] + [f"_n_{o}" for o in ("all", "fail", "night", "success")]
    out = rows.with_columns(feats).with_columns(pl.lit(day).cast(pl.Int32).alias("day"))
    out = out.select(ACCOUNT_KEYS + [_ACCT, "day"] + [f.meta.output_name() for f in feats] + keep_base)
    profiles = {k: v.with_columns(pl.lit(day).cast(pl.Int32).alias("day")) for k, v in profiles.items()}
    return DayOutput(out, profiles)


def _bucket_max(ev: pl.DataFrame, c: Candidate) -> pl.DataFrame:
    """Giá trị lớn nhất trên các khung cố định ``window`` trong ngày (log1p)."""
    b = (pl.col("Time") // _BUCKET_SECONDS[c.window]).alias("_b")
    sub = ev.filter(_obj_filter(c.obj))
    if c.measure == "count":
        per = sub.group_by([_ACCT, b]).agg(pl.len().alias("_k"))
    else:
        per = (
            sub.filter(pl.col(f"_e_{c.entity}").is_not_null())
            .group_by([_ACCT, b]).agg(pl.col(f"_e_{c.entity}").n_unique().alias("_k"))
        )
    return per.group_by(_ACCT).agg(pl.col("_k").max().cast(pl.Float64).log1p().alias(c.name))


def _fanout(profile: pl.DataFrame, name: str) -> pl.DataFrame:
    """Với mỗi thực thể: số tài khoản chạm tới nó trong ngày; tài khoản lấy max (log1p)."""
    per_value = profile.group_by("v").agg(pl.len().alias("_accts"))
    return (
        profile.join(per_value, on="v", how="left")
        .group_by(_ACCT).agg(pl.col("_accts").max().cast(pl.Float64).log1p().alias(name))
    )


# ---------------------------------------------------------------------------
# Tầng 2 — liên ngày (chỉ ≤ t−1) + peer cắt ngang
# ---------------------------------------------------------------------------
def compute_history(
    intraday: pl.DataFrame,
    profiles: Dict[Tuple[str, str], pl.DataFrame],
    specs: Sequence[Candidate],
    entity_type: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Ghép mọi ngày: trả về ``intraday`` + cột của các ``specs`` cần lịch sử / peer.

    ``entity_type`` = khung ``(DomainName, UserName, day, entity_type)`` cho nhóm peer;
    bắt buộc khi có ``peer_z``.
    """
    out = intraday
    key = [_ACCT, "day"]
    day_min = int(intraday["day"].min())

    for c in specs:
        if c.measure in ("novelty", "novelty_share", "jaccard", "dist_shift", "recency"):
            out = out.join(_entity_history(profiles[(c.obj, c.entity)], c, day_min), on=key, how="left")
        elif c.measure in ("delta_mean", "active_days"):
            out = out.join(_daily_history(intraday, c, day_min), on=key, how="left")

    # novelty / fanout: không có thực thể ⇒ 0 (đúng nghĩa đếm)
    zero_fill = [c.name for c in specs if c.measure == "novelty"]
    if zero_fill:
        out = out.with_columns([pl.col(n).fill_null(0.0) for n in zero_fill])

    peers = [c for c in specs if c.measure == "peer_z"]
    if peers:
        if entity_type is None:
            raise ValueError("peer_z cần khung entity_type")
        out = out.join(entity_type.select(ACCOUNT_KEYS + ["day", "entity_type"]),
                       on=ACCOUNT_KEYS + ["day"], how="left")
        for c in peers:
            out = out.with_columns(_robust_z(pl.col(base_column(c)), ["day", "entity_type"]).alias(c.name))
        out = out.drop("entity_type")
    return out


def _robust_z(x: pl.Expr, over: List[str]) -> pl.Expr:
    med = x.median().over(over)
    iqr = (x.quantile(0.75, "linear") - x.quantile(0.25, "linear")).over(over) / 1.349
    std = x.std().over(over)
    return (
        pl.when(iqr > _MIN_SCALE).then((x - med) / iqr)
        .when(std.fill_null(0.0) > _MIN_SCALE).then((x - med) / std)
        .otherwise(0.0)
        .clip(-_PEER_CLIP, _PEER_CLIP)
    )


def _entity_history(prof: pl.DataFrame, c: Candidate, day_min: int) -> pl.DataFrame:
    """novelty / novelty_share / jaccard / dist_shift / recency trên hồ sơ ``(acct, day, v, n)``."""
    key = [_ACCT, "day"]
    p = (
        prof.sort([_ACCT, "v", "day"])
        .with_columns(pl.col("day").shift(1).over([_ACCT, "v"]).alias("_prev"))
    )
    if c.measure == "recency":
        return p.group_by(key).agg(
            (pl.col("day") - pl.col("_prev")).min().cast(pl.Float64).alias(c.name)
        )

    w = _HIST_DAYS[c.window]
    p = p.with_columns(
        (pl.col("_prev").is_null() | (pl.col("_prev") < pl.col("day") - w)).alias("_new")
    )
    today = p.group_by(key).agg([
        pl.len().alias("_S"),
        pl.col("_new").sum().alias("_new_cnt"),
        pl.col("n").sum().cast(pl.Float64).alias("_n_tot"),
        (pl.col("n").cast(pl.Float64) * pl.col("_new").cast(pl.Float64)).sum().alias("_n_new"),
    ])
    if c.measure == "novelty":
        return today.select(key + [pl.col("_new_cnt").cast(pl.Float64).log1p().alias(c.name)])
    if c.measure == "novelty_share":
        return today.select(key + [(pl.col("_n_new") / pl.col("_n_tot")).alias(c.name)])

    # Tập/phân bố của cửa sổ [t−w, t−1]: mỗi lần hiện diện ngày d phủ các t ∈ [d+1, d+w]
    hist = (
        prof.with_columns(pl.int_ranges(pl.col("day") + 1, pl.col("day") + w + 1).alias("_t"))
        .explode("_t")
        .group_by([_ACCT, "_t", "v"]).agg(pl.col("n").sum().alias("_q_n"))
        .rename({"_t": "day"})
        .join(today.select(key), on=key, how="semi")
    )
    h_size = hist.group_by(key).agg(pl.len().alias("_H"))
    if c.measure == "jaccard":
        j = today.join(h_size, on=key, how="left").with_columns(pl.col("_H").fill_null(0))
        inter = pl.col("_S") - pl.col("_new_cnt")
        return j.select(key + [
            pl.when(pl.col("_H") > 0)
            .then(inter.cast(pl.Float64) / (pl.col("_S") + pl.col("_H") - inter).cast(pl.Float64))
            .alias(c.name)
        ])

    # dist_shift: total variation giữa phân bố hôm nay (p) và phân bố cộng dồn cửa sổ (q)
    p_today = prof.select([_ACCT, "day", "v", pl.col("n").cast(pl.Float64).alias("_p_n")])
    p_today = p_today.with_columns((pl.col("_p_n") / pl.col("_p_n").sum().over(key)).alias("_p"))
    q = hist.with_columns(
        (pl.col("_q_n").cast(pl.Float64) / pl.col("_q_n").sum().over(key).cast(pl.Float64)).alias("_q")
    )
    tv = (
        p_today.select([_ACCT, "day", "v", "_p"])
        .join(q.select([_ACCT, "day", "v", "_q"]), on=[_ACCT, "day", "v"], how="full", coalesce=True)
        .with_columns(pl.col("_p").fill_null(0.0), pl.col("_q").fill_null(0.0))
        .group_by(key).agg((0.5 * (pl.col("_p") - pl.col("_q")).abs().sum()).alias(c.name))
    )
    return today.select(key).join(h_size, on=key, how="left").join(tv, on=key, how="left").select(
        key + [pl.when(pl.col("_H").fill_null(0) > 0).then(pl.col(c.name)).alias(c.name)]
    )


def _daily_history(intraday: pl.DataFrame, c: Candidate, day_min: int) -> pl.DataFrame:
    """delta_mean / active_days trên LƯỚI DÀY nội bộ (ngày trống = 0 với nền đếm)."""
    key = [_ACCT, "day"]
    w = _HIST_DAYS[c.window]
    day_max = int(intraday["day"].max())
    grid = (
        intraday.select(_ACCT).unique()
        .with_columns(pl.int_ranges(day_min, day_max + 1).alias("day")).explode("day")
        .with_columns(pl.col("day").cast(pl.Int32))
    )
    active = intraday.select(key + [pl.lit(1.0).alias("_act")])
    warm = pl.col("day") - w >= day_min                         # đủ w ngày lịch sử quan sát

    if c.measure == "active_days":
        dense = grid.join(active, on=key, how="left").with_columns(pl.col("_act").fill_null(0.0)).sort(key)
        s = pl.col("_act").shift(1).rolling_sum(w, min_samples=w).over(_ACCT)
        res = dense.with_columns(pl.when(warm).then(s).alias(c.name))
    else:
        col = base_column(c)
        dense = (
            grid.join(intraday.select(key + [col]), on=key, how="left")
            .join(active, on=key, how="left").sort(key)
        )
        if c.base == "share":
            # tỷ lệ chỉ xác định ở ngày có hoạt động ⇒ trung bình trên các ngày CÓ hoạt động
            val = pl.col(col).fill_null(0.0).shift(1).rolling_sum(w, min_samples=w).over(_ACCT)
            cnt = pl.col("_act").fill_null(0.0).shift(1).rolling_sum(w, min_samples=w).over(_ACCT)
            mean = pl.when(cnt > 0).then(val / cnt)
        else:
            mean = pl.col(col).fill_null(0.0).shift(1).rolling_mean(w, min_samples=w).over(_ACCT)
        res = dense.with_columns(pl.when(warm).then(pl.col(col) - mean).alias(c.name))
    return res.join(active.select(key), on=key, how="semi").select(key + [c.name])
