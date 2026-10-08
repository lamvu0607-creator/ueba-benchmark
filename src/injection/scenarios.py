"""
Sáu kịch bản tấn công, ghép từ các thao tác của :mod:`src.injection.operations`.

Mỗi hàm kịch bản nhận hồ sơ train, kho khuôn, một nạn nhân (đã được :mod:`inject` chọn và xác nhận hợp
lệ), tham số đã rút từ config, một RNG có seed, và trả về :class:`ScenarioResult`: các sự kiện tiêm (21
cột interim, đã qua ``assert_within_days``) cùng ``params`` (tham số THỰC TẾ để ghi manifest). Kịch bản
không đọc/ghi file, không tự chọn nạn nhân, không quyết định ngày — những việc đó thuộc ``inject.py``.

Mọi căn cứ (vì sao mỗi kịch bản dựng sự kiện như vậy) nằm trong docstring của từng hàm. Nguyên tắc
xuyên suốt: **nhân bản sự kiện thật rồi sửa** (xem operations), **chỉ tiêm trong một ngày** cho tới khi
bật ``n_days`` trong config, và mọi thống kê hồ sơ đều TỪ TRAIN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import polars as pl

from src.injection.operations import (
    Account,
    TemplatePool,
    TimeProfile,
    add_failures,
    add_successes,
    conform_to_interim,
    draw_int,
    retype_logon,
    schedule,
    strip_helpers,
)
from src.injection.profiles import TrainProfiles

__all__ = [
    "SCENARIOS",
    "ScenarioError",
    "ScenarioResult",
    "VictimContext",
    "pick_unseen_sources",
    "scenario_brute_force",
    "scenario_password_spraying",
    "scenario_off_hours",
    "scenario_new_workstation_burst",
    "scenario_dormant_wakeup",
    "scenario_logon_type_switch",
]


class ScenarioError(ValueError):
    """Không dựng được kịch bản cho nạn nhân này (thiếu khuôn, thiếu máy mới, cửa sổ giờ quá hẹp…)."""


@dataclass
class ScenarioResult:
    """Kết quả một lần tiêm cho một nạn nhân."""

    events: pl.DataFrame                       # 21 cột interim, đã strip cột phụ
    params: Dict[str, Any] = field(default_factory=dict)
    campaign_id: Optional[str] = None

    @property
    def n_events(self) -> int:
        return self.events.height


@dataclass
class VictimContext:
    """Mọi thứ một kịch bản cần biết về một nạn nhân, do ``inject.py`` chuẩn bị."""

    account: Account
    day: int                                   # ngày dự định tiêm (ngày test)
    profiles: TrainProfiles
    pool: TemplatePool
    prefer_own: bool = True
    allow_peer: bool = True

    # tiện ích đọc hồ sơ (tất cả TỪ TRAIN)
    def known_sources(self) -> List[str]:
        return sorted(self.profiles.known_sources(self.account.domain, self.account.user))

    def known_loghosts(self) -> List[str]:
        return sorted(self.profiles.known_loghosts(self.account.domain, self.account.user))

    def logon_shares(self) -> Dict[int, float]:
        return self.profiles.logon_type_shares(self.account.domain, self.account.user)


# --------------------------------------------------------------------------- chọn máy mới (Source)

def pick_unseen_sources(
    ctx: VictimContext, n: int, rng: np.random.Generator, popularity_quantile: Sequence[float]
) -> List[str]:
    """
    ``n`` tên máy CÓ THẬT trong mạng (hồ sơ ``hosts``) mà nạn nhân CHƯA từng dùng làm Source, lọc theo độ
    phổ biến để không lấy máy "ai cũng dùng".

    Căn cứ: đặc tả cấm bịa tên máy. ``hosts.n_user_accounts`` (số tài khoản User từng gắn với máy, TỪ
    TRAIN) đo độ phổ biến; ``popularity_quantile = [q_lo, q_hi]`` giữ các máy có ``n_user_accounts`` nằm
    trong khoảng phân vị đó. Loại luôn các máy nạn nhân đã dùng làm Source HOẶC LogHost (máy "quen").
    """
    hosts = ctx.profiles.hosts
    if hosts.is_empty():
        raise ScenarioError("Hồ sơ không có bảng hosts — không lấy được máy mới.")
    known = set(ctx.known_sources()) | set(ctx.known_loghosts())
    pop = hosts.select("host", pl.col("n_user_accounts").fill_null(0)).filter(~pl.col("host").is_in(list(known)))
    q_lo, q_hi = float(popularity_quantile[0]), float(popularity_quantile[1])
    lo = pop["n_user_accounts"].quantile(q_lo, "nearest") if pop.height else None
    hi = pop["n_user_accounts"].quantile(q_hi, "nearest") if pop.height else None
    if lo is not None:
        pool_hosts = pop.filter((pl.col("n_user_accounts") >= lo) & (pl.col("n_user_accounts") <= hi))["host"].to_list()
    else:
        pool_hosts = []
    if len(pool_hosts) < n:          # nới khoảng phân vị nếu quá ít máy
        pool_hosts = pop["host"].to_list()
    if len(pool_hosts) < n:
        raise ScenarioError(
            f"Chỉ có {len(pool_hosts)} máy mới khả dụng cho {ctx.account.key}, cần {n}."
        )
    idx = rng.choice(len(pool_hosts), size=n, replace=False)
    return [pool_hosts[i] for i in idx]


def _one_known_loghost(ctx: VictimContext, rng: np.random.Generator) -> Optional[str]:
    hosts = ctx.known_loghosts()
    return hosts[int(rng.integers(len(hosts)))] if hosts else None


def _finalize(events: pl.DataFrame, day: int) -> pl.DataFrame:
    """Bỏ cột phụ, ép 21 cột interim, đồng thời đã được ``schedule`` kiểm tra nửa đêm."""
    return conform_to_interim(strip_helpers(events))


# =========================================================================== 1. brute-force (T1110.001)

def scenario_brute_force(ctx: VictimContext, params: Dict[str, Any], rng: np.random.Generator) -> ScenarioResult:
    """
    Dồn nhiều lần 4625 "bad password" cho MỘT tài khoản, từ MỘT Source, trên MỘT LogHost quen, LogonType 3.

    Căn cứ: brute-force một tài khoản là chuỗi thất bại dày trong thời gian ngắn từ một nguồn. Source lấy
    theo ``source_mode`` (``unseen`` = một máy thật nạn nhân chưa dùng, giống kẻ tấn công từ máy lạ;
    ``known`` = máy quen). LogHost giữ trong tập quen của nạn nhân (máy đích là máy nạn nhân hay đăng
    nhập). Nếu số lần thất bại liên tiếp đạt ngưỡng khoá ``L`` (từ hồ sơ train) thì các lần sau được
    nhân bản từ khuôn "Account locked out." thật — mô phỏng chính sách khoá của Windows.
    """
    n_fail = draw_int(params["n_failures"], rng)
    logon_type = int(params.get("logon_type", 3))
    lock_cfg = params.get("lockout", {}) or {}
    L = ctx.profiles.lockout_threshold if lock_cfg.get("L", "auto") == "auto" else int(lock_cfg["L"])

    if params.get("source_mode", "unseen") == "unseen":
        source = pick_unseen_sources(ctx, 1, rng, params.get("source_popularity_quantile", [0.0, 0.9]))[0]
    else:
        known = ctx.known_sources()
        if not known:
            raise ScenarioError(f"{ctx.account.key} không có Source quen cho source_mode=known.")
        source = known[int(rng.integers(len(known)))]
    loghost = _one_known_loghost(ctx, rng)

    sched = schedule(TimeProfile.from_config(params["time_profile"]), ctx.day, n_fail, rng)
    times = sched["Time"].to_list()

    lock_hit = bool(lock_cfg.get("enabled", True)) and L is not None and n_fail >= int(L)
    n_locked = 0
    if lock_hit and lock_cfg.get("after_lockout", "locked_out") == "locked_out":
        n_bad = int(L)                       # L lần bad password rồi các lần sau là "locked out"
        n_locked = n_fail - n_bad
    elif lock_hit and lock_cfg.get("after_lockout") == "stop":
        n_bad, times = int(L), times[: int(L)]
    else:
        n_bad = n_fail

    parts = [add_failures(ctx.pool, ctx.account, times[:n_bad], rng, source=source, loghost=loghost,
                          fail_kind="bad_pw", logon_type=logon_type,
                          prefer_own=ctx.prefer_own, allow_peer=ctx.allow_peer)]
    if n_locked:
        parts.append(add_failures(ctx.pool, ctx.account, times[n_bad:], rng, source=source, loghost=loghost,
                                  fail_kind="locked", logon_type=logon_type,
                                  prefer_own=ctx.prefer_own, allow_peer=ctx.allow_peer))
    events = _finalize(pl.concat(parts, how="vertical"), ctx.day)
    return ScenarioResult(
        events,
        params={"n_failures": n_fail, "n_bad_password": n_bad, "n_locked_out": n_locked,
                "lockout_threshold": None if L is None else int(L), "source": source, "loghost": loghost,
                "logon_type": logon_type, "source_mode": params.get("source_mode", "unseen"),
                "time_profile": TimeProfile.from_config(params["time_profile"]).to_dict()},
    )


# =========================================================================== 2. password spraying (T1110.003)

def scenario_password_spraying(
    ctx: VictimContext, params: Dict[str, Any], rng: np.random.Generator, campaign: Dict[str, Any]
) -> ScenarioResult:
    """
    MỘT Source thử NHIỀU tài khoản, mỗi tài khoản ÍT lần (< L) — dấu hiệu spray (ngược brute-force).

    Kịch bản này tiêm cho MỘT nạn nhân trong một chiến dịch; ``inject.py`` gọi nó nhiều lần với cùng
    ``campaign['source']`` và ``campaign['id']`` để mọi nạn nhân cùng chiến dịch chia sẻ một Source mới và
    một ``campaign_id``. ``attempts_per_victim`` luôn nhỏ hơn ngưỡng khoá nên không tài khoản nào bị khoá.
    LogHost là một máy quen của chính nạn nhân (``target_loghost = victim_known``).
    """
    n_try = draw_int(params["attempts_per_victim"], rng)
    logon_type = int(params.get("logon_type", 3))
    loghost = _one_known_loghost(ctx, rng) if params.get("target_loghost") == "victim_known" else None
    sched = schedule(TimeProfile.from_config(params["time_profile"]), ctx.day, n_try, rng)
    events = add_failures(ctx.pool, ctx.account, sched["Time"].to_list(), rng, source=campaign["source"],
                          loghost=loghost, fail_kind="bad_pw", logon_type=logon_type,
                          prefer_own=ctx.prefer_own, allow_peer=ctx.allow_peer)
    return ScenarioResult(
        _finalize(events, ctx.day),
        params={"attempts": n_try, "source": campaign["source"], "loghost": loghost, "logon_type": logon_type,
                "campaign_id": campaign["id"], "campaign_size": campaign["size"],
                "time_profile": TimeProfile.from_config(params["time_profile"]).to_dict()},
        campaign_id=campaign["id"],
    )


# =========================================================================== 3. hoạt động ngoài giờ (T1078)

def scenario_off_hours(ctx: VictimContext, params: Dict[str, Any], rng: np.random.Generator) -> ScenarioResult:
    """
    Sao chép một chuỗi 4624 THẬT của chính nạn nhân sang khung giờ đêm CÙNG NGÀY, giữ nguyên Source/LogHost
    và khoảng cách giữa các sự kiện (``mode: replay``).

    Căn cứ: đặc tả yêu cầu lấy sự kiện thật của chính nạn nhân và giữ interarrival. Ta rút một CHUỖI 4624
    liên tiếp từ một ngày train của nạn nhân (giữ nguyên thứ tự thời gian để offsets có nghĩa), rồi replay
    vào ``hour_window`` ban đêm. Source/LogHost đi theo khuôn nên không đổi; chỉ danh tính (vốn đã đúng)
    và Time đổi. Nạn nhân phải là người thường làm ban ngày (``off_hours_ratio`` train thấp) để sự kiện
    đêm thực sự bất thường — điều kiện này do ``inject.py`` lọc.
    """
    tpl = _own_success_chain(ctx, draw_int(params["n_events"], rng), rng)
    offsets = tpl["Time"].to_list()
    sched = schedule(TimeProfile.from_config(params["time_profile"]), ctx.day, tpl.height, rng, offsets=offsets)
    from src.injection.operations import retarget_account, set_times

    # gắn thời gian đêm vào đúng các khuôn đã rút (giữ Source/LogHost của từng khuôn)
    placed = set_times(retarget_account(tpl, ctx.account), sched["Time"].to_list())
    return ScenarioResult(
        _finalize(placed, ctx.day),
        params={"n_events": tpl.height, "source_train_day": int(tpl["_tpl_Time"][0] // 86400 + 1),
                "hour_window": list(TimeProfile.from_config(params["time_profile"]).hour_window),
                "time_profile": TimeProfile.from_config(params["time_profile"]).to_dict()},
    )


def _own_success_chain(ctx: VictimContext, n: int, rng: np.random.Generator) -> pl.DataFrame:
    """Một chuỗi ``n`` sự kiện 4624 LIÊN TIẾP theo thời gian của chính nạn nhân trong MỘT ngày train (giữ offsets).

    Chuỗi không được vắt qua nhiều ngày: offsets khi đó dài cỡ ≥ 1 ngày, không bao giờ replay vừa
    ``hour_window`` (vd. [0, 6]).
    """
    own = ctx.pool.candidates(account=ctx.account, event_id=4624, entity_type=ctx.account.entity_type)
    own = own.with_columns((pl.col("Time") // 86400).alias("_chain_day")).sort("Time")
    per_day = own.group_by("_chain_day").len().filter(pl.col("len") >= n).sort("_chain_day")
    if per_day.is_empty():
        raise ScenarioError(f"{ctx.account.key} không có ngày train nào đủ {n} sự kiện 4624 cho chuỗi.")
    day = per_day["_chain_day"][int(rng.integers(per_day.height))]
    own = own.filter(pl.col("_chain_day") == day).drop("_chain_day")
    start = int(rng.integers(0, own.height - n + 1))
    chain = own.slice(start, n)
    # thêm cột phụ _tpl_* như clone_templates để retarget nhận diện khuôn
    return chain.with_columns(
        pl.col("UserName").alias("_tpl_UserName"), pl.col("DomainName").alias("_tpl_DomainName"),
        pl.col("Time").alias("_tpl_Time"), pl.lit("own").alias("_tpl_origin"),
    )


# =========================================================================== 4. bùng nổ máy trạm mới (T1078/T1021)

def scenario_new_workstation_burst(
    ctx: VictimContext, params: Dict[str, Any], rng: np.random.Generator
) -> ScenarioResult:
    """
    Nhiều 4624 thành công, mỗi lần từ một máy nguồn nạn nhân CHƯA từng dùng (lấy từ máy có THẬT trong mạng).

    Căn cứ: chiếm đoạt tài khoản thường lộ ra ở việc đăng nhập từ loạt máy lạ. Ta chọn ``n_new_sources``
    máy mới (``pick_unseen_sources``), mỗi máy ``events_per_source`` lần 4624 nhân bản từ khuôn thật của
    nạn nhân (giữ LogonType/AuthenticationPackage của nạn nhân), chỉ thay Source và thời gian.
    """
    n_src = draw_int(params["n_new_sources"], rng)
    sources = pick_unseen_sources(ctx, n_src, rng, params.get("source_popularity_quantile", [0.0, 0.9]))
    per = [draw_int(params["events_per_source"], rng) for _ in sources]
    total = int(sum(per))
    sched = schedule(TimeProfile.from_config(params["time_profile"]), ctx.day, total, rng)
    times = sched["Time"].to_list()
    src_per_event: List[str] = [s for s, k in zip(sources, per) for _ in range(k)]
    events = add_successes(ctx.pool, ctx.account, times, rng, source=src_per_event,
                           prefer_own=ctx.prefer_own, allow_peer=ctx.allow_peer)
    return ScenarioResult(
        _finalize(events, ctx.day),
        params={"n_new_sources": n_src, "events_per_source": per, "n_events": total, "sources": sources,
                "time_profile": TimeProfile.from_config(params["time_profile"]).to_dict()},
    )


# =========================================================================== 5. ngủ đông hoạt động lại (T1078)

def scenario_dormant_wakeup(ctx: VictimContext, params: Dict[str, Any], rng: np.random.Generator) -> ScenarioResult:
    """
    Cấy MỘT ngày hoạt động thật của chính nạn nhân vào ngày ``ctx.day`` (ngày trống cuối một khoảng không
    hoạt động). Giữ nguyên giờ trong ngày (``mode: replay, hour_window: [0, 24]``).

    Căn cứ: "ngủ đông thức dậy" = tài khoản im lặng rồi đột nhiên hoạt động lại y như trước. Ta lấy toàn
    bộ sự kiện của MỘT ngày train của nạn nhân (ngày gần ``median_daily_events`` nhất để khối lượng điển
    hình), giữ nguyên offsets trong ngày rồi đặt vào ngày test. ``inject.py`` đã xác nhận ``ctx.day`` đứng
    sau một khoảng trống đủ dài và nạn nhân không hoạt động ngày đó, nên đây tạo khoá (tài khoản, ngày)
    MỚI — điều mà hợp đồng cho phép.
    """
    source_day = _dormant_source_day(ctx, params.get("source_day", "closest_to_median"))
    day_events = ctx.pool.candidates(account=ctx.account, event_id=4624, entity_type=ctx.account.entity_type).filter(
        (pl.col("Time") >= (source_day - 1) * 86400) & (pl.col("Time") < source_day * 86400)
    )
    day_events = pl.concat([
        day_events,
        ctx.pool.candidates(account=ctx.account, event_id=4625, entity_type=ctx.account.entity_type).filter(
            (pl.col("Time") >= (source_day - 1) * 86400) & (pl.col("Time") < source_day * 86400)
        ),
    ], how="vertical").sort("Time")
    if day_events.is_empty():
        raise ScenarioError(f"{ctx.account.key} không có sự kiện ngày train {source_day} để cấy.")
    offsets = [int(t % 86400) for t in day_events["Time"].to_list()]      # giữ giờ trong ngày
    sched = schedule(TimeProfile.from_config(params["time_profile"]), ctx.day, day_events.height, rng, offsets=offsets)
    from src.injection.operations import retarget_account, set_times

    tpl = day_events.with_columns(
        pl.col("UserName").alias("_tpl_UserName"), pl.col("DomainName").alias("_tpl_DomainName"),
        pl.col("Time").alias("_tpl_Time"),
    )
    placed = set_times(retarget_account(tpl, ctx.account), sched["Time"].to_list())
    return ScenarioResult(
        _finalize(placed, ctx.day),
        params={"n_events": day_events.height, "source_train_day": source_day,
                "time_profile": TimeProfile.from_config(params["time_profile"]).to_dict()},
    )


def _dormant_source_day(ctx: VictimContext, rule: str) -> int:
    dc = ctx.profiles.daily_counts.filter(
        (pl.col("DomainName") == ctx.account.domain) & (pl.col("UserName") == ctx.account.user)
    )
    if dc.is_empty():
        raise ScenarioError(f"{ctx.account.key} không có ngày train nào để làm nguồn.")
    if rule == "closest_to_median":
        med = dc["n_events"].median()
        return int(dc.with_columns((pl.col("n_events") - med).abs().alias("_d")).sort(["_d", "day"])["day"][0])
    return int(dc.sort("day")["day"][-1])      # mặc định: ngày hoạt động cuối


# =========================================================================== 6. đổi logon type (T1021)

def scenario_logon_type_switch(
    ctx: VictimContext, params: Dict[str, Any], rng: np.random.Generator
) -> ScenarioResult:
    """
    Thêm 4624 với LogonType nạn nhân CHƯA dùng, khuôn lấy từ sự kiện thật có đúng LogonType đó (để
    AuthenticationPackage/LogonTypeDescription/ProcessName khớp). Source và LogHost lấy từ tập quen.

    Căn cứ: đổi cơ chế đăng nhập (vd. sang RDP = type 10) là dấu hiệu di chuyển ngang. ``target_logon_types``
    là danh sách ưu tiên; chọn type đầu tiên mà (a) nạn nhân chưa dùng trong train và (b) có khuôn User
    thật. Nạn nhân phải có một LogonType chiếm ưu thế rõ (``min_dominant_share``) để "đổi type" thực sự lệch
    hồ sơ — điều kiện này do ``inject.py`` lọc.
    """
    shares = ctx.logon_shares()
    target = _choose_logon_type(ctx, params["target_logon_types"], shares)
    n_ev = draw_int(params["n_events"], rng)
    known_src = ctx.known_sources()
    known_host = ctx.known_loghosts()
    source = known_src[int(rng.integers(len(known_src)))] if known_src else None
    loghost = known_host[int(rng.integers(len(known_host)))] if known_host else None
    sched = schedule(TimeProfile.from_config(params["time_profile"]), ctx.day, n_ev, rng)
    events = retype_logon(ctx.pool, ctx.account, sched["Time"].to_list(), rng, logon_type=target,
                          source=source, loghost=loghost, prefer_own=ctx.prefer_own, allow_peer=ctx.allow_peer)
    dominant = max(shares, key=shares.get) if shares else None
    return ScenarioResult(
        _finalize(events, ctx.day),
        params={"target_logon_type": target, "n_events": n_ev, "source": source, "loghost": loghost,
                "dominant_logon_type": None if dominant is None else int(dominant),
                "dominant_share": None if dominant is None else float(shares[dominant]),
                "time_profile": TimeProfile.from_config(params["time_profile"]).to_dict()},
    )


def _choose_logon_type(ctx: VictimContext, targets: Sequence[int], shares: Dict[int, float]) -> int:
    """Type đầu tiên trong ``targets`` mà nạn nhân chưa dùng và có khuôn 4624 thật của tài khoản User."""
    used = set(shares)
    for t in targets:
        if int(t) in used:
            continue
        has_tpl = ctx.pool.candidates(event_id=4624, logon_type=int(t), entity_type=ctx.account.entity_type).height
        if has_tpl:
            return int(t)
    raise ScenarioError(
        f"{ctx.account.key}: không LogonType mục tiêu nào vừa chưa dùng vừa có khuôn (đã dùng {sorted(used)})."
    )


#: Bảng điều phối cho ``inject.py``. password_spraying có chữ ký khác (thêm campaign) nên gọi riêng.
SCENARIOS: Dict[str, Callable[..., ScenarioResult]] = {
    "brute_force": scenario_brute_force,
    "password_spraying": scenario_password_spraying,
    "off_hours": scenario_off_hours,
    "new_workstation_burst": scenario_new_workstation_burst,
    "dormant_wakeup": scenario_dormant_wakeup,
    "logon_type_switch": scenario_logon_type_switch,
}
