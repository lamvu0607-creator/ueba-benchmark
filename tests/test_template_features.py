"""
Kiểm thử cho schema v4.0: Combinatorial Template, engine tính tự động và bộ tiêm bất thường.
"""

import math

import polars as pl
import pytest

from src.evaluation.injection import InjectionPlan, inject_day
from src.features.extractor import DEFAULT_FEATURE_CFG
from src.features.template_engine import compute_day, compute_history
from src.features.templates import Candidate, build_catalog, candidate_by_name

DAY = 86400


# ---------------------------------------------------------------------------
# Template + vòng 1
# ---------------------------------------------------------------------------
def test_catalog_is_deterministic_and_round2_is_well_formed():
    cat = build_catalog()
    assert len(cat) == 14 * 4 * 8 * 7           # đủ tích Đề-các 4 trục
    round2 = [c for c in cat if c.status == "round2"]
    assert len(round2) == 78
    names = [c.name for c in round2]
    assert len(names) == len(set(names)), "tên sinh tự động phải duy nhất"
    for c in round2:
        assert c.scenarios, f"{c.name}: vòng 2 chỉ nhận biến gắn kịch bản 5.2"
        assert c.cost <= 3, f"{c.name}: vượt ngưỡng diễn giải"
        assert c.description and "{" not in c.description
    # các tổ hợp trùng core v3.0 không bao giờ lọt vào vòng 2
    for key in [("count", "all", "event", "1d"), ("novelty", "all", "Source", "7d")]:
        c = next(x for x in cat if (x.measure, x.obj, x.entity, x.window) == key)
        assert c.status == "dropped_r1"


def test_every_dropped_candidate_has_a_reason():
    for c in build_catalog():
        if c.status != "round2":
            assert c.reason, f"{c} bị loại nhưng không có lý do"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def _events(rows):
    """rows: (day, user, time_in_day, eventid, source, host, logontype)"""
    return pl.DataFrame(
        {
            "Time": [(d - 1) * DAY + t for d, _, t, *_ in rows],
            "EventID": [r[3] for r in rows],
            "UserName": [r[1] for r in rows],
            "LogHost": [r[5] for r in rows],
            "LogonType": [r[6] for r in rows],
            "AuthenticationPackage": ["Kerberos"] * len(rows),
            "Source": [r[4] for r in rows],
            "DomainName": ["dom"] * len(rows),
            "ProcessName": [None] * len(rows),
            "FailureReason": [None if r[3] == 4624 else "Unknown user name or bad password." for r in rows],
        },
        schema_overrides={"EventID": pl.Int32, "LogonType": pl.Int32, "ProcessName": pl.String},
    ).with_columns(pl.col("Time").cast(pl.Int64))


ROWS = [
    (1, "UserA", 36000, 4624, "s1", "h1", 3), (1, "UserA", 36100, 4624, "s2", "h1", 3),
    (2, "UserA", 36000, 4624, "s1", "h1", 3),
    (3, "UserA", 36000, 4624, "s1", "h1", 3), (3, "UserA", 36010, 4624, "s3", "h2", 3),
    (3, "UserA", 36020, 4624, "s3", "h2", 3),
    # thất bại: s9 tấn công 2 tài khoản trong cùng 15 phút của ngày 3
    (3, "UserB", 40000, 4625, "s9", "h1", 3), (3, "UserB", 40100, 4625, "s9", "h1", 3),
    (3, "UserA", 40200, 4625, "s9", "h1", 3),
]

SPECS = [
    Candidate("novelty", "success", "Source", "7d"),
    Candidate("novelty_share", "success", "Source", "7d"),
    Candidate("jaccard", "all", "Source", "7d"),
    Candidate("dist_shift", "all", "Source", "7d"),
    Candidate("count", "fail", "event", "15m"),
    Candidate("fanout", "fail", "Source", "1d"),
    Candidate("active_days", "all", "event", "7d"),
    Candidate("delta_mean", "all", "Source", "7d"),
]


def _matrix(rows, specs=SPECS, max_day=None):
    ev = _events(rows)
    outs = [compute_day(ev.filter(pl.col("Time") // DAY == d - 1), d, specs, DEFAULT_FEATURE_CFG)
            for d in sorted({r[0] for r in rows}) if max_day is None or d <= max_day]
    intraday = pl.concat([o.intraday for o in outs])
    prof = {k: pl.concat([o.profiles[k] for o in outs]) for k in outs[0].profiles}
    return compute_history(intraday, prof, specs)


def test_template_engine_values():
    m = _matrix(ROWS)
    a3 = m.filter((pl.col("UserName") == "UserA") & (pl.col("day") == 3)).row(0, named=True)
    # thành công hôm nay {s1, s3}; lịch sử 7 ngày H = {s1, s2}
    assert a3["novelty_success_source_7d"] == pytest.approx(math.log1p(1))
    assert a3["novelty_share_success_source_7d"] == pytest.approx(2 / 3)
    assert a3["jaccard_source_7d"] == pytest.approx(1 / 4)        # {s1,s3,s9} vs {s1,s2}
    # hôm nay (mọi sự kiện): s1 ×1, s3 ×2, s9 ×1 ⇒ p = 1/4, 1/2, 1/4; q = s1 2/3, s2 1/3
    assert a3["dist_shift_source_7d"] == pytest.approx(0.5 * (abs(1 / 4 - 2 / 3) + 1 / 2 + 1 / 4 + 1 / 3))
    assert a3["fanout_fail_source"] == pytest.approx(math.log1p(2))       # s9 chạm UserA & UserB
    b3 = m.filter((pl.col("UserName") == "UserB") & (pl.col("day") == 3)).row(0, named=True)
    assert b3["count_fail_15m"] == pytest.approx(math.log1p(2))
    assert b3["jaccard_source_7d"] is None                                 # chưa có lịch sử
    # 3 ngày < cửa sổ 7 ngày ⇒ warm-up = NULL, không bịa 0
    assert a3["active_days_7d"] is None and a3["delta_mean_distinct_source_7d"] is None


def test_template_engine_is_causal():
    """Thêm ngày tương lai không được làm đổi giá trị của các ngày trước đó."""
    full = _matrix(ROWS).filter(pl.col("day") <= 2).sort(["UserName", "day"])
    cut = _matrix(ROWS, max_day=2).sort(["UserName", "day"])
    names = [c.name for c in SPECS]
    assert full.select(names).equals(cut.select(names))


# ---------------------------------------------------------------------------
# Tiêm bất thường
# ---------------------------------------------------------------------------
def _plan(**victims):
    plan = InjectionPlan(target_day=3, dormant_from=2)
    for s, users in victims.items():
        plan.victims[s] = pl.DataFrame({"DomainName": ["dom"] * len(users), "UserName": users})
    return plan


def test_injection_scenarios_change_the_event_stream():
    ev = _events([r for r in ROWS if r[0] == 3]).filter(pl.col("EventID") == 4624)
    plan = _plan(brute_force=["UserA"])
    out = inject_day(ev, 3, plan)
    added = out.filter(pl.col("EventID") == 4625)
    assert 100 <= added.height <= 300
    assert added["Source"].n_unique() == 1 and added["Source"][0].startswith("INJ-BF")
    assert (added["Time"].max() - added["Time"].min()) < 20 * 60
    assert out.schema == ev.schema

    out = inject_day(ev, 3, _plan(off_hours=["UserA"]))
    hours = ((out["Time"] % DAY) // 3600).to_list()
    assert all(1 <= h <= 6 for h in hours)

    out = inject_day(ev, 3, _plan(logon_type_switch=["UserA"]))
    assert out["LogonType"].unique().to_list() == [5]


def test_dormant_injection_silences_then_amplifies():
    plan = _plan(dormant_wakeup=["UserA"])
    day2 = _events([r for r in ROWS if r[0] == 2])
    assert inject_day(day2, 2, plan).height == 0                    # ngủ đông: xoá hoạt động
    day3 = _events([r for r in ROWS if r[0] == 3 and r[1] == "UserA"])
    assert inject_day(day3, 3, plan).height == 5 * day3.height      # thức dậy: khối lượng ×5


def test_candidate_lookup_by_name():
    c = candidate_by_name("novelty_success_source_7d")
    assert (c.measure, c.obj, c.entity, c.window) == ("novelty", "success", "Source", "7d")
