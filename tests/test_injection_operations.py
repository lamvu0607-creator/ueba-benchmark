"""
Kiểm thử ``src/injection/operations.py`` bằng log interim tổng hợp (21 cột, đúng dtype).

Ba nhóm bắt buộc theo đặc tả:
  1. hàm nhân bản khuôn (giữ nguyên mọi trường, chỉ lấy từ train, đúng loại tài khoản);
  2. "không vượt nửa đêm" cho mọi mode / cửa sổ giờ / số ngày;
  3. mỗi thao tác chỉ đổi đúng trường khai báo trong ``OP_FIELDS``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import polars as pl
import pytest

from src.features.extractor import DEFAULT_FEATURE_CFG, entity_type_expr
from src.injection.layout import SECONDS_PER_DAY, InjectionLayoutError, day_window
from src.injection.operations import (
    INTERIM_COLUMNS,
    INTERIM_SCHEMA,
    OP_FIELDS,
    Account,
    ScheduleError,
    TemplateNotFoundError,
    TemplatePool,
    TimeProfile,
    add_failures,
    add_successes,
    allowed_segments,
    annotate_events,
    changed_fields,
    clone_templates,
    conform_to_interim,
    draw_int,
    retarget_account,
    retype_logon,
    schedule,
    set_loghost,
    set_source,
    set_times,
    shift_time,
    strip_helpers,
)

BAD = "Unknown user name or bad password."
LOCKED = "Account locked out."
TRAIN_END = 42
ET = entity_type_expr(DEFAULT_FEATURE_CFG)


def _ev(**kw: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {c: None for c in INTERIM_COLUMNS}
    base.update(
        Time=3 * SECONDS_PER_DAY + 9 * 3600, EventID=4624, LogHost="C100", LogonType=3,
        LogonTypeDescription="Network", UserName="User111", DomainName="DOM1", LogonID="0x1a2b",
        SubjectUserName="User111", SubjectDomainName="DOM1", SubjectLogonID="0x3e7",
        Source="C200", AuthenticationPackage="Kerberos", ProcessName="-", ProcessID="0x0",
    )
    base.update(kw)
    return base


def _frame(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=INTERIM_SCHEMA)


def _pool_rows() -> List[Dict[str, Any]]:
    rows = [
        # nạn nhân User111: 4624 type 3 và một 4625 bad password của chính nó
        _ev(LogonID="0xa1"), _ev(LogonID="0xa2", Time=5 * SECONDS_PER_DAY + 10 * 3600),
        _ev(EventID=4625, FailureReason=BAD, LogonID="0x0", AuthenticationPackage="NTLM"),
        # tài khoản User khác: type 10 (RemoteInteractive) + 4625 locked out
        _ev(UserName="User222", SubjectUserName="User222", LogonType=10, LogonTypeDescription="RemoteInteractive",
            AuthenticationPackage="Negotiate", ProcessName="svchost.exe", LogonID="0xb1"),
        _ev(UserName="User222", SubjectUserName="User222", EventID=4625, FailureReason=LOCKED,
            AuthenticationPackage="NTLM", LogonID="0x0"),
        _ev(UserName="User222", SubjectUserName="User222", EventID=4625, FailureReason=BAD,
            AuthenticationPackage="NTLM", LogonID="0x0"),
        # tài khoản máy có type 10 — KHÔNG được dùng làm khuôn cho User
        _ev(UserName="C999$", SubjectUserName="C999$", LogonType=10, LogonTypeDescription="RemoteInteractive",
            AuthenticationPackage="Kerberos", ProcessName="machine.exe"),
        # Subject KHÔNG phản chiếu danh tính (vd. tài khoản máy ánh xạ)
        _ev(UserName="User333", SubjectUserName="C555$", SubjectDomainName="DOM1", LogonType=2,
            LogonTypeDescription="Interactive", AuthenticationPackage="Negotiate", LogonID="0xc1"),
    ]
    return rows


@pytest.fixture()
def pool() -> TemplatePool:
    return TemplatePool(annotate_events(_frame(_pool_rows()), ET), train_end_day=TRAIN_END)


def _assert_interim_schema(df: pl.DataFrame) -> None:
    assert df.columns == INTERIM_COLUMNS
    for c, t in INTERIM_SCHEMA.items():
        assert df.schema[c] == t, c


VICTIM = Account(domain="dom1", user="User111", domain_raw="DOM1", entity_type="User")
OTHER = Account(domain="dom2", user="User444", domain_raw="Dom2", entity_type="User")


# --------------------------------------------------------------------------- schema

def test_interim_schema_matches_raw_to_interim():
    pa = pytest.importorskip("pyarrow")
    from src.data.raw_to_interim import INTERIM_SCHEMA as ARROW_SCHEMA

    assert [f.name for f in ARROW_SCHEMA] == INTERIM_COLUMNS
    mapping = {pa.int64(): pl.Int64, pa.int32(): pl.Int32, pa.string(): pl.String}
    for f in ARROW_SCHEMA:
        assert INTERIM_SCHEMA[f.name] == mapping[f.type], f.name


def test_conform_rejects_missing_columns():
    with pytest.raises(InjectionLayoutError):
        conform_to_interim(pl.DataFrame({"Time": [1]}))


# --------------------------------------------------------------------------- 1. nhân bản khuôn

def test_clone_keeps_every_field_of_a_real_template(pool):
    tpl = pool.candidates(event_id=4624, entity_type="User")
    out = clone_templates(tpl, 4, np.random.default_rng(0))
    assert out.height == 4
    real = set(tpl.select(INTERIM_COLUMNS).rows())
    # mỗi dòng nhân bản trùng NGUYÊN VẸN một dòng khuôn thật (không trường nào bị bịa)
    assert set(out.select(INTERIM_COLUMNS).rows()) <= real
    _assert_interim_schema(strip_helpers(out))
    assert out["_tpl_UserName"].to_list() == out["UserName"].to_list()


def test_clone_without_replacement_when_enough_and_with_when_not(pool):
    tpl = pool.candidates(event_id=4624, entity_type="User")
    few = clone_templates(tpl, tpl.height, np.random.default_rng(1))
    assert few["LogonID"].n_unique() == tpl["LogonID"].n_unique()
    many = clone_templates(tpl, 50, np.random.default_rng(1))
    assert many.height == 50


def test_clone_is_deterministic_for_a_seed(pool):
    tpl = pool.candidates(event_id=4624)
    a = clone_templates(tpl, 10, np.random.default_rng(7))
    b = clone_templates(tpl, 10, np.random.default_rng(7))
    assert a.equals(b)


def test_clone_empty_templates_raises():
    with pytest.raises(TemplateNotFoundError):
        clone_templates(_frame([]), 1, np.random.default_rng(0))


def test_pool_refuses_events_outside_train():
    rows = _pool_rows() + [_ev(Time=TRAIN_END * SECONDS_PER_DAY)]  # 00:00 ngày 43
    with pytest.raises(InjectionLayoutError, match="ngoài train"):
        TemplatePool(annotate_events(_frame(rows), ET), train_end_day=TRAIN_END)


def test_pick_prefers_own_account_then_peer_same_entity_type(pool):
    rng = np.random.default_rng(0)
    own = pool.pick(3, rng, account=VICTIM, event_id=4625, fail_kind="bad_pw")
    assert set(own["_tpl_origin"]) == {"own"} and set(own["_tpl_UserName"]) == {"User111"}
    # nạn nhân không có khuôn locked -> lấy của tài khoản User khác
    peer = pool.pick(2, rng, account=VICTIM, event_id=4625, fail_kind="locked")
    assert set(peer["_tpl_origin"]) == {"peer"} and set(peer["FailureReason"]) == {LOCKED}
    # type 10: chỉ User222 (User) — không bao giờ lấy khuôn của tài khoản máy C999$
    t10 = pool.pick(20, rng, account=VICTIM, event_id=4624, logon_type=10)
    assert set(t10["_tpl_UserName"]) == {"User222"}


def test_pick_without_any_template_raises(pool):
    with pytest.raises(TemplateNotFoundError):
        pool.pick(1, np.random.default_rng(0), account=VICTIM, event_id=4624, logon_type=11)
    with pytest.raises(TemplateNotFoundError):
        pool.pick(1, np.random.default_rng(0), account=VICTIM, event_id=4625, fail_kind="locked", allow_peer=False)


def test_annotate_normalizes_domain_and_classifies_failures(pool):
    ev = pool.events
    assert set(ev["_dom"]) == {"dom1"}
    kinds = dict(zip(ev["FailureReason"].to_list(), ev["_fail_kind"].to_list()))
    assert kinds[BAD] == "bad_pw" and kinds[LOCKED] == "locked" and kinds[None] is None
    assert set(ev.filter(pl.col("UserName") == "C999$")["_entity_type"]) == {"Machine"}


# --------------------------------------------------------------------------- 2. không vượt nửa đêm

@pytest.mark.parametrize("mode", ["burst", "spread", "replay"])
@pytest.mark.parametrize("hw", [(0, 24), (0, 6), (22, 6), (18, 0), (23, 1), (8, 18)])
@pytest.mark.parametrize("n_days", [1, 3])
def test_schedule_never_crosses_midnight(mode, hw, n_days):
    profile = TimeProfile(mode=mode, n_days=n_days, hour_window=hw, burst_duration_s=(60, 50_000))
    segs = allowed_segments(hw)
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 60))
        offs = np.sort(rng.integers(0, 3000, size=n)) if mode == "replay" else None
        out = schedule(profile, 52, n, rng, offsets=offs)
        assert out.height == n
        assert sorted(set(out["day"])) == list(range(52, 52 + n_days))[: min(n, n_days)]
        for d, t in zip(out["day"], out["Time"]):
            start, end = day_window(d)
            assert start <= t < end
            assert any(a <= t - start < b for a, b in segs)


def test_schedule_replay_keeps_gaps_on_one_day():
    offs = np.array([0, 5, 65, 3600, 3601])
    out = schedule(TimeProfile(mode="replay", hour_window=(1, 5)), 50, 5, np.random.default_rng(3), offsets=offs)
    assert np.diff(out["Time"].to_numpy()).tolist() == np.diff(offs).tolist()


def test_schedule_replay_too_long_for_window_raises():
    offs = np.array([0, 5 * 3600])
    with pytest.raises(ScheduleError):
        schedule(TimeProfile(mode="replay", hour_window=(0, 2)), 50, 2, np.random.default_rng(0), offsets=offs)


def test_schedule_burst_longer_than_window_is_clipped_inside_window():
    out = schedule(TimeProfile(mode="burst", hour_window=(2, 3), burst_duration_s=(10_000, 20_000)),
                   45, 30, np.random.default_rng(0))
    start, _ = day_window(45)
    assert ((out["Time"] - start) // 3600 == 2).all()


def test_set_times_outside_day_is_caught_by_check():
    from src.injection.operations import assert_within_days

    start, end = day_window(52)
    with pytest.raises(InjectionLayoutError):
        assert_within_days(pl.DataFrame({"day": [52], "Time": [end]}), [52])
    with pytest.raises(InjectionLayoutError):
        assert_within_days(pl.DataFrame({"Time": [start - 1]}), [52], day_col=None)
    assert_within_days(pl.DataFrame({"Time": [start, end - 1]}), [52], day_col=None)


def test_time_profile_validation_and_config():
    with pytest.raises(ValueError):
        TimeProfile(mode="x")
    with pytest.raises(ValueError):
        TimeProfile(hour_window=(5, 5))
    with pytest.raises(ValueError):
        TimeProfile(n_days=0)
    p = TimeProfile.from_config({"mode": "burst", "hour_window": [22, 6], "burst_duration_s": 900})
    assert p.n_days == 1 and p.burst_duration_s == (900, 900) and p.hour_window == (22, 6)


# --------------------------------------------------------------------------- 3. chỉ đổi đúng trường

def _base(pool: TemplatePool) -> pl.DataFrame:
    return clone_templates(pool.candidates(event_id=4624, entity_type="User"), 5, np.random.default_rng(0))


@pytest.mark.parametrize(
    "op, call",
    [
        ("set_times", lambda df: set_times(df, list(range(df.height)))),
        ("shift_time", lambda df: shift_time(df, 7200)),
        ("set_source", lambda df: set_source(df, "C777")),
        ("set_loghost", lambda df: set_loghost(df, ["C1", "C2", "C3", "C4", "C5"])),
        ("retarget_account", lambda df: retarget_account(df, OTHER)),
    ],
)
def test_basic_ops_change_only_declared_fields(pool, op, call):
    before = _base(pool)
    after = call(before)
    diff = changed_fields(before, after)
    assert diff, f"{op} không đổi gì"
    assert diff <= OP_FIELDS[op], f"{op} đổi ngoài khai báo: {diff - OP_FIELDS[op]}"
    _assert_interim_schema(strip_helpers(after))


def test_retarget_only_rewrites_subject_that_mirrors_template(pool):
    tpl = pool.candidates(event_id=4624, logon_type=2)          # User333, Subject = C555$
    out = retarget_account(clone_templates(tpl, 1, np.random.default_rng(0)), OTHER)
    assert out["UserName"][0] == "User444" and out["DomainName"][0] == "Dom2"
    assert out["SubjectUserName"][0] == "C555$"                   # không phản chiếu -> giữ nguyên cả cặp
    assert out["SubjectDomainName"][0] == "DOM1"
    tpl2 = pool.candidates(event_id=4624, logon_type=10)         # User222, Subject = User222
    out2 = retarget_account(clone_templates(tpl2, 1, np.random.default_rng(0)), OTHER)
    assert out2["SubjectUserName"][0] == "User444"
    assert out2["SubjectDomainName"][0] == "Dom2"


@pytest.mark.parametrize("fail_kind", ["bad_pw", "locked"])
def test_add_failures_changes_only_declared_fields_vs_its_template(pool, fail_kind):
    rng = np.random.default_rng(5)
    start, _ = day_window(52)
    times = [start + 100, start + 105, start + 110]
    out = add_failures(pool, VICTIM, times, rng, source="C888", loghost="C100", fail_kind=fail_kind)
    assert out["EventID"].to_list() == [4625] * 3
    assert set(out["UserName"]) == {"User111"} and set(out["DomainName"]) == {"DOM1"}
    assert out["Time"].to_list() == times
    # đối chiếu từng dòng với ĐÚNG khuôn của nó (cùng seed -> cùng khuôn)
    tpl = pool.pick(3, np.random.default_rng(5), account=VICTIM, event_id=4625, logon_type=3, fail_kind=fail_kind)
    assert changed_fields(tpl, out) <= OP_FIELDS["add_failures"]
    assert out["FailureReason"].to_list() == tpl["FailureReason"].to_list()
    assert out["AuthenticationPackage"].to_list() == tpl["AuthenticationPackage"].to_list()


def test_add_successes_keeps_template_source_when_not_given(pool):
    start, _ = day_window(50)
    out = add_successes(pool, VICTIM, [start + 1, start + 2], np.random.default_rng(0))
    tpl = pool.pick(2, np.random.default_rng(0), account=VICTIM, event_id=4624)
    assert out["Source"].to_list() == tpl["Source"].to_list()
    assert changed_fields(tpl, out) <= {"Time"}  # khuôn của chính nạn nhân: chỉ Time đổi


def test_retype_logon_takes_matching_package_from_real_template(pool):
    start, _ = day_window(55)
    out = retype_logon(pool, VICTIM, [start + 50], np.random.default_rng(0), logon_type=10,
                       source="C200", loghost="C100")
    row = out.row(0, named=True)
    assert row["LogonType"] == 10
    assert row["AuthenticationPackage"] == "Negotiate"            # của khuôn type 10 (User222), không phải Kerberos
    assert row["LogonTypeDescription"] == "RemoteInteractive"
    assert row["ProcessName"] == "svchost.exe"
    tpl = pool.pick(1, np.random.default_rng(0), account=VICTIM, event_id=4624, logon_type=10)
    assert changed_fields(tpl, out) <= OP_FIELDS["retype_logon"]


# --------------------------------------------------------------------------- tiện ích

def test_draw_int():
    rng = np.random.default_rng(0)
    assert draw_int(5, rng) == 5
    vals = {draw_int([2, 4], rng) for _ in range(200)}
    assert vals == {2, 3, 4}
    with pytest.raises(ValueError):
        draw_int([4, 2], rng)
