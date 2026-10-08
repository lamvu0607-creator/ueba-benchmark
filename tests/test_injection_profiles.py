"""
Kiểm thử hồ sơ train của bộ tiêm (src/injection/profiles.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import pytest

from src.features.extractor import features_from_events
from src.injection.layout import day_window, interim_day_path
from src.injection.profiles import (
    ProfileConfig,
    TrainProfiles,
    build_train_profiles,
    estimate_lockout_threshold,
    inactive_gaps,
    lockout_episodes,
    profiles_from_day_events,
)

BAD = "Unknown user name or bad password."
LOCKED = "Account locked out."
OTHER = "An Error occured during Logon."

SCHEMA = {
    "Time": pl.Int64, "EventID": pl.Int32, "UserName": pl.String, "LogHost": pl.String,
    "LogonType": pl.Int32, "AuthenticationPackage": pl.String, "Source": pl.String,
    "DomainName": pl.String, "ProcessName": pl.String, "FailureReason": pl.String,
}


def ev(day: int, hour: int, user: str = "User1", *, eid: int = 4624, host: str = "Comp1",
       src: Optional[str] = "Comp9", lt: int = 3, reason: Optional[str] = None,
       dom: str = "domain001", sec: int = 0) -> Dict[str, Any]:
    start, _ = day_window(day)
    return {
        "Time": start + hour * 3600 + sec, "EventID": eid, "UserName": user, "LogHost": host,
        "LogonType": lt, "AuthenticationPackage": "Kerberos", "Source": src, "DomainName": dom,
        "ProcessName": None, "FailureReason": reason if eid == 4625 else None,
    }


def frame(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA)


def by_day(rows: List[Dict[str, Any]]) -> Dict[int, pl.DataFrame]:
    df = frame(rows).with_columns((pl.col("Time") // 86400 + 1).alias("_d"))
    return {int(k[0]): g.drop("_d") for k, g in df.partition_by("_d", as_dict=True).items()}


CFG = ProfileConfig(start_day=1, train_end_day=5)


# --------------------------------------------------------------------------- hồ sơ tài khoản

def test_known_sources_and_loghosts_skip_invalid_values():
    rows = [
        ev(1, 9, src="Comp9", host="Comp1"),
        ev(1, 10, src=" Comp8 ", host="Comp2"),
        ev(2, 9, src="Unknown", host="Comp1"),
        ev(2, 10, src=None, host=None),
        ev(3, 9, src="", host="  "),
    ]
    p = profiles_from_day_events(by_day(rows), CFG)
    assert p.known_sources("domain001", "User1") == {"Comp9", "Comp8"}
    assert p.known_loghosts("domain001", "User1") == {"Comp1", "Comp2"}
    acc = p.account("domain001", "User1")
    assert (acc["n_known_sources"], acc["n_known_loghosts"]) == (2, 2)
    comp1 = p.account_loghosts.filter(pl.col("LogHost") == "Comp1").row(0, named=True)
    assert (comp1["n_events"], comp1["n_days"]) == (2, 2)


def test_off_hours_ratio_matches_extractor_definition():
    # biên: 7h và 18h là ngoài giờ, 8h và 17h trong giờ (giống extractor: hour <= 7 | hour >= 18)
    rows = [ev(2, h) for h in (0, 7, 8, 12, 17, 18, 23)] + [ev(2, 3, user="User2")]
    events = frame(rows)
    p = profiles_from_day_events({2: events}, CFG)
    ref = features_from_events(events, 2).select("DomainName", "UserName", "off_hours_ratio")
    got = p.accounts.select("DomainName", "UserName", "off_hours_ratio")
    assert got.sort("UserName").equals(ref.sort("UserName"))
    assert p.account("domain001", "User1")["off_hours_ratio"] == pytest.approx(4 / 7)


def test_logon_type_shares_sum_to_one():
    rows = [ev(1, 9, lt=3), ev(1, 10, lt=3), ev(2, 9, lt=2), ev(3, 9, lt=10)]
    p = profiles_from_day_events(by_day(rows), CFG)
    shares = p.logon_type_shares("domain001", "User1")
    assert shares == pytest.approx({3: 0.5, 2: 0.25, 10: 0.25})
    assert p.account_logon_types["share"].sum() == pytest.approx(1.0)


def test_median_daily_events_uses_active_days_only():
    rows = [ev(1, 9)] + [ev(3, 9, sec=i) for i in range(3)] + [ev(5, 9, sec=i) for i in range(10)]
    acc = profiles_from_day_events(by_day(rows), CFG).account("domain001", "User1")
    assert acc["median_daily_events"] == 3
    assert (acc["n_active_days"], acc["first_day"], acc["last_day"], acc["n_events"]) == (3, 1, 5, 14)


def test_success_failure_counts_and_entity_type():
    rows = [ev(1, 9), ev(1, 9, eid=4625, reason=BAD), ev(1, 9, user="Comp5$")]
    p = profiles_from_day_events(by_day(rows), CFG)
    acc = p.account("domain001", "User1")
    assert (acc["n_success"], acc["n_failure"], acc["entity_type"]) == (1, 1, "User")
    assert p.account("domain001", "Comp5$")["entity_type"] == "Machine"


# --------------------------------------------------------------------------- khoảng không hoạt động

def test_inactive_gaps_kinds_and_missing_days():
    daily = pl.DataFrame(
        {"DomainName": ["d"] * 3, "UserName": ["u"] * 3, "day": [3, 4, 8], "n_events": [1, 1, 1]},
        schema_overrides={"day": pl.Int32},
    )
    # ngày 6 không có file -> không tính là ngày không hoạt động
    cal = [1, 2, 3, 4, 5, 7, 8, 9, 10]
    gaps = inactive_gaps(daily, cal).select("gap_start_day", "gap_end_day", "n_days", "kind").rows()
    assert gaps == [(1, 2, 2, "leading"), (5, 7, 2, "internal"), (9, 10, 2, "trailing")]


def test_account_without_gaps_has_none():
    rows = [ev(d, 9) for d in range(1, 6)]
    p = profiles_from_day_events(by_day(rows), CFG)
    assert p.gaps("domain001", "User1").is_empty()
    acc = p.account("domain001", "User1")
    assert acc["n_inactive_gaps"] == 0 and acc["max_internal_gap_days"] is None


def test_gap_summary_in_accounts():
    rows = [ev(1, 9), ev(4, 9)]
    acc = profiles_from_day_events(by_day(rows), CFG).account("domain001", "User1")
    assert acc["n_inactive_gaps"] == 2 and acc["max_internal_gap_days"] == 2  # 2-3 internal, 5 trailing


# --------------------------------------------------------------------------- hồ sơ máy

def test_hosts_count_accounts_per_machine():
    rows = [
        ev(1, 9, user="User1", host="CompA", src="CompX"),
        ev(1, 9, user="User2", host="CompA", src="CompA"),
        ev(2, 9, user="Comp7$", host="CompX", src=None),
        ev(2, 9, user="User1", host="CompA", src="CompX"),
    ]
    hosts = profiles_from_day_events(by_day(rows), CFG).hosts
    a = hosts.filter(pl.col("host") == "CompA").row(0, named=True)
    assert (a["n_accounts"], a["n_accounts_as_loghost"], a["n_accounts_as_source"], a["n_user_accounts"]) == (2, 2, 1, 2)
    assert (a["n_events_as_loghost"], a["n_days"]) == (3, 2)
    x = hosts.filter(pl.col("host") == "CompX").row(0, named=True)
    assert (x["n_accounts"], x["n_accounts_as_loghost"], x["n_accounts_as_source"], x["n_user_accounts"]) == (2, 1, 1, 1)
    assert hosts["n_accounts"].to_list() == sorted(hosts["n_accounts"].to_list(), reverse=True)


# --------------------------------------------------------------------------- ngưỡng khoá L

def _failures(seq: List[tuple], user: str = "User1") -> pl.DataFrame:
    """seq: (Time, kind)."""
    return pl.DataFrame(
        {
            "DomainName": ["domain001"] * len(seq), "UserName": [user] * len(seq),
            "Time": [t for t, _ in seq], "_seq": list(range(len(seq))), "kind": [k for _, k in seq],
            "day": [1] * len(seq),
        },
        schema_overrides={"day": pl.Int32},
    )


def test_lockout_streak_counts_consecutive_bad_passwords():
    seq = [(10, "bad_pw"), (20, "other"), (30, "bad_pw"), (31, "bad_pw"), (32, "bad_pw"),
           (33, "locked"), (34, "locked"), (40, "locked")]
    ep = lockout_episodes(_failures(seq))
    assert ep.height == 1  # 34, 40 cùng một lần khoá
    assert (ep["streak"][0], ep["n_bad_pw_window"][0], ep["Time"][0]) == (3, 4, 33)


def test_lockout_streak_breaks_on_time_gap_and_new_episode_after_gap():
    seq = [(0, "bad_pw"), (5000, "bad_pw"), (5001, "bad_pw"), (5002, "locked"),
           (9000, "bad_pw"), (9001, "locked")]
    ep = lockout_episodes(_failures(seq), window_s=1800, episode_gap_s=1800)
    assert ep["Time"].to_list() == [5002, 9001]
    assert ep["streak"].to_list() == [2, 1]
    assert ep["n_bad_pw_window"].to_list() == [2, 1]


def test_lockout_same_second_uses_file_order():
    # cùng giây: bad_pw đứng SAU lần khoá trong file không được tính vào chuỗi
    seq = [(100, "bad_pw"), (100, "locked"), (100, "bad_pw")]
    ep = lockout_episodes(_failures(seq))
    assert ep["streak"].to_list() == [1] and ep["n_bad_pw_window"].to_list() == [1]


def test_estimate_lockout_threshold_mode_ignores_zero_streaks():
    ep = pl.DataFrame({
        "DomainName": ["d"] * 6, "UserName": [f"u{i}" for i in range(6)],
        "streak": [0, 0, 0, 5, 5, 3],
    })
    res = estimate_lockout_threshold(ep)
    assert res["L"] == 5 and res["n_onsets_streak0"] == 3 and res["streak_hist"] == {"3": 1, "5": 2}
    assert res["L_support"] == pytest.approx(2 / 3)
    assert estimate_lockout_threshold(ep.head(3))["L"] is None


def test_lockout_threshold_from_day_events():
    rows = []
    for i, user in enumerate(["User1", "User2", "User3"]):
        n_bad = 2 if user == "User3" else 4
        rows += [ev(1, 10 + i, user, eid=4625, reason=BAD, sec=s) for s in range(n_bad)]
        rows += [ev(1, 10 + i, user, eid=4625, reason=LOCKED, sec=10)]
    p = profiles_from_day_events(by_day(rows), CFG)
    assert p.lockout_threshold == 4
    assert p.network["lockout"]["n_onsets"] == 3
    assert p.network["lockout"]["by_entity_type"] == {"User": 4}


# --------------------------------------------------------------------------- chỉ train, lưu/nạp, đọc đĩa

def test_rejects_days_outside_train():
    with pytest.raises(ValueError, match="ngoài train"):
        profiles_from_day_events(by_day([ev(1, 9), ev(6, 9)]), CFG)


def test_save_load_roundtrip(temp_artifact_dir: Path):
    rows = [ev(1, 9), ev(3, 22, src="CompZ"), ev(3, 9, eid=4625, reason=BAD), ev(3, 9, eid=4625, reason=LOCKED, sec=1)]
    p = profiles_from_day_events(by_day(rows), CFG)
    q = TrainProfiles.load(p.save(temp_artifact_dir / "prof"))
    assert q.network == p.network
    for name in ("accounts", "account_sources", "account_gaps", "hosts", "lockout_episodes"):
        assert getattr(q, name).equals(getattr(p, name))


def _write_interim(root: Path, day: int, rows: List[Dict[str, Any]]) -> None:
    df = frame(rows)
    for eid in (4624, 4625):
        path = interim_day_path(root, eid, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = df.filter(pl.col("EventID") == eid)
        if eid == 4624:
            part = part.drop("FailureReason")
        part.write_parquet(path)


def test_build_train_profiles_reads_interim_train_days_only(temp_artifact_dir: Path):
    root = temp_artifact_dir / "interim"
    _write_interim(root, 1, [ev(1, 9, dom="Domain001"), ev(1, 9, user="null"), ev(1, 9, eid=4625, reason=BAD)])
    _write_interim(root, 2, [ev(2, 9, dom=" ")])
    _write_interim(root, 3, [ev(3, 9, user="UserTest")])  # ngày test
    p = build_train_profiles(root, ProfileConfig(start_day=1, train_end_day=2))
    assert p.network["train_days"] == [1, 2]
    # DomainName chuẩn hoá như extractor, "null" bị loại, ngày 3 không bao giờ được đọc
    assert set(p.accounts.select("DomainName", "UserName").rows()) == {("domain001", "User1"), ("Unknown", "User1")}
    acc = p.account("domain001", "User1")
    assert (acc["n_events"], acc["n_failure"], acc["first_day"]) == (2, 1, 1)
    assert p.network["days_outside_window"] == {}


def test_profile_config_from_system_config():
    cfg = ProfileConfig.from_system_config(
        {"evaluation": {"split_day": 40}, "features": {"off_hours_start": 19, "off_hours_end": 6}},
        lockout_window_s=600,
    )
    assert (cfg.train_end_day, cfg.off_hours_start, cfg.off_hours_end, cfg.lockout_window_s) == (40, 19, 6, 600)
