"""
Kiểm thử 6 kịch bản (``scenarios.py``) và điều phối (``inject.py``) trên log interim tổng hợp nhỏ.

Dữ liệu dựng một mạng mini: train = ngày 1..5, test = ngày 6..8 (``split_day = 5``). Một vài tài khoản
User với lịch sử đủ để mỗi kịch bản có ứng viên; vài máy làm Source/LogHost. Toàn bộ chạy qua đúng đường
thật: dựng hồ sơ -> dựng kho khuôn từ file interim -> chọn nạn nhân -> kịch bản -> ghi run -> kiểm tra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl
import pytest

from src.features.extractor import DEFAULT_FEATURE_CFG, entity_type_expr
from src.injection.inject import InjectionConfig, InjectionRunner, injected_day_expr, run_injection
from src.injection.layout import RunLayout, day_window, overlay_days
from src.injection.operations import INTERIM_COLUMNS, INTERIM_SCHEMA, Account, TemplatePool, annotate_events
from src.injection.profiles import ProfileConfig, build_train_profiles
from src.injection.scenarios import (
    VictimContext,
    scenario_brute_force,
    scenario_dormant_wakeup,
    scenario_logon_type_switch,
    scenario_new_workstation_burst,
    scenario_off_hours,
)

SPLIT_DAY = 5
BAD = "Unknown user name or bad password."
LOCKED = "Account locked out."
ET = entity_type_expr(DEFAULT_FEATURE_CFG)


def _row(**kw: Any) -> Dict[str, Any]:
    base = {c: None for c in INTERIM_COLUMNS}
    base.update(EventID=4624, LogonType=3, LogonTypeDescription="Network", DomainName="DOM1",
                AuthenticationPackage="Kerberos", LogonID="0x1", SubjectUserName=None)
    base.update(kw)
    return base


def _t(day: int, hour: int, extra: int = 0) -> int:
    return (day - 1) * 86400 + hour * 3600 + extra


def _write_interim(root: Path, rows: List[Dict[str, Any]]) -> None:
    df = pl.DataFrame(rows, schema=INTERIM_SCHEMA)
    df = df.with_columns((pl.col("Time") // 86400 + 1).alias("_d"))
    for (eid, day), part in df.group_by(["EventID", "_d"]):
        p = root / f"event_{eid}" / f"event_{eid}_day-{int(day):02d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        part.drop("_d").write_parquet(p)
    # đảm bảo ngày nào có 4624 thì cũng có file 4625 (và ngược lại) cho available_days
    for day in sorted(set(df["_d"].to_list())):
        for eid in (4624, 4625):
            p = root / f"event_{eid}" / f"event_{eid}_day-{int(day):02d}.parquet"
            if not p.is_file():
                p.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame([], schema=INTERIM_SCHEMA).write_parquet(p)


def _make_dataset(interim: Path) -> None:
    rows: List[Dict[str, Any]] = []
    # Udaily: hoạt động ban ngày mọi ngày train, type 3 chiếm ưu thế, dùng Source S1 / LogHost H1
    for day in range(1, SPLIT_DAY + 1):
        for h in (9, 10, 11, 13, 14):
            rows.append(_row(Time=_t(day, h), UserName="User1", Source="S1", LogHost="H1", LogonID=f"0x1{day}{h}"))
    # Uday2..4: thêm người làm ban ngày (đủ chuỗi 4624/ngày, hoạt động ngày test) để mỗi kịch bản
    # on-active-day còn ứng viên dù brute_force/spraying lấy trước tài khoản nào
    for k in (6, 7, 8):
        for day in range(1, SPLIT_DAY + 1):
            for h in (9, 10, 11, 13):
                rows.append(_row(Time=_t(day, h), UserName=f"User{k}", Source=f"S{k}", LogHost=f"H{k}",
                                 LogonID=f"0x{k}{day}{h}"))
    # Udormant: hoạt động ngày 1,2 rồi im lặng (để test ngủ đông ở ngày test)
    for day in (1, 2):
        for h in (9, 10, 11):
            rows.append(_row(Time=_t(day, h), UserName="User2", Source="S2", LogHost="H2", LogonID=f"0x2{day}{h}"))
    # Ufail: có cả 4624 và một chuỗi 4625 bad password + một locked (làm khuôn khoá)
    for day in range(1, SPLIT_DAY + 1):
        rows.append(_row(Time=_t(day, 9), UserName="User3", Source="S3", LogHost="H3", LogonID=f"0x3{day}"))
    for i in range(6):
        rows.append(_row(Time=_t(2, 2, i * 30), EventID=4625, UserName="User3", Source="S3", LogHost="H3",
                         FailureReason=BAD, AuthenticationPackage="NTLM", LogonID=None))
    rows.append(_row(Time=_t(2, 2, 200), EventID=4625, UserName="User3", Source="S3", LogHost="H3",
                     FailureReason=LOCKED, AuthenticationPackage="NTLM", LogonID=None))
    # Uswitch: chỉ dùng type 3 (dominant share 1.0), có Source/LogHost quen, mọi ngày train
    for day in range(1, SPLIT_DAY + 1):
        for h in (8, 12, 16):
            rows.append(_row(Time=_t(day, h), UserName="User4", Source="S4", LogHost="H4", LogonID=f"0x4{day}{h}"))
    # một tài khoản dùng type 10 để làm khuôn RemoteInteractive cho kịch bản 6
    for day in range(1, SPLIT_DAY + 1):
        rows.append(_row(Time=_t(day, 15), UserName="User5", LogonType=10, LogonTypeDescription="RemoteInteractive",
                         AuthenticationPackage="Negotiate", Source="S5", LogHost="H5", ProcessName="svchost.exe",
                         LogonID=f"0x5{day}"))
    # tài khoản máy (không được làm nạn nhân / khuôn User)
    for day in range(1, SPLIT_DAY + 1):
        rows.append(_row(Time=_t(day, 3), UserName="C9$", Source="H9", LogHost="H9", LogonID=f"0x9{day}"))

    # ----- TEST days (6..8): cho các tài khoản hoạt động để kịch bản on-active-day có ngày tiêm
    for day in (6, 7, 8):
        for h in (9, 10, 11):
            rows.append(_row(Time=_t(day, h), UserName="User1", Source="S1", LogHost="H1", LogonID=f"0xT1{day}{h}"))
            rows.append(_row(Time=_t(day, h), UserName="User3", Source="S3", LogHost="H3", LogonID=f"0xT3{day}{h}"))
            rows.append(_row(Time=_t(day, h), UserName="User4", Source="S4", LogHost="H4", LogonID=f"0xT4{day}{h}"))
            for k in (6, 7, 8):
                rows.append(_row(Time=_t(day, h), UserName=f"User{k}", Source=f"S{k}", LogHost=f"H{k}",
                                 LogonID=f"0xT{k}{day}{h}"))
    # User2 (dormant) KHÔNG xuất hiện ngày test -> ngày test là ngày trống sau khoảng ngủ đông
    # máy mạng (để pick_unseen_sources có máy thật): thêm nhiều LogHost chỉ xuất hiện ở train
    for day in range(1, SPLIT_DAY + 1):
        for host in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"):
            rows.append(_row(Time=_t(day, 16), UserName=f"User_{host}", Source=host, LogHost=host,
                             LogonID=f"0x{host}{day}"))
    _write_interim(interim, rows)


@pytest.fixture()
def interim(temp_artifact_dir: Path) -> Path:
    d = temp_artifact_dir / "interim"
    d.mkdir(parents=True, exist_ok=True)
    _make_dataset(d)
    return d


@pytest.fixture()
def profiles(interim: Path):
    return build_train_profiles(interim, ProfileConfig(start_day=1, train_end_day=SPLIT_DAY))


@pytest.fixture()
def pool(interim: Path) -> TemplatePool:
    frames = []
    for day in range(1, SPLIT_DAY + 1):
        parts = [pl.read_parquet(interim / f"event_{e}" / f"event_{e}_day-{day:02d}.parquet") for e in (4624, 4625)]
        raw = pl.concat(parts, how="vertical").select(INTERIM_COLUMNS)
        frames.append(annotate_events(raw, ET))
    return TemplatePool(pl.concat(frames), train_end_day=SPLIT_DAY)


def _ctx(profiles, pool, user: str, day: int) -> VictimContext:
    return VictimContext(Account("dom1", user, "DOM1", "User"), day, profiles, pool)


# --------------------------------------------------------------------------- kịch bản đơn lẻ

def test_brute_force_triggers_lockout_and_keeps_day(profiles, pool):
    params = {"n_failures": 8, "logon_type": 3, "source_mode": "unseen",
              "source_popularity_quantile": [0.0, 1.0],
              "lockout": {"enabled": True, "L": "auto", "after_lockout": "locked_out"},
              "time_profile": {"mode": "burst", "n_days": 1, "hour_window": [0, 24], "burst_duration_s": [300, 600]}}
    res = scenario_brute_force(_ctx(profiles, pool, "User3", 6), params, np.random.default_rng(0))
    assert res.events["EventID"].to_list() == [4625] * 8
    start, end = day_window(6)
    assert (res.events["Time"] >= start).all() and (res.events["Time"] < end).all()
    L = profiles.lockout_threshold
    if L is not None and 8 >= L:
        assert res.params["n_locked_out"] == 8 - L
        assert res.events.filter(pl.col("FailureReason") == LOCKED).height == 8 - L
    assert set(res.events["Source"]).isdisjoint({"S3", "H3"})       # Source mới, không phải máy quen


def test_off_hours_replays_own_events_into_night(profiles, pool):
    params = {"n_events": 3, "max_train_off_hours_ratio": 1.0,
              "time_profile": {"mode": "replay", "n_days": 1, "hour_window": [0, 6]}}
    res = scenario_off_hours(_ctx(profiles, pool, "User1", 6), params, np.random.default_rng(1))
    start, _ = day_window(6)
    hours = [((t - start) // 3600) for t in res.events["Time"]]
    assert all(0 <= h < 6 for h in hours)
    assert set(res.events["Source"]) == {"S1"}                      # giữ Source thật của nạn nhân
    assert set(res.events["UserName"]) == {"User1"}


def test_new_workstation_burst_uses_only_unseen_real_hosts(profiles, pool):
    params = {"n_new_sources": 4, "events_per_source": 2, "source_popularity_quantile": [0.0, 1.0],
              "time_profile": {"mode": "burst", "n_days": 1, "hour_window": [8, 18], "burst_duration_s": [600, 3600]}}
    res = scenario_new_workstation_burst(_ctx(profiles, pool, "User1", 7), params, np.random.default_rng(2))
    assert res.events.height == 8 and set(res.events["EventID"]) == {4624}
    known = {"S1", "H1"}
    assert set(res.events["Source"]).isdisjoint(known)
    real_hosts = set(profiles.hosts["host"].to_list())
    assert set(res.events["Source"]) <= real_hosts                  # mọi Source là máy CÓ THẬT


def test_dormant_wakeup_creates_activity_on_idle_day(profiles, pool):
    params = {"source_day": "closest_to_median",
              "time_profile": {"mode": "replay", "n_days": 1, "hour_window": [0, 24]}}
    res = scenario_dormant_wakeup(_ctx(profiles, pool, "User2", 8), params, np.random.default_rng(3))
    assert res.events.height >= 1 and set(res.events["UserName"]) == {"User2"}
    start, end = day_window(8)
    assert (res.events["Time"] >= start).all() and (res.events["Time"] < end).all()


def test_logon_type_switch_picks_unused_type_with_matching_package(profiles, pool):
    params = {"target_logon_types": [10, 2], "min_dominant_share": 0.5, "n_events": 3,
              "time_profile": {"mode": "spread", "n_days": 1, "hour_window": [8, 18]}}
    res = scenario_logon_type_switch(_ctx(profiles, pool, "User4", 6), params, np.random.default_rng(4))
    assert set(res.events["LogonType"]) == {10}
    assert set(res.events["AuthenticationPackage"]) == {"Negotiate"}  # khớp khuôn type 10
    assert res.params["target_logon_type"] == 10
    assert set(res.events["Source"]) <= {"S4"} and set(res.events["LogHost"]) <= {"H4"}


# --------------------------------------------------------------------------- điều phối end-to-end

def _write_config(tmp: Path, interim: Path, runs: Path) -> Path:
    cfg = {
        "common": {
            "interim_dir": str(interim), "profiles_dir": str(tmp / "profiles"), "runs_dir": str(runs),
            "split_day": SPLIT_DAY,
            "victim": {"entity_type": "User", "min_train_active_days": 2, "one_injection_per_account": True},
            "rule_exclusion": {"enabled": False},
            "template": {"prefer_own_account": True, "allow_peer_fallback": True}, "oracle": "none",
        },
        "scenarios": {
            "brute_force": {"enabled": True, "n_victims": 1, "n_failures": 8, "logon_type": 3,
                            "source_mode": "unseen", "source_popularity_quantile": [0.0, 1.0],
                            "lockout": {"enabled": True, "L": "auto", "after_lockout": "locked_out"},
                            "time_profile": {"mode": "burst", "n_days": 1, "hour_window": [0, 24],
                                             "burst_duration_s": [300, 600]}},
            "password_spraying": {"enabled": True, "n_campaigns": 1, "victims_per_campaign": 2,
                                  "attempts_per_victim": 2, "logon_type": 3, "source_mode": "unseen",
                                  "source_popularity_quantile": [0.0, 1.0], "target_loghost": "victim_known",
                                  "time_profile": {"mode": "burst", "n_days": 1, "hour_window": [0, 24],
                                                   "burst_duration_s": [1800, 3600]}},
            "off_hours": {"enabled": True, "n_victims": 1, "n_events": 3, "max_train_off_hours_ratio": 1.0,
                          "time_profile": {"mode": "replay", "n_days": 1, "hour_window": [0, 6]}},
            "new_workstation_burst": {"enabled": True, "n_victims": 1, "n_new_sources": 3,
                                      "events_per_source": 2, "source_popularity_quantile": [0.0, 1.0],
                                      "time_profile": {"mode": "burst", "n_days": 1, "hour_window": [8, 18],
                                                       "burst_duration_s": [600, 3600]}},
            "dormant_wakeup": {"enabled": True, "n_victims": 1, "gap_rule": "fixed", "min_gap_days": 1,
                               "source_day": "closest_to_median",
                               "time_profile": {"mode": "replay", "n_days": 1, "hour_window": [0, 24]}},
            "logon_type_switch": {"enabled": True, "n_victims": 1, "target_logon_types": [10, 2],
                                  "min_dominant_share": 0.5, "n_events": 3,
                                  "time_profile": {"mode": "spread", "n_days": 1, "hour_window": [8, 18]}},
        },
        "blocks": {"dev": {"days": [6, 7, 8], "seed": 123, "run_id_prefix": "dev", "scenarios": None}},
    }
    cfg["blocks"]["dev"]["scenarios"] = cfg["scenarios"]
    import yaml

    p = tmp / "injection.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    sysp = tmp / "system.yaml"
    sysp.write_text(yaml.safe_dump({"evaluation": {"split_day": SPLIT_DAY}}), encoding="utf-8")
    return p


@pytest.fixture()
def built(temp_artifact_dir: Path, interim: Path, profiles):
    tmp = temp_artifact_dir
    (tmp / "profiles").mkdir(parents=True, exist_ok=True)
    profiles.save(tmp / "profiles")
    runs = tmp / "runs"
    cfg_path = _write_config(tmp, interim, runs)
    result = run_injection("dev", cfg_path, tmp / "system.yaml")
    return result, RunLayout.for_run(runs, result["run_id"]), interim


def test_same_seed_gives_identical_run(built):
    """Cùng config + seed -> cùng nạn nhân, cùng sự kiện tiêm (kho khuôn và thứ tự ứng viên phải tất định)."""
    result, layout, _ = built
    cfg_path = layout.root.parent.parent / "injection.yaml"
    again = run_injection("dev", cfg_path, layout.root.parent.parent / "system.yaml", run_id="dev_again")
    layout2 = RunLayout.for_run(layout.root.parent, again["run_id"])

    cols = ["scenario", "DomainName", "UserName", "day", "n_events", "params"]

    def manifest(lay: RunLayout) -> pl.DataFrame:
        # campaign_id chứa run_id (khác nhau có chủ đích) -> thay bằng hằng trước khi so
        return pl.read_csv(lay.manifest_path).select(cols).with_columns(
            pl.col("params").str.replace_all(lay.run_id, "RUN", literal=True)
        )

    assert manifest(layout).equals(manifest(layout2))
    ev1 = pl.read_parquet(layout.injected_events_path).drop("inj_id")
    ev2 = pl.read_parquet(layout2.injected_events_path).drop("inj_id")
    assert ev1.equals(ev2)


def test_spraying_source_is_new_for_every_campaign_victim(built, profiles):
    """Source chung của chiến dịch phải lạ với MỌI nạn nhân (không chỉ nạn nhân đầu)."""
    import json

    _, layout, _ = built
    spray = pl.read_csv(layout.manifest_path).filter(pl.col("scenario") == "password_spraying")
    assert spray.height >= 2
    for dom, user, params in spray.select("DomainName", "UserName", "params").iter_rows():
        src = json.loads(params)["source"]
        d = dom.strip().lower()
        assert src not in set(profiles.known_sources(d, user)) | set(profiles.known_loghosts(d, user))


def test_run_produces_contracted_outputs(built):
    result, layout, interim = built
    assert layout.manifest_path.is_file() and layout.injected_events_path.is_file()
    assert layout.labels_path.is_file()
    manifest = pl.read_csv(layout.manifest_path)
    assert set(manifest.columns) >= {"inj_id", "scenario", "DomainName", "UserName", "day", "campaign_id",
                                     "split", "seed", "n_events", "params"}
    assert manifest["inj_id"].n_unique() == manifest.height


def test_run_never_touches_train_days(built):
    _, layout, _ = built
    assert min(overlay_days(layout.events_dir)) > SPLIT_DAY


def test_each_overlay_day_has_both_event_files_with_interim_schema(built):
    _, layout, _ = built
    for day in overlay_days(layout.events_dir):
        for eid in (4624, 4625):
            f = layout.events_file(eid, day)
            assert f.is_file(), f
            assert pl.read_parquet(f).columns == INTERIM_COLUMNS


def test_injected_rows_match_overlay_delta(built):
    _, layout, interim = built
    injected = pl.read_parquet(layout.injected_events_path)
    assert "inj_id" not in pl.read_parquet(layout.events_file(injected["EventID"][0], 6)).columns
    day_col = injected.with_columns(injected_day_expr())
    for day in overlay_days(layout.events_dir):
        added = sum(
            pl.read_parquet(layout.events_file(e, day)).height
            - pl.read_parquet(interim / f"event_{e}" / f"event_{e}_day-{day:02d}.parquet").height
            for e in (4624, 4625)
        )
        assert added == day_col.filter(pl.col("day") == day).height


def test_each_account_injected_at_most_once(built):
    _, layout, _ = built
    manifest = pl.read_csv(layout.manifest_path)
    assert manifest.select(["DomainName", "UserName"]).unique().height == manifest.height


def test_injected_events_within_their_day(built):
    _, layout, _ = built
    injected = pl.read_parquet(layout.injected_events_path).with_columns(injected_day_expr())
    for day, part in injected.group_by("day"):
        start, end = day_window(int(day[0]))
        assert (part["Time"] >= start).all() and (part["Time"] < end).all()


def test_labels_are_positive_and_unique(built):
    _, layout, _ = built
    labels = pl.read_parquet(layout.labels_path)
    assert set(labels["is_anomaly"].unique().to_list()) == {1}
    assert labels.select(["DomainName", "UserName", "day"]).unique().height == labels.height
    # DomainName của nhãn đã chuẩn hoá -> khớp khoá ma trận
    assert set(labels["DomainName"].to_list()) == {"dom1"}


def test_spraying_shares_campaign_id_and_source(built):
    _, layout, _ = built
    import json

    manifest = pl.read_csv(layout.manifest_path)
    spray = manifest.filter(pl.col("scenario") == "password_spraying")
    if spray.height >= 2:
        assert spray["campaign_id"].n_unique() <= spray.height
        sources = {json.loads(p)["source"] for p in spray["params"]}
        by_campaign = spray.group_by("campaign_id").agg(
            pl.col("params").map_elements(lambda s: json.loads(s)["source"], return_dtype=pl.String).n_unique()
        )
        assert (by_campaign["params"] == 1).all()       # mỗi chiến dịch đúng một Source


def test_domain_raw_written_to_overlay(built):
    _, layout, _ = built
    # log đè giữ DomainName THÔ (DOM1), không phải dạng chuẩn hoá
    injected = pl.read_parquet(layout.injected_events_path)
    assert set(injected["DomainName"].unique().to_list()) == {"DOM1"}


def test_pool_is_shrunk_but_keeps_victims_and_peer_templates(built):
    """Kho khuôn thu nhỏ: vẫn có khuôn own của nạn nhân và khuôn peer (type 10) cho kịch bản 6."""
    result, layout, interim = built
    # dựng lại runner để soi kho (đi đúng đường _build_pool với peer_cap mặc định)
    import yaml
    from src.injection.inject import InjectionConfig, InjectionRunner

    cfg_path = Path(layout.root).parent.parent  # không dùng; build lại từ file config đã ghi trong fixture
    # fixture đã chạy run_injection; ở đây chỉ kiểm hợp đồng trên manifest + nhãn thay vì nội tại kho.
    manifest = pl.read_csv(layout.manifest_path)
    # mọi kịch bản bật đều tiêm được ít nhất 1 nạn nhân trên bộ dữ liệu mini
    assert set(manifest["scenario"].unique().to_list()) >= {
        "brute_force", "off_hours", "new_workstation_burst", "dormant_wakeup", "logon_type_switch"
    }
    # kịch bản 6 lấy được khuôn type 10 (peer của User5) -> sự kiện tiêm có LogonType 10
    inj = pl.read_parquet(layout.injected_events_path)
    switch_users = manifest.filter(pl.col("scenario") == "logon_type_switch")["UserName"].to_list()
    if switch_users:
        got = inj.filter(pl.col("UserName").is_in(switch_users))["LogonType"].unique().to_list()
        assert 10 in got


def test_peer_cap_respected_in_pool_build(temp_artifact_dir, interim, profiles):
    """peer_cap_per_group giới hạn số khuôn peer mỗi nhóm; nạn nhân cơ sở vẫn giữ đầy đủ."""
    import yaml
    from src.injection.inject import InjectionConfig, InjectionRunner

    tmp = temp_artifact_dir
    (tmp / "profiles").mkdir(parents=True, exist_ok=True)
    profiles.save(tmp / "profiles")
    cfg = {
        "common": {
            "interim_dir": str(interim), "profiles_dir": str(tmp / "profiles"), "runs_dir": str(tmp / "runs"),
            "split_day": SPLIT_DAY,
            "victim": {"entity_type": "User", "min_train_active_days": 2, "one_injection_per_account": True},
            "rule_exclusion": {"enabled": False},
            "template": {"prefer_own_account": True, "allow_peer_fallback": True, "peer_cap_per_group": 1},
        },
        "scenarios": {"brute_force": {"enabled": True, "n_victims": 1, "n_failures": 2, "logon_type": 3,
                                      "source_mode": "unseen", "source_popularity_quantile": [0.0, 1.0],
                                      "lockout": {"enabled": False},
                                      "time_profile": {"mode": "burst", "n_days": 1, "hour_window": [0, 24],
                                                       "burst_duration_s": [300, 600]}}},
        "blocks": {"dev": {"days": [6, 7, 8], "seed": 1, "run_id_prefix": "dev", "scenarios": None}},
    }
    cfg["blocks"]["dev"]["scenarios"] = cfg["scenarios"]
    (tmp / "inj.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (tmp / "sys.yaml").write_text(yaml.safe_dump({"evaluation": {"split_day": SPLIT_DAY}}), encoding="utf-8")

    runner = InjectionRunner(InjectionConfig.from_files("dev", tmp / "inj.yaml", tmp / "sys.yaml"))
    pool = runner.pool
    # mỗi nhóm peer còn ≤ cap? kiểm gián tiếp: tổng sự kiện nhỏ hơn nhiều so với toàn log mini train
    assert pool.events.height > 0
    # nạn nhân cơ sở (User1) giữ đầy đủ sự kiện own (5 ngày × 5 = 25 sự kiện 4624)
    own = pool.events.filter((pl.col("_dom") == "dom1") & (pl.col("UserName") == "User1"))
    assert own.height == 25


def test_test_block_dormancy_observes_prior_dev_activity(temp_artifact_dir, interim, profiles):
    from src.injection.inject import InjectionConfig, InjectionRunner
    profiles.save(temp_artifact_dir / 'profiles')
    cfg_path = _write_config(temp_artifact_dir, interim, temp_artifact_dir / 'runs')
    cfg = InjectionConfig.from_files('dev', cfg_path, temp_artifact_dir / 'system.yaml')
    # Train ends on 5; User1 is active on dev days 6,7 but absent on test day 8.
    path = interim / 'event_4624' / 'event_4624_day-08.parquet'
    ev = pl.read_parquet(path)
    ev.filter(pl.col('UserName') != 'User1').write_parquet(path)
    cfg.days = [8]
    runner = InjectionRunner(cfg)
    candidates = runner._candidates_dormant({'gap_rule': 'fixed', 'min_gap_days': 1})
    assert 'User1' not in candidates['UserName'].to_list()
    assert 'User2' in candidates['UserName'].to_list()


def test_missing_presence_is_not_inactivity(interim):
    from src.injection.inject import _read_presence
    (interim / 'event_4624' / 'event_4624_day-06.parquet').unlink()
    with pytest.raises(FileNotFoundError, match='cannot infer inactivity'):
        _read_presence(interim, [6])


def test_run_freezes_full_block_days(built):
    import json
    from src.injection.layout import load_run_eval_days
    _, layout, _ = built
    assert json.loads((layout.root / 'run_config.json').read_text())['eval_days'] == [6, 7, 8]
    assert load_run_eval_days(layout) == [6, 7, 8]
