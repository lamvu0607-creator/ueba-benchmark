"""
Kiểm thử hạ tầng run log đè (src/injection: layout, labels, difficulty) và chọn nguồn ngày của extractor.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.evaluation.labeled_eval import load_eval_labels
from src.features.extractor import resolve_day_source
from src.injection.difficulty import NullOracle, get_oracle, score_difficulty, validate_difficulty
from src.injection.labels import labels_from_manifest, write_run_labels
from src.injection.layout import (
    InjectionLayoutError,
    RunLayout,
    SECONDS_PER_DAY,
    assert_events_within_day,
    day_of_time,
    day_window,
    events_outside_day,
    interim_day_path,
    overlay_days,
)


# --------------------------------------------------------------------------- layout: "không vượt nửa đêm"

def test_day_window_matches_interim_convention():
    assert day_window(1) == (0, SECONDS_PER_DAY)
    assert day_window(43) == (42 * SECONDS_PER_DAY, 43 * SECONDS_PER_DAY)
    with pytest.raises(ValueError):
        day_window(0)


def test_day_of_time_is_inverse_of_day_window():
    day = 43
    start, end = day_window(day)
    df = pl.DataFrame({"Time": [start, start + 1, end - 1, end, start - 1]})
    assert df.select(day_of_time())["Time"].to_list() == [43, 43, 43, 44, 42]


def test_events_outside_day_boundaries():
    day = 43
    start, end = day_window(day)
    df = pl.DataFrame({"Time": [start, end - 1, end, start - 1, None]}, schema={"Time": pl.Int64})
    bad = events_outside_day(df, day)
    # đầu cửa sổ tính vào ngày, cuối cửa sổ (00:00 hôm sau) thì không; thiếu Time cũng là vi phạm
    assert sorted(bad["Time"].to_list(), key=lambda x: (x is None, x)) == [start - 1, end, None]


def test_assert_events_within_day_passes_on_full_day():
    start, end = day_window(52)
    assert_events_within_day(pl.DataFrame({"Time": [start, start + 3600, end - 1]}), 52)


@pytest.mark.parametrize("offset", [-1, SECONDS_PER_DAY])
def test_assert_events_within_day_rejects_midnight_crossing(offset):
    start, _ = day_window(52)
    df = pl.DataFrame({"Time": [start + 10, start + offset]})
    with pytest.raises(InjectionLayoutError, match="ngoài ngày 52"):
        assert_events_within_day(df, 52)


def test_run_layout_paths_and_run_id_validation(temp_artifact_dir: Path):
    layout = RunLayout.for_run(temp_artifact_dir, "run_001")
    assert layout.run_id == "run_001"
    assert layout.events_file(4625, 7) == layout.events_dir / "event_4625" / "event_4625_day-07.parquet"
    assert RunLayout.from_events_dir(layout.events_dir).root == layout.root.resolve()
    for bad in ["", "a/b", "a\\b", "c:x"]:
        with pytest.raises(ValueError):
            RunLayout.for_run(temp_artifact_dir, bad)


def test_run_layout_results_live_under_experiments(temp_artifact_dir: Path):
    """Kết quả benchmark của run ghi vào <experiments>/<run_id>/, không vào data/injection_runs."""
    exp = temp_artifact_dir / "experiments" / "injection_runs"
    layout = RunLayout.for_run(temp_artifact_dir / "runs", "r9", experiments_dir=exp)
    assert layout.results_dir == exp / "r9" / "results"
    assert layout.models_dir == exp / "r9" / "models"
    assert RunLayout.from_events_dir(layout.events_dir, exp).results_dir == exp / "r9" / "results"
    layout.root.mkdir(parents=True)
    pl.DataFrame({"x": [1]}).write_parquet(layout.labels_path)
    (layout.root / "run_config.json").write_text("{}", encoding="utf-8")
    copied = layout.export_run_metadata()
    assert sorted(p.name for p in copied) == ["labels.parquet", "run_config.json"]
    assert (layout.results_dir.parent / "labels.parquet").is_file()  # plots tìm nhãn ở <results>/../


def _touch_day(root: Path, eid: int, day: int) -> Path:
    p = interim_day_path(root, eid, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"Time": [0]}).write_parquet(p)
    return p


def test_overlay_days_lists_days_of_either_event(temp_artifact_dir: Path):
    _touch_day(temp_artifact_dir, 4624, 45)
    _touch_day(temp_artifact_dir, 4625, 45)
    _touch_day(temp_artifact_dir, 4625, 52)
    assert overlay_days(temp_artifact_dir) == [45, 52]
    assert overlay_days(temp_artifact_dir / "missing") == []


# --------------------------------------------------------------------------- extractor: nguồn của từng ngày

def test_resolve_day_source_uses_overlay_only_for_its_days(temp_artifact_dir: Path):
    base, over = temp_artifact_dir / "interim", temp_artifact_dir / "overlay"
    _touch_day(over, 4624, 45)
    _touch_day(over, 4625, 45)
    assert resolve_day_source(45, base, over, [45]) == over
    assert resolve_day_source(44, base, over, [45]) == base
    assert resolve_day_source(45, base, None, []) == base


def test_resolve_day_source_requires_both_event_files(temp_artifact_dir: Path):
    over = temp_artifact_dir / "overlay"
    _touch_day(over, 4624, 46)
    with pytest.raises(FileNotFoundError, match="4624 và 4625"):
        resolve_day_source(46, temp_artifact_dir, over, [46])


# --------------------------------------------------------------------------- nhãn từ manifest

def _manifest() -> pl.DataFrame:
    return pl.DataFrame({
        "inj_id": ["dev-0001", "dev-0002", "dev-0003"],
        "scenario": ["s1", "s2", "s2"],
        "DomainName": ["Domain001", " domain001 ", None],
        "UserName": ["User1", "User2", "User3"],
        "day": [44, 45, 45],
        "campaign_id": [None, "c1", "c1"],
    })


def test_labels_from_manifest_normalizes_keys_and_marks_positive():
    labels = labels_from_manifest(_manifest())
    assert labels.columns == ["DomainName", "UserName", "day", "is_anomaly", "eval_exclude", "scenario", "campaign_id"]
    domains = dict(zip(labels["UserName"].to_list(), labels["DomainName"].to_list()))
    assert domains == {"User1": "domain001", "User2": "domain001", "User3": "Unknown"}
    assert labels["is_anomaly"].to_list() == [1, 1, 1]
    assert not labels["eval_exclude"].any()
    assert labels.filter(pl.col("campaign_id") == "c1").height == 2


def test_labels_from_manifest_rejects_duplicate_keys():
    dup = _manifest().with_columns(pl.lit("domain001").alias("DomainName"), pl.lit("User1").alias("UserName"), pl.lit(44).alias("day"))
    with pytest.raises(ValueError, match="nhiều lần"):
        labels_from_manifest(dup)


def test_labels_from_manifest_rejects_missing_columns():
    with pytest.raises(ValueError, match="thiếu cột"):
        labels_from_manifest(_manifest().drop("day"))


def test_labels_from_manifest_expands_multi_day_column():
    m = _manifest().head(1).with_columns(pl.lit("44;45").alias("days"))
    labels = labels_from_manifest(m)
    assert labels["day"].to_list() == [44, 45]


def test_write_run_labels_roundtrips_through_load_eval_labels(temp_artifact_dir: Path):
    layout = RunLayout.for_run(temp_artifact_dir, "r1")
    layout.ensure_dirs()
    _manifest().write_csv(layout.manifest_path)
    path = write_run_labels(layout)
    loaded = load_eval_labels(path)
    assert loaded.height == 3 and loaded["is_anomaly"].sum() == 3


# --------------------------------------------------------------------------- hook oracle độ khó

def test_null_oracle_returns_one_null_per_label():
    labels = labels_from_manifest(_manifest())
    out = validate_difficulty(NullOracle().score(pl.DataFrame(), labels), labels)
    assert out.height == labels.height and out["difficulty"].null_count() == labels.height


def test_get_oracle_unknown_name_raises():
    assert get_oracle(None).name == "none"
    with pytest.raises(KeyError):
        get_oracle("does-not-exist")


def test_validate_difficulty_rejects_misaligned_output():
    labels = labels_from_manifest(_manifest())
    with pytest.raises(ValueError, match="lệch nhãn"):
        validate_difficulty(NullOracle().score(pl.DataFrame(), labels.head(2)), labels)


def test_score_difficulty_writes_file(temp_artifact_dir: Path):
    layout = RunLayout.for_run(temp_artifact_dir, "r2")
    layout.ensure_dirs([layout.processed_dir])
    _manifest().write_csv(layout.manifest_path)
    write_run_labels(layout)
    labels = pl.read_parquet(layout.labels_path)
    labels.select(["DomainName", "UserName", "day"]).write_parquet(layout.processed_dir / "feature_matrix_processed.parquet")
    out = score_difficulty(layout)
    assert pl.read_parquet(out).height == labels.height
    assert out.with_suffix(".json").is_file()
