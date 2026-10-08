"""Hạ tầng của một run log đè: bố cục run, nhãn từ manifest, hook oracle độ khó."""

from src.injection.difficulty import DifficultyOracle, NullOracle, get_oracle, score_difficulty
from src.injection.labels import labels_from_manifest, write_run_labels
from src.injection.layout import (
    RunLayout,
    assert_events_within_day,
    day_of_time,
    day_window,
    events_outside_day,
    overlay_days,
)
from src.injection.profiles import ProfileConfig, TrainProfiles, build_train_profiles

__all__ = [
    "ProfileConfig",
    "TrainProfiles",
    "build_train_profiles",
    "DifficultyOracle",
    "NullOracle",
    "get_oracle",
    "score_difficulty",
    "labels_from_manifest",
    "write_run_labels",
    "RunLayout",
    "assert_events_within_day",
    "day_of_time",
    "day_window",
    "events_outside_day",
    "overlay_days",
]
