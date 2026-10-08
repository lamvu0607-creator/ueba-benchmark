"""
Bố cục một lần tiêm (run) và quy ước "ngày" của tầng log interim.

**Quy ước ngày (đã kiểm chứng trên dữ liệu thật, ngày 1/42/43):** ``day`` của pipeline KHÔNG được
tính từ timestamp mà là số thứ tự file ``event_XXXX_day-NN.parquet``. Ở tầng interim, mọi sự kiện
của file ngày ``d`` có ``Time ∈ [(d−1)·86400, d·86400)`` và extractor lấy giờ bằng
``(Time % 86400) // 3600``. Bộ tiêm ghi vào tầng interim nên một sự kiện "nằm trong ngày dự định"
khi và chỉ khi ``Time`` thuộc đúng cửa sổ đó — không vượt nửa đêm ở cả hai phía.

(Tầng cleaned trừ 3600s cho ngày ≥ 42 để chỉnh DST ⇒ cửa sổ bị lệch 1 giờ. Vì vậy chạy features
trên log đã tiêm bắt buộc đọc tầng interim — xem ``build_account_day_matrix(events_dir=...)``.)

Một run nằm trong ``<injection_runs_dir>/<run_id>/``::

    events_injected/event_4624/event_4624_day-NN.parquet   log đã tiêm, CÙNG tên/schema file interim,
    events_injected/event_4625/event_4625_day-NN.parquet   chỉ các ngày test (ngày train đọc từ gốc)
    injected_events.parquet      bảng phụ: mỗi sự kiện tiêm -> inj_id (log chính không có cột đánh dấu)
    injection_manifest.csv       mỗi lần tiêm một dòng
    labels.parquet               nhãn chuẩn is_anomaly / eval_exclude sinh từ manifest
    run_config.json              block, seed, split_day, đầy đủ eval_days của run
    features/raw/, processed/    ma trận đặc trưng tính lại trên log đã tiêm
    results/, models/            kết quả benchmark của run
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import polars as pl
import yaml


def load_run_eval_days(layout: "RunLayout", config_path: Path | str = "configs/injection.yaml") -> List[int]:
    """Use frozen run days; older runs fall back to their manifest block, never overlay days alone."""
    metadata = layout.root / "run_config.json"
    manifest = pl.read_csv(layout.manifest_path, infer_schema_length=None, schema_overrides={"campaign_id": pl.String})
    if manifest.is_empty():
        raise InjectionLayoutError("Injection manifest is empty.")
    if metadata.is_file():
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        days = payload["eval_days"]
        split_day = int(payload["split_day"])
    else:
        if "split" not in manifest.columns or manifest["split"].null_count():
            raise InjectionLayoutError("Older run needs a manifest with a dev/test split column.")
        blocks = manifest["split"].unique().to_list()
        if len(blocks) != 1:
            raise InjectionLayoutError("Manifest must belong to exactly one dev/test block.")
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        days = cfg["blocks"][blocks[0]]["days"]
        split_day = cfg["common"].get("split_day")
    days = sorted(set(int(d) for d in days))
    if not days or (split_day is not None and min(days) <= int(split_day)):
        raise InjectionLayoutError("Run evaluation days must be nonempty and after training.")
    if not set(manifest["day"].to_list()).issubset(days):
        raise InjectionLayoutError("Manifest contains injections outside the run evaluation block.")
    return days

__all__ = [
    "load_run_eval_days",
    "SECONDS_PER_DAY",
    "EVENT_IDS",
    "InjectionLayoutError",
    "RunLayout",
    "day_window",
    "day_of_time",
    "interim_day_path",
    "overlay_days",
    "events_outside_day",
    "assert_events_within_day",
]

SECONDS_PER_DAY = 86400
EVENT_IDS: Tuple[int, ...] = (4624, 4625)


class InjectionLayoutError(ValueError):
    """Vi phạm hợp đồng của run tiêm (sai ngày, đụng train, bảng phụ lệch log...)."""


def day_window(day: int) -> Tuple[int, int]:
    """Cửa sổ ``[start, end)`` (giây epoch LANL) của ngày ``day`` ở tầng interim."""
    if int(day) < 1:
        raise ValueError(f"day phải ≥ 1, nhận được {day}.")
    return (int(day) - 1) * SECONDS_PER_DAY, int(day) * SECONDS_PER_DAY


def day_of_time(time_col: str = "Time") -> pl.Expr:
    """Biểu thức ngày (đánh số từ 1) của một timestamp interim — nghịch đảo của :func:`day_window`."""
    return (pl.col(time_col) // SECONDS_PER_DAY + 1).cast(pl.Int32)


def interim_day_path(root: Path | str, event_id: int, day: int) -> Path:
    """Đường dẫn file interim của một EventID × ngày, đúng quy ước tên của ``data/interim``."""
    return Path(root) / f"event_{int(event_id)}" / f"event_{int(event_id)}_day-{int(day):02d}.parquet"


def overlay_days(events_dir: Path | str) -> List[int]:
    """Các ngày có ít nhất một file trong thư mục log đã tiêm (bất kể EventID)."""
    days: set[int] = set()
    for eid in EVENT_IDS:
        for p in (Path(events_dir) / f"event_{eid}").glob(f"event_{eid}_day-*.parquet"):
            try:
                days.add(int(p.stem.split("-")[-1]))
            except ValueError:
                continue
    return sorted(days)


def events_outside_day(events: pl.DataFrame, day: int, time_col: str = "Time") -> pl.DataFrame:
    """Các dòng có ``Time`` nằm NGOÀI cửa sổ của ngày ``day`` (rỗng = hợp lệ)."""
    start, end = day_window(day)
    return events.filter(
        pl.col(time_col).is_null() | (pl.col(time_col) < start) | (pl.col(time_col) >= end)
    )


def assert_events_within_day(
    events: pl.DataFrame, day: int, time_col: str = "Time", what: str = "sự kiện"
) -> None:
    """Báo lỗi nếu có sự kiện vượt nửa đêm (hoặc thiếu ``Time``) so với ngày dự định ``day``."""
    bad = events_outside_day(events, day, time_col)
    if bad.height:
        start, end = day_window(day)
        sample = bad[time_col].head(5).to_list()
        raise InjectionLayoutError(
            f"{bad.height:,} {what} nằm ngoài ngày {day} (cửa sổ [{start}, {end})): ví dụ Time = {sample}."
        )


@dataclass(frozen=True)
class RunLayout:
    """Đường dẫn của mọi artifact trong một run tiêm (gắn với ``run_id``)."""

    root: Path

    @classmethod
    def for_run(cls, runs_dir: Path | str, run_id: str) -> "RunLayout":
        if not run_id or any(ch in run_id for ch in r"\/:"):
            raise ValueError(f"run_id không hợp lệ: {run_id!r}.")
        return cls(Path(runs_dir) / run_id)

    @classmethod
    def from_events_dir(cls, events_dir: Path | str) -> "RunLayout":
        """Suy ra run từ thư mục ``events_injected`` (thư mục cha của nó là gốc run)."""
        return cls(Path(events_dir).resolve().parent)

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def events_dir(self) -> Path:
        return self.root / "events_injected"

    @property
    def injected_events_path(self) -> Path:
        return self.root / "injected_events.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "injection_manifest.csv"

    @property
    def labels_path(self) -> Path:
        return self.root / "labels.parquet"

    @property
    def difficulty_path(self) -> Path:
        return self.root / "difficulty.parquet"

    @property
    def features_raw_dir(self) -> Path:
        return self.root / "features" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def events_file(self, event_id: int, day: int) -> Path:
        return interim_day_path(self.events_dir, event_id, day)

    def ensure_dirs(self, extra: Iterable[Path] = ()) -> None:
        for d in [self.root, *(self.events_dir / f"event_{e}" for e in EVENT_IDS), *extra]:
            d.mkdir(parents=True, exist_ok=True)
