"""Validate an uploaded interim batch and score User account-days."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from demo.prepare import (
    DEFAULT_ARTIFACTS, DEFAULT_INTERIM, GRID, MODELS, N_FEATURES, PARAMS, ROOT, SCHEMA, TRAIN_END, frozen_config,
    interim_day_files,
)
from src.data.raw_to_interim import COLUMN_NAMES, INTERIM_SCHEMA
from src.evaluation.manifest import file_sha256
from src.features.extractor import build_account_day_matrix
from src.features.preprocessor import normalize_features
from src.models.pipeline import AnomalyPipeline


class DemoInputError(Exception):
    def __init__(self, error: str, message: str, status_code: int = 400):
        self.error, self.message, self.status_code = error, message, status_code
        super().__init__(message)


def _compatible(got: pa.DataType, want: pa.DataType) -> bool:
    """Cùng kiểu, hoặc cùng họ: chuỗi (string / large_string / string_view — polars ghi large_string) hay số nguyên."""
    if got == want:
        return True
    is_str = lambda t: pa.types.is_string(t) or pa.types.is_large_string(t) or str(t) == "string_view"
    if is_str(want):
        return is_str(got) or pa.types.is_null(got)
    if pa.types.is_integer(want):
        return pa.types.is_integer(got)
    return False


def _has_context(interim_dir: Path, day: int) -> bool:
    return all(p.is_file() for p in interim_day_files(interim_dir, day))


class DemoService:
    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACTS, interim_dir: Path = DEFAULT_INTERIM):
        self.artifact_dir, self.interim_dir = Path(artifact_dir), Path(interim_dir)
        self.models: dict[str, AnomalyPipeline] = {}
        self.metadata: dict = {}
        self.activity = pl.DataFrame()

    def load(self) -> None:
        self.models.clear()
        self.metadata = {}
        self.activity = pl.DataFrame()
        metadata = json.loads((self.artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        _, features = frozen_config()
        if (metadata.get("format") != "ueba-demo-artifacts/1"
                or metadata.get("feature_names") != features
                or metadata.get("feature_schema_sha256") != file_sha256(SCHEMA)
                or metadata.get("grid_sha256") != file_sha256(GRID)
                or metadata.get("params_sha256") != file_sha256(PARAMS)
                or metadata.get("train_start") != 1
                or metadata.get("train_end") != TRAIN_END
                or metadata.get("seed") != 42
                or metadata.get("segment") != "User"
                or set(metadata.get("models", {})) != set(MODELS)):
            raise ValueError(f"Artifact không khớp cấu hình User/{N_FEATURES} đặc trưng hiện tại.")
        activity = pl.read_parquet(self.artifact_dir / "historical_activity.parquet")
        if not {"DomainName", "UserName", "day"}.issubset(activity.columns):
            raise ValueError("Bảng hiện diện tài khoản trong artifact sai schema.")
        self.metadata, self.activity = metadata, activity
        for name in MODELS:
            pipeline = AnomalyPipeline.load(self.artifact_dir / f"{name}.joblib")
            if (not pipeline.is_fitted_ or not pipeline.model.is_fitted_
                    or pipeline.model.name != name or pipeline.feature_names != features
                    or pipeline.random_state != 42 or pipeline.model.random_state != 42
                    or not math.isclose(float(pipeline.model.contamination), 0.05)
                    or pipeline.model.fit_scores_ is None or not np.isfinite(pipeline.model.fit_scores_).all()
                    or not math.isfinite(float(pipeline.model.threshold_))
                    or not math.isclose(float(pipeline.model.threshold_), float(metadata["models"][name]["threshold"]))):
                raise ValueError(f"Artifact mô hình {name} không hợp lệ.")
            self.models[name] = pipeline

    def health(self) -> dict:
        return {
            "status": "ready" if len(self.models) == len(MODELS) else "unavailable",
            "artifact_version": self.metadata.get("model_version"),
            "schema_version": self.metadata.get("feature_schema_version"),
            "schema_sha256": self.metadata.get("feature_schema_sha256"),
            "models": {name: name in self.models for name in MODELS},
        }

    def _inspect(self, path: Path, filename: str) -> tuple[int, int, int]:
        if Path(filename).suffix.lower() != ".parquet":
            raise DemoInputError("invalid_file", "Chỉ nhận file .parquet.")
        try:
            schema = pq.read_schema(path)
        except Exception as exc:
            raise DemoInputError("invalid_file", "File không phải Parquet hợp lệ.") from exc
        if len(schema.names) != len(set(schema.names)):
            raise DemoInputError("invalid_schema", "Tên cột trong Parquet bị trùng.")
        missing = [name for name in COLUMN_NAMES if name not in schema.names]
        if missing:
            raise DemoInputError("invalid_schema", f"Thiếu cột: {', '.join(missing)}.")
        wrong = [field.name for field in INTERIM_SCHEMA if not _compatible(schema.field(field.name).type, field.type)]
        if wrong:
            raise DemoInputError("invalid_schema", f"Sai kiểu dữ liệu ở cột: {', '.join(wrong)}.")
        try:
            stats = pl.scan_parquet(path).select(
                pl.len().alias("events"),
                pl.col("Time").min().alias("min_time"),
                pl.col("Time").max().alias("max_time"),
                pl.col("Time").null_count().alias("null_time"),
                pl.col("EventID").null_count().alias("null_event"),
                (~pl.col("EventID").is_in([4624, 4625])).sum().alias("bad_event"),
                ((pl.col("Time") // 86400 + 1).n_unique()).alias("days"),
            ).collect().row(0, named=True)
        except Exception as exc:
            raise DemoInputError("invalid_file", "Không thể đọc dữ liệu Parquet.") from exc
        if (not stats["events"] or stats["null_time"] or stats["null_event"] or stats["bad_event"]
                or stats["min_time"] < 0):
            raise DemoInputError("invalid_file", "Time phải không âm và EventID chỉ gồm 4624/4625, không được NULL.")
        start, end = stats["min_time"] // 86400 + 1, stats["max_time"] // 86400 + 1
        if start <= TRAIN_END or stats["days"] != end - start + 1:
            raise DemoInputError("invalid_window", "Upload cần các ngày liên tục sau Day 42, không chạm tập train.")
        missing_context = [day for day in range(start - 7, start) if not _has_context(self.interim_dir, day)]
        if missing_context:
            raise DemoInputError("missing_context", f"Kho lịch sử thiếu context: {', '.join(f'Day {day}' for day in missing_context)}.")
        if start > self.metadata["context_archive_end"] + 1:
            raise DemoInputError("missing_context", "Artifact hiện diện tài khoản chưa bao phủ lịch sử trước batch.")
        return int(stats["events"]), int(start), int(end)

    def analyze(self, path: Path, filename: str) -> dict:
        if len(self.models) != len(MODELS):
            raise DemoInputError("artifact_unavailable", "Artifact demo chưa sẵn sàng.", 503)
        event_count, start, end = self._inspect(path, filename)
        with tempfile.TemporaryDirectory(prefix="ueba-demo-", dir=self.interim_dir.parent) as tmp_name:
            tmp = Path(tmp_name)
            # Batch được tách theo ngày × EventID thành layout interim riêng (events_dir); 7 ngày context đọc thẳng
            # từ kho interim — cùng nguồn và cùng đường tính với ma trận train, không chép/symlink file lịch sử.
            events = tmp / "events"
            dtypes = {f.name: (pl.Int64 if pa.types.is_int64(f.type) else pl.Int32 if pa.types.is_integer(f.type) else pl.String)
                      for f in INTERIM_SCHEMA}
            source = pl.scan_parquet(path).select(COLUMN_NAMES).cast(dtypes)
            for day in range(start, end + 1):
                for event_id in (4624, 4625):
                    destination = events / f"event_{event_id}" / f"event_{event_id}_day-{day:02d}.parquet"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source.filter((pl.col("Time") // 86400 + 1 == day) & (pl.col("EventID") == event_id)).collect().write_parquet(destination)
            config = yaml.safe_load((ROOT / "configs/system_config.yaml").read_text(encoding="utf-8"))
            prior = (self.activity.filter(pl.col("day") < start - 7)
                     .group_by(["DomainName", "UserName"]).agg(pl.col("day").max()))
            raw = build_account_day_matrix(start - 7, end, interim_dir=self.interim_dir, output_dir=tmp / "features",
                                           config=config, events_dir=events, prior_activity=prior)
            if raw.is_empty():
                raise DemoInputError("invalid_file", "Không có sự kiện hợp lệ sau khi làm sạch.")
            current = normalize_features(raw.filter(pl.col("day") >= start))
            scored = current.filter(pl.col("entity_type") == "User").sort(["day", "DomainName", "UserName"])
            return self._result(event_count, start, end, current, scored)

    def _result(self, event_count: int, start: int, end: int, current: pl.DataFrame, scored: pl.DataFrame) -> dict:
        n = scored.height
        keys = ["DomainName", "UserName"]
        rows = scored.select([*keys, "day", *self.metadata["feature_names"]]).to_dicts()
        for row in rows:
            for name in self.metadata["feature_names"]:
                if isinstance(row[name], float) and not math.isfinite(row[name]):
                    row[name] = None
        model_stats = {}
        agreements = np.zeros(n, dtype=np.int8)
        ranking = np.zeros(n, dtype=float)
        for name in MODELS:
            pipeline = self.models[name]
            scores = pipeline.score(scored) if n else np.array([], dtype=float)
            flags = scores >= pipeline.model.threshold_
            reference = np.sort(pipeline.model.fit_scores_)
            pct = np.searchsorted(reference, scores, side="right") / len(reference)
            agreements += flags.astype(np.int8)
            ranking += pct
            for i, row in enumerate(rows):
                row[name] = {"score": float(scores[i]), "percentile": float(pct[i]), "alert": bool(flags[i])}
            counts, edges = np.histogram(scores, bins=16) if n else (np.zeros(16, dtype=int), np.linspace(0, 1, 17))
            model_stats[name] = {
                "threshold": float(pipeline.model.threshold_),
                "alerts": int(flags.sum()), "alert_rate": float(flags.mean()) if n else 0.0,
                "distribution": {"edges": edges.tolist(), "counts": counts.tolist()},
            }
        for i, row in enumerate(rows):
            row["agreement_count"] = int(agreements[i])
        order = sorted(range(n), key=lambda i: (-int(agreements[i]), -float(ranking[i]), rows[i]["day"], rows[i]["DomainName"], rows[i]["UserName"]))
        types = {r["entity_type"]: r["len"] for r in current.group_by("entity_type").agg(pl.len()).to_dicts()}
        accounts = current.select(keys).unique().height
        return {
            "status": "ok", "day_start": start, "day_end": end,
            "totals": {"events": event_count, "accounts": accounts, "account_days": current.height,
                       "scored_user_account_days": n, "unscored_by_type": {k: v for k, v in types.items() if k != "User"}},
            "models": model_stats,
            "agreement": {str(k): int((agreements == k).sum()) for k in range(4)},
            "feature_names": self.metadata["feature_names"],
            "top_50": [rows[i] for i in order[:50]],
        }
