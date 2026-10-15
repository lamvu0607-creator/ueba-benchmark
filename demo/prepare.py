"""Fit the frozen User/30-feature benchmark configuration for local inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml

from src.evaluation.manifest import file_sha256, git_state
from src.features.schema import FeatureSchema
from src.models.registry import create_pipeline, load_params

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "configs/benchmark_grid_test_30f.yaml"
PARAMS = ROOT / "configs/model_params.yaml"
SCHEMA = ROOT / "configs/feature_schema.yaml"
DEFAULT_PROCESSED = ROOT / "data/processed/feature_matrix_processed.parquet"
DEFAULT_CLEANED = ROOT / "data/cleaned"
DEFAULT_ARTIFACTS = ROOT / "experiments/models/demo_30f_user"
MODELS = ("isolation_forest", "local_outlier_factor", "one_class_svm")
TRAIN_END = 42


def frozen_config() -> tuple[dict, list[str]]:
    grid = yaml.safe_load(GRID.read_text(encoding="utf-8"))
    if grid.get("segment") != "User" or set(grid.get("models", {})) != set(MODELS):
        raise ValueError("Cấu hình benchmark cuối phải có đúng ba mô hình cho User.")
    drops = grid.get("drop_features") or []
    core = FeatureSchema(SCHEMA).core_features
    if len(drops) != len(set(drops)) or set(drops) - set(core):
        raise ValueError("drop_features không khớp feature schema.")
    features = [name for name in core if name not in set(drops)]
    if len(features) != 30 or any(len(grid["models"][name]) != 1 for name in MODELS):
        raise ValueError("Cấu hình cuối không còn đúng 30 đặc trưng/một cấu hình mỗi mô hình.")
    return grid, features


def prepare_artifacts(
    processed_path: Path = DEFAULT_PROCESSED,
    cleaned_dir: Path = DEFAULT_CLEANED,
    output_dir: Path = DEFAULT_ARTIFACTS,
) -> Path:
    processed_path, cleaned_dir, output_dir = map(Path, (processed_path, cleaned_dir, output_dir))
    if not processed_path.is_file():
        raise FileNotFoundError(f"Thiếu {processed_path}; chạy 'python main.py --stage features' trước.")
    if output_dir.exists():
        raise FileExistsError(f"Artifact đã tồn tại: {output_dir}. Chuyển/xóa bản cũ trước khi chuẩn bị lại.")
    grid, features = frozen_config()
    params = load_params(PARAMS)
    all_days = pl.scan_parquet(processed_path).select("day").unique().collect()["day"].to_list()
    if not set(range(1, TRAIN_END + 1)).issubset(all_days):
        raise ValueError("Ma trận processed thiếu ngày train trong Day 1–42.")
    archive_days = []
    for path in cleaned_dir.glob("cleaned_day-*.parquet"):
        try:
            archive_days.append(int(path.stem.rsplit("-", 1)[1]))
        except ValueError:
            continue
    if not archive_days or set(range(1, max(archive_days) + 1)) != set(archive_days):
        raise ValueError("Kho cleaned phải có các file ngày liên tục từ Day 1.")
    archive_end = max(archive_days)
    if not set(range(1, archive_end + 1)).issubset(all_days):
        raise ValueError("Ma trận processed thiếu ngày thuộc kho cleaned; chạy lại stage features.")

    frame = pl.scan_parquet(processed_path)
    missing = [name for name in ["DomainName", "UserName", "day", "entity_type", *features] if name not in frame.collect_schema()]
    if missing:
        raise ValueError(f"Ma trận processed thiếu cột: {missing}.")
    train = frame.filter(pl.col("day").is_between(1, TRAIN_END) & (pl.col("entity_type") == "User")).collect()
    train = train.sort(["day", "DomainName", "UserName"])
    if train.is_empty():
        raise ValueError("Không có User account-day trong train.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    try:
        activity = frame.select("DomainName", "UserName", "day").filter(pl.col("day").is_between(1, archive_end)).collect()
        activity.write_parquet(output_dir / "historical_activity.parquet")
        metadata = {
            "format": "ueba-demo-artifacts/1",
            "model_version": hashlib.sha256(
                (file_sha256(processed_path) + file_sha256(GRID) + file_sha256(PARAMS) + file_sha256(SCHEMA)).encode()
            ).hexdigest()[:16],
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_start": 1,
            "train_end": TRAIN_END,
            "context_archive_end": archive_end,
            "segment": "User",
            "feature_schema_version": "v4.1",
            "feature_schema_sha256": file_sha256(SCHEMA),
            "grid_sha256": file_sha256(GRID),
            "params_sha256": file_sha256(PARAMS),
            "processed_sha256": file_sha256(processed_path),
            "feature_names": features,
            "seed": 42,
            "git_commit": git_state().get("commit"),
            "models": {},
        }
        for name in MODELS:
            spec = grid["models"][name][0]
            model_params = dict(params)
            model_params[name] = {**(params.get(name) or {}), **(spec.get("params") or {})}
            if name == "one_class_svm":
                model_params[name]["nu"] = float(grid["contamination"])
            model_params["preprocessing"] = {**(params.get("preprocessing") or {}), "scaler": spec["scaler"]}
            pipeline = create_pipeline(
                name, params=model_params, feature_names=features,
                contamination=float(grid["contamination"]), random_state=42,
            )
            pipeline.fit(train)
            pipeline.save(output_dir / f"{name}.joblib")
            metadata["models"][name] = {
                "config_id": spec["id"], "scaler": spec["scaler"],
                "params": pipeline.model.get_metadata()["native_params"],
                "threshold": float(pipeline.model.threshold_),
                "n_train_rows": pipeline.n_train_rows_,
            }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn bị artifact cho UEBA demo (User, 30 đặc trưng).")
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--cleaned-dir", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    print(f"Đã tạo artifact tại {prepare_artifacts(args.processed, args.cleaned_dir, args.output)}")


if __name__ == "__main__":
    main()
