"""
Fit và lưu 3 mô hình ML của bộ 22 đặc trưng cho demo (pipeline imputer + scaler + mô hình, file .joblib).

Cùng thiết lập với lưới test (configs/benchmark_grid_test_22f.yaml): cấu hình chốt từ dev, ``drop_features``,
``sort_rows``, phân khúc User, contamination của lưới. Fit trên log GỐC ngày 1–split_day (mặc định 42), seed 42.

Kiểm tra: nạp lại từng file .joblib, chấm ngày đánh giá của run test seed 42 rồi so PR-AUC với dòng tương ứng
trong grid_runs.csv của lưới (ngày 1–42 của run test không bị tiêm nên mô hình phải trùng với mô hình của lưới).

    python scripts/evaluation/train_demo_models.py
    # nạp lại: AnomalyPipeline.load("experiments/models/User_22f/one_class_svm.joblib").score(df)

Đầu ra: experiments/models/User_22f/<model>.joblib, demo_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.grid import _params_with, fit_and_score, load_grid_config, load_run_data  # noqa: E402
from src.evaluation.manifest import file_sha256, git_state, library_versions  # noqa: E402
from src.models.pipeline import AnomalyPipeline  # noqa: E402
from src.models.registry import load_params  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("ueba_benchmark.demo_models")


def ml_features(cfg: dict) -> list:
    from src.features.schema import FeatureSchema

    drop = set(cfg.get("drop_features") or [])
    return [f for f in FeatureSchema().core_features if f not in drop]


def main() -> None:
    p = argparse.ArgumentParser(description="Fit và lưu mô hình 22 đặc trưng cho demo")
    p.add_argument("--config", default="configs/benchmark_grid_test_22f.yaml")
    p.add_argument("--model-params", default="configs/model_params.yaml")
    p.add_argument("--matrix", default="data/processed/feature_matrix_processed.parquet")
    p.add_argument("--split-day", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="experiments/models/User_22f")
    p.add_argument("--no-check", action="store_true", help="Bỏ bước so PR-AUC với lưới")
    args = p.parse_args()

    cfg = load_grid_config(args.config)
    params = load_params(args.model_params)
    feats = ml_features(cfg)
    c0 = float(cfg.get("contamination", 0.05))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(args.matrix)
    if cfg.get("sort_rows"):
        df = df.sort(["day", "DomainName", "UserName"])
    if cfg.get("segment"):
        df = df.filter(pl.col("entity_type") == cfg["segment"])
    train = df.filter(pl.col("day") <= args.split_day).to_pandas()
    logger.info("Train: %s dòng (ngày 1–%d), %d đặc trưng.", f"{len(train):,}", args.split_day, len(feats))

    models = {}
    for model, confs in cfg["models"].items():
        c = confs[0]  # lưới test: một cấu hình chốt / mô hình
        res = fit_and_score({"train": train, "eval": train.head(1)}, model,
                            _params_with(params, model, c.get("params") or {}), c.get("scaler", "robust"),
                            args.seed, c0, feature_names=feats)
        path = res["pipeline"].save(out / f"{model}.joblib")
        models[model] = {"config_id": c["id"], "scaler": c.get("scaler", "robust"), "params": c.get("params") or {},
                         "path": str(path), "threshold": res["threshold"], "n_fit": res["n_fit"],
                         "fit_seconds": round(res["fit_seconds"], 1)}
        logger.info("%s (%s): fit %.0fs, ngưỡng %.4f -> %s", model, c["id"], res["fit_seconds"], res["threshold"], path)

    check = []
    if not args.no_check:
        idx = [int(s) for s in cfg["model_seeds"]].index(args.seed)
        run = cfg["runs"][idx]
        data = load_run_data(run, cfg.get("segment"), args.split_day, sort_rows=bool(cfg.get("sort_rows")))
        grid = pd.read_csv(Path(cfg["output_dir"]) / "grid_runs.csv")
        for model, info in models.items():
            scores = AnomalyPipeline.load(info["path"]).score(data["eval"])
            pr = float(average_precision_score(data["y"], np.asarray(scores, dtype=np.float64)))
            ref = grid[(grid["model"] == model) & (grid["config_id"] == info["config_id"])
                       & (grid["model_seed"] == args.seed)]["pr_auc"]
            ref = float(ref.iloc[0]) if len(ref) else float("nan")
            check.append({"model": model, "run": data["run_id"], "pr_auc_demo": round(pr, 6),
                          "pr_auc_grid": round(ref, 6), "rel_diff": round(abs(pr - ref) / ref, 6) if ref else None})
            logger.info("[check] %-22s PR-AUC demo %.4f | lưới %.4f", model, pr, ref)

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "git": git_state(), "libraries": library_versions(),
        "config_path": args.config, "matrix": args.matrix, "matrix_sha256": file_sha256(Path(args.matrix)),
        "segment": cfg.get("segment"), "split_day": args.split_day, "seed": args.seed, "contamination": c0,
        "features": feats, "models": models, "check": check,
    }
    (out / "demo_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                            encoding="utf-8")
    logger.info("Xong -> %s", out)


if __name__ == "__main__":
    main()
