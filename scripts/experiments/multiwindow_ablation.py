"""
Gộp nhiều cửa sổ lịch sử trên khối DEV (tuần 5): mô hình thấy CÙNG LÚC cột 7 ngày và bản 14 / 21 ngày của chúng.

    python scripts/experiments/multiwindow_ablation.py

Khác ``window_ablation.py`` (thay cửa sổ 7 ngày bằng N ngày, giữ 41 cột): ở đây cột 14 / 21 ngày được THÊM vào
bộ 30 đặc trưng với hậu tố ``_14d`` / ``_21d``.

- ``base30``: bộ 30 đặc trưng (core trừ ``drop_features`` của ``configs/benchmark_grid_test_30f.yaml``).
- ``A_all_hist``: base30 + bản 14 và 21 ngày của mọi đặc trưng ``*_7d`` còn trong base30 (17 × 2 = +34 -> 64).
- ``B_failure_hist``: base30 + bản 14 và 21 ngày của 4 đặc trưng lịch sử nhóm failure (+8 -> 38), nhắm spraying.

Ma trận 14 / 21 ngày tính bằng ``window_ablation.build_matrix`` vào ``data/window_ablation/w{N}/<run_id>/``
(tiêm lại cùng seed nếu thiếu ``events_injected/``). Cửa sổ chỉ đổi 18 cột ``*_7d``; script kiểm tra mọi cột khác
khớp ma trận gốc của run trước khi ghép.

Mô hình, run và seed như ``feature_ablation.py``; so bằng hiệu ghép cặp với ``base30``.
Đầu ra: experiments/ablation/multiwindow/
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from feature_ablation import ML, OVERALL, SCEN, _append, _done, paired_delta  # noqa: E402
from window_ablation import build_matrix, ensure_injected  # noqa: E402

from src.evaluation.feature_groups import ABLATION_GROUPS  # noqa: E402
from src.evaluation.grid import _params_with, _scenario_rows, fit_and_score, load_grid_config, load_run_data  # noqa: E402
from src.evaluation.manifest import file_sha256  # noqa: E402
from src.evaluation.metrics import labeled_metrics  # noqa: E402
from src.evaluation.scenario import scenario_detection_rows  # noqa: E402
from src.features.schema import FeatureSchema  # noqa: E402
from src.models.registry import load_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("ueba_benchmark.multiwindow_ablation")

KEYS = ["day", "DomainName", "UserName"]
WINDOWS = [14, 21]
REF = "base30"


def history_features(core: list) -> list:
    return [f for f in core if f.endswith("_7d")]


def multiwindow_configs(base: list) -> dict:
    """Tên cấu hình -> danh sách đặc trưng (cột thêm mang hậu tố _14d / _21d)."""
    hist = history_features(base)
    fail = [f for f in ABLATION_GROUPS["failure"] if f in hist]

    def extra(fs):
        return [f.removesuffix("_7d") + f"_{w}d" for w in WINDOWS for f in fs]

    return {REF: list(base), "A_all_hist": list(base) + extra(hist), "B_failure_hist": list(base) + extra(fail)}


def window_columns(run: Path, data_out: Path, hist: list, sys_cfg: dict) -> pl.DataFrame:
    """Khoá + cột lịch sử 14 / 21 ngày của một run, sau khi kiểm tra các cột không phụ thuộc cửa sổ khớp bản gốc."""
    orig = pl.read_parquet(run / "processed" / "feature_matrix_processed.parquet").sort(KEYS)
    labels_sha = file_sha256(run / "labels.parquet")
    out = orig.select(KEYS)
    events = None
    for w in WINDOWS:
        if not (data_out / f"w{w}" / run.name / "processed" / "feature_matrix_processed.parquet").is_file():
            events = events or ensure_injected(run)
            if file_sha256(run / "labels.parquet") != labels_sha:
                raise RuntimeError(f"{run}: tiêm lại làm đổi labels.parquet -> không cùng dữ liệu")
        dst = build_matrix(run, events, w, data_out, sys_cfg)
        m = pl.read_parquet(dst / "processed" / "feature_matrix_processed.parquet").sort(KEYS)
        if not m.select(KEYS).equals(orig.select(KEYS)):
            raise RuntimeError(f"{run.name} w{w}: khoá dòng khác ma trận gốc")
        same = [c for c in orig.columns if c not in hist and c not in KEYS and orig[c].dtype.is_numeric()]
        bad = [c for c in same if float((orig[c].cast(pl.Float64) - m[c].cast(pl.Float64)).abs().max() or 0) > 1e-9]
        if bad:
            raise RuntimeError(f"{run.name} w{w}: cột ngoài *_7d lệch bản gốc: {bad}")
        out = out.with_columns([m[f].alias(f.removesuffix("_7d") + f"_{w}d") for f in hist])
    return out


def run(args) -> None:
    dev = load_grid_config(args.dev_grid)
    frozen = load_grid_config(args.frozen_grid)
    params = load_params("configs/model_params.yaml")
    sys_cfg = yaml.safe_load((ROOT / "configs/system_config.yaml").read_text(encoding="utf-8")) or {}
    c0 = float(frozen.get("contamination", 0.05))
    core = list(FeatureSchema().core_features)
    drop = set(frozen.get("drop_features") or [])
    base = [f for f in core if f not in drop]
    cfgs = multiwindow_configs(base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "configs.json").write_text(json.dumps({k: {"n_features": len(v), "added": v[len(base):]}
                                                  for k, v in cfgs.items()}, indent=2), encoding="utf-8")
    runs_csv, scen_csv = out / "runs.csv", out / "scenario_runs.csv"
    done = _done(runs_csv)
    hist = history_features(core)

    for i, (run_dir, mseed) in enumerate(zip(dev["runs"], dev["model_seeds"])):
        run_path = Path(run_dir)
        extra = window_columns(run_path, Path(args.data_out), hist, sys_cfg)  # tính ma trận trước, kể cả khi đã chấm
        todo = [(a, m) for a in cfgs for m in ML if (run_path.name, a, m) not in done]
        if not todo:
            continue
        data = load_run_data(run_dir, frozen.get("segment"), sort_rows=True)
        ext = extra.to_pandas()
        for part in ("train", "eval"):
            merged = data[part].merge(ext, on=KEYS, how="left", validate="one_to_one")
            if len(merged) != len(data[part]) or merged[ext.columns[-1]].isna().all():
                raise RuntimeError(f"{run_path.name}: ghép cột cửa sổ thất bại ({part})")
            data[part] = merged
        for a, m in todo:
            c = frozen["models"][m][0]
            res = fit_and_score(data, m, _params_with(params, m, c.get("params") or {}), c.get("scaler", "robust"),
                                int(mseed), c0, feature_names=cfgs[a])
            row = {"run_id": run_path.name, "seed_index": i, "ablation": a, "model": m}
            met = labeled_metrics(data["y"], res["scores"], data["days"], ks=[50, 100], budget_ratio=c0,
                                  daily_budgets=[100])
            _append(runs_csv, [{**row, "n_features": len(cfgs[a]), **{k: met[k] for k in OVERALL},
                                "fit_seconds": round(res["fit_seconds"] + res["score_seconds"], 1)}])
            det = pd.DataFrame(scenario_detection_rows(data["y"], data["scenario"].astype(object), res["scores"],
                                                       res["flags"], data["days"], daily_ks=(100,)))
            auc = pd.DataFrame(_scenario_rows(data, res["scores"], {}))[["scenario", "pr_auc", "roc_auc"]]
            sc = det[["scenario", "recall_top100_day"]].merge(auc, on="scenario", how="left")
            _append(scen_csv, [{**row, **r} for r in sc.to_dict(orient="records")])
            logger.info("run %d/%d %s  %-15s %-22s PR-AUC %.3f  (%.0fs)", i + 1, len(dev["runs"]), run_path.name, a,
                        m, met["pr_auc"], res["fit_seconds"] + res["score_seconds"])


def analyze(out: Path) -> None:
    runs = pd.read_csv(out / "runs.csv")
    scen = pd.read_csv(out / "scenario_runs.csv")
    dov = paired_delta(runs.drop(columns=["n_features", "fit_seconds"]), ["model"], OVERALL, ref=REF)
    dov.round(4).to_csv(out / "delta_overall.csv", index=False, encoding="utf-8-sig")
    dsc = paired_delta(scen, ["model", "scenario"], SCEN, ref=REF)
    dsc.round(4).to_csv(out / "delta_scenario.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(dov[["ablation", "model", "pr_auc_full", "d_pr_auc_mean", "d_pr_auc_std", "recall_at_100_per_day_full",
                   "d_recall_at_100_per_day_mean", "d_recall_at_100_per_day_std"]].round(3).to_string(index=False))
        print(dsc[["ablation", "model", "scenario", "recall_top100_day_full", "d_recall_top100_day_mean",
                   "d_recall_top100_day_std", "d_roc_auc_mean"]].round(3).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dev-grid", default="configs/benchmark_grid.yaml")
    p.add_argument("--frozen-grid", default="configs/benchmark_grid_test_30f.yaml",
                   help="cấu hình mô hình chốt + drop_features của bộ 30")
    p.add_argument("--data-out", default="data/window_ablation")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--out", default="experiments/ablation/multiwindow")
    args = p.parse_args()
    if not args.analyze_only:
        run(args)
    analyze(Path(args.out))


if __name__ == "__main__":
    main()
