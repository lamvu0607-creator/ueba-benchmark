"""Lưới thí nghiệm tuần 4 (src/evaluation/grid.py) trên 2 run tổng hợp nhỏ."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from src.evaluation.grid import load_grid_config, load_run_data, mean_pr_curve, run_grid, select_best, summarize
from src.features.schema import FeatureSchema

FEATURES = list(FeatureSchema().core_features)


def _make_run(root: Path, seed: int) -> Path:
    """30 tài khoản × 12 ngày (train 1–8, eval 9–10, ngày 11–12 ngoài khối); 6 nạn nhân lệch mạnh ở eval."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(30):
        for d in range(1, 13):
            rows.append({"DomainName": "D", "UserName": f"U{u}", "day": d,
                         "entity_type": "User" if u < 25 else "Machine"})
    df = pl.DataFrame(rows)
    X = rng.normal(size=(df.height, len(FEATURES)))
    victims = [(f"U{u}", d) for u, d in [(1, 9), (2, 9), (3, 10), (4, 10), (5, 9), (6, 10)]]
    vmask = np.array([(r["UserName"], r["day"]) in victims for r in df.iter_rows(named=True)])
    X[vmask, :3] += 8.0
    df = df.with_columns([pl.Series(f, X[:, j]) for j, f in enumerate(FEATURES)])
    (root / "processed").mkdir(parents=True)
    df.write_parquet(root / "processed" / "feature_matrix_processed.parquet")
    pl.DataFrame({"DomainName": ["D"] * 6, "UserName": [u for u, _ in victims], "day": [d for _, d in victims],
                  "is_anomaly": [1] * 6, "eval_exclude": [False] * 6,
                  "scenario": ["a", "a", "a", "b", "b", "b"], "campaign_id": [None] * 6},
                 schema_overrides={"campaign_id": pl.String}).write_parquet(root / "labels.parquet")
    (root / "run_config.json").write_text(json.dumps({"split_day": 8, "eval_days": [9, 10], "seed": seed}),
                                          encoding="utf-8")
    return root


@pytest.fixture()
def grid_setup(temp_artifact_dir: Path):
    runs = [_make_run(temp_artifact_dir / f"run{s}", s) for s in (1, 2)]
    cfg = {
        "segment": "User", "output_dir": str(temp_artifact_dir / "out"), "runs": [str(r) for r in runs],
        "model_seeds": [42, 7], "contamination": 0.1, "precision_ks": [5], "daily_budgets": [3],
        "models": {
            "isolation_forest": [{"id": "if_a", "scaler": "robust", "params": {"n_estimators": 20}},
                                 {"id": "if_b", "scaler": "robust", "params": {"n_estimators": 40}}],
            "one_class_svm": [{"id": "oc_std", "scaler": "standard", "params": {"n_components": 20}}],
        },
        "baselines": ["zscore_baseline", "random_baseline"],
        "contamination_sweep": [0.05, 0.2],
    }
    cpath = temp_artifact_dir / "grid.yaml"
    cpath.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    sys_path = temp_artifact_dir / "system.yaml"
    sys_path.write_text(yaml.safe_dump({"evaluation": {"split_day": 8}}), encoding="utf-8")
    return cpath, sys_path, runs


def test_load_run_data_uses_frozen_eval_days_and_segment(grid_setup):
    _, _, runs = grid_setup
    data = load_run_data(runs[0], "User", split_day=8)
    assert set(data["days"]) == {9, 10}  # ngày 11–12 ngoài khối bị loại
    assert data["y"].sum() == 6 and data["y"].size == 25 * 2
    assert (data["train"]["day"] <= 8).all() and (data["train"]["entity_type"] == "User").all()
    with pytest.raises(ValueError, match="split_day"):
        load_run_data(runs[0], "User", split_day=7)


def test_run_grid_end_to_end(grid_setup):
    cpath, sys_path, _ = grid_setup
    log_path = cpath.parent / "experiment_log.csv"
    res = run_grid(cpath, system_config_path=sys_path, formats=["png"], experiment_log=log_path)
    out = Path(res["output_dir"])
    runs = pd.read_csv(out / "grid_runs.csv")
    assert len(runs) == 2 * (3 + 2)  # 2 seed × (3 cấu hình ML + 2 baseline)
    assert set(runs["model_seed"]) == {42, 7}
    summary = pd.read_csv(out / "grid_summary.csv")
    assert (summary["pr_auc_n"] == 2).all() and summary["pr_auc_std"].notna().all()
    assert res["best"]["isolation_forest"] in {"if_a", "if_b"} and res["best"]["one_class_svm"] == "oc_std"
    # tín hiệu lệch +8σ: mô hình ML phải vượt random
    ml = summary[summary["kind"] == "ml"]["pr_auc_mean"].max()
    assert ml > summary.loc[summary["model"] == "random_baseline", "pr_auc_mean"].iloc[0]
    cont = pd.read_csv(out / "contamination_summary.csv")
    assert len(cont) == 2 * 2  # 2 mô hình × 2 mức
    assert set(pd.read_csv(out / "grid_scenarios.csv")["scenario"]) == {"a", "b"}
    assert {"grid_pr_auc.png", "pr_curves.png", "contamination_sensitivity.png", "scenario_roc_auc.png"} <= {
        p.name for p in (out / "figures").iterdir()}
    manifest = json.loads((out / "grid_manifest.json").read_text(encoding="utf-8"))
    assert manifest["best_configs"] == res["best"] and len(manifest["inputs"]) == 2
    # mục 5.6: mọi lần fit của lưới (và quét contamination) vào nhật ký chung, kèm cấu hình + kết quả
    log = pd.read_csv(log_path)
    assert len(log) == len(runs) + 2 * 2 * 2      # lưới + 2 mô hình × 2 mức × 2 seed
    assert {"config_id", "run_id", "pr_auc", "experiment"} <= set(log.columns) and log["pr_auc"].notna().all()
    run_grid(cpath, system_config_path=sys_path, formats=["png"], experiment_log=log_path)
    assert len(pd.read_csv(log_path)) == len(log)  # chạy lại cùng commit/cấu hình: không ghi trùng


def test_summarize_and_select_best():
    runs = pd.DataFrame({"kind": ["ml"] * 4, "model": ["m"] * 4, "config_id": ["a", "a", "b", "b"],
                         "pr_auc": [0.1, 0.3, 0.2, 0.2]})
    s = summarize(runs, ["kind", "model", "config_id"], ["pr_auc"])
    assert s.set_index("config_id").loc["a", "pr_auc_std"] == pytest.approx(np.std([0.1, 0.3], ddof=1))
    assert select_best(s) == {"m": "a"}


def test_grid_config_validation(temp_artifact_dir: Path):
    p = temp_artifact_dir / "bad.yaml"
    p.write_text(yaml.safe_dump({"runs": ["a", "b"], "model_seeds": [1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="cùng độ dài"):
        load_grid_config(p)
    p.write_text(yaml.safe_dump({"runs": ["a"], "model_seeds": [1], "models": {
        "x": [{"id": "dup"}], "y": [{"id": "dup"}]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="trùng"):
        load_grid_config(p)


def test_mean_pr_curve_averages_precision_on_common_recall_grid():
    y = np.array([1, 0, 1, 0])
    grid = np.array([0.5, 1.0])
    # seed 1: dương xếp 1, 3 -> P(r=0,5)=1, P(r=1)=2/3; seed 2: dương xếp 2, 3 -> 1/2, 2/3
    df = mean_pr_curve([y, y], [np.array([4, 3, 2, 1.0]), np.array([3, 4, 2, 1.0])], grid)
    assert df["precision"].tolist() == pytest.approx([0.75, 2 / 3])
    assert df["precision_std"].iloc[0] == pytest.approx(np.std([1, 0.5], ddof=1))
    assert (df["n_seeds"] == 2).all()
