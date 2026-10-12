"""
Ablation theo nhóm đặc trưng trên khối DEV (tuần 5): bỏ từng nhóm rồi đo lại OCSVM / IF / LOF.

    python scripts/evaluation/feature_ablation.py                  # 10 cấu hình × 3 mô hình × 5 run dev
    python scripts/evaluation/feature_ablation.py --only full drop_rhythm

- Run và model seed lấy từ ``configs/benchmark_grid.yaml`` (dev), cấu hình mô hình chốt lấy từ
  ``configs/benchmark_grid_test.yaml`` -> không chọn lại tham số, không nhìn khối test.
- Dòng được sắp cố định (day, DomainName, UserName) trước khi fit.
- Ghi dần từng (run, cấu hình, mô hình) vào ``runs.csv`` / ``scenario_runs.csv``; chạy lại sẽ bỏ qua phần đã có.
- So với bản đủ 41 đặc trưng bằng **hiệu ghép cặp** trên cùng run + seed: TB ± std của hiệu qua 5 run.

Đầu ra: experiments/ablation/feature_groups/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.feature_groups import ABLATION_GROUPS, LOW_SHAP_FEATURES, schema_origin  # noqa: E402
from src.evaluation.grid import (  # noqa: E402
    _params_with, _scenario_rows, fit_and_score, load_grid_config, load_run_data, summarize,
)
from src.evaluation.metrics import labeled_metrics  # noqa: E402
from src.evaluation.scenario import scenario_detection_rows  # noqa: E402
from src.models.registry import load_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("ueba_benchmark.feature_ablation")

ML = ["one_class_svm", "isolation_forest", "local_outlier_factor"]
SHORT = {"one_class_svm": "OCSVM", "isolation_forest": "IF", "local_outlier_factor": "LOF"}
KEYS = ["run_id", "seed_index", "ablation", "model"]
OVERALL = ["pr_auc", "roc_auc", "precision_at_50", "precision_at_100_per_day", "recall_at_100_per_day"]
SCEN = ["pr_auc", "roc_auc", "recall_top100_day"]
CONTROL = "control_shuffle_rows"


def ablation_configs(origin: dict) -> dict:
    """Tên cấu hình -> danh sách đặc trưng BỊ BỎ (thứ tự theo reports/week5/shap_lime.md §3)."""
    g = ABLATION_GROUPS
    cfgs = {"full": [], "drop_rhythm": g["rhythm"], "drop_volume_context": g["volume"] + g["context"],
            "drop_template_v4": [f for f, o in origin.items() if o == "template_v4"]}
    for grp in ["failure", "time", "logon_type", "fanout", "novelty"]:
        cfgs[f"drop_{grp}"] = g[grp]
    cfgs["drop_low_shap10"] = LOW_SHAP_FEATURES
    cfgs["drop_rhythm_low_shap10"] = list(dict.fromkeys(g["rhythm"] + LOW_SHAP_FEATURES))  # bộ gọn ứng viên (30)
    cfgs[CONTROL] = []  # đủ 41 đặc trưng nhưng xáo thứ tự dòng train -> mức nhiễu của riêng thứ tự dòng
    return cfgs


def _done(path: Path) -> set:
    if not path.is_file():
        return set()
    d = pd.read_csv(path, usecols=["run_id", "ablation", "model"])
    return set(map(tuple, d.to_numpy()))


def _append(path: Path, rows: list) -> None:
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.is_file(), index=False, encoding="utf-8")


def run(args) -> None:
    dev = load_grid_config(args.dev_grid)
    frozen = load_grid_config(args.frozen_grid)
    params = load_params("configs/model_params.yaml")
    c0 = float(frozen.get("contamination", 0.05))
    origin = schema_origin(ROOT / "configs/feature_schema.yaml")
    core = list(origin)
    cfgs = ablation_configs(origin)
    if args.only:
        cfgs = {k: v for k, v in cfgs.items() if k in set(args.only)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_configs.json").write_text(json.dumps(
        {k: {"dropped": v, "n_features": len(core) - len(v)} for k, v in ablation_configs(origin).items()},
        indent=2, ensure_ascii=False), encoding="utf-8")
    runs_csv, scen_csv = out / "runs.csv", out / "scenario_runs.csv"
    done = _done(runs_csv)

    for i, (run_dir, mseed) in enumerate(zip(dev["runs"], dev["model_seeds"])):
        run_id = Path(run_dir).name
        todo = [(a, m) for a in cfgs for m in ML if (run_id, a, m) not in done]
        if not todo:
            continue
        data = load_run_data(run_dir, frozen.get("segment"), sort_rows=True)
        for a, m in todo:
            keep = [f for f in core if f not in set(cfgs[a])]
            c = frozen["models"][m][0]
            d = data
            if a == CONTROL:
                d = {**data, "train": data["train"].sample(frac=1.0, random_state=int(mseed)).reset_index(drop=True)}
            res = fit_and_score(d, m, _params_with(params, m, c.get("params") or {}), c.get("scaler", "robust"),
                                int(mseed), c0, feature_names=keep)
            base = {"run_id": run_id, "seed_index": i, "ablation": a, "model": m}
            met = labeled_metrics(data["y"], res["scores"], data["days"], ks=[50, 100], budget_ratio=c0,
                                  daily_budgets=[100])
            _append(runs_csv, [{**base, "n_features": len(keep), **{k: met[k] for k in OVERALL},
                                "fit_seconds": round(res["fit_seconds"] + res["score_seconds"], 1)}])
            det = pd.DataFrame(scenario_detection_rows(data["y"], data["scenario"].astype(object), res["scores"],
                                                       res["flags"], data["days"], daily_ks=(100,)))
            auc = pd.DataFrame(_scenario_rows(data, res["scores"], {}))[["scenario", "pr_auc", "roc_auc"]]
            sc = det[["scenario", "recall_top100_day"]].merge(auc, on="scenario", how="left")
            _append(scen_csv, [{**base, **r} for r in sc.to_dict(orient="records")])
            logger.info("run %d/%d %s  %-20s %-22s PR-AUC %.3f  (%.0fs)", i + 1, len(dev["runs"]), run_id, a, m,
                        met["pr_auc"], res["fit_seconds"] + res["score_seconds"])


def paired_delta(df: pd.DataFrame, keys: list, metrics: list) -> pd.DataFrame:
    """Hiệu (cấu hình − full) trên cùng run + mô hình (+ kịch bản), rồi TB ± std qua run."""
    full = df[df["ablation"] == "full"].drop(columns="ablation")
    d = df[df["ablation"] != "full"].merge(full, on=["run_id", "seed_index", *keys], suffixes=("", "_full"))
    for m in metrics:
        d[f"d_{m}"] = d[m] - d[f"{m}_full"]
    g = d.groupby(["ablation", *keys], sort=False)
    res = pd.concat([g[[f"{m}_full" for m in metrics]].mean(),
                     g[[f"d_{m}" for m in metrics]].mean().add_suffix("_mean"),
                     g[[f"d_{m}" for m in metrics]].std(ddof=1).add_suffix("_std"),
                     g.size().rename("n_runs")], axis=1).reset_index()
    for m in metrics:  # rõ = |hiệu TB| vượt 1 std của hiệu (cùng quy ước "1 độ lệch chuẩn" của các tuần trước)
        res[f"d_{m}_clear"] = res[f"d_{m}_mean"].abs() > res[f"d_{m}_std"]
    return res


def analyze(out: Path) -> None:
    runs = pd.read_csv(out / "runs.csv")
    scen = pd.read_csv(out / "scenario_runs.csv")
    summarize(runs, ["ablation", "model"], OVERALL + ["n_features"]).to_csv(
        out / "summary.csv", index=False, encoding="utf-8-sig")
    summarize(scen, ["ablation", "model", "scenario"], SCEN).to_csv(
        out / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    dov = paired_delta(runs.drop(columns=["n_features", "fit_seconds"]), ["model"], OVERALL)
    dov.round(4).to_csv(out / "delta_overall.csv", index=False, encoding="utf-8-sig")
    dsc = paired_delta(scen, ["model", "scenario"], SCEN)
    dsc.round(4).to_csv(out / "delta_scenario.csv", index=False, encoding="utf-8-sig")
    plot(dsc, out)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(dov[["ablation", "model", "pr_auc_full", "d_pr_auc_mean", "d_pr_auc_std",
                   "recall_at_100_per_day_full", "d_recall_at_100_per_day_mean", "d_recall_at_100_per_day_std",
                   "d_precision_at_50_mean"]].round(3).to_string(index=False))


def plot(dsc: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "figures").mkdir(parents=True, exist_ok=True)
    abl = list(dict.fromkeys(dsc["ablation"]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, m in zip(axes, ML):
        piv = dsc[dsc["model"] == m].pivot(index="ablation", columns="scenario",
                                            values="d_recall_top100_day_mean").reindex(abl)
        clear = dsc[dsc["model"] == m].pivot(index="ablation", columns="scenario",
                                              values="d_recall_top100_day_clear").reindex(abl)
        cols = [c for c in piv.columns if c != "all"] + ["all"]
        piv, clear = piv[cols], clear[cols]
        im = ax.imshow(piv.to_numpy(), cmap="RdBu", vmin=-0.4, vmax=0.4, aspect="auto")
        ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(abl)), abl, fontsize=8)
        for (r, c), v in np.ndenumerate(piv.to_numpy()):
            ax.text(c, r, f"{v:+.2f}" + ("*" if clear.iat[r, c] else ""), ha="center", va="center", fontsize=7)
        ax.set_title(f"{SHORT[m]}: Δ recall top-100/ngày so với đủ 41 đặc trưng")
    fig.colorbar(im, ax=axes, fraction=0.015)
    fig.text(0.01, -0.12, "* = |Δ TB| > 1 std của Δ qua 5 run dev; control_shuffle_rows = mức nhiễu do thứ tự dòng",
             fontsize=8)
    fig.savefig(out / "figures" / "delta_recall_by_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dev-grid", default="configs/benchmark_grid.yaml")
    p.add_argument("--frozen-grid", default="configs/benchmark_grid_test.yaml")
    p.add_argument("--only", nargs="*", help="chỉ chạy các cấu hình này (luôn nên có 'full')")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--out", default="experiments/ablation/feature_groups")
    args = p.parse_args()
    if not args.analyze_only:
        run(args)
    analyze(Path(args.out))


if __name__ == "__main__":
    main()
