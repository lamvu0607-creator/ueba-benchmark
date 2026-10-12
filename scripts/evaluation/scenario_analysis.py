"""
Phân tích theo loại bất thường (tuần 5): chấm lại 5 run test với cấu hình chốt, lưu điểm từng dòng và tính
recall theo ngân sách cảnh báo cho từng kịch bản, tính bổ trợ OCSVM–IF và danh sách nạn nhân bị mọi mô hình bỏ sót.

    python scripts/evaluation/scenario_analysis.py                       # chấm + phân tích
    python scripts/evaluation/scenario_analysis.py --reuse-scores        # chỉ phân tích lại từ scores.parquet

Đầu ra: experiments/analysis/scenario/
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

from src.evaluation.grid import (  # noqa: E402
    _params_with, _scenario_rows, fit_and_score, load_grid_config, load_run_data, run_split_day, summarize,
)
from src.evaluation.scenario import daily_topk_mask, scenario_detection_rows  # noqa: E402
from src.models.registry import load_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("ueba_benchmark.scenario_analysis")

ML = ["one_class_svm", "isolation_forest", "local_outlier_factor"]
SHORT = {"one_class_svm": "OCSVM", "isolation_forest": "IF", "local_outlier_factor": "LOF",
         "zscore_baseline": "z-score", "rule_threshold_baseline": "luật 6 đk",
         "failure_count_baseline": "luật thất bại", "random_baseline": "ngẫu nhiên"}
DAILY_K = 100


def score_runs(cfg: dict, params_path: str):
    params = load_params(params_path)
    c0 = float(cfg.get("contamination", 0.05))
    jobs = [(m, c["id"], c.get("scaler", "robust"), c.get("params") or {}) for m, cs in cfg["models"].items() for c in cs]
    jobs += [(b, "default", None, {}) for b in cfg.get("baselines") or []]
    parts, auc_rows = [], []
    for i, (run_dir, mseed) in enumerate(zip(cfg["runs"], cfg["model_seeds"])):
        data = load_run_data(run_dir, cfg.get("segment"), run_split_day(run_dir), sort_rows=True)
        keys = data["eval"][["DomainName", "UserName", "day"]].reset_index(drop=True)
        for model, cid, scaler, override in jobs:
            res = fit_and_score(data, model, _params_with(params, model, override), scaler, int(mseed), c0)
            parts.append(keys.assign(run_id=data["run_id"], seed_index=i, model=model, config_id=cid,
                                     y=data["y"], scenario=data["scenario"], score=res["scores"],
                                     flag=res["flags"]))
            auc_rows += _scenario_rows(data, res["scores"], {"seed_index": i, "model": model})
            logger.info("run %d/%d %s  %-24s xong (%.0fs)", i + 1, len(cfg["runs"]), data["run_id"], model,
                        res["fit_seconds"] + res["score_seconds"])
    scores = pd.concat(parts, ignore_index=True)
    return scores, pd.DataFrame(auc_rows)


def detection_table(scores: pd.DataFrame, auc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, model), g in scores.groupby(["run_id", "model"], sort=False):
        for r in scenario_detection_rows(g["y"].to_numpy(), g["scenario"].to_numpy(dtype=object),
                                         g["score"].to_numpy(), g["flag"].to_numpy(), g["day"].to_numpy(),
                                         daily_ks=(10, DAILY_K)):
            rows.append({"run_id": run, "model": model, **r})
    det = summarize(pd.DataFrame(rows), ["model", "scenario"],
                    ["recall_threshold", "recall_top10_day", f"recall_top{DAILY_K}_day", "recall_top1pct",
                     "median_rank_pct"])
    if not auc.empty:
        det = det.merge(summarize(auc, ["model", "scenario"], ["roc_auc", "pr_auc"]), on=["model", "scenario"],
                        how="left")
    return det


def caught_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    """Mỗi nạn nhân (run, tài khoản, ngày) × mô hình ML: có nằm trong top-K/ngày không."""
    keys = ["run_id", "DomainName", "UserName", "day", "scenario"]
    per_model = {}
    for (run, model), g in scores[scores["model"].isin(ML)].groupby(["run_id", "model"], sort=False):
        m = daily_topk_mask(g["score"].to_numpy(), g["day"].to_numpy(), DAILY_K)
        pos = (g["y"] == 1).to_numpy()
        per_model.setdefault(model, []).append(g.loc[pos, keys].assign(**{model: m[pos]}))
    caught = None
    for model, parts in per_model.items():
        df = pd.concat(parts, ignore_index=True)
        caught = df if caught is None else caught.merge(df, on=keys, how="outer")
    if caught[ML].isna().any(axis=None):
        raise ValueError("Nạn nhân không khớp giữa các mô hình")
    caught[ML] = caught[ML].astype(bool)
    return caught


def complementarity(caught: pd.DataFrame) -> pd.DataFrame:
    o, i = caught["one_class_svm"], caught["isolation_forest"]
    c = caught.assign(only_ocsvm=o & ~i, only_if=i & ~o, both=o & i, neither_ocsvm_if=~o & ~i,
                      any_ml=caught[ML].any(axis=1), none_ml=~caught[ML].any(axis=1))
    cols = ["only_ocsvm", "only_if", "both", "neither_ocsvm_if", "any_ml", "none_ml"]
    by = c.groupby("scenario")[cols].mean()
    by.loc["all"] = c[cols].mean()
    n = c.groupby("scenario").size()
    n.loc["all"] = len(c)
    return by.assign(n_victims=n.reindex(by.index)).reset_index()


def missed_victims(caught: pd.DataFrame, runs: list) -> pd.DataFrame:
    missed = caught[~caught[ML].any(axis=1)].copy()
    man = pl.concat([pl.read_csv(Path(r) / "injection_manifest.csv").select(
        pl.lit(Path(r).name).alias("run_id"), "UserName", "day", "scenario", "n_events", "params") for r in runs
        if (Path(r) / "injection_manifest.csv").is_file()])
    if man.height == 0:
        return missed
    return missed.merge(man.to_pandas(), on=["run_id", "UserName", "day", "scenario"], how="left")


def plot(det: pd.DataFrame, scores: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluation.scenario import rank_percentile

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = det[det["scenario"] != "all"]
    models = [m for m in SHORT if m in set(d["model"])]
    col = f"recall_top{DAILY_K}_day_mean"
    piv = d.pivot(index="model", columns="scenario", values=col).reindex(models)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=25, ha="right")
    ax.set_yticks(range(piv.shape[0]), [SHORT[m] for m in piv.index])
    for (r, c), v in np.ndenumerate(piv.to_numpy()):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else "black", fontsize=9)
    ax.set_title(f"Recall trong top-{DAILY_K} cảnh báo/ngày theo kịch bản (TB 5 run test)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "scenario_recall_heatmap.png", dpi=150)
    plt.close(fig)

    rows = []
    for (run, model), g in scores[scores["model"].isin(ML)].groupby(["run_id", "model"], sort=False):
        pct = rank_percentile(g["score"].to_numpy())
        pos = (g["y"] == 1).to_numpy()
        rows.append(pd.DataFrame({"model": model, "scenario": g["scenario"].to_numpy()[pos], "pct": pct[pos]}))
    r = pd.concat(rows)
    scen = sorted(r["scenario"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.26
    for j, m in enumerate(ML):
        data = [r[(r["model"] == m) & (r["scenario"] == s)]["pct"].to_numpy() * 100 for s in scen]
        bp = ax.boxplot(data, positions=np.arange(len(scen)) + (j - 1) * w, widths=w * 0.9, patch_artist=True,
                        showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(["#0e7490", "#f59e0b", "#94a3b8"][j])
        ax.plot([], [], color=["#0e7490", "#f59e0b", "#94a3b8"][j], lw=6, label=SHORT[m])
    ax.set_yscale("log")
    ax.set_xticks(range(len(scen)), scen, rotation=20, ha="right")
    ax.set_ylabel("Hạng của nạn nhân (% từ trên xuống, log)")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Nạn nhân nằm ở top bao nhiêu % danh sách (càng cao càng tốt)")
    fig.tight_layout()
    fig.savefig(fig_dir / "victim_rank_boxplot.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Phân tích theo loại bất thường trên khối test")
    p.add_argument("--config", default="configs/benchmark_grid_test.yaml")
    p.add_argument("--model-params", default="configs/model_params.yaml")
    p.add_argument("--out", default="experiments/analysis/scenario")
    p.add_argument("--reuse-scores", action="store_true")
    args = p.parse_args()

    cfg = load_grid_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse_scores:
        scores = pd.read_parquet(out / "scores.parquet")
        auc = pd.read_csv(out / "scenario_auc_runs.csv")
    else:
        scores, auc = score_runs(cfg, args.model_params)
        scores.to_parquet(out / "scores.parquet", index=False)
        auc.to_csv(out / "scenario_auc_runs.csv", index=False)

    det = detection_table(scores, auc)
    det.to_csv(out / "scenario_detection.csv", index=False)
    caught = caught_matrix(scores)
    comp = complementarity(caught)
    comp.to_csv(out / "complementarity.csv", index=False)
    missed = missed_victims(caught, cfg["runs"])
    missed.to_csv(out / "missed_victims.csv", index=False)
    plot(det, scores, out)
    (out / "analysis_manifest.json").write_text(json.dumps({
        "config": args.config, "runs": cfg["runs"], "model_seeds": cfg["model_seeds"], "daily_k": DAILY_K,
        "sorted_rows": True, "n_scores": int(len(scores))}, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Đã ghi %s", out)


if __name__ == "__main__":
    main()
