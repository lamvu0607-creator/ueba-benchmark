"""
Dựng lại riêng đường PR trung bình ± std của một lưới đã chạy, không chạy lại cả lưới:
python scripts/evaluation/rebuild_pr_curves.py --grid-dir experiments/grid/test

Đọc cấu hình + cấu hình tốt nhất đã chốt trong ``grid_manifest.json``, refit đúng cấu hình đó và các baseline
trên mọi seed (cùng run tiêm, cùng random_state), ghi đè ``pr_curves.csv`` và ``figures/pr_curves.*``.
Không ghi vào nhật ký thí nghiệm (các lần fit này trùng các lần fit đã ghi của lưới).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.grid import _params_with, fit_and_score, load_run_data, mean_pr_curve  # noqa: E402
from src.evaluation.grid_plots import plot_pr_curves  # noqa: E402
from src.models.registry import load_params  # noqa: E402

logger = logging.getLogger("ueba_benchmark.grid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild mean ± std PR curves of a finished grid")
    parser.add_argument("--grid-dir", default="experiments/grid/test")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")

    out = Path(args.grid_dir)
    manifest = json.loads((out / "grid_manifest.json").read_text(encoding="utf-8"))
    cfg, best, split_day = manifest["config"], manifest["best_configs"], manifest["split_day"]
    params = load_params(manifest["params_path"])
    conf = {c["id"]: c for confs in cfg["models"].values() for c in confs}
    jobs = [(m, cid, conf[cid].get("scaler", "robust"), conf[cid].get("params") or {}) for m, cid in best.items()]
    jobs += [(b, "default", None, {}) for b in cfg.get("baselines") or []]
    c0 = float(cfg.get("contamination", 0.05))

    labels, scores = [], {f"{m}|{cid}": [] for m, cid, _, _ in jobs}
    for i, (run_dir, mseed) in enumerate(zip(cfg["runs"], cfg["model_seeds"])):
        data = load_run_data(run_dir, cfg.get("segment"), split_day)
        labels.append(data["y"])
        logger.info("[pr] seed %d/%d: %s, model seed %s", i + 1, len(cfg["runs"]), data["run_id"], mseed)
        for model, cid, scaler, override in jobs:
            res = fit_and_score(data, model, _params_with(params, model, override), scaler, int(mseed), c0)
            scores[f"{model}|{cid}"].append(res["scores"])

    pr_df = pd.concat([mean_pr_curve(labels, s).assign(model=k.split("|")[0], config_id=k.split("|")[1])
                       for k, s in scores.items()], ignore_index=True)
    pr_df = pr_df[["model", "config_id", "recall", "precision", "precision_std", "n_seeds"]]
    pr_df.to_csv(out / "pr_curves.csv", index=False, float_format="%.6g")
    summary = pd.read_csv(out / "grid_summary.csv")
    positive_rate = float(pd.concat([pd.Series(y) for y in labels]).mean())
    figs = plot_pr_curves(pr_df, summary, best, out / "figures", positive_rate, args.formats)
    logger.info("[pr] đã ghi %s và %s", out / "pr_curves.csv", ", ".join(map(str, figs)))


if __name__ == "__main__":
    main()
