"""
Heatmap PR-AUC theo (mô hình × kịch bản) từ ``grid_scenarios_summary.csv`` của lưới test.

    python scripts/evaluation/plot_scenario_heatmap.py

- PR-AUC theo kịch bản = nạn nhân của kịch bản đó so với MỌI dòng bình thường (``grid._scenario_rows``), nên mức
  ngẫu nhiên ≈ tỷ lệ nạn nhân của kịch bản (~0,0018), không phải ~0,01 như PR-AUC tổng. Ô ghi PR-AUC và bội số so với
  mức ngẫu nhiên (lift = PR-AUC / PR-AUC của random_baseline cùng kịch bản). Thang màu log.
- Hình phải: Δ PR-AUC (bộ 30 − bộ 41 của lưới tuần 4) cho 3 mô hình ML; baseline không đổi đặc trưng.
- Viền đen: mô hình tốt nhất của từng kịch bản.

Đầu ra: ``<main>/figures/scenario_pr_auc_heatmap.png`` + ``scenario_pr_auc_table.csv``, và bản sao ``--copy-to`` (đường dẫn file).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

MODELS = {"one_class_svm": "OCSVM", "isolation_forest": "IF", "local_outlier_factor": "LOF",
          "zscore_baseline": "z-score", "failure_count_baseline": "luật thất bại ≥ 5",
          "rule_threshold_baseline": "luật 6 điều kiện", "random_baseline": "ngẫu nhiên"}
ML = ["OCSVM", "IF", "LOF"]
SCEN = ["brute_force", "logon_type_switch", "new_workstation_burst", "dormant_wakeup", "password_spraying",
        "off_hours"]


def pivot(path: Path, col: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["model"] = d["model"].map(MODELS)
    return d.pivot_table(index="model", columns="scenario", values=col).reindex(index=list(MODELS.values()),
                                                                                 columns=SCEN)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--main", default="experiments/grid/test_30f", help="lưới chính (bộ 30 đặc trưng)")
    p.add_argument("--ref", default="experiments/grid/test", help="lưới so sánh (41 đặc trưng, tuần 4)")
    p.add_argument("--copy-to", default="")
    args = p.parse_args()
    main_dir, ref_dir = Path(args.main), Path(args.ref)

    pr = pivot(main_dir / "grid_scenarios_summary.csv", "pr_auc_mean")
    sd = pivot(main_dir / "grid_scenarios_summary.csv", "pr_auc_std")
    ref = pivot(ref_dir / "grid_scenarios_summary.csv", "pr_auc_mean")
    lift = pr / pr.loc["ngẫu nhiên"]
    delta = (pr - ref).loc[ML]

    table = pd.concat({"pr_auc_mean": pr, "pr_auc_std": sd, "lift_vs_random": lift, "pr_auc_41f": ref,
                       "delta_vs_41f": pr - ref}, names=["metric"]).round(4)
    table.to_csv(main_dir / "scenario_pr_auc_table.csv", encoding="utf-8-sig")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [6, 3.6]})
    im = ax.imshow(pr.to_numpy(), cmap="viridis", norm=LogNorm(vmin=1e-3, vmax=0.3), aspect="auto")
    for (r, c), v in np.ndenumerate(pr.to_numpy()):
        dark = v < 0.02
        ax.text(c, r, f"{v:.3f}\n×{lift.iat[r, c]:.0f}", ha="center", va="center", fontsize=8,
                color="white" if dark else "black")
    best = pr.idxmax()
    for c, s in enumerate(SCEN):
        r = list(pr.index).index(best[s])
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
    ax.axhline(2.5, color="white", lw=2)
    ax.set_xticks(range(len(SCEN)), SCEN, rotation=25, ha="right")
    ax.set_yticks(range(len(pr)), pr.index)
    ax.set_title("PR-AUC theo kịch bản, bộ 30 đặc trưng (5 run test)\n"
                 "ô: PR-AUC và ×lần so với ngẫu nhiên · viền đỏ: tốt nhất mỗi kịch bản")
    fig.colorbar(im, ax=ax, fraction=0.03, label="PR-AUC (thang log)")

    lim = float(np.nanmax(np.abs(delta.to_numpy())))
    im2 = ax2.imshow(delta.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for (r, c), v in np.ndenumerate(delta.to_numpy()):
        ax2.text(c, r, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                 color="white" if abs(v) > 0.6 * lim else "black")
    ax2.set_xticks(range(len(SCEN)), SCEN, rotation=25, ha="right")
    ax2.set_yticks(range(len(ML)), ML)
    ax2.set_title("Δ PR-AUC: bộ 30 − bộ 41 (lưới tuần 4)")
    fig.colorbar(im2, ax=ax2, fraction=0.05)
    fig.text(0.01, -0.03, "PR-AUC theo kịch bản = nạn nhân của kịch bản so với mọi dòng bình thường; mức ngẫu nhiên "
             "≈ 0,0018 (tỷ lệ nạn nhân của một kịch bản).", fontsize=8)
    fig.tight_layout()

    out = main_dir / "figures" / "scenario_pr_auc_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if args.copy_to:
        Path(args.copy_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, args.copy_to)
    print(table.loc["pr_auc_mean"].round(3).to_string())
    print(table.loc["lift_vs_random"].round(0).to_string())
    print(table.loc["delta_vs_41f"].loc[ML].round(3).to_string())


if __name__ == "__main__":
    main()
