"""
Recall theo ngân sách cảnh báo/ngày trên TOÀN danh sách xếp hạng, theo kịch bản: phân biệt "mô hình không thấy
nạn nhân" (đường bám đường ngẫu nhiên) với "ngân sách top-100 chưa đủ" (đường vọt lên ngay sau k = 100).

    python scripts/evaluation/plot_scenario_budget.py      # đọc experiments/analysis/scenario/scores.parquet

Hạng trong ngày dùng cùng quy ước với ``daily_topk_mask``: điểm giảm dần, hoà điểm giữ thứ tự dòng.
Đầu ra (cùng thư mục):
  * ``scenario_recall_budget.csv``  — recall@k/ngày, mỗi (mô hình, kịch bản, k), trung bình ± std 5 run,
  * ``scenario_k50.csv``            — số cảnh báo/ngày cần để bắt 50% / 80% nạn nhân (trung vị hạng trong ngày),
  * ``figures/scenario_recall_budget.png``, ``figures/scenario_k50_heatmap.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODELS = ["one_class_svm", "isolation_forest", "local_outlier_factor", "zscore_baseline",
          "rule_threshold_baseline", "failure_count_baseline", "random_baseline"]
SHORT = {"one_class_svm": "OCSVM", "isolation_forest": "IF", "local_outlier_factor": "LOF",
         "zscore_baseline": "z-score", "rule_threshold_baseline": "luật 6 đk",
         "failure_count_baseline": "luật thất bại", "random_baseline": "ngẫu nhiên"}
STYLE = {"one_class_svm": ("#d62728", "-", 2.0), "isolation_forest": ("#1f77b4", "-", 2.0),
         "local_outlier_factor": ("#2ca02c", "-", 2.0), "zscore_baseline": ("#7f7f7f", "--", 1.3),
         "rule_threshold_baseline": ("#bcbd22", "--", 1.3), "failure_count_baseline": ("#8c564b", "--", 1.3),
         "random_baseline": ("#9ca3af", ":", 1.5)}
SCEN = {"brute_force": "Brute-force", "logon_type_switch": "Đổi LogonType",
        "new_workstation_burst": "Máy trạm mới", "dormant_wakeup": "Ngủ đông", "off_hours": "Ngoài giờ",
        "password_spraying": "Spraying"}


def victim_day_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    """Hạng trong ngày (1 = điểm cao nhất) của mọi nạn nhân, mỗi (run, mô hình), kèm số dòng của ngày."""
    s = scores[["run_id", "model", "day", "y", "scenario", "score"]].copy()
    s["pos"] = np.arange(len(s))
    s = s.sort_values(["run_id", "model", "day", "score", "pos"], ascending=[True, True, True, False, True],
                      kind="stable")
    g = s.groupby(["run_id", "model", "day"], sort=False)
    s["rank"] = g.cumcount() + 1
    s["n_day"] = g["score"].transform("size")
    return s[s["y"] == 1][["run_id", "model", "day", "scenario", "rank", "n_day"]]


def recall_budget(v: pd.DataFrame, ks: np.ndarray) -> pd.DataFrame:
    rows = []
    for (run, model, sc), g in v.groupby(["run_id", "model", "scenario"]):
        r = np.sort(g["rank"].to_numpy())
        rec = np.searchsorted(r, ks, side="right") / r.size
        rows.append(pd.DataFrame({"run_id": run, "model": model, "scenario": sc, "k": ks, "recall": rec}))
    out = pd.concat(rows).groupby(["model", "scenario", "k"])["recall"].agg(["mean", "std"]).reset_index()
    return out.rename(columns={"mean": "recall_mean", "std": "recall_std"})


def k_needed(v: pd.DataFrame) -> pd.DataFrame:
    """Số cảnh báo/ngày cần để bắt 50% / 80% nạn nhân của kịch bản: phân vị hạng trong ngày, TB 5 run."""
    q = v.groupby(["run_id", "model", "scenario"])["rank"].quantile([0.5, 0.8]).unstack()
    q.columns = ["k50", "k80"]
    return q.groupby(["model", "scenario"]).mean().reset_index()


def plot_curves(rb: pd.DataFrame, n_day: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, sharey=True)
    for ax, (sc, title) in zip(axes.flat, SCEN.items()):
        for m in MODELS:
            g = rb[(rb["model"] == m) & (rb["scenario"] == sc)]
            if g.empty:
                continue
            c, ls, lw = STYLE[m]
            ax.plot(g["k"], g["recall_mean"], color=c, ls=ls, lw=lw, label=SHORT[m])
        ax.axvline(100, color="k", lw=1, ls="-.", alpha=0.7)
        ax.axvline(n_day * 0.05, color="#0e7490", lw=1, ls=":", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlim(10, n_day)
        ax.set_ylim(0, 1.02)
        ax.grid(True, which="major", alpha=0.3)
        ax.set_title(title, fontsize=12)
    for ax in axes[1]:
        ax.set_xlabel("Ngân sách k cảnh báo/ngày (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recall (tỷ lệ nạn nhân bắt được)")
    axes[0, 0].text(105, 0.96, "k = 100", fontsize=8, va="top")
    axes[0, 0].text(n_day * 0.05 * 1.05, 0.96, "5%/ngày", fontsize=8, va="top", color="#0e7490")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Recall theo ngân sách cảnh báo/ngày trên toàn danh sách (TB 5 run test, ~{n_day:,.0f} dòng/ngày)"
                 .replace(",", "."), fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_k50(kn: pd.DataFrame, n_day: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    piv = kn.pivot(index="model", columns="scenario", values="k50").reindex(index=MODELS, columns=list(SCEN))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn_r", norm=LogNorm(vmin=10, vmax=n_day), aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [SCEN[c] for c in piv.columns], rotation=20, ha="right")
    ax.set_yticks(range(piv.shape[0]), [SHORT[m] for m in piv.index])
    for (r, c), v in np.ndenumerate(piv.to_numpy()):
        ax.text(c, r, f"{v:,.0f}".replace(",", "."), ha="center", va="center", fontsize=9,
                fontweight="bold" if v <= 100 else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.035)
    cb.set_label("cảnh báo/ngày (log)")
    ax.set_title(f"Số cảnh báo/ngày cần để bắt 50% nạn nhân (≤ 100 in đậm; ngẫu nhiên ≈ {n_day / 2:,.0f})"
                 .replace(",", "."), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Recall theo ngân sách cảnh báo/ngày, theo kịch bản")
    p.add_argument("--dir", default="experiments/analysis/scenario")
    args = p.parse_args()
    out = Path(args.dir)
    scores = pd.read_parquet(out / "scores.parquet")
    v = victim_day_ranks(scores)
    n_day = float(scores.groupby(["run_id", "model", "day"]).size().mean())
    ks = np.unique(np.r_[np.round(np.geomspace(1, scores.groupby(["run_id", "model", "day"]).size().max(), 300)),
                         [100, 300, 500, 1000]]).astype(int)
    rb = recall_budget(v, ks)
    kn = k_needed(v)
    rb.to_csv(out / "scenario_recall_budget.csv", index=False, float_format="%.4g")
    kn.to_csv(out / "scenario_k50.csv", index=False, float_format="%.1f")
    (out / "figures").mkdir(exist_ok=True)
    plot_curves(rb, n_day, out / "figures" / "scenario_recall_budget.png")
    plot_k50(kn, n_day, out / "figures" / "scenario_k50_heatmap.png")
    print(kn.pivot(index="model", columns="scenario", values="k50").reindex(MODELS).round(0).to_string())
    print(kn.pivot(index="model", columns="scenario", values="k80").reindex(MODELS).round(0).to_string())
    sel = rb[rb["k"].isin([100, 300, 500, 1000])].pivot_table(index=["model", "k"], columns="scenario",
                                                         values="recall_mean")
    print(sel.reindex(MODELS, level=0).round(2).to_string())


if __name__ == "__main__":
    main()
