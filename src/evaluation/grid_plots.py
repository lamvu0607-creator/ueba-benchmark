"""Hình của lưới thí nghiệm tuần 4 (src/evaluation/grid.py): PR-AUC ± std, đường PR, contamination, kịch bản."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {
    "isolation_forest": "Isolation Forest", "local_outlier_factor": "LOF", "one_class_svm": "One-Class SVM",
    "zscore_baseline": "Z-score (global)", "rule_threshold_baseline": "Rule (6 luật)",
    "failure_count_baseline": "Rule (thất bại ≥ 5)", "random_baseline": "Random",
}
COLORS = {
    "isolation_forest": "#1f77b4", "local_outlier_factor": "#2ca02c", "one_class_svm": "#d62728",
    "zscore_baseline": "#7f7f7f", "rule_threshold_baseline": "#bcbd22",
    "failure_count_baseline": "#8c564b", "random_baseline": "#c7c7c7",
}


def _save(fig, out: Path, name: str, formats: Sequence[str]) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def plot_pr_curves(pr_df: pd.DataFrame, summary: pd.DataFrame, best: Dict[str, str], out: Path,
                   positive_rate: float, formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    """Đường PR trung bình qua các seed (dải ±1 std cho mô hình ML); chú thích gọn, đặt dưới khung vẽ."""
    fig, ax = plt.subplots(figsize=(8, 5.4))
    n_seeds = int(pr_df["n_seeds"].max()) if "n_seeds" in pr_df else 1
    for (model, cid), g in pr_df.groupby(["model", "config_id"], sort=False):
        auc = summary.loc[(summary["model"] == model) & (summary["config_id"] == cid), "pr_auc_mean"]
        is_ml = model in best
        color = COLORS.get(model)
        ax.plot(g["recall"], g["precision"], color=color, lw=1.8 if is_ml else 1.1, ls="-" if is_ml else "--",
                label=f"{LABELS.get(model, model)} ({float(auc.iloc[0]):.3f})")
        if is_ml and "precision_std" in g and g["precision_std"].notna().any():
            lo = np.clip(g["precision"] - g["precision_std"], positive_rate / 3, 1.0)
            hi = np.clip(g["precision"] + g["precision_std"], positive_rate / 3, 1.0)
            ax.fill_between(g["recall"], lo, hi, color=color, alpha=0.15, lw=0)
    ax.axhline(positive_rate, ls=":", color="k", lw=1, label=f"Tỷ lệ dương ({positive_rate:.2%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision (thang log)")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(max(positive_rate / 3, 1e-4), 1.0)
    ax.grid(True, which="major", alpha=0.3)
    ax.set_title(f"Đường Precision–Recall, trung bình {n_seeds} seed (dải = ±1 std)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False,
              title="Mô hình (PR-AUC trung bình)", title_fontsize=8)
    return _save(fig, out, "pr_curves", formats)


SHORT = {"isolation_forest": "IF", "local_outlier_factor": "LOF", "one_class_svm": "OCSVM"}


def plot_contamination(cont_summary: pd.DataFrame, out: Path, formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    """Lưới 2×2 (tỷ lệ cảnh báo, precision, recall, PR-AUC) theo contamination; chú thích chung dưới hình."""
    metrics = [("alert_rate", "Tỷ lệ cảnh báo thực tế"), ("precision", "Precision tại ngưỡng"),
               ("recall", "Recall tại ngưỡng"), ("pr_auc", "PR-AUC")]
    c = np.sort(cont_summary["contamination"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
    for ax, (m, title) in zip(axes.flat, metrics):
        for model, g in cont_summary.groupby("model"):
            g = g.sort_values("contamination")
            ax.errorbar(g["contamination"], g[f"{m}_mean"], yerr=g[f"{m}_std"].fillna(0), marker="o", ms=4,
                        capsize=3, color=COLORS.get(model), label=SHORT.get(model, LABELS.get(model, model)))
        if m == "alert_rate":
            ax.plot(c, c, ls=":", color="k", lw=1, label="= contamination")
        ax.set_xscale("log")
        ax.set_xticks(c, [f"{v:.1%}".replace(".0%", "%") for v in c], fontsize=8)
        ax.minorticks_off()
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("contamination (ngân sách cảnh báo trên train)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Độ nhạy contamination (trung bình ± std qua các seed)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, "contamination_sensitivity", formats)


def plot_grid(summary: pd.DataFrame, scen_summary: pd.DataFrame, cont_summary: pd.DataFrame, pr_df: pd.DataFrame,
              best: Dict[str, str], out: Path, positive_rate: float,
              formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    figures: List[Path] = []

    # 1. PR-AUC trung bình ± std của mọi cấu hình (ML) và baseline
    s = summary.sort_values(["kind", "model", "pr_auc_mean"], ascending=[False, True, False])
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(s) + 1.2))
    y = np.arange(len(s))
    ax.barh(y, s["pr_auc_mean"], xerr=s["pr_auc_std"].fillna(0), color=[COLORS.get(m, "#999") for m in s["model"]],
            capsize=3, alpha=0.9)
    names = [LABELS.get(m, m) if k == "baseline" else c for k, m, c in zip(s["kind"], s["model"], s["config_id"])]
    ax.set_yticks(y, [n + (" ★" if best.get(m) == c else "") for n, m, c in zip(names, s["model"], s["config_id"])])
    ax.invert_yaxis()
    ax.axvline(positive_rate, ls="--", color="k", lw=1, label=f"tỷ lệ dương = {positive_rate:.3%}")
    ax.set_xlabel(f"PR-AUC (trung bình ± độ lệch chuẩn, n = {int(s['pr_auc_n'].max())} seed)")
    ax.set_title("PR-AUC theo cấu hình — ★ = cấu hình tốt nhất của mô hình")
    ax.legend(loc="lower right")
    figures += _save(fig, out, "grid_pr_auc", formats)

    # 2. Đường PR trung bình ± std của cấu hình tốt nhất + baseline
    figures += plot_pr_curves(pr_df, summary, best, out, positive_rate, formats)

    # 3. Độ nhạy contamination: alert rate thực tế, precision, recall, PR-AUC tại ngưỡng train
    figures += plot_contamination(cont_summary, out, formats)

    # 4. ROC-AUC theo kịch bản của cấu hình tốt nhất + baseline
    keep = [(m, c) for m, c in best.items()] + [(m, "default") for m in scen_summary.loc[
        scen_summary["kind"] == "baseline", "model"].unique()]
    sub = pd.concat([scen_summary[(scen_summary["model"] == m) & (scen_summary["config_id"] == c)] for m, c in keep])
    piv = sub.pivot_table(index="scenario", columns="model", values="roc_auc_mean", sort=False)
    piv = piv[[m for m, _ in keep if m in piv.columns]]
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 1.5))
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [LABELS.get(m, m) for m in piv.columns], rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for (r, c), v in np.ndenumerate(piv.to_numpy()):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="ROC-AUC trung bình")
    ax.set_title("ROC-AUC theo kịch bản (cấu hình tốt nhất)")
    figures += _save(fig, out, "scenario_roc_auc", formats)
    return figures
