"""
Vẽ đường cong chọn K (số đặc trưng SHAP thấp bị bỏ thêm sau rhythm) từ experiments/ablation/kcurve/.

    python scripts/evaluation/plot_kcurve.py

Trục x = số đặc trưng còn lại; mỗi điểm = TB qua 5 run dev, râu = std của hiệu ghép cặp so với cấu hình
tốt nhất của mô hình đó (đúng thước đo dùng cho quy tắc chọn K). Bộ 30 cũ (danh sách 10 từ 1 run) vẽ riêng.

Đầu ra: experiments/analysis/.../figures/kcurve.png + bản sao --copy-to.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SHORT = {"one_class_svm": "OCSVM", "isolation_forest": "IF", "local_outlier_factor": "LOF"}
COLOR = {"one_class_svm": "#2a78d6", "isolation_forest": "#eb6834", "local_outlier_factor": "#1baf7a"}
CURVE = ["full", "drop_rhythm", "drop_rhythm_k5", "drop_rhythm_k10", "drop_rhythm_k15", "drop_rhythm_k18"]
OLD30 = "drop_rhythm_low_shap10"
METRICS = {"recall_at_100_per_day": "Recall top-100/ngày (chỉ số chính)", "pr_auc": "PR-AUC"}


def paired_to_best(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mỗi (mô hình, cấu hình): TB metric, và TB ± std của hiệu ghép cặp so với cấu hình tốt nhất."""
    rows = []
    for m, g in runs.groupby("model"):
        p = g.pivot(index="run_id", columns="ablation", values=metric)
        best = p.mean().idxmax()
        d = p.sub(p[best], axis=0)
        for a in p.columns:
            rows.append({"model": m, "ablation": a, "mean": p[a].mean(), "d_best_mean": d[a].mean(),
                         "d_best_std": d[a].std(ddof=1), "best": best})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dir", default="experiments/ablation/kcurve")
    p.add_argument("--copy-to", default="reports/week5/figures/13-kcurve.png")
    args = p.parse_args()
    d = Path(args.dir)
    runs = pd.read_csv(d / "runs.csv")
    nfeat = runs.groupby("ablation")["n_features"].first()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    tables = []
    for ax, (metric, title) in zip(axes, METRICS.items()):
        t = paired_to_best(runs, metric).assign(metric=metric)
        tables.append(t)
        for m in SHORT:
            s = t[t["model"] == m].set_index("ablation")
            x = [nfeat[a] for a in CURVE]
            ax.errorbar(x, s.loc[CURVE, "mean"], yerr=s.loc[CURVE, "d_best_std"], color=COLOR[m], marker="o",
                        lw=2, capsize=4, label=SHORT[m])
            ax.plot(nfeat[OLD30], s.loc[OLD30, "mean"], marker="*", ms=14, color=COLOR[m], mec="black", ls="none")
            b = s["best"].iloc[0]
            ax.plot(nfeat[b], s.loc[b, "mean"], marker="o", ms=13, mfc="none", mec="red", mew=2, ls="none")
        ax.set_xticks([41, 37, 33, 30, 29, 25, 22])
        ax.invert_xaxis()
        ax.set_xlabel("số đặc trưng còn lại  (41 → bỏ rhythm 37 → K = 5 / 10 / 15 / 18)")
        ax.set_title(title)
        ax.grid(axis="y", color="#ddd")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Chọn K: bỏ rhythm + K đặc trưng SHAP thấp nhất (TB 5 run dev)")
    fig.text(0.01, -0.1, "Râu = std của hiệu ghép cặp so với cấu hình tốt nhất của mô hình (vòng đỏ). "
             "★ = bộ 30 hiện tại (rhythm + 10 đặc trưng SHAP thấp lấy từ 1 run, danh sách khác K = 10).",
             fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = d / "figures" / "kcurve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.concat(tables).round(4).to_csv(d / "kcurve_vs_best.csv", index=False, encoding="utf-8-sig")
    if args.copy_to:
        Path(args.copy_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, args.copy_to)


if __name__ == "__main__":
    main()
