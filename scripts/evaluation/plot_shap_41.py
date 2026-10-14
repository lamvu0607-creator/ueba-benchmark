"""
Vẽ 41 đặc trưng xếp theo tỷ phần |SHAP| trên nạn nhân (lấy mức cao nhất qua OCSVM, IF, LOF) cho slide tuần 5.

    python scripts/evaluation/plot_shap_41.py

Đọc ``experiments/analysis/explain/feature_importance.csv`` (1 run dev, seed 42). Đỏ = 10 đặc trưng thấp nhất (ứng viên
cắt tỉa); cam = các đặc trưng rhythm còn lại (bị bỏ vì thuộc nhóm rhythm, không phải vì SHAP thấp); nét đứt = 1/41.

Đầu ra: reports/week5/figures/05-shap_41_dac_trung.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

N_LOW = 10


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--csv", default="experiments/analysis/explain/feature_importance.csv")
    p.add_argument("--out", default="reports/week5/figures/05-shap_41_dac_trung.png")
    args = p.parse_args()

    fi = pd.read_csv(args.csv)
    s = fi.groupby("feature").agg(share=("shap_share_victim", "max"), group=("group", "first"))
    s = s.sort_values("share")
    low = set(s.index[:N_LOW])
    colors = ["#c0392b" if f in low else "#e67e22" if g == "rhythm" else "#2a78d6"
              for f, g in zip(s.index, s["group"])]

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(range(len(s)), s["share"], color=colors)
    ax.set_yticks(range(len(s)), s.index, fontsize=7.5)
    for t, f, g in zip(ax.get_yticklabels(), s.index, s["group"]):
        if f in low:
            t.set_color("#c0392b")
        elif g == "rhythm":
            t.set_color("#e67e22")
    ax.axvline(1 / len(s), color="black", ls="--", lw=1)
    ax.text(1 / len(s), 8, " chia đều 1/41 = 0,024", fontsize=8, va="center")
    ax.set_xlabel("tỷ phần |SHAP| trên nạn nhân (mức cao nhất qua OCSVM, IF, LOF)")
    ax.set_title("41 đặc trưng xếp theo mức được dùng (1 run dev, seed 42)")
    ax.margins(y=0.01)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#c0392b", "#e67e22", "#2a78d6")]
    ax.legend(handles, [f"{N_LOW} đặc trưng SHAP thấp nhất", "rhythm (bỏ vì thuộc nhóm rhythm)", "giữ"],
              loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(s.head(N_LOW + 2).round(4).to_string())


if __name__ == "__main__":
    main()
