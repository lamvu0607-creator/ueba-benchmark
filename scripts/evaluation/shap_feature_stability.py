"""
SHAP theo TỪNG đặc trưng: độ ổn định giữa các run dev, top đặc trưng mỗi kịch bản, và cụm đặc trưng theo tương quan.

    python scripts/evaluation/shap_feature_stability.py

Đọc ``shap_values.parquet`` của ``explain_models.py`` trên nhiều run dev (mặc định 36, 37, 38; seed 42). Không dùng
khối test.

1. **Độ ổn định:** với mỗi mô hình, tỷ phần |SHAP| trung bình trên nạn nhân của 41 đặc trưng ở từng run; Spearman
   của hạng giữa các cặp run, trùng top-10; cùng phép đo ở mức nhóm theo nghĩa (8 nhóm) và mức cụm tương quan.
2. **Top đặc trưng theo kịch bản:** tỷ phần trung bình qua run; ``n_runs_top5`` = số run mà đặc trưng lọt top-5.
3. **Cụm theo tương quan:** |Spearman| giữa 41 đặc trưng trên dòng train User (ngày 1–35) của run dev đầu, phân cụm
   phân cấp (average linkage, khoảng cách 1 − |ρ|, cắt tại ``--cut``). Tỷ phần của cụm = tổng tỷ phần các đặc trưng
   trong cụm, không phụ thuộc cách KernelSHAP chia công giữa các đặc trưng tương quan trong cụm.
4. **11 đặc trưng đã bỏ** (bộ 30): tỷ phần và hạng ở từng run, vì danh sách SHAP thấp ban đầu chỉ lấy từ một run.

Đầu ra: experiments/analysis/explain_stability/
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.feature_groups import ABLATION_GROUPS, FEATURE_GROUP  # noqa: E402
from src.features.schema import FeatureSchema  # noqa: E402

SHORT = {"one_class_svm": "OCSVM", "isolation_forest": "IF", "local_outlier_factor": "LOF"}
ML = list(SHORT)
COL_LABEL = {"false_alarm": "BÁO NHẦM\n(bình thường, điểm cao)", "all_victims": "tất cả nạn nhân"}
SCEN = ["brute_force", "password_spraying", "off_hours", "logon_type_switch", "new_workstation_burst",
        "dormant_wakeup"]
GROUP_COLOR = {"volume": "#8c564b", "failure": "#d62728", "time": "#9467bd", "rhythm": "#7f7f7f",
               "logon_type": "#1f77b4", "fanout": "#2ca02c", "novelty": "#ff7f0e", "context": "#17becf"}


def load_shap(dirs: dict) -> pd.DataFrame:
    parts = []
    for run, d in dirs.items():
        s = pd.read_parquet(Path(d) / "shap_values.parquet", columns=["model", "row", "kind", "scenario", "feature",
                                                                      "share"])
        parts.append(s.assign(run=run))
    return pd.concat(parts, ignore_index=True)


def unit_share(shap: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Tổng tỷ phần theo đơn vị (đặc trưng / nhóm / cụm) trên từng dòng, rồi TB trên nạn nhân -> (model, run, unit)."""
    v = shap[shap["kind"] == "victim"].assign(unit=lambda d: d["feature"].map(mapping))
    per_row = v.groupby(["model", "run", "row", "unit"], observed=True)["share"].sum().reset_index()
    return per_row.groupby(["model", "run", "unit"])["share"].mean().reset_index()


def stability(share: pd.DataFrame, level: str, k: int) -> pd.DataFrame:
    rows = []
    for m in ML:
        piv = share[share["model"] == m].pivot(index="unit", columns="run", values="share")
        for a, b in itertools.combinations(piv.columns, 2):
            top_a, top_b = set(piv[a].nlargest(k).index), set(piv[b].nlargest(k).index)
            rows.append({"level": level, "n_units": len(piv), "model": SHORT[m], "run_a": a, "run_b": b,
                         "spearman": spearmanr(piv[a], piv[b]).statistic, f"top_overlap": len(top_a & top_b) / k,
                         "top_k": k})
    return pd.DataFrame(rows)


def correlation_clusters(run_dir: Path, features: list, max_day: int, cut: float, n_rows: int):
    df = (pl.read_parquet(run_dir / "processed/feature_matrix_processed.parquet", columns=["entity_type", "day",
                                                                                         *features])
          .filter((pl.col("entity_type") == "User") & (pl.col("day") <= max_day)).select(features).to_pandas())
    df = df.astype(float).fillna(df.median())
    df = df.sample(n=min(n_rows, len(df)), random_state=0)
    rho = pd.DataFrame(spearmanr(df.to_numpy()).statistic, index=features, columns=features).fillna(0)
    dist = 1 - rho.abs().to_numpy()
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(np.clip(dist, 0, None), checks=False), method="average")
    labels = fcluster(z, t=cut, criterion="distance")
    clusters = pd.Series(labels, index=features)
    # đặt tên cụm theo đặc trưng đầu tiên (thứ tự schema) + số thành viên
    names = {c: (f"{g.index[0]}" if len(g) == 1 else f"C{c}:{g.index[0]}+{len(g) - 1}")
             for c, g in clusters.groupby(clusters)}
    return rho, z, clusters.map(names)


def plot_feature_stability(feat: pd.DataFrame, dropped: set, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 10), sharey=False)
    for ax, m in zip(axes, ML):
        d = feat[feat["model"] == SHORT[m]].sort_values("share_mean")
        y = np.arange(len(d))
        ax.barh(y, d["share_mean"], color=[GROUP_COLOR[g] for g in d["group"]],
                xerr=[d["share_mean"] - d["share_min"], d["share_max"] - d["share_mean"]], error_kw={"lw": 0.8})
        ax.set_yticks(y, [f"{f} ✕" if f in dropped else f for f in d["feature"]], fontsize=7)
        for t, f in zip(ax.get_yticklabels(), d["feature"]):
            if f in dropped:
                t.set_color("red")
        ax.axvline(1 / len(d), color="black", ls="--", lw=0.8)
        ax.set_title(f"{SHORT[m]}: tỷ phần |SHAP| trên nạn nhân\nTB và min–max qua {d['n_runs'].iloc[0]} run dev")
        ax.set_xlabel("tỷ phần |SHAP|")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
    fig.legend(handles, list(GROUP_COLOR), loc="lower center", ncol=8, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, -0.045, "✕ đỏ = 11 đặc trưng đã bỏ ở bộ 30 · nét đứt = mức chia đều 1/41", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "feature_shap_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_by_scenario(scen: pd.DataFrame, order: list, dropped: set, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = SCEN + ["false_alarm"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 11), sharey=True)
    for ax, m in zip(axes, ML):
        piv = scen[scen["model"] == SHORT[m]].pivot(index="feature", columns="col", values="share").reindex(
            index=order, columns=cols)
        im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0, vmax=0.4, aspect="auto")
        for (r, c), v in np.ndenumerate(piv.to_numpy()):
            if v >= 0.05:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.25 else "black")
        ax.set_xticks(range(len(cols)), [COL_LABEL.get(c, c) for c in cols], rotation=40, ha="right", fontsize=8)
        ax.axvline(len(SCEN) - 0.5, color="black", lw=1.5)
        ax.set_title(SHORT[m])
        bounds = np.flatnonzero(np.diff([list(ABLATION_GROUPS).index(FEATURE_GROUP[f]) for f in order])) + 0.5
        for b in bounds:
            ax.axhline(b, color="gray", lw=0.6)
    axes[0].set_yticks(range(len(order)), [f"{f} ✕" if f in dropped else f for f in order], fontsize=7)
    for t, f in zip(axes[0].get_yticklabels(), order):
        if f in dropped:
            t.set_color("red")
    fig.colorbar(im, ax=axes, fraction=0.02, label="tỷ phần |SHAP| TB qua run dev")
    fig.suptitle("Tỷ phần |SHAP| theo từng đặc trưng × kịch bản (vạch xám tách 8 nhóm; số ghi khi ≥ 0,05)")
    fig.text(0.01, -0.06, "Cột cuối (sau vạch đen) không phải kịch bản: 30 dòng bình thường (không tiêm) có điểm cao "
             "nhất của mỗi mô hình — cho biết đặc trưng nào làm mô hình báo sai.", fontsize=8)
    fig.savefig(out / "figures" / "feature_shap_by_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clusters(rho: pd.DataFrame, z, clusters: pd.Series, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(rho.index[leaves_list(z)])
    r = rho.loc[order, order].abs()
    fig, ax = plt.subplots(figsize=(12, 10.5))
    im = ax.imshow(r.to_numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(order)), [f"{f}  [{FEATURE_GROUP[f]}]" for f in order], fontsize=6.5)
    c = clusters.loc[order].to_numpy()
    start = 0
    for i in range(1, len(c) + 1):
        if i == len(c) or c[i] != c[start]:
            if i - start > 1:
                ax.add_patch(plt.Rectangle((start - .5, start - .5), i - start, i - start, fill=False,
                                           edgecolor="red", lw=1.8))
            start = i
    fig.colorbar(im, ax=ax, fraction=0.04, label="|Spearman|")
    ax.set_title("|Spearman| giữa 41 đặc trưng (train User, ngày 1–35) · khung đỏ = cụm tương quan")
    fig.savefig(out / "figures" / "feature_correlation_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--explain-dirs", nargs="+", default=[
        "dev_seed20261036=experiments/analysis/explain",
        "dev_seed20261037=experiments/analysis/explain_runs/dev_seed20261037",
        "dev_seed20261038=experiments/analysis/explain_runs/dev_seed20261038"])
    p.add_argument("--corr-run", default="data/injection_runs/dev_seed20261036")
    p.add_argument("--max-day", type=int, default=35)
    p.add_argument("--cut", type=float, default=0.3, help="cắt cây tại khoảng cách 1 − |ρ| (0,3 ⇔ |ρ| TB ≥ 0,7)")
    p.add_argument("--corr-rows", type=int, default=200_000)
    p.add_argument("--frozen-grid", default="configs/benchmark_grid_test_30f.yaml")
    p.add_argument("--out", default="experiments/analysis/explain_stability")
    args = p.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    dirs = dict(s.split("=", 1) for s in args.explain_dirs)
    shap = load_shap(dirs)
    core = list(FeatureSchema().core_features)
    dropped = set(yaml.safe_load(Path(args.frozen_grid).read_text(encoding="utf-8")).get("drop_features") or [])
    n_runs = shap["run"].nunique()

    # 3. cụm tương quan
    rho, z, clusters = correlation_clusters(Path(args.corr_run), core, args.max_day, args.cut, args.corr_rows)
    cl = pd.DataFrame({"feature": core, "group": [FEATURE_GROUP[f] for f in core], "cluster": clusters[core].values})
    cl["cluster_size"] = cl.groupby("cluster")["feature"].transform("size")
    cl.sort_values(["cluster_size", "cluster"], ascending=[False, True]).to_csv(
        out / "correlation_clusters.csv", index=False, encoding="utf-8-sig")
    plot_clusters(rho, z, clusters, out)

    # 1. độ ổn định: đặc trưng / nhóm / cụm
    f_share = unit_share(shap, {f: f for f in core})
    g_share = unit_share(shap, FEATURE_GROUP)
    c_share = unit_share(shap, clusters.to_dict())
    stab = pd.concat([stability(f_share, "feature", 10), stability(g_share, "group", 3),
                      stability(c_share, "cluster", 5)])
    stab.round(3).to_csv(out / "rank_stability.csv", index=False, encoding="utf-8-sig")
    stab_sum = stab.groupby(["level", "model"])[["spearman", "top_overlap"]].mean().round(3).reset_index()
    stab_sum.to_csv(out / "rank_stability_summary.csv", index=False, encoding="utf-8-sig")

    feat = (f_share.groupby(["model", "unit"])["share"].agg(share_mean="mean", share_min="min", share_max="max")
            .reset_index().rename(columns={"unit": "feature"}))
    feat["model"] = feat["model"].map(SHORT)
    f_share["rank"] = f_share.groupby(["model", "run"])["share"].rank(ascending=False)
    rk = f_share.groupby(["model", "unit"])["rank"].agg(rank_best="min", rank_worst="max").reset_index()
    rk["model"] = rk["model"].map(SHORT)
    feat = feat.merge(rk.rename(columns={"unit": "feature"}), on=["model", "feature"])
    feat["group"] = feat["feature"].map(FEATURE_GROUP)
    feat["cluster"] = feat["feature"].map(clusters)
    feat["dropped_in_30"] = feat["feature"].isin(dropped)
    feat["n_runs"] = n_runs
    feat.sort_values(["model", "share_mean"], ascending=[True, False]).round(4).to_csv(
        out / "feature_share_by_run.csv", index=False, encoding="utf-8-sig")
    plot_feature_stability(feat, dropped, out)

    # 2. top đặc trưng theo kịch bản (và báo nhầm)
    rows = shap.assign(col=np.where(shap["kind"] == "victim", shap["scenario"], "false_alarm"))
    per_run = rows.groupby(["model", "run", "col", "feature"])["share"].mean().reset_index()
    per_run["top5"] = per_run.groupby(["model", "run", "col"])["share"].rank(ascending=False) <= 5
    scen = per_run.groupby(["model", "col", "feature"]).agg(share=("share", "mean"),
                                                             n_runs_top5=("top5", "sum")).reset_index()
    scen["model"] = scen["model"].map(SHORT)
    top = (scen.sort_values("share", ascending=False).groupby(["model", "col"]).head(5)
           .sort_values(["col", "model", "share"], ascending=[True, True, False]))
    top.round(3).to_csv(out / "top5_by_scenario.csv", index=False, encoding="utf-8-sig")
    order = [f for g in ABLATION_GROUPS.values() for f in g]
    plot_feature_by_scenario(scen, order, dropped, out)

    # 4. 11 đặc trưng đã bỏ
    drop_tab = feat[feat["dropped_in_30"]].pivot(index="feature", columns="model",
                                                 values=["share_mean", "rank_best"]).round(3)
    drop_tab.to_csv(out / "dropped_features.csv", encoding="utf-8-sig")

    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(f"{n_runs} run dev: {sorted(dirs)}")
        print(stab_sum.to_string(index=False))
        print(cl[cl["cluster_size"] > 1].sort_values("cluster").to_string(index=False))
        print(drop_tab.to_string())
        print(top.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
