"""
Giải thích điểm của OCSVM / IF / LOF bằng SHAP (KernelSHAP) và LIME, để chuẩn bị ablation theo nhóm đặc trưng (tuần 5).

    python scripts/evaluation/explain_models.py                          # run dev mặc định, seed 42
    python scripts/evaluation/explain_models.py --n-victims 10 --skip-lime   # thử nhanh

Chạy trên khối **dev** (fit ngày 1–35, chấm 36–42): kết quả dùng để chọn nhóm cần ablation, nên không được nhìn test.

Không gian giải thích = 41 đặc trưng core **sau imputer, trước scaler** (đơn vị gốc, LIME rời rạc hoá đọc được);
hàm được giải thích là điểm dị biệt của pipeline (scaler -> mô hình), CAO = DỊ BIỆT.

Mẫu được giải thích (cùng dòng cho cả 3 mô hình, trừ báo nhầm):
- nạn nhân: ``--n-victims`` mỗi kịch bản (lấy ngẫu nhiên);
- báo nhầm: ``--n-false-alarms`` dòng không tiêm có điểm cao nhất của từng mô hình.

Để so được giữa các mô hình (thang điểm khác nhau), mỗi dòng được chuẩn hoá thành **tỷ phần |đóng góp|**
(|φ_j| / Σ|φ|). Đầu ra: experiments/analysis/explain/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.feature_groups import ABLATION_GROUPS, FEATURE_GROUP, schema_origin  # noqa: E402
from src.evaluation.grid import _params_with, fit_and_score, load_grid_config, load_run_data  # noqa: E402
from src.models.registry import load_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("ueba_benchmark.explain_models")
logging.getLogger("shap").setLevel(logging.WARNING)

ML = ["one_class_svm", "isolation_forest", "local_outlier_factor"]
SHORT = {"one_class_svm": "OCSVM", "isolation_forest": "IF", "local_outlier_factor": "LOF"}


def score_fn(pipe):
    """Điểm của pipeline trên ma trận đã impute (đơn vị gốc): scaler -> mô hình."""
    def f(Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        if pipe.scaler_ is not None:
            Z = pipe.scaler_.transform(Z)
        return pipe.model.score(Z)
    return f


def pick_rows(data: dict, scores: dict, n_victims: int, n_fa: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y, scen = data["y"], data["scenario"].astype(object)
    rows = []
    for s in sorted(set(scen[y == 1])):
        idx = np.flatnonzero((y == 1) & (scen == s))
        for i in rng.choice(idx, size=min(n_victims, idx.size), replace=False):
            rows += [{"model": m, "row": int(i), "kind": "victim", "scenario": s} for m in ML]
    for m in ML:
        neg = np.flatnonzero(y == 0)
        for i in neg[np.argsort(-scores[m][neg], kind="stable")[:n_fa]]:
            rows.append({"model": m, "row": int(i), "kind": "false_alarm", "scenario": "false_alarm"})
    return pd.DataFrame(rows)


def run_shap(f, X_bg: np.ndarray, X: np.ndarray, nsamples: int) -> tuple[np.ndarray, float]:
    import shap
    explainer = shap.KernelExplainer(f, X_bg)
    phi = explainer.shap_values(X, nsamples=nsamples, l1_reg=False, silent=True)
    return np.asarray(phi, dtype=np.float64), float(np.ravel(explainer.expected_value)[0])


def run_lime(f, X_train: np.ndarray, X: np.ndarray, names: list, num_samples: int, seed: int):
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(X_train, mode="regression", feature_names=names, discretize_continuous=True,
                                     random_state=seed)
    W, r2 = np.zeros_like(X), np.zeros(len(X))
    for i, x in enumerate(X):
        exp = explainer.explain_instance(x, f, num_features=len(names), num_samples=num_samples)
        for j, w in next(iter(exp.local_exp.values())):
            W[i, j] = w
        r2[i] = exp.score
    return W, r2


def to_long(sel: pd.DataFrame, X: np.ndarray, contrib: np.ndarray, names: list, origin: dict) -> pd.DataFrame:
    total = np.abs(contrib).sum(axis=1, keepdims=True)
    share = np.divide(np.abs(contrib), total, out=np.zeros_like(contrib), where=total > 0)
    n, p = contrib.shape
    out = sel.loc[sel.index.repeat(p)].reset_index(drop=True)
    return out.assign(feature=np.tile(names, n), group=[FEATURE_GROUP[c] for c in np.tile(names, n)],
                      origin=[origin[c] for c in np.tile(names, n)], value=X.ravel(), contrib=contrib.ravel(),
                      share=share.ravel())


def feature_table(long: pd.DataFrame, method: str) -> pd.DataFrame:
    """Tỷ phần trung bình theo (mô hình, đặc trưng), tách nạn nhân / báo nhầm, kèm hạng trong mô hình."""
    t = long.pivot_table(index=["model", "feature", "group", "origin"], columns="kind", values="share",
                         aggfunc="mean").reset_index()
    t = t.rename(columns={"victim": f"{method}_share_victim", "false_alarm": f"{method}_share_false_alarm"})
    t[f"{method}_rank_victim"] = t.groupby("model")[f"{method}_share_victim"].rank(ascending=False, method="min")
    return t


def group_table(long: pd.DataFrame, method: str) -> pd.DataFrame:
    """Tỷ phần mỗi nhóm trong điểm, theo (mô hình, kịch bản); dòng 'all_victims' = mọi nạn nhân."""
    per_row = long.groupby(["model", "kind", "scenario", "row", "group"], as_index=False)["share"].sum()
    by_scen = per_row.groupby(["model", "scenario", "group"], as_index=False)["share"].mean()
    allv = per_row[per_row["kind"] == "victim"].groupby(["model", "group"], as_index=False)["share"].mean()
    return pd.concat([by_scen, allv.assign(scenario="all_victims")], ignore_index=True).rename(
        columns={"share": f"{method}_share"})


def agreement(shap_long: pd.DataFrame, lime_long: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Mức khớp SHAP–LIME: Spearman hạng toàn cục, và trùng top-k đặc trưng trên từng dòng."""
    from scipy.stats import spearmanr
    rows = []
    for m in ML:
        a = shap_long[shap_long["model"] == m]
        b = lime_long[lime_long["model"] == m]
        if b.empty:
            continue
        ga, gb = a.groupby("feature")["share"].mean(), b.groupby("feature")["share"].mean()
        rho = spearmanr(ga, gb.reindex(ga.index)).statistic
        ov = []
        for (kind, row), g in b.groupby(["kind", "row"]):
            h = a[(a["kind"] == kind) & (a["row"] == row)]
            ov.append(len(set(g.nlargest(k, "share")["feature"]) & set(h.nlargest(k, "share")["feature"])) / k)
        rows.append({"model": m, "spearman_global": round(float(rho), 3), f"top{k}_overlap_mean": round(float(np.mean(ov)), 3),
                     "n_rows": len(ov), "lime_r2_mean": round(float(b.drop_duplicates(["kind", "row"])["lime_r2"].mean()), 3)})
    return pd.DataFrame(rows)


def ablation_plan(groups: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Mỗi nhóm: kịch bản phụ thuộc nhiều nhất (theo SHAP), tỷ phần trong báo nhầm; đặc trưng gần như không dùng."""
    g = groups[groups["scenario"].ne("all_victims") & groups["scenario"].ne("false_alarm")]
    rows = []
    for grp, fs in ABLATION_GROUPS.items():
        r = {"group": grp, "n_features": len(fs), "uniform_share": round(len(fs) / len(FEATURE_GROUP), 3)}
        for m in ML:
            gm = g[(g["model"] == m) & (g["group"] == grp)]
            top = gm.loc[gm["shap_share"].idxmax()]
            r[f"{SHORT[m]}_victims"] = round(float(groups.query("model == @m and group == @grp and "
                                                                 "scenario == 'all_victims'")["shap_share"].iloc[0]), 3)
            r[f"{SHORT[m]}_top_scenario"] = f"{top['scenario']} ({top['shap_share']:.2f})"
            r[f"{SHORT[m]}_false_alarm"] = round(float(groups.query("model == @m and group == @grp and "
                                                                     "scenario == 'false_alarm'")["shap_share"].iloc[0]), 3)
        low = feats.groupby("feature")["shap_share_victim"].max()
        r["near_unused_features"] = ", ".join(f for f in fs if low.get(f, 0) < 0.25 / len(FEATURE_GROUP))
        rows.append(r)
    return pd.DataFrame(rows)


def plot(groups: pd.DataFrame, feats: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = list(ABLATION_GROUPS)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    for ax, m in zip(axes, ML):
        piv = groups[groups["model"] == m].pivot(index="group", columns="scenario", values="shap_share").reindex(order)
        cols = [c for c in piv.columns if c not in ("all_victims", "false_alarm")] + ["all_victims", "false_alarm"]
        piv = piv[cols]
        im = ax.imshow(piv.to_numpy(), cmap="Oranges", vmin=0, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(order)), order)
        for (r, c), v in np.ndenumerate(piv.to_numpy()):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.3 else "black")
        ax.set_title(f"{SHORT[m]}: tỷ phần |SHAP| theo nhóm")
    fig.colorbar(im, ax=axes, fraction=0.015)
    fig.savefig(fig_dir / "shap_group_by_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, m in zip(axes, ML):
        t = feats[feats["model"] == m].nlargest(15, "shap_share_victim").iloc[::-1]
        yy = np.arange(len(t))
        ax.barh(yy - 0.2, t["shap_share_victim"], height=0.4, label="SHAP")
        if "lime_share_victim" in t:
            ax.barh(yy + 0.2, t["lime_share_victim"], height=0.4, label="LIME")
        ax.set_yticks(yy, t["feature"], fontsize=8)
        ax.set_title(f"{SHORT[m]}: 15 đặc trưng đóng góp nhiều nhất (nạn nhân)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "top_features_shap_lime.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--grid", default="configs/benchmark_grid_test.yaml", help="lấy cấu hình mô hình đã chốt")
    p.add_argument("--run", default="data/injection_runs/dev_seed20261036")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-victims", type=int, default=30, help="số nạn nhân mỗi kịch bản")
    p.add_argument("--n-false-alarms", type=int, default=30)
    p.add_argument("--n-background", type=int, default=50, help="số dòng train làm nền cho KernelSHAP")
    p.add_argument("--shap-nsamples", type=int, default=1000)
    p.add_argument("--lime-samples", type=int, default=2000)
    p.add_argument("--lime-every", type=int, default=2, help="LIME trên 1/N số dòng của SHAP (LIME chậm hơn)")
    p.add_argument("--skip-lime", action="store_true")
    p.add_argument("--out", default="experiments/analysis/explain")
    args = p.parse_args()

    cfg = load_grid_config(args.grid)
    params = load_params("configs/model_params.yaml")
    c0 = float(cfg.get("contamination", 0.05))
    origin = schema_origin(ROOT / "configs/feature_schema.yaml")
    data = load_run_data(args.run, cfg.get("segment"), sort_rows=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipes, scores = {}, {}
    for m in ML:
        c = cfg["models"][m][0]
        res = fit_and_score(data, m, _params_with(params, m, c.get("params") or {}), c.get("scaler", "robust"),
                            args.seed, c0)
        pipes[m], scores[m] = res["pipeline"], res["scores"]
        logger.info("%s (%s) fit xong %.0fs", m, c["id"], res["fit_seconds"])
    names = list(pipes[ML[0]].feature_names)
    if set(names) != set(FEATURE_GROUP):
        raise ValueError(f"ABLATION_GROUPS không khớp đặc trưng của pipeline: {set(names) ^ set(FEATURE_GROUP)}")

    # Cả 3 pipeline dùng cùng imputer median trên cùng train -> chung không gian giải thích.
    imp = pipes[ML[0]].imputer_
    X_ev = imp.transform(data["eval"][names].to_numpy(dtype=np.float64))
    X_tr = imp.transform(data["train"][names].to_numpy(dtype=np.float64))
    rng = np.random.default_rng(args.seed)
    X_bg = X_tr[rng.choice(len(X_tr), size=args.n_background, replace=False)]
    X_lime_train = X_tr[rng.choice(len(X_tr), size=min(20000, len(X_tr)), replace=False)]

    sel = pick_rows(data, scores, args.n_victims, args.n_false_alarms, args.seed)
    keys = data["eval"][["DomainName", "UserName", "day"]].reset_index(drop=True)
    sel = sel.join(keys, on="row")
    sel["score"] = [scores[m][r] for m, r in zip(sel["model"], sel["row"])]
    sel["rank_pct"] = [float((scores[m] > scores[m][r]).mean()) for m, r in zip(sel["model"], sel["row"])]

    shap_parts, lime_parts = [], []
    for m in ML:
        s = sel[sel["model"] == m].reset_index(drop=True)
        X = X_ev[s["row"].to_numpy()]
        f = score_fn(pipes[m])
        t0 = time.perf_counter()
        phi, base = run_shap(f, X_bg, X, args.shap_nsamples)
        gap = np.abs(phi.sum(axis=1) + base - f(X)).max()
        logger.info("%s SHAP %d dòng: %.0fs (sai số cộng tính tối đa %.2e)", m, len(X), time.perf_counter() - t0, gap)
        shap_parts.append(to_long(s.assign(shap_base=base), X, phi, names, origin))
        if not args.skip_lime:
            li = np.flatnonzero(s.groupby("scenario").cumcount().to_numpy() % args.lime_every == 0)
            sl = s.iloc[li].reset_index(drop=True)
            t0 = time.perf_counter()
            W, r2 = run_lime(f, X_lime_train, X[li], names, args.lime_samples, args.seed)
            logger.info("%s LIME %d dòng: %.0fs (R² cục bộ TB %.2f)", m, len(li), time.perf_counter() - t0, r2.mean())
            lime_parts.append(to_long(sl.assign(lime_r2=r2), X[li], W, names, origin))

    shap_long = pd.concat(shap_parts, ignore_index=True)
    shap_long.to_parquet(out / "shap_values.parquet", index=False)
    feats = feature_table(shap_long, "shap")
    groups = group_table(shap_long, "shap")
    if lime_parts:
        lime_long = pd.concat(lime_parts, ignore_index=True)
        lime_long.to_parquet(out / "lime_weights.parquet", index=False)
        feats = feats.merge(feature_table(lime_long, "lime"), on=["model", "feature", "group", "origin"], how="left")
        groups = groups.merge(group_table(lime_long, "lime"), on=["model", "scenario", "group"], how="left")
        agree = agreement(shap_long, lime_long)
        agree.to_csv(out / "shap_lime_agreement.csv", index=False, encoding="utf-8-sig")
        print(agree.to_string(index=False))
    feats.sort_values(["model", "shap_share_victim"], ascending=[True, False]).to_csv(
        out / "feature_importance.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(out / "group_importance.csv", index=False, encoding="utf-8-sig")
    plan = ablation_plan(groups, feats)
    plan.to_csv(out / "ablation_plan.csv", index=False, encoding="utf-8-sig")
    origin_share = shap_long[shap_long["kind"] == "victim"].groupby(["model", "row", "origin"])["share"].sum() \
        .groupby(["model", "origin"]).mean().unstack().round(3)
    origin_share.to_csv(out / "origin_share.csv", encoding="utf-8-sig")
    plot(groups, feats, out)
    (out / "explain_manifest.json").write_text(json.dumps({
        "run": str(args.run), "seed": args.seed, "segment": cfg.get("segment"),
        "configs": {m: cfg["models"][m][0]["id"] for m in ML}, "n_features": len(names),
        "n_victims_per_scenario": args.n_victims, "n_false_alarms_per_model": args.n_false_alarms,
        "n_background": args.n_background, "shap_nsamples": args.shap_nsamples,
        "lime_samples": None if args.skip_lime else args.lime_samples, "lime_every": args.lime_every,
        "ablation_groups": ABLATION_GROUPS}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(plan.to_string(index=False))
    print(origin_share.to_string())


if __name__ == "__main__":
    main()
