"""
Đa cộng tuyến khi THÊM cột lịch sử 14 / 21 ngày vào bộ 30 đặc trưng (trước khi chạy multiwindow_ablation.py).

    python scripts/experiments/window_collinearity.py

- Dữ liệu: ma trận w7 / w14 / w21 của ``data/window_ablation/w{N}/test_seed20261061`` nhưng CHỈ ngày ≤ 35, phân khúc
  User. Các ngày này là log sạch (tiêm test chỉ ở ngày 43–60, cột lịch sử chỉ nhìn về trước), trùng với phần train
  của khối dev -> không nhìn ngày tiêm hay nhãn. Script kiểm tra điều này bằng cách so với ma trận gốc của một run dev.
- Tiền xử lý như mô hình: imputer median rồi StandardScaler (fit trên chính các dòng này).
- Đo:
  1. từng cặp ``x_7d`` với ``x_14d`` / ``x_21d``: Pearson, Spearman, tỷ lệ dòng có giá trị khác nhau;
  2. VIF = diag(R⁻¹) của ma trận tương quan cho base30, base30 + 21 ngày (A21: +17), base30 + 21 ngày nhóm failure
     (B21: +4), và base30 + 14 + 21 ngày (A: +34); số VIF > 10, VIF lớn nhất, số điều kiện;
  3. R² của mỗi cột 21 ngày khi hồi quy trên toàn bộ base30 (= 1 − 1/VIF trong bộ base30 + cột đó);
     C21 = base30 + các cột 21 ngày có R² < ``--r2-max`` (còn mang thông tin mới).

Đầu ra: experiments/ablation/multiwindow/collinearity/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from multiwindow_ablation import KEYS, history_features, multiwindow_configs  # noqa: E402

from src.evaluation.feature_groups import ABLATION_GROUPS  # noqa: E402
from src.features.schema import FeatureSchema  # noqa: E402


def load(path: Path, max_day: int) -> pl.DataFrame:
    return (pl.read_parquet(path).filter((pl.col("entity_type") == "User") & (pl.col("day") <= max_day)).sort(KEYS))


def vif_table(X: pd.DataFrame) -> pd.Series:
    r = np.corrcoef(X.to_numpy(), rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(r)), index=X.columns)


def summarize_set(name: str, X: pd.DataFrame) -> dict:
    v = vif_table(X)
    eig = np.linalg.eigvalsh(np.corrcoef(X.to_numpy(), rowvar=False))
    return {"set": name, "n_features": X.shape[1], "max_vif": float(v.max()), "argmax_vif": v.idxmax(),
            "n_vif_gt_5": int((v > 5).sum()), "n_vif_gt_10": int((v > 10).sum()),
            "condition_number": float(np.sqrt(eig.max() / max(eig.min(), 1e-12)))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run", default="test_seed20261061")
    p.add_argument("--dev-check", default="data/injection_runs/dev_seed20261036")
    p.add_argument("--max-day", type=int, default=35)
    p.add_argument("--frozen-grid", default="configs/benchmark_grid_test_30f.yaml")
    p.add_argument("--r2-max", type=float, default=0.75, help="ngưỡng R² theo base30 để giữ cột 21 ngày (bộ C21)")
    p.add_argument("--spearman-rows", type=int, default=200_000)
    p.add_argument("--out", default="experiments/ablation/multiwindow/collinearity")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    core = list(FeatureSchema().core_features)
    drop = set(yaml.safe_load(Path(args.frozen_grid).read_text(encoding="utf-8")).get("drop_features") or [])
    base = [f for f in core if f not in drop]
    hist = history_features(base)

    m = {w: load(ROOT / f"data/window_ablation/w{w}/{args.run}/processed/feature_matrix_processed.parquet",
                 args.max_day) for w in (7, 14, 21)}
    for w in (14, 21):
        assert m[w].select(KEYS).equals(m[7].select(KEYS)), f"khoá dòng w{w} khác w7"
    dev = load(ROOT / args.dev_check / "processed/feature_matrix_processed.parquet", args.max_day)
    assert dev.select(KEYS).equals(m[7].select(KEYS)), "ngày ≤ max_day của run dev và run test khác dòng"
    gap = max(float((dev[c].cast(pl.Float64) - m[7][c].cast(pl.Float64)).abs().max() or 0) for c in core)
    print(f"Kiểm tra ngày ≤ {args.max_day}: run dev và run test lệch tối đa {gap:.2e} trên {len(core)} đặc trưng core")

    df = m[7].select(base).to_pandas()
    for w in (14, 21):
        for f in hist:
            df[f.removesuffix("_7d") + f"_{w}d"] = m[w][f].to_numpy()
    df = df.astype(float)
    df = df.fillna(df.median())
    Z = (df - df.mean()) / df.std(ddof=0).replace(0, 1)
    print(f"{len(Z):,} dòng User, ngày 1–{args.max_day}")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(Z), size=min(args.spearman_rows, len(Z)), replace=False)
    pairs = []
    for f in hist:
        for w in (14, 21):
            g = f.removesuffix("_7d") + f"_{w}d"
            rho = spearmanr(Z[f].to_numpy()[idx], Z[g].to_numpy()[idx]).statistic
            pairs.append({"feature": f, "window": w, "group": next(k for k, v in ABLATION_GROUPS.items() if f in v),
                          "pearson": float(np.corrcoef(Z[f], Z[g])[0, 1]), "spearman": float(rho),
                          "share_rows_differ": float((df[f] != df[g]).mean())})
    pairs = pd.DataFrame(pairs)
    pairs.round(4).to_csv(out / "pair_correlation.csv", index=False, encoding="utf-8-sig")

    cfgs = multiwindow_configs(base)
    fail = [f for f in ABLATION_GROUPS["failure"] if f in hist]
    sets = {"base30": base, "A21 (base30 + 21d)": base + [f.removesuffix("_7d") + "_21d" for f in hist],
            "B21 (base30 + 21d failure)": base + [f.removesuffix("_7d") + "_21d" for f in fail],
            "A (base30 + 14d + 21d)": cfgs["A_all_hist"], "B (base30 + 14d/21d failure)": cfgs["B_failure_hist"]}
    summary = pd.DataFrame([summarize_set(k, Z[v]) for k, v in sets.items()])
    summary.round(2).to_csv(out / "vif_summary.csv", index=False, encoding="utf-8-sig")

    vif_a21 = vif_table(Z[sets["A21 (base30 + 21d)"]]).rename("vif_in_A21")
    vif_base = vif_table(Z[base]).rename("vif_in_base30")
    per = pd.concat([vif_base, vif_a21], axis=1)
    # R² của cột 21 ngày theo base30: thông tin "mới" mà cột đó còn mang
    r2 = {f.removesuffix("_7d") + "_21d": 1 - 1 / vif_table(Z[base + [f.removesuffix("_7d") + "_21d"]]).iloc[-1]
          for f in hist}
    per["r2_on_base30"] = pd.Series(r2)
    # C21: chỉ thêm cột 21 ngày còn mang thông tin mới (R² theo base30 < ngưỡng)
    c21 = [c for c, v in r2.items() if v < args.r2_max]
    summary = pd.concat([summary, pd.DataFrame([summarize_set(f"C21 (base30 + 21d, R² < {args.r2_max})", Z[base + c21])])])
    summary.round(2).to_csv(out / "vif_summary.csv", index=False, encoding="utf-8-sig")
    print("C21 thêm:", c21)
    per.round(3).sort_values("vif_in_A21", ascending=False).to_csv(out / "vif_per_feature.csv", encoding="utf-8-sig")

    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summary.round(2).to_string(index=False))
        print(pairs[pairs["window"] == 21].round(3).sort_values("spearman", ascending=False).to_string(index=False))
        print(per.loc[[c for c in per.index if c.endswith("_21d")]].round(3)
              .sort_values("r2_on_base30").to_string())


if __name__ == "__main__":
    main()
