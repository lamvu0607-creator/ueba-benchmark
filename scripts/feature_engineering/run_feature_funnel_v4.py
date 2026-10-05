"""Phễu chọn đặc trưng schema v4.0: Combinatorial Template → vòng 1 (lập luận) → vòng 2 (số liệu).

Cách dùng (≈ 10 phút cho 21 ngày):
    python scripts/feature_engineering/run_feature_funnel_v4.py
    python scripts/feature_engineering/run_feature_funnel_v4.py --end-day 21 --target-day 21

Vòng 2 gồm 4 cổng, áp THEO THỨ TỰ do thực tập sinh chốt (2026-10-04):

    E. Không rò rỉ  — tính lại tầng lịch sử khi CẮT BỎ ngày cuối; mọi giá trị ngày ≤ t−1 phải y hệt.
    B. Có thông tin — trên ngày sau warm-up: NULL ≤ 50% và giá trị phổ biến nhất < 99% số dòng.
    A. Không trùng  — max(|Spearman|, |Pearson|) < 0,85 với 24 core v3.0 VÀ với ứng viên đã nhận
                      trước đó (thứ tự: ưu tiên kịch bản → đơn vị giải thích → tên).
    C. Phân tách    — trên ngày tiêm, AUC theo đúng chiều bất thường ≥ 0,70 cho ÍT NHẤT một
                      kịch bản mà ứng viên được thiết kế để bắt (bảng ánh xạ luật R2).

Đầu ra (docs/feature_engineering/tables/):
    v4_template_space.csv     toàn bộ 3.136 tổ hợp + trạng thái bước 0 / vòng 1 + lý do
    v4_round2_gates.csv       78 ứng viên vòng 2 × số đo của 4 cổng + quyết định
    v4_scenario_auc.csv       AUC theo chiều (đặc trưng × kịch bản) cho 24 core + mọi ứng viên
    v4_funnel_summary.csv     số lượng qua từng tầng
    v4_final_vif.csv          VIF của bộ core v4 (24 + biến giữ lại)
Ma trận trung gian: data/features/funnel_v4/{clean,injected}.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.injection import INJECTION_SCENARIOS, inject_day, select_victims  # noqa: E402
from src.features.extractor import (  # noqa: E402
    MATRIX_ORDERED_COLS, RAW_ORDERED_COLS, features_from_events, load_day_events, resolve_feature_config,
)
from src.features.history import add_history_features  # noqa: E402
from src.features.preprocessor import normalize_features  # noqa: E402
from src.features.schema import FeatureSchema  # noqa: E402
from src.features.template_engine import ACCOUNT_KEYS, compute_day, compute_history  # noqa: E402
from src.features.templates import MEASURES, Candidate, build_catalog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger("funnel_v4")

RHO_MAX = 0.85
NULL_MAX = 0.50
TOP_SHARE_MAX = 0.99
AUC_MIN = 0.70
TARGET_CORE = 15          # quy mô do thực tập sinh chốt (2026-10-04)
VIF_MAX = 10.0            # ngưỡng "nghiêm trọng" đang dùng ở check_multicollinearity.py / báo cáo v3 §7
SCENARIO_ORDER = ["brute_force", "password_spraying", "new_workstation_burst", "off_hours",
                  "lateral_fanout", "dormant_wakeup", "logon_type_switch"]
WARMUP_DAYS = 7
TABLES = REPO_ROOT / "docs" / "feature_engineering" / "tables"
MATRIX_DIR = REPO_ROOT / "data" / "features" / "funnel_v4"
KEYS = ACCOUNT_KEYS + ["day"]


# ---------------------------------------------------------------------------
# Dựng hai ma trận (sạch / đã tiêm) bằng đúng đường tính của pipeline
# ---------------------------------------------------------------------------
def build_matrices(start: int, end: int, target: int, specs: List[Candidate], cfg: dict):
    fcfg = resolve_feature_config(cfg)
    raw = pl.read_parquet(REPO_ROOT / "data" / "features" / "raw" / "feature_matrix_raw.parquet")
    plan = select_victims(raw.filter(pl.col("day").is_between(start, end)), target_day=target)
    log.info("Nạn nhân: %s", {s: v.height for s, v in plan.victims.items()})
    dormant = plan.dormant_accounts().select(ACCOUNT_KEYS)
    dormant_acct = set(
        dormant.select(pl.concat_str(ACCOUNT_KEYS, separator="\x1f").hash()).to_series().to_list()
    )

    parts: Dict[str, Dict[str, list]] = {
        v: {"core": [], "pres": [], "eng": [], "prof": []} for v in ("clean", "injected")
    }
    data_dir = REPO_ROOT / "data" / "interim"
    for d in range(start, end + 1):
        t0 = time.time()
        ev = load_day_events(d, data_dir)
        if ev is None:
            continue
        core, pres = features_from_events(ev, d, feature_cfg=fcfg, return_presences=True)
        eng = compute_day(ev, d, specs, fcfg)
        _append(parts["clean"], core, pres, eng)

        if d == target:
            ev_i = inject_day(ev, d, plan)
            core_i, pres_i = features_from_events(ev_i, d, feature_cfg=fcfg, return_presences=True)
            _append(parts["injected"], core_i, pres_i, compute_day(ev_i, d, specs, fcfg))
        elif plan.dormant_from <= d < target:
            # chỉ xoá nạn nhân ngủ đông ⇒ lọc trực tiếp kết quả sạch (tương đương chạy lại)
            _append(parts["injected"],
                    core.join(dormant, on=ACCOUNT_KEYS, how="anti"),
                    pres.join(dormant, on=ACCOUNT_KEYS, how="anti"),
                    _drop_accounts(eng, dormant_acct))
        else:
            _append(parts["injected"], core, pres, eng)
        del ev
        log.info("Day %02d xong (%.1fs)", d, time.time() - t0)

    out = {}
    for variant, p in parts.items():
        out[variant] = _assemble(p, specs)
        log.info("Ma trận %s: %d dòng × %d cột", variant, out[variant].height, out[variant].width)
    return out, plan, parts["clean"]


def _append(store, core, pres, eng):
    store["core"].append(core)
    store["pres"].append(pres)
    store["eng"].append(eng.intraday)
    store["prof"].append(eng.profiles)


def _drop_accounts(eng, accts):
    from src.features.template_engine import DayOutput
    keep = ~pl.col("_acct").is_in(list(accts))
    return DayOutput(eng.intraday.filter(keep), {k: v.filter(keep) for k, v in eng.profiles.items()})


def _concat_profiles(profs: List[dict]) -> dict:
    keys = profs[0].keys()
    return {k: pl.concat([p[k] for p in profs]) for k in keys}


def _assemble(p, specs) -> pl.DataFrame:
    core = pl.concat(p["core"])
    pres = pl.concat(p["pres"])
    core = add_history_features(core, entities=pres).select(MATRIX_ORDERED_COLS)
    core = normalize_features(core)
    tmpl = _template_matrix(p, specs, core)
    return core.join(tmpl, on=KEYS, how="left")


def _template_matrix(p, specs, core: pl.DataFrame, max_day: int | None = None) -> pl.DataFrame:
    eng = pl.concat(p["eng"])
    prof = _concat_profiles(p["prof"])
    if max_day is not None:
        eng = eng.filter(pl.col("day") <= max_day)
        prof = {k: v.filter(pl.col("day") <= max_day) for k, v in prof.items()}
    et = core.select(KEYS + ["entity_type"])
    hist = compute_history(eng, prof, specs, entity_type=et)
    return hist.select(KEYS + [c.name for c in specs])


# ---------------------------------------------------------------------------
# Các cổng
# ---------------------------------------------------------------------------
def gate_leakage(clean_parts, specs, core, end: int) -> Dict[str, float]:
    """Cổng E: cắt bỏ ngày cuối rồi tính lại; giá trị các ngày còn lại phải y hệt."""
    full = _template_matrix(clean_parts, specs, core)
    cut = _template_matrix(clean_parts, specs, core.filter(pl.col("day") <= end - 1), max_day=end - 1)
    both = full.filter(pl.col("day") <= end - 1).join(cut, on=KEYS, how="inner", suffix="__cut")
    out = {}
    for c in specs:
        a = both[c.name].cast(pl.Float64).to_numpy()
        b = both[f"{c.name}__cut"].cast(pl.Float64).to_numpy()
        same_null = np.array_equal(np.isnan(a), np.isnan(b))
        diff = np.nanmax(np.abs(a - b)) if np.isfinite(a - b).any() else 0.0
        out[c.name] = float(diff) if same_null else float("inf")
    return out


def gate_information(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c]
        null_share = float(s.isna().mean())
        nn = s.dropna().round(9)
        top = float(nn.value_counts(normalize=True).iloc[0]) if len(nn) else 1.0
        rows.append({"feature": c, "null_share": null_share, "top_value_share": top})
    return pd.DataFrame(rows).set_index("feature")


def correlation(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """max(|Spearman|, |Pearson|) — cùng quy ước fillna(0) của check_multicollinearity.py."""
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sp = X.rank().corr(method="pearson").abs()
    pe = X.corr(method="pearson").abs()
    return np.maximum(sp.fillna(0.0), pe.fillna(0.0))


def directional_auc(x: np.ndarray, y: np.ndarray, direction: str) -> float:
    """AUC (Mann–Whitney) với nhãn y=1 là bất thường; 'down' ⇒ đảo dấu; 'both' ⇒ max(AUC, 1−AUC)."""
    from scipy.stats import rankdata
    r = rankdata(x)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    if direction == "down":
        return 1.0 - auc
    if direction == "both":
        return max(auc, 1.0 - auc)
    return auc


def scenario_auc(inj: pl.DataFrame, labels: pl.DataFrame, feats: Dict[str, str], target: int) -> pd.DataFrame:
    day = inj.filter(pl.col("day") == target).join(labels.drop("day"), on=ACCOUNT_KEYS, how="left")
    pdf = day.select(list(feats) + ["scenario"]).to_pandas()
    normal = pdf["scenario"].isna()
    rows = []
    for f, direction in feats.items():
        x = pdf[f].astype(float).replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x[normal].median())                 # như imputer median của pipeline
        rec = {"feature": f, "direction": direction}
        for s in INJECTION_SCENARIOS:
            m = normal | (pdf["scenario"] == s)
            y = (pdf.loc[m, "scenario"] == s).to_numpy().astype(int)
            rec[s] = directional_auc(x[m].to_numpy(), y, direction) if y.sum() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("feature")


def vif(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.loc[:, X.std() > 0]
    corr = np.corrcoef(X.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(corr)
    return pd.Series(np.diag(inv), index=X.columns, name="VIF").sort_values(ascending=False)


def finalize(post: pd.DataFrame, gates: pd.DataFrame, auc: pd.DataFrame, core_names: List[str]):
    """
    Bước chốt quy mô: từ các biến đã qua 4 cổng, chọn ~``TARGET_CORE`` biến vào core.

    Xoay vòng theo thứ tự ưu tiên kịch bản; mỗi lượt, mỗi kịch bản lấy biến (nhắm kịch bản đó)
    có AUC cao nhất — hoà thì lấy biến có max |ρ| nhỏ hơn (mang thông tin mới hơn) — và CHỈ nhận
    nếu VIF lớn nhất của cả bộ vẫn ≤ ``VIF_MAX``. Biến qua cổng nhưng không được chọn = DỰ BỊ.
    """
    passed = gates[gates["decision"] == "giữ"].set_index("feature")
    targets = {f: set(passed.loc[f, "scenarios"].split(",")) for f in passed.index}
    chosen: List[str] = []
    vif_rejected: Dict[str, float] = {}
    progress = True
    while len(chosen) < TARGET_CORE and progress:
        progress = False
        for s in SCENARIO_ORDER:
            if len(chosen) >= TARGET_CORE:
                break
            pool = [f for f in passed.index if s in targets[f] and f not in chosen and f not in vif_rejected]
            pool.sort(key=lambda f: (-round(auc.loc[f, s], 3), passed.loc[f, "max_rho"], f))
            for f in pool:
                v = vif(post, core_names + chosen + [f]).max()
                if v <= VIF_MAX:
                    chosen.append(f)
                    progress = True
                    break
                vif_rejected[f] = float(v)
    reserve = [f for f in passed.index if f not in chosen]
    return chosen, reserve, vif_rejected


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-day", type=int, default=1)
    ap.add_argument("--end-day", type=int, default=21)
    ap.add_argument("--target-day", type=int, default=None)
    ap.add_argument("--from-cache", action="store_true",
                    help="dùng lại ma trận ở data/features/funnel_v4/ (bỏ qua bước dựng, cổng E đọc từ bảng cũ)")
    args = ap.parse_args()
    target = args.target_day or args.end_day

    import yaml
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "system_config.yaml", encoding="utf-8"))
    TABLES.mkdir(parents=True, exist_ok=True)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Bước 0 + vòng 1 -------------------------------------------------
    catalog = build_catalog()
    space = pd.DataFrame([{
        "name": c.name if c.status != "grammar_invalid" else "",
        "measure": c.measure, "object": c.obj, "entity": c.entity, "window": c.window,
        "status": c.status, "reason": c.reason, "scenarios": ",".join(sorted(c.scenarios)),
        "cost": c.cost, "description": c.description if c.status != "grammar_invalid" else "",
    } for c in catalog])
    space.to_csv(TABLES / "v4_template_space.csv", index=False, encoding="utf-8")
    specs = [c for c in catalog if c.status == "round2"]
    log.info("Không gian %d tổ hợp → %d ứng viên vòng 2", len(catalog), len(specs))

    # ---- Dựng ma trận -----------------------------------------------------
    if args.from_cache:
        mats = {v: pl.read_parquet(MATRIX_DIR / f"{v}.parquet") for v in ("clean", "injected")}
        labels = pl.read_parquet(MATRIX_DIR / "labels.parquet")
        old = pd.read_csv(TABLES / "v4_round2_gates.csv").set_index("feature")
        victims = labels.group_by("scenario").len()
    else:
        mats, plan, clean_parts = build_matrices(args.start_day, args.end_day, target, specs, cfg)
        for v, m in mats.items():
            m.write_parquet(MATRIX_DIR / f"{v}.parquet")
        labels = plan.labels()
        labels.write_parquet(MATRIX_DIR / "labels.parquet")
        victims = labels.group_by("scenario").len()

    core_names = FeatureSchema().core_features
    names = [c.name for c in specs]
    post = mats["clean"].filter(pl.col("day") >= args.start_day + WARMUP_DAYS).to_pandas()

    # ---- Cổng E -----------------------------------------------------------
    t0 = time.time()
    if args.from_cache:
        leak = old["leak_max_abs_diff"].to_dict()
    else:
        core_clean = mats["clean"].select(KEYS + ["entity_type"])
        leak = gate_leakage(clean_parts, specs, core_clean, args.end_day)
    log.info("Cổng E xong (%.1fs)", time.time() - t0)

    # ---- Cổng B -----------------------------------------------------------
    info = gate_information(post, names)

    # ---- Cổng C (đo cho MỌI biến để có bảng bao phủ kịch bản) -------------
    feats_dir = {c.name: c.direction for c in specs}
    feats_dir.update({f: "both" for f in core_names})
    auc = scenario_auc(mats["injected"], labels, feats_dir, target)
    auc.to_csv(TABLES / "v4_scenario_auc.csv", encoding="utf-8", float_format="%.4f")

    # ---- Áp cổng theo thứ tự E → B → A → C ---------------------------------
    order = sorted(specs, key=lambda c: (c.priority, c.cost, c.name))
    corr = correlation(post, core_names + names)
    accepted: List[str] = []
    records = []
    for c in order:
        rec = {
            "feature": c.name, "description": c.description, "scenarios": ",".join(sorted(c.scenarios)),
            "priority": c.priority, "cost": c.cost, "direction": c.direction,
            "leak_max_abs_diff": leak[c.name],
            "null_share": info.loc[c.name, "null_share"], "top_value_share": info.loc[c.name, "top_value_share"],
        }
        others = core_names + accepted
        row = corr.loc[c.name, others]
        rec["max_rho"] = float(row.max())
        rec["max_rho_with"] = str(row.idxmax())
        target_auc = {s: auc.loc[c.name, s] for s in c.scenarios}
        best_s = max(target_auc, key=lambda s: target_auc[s])
        rec["best_target_scenario"] = best_s
        rec["best_target_auc"] = float(target_auc[best_s])

        if not (rec["leak_max_abs_diff"] <= 1e-9):
            rec["decision"], rec["gate"] = "loại", "E"
            rec["reason"] = f"giá trị ngày cũ đổi khi bỏ ngày tương lai (Δmax = {leak[c.name]:.3g})"
        elif rec["null_share"] > NULL_MAX:
            rec["decision"], rec["gate"] = "loại", "B"
            rec["reason"] = f"NULL {rec['null_share']:.1%} > {NULL_MAX:.0%}"
        elif rec["top_value_share"] >= TOP_SHARE_MAX:
            rec["decision"], rec["gate"] = "loại", "B"
            rec["reason"] = f"một giá trị chiếm {rec['top_value_share']:.2%} dòng (≥ {TOP_SHARE_MAX:.0%})"
        elif rec["max_rho"] >= RHO_MAX:
            rec["decision"], rec["gate"] = "loại", "A"
            rec["reason"] = f"|ρ| = {rec['max_rho']:.4f} với {rec['max_rho_with']}"
        elif rec["best_target_auc"] < AUC_MIN:
            rec["decision"], rec["gate"] = "loại", "C"
            rec["reason"] = (f"AUC tốt nhất trên kịch bản mục tiêu = {rec['best_target_auc']:.3f} "
                             f"({best_s}) < {AUC_MIN}")
        else:
            rec["decision"], rec["gate"] = "giữ", ""
            rec["reason"] = f"AUC {rec['best_target_auc']:.3f} trên {best_s}"
            accepted.append(c.name)
        records.append(rec)

    gates = pd.DataFrame(records)
    gates.to_csv(TABLES / "v4_round2_gates.csv", index=False, encoding="utf-8", float_format="%.4f")

    # ---- Tổng kết ---------------------------------------------------------
    counts = space["status"].value_counts()
    summary = [
        ("Bước 0 — tổ hợp sinh ra", len(space)),
        ("Bước 0 — vô nghĩa về ngữ pháp", int(counts.get("grammar_invalid", 0))),
        ("Vòng 1 — R1 đã có / đã đo và loại", int(counts.get("dropped_r1", 0))),
        ("Vòng 1 — R2 không gắn kịch bản 5.2", int(counts.get("dropped_r2", 0))),
        ("Vòng 1 — R3 khó diễn giải", int(counts.get("dropped_r3", 0))),
        ("Vòng 1 — R4 biến thể dư cùng họ", int(counts.get("dropped_r4", 0))),
        ("Vòng 2 — ứng viên được cài đặt", len(specs)),
    ]
    for g in ("E", "B", "A", "C"):
        summary.append((f"Vòng 2 — loại ở cổng {g}", int((gates["gate"] == g).sum())))
    summary.append(("Giữ lại", len(accepted)))
    chosen, reserve, vif_rejected = finalize(post, gates, auc, core_names)
    gates["tier"] = gates["feature"].map(
        lambda f: "core" if f in chosen else ("dự bị" if f in reserve else ""))
    gates["vif_if_added"] = gates["feature"].map(vif_rejected)
    gates.to_csv(TABLES / "v4_round2_gates.csv", index=False, encoding="utf-8", float_format="%.4f")
    summary += [("Chốt — vào core v4", len(chosen)), ("Chốt — dự bị (qua 4 cổng, chưa vào core)", len(reserve))]
    pd.DataFrame(summary, columns=["stage", "count"]).to_csv(
        TABLES / "v4_funnel_summary.csv", index=False, encoding="utf-8")

    v = vif(post, core_names + chosen)
    v.to_csv(TABLES / "v4_final_vif.csv", encoding="utf-8", float_format="%.3f")
    final_corr = correlation(post, core_names + chosen).to_numpy(copy=True)
    np.fill_diagonal(final_corr, 0)
    for s, n in summary:
        print(f"{s:<45} {n:>6}")
    print(f"\nQua 4 cổng ({len(accepted)}):")
    for f in accepted:
        r = gates.set_index("feature").loc[f]
        print(f"  {f:<40} [{r['tier']}] {r['reason']}")
    print(f"\nBị VIF chặn khi thử thêm vào core: { {k: round(x, 2) for k, x in vif_rejected.items()} }")
    print(f"\nCore v4 ({len(core_names) + len(chosen)} biến = {len(core_names)} + {len(chosen)}): "
          f"max |ρ| = {final_corr.max():.4f}, "
          f"VIF max = {v.max():.2f} ({v.idxmax()})")
    print(f"Nạn nhân theo kịch bản: {dict(victims.sort('scenario').iter_rows())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
