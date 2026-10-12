"""
Tạo bộ review thủ công top 50 cảnh báo (tuần 5, đề cương mục 7 và mục "review top 50").

Log GỐC, không tiêm: fit trên ngày 1–split, chấm ngày split+1..60 (phân khúc User, cấu hình chốt từ dev).
Lấy 50 cảnh báo điểm cao nhất của MỖI mô hình ML, hợp lại, rồi dựng "thẻ cảnh báo" cho từng cặp (tài khoản, ngày):
đặc trưng lệch nhất so với train, ngữ cảnh tài khoản, tóm tắt log thô ngày đó.

Làm mù: thứ tự xáo ngẫu nhiên, phiếu chấm không có tên mô hình hay điểm; bảng khoá để riêng.

    python scripts/evaluation/build_review_set.py

Đầu ra: experiments/analysis/review_top50/
  review_sheet.csv      phiếu chấm (mở bằng Excel / Google Sheets) — cột nhãn cho người chấm 1 và 2
  alert_cards.html      thẻ cảnh báo để đọc khi chấm
  answer_key.csv        alert_id -> mô hình, hạng, điểm (KHÔNG mở trước khi chấm xong)
  review_manifest.json  cấu hình, độ trùng top 50 giữa các mô hình
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.grid import _params_with, fit_and_score, load_grid_config  # noqa: E402
from src.features.extractor import load_day_events  # noqa: E402
from src.models.registry import load_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("ueba_benchmark.review_set")

KEYS = ["DomainName", "UserName", "day"]
NON_FEATURES = set(KEYS) | {"entity_type"}
LABELS = "a = đáng ngờ, cần điều tra | b = bất thường nhưng giải thích được | c = nhiễu"
LOGON_TYPES = {2: "Interactive", 3: "Network", 4: "Batch", 5: "Service", 7: "Unlock", 8: "NetworkCleartext",
               9: "NewCredentials", 10: "RemoteInteractive", 11: "CachedInteractive"}


def score_original(matrix: Path, cfg: dict, params_path: str, split_day: int, eval_days: list, seed: int):
    df = pl.read_parquet(matrix).sort(["day", "DomainName", "UserName"])
    if cfg.get("segment"):
        df = df.filter(pl.col("entity_type") == cfg["segment"])
    train = df.filter(pl.col("day") <= split_day).to_pandas()
    ev = df.filter(pl.col("day").is_in(eval_days)).to_pandas()
    data = {"train": train, "eval": ev}
    params = load_params(params_path)
    c0 = float(cfg.get("contamination", 0.05))
    scores = {}
    for model, confs in cfg["models"].items():
        c = confs[0]  # lưới test: một cấu hình chốt / mô hình
        res = fit_and_score(data, model, _params_with(params, model, c.get("params") or {}),
                            c.get("scaler", "robust"), seed, c0)
        scores[model] = res["scores"]
        logger.info("%s (%s): fit %.0fs, ngưỡng %.4f", model, c["id"], res["fit_seconds"], res["threshold"])
    return train, ev, scores


def feature_deviation(train: pd.DataFrame, rows: pd.DataFrame, top: int = 5) -> list:
    feats = [c for c in train.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(train[c])]
    t = train[feats]
    med, q99, q999 = t.median(), t.quantile(0.99), t.quantile(0.999)
    scale = ((t.quantile(0.75) - t.quantile(0.25)) / 1.349).where(lambda s: s > 1e-9, t.std()).replace(0, np.nan)
    out = []
    for _, r in rows.iterrows():
        z = ((r[feats].astype(float) - med) / scale).replace([np.inf, -np.inf], np.nan).dropna()
        best = z.abs().sort_values(ascending=False).head(top).index
        out.append([{"feature": f, "value": float(r[f]), "train_median": float(med[f]), "train_p99": float(q99[f]),
                     "z": float(z[f]), "beyond_p999": bool(r[f] > q999[f])} for f in best])
    return out


def account_context(train: pd.DataFrame, rows: pd.DataFrame) -> list:
    g = train.groupby(["DomainName", "UserName"])
    ctx = pd.DataFrame({"train_active_days": g.size(), "train_median_logons": g["total_logons"].median()})
    out = []
    for _, r in rows.iterrows():
        k = (r["DomainName"], r["UserName"])
        out.append({"train_active_days": int(ctx.loc[k, "train_active_days"]) if k in ctx.index else 0,
                    "train_median_logons": float(ctx.loc[k, "train_median_logons"]) if k in ctx.index else None})
    return out


def raw_summaries(alerts: pd.DataFrame, interim: Path, lookback: int = 7) -> dict:
    """Tóm tắt log thô ngày cảnh báo + Source/LogHost đã thấy trong `lookback` ngày trước (để đếm cái mới)."""
    users = set(alerts["UserName"])
    need = sorted({d for day in alerts["day"] for d in range(int(day) - lookback, int(day) + 1) if d >= 1})
    by_day = {}
    for d in need:
        ev = load_day_events(d, interim)
        if ev is not None:
            by_day[d] = ev.filter(pl.col("UserName").is_in(list(users)))
        logger.info("Đọc log thô ngày %02d (%d sự kiện của tài khoản cảnh báo)", d, by_day[d].height if d in by_day else 0)
    out = {}
    for _, a in alerts.iterrows():
        dom, user, day = a["DomainName"], a["UserName"], int(a["day"])
        acc = (pl.col("DomainName") == dom) & (pl.col("UserName") == user)
        today = by_day.get(day, pl.DataFrame()).filter(acc) if day in by_day else pl.DataFrame()
        past = [by_day[d].filter(acc) for d in range(day - lookback, day) if d in by_day]
        past = pl.concat(past) if past else today.clear()
        if today.height == 0:
            out[(dom, user, day)] = {"n_events": 0}
            continue
        t = today.with_columns(((pl.col("Time") % 86400) // 3600).alias("hour"))
        seen_src = set(past["Source"].drop_nulls()) if past.height else set()
        seen_host = set(past["LogHost"].drop_nulls()) if past.height else set()
        src, host = set(t["Source"].drop_nulls()), set(t["LogHost"].drop_nulls())
        vc = lambda col: {str(k): int(v) for k, v in t[col].value_counts(sort=True).head(6).iter_rows()}  # noqa: E731
        out[(dom, user, day)] = {
            "n_events": t.height,
            "n_4624": int((t["EventID"] == 4624).sum()), "n_4625": int((t["EventID"] == 4625).sum()),
            "logon_types": {LOGON_TYPES.get(int(k), str(k)) if k is not None else "null": v
                            for k, v in t["LogonType"].value_counts(sort=True).head(6).iter_rows()},
            "n_sources": len(src), "n_new_sources_7d": len(src - seen_src),
            "n_loghosts": len(host), "n_new_loghosts_7d": len(host - seen_host),
            "past_7d_events": past.height,
            "failure_reasons": vc("FailureReason") if t["FailureReason"].drop_nulls().len() else {},
            "top_sources": vc("Source"), "top_loghosts": vc("LogHost"), "processes": vc("ProcessName"),
            "auth_packages": vc("AuthenticationPackage"),
            "hours": {int(k): int(v) for k, v in t["hour"].value_counts().sort("hour").iter_rows()},
            "sample": t.sort("Time").head(12).select(
                ["Time", "EventID", "LogonType", "Source", "LogHost", "AuthenticationPackage", "ProcessName",
                 "FailureReason"]).to_dicts(),
        }
    return out


def render_html(cards: list, out: Path) -> None:
    def kv(d):
        return ", ".join(f"{html.escape(str(k))}: {v}" for k, v in d.items()) if d else "—"

    parts = ["""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Thẻ cảnh báo top 50</title>
<style>body{font-family:system-ui,sans-serif;margin:16px;max-width:1100px;color:#1e293b;background:#fff}
.card{border:1px solid #cbd5e1;border-radius:8px;padding:12px 16px;margin:14px 0}
h2{margin:0 0 6px;font-size:18px;color:#12305c}table{border-collapse:collapse;font-size:13px;margin:6px 0}
td,th{border:1px solid #e2e8f0;padding:3px 7px;text-align:left}th{background:#f1f5f9}
.hi{color:#b91c1c;font-weight:600}.muted{color:#64748b;font-size:13px}</style></head><body>
<h1>Thẻ cảnh báo — review top 50 (log gốc, ngày test)</h1>
<p class="muted">Nhãn: """ + html.escape(LABELS) + """. Chấm độc lập, ghi nhãn vào review_sheet.csv theo alert_id.
Không mở answer_key.csv trước khi chấm xong.</p>"""]
    for c in cards:
        r = c["raw"]
        feat_rows = "".join(
            f"<tr><td>{html.escape(f['feature'])}</td><td>{f['value']:.4g}</td><td>{f['train_median']:.4g}</td>"
            f"<td>{f['train_p99']:.4g}</td><td class=\"{'hi' if f['beyond_p999'] else ''}\">{f['z']:+.1f}</td></tr>"
            for f in c["features"])
        sample = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in s.values()) + "</tr>"
                         for s in r.get("sample", []))
        parts.append(f"""<div class="card" id="{c['alert_id']}"><h2>{c['alert_id']} — {html.escape(c['UserName'])}
@ {html.escape(c['DomainName'])}, ngày {c['day']}</h2>
<p>Train: hoạt động {c['context']['train_active_days']}/42 ngày, trung vị {c['context']['train_median_logons']} đăng nhập/ngày.
Hôm nay: <b>{r.get('n_events', 0)}</b> sự kiện (4624: {r.get('n_4624', 0)}, 4625: {r.get('n_4625', 0)}); 7 ngày trước: {r.get('past_7d_events', 0)} sự kiện.</p>
<p>Source: {r.get('n_sources', 0)} ({r.get('n_new_sources_7d', 0)} mới so với 7 ngày) · LogHost: {r.get('n_loghosts', 0)}
({r.get('n_new_loghosts_7d', 0)} mới)<br>LogonType: {kv(r.get('logon_types'))}<br>Lý do thất bại: {kv(r.get('failure_reasons'))}
<br>Giờ (UTC giờ trong ngày: số sự kiện): {kv(r.get('hours'))}<br>Source hay gặp: {kv(r.get('top_sources'))}
<br>LogHost hay gặp: {kv(r.get('top_loghosts'))}<br>Tiến trình: {kv(r.get('processes'))} · Xác thực: {kv(r.get('auth_packages'))}</p>
<table><tr><th>Đặc trưng lệch nhất</th><th>Giá trị</th><th>Trung vị train</th><th>p99 train</th><th>z bền vững</th></tr>{feat_rows}</table>
<details><summary>12 sự kiện đầu ngày</summary><table><tr><th>Time</th><th>EventID</th><th>LogonType</th><th>Source</th>
<th>LogHost</th><th>Auth</th><th>Process</th><th>FailureReason</th></tr>{sample}</table></details></div>""")
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Tạo bộ review top 50 cảnh báo trên log gốc")
    p.add_argument("--config", default="configs/benchmark_grid_test.yaml")
    p.add_argument("--model-params", default="configs/model_params.yaml")
    p.add_argument("--matrix", default="data/processed/feature_matrix_processed.parquet")
    p.add_argument("--interim", default="data/interim")
    p.add_argument("--split-day", type=int, default=42)
    p.add_argument("--last-day", type=int, default=60)
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="experiments/analysis/review_top50")
    args = p.parse_args()

    cfg = load_grid_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_days = list(range(args.split_day + 1, args.last_day + 1))
    train, ev, scores = score_original(Path(args.matrix), cfg, args.model_params, args.split_day, eval_days, args.seed)

    key_rows, tops = [], {}
    for model, s in scores.items():
        order = np.argsort(-s, kind="stable")[: args.top]
        tops[model] = set(map(tuple, ev.iloc[order][KEYS].astype(str).to_numpy()))
        for rank, i in enumerate(order, 1):
            key_rows.append({**ev.iloc[i][KEYS].to_dict(), "model": model, "rank": rank, "score": float(s[i])})
    key = pd.DataFrame(key_rows)
    alerts = key[KEYS].drop_duplicates().sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    alerts.insert(0, "alert_id", [f"A{i + 1:03d}" for i in range(len(alerts))])
    key = key.merge(alerts, on=KEYS)
    logger.info("%d cảnh báo duy nhất từ %d mô hình × top %d", len(alerts), len(scores), args.top)

    rows = alerts.merge(ev, on=KEYS, how="left")
    feats, ctx = feature_deviation(train, rows), account_context(train, rows)
    raw = raw_summaries(alerts, Path(args.interim))
    cards = [{**a, "features": f, "context": c, "raw": raw.get((a["DomainName"], a["UserName"], int(a["day"])), {})}
             for a, f, c in zip(alerts.to_dict("records"), feats, ctx)]
    render_html(cards, out / "alert_cards.html")

    sheet = alerts.copy()
    sheet["tom_tat"] = [f"{c['raw'].get('n_events', 0)} sự kiện, {c['raw'].get('n_4625', 0)} thất bại, "
                        f"{c['raw'].get('n_new_sources_7d', 0)} Source mới, {c['raw'].get('n_new_loghosts_7d', 0)} LogHost mới; "
                        f"lệch nhất: {', '.join(f['feature'] for f in c['features'][:3])}" for c in cards]
    for col in ["nhan_nguoi_1", "ghi_chu_1", "nhan_nguoi_2", "ghi_chu_2", "nhan_thong_nhat", "ghi_chu_thong_nhat"]:
        sheet[col] = ""
    sheet.to_csv(out / "review_sheet.csv", index=False, encoding="utf-8-sig")
    key.sort_values(["model", "rank"]).to_csv(out / "answer_key.csv", index=False, encoding="utf-8-sig")
    (out / "alert_cards.json").write_text(json.dumps(cards, ensure_ascii=False, default=str, indent=1),
                                          encoding="utf-8")

    names = list(tops)
    overlap = {f"{a}&{b}": len(tops[a] & tops[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    overlap["all"] = len(set.intersection(*tops.values()))
    (out / "review_manifest.json").write_text(json.dumps({
        "matrix": args.matrix, "segment": cfg.get("segment"), "split_day": args.split_day, "eval_days": eval_days,
        "models": {m: cfg["models"][m][0]["id"] for m in cfg["models"]}, "seed": args.seed, "top": args.top,
        "n_alerts": len(alerts), "top_overlap": overlap, "labels": LABELS}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    logger.info("Đã ghi bộ review vào %s (trùng top %d: %s)", out, args.top, overlap)


if __name__ == "__main__":
    main()
