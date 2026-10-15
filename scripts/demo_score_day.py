"""
Demo: chấm điểm log MỘT NGÀY MỚI (vd. ngày 61) bằng mô hình 22 đặc trưng đã train (ngày 1–42).

Luồng (in từng bước để trình bày):
  1. Nạp log ngày mới   : file thô ``wls_day-61.bz2`` (JSON Lines LANL) -> lọc 4624/4625 -> parquet interim;
                          hoặc thư mục đã có ``event_4624/event_4624_day-61.parquet`` + ``event_4625/...``.
  1b. (tuỳ chọn --inject) Tiêm tấn công: chạy ĐÚNG bộ tiêm của benchmark (configs/injection.yaml, khối test,
                          hồ sơ + kho khuôn từ train 1–42) chỉ cho ngày t -> log đè + labels.parquet; bước 4
                          cho biết nạn nhân nằm ở hạng nào (recall top-K, PR-AUC theo kịch bản).
  2. Trích đặc trưng    : cùng code với pipeline chính (``build_account_day_matrix``) trên ngày t−7..t —
                          các ngày lịch sử đọc từ ``data/interim``, ngày t đọc từ log mới. Đặc trưng ``_7d`` cần
                          đủ 7 ngày trước t; ``peer_z_*`` so với mọi tài khoản User cùng ngày nên log phải đủ cả ngày.
                          ``days_since_last_activity`` của tài khoản vắng mặt cả 7 ngày được bổ sung từ ma trận
                          lịch sử ``data/processed/feature_matrix_processed.parquet``.
  3. Chấm điểm          : nạp ``experiments/models/User_22f/<model>.joblib`` (imputer + scaler + mô hình), chấm
                          các dòng User của ngày t, xếp hạng, so ngưỡng train (contamination 5%).
  4. Cảnh báo           : top-K kèm 3 đặc trưng nằm xa nhất ở đuôi phân phối train (phân vị so với train).

    python scripts/demo_score_day.py --log path/to/wls_day-61.bz2
    python scripts/demo_score_day.py --log data/demo/events --day 61 --model isolation_forest --top-k 30
    # diễn tập khi chưa có log thật: tạo "ngày 61" bằng cách dời log ngày 47 (cùng thứ trong tuần) sang ngày 61
    python scripts/demo_score_day.py --simulate-from-day 47 --day 61
    # ... và tiêm 6 kịch bản tấn công vào ngày 61 (1% User như benchmark, hoặc --victims-per-scenario 3)
    python scripts/demo_score_day.py --simulate-from-day 47 --day 61 --inject
    # kiểm tra: chấm lại ngày 60 (đã có trong ma trận) và so 22 đặc trưng với ma trận gốc
    python scripts/demo_score_day.py --day 60 --check-against-matrix

Đầu ra: experiments/demo/day<t>/ (alerts_<model>.csv, scores_<model>.parquet, features/).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path, PureWindowsPath

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_account_day_matrix  # noqa: E402
from src.features.preprocessor import normalize_features  # noqa: E402
from src.injection.layout import SECONDS_PER_DAY, day_of_time, interim_day_path  # noqa: E402
from src.models.pipeline import AnomalyPipeline  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
KEYS = ["DomainName", "UserName"]


def banner(msg: str) -> None:
    print(f"\n=== {msg} " + "=" * max(0, 70 - len(msg)), flush=True)


def ingest(args, events_dir: Path) -> None:
    """Đưa log ngày t về layout interim trong ``events_dir``."""
    if args.simulate_from_day:
        src = args.simulate_from_day
        shift = (args.day - src) * SECONDS_PER_DAY
        for eid in (4624, 4625):
            df = pl.read_parquet(interim_day_path(args.interim, eid, src))
            out = interim_day_path(events_dir, eid, args.day)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.with_columns(pl.col("Time") + shift).write_parquet(out)
            print(f"  [mô phỏng] event_{eid}: ngày {src} -> ngày {args.day}, {df.height:,} sự kiện")
        return
    log = Path(args.log)
    if log.is_dir():
        for eid in (4624, 4625):
            p = interim_day_path(log, eid, args.day)
            if not p.is_file():
                sys.exit(f"Thiếu {p}")
            out = interim_day_path(events_dir, eid, args.day)
            out.parent.mkdir(parents=True, exist_ok=True)
            if p.resolve() != out.resolve():
                shutil.copyfile(p, out)
        return
    from src.data.raw_to_interim import convert_single_bz2, extract_day_tag

    if extract_day_tag(log) != f"day-{args.day:02d}":
        sys.exit(f"Tên file '{log.name}' phải chứa 'day-{args.day:02d}' (vd. wls_day-{args.day:02d}.bz2).")
    counts = convert_single_bz2(log, events_dir, overwrite=True)
    print(f"  Đã lọc từ log thô: {counts}")


def check_day_window(events_dir: Path, day: int) -> None:
    for eid in (4624, 4625):
        p = interim_day_path(events_dir, eid, day)
        d = pl.scan_parquet(p).select(day_of_time().alias("d")).group_by("d").len().collect()
        bad = d.filter(pl.col("d") != day)["len"].sum()
        tot = d["len"].sum()
        print(f"  event_{eid}: {tot:,} sự kiện" + (f" — CẢNH BÁO {bad:,} sự kiện có Time ngoài ngày {day}" if bad else ""))


def fill_recency(day_df: pl.DataFrame, matrix: pl.DataFrame, day: int) -> pl.DataFrame:
    """Tài khoản không hoạt động trong 7 ngày nạp: lấy ngày hoạt động gần nhất từ ma trận lịch sử."""
    last = (matrix.filter(pl.col("day") < day).group_by(KEYS).agg(pl.col("day").max().alias("_last")))
    out = day_df.join(last, on=KEYS, how="left").with_columns(
        pl.when(pl.col("days_since_last_activity").is_null() & pl.col("_last").is_not_null())
        .then(pl.lit(day) - pl.col("_last")).otherwise(pl.col("days_since_last_activity"))
        .cast(pl.Int32).alias("days_since_last_activity")
    )
    n = int((day_df["days_since_last_activity"].is_null() & out["days_since_last_activity"].is_not_null()).sum())
    print(f"  days_since_last_activity: bổ sung từ lịch sử cho {n:,} tài khoản vắng mặt 7 ngày qua")
    return out.drop("_last")


def inject_day(args, events_dir: Path | None) -> tuple[Path, pl.DataFrame]:
    """Chạy bộ tiêm của benchmark cho riêng ngày t; trả (thư mục log đè, nhãn)."""
    import dataclasses

    from src.evaluation.labeled_eval import load_eval_labels
    from src.injection.inject import InjectionConfig, InjectionRunner
    from src.injection.layout import RunLayout

    cfg = InjectionConfig.from_files("test", run_id=f"demo_day{args.day:02d}", seed=args.inject_seed)
    common = dict(cfg.common, eval_interim_dir=str(events_dir) if events_dir else None)
    scen = {k: dict(v) for k, v in cfg.scenarios.items()}
    if args.victims_per_scenario:
        common["target_rate"] = {"enabled": False}
        for name, v in scen.items():
            if name == "password_spraying":
                v.update(n_campaigns=1, victims_per_campaign=[args.victims_per_scenario] * 2)
            else:
                v["n_victims"] = args.victims_per_scenario
    cfg = dataclasses.replace(cfg, days=[args.day], common=common, scenarios=scen)
    root = RunLayout.for_run(common["runs_dir"], cfg.run_id).root
    if root.exists():
        shutil.rmtree(root)
    t0 = time.time()
    info = InjectionRunner(cfg).build()
    man = pl.read_csv(info["manifest"], infer_schema_length=None)
    print(f"  {info['n_injections']} lần tiêm, {info['n_injected']:,} sự kiện giả mạo, {time.time() - t0:.0f}s "
          f"-> {info['root']}")
    per = man.group_by("scenario").agg(pl.len().alias("n"), pl.col("n_events").sum().alias("ev")).sort("scenario")
    for r in per.iter_rows(named=True):
        print(f"    {r['scenario']:24s} {r['n']:3d} nạn nhân, {r['ev']:6,} sự kiện")
    return RunLayout.for_run(common["runs_dir"], cfg.run_id).events_dir, load_eval_labels(info["labels"])


def report_detection(res: pl.DataFrame, top_k: int) -> None:
    """Nạn nhân được tiêm nằm ở hạng nào (res đã sort theo rank)."""
    from sklearn.metrics import average_precision_score

    y = res["is_anomaly"].to_numpy()
    n_pos = int(y.sum())
    print(f"\n  Nhãn: {n_pos} User bị tiêm / {res.height:,} User ngày này ({n_pos / res.height:.2%}); "
          f"PR-AUC = {average_precision_score(y, res['score'].to_numpy()):.3f} (ngẫu nhiên ≈ {n_pos / res.height:.3f})")
    for k in sorted({top_k, 50, 100}):
        hit = int(y[:k].sum())
        print(f"  Top {k:<4}: bắt {hit}/{n_pos} nạn nhân (recall {hit / max(n_pos, 1):.0%}, precision {hit / k:.0%})")
    pos = res.filter(pl.col("is_anomaly") == 1)
    print("  Theo kịch bản (hạng của từng nạn nhân, trên tổng {:,}):".format(res.height))
    for r in pos.group_by("scenario").agg(pl.col("rank").sort()).sort("scenario").iter_rows(named=True):
        ranks = r["rank"]
        print(f"    {r['scenario']:24s} trong top 100: {sum(x <= 100 for x in ranks)}/{len(ranks)}  | hạng: "
              + ", ".join(str(x) for x in ranks[:12]) + (" ..." if len(ranks) > 12 else ""))


def main() -> None:
    p = argparse.ArgumentParser(description="Demo chấm điểm log một ngày mới bằng mô hình đã train")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--log", help="wls_day-NN.bz2 thô hoặc thư mục layout interim chứa ngày --day")
    g.add_argument("--simulate-from-day", type=int, help="Diễn tập: dời log ngày này (data/interim) sang --day")
    p.add_argument("--day", type=int, default=61)
    p.add_argument("--model", default="one_class_svm",
                   choices=["one_class_svm", "isolation_forest", "local_outlier_factor"])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--history-days", type=int, default=7)
    p.add_argument("--interim", default="data/interim")
    p.add_argument("--models-dir", default="experiments/models/User_22f")
    p.add_argument("--matrix", default="data/processed/feature_matrix_processed.parquet",
                   help="Ma trận lịch sử (bổ sung recency, thống kê train để giải thích, kiểm tra)")
    p.add_argument("--out", default=None)
    p.add_argument("--inject", action="store_true", help="Tiêm 6 kịch bản tấn công vào ngày --day (có nhãn)")
    p.add_argument("--inject-seed", type=int, default=20261061)
    p.add_argument("--victims-per-scenario", type=int, default=None,
                   help="Số nạn nhân mỗi kịch bản (mặc định: tỉ lệ mục tiêu 1%% User như benchmark)")
    p.add_argument("--check-against-matrix", action="store_true",
                   help="Ngày --day đã có trong --matrix: so 22 đặc trưng tính lại với ma trận")
    args = p.parse_args()

    out = Path(args.out or f"experiments/demo/day{args.day:02d}")
    manifest = json.loads((Path(args.models_dir) / "demo_manifest.json").read_text(encoding="utf-8"))
    feats = manifest["features"]
    t_all = time.time()

    banner(f"1. Nạp log ngày {args.day}")
    events_dir = None
    if args.log or args.simulate_from_day:
        events_dir = out / "events"
        if events_dir.exists():
            shutil.rmtree(events_dir)
        ingest(args, events_dir)
        check_day_window(events_dir, args.day)
    elif not interim_day_path(args.interim, 4624, args.day).is_file():
        sys.exit(f"Ngày {args.day} không có trong {args.interim}: truyền --log hoặc --simulate-from-day.")
    else:
        print(f"  Dùng log ngày {args.day} có sẵn trong {args.interim}")

    labels = None
    if args.inject:
        banner(f"1b. Tiêm tấn công vào ngày {args.day}")
        events_dir, labels = inject_day(args, events_dir)

    start = args.day - args.history_days
    banner(f"2. Trích đặc trưng (lịch sử ngày {start}–{args.day - 1} + ngày {args.day})")
    t0 = time.time()
    raw = build_account_day_matrix(start_day=start, end_day=args.day, interim_dir=args.interim,
                                   output_dir=out / "features", events_dir=events_dir,
                                   history_window_days=args.history_days)
    day_df = normalize_features(raw).filter(pl.col("day") == args.day)
    print(f"  {raw.height:,} dòng (tài khoản × ngày) cho {args.history_days + 1} ngày, {time.time() - t0:.0f}s")
    matrix = pl.read_parquet(args.matrix, columns=KEYS + ["day", "entity_type"] + feats)
    day_df = fill_recency(day_df, matrix, args.day)
    users = day_df.filter(pl.col("entity_type") == manifest["segment"]).sort(KEYS)
    print(f"  Ngày {args.day}: {day_df.height:,} tài khoản, trong đó {users.height:,} User được chấm "
          f"({len(feats)} đặc trưng)")

    if args.check_against_matrix:
        ref = matrix.filter((pl.col("day") == args.day) & (pl.col("entity_type") == manifest["segment"]))
        j = users.select(KEYS + feats).join(ref.select(KEYS + feats), on=KEYS, suffix="_ref")
        print(f"  [kiểm tra] khớp {j.height:,}/{ref.height:,} User với ma trận gốc; số dòng lệch theo đặc trưng:")
        for f in feats:
            a, b = j[f].cast(pl.Float64), j[f + "_ref"].cast(pl.Float64)
            diff = ((a - b).abs() > 1e-6) | (a.is_null() != b.is_null())
            n = int(diff.fill_null(False).sum())
            if n:
                print(f"    {f:32s} {n:,}")
        print("    (không in đặc trưng nào = khớp hoàn toàn)")

    banner(f"3. Chấm điểm bằng {args.model} ({manifest['models'][args.model]['config_id']})")
    # manifest ghi đường dẫn của máy train (Windows: dấu \) -> chỉ lấy tên file, nạp từ --models-dir
    pipe = AnomalyPipeline.load(Path(args.models_dir) / PureWindowsPath(manifest["models"][args.model]["path"]).name)
    t0 = time.time()
    scores = np.asarray(pipe.score(users.select(feats).to_pandas()), dtype=np.float64)
    thr = float(manifest["models"][args.model]["threshold"])
    print(f"  Chấm {users.height:,} User trong {time.time() - t0:.1f}s; ngưỡng train (top 5% train) = {thr:.6g}; "
          f"vượt ngưỡng: {(scores >= thr).sum():,} ({(scores >= thr).mean():.1%})")
    n_tie = int((scores == scores.max()).sum())
    if n_tie > 1:
        print(f"  Lưu ý: {n_tie:,} User cùng điểm cao nhất (điểm bão hoà) — thứ hạng trong nhóm này không có ý nghĩa.")
    res = users.select(KEYS + feats).with_columns(pl.Series("score", scores)).with_columns(
        pl.col("score").rank("ordinal", descending=True).cast(pl.Int32).alias("rank"),
        (pl.col("score") >= thr).alias("above_train_threshold"),
    ).sort("rank")

    if labels is not None:
        from src.evaluation.labeled_eval import align_eval_labels

        lab = align_eval_labels(res.select(KEYS).with_columns(pl.lit(args.day, pl.Int64).alias("day")), labels)
        res = res.with_columns(lab["is_anomaly"].alias("is_anomaly"), lab["scenario"].alias("scenario"))

    banner(f"4. Top {args.top_k} cảnh báo ngày {args.day}")
    train = matrix.filter((pl.col("day") <= manifest["split_day"]) & (pl.col("entity_type") == manifest["segment"]))
    ref = {f: np.sort(train[f].drop_nulls().cast(pl.Float64).to_numpy()) for f in feats}

    def pct(f: str, x: float) -> float:
        """Phân vị (mid-rank) của x trong phân phối train của đặc trưng f."""
        a = ref[f]
        return (np.searchsorted(a, x, "left") + np.searchsorted(a, x, "right")) / 2 / len(a)

    rows = []
    for r in res.head(args.top_k).iter_rows(named=True):
        q = {f: pct(f, r[f]) for f in feats if r[f] is not None}
        why = sorted(q, key=lambda f: min(q[f], 1 - q[f]))[:3]  # đặc trưng nằm xa nhất ở đuôi phân phối train
        reason = "; ".join(f"{f}={r[f]:.3g} ({'>' if q[f] >= 0.5 else '<'}{max(q[f], 1 - q[f]):.2%} train)"
                           for f in why)
        tag = f"  <== TIÊM: {r['scenario']}" if labels is not None and r["is_anomaly"] == 1 else ""
        rows.append({"rank": r["rank"], "injected": r.get("scenario"), "account": f"{r['DomainName']}\\{r['UserName']}",
                     "score": r["score"], "why": reason})
        print(f"  #{r['rank']:<3} {r['DomainName']}\\{r['UserName']:<14} score {r['score']:.6g}  | {reason}{tag}")

    if labels is not None:
        report_detection(res, args.top_k)

    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out / f"alerts_{args.model}.csv")
    res.write_parquet(out / f"scores_{args.model}.parquet")
    print(f"\nXong trong {time.time() - t_all:.0f}s -> {out}/alerts_{args.model}.csv, scores_{args.model}.parquet")


if __name__ == "__main__":
    main()
