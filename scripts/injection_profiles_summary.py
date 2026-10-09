"""
Dựng (hoặc nạp lại) hồ sơ TRAIN của bộ tiêm rồi in thống kê tóm tắt để kiểm tra bằng mắt.

Chạy:
    python scripts/injection_profiles_summary.py                 # dựng từ data/interim, ngày 1..split_day
    python scripts/injection_profiles_summary.py --load          # chỉ nạp hồ sơ đã lưu và in lại
    python scripts/injection_profiles_summary.py --lockout-window 600 --no-save

Hồ sơ được lưu ở ``--out`` (mặc định ``data/features/train_profiles/``, đã nằm trong .gitignore).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.extractor import resolve_feature_config  # noqa: E402
from src.injection.profiles import ProfileConfig, TrainProfiles, build_train_profiles  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - chỉ ảnh hưởng hiển thị console
        pass

QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
MAIN_TYPES = ("User", "Machine")


def _h(title: str) -> None:
    print()
    print(f"== {title} " + "=" * max(0, 76 - len(title)))


def _quantile_table(df: pl.DataFrame, cols, by: str = "entity_type", types=MAIN_TYPES) -> None:
    """Một dòng / (nhóm, cột): n, null, các phân vị."""
    head = f"{'nhóm':<9} {'cột':<24} {'n':>8} {'null':>6} " + " ".join(f"{'q' + format(q, '.2g'):>9}" for q in QUANTILES)
    print(head)
    groups = [("ALL", df)] + [(t, df.filter(pl.col(by) == t)) for t in types]
    for name, g in groups:
        for c in cols:
            s = g[c].drop_nulls().cast(pl.Float64)
            qs = " ".join(f"{s.quantile(q, 'nearest'):>9.3g}" if s.len() else f"{'-':>9}" for q in QUANTILES)
            print(f"{name:<9} {c:<24} {g.height:>8,} {g[c].null_count():>6,} {qs}")


def summarize(p: TrainProfiles, n_samples: int, seed: int) -> None:
    net = p.network
    cfg = net["config"]
    _h("Cấu hình")
    days = net["train_days"]
    print(f"Ngày train đã đọc: {len(days)} ngày ({min(days)}..{max(days)}); thiếu: "
          f"{sorted(set(range(cfg['start_day'], cfg['train_end_day'] + 1)) - set(days)) or 'không'}")
    print(f"Ngoài giờ: hour >= {cfg['off_hours_start']} | hour <= {cfg['off_hours_end']}")
    print(f"Sự kiện có Time ngoài cửa sổ ngày của file: {net['days_outside_window'] or 'không có'}")

    acc = p.accounts
    _h("Tài khoản")
    print(f"{acc.height:,} tài khoản, {int(acc['n_events'].sum()):,} sự kiện")
    print(acc.group_by("entity_type").agg(
        pl.len().alias("n_accounts"), pl.col("n_events").sum(),
        (pl.col("n_active_days") >= 7).sum().alias("active_ge_7d"),
    ).sort("n_accounts", descending=True))
    _quantile_table(acc, ["n_active_days", "median_daily_events", "off_hours_ratio",
                          "n_known_sources", "n_known_loghosts", "max_internal_gap_days"])

    _h("LogonType")
    lt = p.account_logon_types.join(acc.select("DomainName", "UserName", "entity_type"), on=["DomainName", "UserName"])
    by_type = (
        lt.group_by("entity_type", "LogonType").agg(pl.col("n_events").sum(), pl.len().alias("n_accounts"))
        .with_columns((pl.col("n_events") / pl.col("n_events").sum().over("entity_type")).round(4).alias("event_share"))
        .filter(pl.col("entity_type").is_in(MAIN_TYPES))
        .sort("entity_type", "n_events", descending=[False, True])
    )
    with pl.Config(tbl_rows=40):
        print(by_type)
    n_types = lt.group_by("DomainName", "UserName").agg(pl.len().alias("k")).join(
        acc.select("DomainName", "UserName", "entity_type"), on=["DomainName", "UserName"])
    print("Số LogonType khác nhau mỗi tài khoản:")
    print(n_types.group_by("entity_type").agg(
        (pl.col("k") == 1).mean().round(3).alias("frac_1_type"), pl.col("k").median().alias("median_k"), pl.col("k").max().alias("max_k"),
    ).sort("entity_type"))

    _h("Khoảng không hoạt động")
    g = p.account_gaps.join(acc.select("DomainName", "UserName", "entity_type"), on=["DomainName", "UserName"])
    print(g.group_by("entity_type", "kind").agg(
        pl.len().alias("n_gaps"), pl.col("n_days").median().alias("median_days"), pl.col("n_days").max().alias("max_days"),
    ).filter(pl.col("entity_type").is_in(MAIN_TYPES)).sort("entity_type", "kind"))
    internal = g.filter(pl.col("kind") == "internal")
    for k in (3, 7, 14):
        n = internal.filter(pl.col("n_days") >= k).select("DomainName", "UserName").unique().height
        print(f"  tài khoản có khoảng internal >= {k:>2} ngày: {n:,}")

    _h("Máy (toàn mạng)")
    hosts = p.hosts
    print(f"{hosts.height:,} máy (LogHost ∪ Source).")
    _quantile_table(hosts, ["n_accounts", "n_accounts_as_loghost", "n_accounts_as_source", "n_user_accounts", "n_days"],
                    types=())
    print(f"Máy chỉ 1 tài khoản dùng: {hosts.filter(pl.col('n_accounts') == 1).height:,}; "
          f"không User nào dùng: {hosts.filter(pl.col('n_user_accounts') == 0).height:,}")
    print("15 máy nhiều tài khoản nhất:")
    with pl.Config(tbl_rows=15, tbl_width_chars=160):
        print(hosts.head(15))

    _h("Ngưỡng khoá L")
    lock = net["lockout"]
    print(f"cửa sổ chuỗi = {cfg['lockout_window_s']}s, tách lần khoá khi cách >= {cfg['lockout_episode_gap_s']}s")
    print(f"{lock['n_onsets']:,} lần khoá trên {lock['n_accounts']:,} tài khoản; "
          f"{lock['n_onsets_streak0']:,} lần không có bad password đứng trước (loại khỏi ước lượng)")
    print(f"L = {lock['L']} (chiếm {lock['L_support']:.1%} số lần khoá có streak >= 1; trung vị streak = {lock['streak_median']})")
    print(f"L theo entity_type: {lock.get('by_entity_type')}")
    hist = sorted(((int(k), v) for k, v in lock["streak_hist"].items()))
    total = sum(v for _, v in hist) or 1
    print("Phân bố streak (streak: số lần khoá):")
    for k, v in hist[:30]:
        print(f"  {k:>4}: {v:>6,}  {'#' * max(1, round(60 * v / total)) if v else ''}")
    if len(hist) > 30:
        print(f"  ... {len(hist) - 30} giá trị streak lớn hơn, max = {hist[-1][0]}")
    ep = p.lockout_episodes
    if ep.height:
        print("Đối chiếu: n_bad_pw_window (mọi bad password trong cửa sổ, không cần liên tiếp):")
        print(ep.filter(pl.col("n_bad_pw_window") >= 1).group_by("n_bad_pw_window").len()
              .sort("len", descending=True).head(8))

    _h(f"{n_samples} tài khoản User mẫu (>= 7 ngày hoạt động)")
    pool = acc.filter((pl.col("entity_type") == "User") & (pl.col("n_active_days") >= 7))
    for row in pool.sample(min(n_samples, pool.height), seed=seed).iter_rows(named=True):
        d, u = row["DomainName"], row["UserName"]
        print(f"- {d}\\{u}: {row['n_events']:,} sự kiện / {row['n_active_days']} ngày (ngày {row['first_day']}..{row['last_day']}), "
              f"median/ngày = {row['median_daily_events']}, off_hours = {row['off_hours_ratio']:.3f}, "
              f"thất bại = {row['n_failure']:,}")
        srcs = sorted(p.known_sources(d, u))
        hostsu = sorted(p.known_loghosts(d, u))
        print(f"    Source quen ({len(srcs)}): {srcs[:8]}{' ...' if len(srcs) > 8 else ''}")
        print(f"    LogHost quen ({len(hostsu)}): {hostsu[:8]}{' ...' if len(hostsu) > 8 else ''}")
        print(f"    LogonType: { {k: round(v, 3) for k, v in sorted(p.logon_type_shares(d, u).items())} }")
        gaps = p.gaps(d, u).select("gap_start_day", "gap_end_day", "kind").rows()
        print(f"    Khoảng không hoạt động: {gaps if gaps else 'không có'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/system_config.yaml")
    ap.add_argument("--interim-dir", default=None, help="mặc định paths.interim_data_dir")
    ap.add_argument("--out", default="data/features/train_profiles")
    ap.add_argument("--load", action="store_true", help="nạp hồ sơ đã lưu ở --out thay vì dựng lại")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--train-end-day", type=int, default=None,
                    help="ngày train cuối (mặc định evaluation.split_day); 35 = hồ sơ cho khối dev 36–42")
    ap.add_argument("--lockout-window", type=int, default=None, help="giây; mặc định ProfileConfig")
    ap.add_argument("--lockout-episode-gap", type=int, default=None)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.load:
        profiles = TrainProfiles.load(args.out)
        print(f"Đã nạp hồ sơ từ '{args.out}'.")
    else:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        overrides = {}
        if args.lockout_window is not None:
            overrides["lockout_window_s"] = args.lockout_window
        if args.lockout_episode_gap is not None:
            overrides["lockout_episode_gap_s"] = args.lockout_episode_gap
        if args.train_end_day is not None:
            overrides["train_end_day"] = args.train_end_day
        pcfg = ProfileConfig.from_system_config(cfg, **overrides)
        interim = args.interim_dir or (cfg.get("paths", {}) or {}).get("interim_data_dir", "data/interim")
        t0 = time.time()
        profiles = build_train_profiles(interim, pcfg, resolve_feature_config(cfg))
        print(f"Dựng hồ sơ train từ '{interim}' trong {time.time() - t0:.0f}s.")
        if not args.no_save:
            print(f"Đã lưu vào '{profiles.save(args.out)}'.")
    summarize(profiles, args.samples, args.seed)


if __name__ == "__main__":
    main()
