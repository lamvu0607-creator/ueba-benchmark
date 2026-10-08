"""
Đếm tài khoản ứng viên cho kịch bản 5 (ngủ đông) và 6 (đổi logon type) TRƯỚC khi viết hai kịch bản đó,
kèm một khảo sát nhanh các trường log để kiểm tra giả định của operations.py.

Chạy (sau khi đã có hồ sơ train và baseline trên log gốc):
    python scripts/injection_candidates.py                    # khối dev
    python scripts/injection_candidates.py --block test
    python scripts/injection_candidates.py --survey-day 16    # khảo sát trường trên một ngày TRAIN

Nguồn dữ liệu và phạm vi nhìn:
  * thống kê hồ sơ (ngày hoạt động, khoảng trống train, LogonType, Source/LogHost quen): CHỈ từ train
    (``data/features/train_profiles/``);
  * ngày test: chỉ đọc SỰ HIỆN DIỆN (tài khoản có/không có sự kiện) — đúng quyết định 2026-10-08
    "kịch bản ngủ đông được xem sự hiện diện ở các ngày test ≤ t";
  * loại (tài khoản, ngày) luật ECDF đã cảnh báo trên log gốc nếu có file baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.injection.profiles import ACCOUNT_KEYS, TrainProfiles  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass


def _h(title: str) -> None:
    print()
    print(f"== {title} " + "=" * max(0, 76 - len(title)))


def _norm_domain() -> pl.Expr:
    c = pl.col("DomainName")
    return (
        pl.when(c.is_null() | (c.str.strip_chars() == "")).then(pl.lit("Unknown"))
        .otherwise(c.str.strip_chars().str.to_lowercase()).alias("DomainName")
    )


def _valid_user() -> pl.Expr:
    u = pl.col("UserName")
    return u.is_not_null() & (u.str.strip_chars() != "") & u.str.to_lowercase().is_in(["null", "none", "nan"]).not_()


def day_presence(interim: Path, day: int) -> Optional[pl.DataFrame]:
    """(DomainName chuẩn hoá, UserName) có ít nhất một sự kiện 4624/4625 trong ngày ``day``."""
    paths = [interim / f"event_{e}" / f"event_{e}_day-{day:02d}.parquet" for e in (4624, 4625)]
    if not all(p.is_file() for p in paths):
        print(f"  [cảnh báo] thiếu file interim ngày {day} — coi như không có dữ liệu")
        return None
    lf = pl.concat([pl.scan_parquet(p).select(["UserName", "DomainName"]) for p in paths])
    return lf.filter(_valid_user()).select(_norm_domain(), "UserName").unique().with_columns(
        pl.lit(day, pl.Int32).alias("day")
    ).collect()


def rule_alerted(common: Dict[str, Any]) -> Optional[pl.DataFrame]:
    """Khoá (DomainName, UserName, day) luật ECDF đã cờ trên log gốc; None nếu chưa có file."""
    rc = common.get("rule_exclusion", {}) or {}
    if not rc.get("enabled", True):
        return None
    scores_p, thr_p = Path(rc["baseline_scores"]), Path(rc["thresholds"])
    if not (scores_p.is_file() and thr_p.is_file()):
        print(f"  [cảnh báo] chưa có '{scores_p}' / '{thr_p}' — CHƯA loại tài khoản-ngày luật đã cảnh báo")
        return None
    meta_p = Path(rc.get("event_stats_meta", ""))
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("injected_events_dir") is not None:
            raise SystemExit(f"'{meta_p}' cho thấy baseline chạy trên log ĐÃ TIÊM — chạy lại trên log gốc.")
    seg, method = rc.get("segment", "User"), rc.get("method", "rule_ecdf")
    thr = float(json.loads(thr_p.read_text(encoding="utf-8"))["thresholds"][seg][method]["threshold"])
    flagged = (
        pl.scan_parquet(scores_p)
        .filter((pl.col("segment") == seg) & (pl.col("split") == "eval") & (pl.col(method) > thr))
        .select(ACCOUNT_KEYS + [pl.col("day").cast(pl.Int32)])
        .collect()
    )
    print(f"  luật {method} (ngưỡng {thr:.5f}) đã cờ {flagged.height:,} dòng {seg} trên tập đánh giá")
    return flagged


def survey_fields(interim: Path, day: int, user_keys: pl.DataFrame) -> None:
    """Khảo sát trường của MỘT ngày train để kiểm tra giả định của bộ tiêm."""
    _h(f"Khảo sát trường — ngày train {day} (chỉ tài khoản User đủ điều kiện)")
    frames = []
    for e in (4624, 4625):
        p = interim / f"event_{e}" / f"event_{e}_day-{day:02d}.parquet"
        if not p.is_file():
            print(f"  thiếu {p}")
            return
        frames.append(pl.read_parquet(p))
    ev = pl.concat(frames, how="vertical_relaxed").filter(_valid_user()).with_columns(_norm_domain().alias("_dom"))
    ev = ev.join(user_keys.rename({"DomainName": "_dom"}), on=["_dom", "UserName"], how="semi")
    print(f"  {ev.height:,} sự kiện; schema: {dict(ev.drop('_dom').schema)}")
    print("  tỷ lệ null theo cột (4624 | 4625):")
    for c in ev.columns:
        if c == "_dom":
            continue
        r = [ev.filter(pl.col("EventID") == e)[c].null_count() / max(1, ev.filter(pl.col("EventID") == e).height)
             for e in (4624, 4625)]
        print(f"    {c:<22} {r[0]:7.1%} | {r[1]:7.1%}")
    same = ev.select((pl.col("SubjectUserName") == pl.col("UserName")).mean()).item()
    print(f"  SubjectUserName == UserName: {same if same is None else f'{same:.1%}'}")
    print(f"  Destination khác null: {ev['Destination'].is_not_null().mean():.2%}; "
          f"Destination == LogHost: {ev.select((pl.col('Destination') == pl.col('LogHost')).mean()).item()}")
    print("  FailureReason (4625):")
    print(ev.filter(pl.col("EventID") == 4625).group_by("FailureReason").len().sort("len", descending=True))
    print("  LogonType × AuthenticationPackage (4624):")
    print(ev.filter(pl.col("EventID") == 4624).group_by(["LogonType", "LogonTypeDescription", "AuthenticationPackage"])
          .len().sort(["LogonType", "len"], descending=[False, True]))
    print("  mẫu LogonID / SubjectLogonID / ProcessName:")
    print(ev.select(["EventID", "LogonType", "LogonID", "SubjectLogonID", "ProcessName"]).head(8))


def dormant_candidates(
    base: pl.DataFrame, prof: TrainProfiles, presence: pl.DataFrame, block_days: List[int], split_day: int
) -> None:
    """Ứng viên kịch bản 5: ngày t của khối mà tài khoản KHÔNG hoạt động, đứng sau ≥ G ngày trống liên tiếp."""
    _h("Kịch bản 5 — ngủ đông hoạt động lại")
    train_days = [int(d) for d in prof.network["train_days"]]
    test_days_read = sorted(set(presence["day"].to_list())) if presence.height else []
    calendar = sorted(set(train_days) | set(range(split_day + 1, max(block_days) + 1)))
    active = pl.concat([
        prof.daily_counts.select(ACCOUNT_KEYS + [pl.col("day").cast(pl.Int32)]),
        presence.select(ACCOUNT_KEYS + ["day"]),
    ]).join(base.select(ACCOUNT_KEYS), on=ACCOUNT_KEYS, how="semi").unique()

    rank = {d: i for i, d in enumerate(calendar)}
    act_sets: Dict[Tuple[str, str], Set[int]] = {}
    for dom, user, d in active.iter_rows():
        act_sets.setdefault((dom, user), set()).add(int(d))
    max_gap = {(r["DomainName"], r["UserName"]): (r["max_internal_gap_days"] or 0) for r in base.iter_rows(named=True)}

    rows = []
    for key, days in act_sets.items():
        for t in block_days:
            if t in days or t not in rank:
                continue
            prev = [d for d in days if d < t]
            if not prev:
                continue
            gap = rank[t] - rank[max(prev)] - 1          # số ngày trống liên tiếp NGAY TRƯỚC t (không tính t)
            rows.append((key[0], key[1], t, gap, max_gap[key]))
    cand = pl.DataFrame(rows, schema={"DomainName": pl.String, "UserName": pl.String, "day": pl.Int32,
                                      "gap": pl.Int32, "max_internal_gap": pl.Int32}, orient="row")
    if not test_days_read:
        print("  [cảnh báo] không đọc được ngày test nào — số dưới đây chỉ dựa vào train")
    print(f"  ngày test đã đọc hiện diện: {test_days_read}")
    print(f"  {'luật':<34} {'(TK, ngày)':>12} {'TK khác nhau':>14}")
    for g in (1, 3, 5, 7, 10, 14, 21):
        sub = cand.filter(pl.col("gap") >= g)
        print(f"  {'fixed: gap ≥ ' + str(g):<34} {sub.height:>12,} {sub.select(ACCOUNT_KEYS).unique().height:>14,}")
    for floor in (3, 7, 10):
        sub = cand.filter((pl.col("gap") > pl.col("max_internal_gap")) & (pl.col("gap") >= floor))
        print(f"  {'account_max: gap > max nội bộ, ≥ ' + str(floor):<34} {sub.height:>12,} "
              f"{sub.select(ACCOUNT_KEYS).unique().height:>14,}")
    print("  (gap = số ngày lịch liên tiếp không hoạt động ngay trước t, xuyên qua ranh train/test)")


def logon_type_candidates(base: pl.DataFrame, prof: TrainProfiles, active_ok: pl.DataFrame) -> None:
    """Ứng viên kịch bản 6: LogonType mới có khuôn User thật + Source/LogHost quen + ngày hoạt động hợp lệ."""
    _h("Kịch bản 6 — đổi logon type")
    users = prof.accounts.filter(pl.col("entity_type") == "User").select(ACCOUNT_KEYS)
    lt = prof.account_logon_types.filter(pl.col("LogonType").is_not_null())
    tpl = (lt.join(users, on=ACCOUNT_KEYS, how="semi").group_by("LogonType")
           .agg(pl.len().alias("n_user_accounts"), pl.col("n_events").sum().alias("n_events")).sort("LogonType"))
    print("  khuôn có sẵn (sự kiện train của tài khoản User, gồm 4624 lẫn 4625) theo LogonType:")
    print(tpl)
    elig = (
        base.filter((pl.col("n_known_sources") > 0) & (pl.col("n_known_loghosts") > 0))
        .join(active_ok.select(ACCOUNT_KEYS).unique(), on=ACCOUNT_KEYS, how="semi")
    )
    dom = lt.group_by(ACCOUNT_KEYS).agg(pl.col("share").max().alias("dominant_share"),
                                         pl.col("LogonType").alias("types"))
    elig = elig.join(dom, on=ACCOUNT_KEYS, how="left")
    print(f"  nạn nhân cơ sở có Source+LogHost quen và ≥ 1 ngày hoạt động hợp lệ trong khối: {elig.height:,}")
    shares = (0.0, 0.5, 0.8, 0.95)
    print(f"  {'LogonType mới':<14} " + " ".join(f"{'dom≥' + format(s, '.2f'):>10}" for s in shares))
    for t in tpl["LogonType"].to_list():
        lacking = elig.filter(~pl.col("types").list.contains(int(t)))
        print(f"  {int(t):<14} " + " ".join(f"{lacking.filter(pl.col('dominant_share') >= s).height:>10,}"
                                             for s in shares))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/injection.yaml")
    ap.add_argument("--system-config", default="configs/system_config.yaml")
    ap.add_argument("--block", default="dev", choices=["dev", "test"])
    ap.add_argument("--survey-day", type=int, default=16, help="ngày TRAIN để khảo sát trường (0 = bỏ qua)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sys_cfg = yaml.safe_load(Path(args.system_config).read_text(encoding="utf-8"))
    common = cfg["common"]
    split_day = int(common.get("split_day") or sys_cfg["evaluation"]["split_day"])
    block = cfg["blocks"][args.block]
    block_days = [int(d) for d in block["days"]]
    interim = Path(common["interim_dir"])
    vic = common["victim"]

    prof = TrainProfiles.load(common["profiles_dir"])
    if max(int(d) for d in prof.network["train_days"]) > split_day:
        raise SystemExit("Hồ sơ chứa ngày > split_day — dựng lại hồ sơ chỉ từ train.")

    _h(f"Nạn nhân cơ sở — khối {args.block} (ngày {block_days[0]}..{block_days[-1]})")
    base = prof.accounts.filter(
        (pl.col("entity_type") == vic["entity_type"]) & (pl.col("n_active_days") >= int(vic["min_train_active_days"]))
    )
    print(f"  tài khoản {vic['entity_type']} trong hồ sơ: "
          f"{prof.accounts.filter(pl.col('entity_type') == vic['entity_type']).height:,}; "
          f"có ≥ {vic['min_train_active_days']} ngày hoạt động train: {base.height:,}")

    print("  đọc hiện diện các ngày test (chỉ cột UserName/DomainName)...")
    pres_parts = [p for d in range(split_day + 1, max(block_days) + 1) if (p := day_presence(interim, d)) is not None]
    presence = (pl.concat(pres_parts) if pres_parts
                else pl.DataFrame(schema={"DomainName": pl.String, "UserName": pl.String, "day": pl.Int32}))
    in_block = presence.filter(pl.col("day").is_in(block_days)).join(base.select(ACCOUNT_KEYS), on=ACCOUNT_KEYS,
                                                                       how="semi")
    flagged = rule_alerted(common)
    active_ok = in_block if flagged is None else in_block.join(flagged, on=ACCOUNT_KEYS + ["day"], how="anti")
    print(f"  (TK, ngày) hoạt động trong khối: {in_block.height:,}; sau khi bỏ dòng luật đã cờ: "
          f"{active_ok.height:,} ({active_ok.select(ACCOUNT_KEYS).unique().height:,} tài khoản)")

    dormant_candidates(base, prof, presence, block_days, split_day)
    logon_type_candidates(base, prof, active_ok)

    if args.survey_day:
        if args.survey_day > split_day:
            raise SystemExit("--survey-day phải là ngày train.")
        survey_fields(interim, args.survey_day, base.select(ACCOUNT_KEYS))


if __name__ == "__main__":
    main()
