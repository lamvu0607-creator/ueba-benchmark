"""
Điều phối một lần tiêm (run): đọc config, chọn nạn nhân từ hồ sơ TRAIN, gọi kịch bản, ghép log đè và
ghi ba đầu ra (``events_injected/``, ``injected_events.parquet``, ``injection_manifest.csv``), kèm các
kiểm tra bắt buộc trước khi ghi.

Một lần chạy = MỘT khối (``dev`` hoặc ``test``) = một ``run_id``. Hồ sơ dùng chung cho mọi khối, không
bao giờ dựng lại từ ngày > ``split_day``.

Chọn nạn nhân (quyết định 2026-10-08)
-------------------------------------
* chỉ ``entity_type == "User"``, có ≥ ``min_train_active_days`` ngày hoạt động train;
* mỗi tài khoản tiêm TỐI ĐA MỘT LẦN trong một run (dùng cho mọi kịch bản);
* bỏ (tài khoản, ngày) mà baseline luật ECDF đã cảnh báo trên log GỐC;
* kịch bản 3/4/6 tiêm vào ngày nạn nhân VỐN có hoạt động trong khối; 5 tiêm vào ngày nạn nhân KHÔNG
  hoạt động, đứng sau một khoảng trống đủ dài; 2 tạo khoá mới (nạn nhân có thể không hoạt động ngày đó).
* nếu một ngày thiếu nạn nhân sạch thì GIẢM số nạn nhân ngày đó rồi vẫn chạy (không dừng).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import polars as pl
import yaml

from src.injection.labels import write_run_labels
from src.injection.layout import (
    EVENT_IDS,
    RunLayout,
    assert_events_within_day,
    interim_day_path,
    overlay_days,
)
from src.injection.operations import (
    INTERIM_COLUMNS,
    Account,
    ScheduleError,
    TemplateNotFoundError,
    TemplatePool,
    annotate_events,
    annotate_events_lazy,
    conform_to_interim,
)
from src.injection.profiles import ACCOUNT_KEYS, TrainProfiles
from src.injection.scenarios import (
    SCENARIOS,
    ScenarioError,
    ScenarioResult,
    VictimContext,
    scenario_password_spraying,
)

logger = logging.getLogger("ueba_benchmark.injection.inject")

__all__ = ["InjectionConfig", "InjectionRunner", "run_injection"]

_SCENARIO_ORDER = [
    "brute_force", "password_spraying", "off_hours", "new_workstation_burst",
    "dormant_wakeup", "logon_type_switch",
]
#: Kịch bản tiêm vào ngày nạn nhân VỐN hoạt động (và chưa bị luật cờ).
_ON_ACTIVE_DAY = {"brute_force", "new_workstation_burst", "logon_type_switch", "off_hours"}
#: Kịch bản tiêm vào ngày nạn nhân KHÔNG hoạt động.
_ON_IDLE_DAY = {"dormant_wakeup"}
#: Kịch bản tạo khoá (tài khoản, ngày) có thể CHƯA tồn tại (nạn nhân không cần hoạt động ngày đó).
_MAY_CREATE_KEY = {"password_spraying", "dormant_wakeup"}


@dataclass
class InjectionConfig:
    """Tham số một khối đã phân giải từ ``configs/injection.yaml``."""

    block: str
    days: List[int]
    seed: int
    run_id: str
    scenarios: Dict[str, Dict[str, Any]]
    common: Dict[str, Any]
    split_day: int

    @classmethod
    def from_files(
        cls,
        block: str,
        config_path: Path | str = "configs/injection.yaml",
        system_config_path: Path | str = "configs/system_config.yaml",
        run_id: Optional[str] = None,
    ) -> "InjectionConfig":
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        sys_cfg = yaml.safe_load(Path(system_config_path).read_text(encoding="utf-8"))
        if block not in cfg["blocks"]:
            raise KeyError(f"Khối '{block}' không có trong {config_path} (có: {sorted(cfg['blocks'])}).")
        blk = cfg["blocks"][block]
        common = cfg["common"]
        split_day = int(common.get("split_day") or sys_cfg["evaluation"]["split_day"])
        days = [int(d) for d in blk["days"]]
        if min(days) <= split_day:
            raise ValueError(f"Khối '{block}' có ngày ≤ split_day ({split_day}) — chỉ được tiêm test.")
        rid = run_id or f"{blk.get('run_id_prefix', block)}_seed{blk['seed']}"
        scen = {k: v for k, v in blk["scenarios"].items() if v.get("enabled", True)}
        return cls(block, days, int(blk["seed"]), rid, scen, common, split_day)


# --------------------------------------------------------------------------- cache khuôn own trên đĩa

_OWN_CACHE_VERSION = 1
_OWN_BUCKETS = 64


def _own_bucket(domain: str, user: str) -> int:
    """Bucket ỔN ĐỊNH (không phụ thuộc phiên bản polars) của một tài khoản trong cache khuôn own."""
    return zlib.crc32(f"{domain}\x00{user}".encode("utf-8")) % _OWN_BUCKETS


# --------------------------------------------------------------------------- phân giải loại trừ của luật

def _rule_flagged(common: Dict[str, Any]) -> pl.DataFrame:
    """Khoá (DomainName, UserName, day) luật ECDF đã cờ trên log gốc (rỗng nếu tắt hoặc thiếu file)."""
    empty = pl.DataFrame(schema={"DomainName": pl.String, "UserName": pl.String, "day": pl.Int64})
    rc = common.get("rule_exclusion", {}) or {}
    if not rc.get("enabled", True):
        return empty
    scores_p, thr_p = Path(rc["baseline_scores"]), Path(rc["thresholds"])
    if not (scores_p.is_file() and thr_p.is_file()):
        logger.warning("Thiếu '%s'/'%s' — KHÔNG loại tài khoản-ngày luật đã cờ.", scores_p, thr_p)
        return empty
    meta_p = Path(rc.get("event_stats_meta", ""))
    if meta_p.is_file() and json.loads(meta_p.read_text(encoding="utf-8")).get("injected_events_dir") is not None:
        raise ValueError(f"'{meta_p}': baseline chạy trên log đã tiêm — cần baseline của log gốc.")
    seg, method = rc.get("segment", "User"), rc.get("method", "rule_ecdf")
    thr = float(json.loads(thr_p.read_text(encoding="utf-8"))["thresholds"][seg][method]["threshold"])
    return (
        pl.scan_parquet(scores_p)
        .filter((pl.col("segment") == seg) & (pl.col("split") == "eval") & (pl.col(method) > thr))
        .select(ACCOUNT_KEYS + [pl.col("day").cast(pl.Int64)]).collect()
    )


# --------------------------------------------------------------------------- hiện diện test

def _read_presence(interim: Path, days: Sequence[int]) -> pl.DataFrame:
    """(DomainName chuẩn hoá, UserName, day) có ≥ 1 sự kiện trong các ngày test (đọc nhẹ 2 cột)."""
    def norm() -> pl.Expr:
        c = pl.col("DomainName")
        return (pl.when(c.is_null() | (c.str.strip_chars() == "")).then(pl.lit("Unknown"))
                .otherwise(c.str.strip_chars().str.to_lowercase()).alias("DomainName"))

    parts = []
    for d in days:
        files = [interim_day_path(interim, e, d) for e in EVENT_IDS]
        if not all(p.is_file() for p in files):
            raise FileNotFoundError(f"Missing interim day {d}: cannot infer inactivity from missing logs.")
        lf = pl.concat([pl.scan_parquet(p).select(["UserName", "DomainName"]) for p in files])
        parts.append(
            lf.filter(pl.col("UserName").is_not_null() & (pl.col("UserName").str.strip_chars() != "")
                      & pl.col("UserName").str.to_lowercase().is_in(["null", "none", "nan"]).not_())
            .select(norm(), "UserName").unique().with_columns(pl.lit(int(d), pl.Int64).alias("day")).collect()
        )
    if not parts:
        return pl.DataFrame(schema={"DomainName": pl.String, "UserName": pl.String, "day": pl.Int64})
    return pl.concat(parts)


# --------------------------------------------------------------------------- runner

class InjectionRunner:
    """Chạy một khối: dựng kho khuôn, chọn nạn nhân, gọi kịch bản, ghi đầu ra, kiểm tra."""

    def __init__(self, cfg: InjectionConfig, feature_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.profiles = TrainProfiles.load(cfg.common["profiles_dir"])
        if max(int(d) for d in self.profiles.network["train_days"]) > cfg.split_day:
            raise ValueError("Hồ sơ chứa ngày > split_day — dựng lại hồ sơ chỉ từ train.")
        self.interim = Path(cfg.common["interim_dir"])
        self._feature_cfg = feature_cfg
        self._pool: Optional[TemplatePool] = None
        self._own_dir: Optional[Path] = None
        self._domain_raw_cache: Dict[Tuple[str, str], str] = {}
        self._used_accounts: Set[Tuple[str, str]] = set()      # mỗi tài khoản tối đa một lần / run
        self._n_campaigns = 0                                   # đánh số campaign_id liên tục qua các vòng
        self._target: Dict[str, Any] = {}                       # thông tin tỉ lệ mục tiêu (ghi run_config.json)
        self._flagged = _rule_flagged(cfg.common)
        # Include earlier evaluation days: a test block must not mistake dev activity for dormancy.
        self._presence = _read_presence(self.interim, range(cfg.split_day + 1, max(cfg.days) + 1))
        self._domain_raw = self._build_domain_raw_map()

    # ---- kho khuôn (đọc train MỘT LẦN bằng lazy scan; own để trên đĩa, chỉ peer nằm trong RAM)
    @property
    def pool(self) -> TemplatePool:
        if self._pool is None:
            self._pool = self._build_pool()
            logger.info("Kho khuôn peer: %s sự kiện trong RAM; khuôn own đọc theo tài khoản từ '%s'.",
                        f"{self._pool.events.height:,}", self._own_dir)
        return self._pool

    def _build_pool(self) -> TemplatePool:
        """
        Dựng kho khuôn mà KHÔNG nạp sự kiện own của mọi nạn nhân cơ sở vào RAM (trên log thật đó là
        ~220 triệu dòng ≈ 20 GB — nguyên nhân tràn RAM làm treo/crash máy). Ngữ nghĩa ``pick`` giữ nguyên:

          * khuôn ``own`` (của chính nạn nhân): ghi MỘT LẦN ra cache trên đĩa
            (``<runs_dir>/_template_cache/<khoá>/b<bucket>/``), chia theo bucket ``crc32(_dom, UserName)``;
            ``pool.candidates(account=...)`` chỉ đọc bucket của tài khoản đó (có cache LRU). Cache dùng lại
            giữa các lần chạy dev/test khi khoá (file train, nạn nhân cơ sở, split_day...) không đổi;
          * khuôn ``peer`` (tài khoản khác cùng ``entity_type``): chỉ giữ tối đa ``peer_cap`` mẫu cho mỗi
            ``(entity_type, EventID, LogonType, fail_kind)`` — nằm trong RAM.
        """
        from src.features.extractor import DEFAULT_FEATURE_CFG, entity_type_expr

        et = entity_type_expr(self._feature_cfg or DEFAULT_FEATURE_CFG)
        peer_cap = int((self.cfg.common.get("template", {}) or {}).get("peer_cap_per_group", 200))

        files = [(int(d), interim_day_path(self.interim, e, int(d)))
                 for d in self.profiles.network["train_days"] for e in EVENT_IDS]
        files = [(d, p) for d, p in files if p.is_file()]
        if not files:
            raise FileNotFoundError("Không có file interim train để dựng kho khuôn.")

        base_keys = self._base_victims().select(pl.col("DomainName").alias("_dom"), pl.col("UserName"))
        base_keys = base_keys.with_columns(
            pl.Series("_bucket", [_own_bucket(d, u) for d, u in base_keys.iter_rows()], dtype=pl.Int32)
        )
        key_src = {
            "version": _OWN_CACHE_VERSION, "split_day": self.cfg.split_day, "entity_type": str(et),
            "peer_cap": peer_cap, "n_buckets": _OWN_BUCKETS,
            "files": [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for _, p in files],
            "base_keys": hashlib.sha1(
                "\n".join(f"{d}\x00{u}" for d, u in sorted(base_keys.select("_dom", "UserName").iter_rows()))
                .encode("utf-8")).hexdigest(),
        }
        cache_key = hashlib.sha1(json.dumps(key_src, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        cache = Path(self.cfg.common["runs_dir"]) / "_template_cache" / cache_key
        self._own_dir = cache
        if not (cache / "meta.json").is_file():
            self._write_template_cache(cache, files, base_keys, et, peer_cap, key_src)
        else:
            logger.info("Dùng lại cache khuôn '%s'.", cache)

        peer = pl.read_parquet(cache / "peer.parquet")
        raw = pl.read_parquet(cache / "domain_raw.parquet")
        self._domain_raw_cache = {(r["_dom"], r["UserName"]): r["raw"] for r in raw.iter_rows(named=True)}
        # sắp theo mọi cột: thứ tự kho tất định -> ``pick`` (rút theo chỉ số) cho cùng khuôn với cùng seed
        peer = peer.unique(maintain_order=True).sort(INTERIM_COLUMNS, nulls_last=True, maintain_order=True)
        columns = peer.columns

        @functools.lru_cache(maxsize=64)
        def load_own(dom: str, user: str) -> pl.DataFrame:
            bdir = cache / f"b{_own_bucket(dom, user):03d}"
            if not any(bdir.glob("*.parquet")):
                return peer.clear()
            own = (pl.scan_parquet(bdir / "*.parquet")
                   .filter((pl.col("_dom") == dom) & (pl.col("UserName") == user)).collect())
            return (own.select(columns).unique(maintain_order=True)
                    .sort(INTERIM_COLUMNS, nulls_last=True, maintain_order=True))

        return TemplatePool(peer, train_end_day=self.cfg.split_day,
                            own_loader=lambda acct: load_own(acct.domain, acct.user))

    def _write_template_cache(self, cache: Path, files: List[Tuple[int, Path]], base_keys: pl.DataFrame,
                              et: pl.Expr, peer_cap: int, key_src: Dict[str, Any]) -> None:
        """Đọc TỪNG file train một (RAM chỉ giữ một ngày), ghi own theo bucket + peer + DomainName thô."""
        tmp = cache.with_name(cache.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        grp = ["_entity_type", "EventID", "LogonType", "_fail_kind"]
        keys_lf = base_keys.lazy()
        # Phần peer / DomainName thô của từng file được GHI RA ĐĨA ngay, không giữ trong list: khung nhỏ cắt
        # từ một file (head/agg trên cột String) vẫn tham chiếu toàn bộ buffer chuỗi của file đó, giữ trong
        # RAM qua 84 file sẽ phình hàng chục GB.
        (tmp / "_peer").mkdir()
        (tmp / "_raw").mkdir()
        n_own = 0
        for i, (_, path) in enumerate(files, 1):
            lf = annotate_events_lazy(pl.scan_parquet(path), et)
            own = lf.join(keys_lf, on=["_dom", "UserName"], how="inner").collect()
            n_own += own.height
            own.group_by(["_dom", "UserName"]).agg(
                pl.col("Time").min().alias("_t"), pl.col("DomainName").sort_by("Time").first().alias("raw")
            ).write_parquet(tmp / "_raw" / f"{i:03d}.parquet")
            for (b,), part in own.partition_by("_bucket", as_dict=True).items():
                bdir = tmp / f"b{int(b):03d}"
                bdir.mkdir(exist_ok=True)
                part.drop("_bucket").write_parquet(bdir / f"{path.stem}.parquet")
            del own
            (lf.group_by(grp, maintain_order=True).head(peer_cap).select(lf.collect_schema().names())
             .collect().write_parquet(tmp / "_peer" / f"{i:03d}.parquet"))
            logger.info("Cache khuôn: %d/%d file (%s), own luỹ kế %s dòng.", i, len(files), path.name, f"{n_own:,}")
        # file đặt tên theo thứ tự đọc (ngày tăng dần) -> ``head`` lần hai = ``peer_cap`` dòng đầu của cả train
        peer = (pl.scan_parquet(tmp / "_peer" / "*.parquet").collect()
                .group_by(grp, maintain_order=True).head(peer_cap))
        peer.select(pl.read_parquet_schema(tmp / "_peer" / "001.parquet").names()).write_parquet(tmp / "peer.parquet")
        raw = (pl.scan_parquet(tmp / "_raw" / "*.parquet").collect().sort("_t", maintain_order=True)
               .group_by(["_dom", "UserName"], maintain_order=True).agg(pl.col("raw").first()))
        raw.write_parquet(tmp / "domain_raw.parquet")
        shutil.rmtree(tmp / "_peer")
        shutil.rmtree(tmp / "_raw")
        (tmp / "meta.json").write_text(json.dumps({**key_src, "n_own_events": n_own}, indent=2), encoding="utf-8")
        if cache.exists():
            shutil.rmtree(cache)
        tmp.rename(cache)

    def _build_domain_raw_map(self) -> Dict[Tuple[str, str], str]:
        """Khoá chuẩn hoá -> một DomainName THÔ mẫu, để sự kiện tiêm ghi domain thô đúng của tài khoản."""
        _ = self.pool
        return self._domain_raw_cache

    def _account(self, domain: str, user: str) -> Account:
        return Account(domain=domain, user=user, domain_raw=self._domain_raw.get((domain, user), domain),
                       entity_type="User")

    # ---- quần thể cơ sở
    def _base_victims(self) -> pl.DataFrame:
        vic = self.cfg.common["victim"]
        return self.profiles.accounts.filter(
            (pl.col("entity_type") == vic["entity_type"])
            & (pl.col("n_active_days") >= int(vic["min_train_active_days"]))
        )

    def _active_in_block(self) -> pl.DataFrame:
        """(DomainName, UserName, day) nạn nhân cơ sở hoạt động trong khối, đã bỏ dòng luật cờ."""
        base = self._base_victims().select(ACCOUNT_KEYS)
        act = self._presence.filter(pl.col("day").is_in(self.cfg.days)).join(base, on=ACCOUNT_KEYS, how="semi")
        return act.join(self._flagged, on=ACCOUNT_KEYS + ["day"], how="anti")

    # ---- chọn (tài khoản, ngày) cho từng kịch bản, loại tài khoản đã dùng
    def _take(self, pairs: pl.DataFrame, n: int) -> pl.DataFrame:
        """Rút tối đa ``n`` dòng (tài khoản, ngày), mỗi tài khoản một lần, tránh tài khoản đã dùng ở run."""
        if pairs.is_empty():
            return pairs
        chosen = self._shuffled_unused(pairs).head(n)
        for d, u in zip(chosen["DomainName"], chosen["UserName"]):
            self._used_accounts.add((d, u))
        if chosen.height < n:
            logger.warning("Chỉ chọn được %d/%d nạn nhân sạch — giảm số nạn nhân.", chosen.height, n)
        return chosen

    def _shuffled_unused(self, pairs: pl.DataFrame) -> pl.DataFrame:
        """Bỏ tài khoản đã dùng ở run, xáo trộn (theo rng của run), giữ một dòng mỗi tài khoản."""
        if pairs.is_empty():
            return pairs
        if self._used_accounts:
            used = pl.DataFrame(
                {"DomainName": [d for d, _ in self._used_accounts], "UserName": [u for _, u in self._used_accounts]},
                schema={"DomainName": pl.String, "UserName": pl.String},
            )
            pairs = pairs.join(used, on=ACCOUNT_KEYS, how="anti")
        # join/unique/group_by của polars không giữ thứ tự dòng -> sắp trước để cùng seed cho cùng lựa chọn
        pairs = pairs.sort(pairs.columns)
        pairs = pairs.sample(fraction=1.0, shuffle=True, seed=int(self.rng.integers(1 << 32)))
        return pairs.unique(subset=ACCOUNT_KEYS, keep="first", maintain_order=True)

    def _candidates_on_active_day(self, params: Dict[str, Any]) -> pl.DataFrame:
        """Nạn nhân×ngày cho kịch bản tiêm vào ngày đang hoạt động (có lọc phụ theo kịch bản nếu cần)."""
        return self._active_in_block()

    def _candidates_dormant(self, params: Dict[str, Any]) -> pl.DataFrame:
        """Nạn nhân×ngày (ngủ đông): ngày khối mà nạn nhân KHÔNG hoạt động, sau một khoảng trống đủ dài."""
        base = self._base_victims()
        base_keys = base.select(ACCOUNT_KEYS)
        train_days = [int(d) for d in self.profiles.network["train_days"]]
        calendar = sorted(set(train_days) | set(range(self.cfg.split_day + 1, max(self.cfg.days) + 1)))
        rank = {d: i for i, d in enumerate(calendar)}
        active = pl.concat([
            self.profiles.daily_counts.select(ACCOUNT_KEYS + [pl.col("day").cast(pl.Int64)]),
            self._presence.select(ACCOUNT_KEYS + ["day"]),
        ]).join(base_keys, on=ACCOUNT_KEYS, how="semi").unique()
        act: Dict[Tuple[str, str], Set[int]] = {}
        for d, u, day in active.iter_rows():
            act.setdefault((d, u), set()).add(int(day))
        max_gap = {(r["DomainName"], r["UserName"]): (r["max_internal_gap_days"] or 0)
                   for r in base.iter_rows(named=True)}
        min_gap = int(params.get("min_gap_days", 7))
        rule = params.get("gap_rule", "account_max")
        rows = []
        for key, days in act.items():
            for t in self.cfg.days:
                if t in days or t not in rank:
                    continue
                prev = [d for d in days if d < t]
                if not prev:
                    continue
                gap = rank[t] - rank[max(prev)] - 1
                ok = gap >= min_gap if rule == "fixed" else (gap > max_gap[key] and gap >= min_gap)
                if ok:
                    rows.append((key[0], key[1], t))
        return pl.DataFrame(rows, schema={"DomainName": pl.String, "UserName": pl.String, "day": pl.Int64},
                            orient="row")

    def _candidates_logon_switch(self, params: Dict[str, Any]) -> pl.DataFrame:
        """Nạn nhân×ngày đang hoạt động + có LogonType chiếm ưu thế ≥ ngưỡng và ≥ 1 Source/LogHost quen."""
        act = self._active_in_block()
        lt = self.profiles.account_logon_types.filter(pl.col("LogonType").is_not_null())
        dom = lt.group_by(ACCOUNT_KEYS).agg(pl.col("share").max().alias("dominant_share"))
        ok_dom = dom.filter(pl.col("dominant_share") >= float(params.get("min_dominant_share", 0.8))).select(ACCOUNT_KEYS)
        has_host = self.profiles.accounts.filter(
            (pl.col("n_known_sources") > 0) & (pl.col("n_known_loghosts") > 0)
        ).select(ACCOUNT_KEYS)
        return act.join(ok_dom, on=ACCOUNT_KEYS, how="semi").join(has_host, on=ACCOUNT_KEYS, how="semi")

    def _candidates_off_hours(self, params: Dict[str, Any]) -> pl.DataFrame:
        """Nạn nhân×ngày đang hoạt động + off_hours_ratio train thấp (người thường làm ban ngày)."""
        act = self._active_in_block()
        day_people = self.profiles.accounts.filter(
            pl.col("off_hours_ratio") <= float(params.get("max_train_off_hours_ratio", 0.2))
        ).select(ACCOUNT_KEYS)
        return act.join(day_people, on=ACCOUNT_KEYS, how="semi")

    # ---- gọi kịch bản cho một nạn nhân, bắt lỗi "không dựng được" để bỏ qua
    def _ctx(self, domain: str, user: str, day: int) -> VictimContext:
        tmpl = self.cfg.common.get("template", {}) or {}
        return VictimContext(
            account=self._account(domain, user), day=int(day), profiles=self.profiles, pool=self.pool,
            prefer_own=bool(tmpl.get("prefer_own_account", True)),
            allow_peer=bool(tmpl.get("allow_peer_fallback", True)),
        )

    def _run_simple(self, name: str, chooser, params: Dict[str, Any], n: int) -> List[Tuple[Account, int, ScenarioResult]]:
        """Duyệt ứng viên đã xáo trộn tới khi đủ ``n`` nạn nhân dựng được (ứng viên hỏng -> thử người kế)."""
        out: List[Tuple[Account, int, ScenarioResult]] = []
        if n <= 0:
            return out
        skipped = 0
        fn = SCENARIOS[name]
        for d, u, day in self._shuffled_unused(chooser(params)).select(ACCOUNT_KEYS + ["day"]).iter_rows():
            if len(out) >= n:
                break
            ctx = self._ctx(d, u, int(day))
            try:
                res = fn(ctx, params, self.rng)
            except (ScenarioError, TemplateNotFoundError, ScheduleError) as exc:
                skipped += 1
                logger.info("[%s] bỏ %s: %s", name, (d, u), exc)
                continue
            self._used_accounts.add((d, u))
            out.append((ctx.account, int(day), res))
        if skipped:
            logger.warning("[%s] bỏ %d ứng viên không dựng được.", name, skipped)
        if len(out) < n:
            logger.warning("Chỉ chọn được %d/%d nạn nhân sạch — giảm số nạn nhân.", len(out), n)
        return out

    def _run_spraying(
        self, params: Dict[str, Any], n_victims: Optional[int] = None
    ) -> List[Tuple[Account, int, ScenarioResult]]:
        """
        Nhiều chiến dịch: mỗi chiến dịch một Source mới, một campaign_id, nhiều nạn nhân cùng ngày.

        ``n_victims`` = None: chạy đúng ``n_campaigns`` chiến dịch. Có giá trị (chế độ ``target_rate``): mở chiến
        dịch tới khi đủ ``n_victims`` nạn nhân (chiến dịch cuối bị cắt cỡ cho vừa).
        """
        from src.injection.scenarios import pick_unseen_sources

        n_camp = int(params.get("n_campaigns", 3)) if n_victims is None else 4 * max(int(n_victims), 1)
        out: List[Tuple[Account, int, ScenarioResult]] = []
        for c in range(n_camp):
            if n_victims is not None and len(out) >= n_victims:
                break
            size = _draw(params["victims_per_campaign"], self.rng)
            if n_victims is not None:
                size = min(size, n_victims - len(out))
            day = int(self.cfg.days[int(self.rng.integers(len(self.cfg.days)))])
            # nạn nhân chiến dịch: tài khoản cơ sở chưa dùng (không cần hoạt động ngày đó — spray tạo khoá mới)
            pool_keys = self._base_victims().select(ACCOUNT_KEYS)
            pool_keys = pool_keys.join(self._flagged.filter(pl.col("day") == day).select(ACCOUNT_KEYS),
                                       on=ACCOUNT_KEYS, how="anti")
            chosen = self._take(pool_keys.with_columns(pl.lit(day).alias("day")), size)
            if chosen.is_empty():
                continue
            # một Source mới cho cả chiến dịch: máy thật mà KHÔNG nạn nhân nào trong nhóm quen dùng
            keys = list(zip(chosen["DomainName"], chosen["UserName"]))
            try:
                source = pick_unseen_sources(_CampaignCtx(self, keys), 1, self.rng,
                                             params.get("source_popularity_quantile", [0.0, 0.9]))[0]
            except ScenarioError as exc:
                logger.info("[password_spraying] bỏ chiến dịch %d: %s", c, exc)
                for d, u in zip(chosen["DomainName"], chosen["UserName"]):
                    self._used_accounts.discard((d, u))     # chiến dịch hỏng: trả lại suất cho cả nhóm
                continue
            cid = f"{self.cfg.run_id}_spray{self._n_campaigns:02d}"
            self._n_campaigns += 1
            campaign = {"id": cid, "source": source, "size": chosen.height}
            for d, u, _ in chosen.iter_rows():
                ctx = self._ctx(d, u, day)
                try:
                    res = scenario_password_spraying(ctx, params, self.rng, campaign)
                except (ScenarioError, TemplateNotFoundError, ScheduleError) as exc:
                    self._used_accounts.discard((d, u))
                    logger.info("[password_spraying] bỏ %s: %s", (d, u), exc)
                    continue
                out.append((ctx.account, day, res))
        return out

    # ---- tỉ lệ mục tiêu (mục 5.2 đề tài: nhãn dương ≈ 0,5–1% số bản ghi của tập kiểm thử)
    def _denominator(self, entity_type: str) -> int:
        """Số dòng (tài khoản, ngày) loại ``entity_type`` trong khối, trên log GỐC (= dòng ma trận đặc trưng)."""
        from src.features.extractor import DEFAULT_FEATURE_CFG, entity_type_expr

        et = entity_type_expr(self._feature_cfg or DEFAULT_FEATURE_CFG)
        pres = self._presence.filter(pl.col("day").is_in(self.cfg.days)).with_columns(et)
        return pres.filter(pl.col("entity_type") == entity_type).height

    def _quotas(self, names: List[str]) -> Dict[str, Optional[int]]:
        """
        Số nạn nhân mỗi kịch bản. ``common.target_rate`` tắt -> ``n_victims`` của từng kịch bản (spraying:
        None = chạy ``n_campaigns``). Bật -> tổng N = ⌈rate·D / (1 − rate)⌉ với D = số dòng (tài khoản, ngày)
        ``entity_type`` của khối trên log gốc, chia theo ``weights``. Chia cho (1 − rate) vì mỗi lần tiêm có
        thể tạo thêm một dòng mới (spraying/dormant) -> tỉ lệ cuối trên ma trận vẫn ≥ rate.
        """
        tr = self.cfg.common.get("target_rate") or {}
        if not tr.get("enabled", False):
            return {n: (None if n == "password_spraying" else int(self.cfg.scenarios[n].get("n_victims", 0)))
                    for n in names}
        rate = float(tr["rate"])
        if not 0.0 < rate < 1.0:
            raise ValueError(f"target_rate.rate phải trong (0, 1), nhận {rate}.")
        denom = self._denominator(str(tr.get("entity_type", "User")))
        total = int(np.ceil(rate * denom / (1.0 - rate)))
        weights = {n: float((tr.get("weights") or {}).get(n, 1.0)) for n in names}
        wsum = sum(weights.values())
        if wsum <= 0:
            raise ValueError("target_rate.weights phải có tổng > 0.")
        # chia nguyên theo phần dư lớn nhất -> tổng đúng bằng ``total``
        raw = {n: total * weights[n] / wsum for n in names}
        quotas = {n: int(np.floor(v)) for n, v in raw.items()}
        for n in sorted(names, key=lambda k: (-(raw[k] - quotas[k]), names.index(k)))[: total - sum(quotas.values())]:
            quotas[n] += 1
        self._target = {"rate": rate, "entity_type": str(tr.get("entity_type", "User")),
                        "denominator": denom, "n_target": total, "quotas": dict(quotas)}
        logger.info("Tỉ lệ mục tiêu %.2f%% × %s dòng %s -> %d lần tiêm: %s",
                    100 * rate, f"{denom:,}", self._target["entity_type"], total, quotas)
        return quotas

    def _run_scenario(self, name: str, n: Optional[int]) -> List[Tuple[Account, int, ScenarioResult]]:
        params = self.cfg.scenarios[name]
        if name == "password_spraying":
            return self._run_spraying(params, n)
        chooser = {
            "dormant_wakeup": self._candidates_dormant,
            "logon_type_switch": self._candidates_logon_switch,
            "off_hours": self._candidates_off_hours,
        }.get(name, self._candidates_on_active_day)
        return self._run_simple(name, chooser, params, int(n or 0))

    # ---- chạy toàn khối
    def build(self) -> Dict[str, Any]:
        _ = self.pool                                       # ép đọc train trước khi chọn nạn nhân
        names = [n for n in _SCENARIO_ORDER if n in self.cfg.scenarios]
        need = self._quotas(names)
        results: List[Tuple[str, Account, int, ScenarioResult]] = []
        got: Dict[str, int] = {n: 0 for n in names}
        # Vòng 1 chạy hạn mức; ở chế độ target_rate, phần THIẾU (kịch bản hết ứng viên dựng được) được chia
        # lại cho các kịch bản đã đủ hạn mức, tối đa 3 vòng.
        for _round in range(3):
            short: Dict[str, int] = {}
            for name in names:
                n = need.get(name, 0)
                if n is not None and n <= 0:
                    continue
                triples = self._run_scenario(name, n)
                for acct, day, res in triples:
                    results.append((name, acct, day, res))
                got[name] += len(triples)
                if n is not None:
                    short[name] = n - len(triples)
                logger.info("[%s] tiêm %d nạn nhân.", name, len(triples))
            deficit = sum(short.values())
            if not self._target or deficit <= 0:
                break
            capable = [n for n in names if short.get(n, 1) == 0]
            if not capable:
                break
            logger.warning("Thiếu %d lần tiêm so với mục tiêu -> chia lại cho %s.", deficit, capable)
            need = {n: 0 for n in names}
            for i, n in enumerate(capable):
                need[n] = deficit // len(capable) + (1 if i < deficit % len(capable) else 0)
        if self._target:
            self._target["achieved"] = dict(got)
            self._target["n_injected"] = sum(got.values())
            self._target["rate_achieved"] = sum(got.values()) / max(self._target["denominator"], 1)
            if sum(got.values()) < self._target["n_target"]:
                logger.warning("Chỉ tiêm được %d/%d lần (%.3f%%).", sum(got.values()), self._target["n_target"],
                               100 * self._target["rate_achieved"])
        return self._assemble_and_write(results)

    # ---- ghép log đè + ghi đầu ra + kiểm tra
    def _assemble_and_write(self, results: List[Tuple[str, Account, int, ScenarioResult]]) -> Dict[str, Any]:
        layout = RunLayout.for_run(self.cfg.common["runs_dir"], self.cfg.run_id)
        layout.ensure_dirs()
        if not results:
            raise RuntimeError("Không tiêm được sự kiện nào — kiểm tra config/nguồn dữ liệu.")

        inj_rows: List[pl.DataFrame] = []
        manifest: List[Dict[str, Any]] = []
        for i, (scenario, acct, day, res) in enumerate(results):
            inj_id = f"{self.cfg.run_id}_{i:05d}"
            assert_events_within_day(res.events, day)       # kiểm tra nửa đêm, lần hai (phòng thủ)
            inj_rows.append(res.events.with_columns(pl.lit(inj_id).alias("inj_id")))
            manifest.append({
                "inj_id": inj_id, "scenario": scenario, "DomainName": acct.domain_raw, "UserName": acct.user,
                "day": int(day), "campaign_id": res.campaign_id, "split": self.cfg.block, "seed": self.cfg.seed,
                "n_events": res.n_events, "params": json.dumps(res.params, ensure_ascii=False, default=str),
            })
        injected = pl.concat(inj_rows, how="vertical")
        manifest_df = pl.DataFrame(manifest, infer_schema_length=None)  # campaign_id null ở >100 dòng đầu

        base_days = self._write_overlay(layout, injected)
        injected.write_parquet(layout.injected_events_path)
        manifest_df.write_csv(layout.manifest_path)
        self._check_run(layout, injected, manifest_df)
        labels_path = write_run_labels(layout)
        (layout.root / "run_config.json").write_text(json.dumps({
            "run_id": self.cfg.run_id, "block": self.cfg.block, "seed": self.cfg.seed,
            "split_day": self.cfg.split_day, "eval_days": self.cfg.days,
            "target_rate": self._target or None,
        }, indent=2), encoding="utf-8")

        logger.info("Run '%s': %s sự kiện tiêm, %d lần tiêm, %d ngày đè.",
                    self.cfg.run_id, f"{injected.height:,}", manifest_df.height, len(base_days))
        return {"run_id": self.cfg.run_id, "root": str(layout.root), "n_injected": injected.height,
                "n_injections": manifest_df.height, "days": base_days, "labels": str(labels_path),
                "manifest": str(layout.manifest_path)}

    def _write_overlay(self, layout: RunLayout, injected: pl.DataFrame) -> List[int]:
        """
        Ghi log ĐÈ: với mỗi ngày có sự kiện tiêm, đọc log gốc interim của CẢ 4624 và 4625 TRỰC TIẾP từ
        file (KHÔNG qua ``load_day_events`` — hàm đó lọc UserName null và chuẩn hoá DomainName, sẽ làm log
        đè khác log gốc), nối sự kiện tiêm (bỏ cột ``inj_id``), ghi lại CẢ HAI file. Đúng hợp đồng "ngày
        nào có thì đủ cả hai file, cùng 21 cột và dtype interim".
        """
        day_col = injected.with_columns(injected_day_expr())
        days = sorted(set(int(d) for d in day_col["day"].to_list()))
        for day in days:
            for e in EVENT_IDS:
                overlay = day_col.filter((pl.col("day") == day) & (pl.col("EventID") == e)).select(INTERIM_COLUMNS)
                orig = conform_to_interim(_as_interim(pl.read_parquet(interim_day_path(self.interim, e, day))))
                merged = pl.concat([orig, overlay], how="vertical").sort("Time")
                out = layout.events_file(e, day)
                out.parent.mkdir(parents=True, exist_ok=True)
                merged.write_parquet(out)
        return days

    def _check_run(self, layout: RunLayout, injected: pl.DataFrame, manifest: pl.DataFrame) -> None:
        """Các kiểm tra bắt buộc trước khi sinh nhãn (mục 4 của quy_trinh_tiem_log.md)."""
        ov = overlay_days(layout.events_dir)
        if ov and min(ov) <= self.cfg.split_day:
            raise ValueError(f"Log đè chạm ngày train {[d for d in ov if d <= self.cfg.split_day]}.")
        # số dòng đè thêm theo ngày = số sự kiện tiêm theo ngày
        inj_day = injected.with_columns(injected_day_expr())
        for day in ov:
            added = sum(
                pl.scan_parquet(layout.events_file(e, day)).select(pl.len()).collect().item()
                - pl.scan_parquet(interim_day_path(self.interim, e, day)).select(pl.len()).collect().item()
                for e in EVENT_IDS
            )
            want = inj_day.filter(pl.col("day") == day).height
            if added != want:
                raise ValueError(f"Ngày {day}: log đè thêm {added} dòng nhưng bảng phụ ghi {want}.")
        # schema file đè khớp interim
        for day in ov:
            for e in EVENT_IDS:
                got = pl.read_parquet_schema(layout.events_file(e, day))
                if list(got.keys()) != INTERIM_COLUMNS:
                    raise ValueError(f"File đè {e}/{day} sai cột interim.")
        # mỗi tài khoản tối đa một lần (labels_from_manifest cũng chặn, nhưng báo sớm ở đây)
        dup = manifest.height - manifest.select(["DomainName", "UserName"]).unique().height
        if dup:
            raise ValueError(f"{dup} tài khoản bị tiêm nhiều lần trong một run.")


class _CampaignCtx:
    """Ngữ cảnh tối thiểu cho ``pick_unseen_sources``: máy "quen" = hợp máy quen của MỌI nạn nhân chiến dịch."""

    def __init__(self, runner: "InjectionRunner", keys: Sequence[Tuple[str, str]]) -> None:
        self.profiles = runner.profiles
        self.account = runner._account(*keys[0])
        self._known: Set[str] = set()
        for d, u in keys:
            self._known |= set(self.profiles.known_sources(d, u)) | set(self.profiles.known_loghosts(d, u))

    def known_sources(self) -> List[str]:
        return sorted(self._known)

    def known_loghosts(self) -> List[str]:
        return []


def injected_day_expr() -> pl.Expr:
    """Ngày của một sự kiện tiêm, suy từ ``Time`` theo quy ước interim (dùng khi ghép/kiểm tra)."""
    return (pl.col("Time") // 86400 + 1).cast(pl.Int64).alias("day")


def _as_interim(ev: pl.DataFrame) -> pl.DataFrame:
    """Bổ các cột interim mà ``load_day_events`` không trả (nó chỉ giữ cột cần cho đặc trưng)."""
    missing = [c for c in INTERIM_COLUMNS if c not in ev.columns]
    if missing:
        ev = ev.with_columns([pl.lit(None, pl.String).alias(c) for c in missing
                              if c not in ("Time", "EventID", "LogonType")])
        for c in ("LogonType",):
            if c in missing:
                ev = ev.with_columns(pl.lit(None, pl.Int32).alias(c))
    return ev


def _draw(spec: Any, rng: np.random.Generator) -> int:
    from src.injection.operations import draw_int

    return draw_int(spec, rng)


def run_injection(
    block: str,
    config_path: Path | str = "configs/injection.yaml",
    system_config_path: Path | str = "configs/system_config.yaml",
    run_id: Optional[str] = None,
    feature_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Điểm vào: dựng và ghi một run cho ``block`` ('dev' hoặc 'test')."""
    cfg = InjectionConfig.from_files(block, config_path, system_config_path, run_id)
    return InjectionRunner(cfg, feature_cfg=feature_cfg).build()
