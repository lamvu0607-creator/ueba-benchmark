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

import json
import logging
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
            logger.warning("Thiếu file interim ngày %d — coi như không có dữ liệu.", d)
            continue
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
        self._used_accounts: Set[Tuple[str, str]] = set()      # mỗi tài khoản tối đa một lần / run
        self._flagged = _rule_flagged(cfg.common)
        self._presence = _read_presence(self.interim, cfg.days)
        self._domain_raw = self._build_domain_raw_map()

    # ---- kho khuôn (đọc train MỘT LẦN bằng lazy scan; thu nhỏ để không giữ cả log train trong RAM)
    @property
    def pool(self) -> TemplatePool:
        if self._pool is None:
            self._pool = self._build_pool()
            logger.info("Kho khuôn: %s sự kiện (đã thu nhỏ từ train).", f"{self._pool.events.height:,}")
        return self._pool

    def _build_pool(self) -> TemplatePool:
        """
        Dựng kho khuôn bằng lazy scan **từng file** interim train, rồi THU NHỎ để chỉ giữ những khuôn
        thực sự có thể được dùng — tránh nạp cả chục triệu sự kiện train vào RAM (nguyên nhân chính gây
        chậm / tràn RAM). Ngữ nghĩa ``pick`` không đổi:

          * khuôn ``own`` (của chính nạn nhân): giữ ĐẦY ĐỦ sự kiện của các tài khoản là **nạn nhân cơ
            sở** (chỉ User, đủ ngày hoạt động) — tập này nhỏ hơn toàn log rất nhiều, và off_hours/dormant
            cần nguyên chuỗi/ngày của nạn nhân nên không được cắt;
          * khuôn ``peer`` (tài khoản khác cùng ``entity_type``): chỉ cần một ít mẫu cho mỗi
            ``(entity_type, EventID, LogonType, fail_kind)`` — giữ tối đa ``peer_cap`` mẫu/nhóm.

        Hai phần được hợp lại; nhờ vậy mọi truy vấn của ``pool.pick`` vẫn có khuôn, nhưng kích thước kho
        giảm vài bậc.
        """
        from src.features.extractor import DEFAULT_FEATURE_CFG, entity_type_expr

        et = entity_type_expr(self._feature_cfg or DEFAULT_FEATURE_CFG)
        peer_cap = int((self.cfg.common.get("template", {}) or {}).get("peer_cap_per_group", 200))

        files = [interim_day_path(self.interim, e, int(d))
                 for d in self.profiles.network["train_days"] for e in EVENT_IDS]
        files = [p for p in files if p.is_file()]
        if not files:
            raise FileNotFoundError("Không có file interim train để dựng kho khuôn.")

        base_keys = self._base_victims().select(
            pl.col("DomainName").alias("_dom"), pl.col("UserName")
        ).lazy()  # khoá đã chuẩn hoá (accounts của hồ sơ dùng _dom)
        grp = ["_entity_type", "EventID", "LogonType", "_fail_kind"]

        # TỪNG FILE một: RAM chỉ giữ một ngày. (Một window ``over`` trên scan cả 42 ngày buộc polars nạp toàn
        # bộ log train.) peer = ``peer_cap`` dòng đầu mỗi nhóm của từng file; ``head`` lần hai trên phần gộp
        # (file theo thứ tự ngày) cho đúng ``peer_cap`` dòng đầu của cả train.
        own_parts: List[pl.DataFrame] = []
        peer_parts: List[pl.DataFrame] = []
        for path in files:
            lf = annotate_events_lazy(pl.scan_parquet(path), et)
            own_parts.append(lf.join(base_keys, on=["_dom", "UserName"], how="semi").collect())
            peer_parts.append(lf.group_by(grp, maintain_order=True).head(peer_cap).select(lf.collect_schema().names()).collect())
        peer = pl.concat(peer_parts, how="vertical").group_by(grp, maintain_order=True).head(peer_cap)
        events = pl.concat([pl.concat(own_parts, how="vertical"), peer.select(own_parts[0].columns)], how="vertical")
        # sắp theo mọi cột: thứ tự kho tất định -> ``pick`` (rút theo chỉ số) cho cùng khuôn với cùng seed
        events = events.unique(maintain_order=True).sort(INTERIM_COLUMNS, nulls_last=True, maintain_order=True)
        return TemplatePool(events, train_end_day=self.cfg.split_day)

    def _build_domain_raw_map(self) -> Dict[Tuple[str, str], str]:
        """Khoá chuẩn hoá -> một DomainName THÔ mẫu, để sự kiện tiêm ghi domain thô đúng của tài khoản."""
        ev = self.pool.events
        raw = ev.group_by(["_dom", "UserName"]).agg(pl.col("DomainName").first().alias("raw"))
        return {(r["_dom"], r["UserName"]): r["raw"] for r in raw.iter_rows(named=True)}

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

    def _run_spraying(self, params: Dict[str, Any]) -> List[Tuple[Account, int, ScenarioResult]]:
        """Nhiều chiến dịch: mỗi chiến dịch một Source mới, một campaign_id, nhiều nạn nhân cùng ngày."""
        from src.injection.scenarios import pick_unseen_sources

        n_camp = int(params.get("n_campaigns", 3))
        out: List[Tuple[Account, int, ScenarioResult]] = []
        for c in range(n_camp):
            size = _draw(params["victims_per_campaign"], self.rng)
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
            cid = f"{self.cfg.run_id}_spray{c:02d}"
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

    # ---- chạy toàn khối
    def build(self) -> Dict[str, Any]:
        _ = self.pool                                       # ép đọc train trước khi chọn nạn nhân
        results: List[Tuple[str, Account, int, ScenarioResult]] = []
        for name in _SCENARIO_ORDER:
            if name not in self.cfg.scenarios:
                continue
            params = self.cfg.scenarios[name]
            if name == "password_spraying":
                triples = self._run_spraying(params)
            else:
                chooser = {
                    "dormant_wakeup": self._candidates_dormant,
                    "logon_type_switch": self._candidates_logon_switch,
                    "off_hours": self._candidates_off_hours,
                }.get(name, self._candidates_on_active_day)
                n = int(params.get("n_victims", 0))
                triples = self._run_simple(name, chooser, params, n)
            for acct, day, res in triples:
                results.append((name, acct, day, res))
            logger.info("[%s] tiêm %d nạn nhân.", name, len(triples))
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
        manifest_df = pl.DataFrame(manifest)

        base_days = self._write_overlay(layout, injected)
        injected.write_parquet(layout.injected_events_path)
        manifest_df.write_csv(layout.manifest_path)
        self._check_run(layout, injected, manifest_df)
        labels_path = write_run_labels(layout)

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
                pl.read_parquet(layout.events_file(e, day)).height
                - pl.read_parquet(interim_day_path(self.interim, e, day)).height
                for e in EVENT_IDS
            )
            want = inj_day.filter(pl.col("day") == day).height
            if added != want:
                raise ValueError(f"Ngày {day}: log đè thêm {added} dòng nhưng bảng phụ ghi {want}.")
        # schema file đè khớp interim
        for day in ov:
            for e in EVENT_IDS:
                got = pl.read_parquet(layout.events_file(e, day)).schema
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
