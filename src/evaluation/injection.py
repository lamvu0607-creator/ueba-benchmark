"""
Tiêm bất thường tổng hợp ở MỨC SỰ KIỆN (Lớp 1 — mục 5.2 của docs/Tong quan de tai ueba.md).

Vì sao tiêm vào luồng sự kiện chứ không sửa thẳng ma trận đặc trưng: nếu sửa ma trận thì
"đặc trưng có phân tách được kịch bản không" trở thành câu hỏi về cách ta sửa, không còn là
câu hỏi về đặc trưng. Tiêm sự kiện rồi chạy *đúng* đường tính của pipeline thì mọi đặc trưng
(cũ lẫn mới) được thử trên cùng một bằng chứng.

Bảy kịch bản (6 của mục 5.2 + di chuyển ngang mà thực tập sinh ưu tiên):

=========================  ==================================================================
``brute_force``            100–300 lần 4625 trong 20 phút từ MỘT nguồn mới → máy đích quen
``password_spraying``      MỘT nguồn mới, mỗi nạn nhân 3–8 lần 4625 trong cùng 30 phút
``off_hours``              nén toàn bộ hoạt động của ngày vào 01:00–06:00
``new_workstation_burst``  10–30 lần 4624 thành công, mỗi lần từ một máy nguồn chưa từng thấy
``dormant_wakeup``         xoá hoạt động ngày 4…t−1 (ngủ đông), ngày t khối lượng ×5
``logon_type_switch``      mọi sự kiện ngày t đổi sang LogonType 5 (chưa từng dùng)
``lateral_fanout``         15–40 lần 4624 type 3 tới các máy đích chưa từng chạm, trong 1 giờ
=========================  ==================================================================

*Cảnh báo phương pháp (giữ nguyên từ đề cương):* bất thường tổng hợp dễ phát hiện hơn tấn công
thật ⇒ độ phân tách đo được là CẬN TRÊN lạc quan. Ở đây nó chỉ dùng làm cổng lọc đặc trưng
("đặc trưng có phản ứng đúng chiều với đúng kịch bản nó nhắm tới không").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import polars as pl

__all__ = ["InjectionPlan", "select_victims", "inject_day", "INJECTION_SCENARIOS"]

INJECTION_SCENARIOS: List[str] = [
    "brute_force", "password_spraying", "off_hours", "new_workstation_burst",
    "dormant_wakeup", "logon_type_switch", "lateral_fanout",
]
_KEYS = ["DomainName", "UserName"]
_DAY = 86400


@dataclass
class InjectionPlan:
    """Kế hoạch tiêm: ngày tiêm, cửa sổ ngủ đông, danh sách nạn nhân theo kịch bản."""
    target_day: int
    dormant_from: int
    victims: Dict[str, pl.DataFrame] = field(default_factory=dict)   # scenario -> (DomainName, UserName)
    seed: int = 42

    def labels(self) -> pl.DataFrame:
        frames = [v.select(_KEYS).with_columns(pl.lit(s).alias("scenario")) for s, v in self.victims.items()]
        return pl.concat(frames).with_columns(pl.lit(self.target_day).cast(pl.Int32).alias("day"))

    def dormant_accounts(self) -> pl.DataFrame:
        return self.victims.get("dormant_wakeup", pl.DataFrame(schema={k: pl.String for k in _KEYS}))


def select_victims(
    matrix: pl.DataFrame,
    target_day: int,
    n_per_scenario: int = 50,
    dormant_from: int = 4,
    seed: int = 42,
) -> InjectionPlan:
    """
    Chọn nạn nhân TẤT ĐỊNH (theo ``seed``) từ ma trận thô v3 (cần ``total_logons``,
    ``failure_ratio``, ``off_hours_ratio``, ``rare_logon_type_count``, ``entity_type``).

    Chỉ chọn tài khoản người (``entity_type == "User"``) có hồ sơ lịch sử (≥ 4/7 ngày hoạt động
    trước ngày tiêm) để "khác hẳn hồ sơ hành vi lịch sử" có nghĩa; các nhóm không giao nhau.
    """
    t = target_day
    users = matrix.filter(pl.col("entity_type") == "User")
    hist = (
        users.filter(pl.col("day").is_between(t - 7, t - 1))
        .group_by(_KEYS).agg([
            pl.len().alias("_active_7d"),
            pl.col("rare_logon_type_count").sum().alias("_rare_7d"),
        ])
    )
    today = users.filter(pl.col("day") == t).join(hist, on=_KEYS, how="left").with_columns(
        pl.col("_active_7d").fill_null(0), pl.col("_rare_7d").fill_null(0)
    )
    regular = today.filter(pl.col("_active_7d") >= 4)

    early = (
        users.filter(pl.col("day") < dormant_from)
        .group_by(_KEYS).agg(pl.len().alias("_early"))
        .filter(pl.col("_early") == dormant_from - 1)
    )
    pools: Dict[str, pl.DataFrame] = {
        # ngủ đông: hoạt động đủ các ngày trước dormant_from và có mặt ở ngày t
        "dormant_wakeup": today.join(early, on=_KEYS, how="semi").filter(pl.col("total_logons") >= 10),
        "brute_force": regular.filter(pl.col("total_logons").is_between(5, 50) & (pl.col("failure_ratio") == 0)),
        "password_spraying": regular.filter(pl.col("total_logons").is_between(5, 500) & (pl.col("failure_ratio") == 0)),
        "off_hours": regular.filter((pl.col("total_logons") >= 20) & (pl.col("off_hours_ratio") <= 0.2)),
        "logon_type_switch": regular.filter(
            (pl.col("total_logons") >= 10) & (pl.col("rare_logon_type_count") == 0) & (pl.col("_rare_7d") == 0)
        ),
        "new_workstation_burst": regular.filter(pl.col("total_logons") >= 5),
        "lateral_fanout": regular.filter(pl.col("total_logons") >= 5),
    }

    rng = np.random.default_rng(seed)
    taken = pl.DataFrame(schema={k: pl.String for k in _KEYS})
    plan = InjectionPlan(target_day=t, dormant_from=dormant_from, seed=seed)
    for scen in INJECTION_SCENARIOS:          # thứ tự cố định ⇒ tất định
        pool = pools[scen].select(_KEYS).join(taken, on=_KEYS, how="anti").sort(_KEYS)
        k = min(n_per_scenario, pool.height)
        idx = np.sort(rng.choice(pool.height, size=k, replace=False)) if k else np.array([], dtype=int)
        chosen = pool[idx.tolist()] if k else pool.head(0)
        taken = pl.concat([taken, chosen])
        plan.victims[scen] = chosen
    return plan


def _event_rows(n: int, **cols) -> pl.DataFrame:
    return pl.DataFrame({k: (v if isinstance(v, list) else [v] * n) for k, v in cols.items()})


def _typical(ev: pl.DataFrame, col: str) -> Dict[tuple, Optional[str]]:
    """Giá trị phổ biến nhất của ``col`` cho từng tài khoản (dùng làm máy đích/nguồn 'quen')."""
    m = (
        ev.filter(pl.col(col).is_not_null() & (pl.col(col) != "Unknown"))
        .group_by(_KEYS + [col]).len()
        .sort(["len", col], descending=[True, False])
        .group_by(_KEYS, maintain_order=True).agg(pl.col(col).first())
    )
    return {(r[0], r[1]): r[2] for r in m.iter_rows()}


def _flag(events: pl.DataFrame, accounts: pl.DataFrame) -> pl.Series:
    """Cờ boolean: sự kiện thuộc một trong các tài khoản ``accounts``."""
    marked = accounts.select(_KEYS).unique().with_columns(pl.lit(True).alias("_v"))
    return events.select(_KEYS).join(marked, on=_KEYS, how="left", maintain_order="left")["_v"].fill_null(False)


def inject_day(events: pl.DataFrame, day: int, plan: InjectionPlan) -> pl.DataFrame:
    """
    Trả về luồng sự kiện ĐÃ TIÊM của ngày ``day`` (schema giữ nguyên).

    * ``dormant_from ≤ day < target_day``: chỉ xoá sự kiện của nạn nhân ngủ đông.
    * ``day == target_day``: áp cả 7 kịch bản.
    * ngày khác: trả nguyên.
    """
    t = plan.target_day
    if plan.dormant_from <= day < t:
        return events.join(plan.dormant_accounts().select(_KEYS), on=_KEYS, how="anti")
    if day != t:
        return events

    rng = np.random.default_rng(plan.seed + day)
    base = (day - 1) * _DAY
    schema = events.schema
    top_host = _typical(events, "LogHost")
    top_src = _typical(events, "Source")
    all_hosts = (
        events.filter(pl.col("LogHost").str.starts_with("Comp")).select("LogHost").unique().sort("LogHost")
        ["LogHost"].to_list()
    )
    used_hosts = events.group_by(_KEYS).agg(pl.col("LogHost").unique())
    used = {(r[0], r[1]): set(r[2]) for r in used_hosts.iter_rows()}
    new_rows: List[pl.DataFrame] = []
    out = events

    def victims(s: str):
        return list(plan.victims.get(s, pl.DataFrame(schema={k: pl.String for k in _KEYS})).iter_rows())

    def fail_rows(dom, user, times, src, host):
        n = len(times)
        return _event_rows(
            n, Time=[int(x) for x in times], EventID=4625, UserName=user, LogHost=host, LogonType=3,
            AuthenticationPackage="NTLM", Source=src, DomainName=dom, ProcessName=None,
            FailureReason="Unknown user name or bad password.",
        )

    def ok_rows(dom, user, times, srcs, hosts):
        n = len(times)
        return _event_rows(
            n, Time=[int(x) for x in times], EventID=4624, UserName=user, LogHost=hosts, LogonType=3,
            AuthenticationPackage="Kerberos", Source=srcs, DomainName=dom, ProcessName=None,
            FailureReason=None,
        )

    # 1. Brute force: một nguồn mới, 100–300 lần thất bại trong 20 phút
    for i, (dom, user) in enumerate(victims("brute_force")):
        n = int(rng.integers(100, 301))
        start = base + int(rng.integers(8 * 3600, 17 * 3600))
        times = np.sort(start + rng.integers(0, 20 * 60, size=n))
        new_rows.append(fail_rows(dom, user, times, f"INJ-BF-{i:03d}", top_host.get((dom, user), "ActiveDirectory")))

    # 2. Password spraying: MỘT nguồn, cùng khung 10:00–10:30, 3–8 lần mỗi nạn nhân
    for dom, user in victims("password_spraying"):
        n = int(rng.integers(3, 9))
        times = np.sort(base + 10 * 3600 + rng.integers(0, 30 * 60, size=n))
        new_rows.append(fail_rows(dom, user, times, "INJ-SPRAY-SRC", top_host.get((dom, user), "ActiveDirectory")))

    # 3. Off-hours: nén cả ngày vào 01:00–06:00 (giữ thứ tự & khoảng cách tương đối)
    oh = plan.victims.get("off_hours")
    if oh is not None and oh.height:
        is_v = _flag(out, oh)
        out = out.with_columns(
            pl.when(pl.lit(is_v))
            .then((base + 3600 + ((pl.col("Time") - base) * 5) // 24).cast(schema["Time"]))
            .otherwise(pl.col("Time")).alias("Time")
        )

    # 4. New workstation burst: 10–30 lần thành công, mỗi lần từ một máy nguồn chưa từng thấy
    for i, (dom, user) in enumerate(victims("new_workstation_burst")):
        n = int(rng.integers(10, 31))
        times = np.sort(base + rng.integers(9 * 3600, 17 * 3600, size=n))
        new_rows.append(ok_rows(dom, user, times, [f"INJ-WS-{i:03d}-{j:02d}" for j in range(n)],
                                top_host.get((dom, user), "ActiveDirectory")))

    # 5. Lateral fan-out: 15–40 máy đích thật mà tài khoản chưa chạm hôm nay, trong 14:00–15:00
    for dom, user in victims("lateral_fanout"):
        n = int(rng.integers(15, 41))
        pool = [h for h in all_hosts if h not in used.get((dom, user), set())]
        hosts = [pool[j] for j in rng.choice(len(pool), size=n, replace=False)]
        times = np.sort(base + 14 * 3600 + rng.integers(0, 3600, size=n))
        src = top_src.get((dom, user)) or top_host.get((dom, user), "ActiveDirectory")
        new_rows.append(ok_rows(dom, user, times, src, hosts))

    # 6. Dormant wake-up: khối lượng ngày t ×5 (nhân bản sự kiện, lệch 0–59 giây)
    dw = plan.dormant_accounts()
    if dw.height:
        mine = out.join(dw.select(_KEYS), on=_KEYS, how="semi")
        for r in range(1, 5):
            jitter = pl.Series(rng.integers(0, 60, size=mine.height))
            new_rows.append(mine.with_columns(
                (pl.col("Time") + jitter).clip(base, base + _DAY - 1).cast(schema["Time"]).alias("Time")
            ))

    # 7. Logon-type switch: mọi sự kiện ngày t chuyển sang type 5
    ls = plan.victims.get("logon_type_switch")
    if ls is not None and ls.height:
        is_v = _flag(out, ls)
        out = out.with_columns(
            pl.when(pl.lit(is_v)).then(pl.lit(5)).otherwise(pl.col("LogonType")).cast(schema["LogonType"]).alias("LogonType")
        )

    if new_rows:
        extra = [
            f.select([pl.col(c).cast(schema[c]) if c in f.columns else pl.lit(None, schema[c]).alias(c)
                      for c in schema])
            for f in new_rows
        ]
        out = pl.concat([out] + extra, how="vertical")
    return out
