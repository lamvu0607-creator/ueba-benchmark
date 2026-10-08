"""
Thao tác cơ bản của bộ tiêm: kho khuôn, nhân bản khuôn, sinh thời gian theo "profile thời gian", và các
thao tác chỉ sửa đúng trường dự kiến. Module này THUẦN (không đọc/ghi file) để kiểm thử bằng dữ liệu
tổng hợp; việc đọc log train để dựng kho khuôn nằm ở ``inject.py``.

Nguyên tắc "NHÂN BẢN SỰ KIỆN THẬT RỒI SỬA"
------------------------------------------
Mọi sự kiện tiêm là bản sao của MỘT sự kiện thật trong train (cùng EventID, LogonType, loại tài khoản,
và lý do thất bại nếu là 4625). Mọi trường không bị kịch bản đổi — ``AuthenticationPackage``,
``LogonTypeDescription``, ``FailureReason``, ``LogonID``, ``Status``, ``Process*``, ``ServiceName``,
``Destination``… — đi nguyên từ khuôn, nên không có giá trị nào bị bịa. Mỗi thao tác khai báo tập
trường nó được phép đổi trong :data:`OP_FIELDS`; test kiểm tra nó không đụng trường nào khác.

Lưu ý ``LogonID``: giữ nguyên giá trị (và do đó định dạng) của khuôn. Kho khuôn rút KHÔNG lặp khi đủ
khuôn nên phần lớn sự kiện tiêm có ``LogonID`` khác nhau; không pipeline nào đang dùng trường này.

Profile thời gian (mở cho tiêm nhiều ngày)
------------------------------------------
:class:`TimeProfile` mô tả CÁCH rải sự kiện, tách khỏi kịch bản: ``mode`` (burst / spread / replay),
``n_days`` (mặc định 1), ``hour_window``, ``burst_duration_s``. :func:`schedule` trả bảng
``(day, Time)``; mọi ``Time`` luôn nằm trong cửa sổ ngày của ``layout.day_window`` (kiểm tra cứng
trước khi trả về). Bật nhiều ngày = đổi ``n_days`` trong config; kịch bản không cần viết lại.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import polars as pl

from src.injection.layout import (
    SECONDS_PER_DAY,
    InjectionLayoutError,
    assert_events_within_day,
    day_of_time,
    day_window,
)

__all__ = [
    "INTERIM_SCHEMA",
    "INTERIM_COLUMNS",
    "HELPER_PREFIX",
    "OP_FIELDS",
    "FAIL_KINDS",
    "TemplateNotFoundError",
    "ScheduleError",
    "Account",
    "TemplatePool",
    "conform_to_interim",
    "strip_helpers",
    "annotate_events",
    "annotate_events_lazy",
    "clone_templates",
    "retarget_account",
    "set_times",
    "set_source",
    "set_loghost",
    "shift_time",
    "add_failures",
    "add_successes",
    "retype_logon",
    "TimeProfile",
    "allowed_segments",
    "schedule",
    "assert_within_days",
    "changed_fields",
    "draw_int",
]

# --------------------------------------------------------------------------- schema interim

#: 21 cột interim đúng thứ tự + dtype của ``src/data/raw_to_interim.INTERIM_SCHEMA`` (test đối chiếu).
INTERIM_SCHEMA: Dict[str, pl.DataType] = {
    "Time": pl.Int64,
    "EventID": pl.Int32,
    "LogHost": pl.String,
    "LogonType": pl.Int32,
    "LogonTypeDescription": pl.String,
    "UserName": pl.String,
    "DomainName": pl.String,
    "LogonID": pl.String,
    "SubjectUserName": pl.String,
    "SubjectDomainName": pl.String,
    "SubjectLogonID": pl.String,
    "Status": pl.String,
    "Source": pl.String,
    "ServiceName": pl.String,
    "Destination": pl.String,
    "AuthenticationPackage": pl.String,
    "FailureReason": pl.String,
    "ProcessName": pl.String,
    "ProcessID": pl.String,
    "ParentProcessName": pl.String,
    "ParentProcessID": pl.String,
}
INTERIM_COLUMNS: List[str] = list(INTERIM_SCHEMA)

#: Cột phụ (nguồn gốc khuôn, khoá chuẩn hoá…) mang tiền tố này và KHÔNG BAO GIỜ được ghi vào log chính.
HELPER_PREFIX = "_"

#: Trường mỗi thao tác được phép đổi. Mọi trường interim khác phải giữ nguyên giá trị của khuôn.
OP_FIELDS: Dict[str, Set[str]] = {
    "retarget_account": {"UserName", "DomainName", "SubjectUserName", "SubjectDomainName"},
    "set_times": {"Time"},
    "shift_time": {"Time"},
    "set_source": {"Source"},
    "set_loghost": {"LogHost"},
}
OP_FIELDS["add_failures"] = OP_FIELDS["retarget_account"] | {"Time", "Source", "LogHost"}
OP_FIELDS["add_successes"] = OP_FIELDS["add_failures"]
OP_FIELDS["retype_logon"] = OP_FIELDS["add_failures"]

#: Phân loại 4625 theo ``FailureReason`` — cùng mẫu chuỗi với ``profiles.py`` và extractor.
FAIL_KINDS = ("bad_pw", "locked", "other")
_BAD_PASSWORD = "bad password"
_LOCKED_OUT = "account locked out"


class TemplateNotFoundError(LookupError):
    """Không có sự kiện thật nào trong train khớp tiêu chí khuôn (không được bịa thay thế)."""


class ScheduleError(ValueError):
    """Profile thời gian không xếp được sự kiện vào cửa sổ cho phép trong ngày."""


def conform_to_interim(df: pl.DataFrame) -> pl.DataFrame:
    """Chọn đúng 21 cột interim theo thứ tự và ép dtype; thiếu cột -> lỗi (không tự điền)."""
    missing = [c for c in INTERIM_COLUMNS if c not in df.columns]
    if missing:
        raise InjectionLayoutError(f"Thiếu cột interim {missing}.")
    return df.select([pl.col(c).cast(t, strict=True) for c, t in INTERIM_SCHEMA.items()])


def strip_helpers(df: pl.DataFrame) -> pl.DataFrame:
    """Bỏ mọi cột phụ (tiền tố ``_``) — dùng trước khi ghép vào log chính."""
    return df.select([c for c in df.columns if not c.startswith(HELPER_PREFIX)])


# --------------------------------------------------------------------------- tài khoản và kho khuôn

@dataclass(frozen=True)
class Account:
    """
    Danh tính một nạn nhân. ``domain`` là dạng CHUẨN HOÁ (khoá hồ sơ/ma trận); ``domain_raw`` là giá trị
    ``DomainName`` đúng như trong log interim — sự kiện tiêm ghi ``domain_raw`` để sau khi
    ``load_day_events`` chuẩn hoá thì khoá trùng khoá thật của tài khoản.
    """

    domain: str
    user: str
    domain_raw: str
    entity_type: str = "User"

    @property
    def key(self) -> Tuple[str, str]:
        return (self.domain, self.user)


def _normalized_domain(col: str = "DomainName") -> pl.Expr:
    c = pl.col(col)
    return (
        pl.when(c.is_null() | (c.str.strip_chars() == ""))
        .then(pl.lit("Unknown"))
        .otherwise(c.str.strip_chars().str.to_lowercase())
    )


def _fail_kind() -> pl.Expr:
    reason = pl.col("FailureReason").str.to_lowercase()
    return (
        pl.when(pl.col("EventID") != 4625).then(pl.lit(None, pl.String))
        .when(reason.str.contains(_LOCKED_OUT, literal=True).fill_null(False)).then(pl.lit("locked"))
        .when(reason.str.contains(_BAD_PASSWORD, literal=True).fill_null(False)).then(pl.lit("bad_pw"))
        .otherwise(pl.lit("other"))
    )


def _annotate_exprs(entity_type: pl.Expr) -> List[pl.Expr]:
    return [_normalized_domain().alias("_dom"), entity_type.alias("_entity_type"), _fail_kind().alias("_fail_kind")]


def _valid_user_filter() -> pl.Expr:
    return (
        pl.col("UserName").is_not_null()
        & (pl.col("UserName").str.strip_chars() != "")
        & pl.col("UserName").str.to_lowercase().is_in(["null", "none", "nan"]).not_()
    )


def annotate_events_lazy(events: pl.LazyFrame, entity_type: pl.Expr) -> pl.LazyFrame:
    """
    Bản LAZY của :func:`annotate_events` cho ``pl.scan_parquet`` — dùng khi dựng kho khuôn từ nhiều file
    train để polars tối ưu/song song hoá và KHÔNG nạp cả log vào RAM. Chọn đúng 21 cột + ép dtype, bỏ
    dòng thiếu UserName, rồi thêm ``_dom``/``_entity_type``/``_fail_kind``.
    """
    typed = events.select([pl.col(c).cast(t, strict=False) for c, t in INTERIM_SCHEMA.items()])
    return typed.filter(_valid_user_filter()).with_columns(_annotate_exprs(entity_type))


def annotate_events(events: pl.DataFrame, entity_type: pl.Expr) -> pl.DataFrame:
    """
    Thêm cột phụ cho log interim THÔ (21 cột): ``_dom`` (DomainName chuẩn hoá như extractor),
    ``_entity_type`` (``entity_type`` là ``src.features.extractor.entity_type_expr(fcfg)``) và
    ``_fail_kind`` (bad_pw / locked / other cho 4625, null cho 4624). Dòng không có UserName bị bỏ
    giống ``load_day_events``.
    """
    ev = conform_to_interim(events).filter(_valid_user_filter())
    return ev.with_columns(_annotate_exprs(entity_type))


@dataclass
class TemplatePool:
    """
    Kho sự kiện thật của TRAIN dùng làm khuôn (đầu vào = :func:`annotate_events` của các ngày train).

    Khởi tạo kiểm tra cứng: mọi ``Time`` phải < ``train_end_day · 86400`` — khuôn không bao giờ lấy từ test.

    ``own_loader`` (tuỳ chọn): hàm trả sự kiện train của MỘT tài khoản (đã ``annotate_events``). Khi có,
    truy vấn kèm ``account`` đọc từ hàm này thay vì ``events`` — để không phải giữ sự kiện own của mọi
    nạn nhân cơ sở trong RAM (hàng trăm triệu dòng); ``events`` khi đó chỉ là kho peer đã thu nhỏ.
    """

    events: pl.DataFrame
    train_end_day: int
    own_loader: Optional[Callable[[Account], pl.DataFrame]] = None

    def __post_init__(self) -> None:
        self._check(self.events)

    def _check(self, events: pl.DataFrame) -> pl.DataFrame:
        need = INTERIM_COLUMNS + ["_dom", "_entity_type", "_fail_kind"]
        missing = [c for c in need if c not in events.columns]
        if missing:
            raise InjectionLayoutError(f"Kho khuôn thiếu cột {missing} — dựng bằng annotate_events().")
        _, end = day_window(self.train_end_day)
        leak = events.filter(pl.col("Time").is_null() | (pl.col("Time") >= end))
        if leak.height:
            raise InjectionLayoutError(
                f"Kho khuôn có {leak.height:,} sự kiện ngoài train (Time ≥ {end}) — khuôn chỉ được lấy từ train."
            )
        return events

    def candidates(
        self,
        *,
        event_id: int,
        logon_type: Optional[int] = None,
        entity_type: Optional[str] = None,
        fail_kind: Optional[str] = None,
        account: Optional[Account] = None,
        extra: Optional[pl.Expr] = None,
    ) -> pl.DataFrame:
        """Các dòng khớp MỌI tiêu chí đã cho (``None`` = không lọc theo tiêu chí đó)."""
        if fail_kind is not None and fail_kind not in FAIL_KINDS:
            raise ValueError(f"fail_kind phải thuộc {FAIL_KINDS}, nhận {fail_kind!r}.")
        cond = pl.col("EventID") == int(event_id)
        if logon_type is not None:
            cond = cond & (pl.col("LogonType") == int(logon_type))
        if entity_type is not None:
            cond = cond & (pl.col("_entity_type") == entity_type)
        if fail_kind is not None:
            cond = cond & (pl.col("_fail_kind") == fail_kind)
        source = self.events
        if account is not None:
            cond = cond & (pl.col("_dom") == account.domain) & (pl.col("UserName") == account.user)
            if self.own_loader is not None:
                source = self._check(self.own_loader(account))
        if extra is not None:
            cond = cond & extra
        return source.filter(cond)

    def pick(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        account: Account,
        event_id: int,
        logon_type: Optional[int] = None,
        fail_kind: Optional[str] = None,
        prefer_own: bool = True,
        allow_peer: bool = True,
        extra: Optional[pl.Expr] = None,
    ) -> pl.DataFrame:
        """
        Rút ``n`` khuôn rồi nhân bản (:func:`clone_templates`). Thứ tự ưu tiên: sự kiện của CHÍNH tài khoản
        -> sự kiện của tài khoản CÙNG ``entity_type``. Không bao giờ lấy khuôn khác loại tài khoản.
        Cột ``_tpl_origin`` ghi ``own``/``peer``.
        """
        crit = dict(event_id=event_id, logon_type=logon_type, fail_kind=fail_kind, extra=extra)
        tiers: List[Tuple[str, pl.DataFrame]] = []
        if prefer_own:
            tiers.append(("own", self.candidates(account=account, entity_type=account.entity_type, **crit)))
        if allow_peer or not prefer_own:
            tiers.append(("peer", self.candidates(entity_type=account.entity_type, **crit)))
        for origin, cand in tiers:
            if cand.height:
                return clone_templates(cand, n, rng).with_columns(pl.lit(origin).alias("_tpl_origin"))
        raise TemplateNotFoundError(
            f"Không có khuôn train: EventID={event_id}, LogonType={logon_type}, fail_kind={fail_kind}, "
            f"entity_type={account.entity_type} (tài khoản {account.key})."
        )


# --------------------------------------------------------------------------- nhân bản khuôn

def clone_templates(
    templates: pl.DataFrame, n: int, rng: np.random.Generator, replace: Optional[bool] = None
) -> pl.DataFrame:
    """
    Nhân bản ``n`` dòng từ ``templates`` (giữ NGUYÊN mọi trường và dtype).

    Rút không lặp khi ``n ≤ số khuôn`` (đa dạng LogonID/ProcessID), ngược lại rút có lặp. Thêm cột phụ
    ``_tpl_UserName``, ``_tpl_DomainName``, ``_tpl_Time`` để truy vết khuôn gốc và để
    :func:`retarget_account` nhận ra trường Subject nào đang phản chiếu danh tính khuôn.
    """
    if n < 0:
        raise ValueError(f"n phải ≥ 0, nhận {n}.")
    if templates.height == 0:
        raise TemplateNotFoundError("Không có khuôn để nhân bản.")
    with_replacement = (n > templates.height) if replace is None else bool(replace)
    idx = rng.choice(templates.height, size=n, replace=with_replacement)
    out = templates.select(pl.all().gather(pl.Series(idx, dtype=pl.Int64)))
    tpl_cols = [
        pl.col(c).alias(f"_tpl_{c}") for c in ("UserName", "DomainName", "Time") if f"_tpl_{c}" not in out.columns
    ]
    return out.with_columns(tpl_cols) if tpl_cols else out


# --------------------------------------------------------------------------- thao tác chỉ sửa đúng trường

def retarget_account(events: pl.DataFrame, account: Account) -> pl.DataFrame:
    """
    Gán danh tính nạn nhân: ``UserName``/``DomainName`` <- nạn nhân (``domain_raw``). Trường Subject chỉ
    bị đổi khi ``SubjectUserName`` đang PHẢN CHIẾU tài khoản của khuôn (``== _tpl_UserName``) — để không
    để lộ tên tài khoản khác; ``SubjectDomainName`` chỉ đổi theo khi cả cặp đều phản chiếu. Subject khác
    (vd. tài khoản máy ánh xạ ``C555$``) giữ nguyên cả cặp.
    """
    src_user = pl.col("_tpl_UserName") if "_tpl_UserName" in events.columns else pl.col("UserName")
    src_dom = pl.col("_tpl_DomainName") if "_tpl_DomainName" in events.columns else pl.col("DomainName")
    mirror = pl.col("SubjectUserName").eq_missing(src_user) & src_user.is_not_null()
    return events.with_columns(
        pl.when(mirror).then(pl.lit(account.user)).otherwise(pl.col("SubjectUserName")).alias("SubjectUserName"),
        pl.when(mirror & pl.col("SubjectDomainName").eq_missing(src_dom))
        .then(pl.lit(account.domain_raw)).otherwise(pl.col("SubjectDomainName")).alias("SubjectDomainName"),
        pl.lit(account.user, pl.String).alias("UserName"),
        pl.lit(account.domain_raw, pl.String).alias("DomainName"),
    )


def _broadcast(values: Any, n: int, name: str, dtype: pl.DataType) -> pl.Series:
    if isinstance(values, (str, int, np.integer)) or values is None:
        return pl.Series(name, [values] * n, dtype=dtype)
    s = pl.Series(name, list(values), dtype=dtype)
    if s.len() != n:
        raise ValueError(f"{name}: cần {n} giá trị, nhận {s.len()}.")
    return s


def set_times(events: pl.DataFrame, times: Sequence[int]) -> pl.DataFrame:
    """Chỉ đổi ``Time``."""
    return events.with_columns(_broadcast(times, events.height, "Time", pl.Int64))


def shift_time(events: pl.DataFrame, delta_s: Any) -> pl.DataFrame:
    """Chỉ đổi ``Time``: cộng ``delta_s`` (một số hoặc một giá trị / dòng). Khoảng cách tương đối được giữ."""
    delta = pl.lit(_broadcast(delta_s, events.height, "_d", pl.Int64))
    return events.with_columns((pl.col("Time") + delta).alias("Time"))


def set_source(events: pl.DataFrame, sources: Any) -> pl.DataFrame:
    """Chỉ đổi ``Source`` (một giá trị hoặc một giá trị / dòng)."""
    return events.with_columns(_broadcast(sources, events.height, "Source", pl.String))


def set_loghost(events: pl.DataFrame, loghosts: Any) -> pl.DataFrame:
    """Chỉ đổi ``LogHost`` (một giá trị hoặc một giá trị / dòng)."""
    return events.with_columns(_broadcast(loghosts, events.height, "LogHost", pl.String))


def _place(events: pl.DataFrame, account: Account, times: Sequence[int], source: Any, loghost: Any) -> pl.DataFrame:
    out = retarget_account(events, account)
    if source is not None:
        out = set_source(out, source)
    if loghost is not None:
        out = set_loghost(out, loghost)
    return set_times(out, times)


def add_failures(
    pool: TemplatePool,
    account: Account,
    times: Sequence[int],
    rng: np.random.Generator,
    *,
    source: Any = None,
    loghost: Any = None,
    fail_kind: str = "bad_pw",
    logon_type: Optional[int] = 3,
    prefer_own: bool = True,
    allow_peer: bool = True,
) -> pl.DataFrame:
    """
    ``len(times)`` sự kiện 4625 nhân bản từ khuôn thật có đúng ``fail_kind`` (FailureReason) và LogonType.
    Chỉ đổi danh tính, ``Time``, và ``Source``/``LogHost`` nếu được truyền (``None`` = giữ của khuôn).
    """
    tpl = pool.pick(len(times), rng, account=account, event_id=4625, logon_type=logon_type,
                    fail_kind=fail_kind, prefer_own=prefer_own, allow_peer=allow_peer)
    return _place(tpl, account, times, source, loghost)


def add_successes(
    pool: TemplatePool,
    account: Account,
    times: Sequence[int],
    rng: np.random.Generator,
    *,
    source: Any = None,
    loghost: Any = None,
    logon_type: Optional[int] = None,
    prefer_own: bool = True,
    allow_peer: bool = True,
    extra: Optional[pl.Expr] = None,
) -> pl.DataFrame:
    """``len(times)`` sự kiện 4624 nhân bản từ khuôn thật (cùng loại tài khoản; LogonType nếu chỉ định)."""
    tpl = pool.pick(len(times), rng, account=account, event_id=4624, logon_type=logon_type,
                    prefer_own=prefer_own, allow_peer=allow_peer, extra=extra)
    return _place(tpl, account, times, source, loghost)


def retype_logon(
    pool: TemplatePool,
    account: Account,
    times: Sequence[int],
    rng: np.random.Generator,
    *,
    logon_type: int,
    source: Any = None,
    loghost: Any = None,
    prefer_own: bool = True,
    allow_peer: bool = True,
) -> pl.DataFrame:
    """
    Đăng nhập với LogonType MỚI. KHÔNG sửa trường ``LogonType`` của sự kiện có sẵn (sẽ làm
    ``AuthenticationPackage``/``LogonTypeDescription``/``ProcessName`` lệch nhau); thay vào đó nhân bản
    sự kiện 4624 thật có ĐÚNG LogonType đó để mọi trường đi kèm khớp nhau, rồi chỉ đặt danh tính,
    thời gian, Source/LogHost.
    """
    out = add_successes(pool, account, times, rng, source=source, loghost=loghost, logon_type=int(logon_type),
                        prefer_own=prefer_own, allow_peer=allow_peer)
    if out.height and not (out["LogonType"] == int(logon_type)).all():
        raise InjectionLayoutError("retype_logon sinh sự kiện sai LogonType.")  # phòng thủ — không xảy ra
    return out


# --------------------------------------------------------------------------- profile thời gian

@dataclass(frozen=True)
class TimeProfile:
    """
    Cách rải sự kiện theo thời gian, độc lập kịch bản.

    ``mode``:
      * ``burst``  — mọi sự kiện của một ngày nằm trong một cửa sổ dài ``burst_duration_s`` (rút trong
        [min, max]) đặt ngẫu nhiên trong ``hour_window``;
      * ``spread`` — rải đều trong ``hour_window``;
      * ``replay`` — giữ NGUYÊN khoảng cách giữa các sự kiện nguồn (``offsets`` của :func:`schedule`),
        chỉ dời cả khối vào ``hour_window``.
    ``n_days``: số ngày trải, bắt đầu từ ngày dự định (mặc định 1 — hiện tại chỉ dùng 1).
    ``hour_window``: ``[h0, h1)``; ``h0 > h1`` = ``[h0,24) ∪ [0,h1)`` của CÙNG ngày lịch.
    """

    mode: str = "spread"
    n_days: int = 1
    hour_window: Tuple[int, int] = (0, 24)
    burst_duration_s: Tuple[int, int] = (600, 600)

    def __post_init__(self) -> None:
        if self.mode not in ("burst", "spread", "replay"):
            raise ValueError(f"mode phải là burst/spread/replay, nhận {self.mode!r}.")
        if int(self.n_days) < 1:
            raise ValueError(f"n_days phải ≥ 1, nhận {self.n_days}.")
        h0, h1 = (int(x) for x in self.hour_window)
        if not (0 <= h0 <= 24 and 0 <= h1 <= 24) or h0 == h1 or (h0 == 24):
            raise ValueError(f"hour_window không hợp lệ: {self.hour_window}.")
        lo, hi = (int(x) for x in self.burst_duration_s)
        if lo < 1 or hi < lo:
            raise ValueError(f"burst_duration_s không hợp lệ: {self.burst_duration_s}.")

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "TimeProfile":
        cfg = dict(cfg or {})
        bd = cfg.get("burst_duration_s", (600, 600))
        bd = (int(bd), int(bd)) if isinstance(bd, (int, float)) else (int(bd[0]), int(bd[1]))
        hw = cfg.get("hour_window", (0, 24))
        return cls(
            mode=str(cfg.get("mode", "spread")),
            n_days=int(cfg.get("n_days", 1)),
            hour_window=(int(hw[0]), int(hw[1])),
            burst_duration_s=bd,
        )

    def days(self, start_day: int) -> List[int]:
        return [int(start_day) + i for i in range(int(self.n_days))]

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "n_days": int(self.n_days), "hour_window": list(self.hour_window),
                "burst_duration_s": list(self.burst_duration_s)}


def allowed_segments(hour_window: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Các đoạn ``[a, b)`` (giây tính từ 00:00) hợp lệ trong MỘT ngày cho ``hour_window``."""
    h0, h1 = int(hour_window[0]), int(hour_window[1])
    if h0 < h1:
        return [(h0 * 3600, h1 * 3600)]
    segs = [(h0 * 3600, SECONDS_PER_DAY)]
    if h1 > 0:
        segs.append((0, h1 * 3600))
    return segs


def _pick_segment(segs: List[Tuple[int, int]], need: int, rng: np.random.Generator) -> Tuple[int, int]:
    """
    Một đoạn ``[a, b)`` có chỗ cho khối mà mốc cuối cách mốc đầu ``need`` giây (vị trí bắt đầu hợp lệ:
    ``a ≤ s < b − need``), xác suất ∝ số vị trí đặt được.
    """
    room = np.array([max(0, (b - a) - need) for a, b in segs], dtype=float)  # số vị trí bắt đầu hợp lệ
    if room.sum() <= 0:
        raise ScheduleError(f"Không đoạn nào của cửa sổ chứa được khối dài {need}s (đoạn: {segs}).")
    return segs[int(rng.choice(len(segs), p=room / room.sum()))]


def _times_one_day(
    profile: TimeProfile, day: int, k: int, rng: np.random.Generator, offsets: Optional[np.ndarray]
) -> np.ndarray:
    start, _ = day_window(day)
    segs = allowed_segments(profile.hour_window)
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    if profile.mode == "replay":
        rel = np.asarray(offsets, dtype=np.int64)
        rel = rel - rel.min()
        a, b = _pick_segment(segs, int(rel.max()), rng)
        s = int(rng.integers(a, b - int(rel.max())))
        return start + s + np.sort(rel)
    if profile.mode == "spread":
        lengths = np.array([b - a for a, b in segs], dtype=float)
        seg_idx = rng.choice(len(segs), size=k, p=lengths / lengths.sum())
        sec = np.array([rng.integers(segs[i][0], segs[i][1]) for i in seg_idx], dtype=np.int64)
        return start + np.sort(sec)
    # burst
    lo, hi = profile.burst_duration_s
    longest = max(b - a for a, b in segs)
    dur = min(int(rng.integers(lo, hi + 1)), longest)
    a, b = _pick_segment(segs, dur - 1, rng)          # mốc cuối ≤ s + dur − 1 < b
    s = int(rng.integers(a, b - dur + 1))
    return start + s + np.sort(rng.integers(0, dur, size=k)).astype(np.int64)


def schedule(
    profile: TimeProfile,
    start_day: int,
    n_events: int,
    rng: np.random.Generator,
    offsets: Optional[Sequence[int]] = None,
) -> pl.DataFrame:
    """
    Sinh ``n_events`` mốc thời gian theo ``profile`` từ ngày ``start_day`` -> bảng ``(day Int32, Time Int64)``
    đã sắp xếp. Với ``n_days > 1`` sự kiện được chia đều liên tiếp cho các ngày (``replay``: chia chuỗi
    ``offsets`` thành các đoạn liên tiếp, mỗi đoạn giữ nguyên khoảng cách nội bộ).

    Kiểm tra cứng trước khi trả: mỗi ``Time`` nằm trong cửa sổ ngày của nó (:func:`assert_within_days`).
    """
    if profile.mode == "replay":
        if offsets is None:
            raise ScheduleError("mode replay cần offsets (mốc thời gian của chuỗi nguồn).")
        offs = np.sort(np.asarray(offsets, dtype=np.int64))
        if offs.size != n_events:
            raise ScheduleError(f"replay: {offs.size} offsets nhưng n_events = {n_events}.")
    days = profile.days(start_day)
    chunks = np.array_split(np.arange(n_events), len(days))
    parts: List[pl.DataFrame] = []
    for d, idx in zip(days, chunks):
        sub = offs[idx] if profile.mode == "replay" else None
        t = _times_one_day(profile, d, int(idx.size), rng, sub)
        parts.append(pl.DataFrame({"day": pl.Series([d] * t.size, dtype=pl.Int32),
                                   "Time": pl.Series(t, dtype=pl.Int64)}))
    out = pl.concat(parts) if parts else pl.DataFrame(schema={"day": pl.Int32, "Time": pl.Int64})
    assert_within_days(out, days)
    return out


def assert_within_days(events: pl.DataFrame, days: Iterable[int], day_col: Optional[str] = "day") -> None:
    """
    Kiểm tra "không vượt nửa đêm". Có cột ``day_col``: mỗi dòng phải nằm trong cửa sổ của ĐÚNG ngày ghi ở
    cột đó, và ngày đó phải thuộc ``days``. Không có cột: ngày suy từ ``Time`` phải thuộc ``days``.
    """
    allowed = sorted(set(int(d) for d in days))
    if day_col and day_col in events.columns:
        bad_day = events.filter(~pl.col(day_col).is_in(allowed))
        if bad_day.height:
            raise InjectionLayoutError(f"{bad_day.height} sự kiện gán ngày ngoài {allowed}.")
        for d in allowed:
            assert_events_within_day(events.filter(pl.col(day_col) == d), d)
        return
    derived = events.select(day_of_time().alias("_d"))["_d"]
    if events["Time"].null_count() or not derived.is_in(allowed).all():
        raise InjectionLayoutError(f"Có sự kiện nằm ngoài các ngày dự định {allowed}.")


# --------------------------------------------------------------------------- tiện ích

def changed_fields(before: pl.DataFrame, after: pl.DataFrame, columns: Optional[Iterable[str]] = None) -> Set[str]:
    """Tập cột interim có ít nhất một giá trị khác nhau (null == null) giữa hai khung cùng số dòng."""
    if before.height != after.height:
        raise ValueError("Hai khung phải cùng số dòng.")
    cols = list(columns) if columns is not None else INTERIM_COLUMNS
    return {c for c in cols if not before[c].eq_missing(after[c]).all()}


def draw_int(spec: Any, rng: np.random.Generator) -> int:
    """Tham số config: số -> chính nó; ``[a, b]`` -> số nguyên rút đều trong [a, b]."""
    if isinstance(spec, (list, tuple)):
        a, b = int(spec[0]), int(spec[1])
        if b < a:
            raise ValueError(f"Khoảng không hợp lệ {spec}.")
        return int(rng.integers(a, b + 1))
    return int(spec)
