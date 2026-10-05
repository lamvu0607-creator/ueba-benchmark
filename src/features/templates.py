"""
Combinatorial Template cho đặc trưng Tài khoản × Ngày (schema v4.0).

Mỗi ứng viên là một tổ hợp 4 trục::

    PHÉP ĐO  ×  ĐỐI TƯỢNG SỰ KIỆN  ×  CHIỀU (thực thể)  ×  CỬA SỔ

Ví dụ ``novelty × success × Source × 7d`` = "số Source (chỉ đăng nhập thành công) hôm nay
chưa xuất hiện trong 7 ngày lịch trước". Vì mọi đặc trưng đều sinh từ cùng một ngữ pháp,
tên và câu diễn giải được sinh TỰ ĐỘNG từ tổ hợp ⇒ không có biến nào "không đọc được".

Module này chỉ làm phần **lý thuyết** của phễu:

* Bước 0 — ``enumerate_space``: sinh mọi tổ hợp, đánh dấu tổ hợp *vô nghĩa về ngữ pháp*.
* Vòng 1 — ``round1_filter``: lọc bằng LẬP LUẬN (4 luật R1–R4, ghi lý do cho từng ứng viên).

Phần tính toán nằm ở ``src/features/template_engine.py``; vòng 2 (lọc bằng SỐ LIỆU) ở
``scripts/feature_engineering/run_feature_funnel_v4.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, FrozenSet, List, Optional, Tuple

__all__ = [
    "MEASURES",
    "OBJECTS",
    "ENTITIES",
    "WINDOWS",
    "SCENARIOS",
    "SCENARIO_PRIORITY",
    "Candidate",
    "enumerate_space",
    "round1_filter",
    "build_catalog",
    "candidate_by_name",
]

# ---------------------------------------------------------------------------
# 4 trục của template (mỗi giá trị kèm cụm từ tiếng Việt để sinh câu diễn giải)
# ---------------------------------------------------------------------------

#: Đối tượng sự kiện — lọc sự kiện TRƯỚC khi đo.
OBJECTS: Dict[str, str] = {
    "all": "mọi sự kiện",
    "fail": "đăng nhập thất bại (4625)",
    "success": "đăng nhập thành công (4624)",
    "night": "sự kiện ngoài giờ (18h–7h)",
}

#: Chiều phân loại để đếm/so sánh. ``event`` = không phân loại (đếm sự kiện).
ENTITIES: Dict[str, str] = {
    "event": "sự kiện",
    "Source": "máy nguồn (Source)",
    "LogHost": "máy đích (LogHost)",
    "pair": "cặp (Source, LogHost)",
    "hour": "khung giờ",
    "logon_type": "loại logon",
    "auth": "gói xác thực",
    "fail_reason": "lý do thất bại",
}

#: Cửa sổ: ``1d`` = trong ngày; ``5m/15m/60m`` = khung cố định trong ngày (lấy khung lớn nhất);
#: ``7d/14d`` = ngày lịch TRƯỚC t (không gồm t); ``hist`` = toàn bộ lịch sử trước t.
WINDOWS: Dict[str, str] = {
    "1d": "trong ngày",
    "5m": "khung 5 phút",
    "15m": "khung 15 phút",
    "60m": "khung 60 phút",
    "7d": "7 ngày lịch trước",
    "14d": "14 ngày lịch trước",
    "hist": "toàn bộ lịch sử trước",
}
BUCKET_WINDOWS = ("5m", "15m", "60m")
HISTORY_WINDOWS = ("7d", "14d")


@dataclass(frozen=True)
class MeasureSpec:
    label: str              # mẫu câu, có {obj} {ent} {win}
    entities: str           # "event" | "non_event" | "any" | "source_only"
    windows: Tuple[str, ...]
    cost: int               # "đơn vị giải thích" (xem luật R3)
    direction: str          # "up" | "down" | "both": chiều được coi là bất thường
    needs_history: bool


MEASURES: Dict[str, MeasureSpec] = {
    "count": MeasureSpec(
        "số {obj}{win_suffix}", "event", ("1d",) + BUCKET_WINDOWS, 1, "up", False),
    "distinct": MeasureSpec(
        "số {ent} khác nhau trong {obj}{win_suffix}", "non_event", ("1d",) + BUCKET_WINDOWS, 1, "up", False),
    "share": MeasureSpec(
        "tỷ lệ {obj} trên tổng sự kiện trong ngày", "event", ("1d",), 1, "up", False),
    "evenness": MeasureSpec(
        "độ trải đều (Pielou) của {obj} theo {ent} trong ngày", "non_event", ("1d",), 2, "both", False),
    "fanout": MeasureSpec(
        "số tài khoản khác mà {ent} của tài khoản này chạm tới qua {obj} trong ngày (lấy max)",
        "source_only", ("1d",), 2, "up", False),
    "novelty": MeasureSpec(
        "số {ent} trong {obj} hôm nay chưa xuất hiện trong {win}", "non_event", HISTORY_WINDOWS, 1, "up", True),
    "novelty_share": MeasureSpec(
        "tỷ lệ {obj} hôm nay thuộc {ent} chưa xuất hiện trong {win}", "non_event", HISTORY_WINDOWS, 1, "up", True),
    "jaccard": MeasureSpec(
        "độ trùng (Jaccard) giữa tập {ent} hôm nay và tập {ent} của {win} ({obj})",
        "non_event", HISTORY_WINDOWS, 2, "down", True),
    "dist_shift": MeasureSpec(
        "khoảng cách phân bố (total variation) của {obj} theo {ent} giữa hôm nay và {win}",
        "non_event", HISTORY_WINDOWS, 2, "up", True),
    "deviation": MeasureSpec(
        "z bền vững của {base} hôm nay so với {win}", "any", HISTORY_WINDOWS, 2, "up", True),
    "delta_mean": MeasureSpec(
        "{base} hôm nay trừ trung bình của {win}", "any", HISTORY_WINDOWS, 1, "up", True),
    "peer_z": MeasureSpec(
        "z bền vững của {base} so với các tài khoản cùng loại (entity_type) trong cùng ngày",
        "any", ("1d",), 2, "up", False),
    "active_days": MeasureSpec(
        "số ngày có hoạt động trong {win}", "event", HISTORY_WINDOWS, 1, "down", True),
    "recency": MeasureSpec(
        "số ngày kể từ lần gần nhất thấy {ent} ({obj})", "recency", ("hist",), 1, "up", True),
}

# ---------------------------------------------------------------------------
# Kịch bản tấn công — mục 5.2 của docs/Tong quan de tai ueba.md (+ di chuyển ngang)
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, str] = {
    "brute_force": "Dò mật khẩu (brute-force)",
    "password_spraying": "Rải mật khẩu (password spraying)",
    "off_hours": "Hoạt động ngoài giờ",
    "new_workstation_burst": "Bùng nổ máy trạm mới",
    "dormant_wakeup": "Tài khoản ngủ đông thức dậy",
    "logon_type_switch": "Đổi loại logon bất thường",
    "lateral_fanout": "Di chuyển ngang (tài khoản chạm nhiều máy đích mới)",
}

#: Ưu tiên do thực tập sinh chốt (2026-10-04): A = brute/spray, B = chiếm đoạt tài khoản
#: (nguồn/giờ lạ), C = di chuyển ngang; còn lại xếp sau. Dùng để phân xử khi hai ứng viên
#: trùng trục ở cổng tương quan (giữ biến thuộc kịch bản ưu tiên cao hơn).
SCENARIO_PRIORITY: Dict[str, int] = {
    "brute_force": 1, "password_spraying": 1,
    "new_workstation_burst": 2, "off_hours": 2,
    "lateral_fanout": 3,
    "dormant_wakeup": 4, "logon_type_switch": 4,
}


@dataclass(frozen=True)
class Candidate:
    measure: str
    obj: str
    entity: str
    window: str
    status: str = "candidate"            # grammar_invalid | dropped_r1..r4 | round2
    reason: str = ""
    scenarios: FrozenSet[str] = field(default_factory=frozenset)

    # ---- tên & câu diễn giải sinh tự động -------------------------------
    @property
    def base(self) -> str:
        """Thống kê nền cho deviation/delta_mean/peer_z (xem ``base_label``)."""
        if self.entity != "event":
            return "distinct"
        return "count" if self.obj == "all" else "share"

    @property
    def name(self) -> str:
        ent = {"event": "", "Source": "source", "LogHost": "host", "pair": "pair",
               "hour": "hour", "logon_type": "logontype", "auth": "auth",
               "fail_reason": "failreason"}[self.entity]
        obj = "" if self.obj == "all" else self.obj
        parts = [self.measure]
        if self.measure in ("deviation", "delta_mean", "peer_z"):
            parts.append(self.base)
        parts += [p for p in (obj, ent) if p]
        if self.window != "1d":
            parts.append(self.window)
        return "_".join(parts)

    @property
    def cost(self) -> int:
        c = MEASURES[self.measure].cost
        c += 0 if self.obj == "all" else 1
        c += 1 if self.entity in ("pair", "auth", "fail_reason") else 0
        c += 1 if self.window in ("14d", "hist") else 0
        return c

    @property
    def direction(self) -> str:
        return MEASURES[self.measure].direction

    @property
    def needs_history(self) -> bool:
        return MEASURES[self.measure].needs_history

    def base_label(self) -> str:
        obj = OBJECTS[self.obj]
        if self.base == "count":
            return f"log số {obj}"
        if self.base == "share":
            return f"tỷ lệ {obj}"
        return f"log số {ENTITIES[self.entity]} khác nhau ({obj})"

    @property
    def description(self) -> str:
        spec = MEASURES[self.measure]
        win = WINDOWS[self.window]
        win_suffix = "" if self.window == "1d" else f" trong {win} đông nhất của ngày"
        return spec.label.format(
            obj=OBJECTS[self.obj], ent=ENTITIES[self.entity], win=win,
            win_suffix=win_suffix, base=self.base_label(),
        )

    @property
    def priority(self) -> int:
        return min((SCENARIO_PRIORITY[s] for s in self.scenarios), default=9)

    def with_status(self, status: str, reason: str = "", scenarios=None) -> "Candidate":
        return Candidate(self.measure, self.obj, self.entity, self.window, status, reason,
                         frozenset(scenarios if scenarios is not None else self.scenarios))


# ---------------------------------------------------------------------------
# Bước 0 — ngữ pháp: tổ hợp nào có NGHĨA về mặt định nghĩa
# ---------------------------------------------------------------------------
def _grammar_error(measure: str, obj: str, entity: str, window: str) -> Optional[str]:
    spec = MEASURES[measure]
    if window not in spec.windows:
        return f"phép đo '{measure}' không dùng cửa sổ '{window}'"
    if spec.entities == "event" and entity != "event":
        return f"phép đo '{measure}' chỉ áp dụng cho đếm sự kiện"
    if spec.entities == "non_event" and entity == "event":
        return f"phép đo '{measure}' cần một chiều thực thể"
    if spec.entities == "source_only" and entity != "Source":
        return "fanout chỉ định nghĩa cho Source (một nguồn → nhiều tài khoản)"
    if spec.entities == "recency" and (entity not in ("event", "Source", "LogHost") or obj != "all"):
        return "recency chỉ định nghĩa cho tài khoản / Source / LogHost trên mọi sự kiện"
    if measure == "share" and obj == "all":
        return "tỷ lệ của 'mọi sự kiện' luôn = 1"
    if measure == "active_days" and obj != "all":
        return "số ngày hoạt động chỉ định nghĩa trên mọi sự kiện"
    if entity == "fail_reason" and obj != "fail":
        return "FailureReason chỉ tồn tại ở 4625"
    if entity == "hour" and obj == "night":
        return "'ngoài giờ' được định nghĩa bằng chính khung giờ (vòng lặp định nghĩa)"
    if entity == "hour" and window in BUCKET_WINDOWS:
        return "khung ≤ 60 phút nằm gọn trong 1–2 khung giờ ⇒ số khung giờ gần như hằng số"
    return None


def enumerate_space() -> List[Candidate]:
    """Sinh MỌI tổ hợp 4 trục; tổ hợp vô nghĩa được gắn ``grammar_invalid`` kèm lý do."""
    out: List[Candidate] = []
    for m, o, e, w in product(MEASURES, OBJECTS, ENTITIES, WINDOWS):
        err = _grammar_error(m, o, e, w)
        c = Candidate(m, o, e, w)
        out.append(c.with_status("grammar_invalid", err) if err else c)
    return out


# ---------------------------------------------------------------------------
# Vòng 1 — lọc bằng LẬP LUẬN (4 luật, áp theo thứ tự; ghi luật đầu tiên bị vi phạm)
# ---------------------------------------------------------------------------

#: R1 — tổ hợp TRÙNG với một đặc trưng đang là core (v3.0) hoặc đã bị đo và loại.
#: Khoá = (measure, obj, entity, window).
EXISTING_EQUIVALENTS: Dict[Tuple[str, str, str, str], str] = {
    ("count", "all", "event", "1d"): "core: log_total_logons",
    ("share", "fail", "event", "1d"): "core: failure_ratio",
    ("share", "success", "event", "1d"): "= 1 − failure_ratio (phụ thuộc tuyến tính tuyệt đối)",
    ("share", "night", "event", "1d"): "core: off_hours_ratio",
    ("count", "fail", "event", "1d"): "đã loại: failure_count (ρ = 0,9969 với failure_ratio)",
    ("distinct", "all", "LogHost", "1d"): "core: log_distinct_hosts",
    ("distinct", "all", "Source", "1d"): "core: distinct_sources_count",
    ("evenness", "all", "hour", "1d"): "core: hour_entropy",
    ("evenness", "all", "LogHost", "1d"): "core: dst_host_entropy",
    ("novelty", "all", "Source", "7d"): "core: new_source_count_7d",
    ("novelty", "all", "LogHost", "7d"): "core: new_host_count_7d",
    ("novelty", "all", "Source", "14d"): "đã loại: new_source_count (toàn lịch sử ρ = 0,9999 với bản 7d)",
    ("novelty", "all", "LogHost", "14d"): "đã loại: new_host_count (ρ = 0,9999 với bản 7d)",
    ("novelty", "all", "pair", "7d"): "đã loại: source_host_pair_novelty (ρ = 0,9052 với new_host_count)",
    ("novelty", "all", "pair", "14d"): "đã loại: source_host_pair_novelty (ρ = 0,9052 với new_host_count)",
    ("deviation", "all", "event", "7d"): "core: volume_robust_z_7d",
    ("recency", "all", "event", "hist"): "core: days_since_last_activity",
    ("recency", "all", "Source", "hist"): "đã loại: source_recency (ρ = 0,9238 với days_since_last_activity)",
    ("evenness", "all", "logon_type", "1d"): "Tier B v3: logon_type_entropy — cờ zero-inflated (71,09% dòng chỉ 1 loại)",
    ("distinct", "all", "logon_type", "1d"): "đã loại: distinct_logon_types (ρ = 0,8799 với network_ratio)",
}

#: Phép đo "so với lịch sử của chính tài khoản" — chỉ những phép đo này mới bắt được
#: một sự THAY ĐỔI (dịch giờ, đổi loại logon, thức dậy sau ngủ đông).
HISTORY_COMPARE = ("novelty", "novelty_share", "jaccard", "dist_shift", "deviation", "delta_mean")


def scenarios_for(c: Candidate) -> FrozenSet[str]:
    """
    Bảng ánh xạ tổ hợp → kịch bản 5.2 mà tổ hợp đó *được thiết kế để bắt* (luật R2).

    Nguyên tắc: một kịch bản 5.2 mô tả MỘT cơ chế; tổ hợp chỉ được gắn kịch bản khi phép đo
    phản ánh đúng cơ chế đó — ví dụ "ngoài giờ" là *dịch chuyển* sang ban đêm (so với hồ sơ
    lịch sử), nên "số Source khác nhau vào ban đêm" (đo *cái gì* xảy ra ban đêm) không được gắn.
    """
    m, o, e, w = c.measure, c.obj, c.entity, c.window
    if o == "fail":
        # brute/spray = cường độ & độ tập trung thất bại theo nguồn/đích
        if e in ("event", "Source", "LogHost", "pair"):
            return frozenset({"brute_force", "password_spraying"})
        if e == "fail_reason":
            return frozenset({"brute_force"})          # "Account locked out" là dấu vết brute-force
        return frozenset()                             # loại logon của 4625: không phải cơ chế 5.2
    if o == "night":
        # dịch chuyển giờ: chỉ các phép đo trên KHỐI LƯỢNG ngoài giờ (không theo thực thể)
        return frozenset({"off_hours"}) if e == "event" else frozenset()
    # o ∈ {all, success}
    if e == "hour":
        return frozenset({"off_hours"}) if m in HISTORY_COMPARE else frozenset()
    if e == "logon_type":
        return frozenset({"logon_type_switch"}) if m in HISTORY_COMPARE else frozenset()
    if e in ("Source", "LogHost", "pair"):
        if m in ("evenness", "fanout"):
            return frozenset()                          # không đo "mới" cũng không đo "nhiều"
        if e == "Source":
            return frozenset({"new_workstation_burst"})
        if e == "LogHost":
            return frozenset({"lateral_fanout"})
        return frozenset({"lateral_fanout", "new_workstation_burst"})
    if e == "event" and m in ("delta_mean", "deviation", "active_days", "recency"):
        return frozenset({"dormant_wakeup"})            # thức dậy = so với lịch sử của CHÍNH tài khoản
    # auth (pass-the-hash), burst mọi sự kiện, peer của khối lượng: không thuộc 5.2
    return frozenset()


#: Ngưỡng luật R3: một câu diễn giải ngắn chịu được tối đa 3 "đơn vị giải thích".
MAX_COST = 3


def round1_filter(space: List[Candidate]) -> List[Candidate]:
    """Áp R1–R4 lên các tổ hợp hợp lệ về ngữ pháp; trả về toàn bộ danh sách kèm trạng thái."""
    out: List[Candidate] = []
    valid = [c for c in space if c.status != "grammar_invalid"]
    valid_keys = {(c.measure, c.obj, c.entity, c.window) for c in valid}
    for c in space:
        if c.status == "grammar_invalid":
            out.append(c)
            continue
        key = (c.measure, c.obj, c.entity, c.window)
        scen = scenarios_for(c)

        # R1 — đã có / đã đo và loại
        if key in EXISTING_EQUIVALENTS:
            out.append(c.with_status("dropped_r1", EXISTING_EQUIVALENTS[key], scen))
            continue
        # R2 — không gắn với kịch bản nào của 5.2
        if not scen:
            out.append(c.with_status("dropped_r2", "không nhắm kịch bản nào ở mục 5.2", scen))
            continue
        # R3 — khó diễn giải
        if c.cost > MAX_COST:
            out.append(c.with_status(
                "dropped_r3", f"cần {c.cost} đơn vị giải thích (> {MAX_COST}): không đọc được bằng một câu ngắn", scen))
            continue
        # R4 — biến thể dư cùng họ (giữ một đại diện để vòng 2 không đo lặp)
        r4 = _redundant_variant(c, valid_keys)
        if r4:
            out.append(c.with_status("dropped_r4", r4, scen))
            continue
        out.append(c.with_status("round2", "", scen))
    return out


def _redundant_variant(c: Candidate, valid_keys) -> Optional[str]:
    # (a) 14d khi đã có 7d: v3.0 đo bản toàn-lịch-sử vs 7d có ρ = 0,9999; 14d còn gấp đôi warm-up.
    if c.window == "14d" and (c.measure, c.obj, c.entity, "7d") in valid_keys:
        return "biến thể 14d của một ứng viên 7d (v3.0: novelty nhiều cửa sổ có ρ = 0,9999; warm-up gấp đôi)"
    # (b) khung trong ngày: giữ 15m làm đại diện (đủ chứa một đợt brute-force/spray 10–30 phút)
    if c.window in ("5m", "60m"):
        return "biến thể khung thời gian: giữ khung 15 phút làm đại diện"
    # (c) cùng thống kê nền & cửa sổ, hai cách chuẩn hoá (z bền vững vs hiệu số): giữ hiệu số
    #     (1 đơn vị giải thích thay vì 2). Riêng khối lượng, bản z đã là core (volume_robust_z_7d).
    if c.measure == "deviation":
        return "cùng nền với delta_mean nhưng khó giải thích hơn: giữ hiệu số so với trung bình 7 ngày"
    # (d) 'success' gần như trùng 'all' (87,36% dòng không có thất bại) — trừ họ novelty,
    #     nơi "thành công từ nguồn lạ" là đúng định nghĩa chiếm đoạt tài khoản.
    if c.obj == "success" and c.measure not in ("novelty", "novelty_share"):
        return "đối tượng 'thành công' gần trùng 'mọi sự kiện' (87,36% dòng không có thất bại)"
    return None


def build_catalog() -> List[Candidate]:
    """Toàn bộ không gian sau bước 0 + vòng 1 (thứ tự tất định)."""
    return round1_filter(enumerate_space())


def candidate_by_name(name: str) -> Candidate:
    for c in build_catalog():
        if c.name == name and c.status != "grammar_invalid":
            return c
    raise KeyError(f"Không có ứng viên template tên '{name}'")
