"""
BASELINE 3 — LUẬT NGƯỠNG: mỗi kịch bản một luật, mỗi luật là một thống kê của dòng tài khoản × ngày.

| Luật | Thống kê                                          | Nguồn                                               |
|------|---------------------------------------------------|-----------------------------------------------------|
| R1   | failure_count                                     | round(total_logons × failure_ratio) — ma trận       |
| R2   | số tài khoản bị cùng một Source gây thất bại      | log 4625 (thông tin CHÉO tài khoản) — rule_stats.py |
| R3   | số sự kiện ngoài giờ                              | round(total_logons × off_hours_ratio) — ma trận     |
| R4   | số Source mới so với lịch sử tài khoản            | log 4624+4625 (mọi ngày < t) — rule_stats.py        |
| R5   | days_since_last_activity                          | đặc trưng core, dùng lại nguyên                     |
| R6   | số sự kiện có LogonType chưa thấy ở tài khoản     | log 4624+4625 (mọi ngày < t) — rule_stats.py        |

R1/R3 suy ra CHÍNH XÁC từ hai cột đã có (total_logons là số nguyên, ratio = count/total) nên không
tính lại từ log. Mọi thống kê đều "càng lớn càng đáng ngờ".

**Biến thể ECDF (mặc định, không tham số):** thành phần của luật r là

    c_r(x) = P_train(S_r < s_r(x))      (ECDF trái của thống kê trên TRAIN)
    score(x) = max_r c_r(x)

  * ECDF đưa 6 thống kê khác đơn vị (lần, tài khoản, ngày) về cùng thang "độ hiếm so với train" mà
    không cần chọn ngưỡng: c_r = 0.999 nghĩa là chỉ 0,1% dòng train có giá trị >= dòng này. Mức cắt
    cuối cùng do ngân sách cảnh báo chung quyết định (thresholding.py), không do từng luật.
  * Dùng ECDF TRÁI (P(S < s), = 1 − p-value đuôi phải) chứ không phải P(S <= s): thống kê đếm có khối
    lượng lớn ở 0 (vd. 95% dòng không có thất bại); P(S <= 0) = 0.95 sẽ biến dòng "bình thường nhất"
    thành dòng có điểm cao. Với ECDF trái, giá trị 0 luôn cho điểm 0.
  * Chỉ fit trên train ⇒ mọi giá trị test vượt max train đều nhận 1.0 (đồng hạng ở đỉnh — được báo
    cáo qua train_alert_rate/số cờ, không che bằng tie-break ngầm).
  * max qua luật = "luật nào kích hoạt cũng đủ" — đúng cách SOC vận hành một bộ luật.

**Biến thể ngưỡng ngoài (để đối chiếu):** c_R1 = 1[failure_count >= L] (L ước lượng từ log train),
c_R2 = 1[spray_accounts >= 30] (mốc Splunk); R3–R6 giữ ECDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import polars as pl

from src.baselines._common import ROW_KEYS, TrainMedianImputer, column_matrix, logger
from src.baselines.rule_stats import EVENT_STAT_COLUMNS

__all__ = ["RuleSpec", "RULES", "RULE_COLUMNS", "attach_rule_statistics", "EcdfRuleBaseline"]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    column: str
    scenario: str
    source: str


RULES: List[RuleSpec] = [
    RuleSpec("R1", "r1_failure_count", "brute force / dò mật khẩu", "round(total_logons × failure_ratio)"),
    RuleSpec("R2", "r2_spray_accounts", "password spraying", "log 4625: max số tài khoản/Source trong ngày"),
    RuleSpec("R3", "r3_off_hours_count", "hoạt động ngoài giờ", "round(total_logons × off_hours_ratio)"),
    RuleSpec("R4", "r4_new_sources", "đăng nhập từ nguồn mới", "log: Source chưa thấy ở mọi ngày < t"),
    RuleSpec("R5", "days_since_last_activity", "tài khoản ngủ đông quay lại", "đặc trưng core"),
    RuleSpec("R6", "r6_new_logontype_events", "đổi kiểu đăng nhập", "log: LogonType chưa thấy ở mọi ngày < t"),
]
RULE_COLUMNS: List[str] = [r.column for r in RULES]


def attach_rule_statistics(matrix: pl.DataFrame, event_stats: pl.DataFrame) -> pl.DataFrame:
    """Thêm 6 cột thống kê luật vào ma trận (R1/R3 suy từ cột có sẵn, R2/R4/R6 nối từ ``event_stats``)."""
    needed = ["total_logons", "failure_ratio", "off_hours_ratio", "days_since_last_activity"]
    missing = [c for c in needed if c not in matrix.columns]
    if missing:
        raise ValueError(f"Ma trận thiếu cột cần cho luật: {missing}.")
    stats = event_stats.select(ROW_KEYS + EVENT_STAT_COLUMNS).with_columns(pl.col("day").cast(matrix.schema["day"]))
    out = matrix.drop([c for c in EVENT_STAT_COLUMNS if c in matrix.columns]).with_columns(
        (pl.col("total_logons") * pl.col("failure_ratio")).round(0).alias("r1_failure_count"),
        (pl.col("total_logons") * pl.col("off_hours_ratio")).round(0).alias("r3_off_hours_count"),
    ).join(stats, on=ROW_KEYS, how="left", maintain_order="left")
    n_unmatched = int(out["r2_spray_accounts"].is_null().sum())
    if n_unmatched:
        logger.warning("[rules] %s dòng ma trận không khớp thống kê log (R2 -> 0, R4/R6 -> NULL).", f"{n_unmatched:,}")
        out = out.with_columns(pl.col("r2_spray_accounts").fill_null(0.0))
    return out


class EcdfRuleBaseline:
    """
    Điểm luật = max_r c_r(x). ``external`` = None -> mọi thành phần là ECDF train (biến thể chính);
    ``external = {"r1_failure_count": L, "r2_spray_accounts": 30}`` -> các luật đó thành chỉ báo 0/1.
    """

    def __init__(self, external: Optional[Mapping[str, float]] = None, name: Optional[str] = None) -> None:
        self.external: Dict[str, float] = {str(k): float(v) for k, v in (external or {}).items()}
        unknown = sorted(set(self.external) - set(RULE_COLUMNS))
        if unknown:
            raise ValueError(f"Ngưỡng ngoài cho cột không phải luật: {unknown}.")
        self.name = name or ("rule_external" if self.external else "rule_ecdf")
        self.imputer_ = TrainMedianImputer()

    def fit(self, train: pl.DataFrame) -> "EcdfRuleBaseline":
        X = column_matrix(train, RULE_COLUMNS, f"[{self.name}] train")
        X = self.imputer_.fit(X, RULE_COLUMNS).transform(X)
        self.sorted_train_ = [np.sort(X[:, j]) for j in range(X.shape[1])]
        self.n_train_ = int(X.shape[0])
        logger.info("[%s] ECDF fit trên %s dòng train; ngưỡng ngoài: %s", self.name, f"{self.n_train_:,}", self.external or "không")
        return self

    def components(self, df: pl.DataFrame) -> np.ndarray:
        """Ma trận (n, 6) thành phần c_r — để giải thích luật nào đẩy điểm của dòng."""
        if not hasattr(self, "sorted_train_"):
            raise RuntimeError(f"[{self.name}] chưa fit.")
        X = self.imputer_.transform(column_matrix(df, RULE_COLUMNS, f"[{self.name}] score"))
        out = np.empty_like(X)
        for j, col in enumerate(RULE_COLUMNS):
            if col in self.external:
                out[:, j] = (X[:, j] >= self.external[col]).astype(np.float64)
            else:
                ref = self.sorted_train_[j]
                out[:, j] = np.searchsorted(ref, X[:, j], side="left") / float(ref.size)
        return out

    def score(self, df: pl.DataFrame) -> np.ndarray:
        return self.components(df).max(axis=1)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_train": self.n_train_,
            "rules": [r.__dict__ for r in RULES],
            "external_thresholds": self.external,
            "impute_medians": self.imputer_.as_dict(RULE_COLUMNS),
        }
