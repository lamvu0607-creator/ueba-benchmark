"""
Phân tích theo loại bất thường (tuần 5): mô hình nào bắt được kịch bản nào **trong ngân sách cảnh báo**.

ROC-AUC / PR-AUC theo kịch bản (``grid._scenario_rows``) đo cả danh sách xếp hạng; ở đây đo thứ analyst thực sự thấy:

* ``recall_threshold``   — tỷ lệ nạn nhân bị cờ tại ngưỡng vận hành (phân vị 1 − contamination của điểm train);
* ``recall_top{k}_day``  — tỷ lệ nạn nhân nằm trong k điểm cao nhất CỦA NGÀY đó (ngân sách k cảnh báo/ngày);
* ``recall_top1pct``     — tỷ lệ nạn nhân nằm trong 1% điểm cao nhất toàn khối;
* ``median_rank_pct``    — hạng phần trăm trung vị của nạn nhân (0,01 = nằm ở top 1%; càng nhỏ càng tốt).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import rankdata

__all__ = ["daily_topk_mask", "top_fraction_mask", "rank_percentile", "scenario_detection_rows"]


def daily_topk_mask(scores: np.ndarray, days: np.ndarray, k: int) -> np.ndarray:
    """True cho k dòng điểm cao nhất của mỗi ngày (hoà điểm: thứ tự ổn định theo vị trí dòng)."""
    scores, days = np.asarray(scores, dtype=np.float64), np.asarray(days)
    mask = np.zeros(scores.size, dtype=bool)
    for d in np.unique(days):
        idx = np.flatnonzero(days == d)
        top = idx[np.argsort(-scores[idx], kind="stable")[:k]]
        mask[top] = True
    return mask


def top_fraction_mask(scores: np.ndarray, frac: float) -> np.ndarray:
    """True cho ceil(frac × n) dòng điểm cao nhất toàn khối."""
    scores = np.asarray(scores, dtype=np.float64)
    k = int(np.ceil(frac * scores.size))
    mask = np.zeros(scores.size, dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


def rank_percentile(scores: np.ndarray) -> np.ndarray:
    """Hạng phần trăm từ trên xuống: điểm cao nhất ≈ 1/n, thấp nhất = 1 (hoà điểm lấy hạng bi quan)."""
    scores = np.asarray(scores, dtype=np.float64)
    return rankdata(-scores, method="max") / scores.size


def scenario_detection_rows(y: np.ndarray, scenario: np.ndarray, scores: np.ndarray, flags: np.ndarray,
                            days: np.ndarray, daily_ks: Sequence[int] = (100,),
                            top_frac: float = 0.01) -> List[Dict[str, Any]]:
    """Một dòng chỉ số cho mỗi kịch bản (và dòng ``all`` cho mọi nạn nhân)."""
    y = np.asarray(y).astype(bool)
    scenario = np.asarray(scenario, dtype=object)
    flags = np.asarray(flags).astype(bool)
    day_masks = {k: daily_topk_mask(scores, days, k) for k in daily_ks}
    top = top_fraction_mask(scores, top_frac)
    pct = rank_percentile(scores)
    groups = [("all", y)] + [(sc, y & (scenario == sc)) for sc in sorted({s for s in scenario[y] if s is not None})]
    rows = []
    for name, pos in groups:
        if not pos.any():
            continue
        row = {"scenario": name, "n_positive": int(pos.sum()), "recall_threshold": float(flags[pos].mean())}
        row.update({f"recall_top{k}_day": float(m[pos].mean()) for k, m in day_masks.items()})
        row[f"recall_top{top_frac * 100:g}pct"] = float(top[pos].mean())
        row["median_rank_pct"] = float(np.median(pct[pos]))
        rows.append(row)
    return rows
