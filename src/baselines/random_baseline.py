"""
BASELINE 1 — NGẪU NHIÊN: điểm đều trong [0, 1), seed cố định.

Đây là mốc dưới (null hypothesis): điểm không mang thông tin gì về dòng, nên PR-AUC kỳ vọng bằng
đúng tỷ lệ dương tính π của tập đánh giá (precision của một bộ chọn ngẫu nhiên ở mọi mức recall là π).
Một phương pháp có PR-AUC không vượt rõ π thì không phát hiện được gì.

Không fit gì từ dữ liệu, không đọc nhãn. Điểm train và điểm đánh giá lấy từ hai luồng ngẫu nhiên
tách biệt ``default_rng([seed, stream])`` để hai tập không dùng chung một dãy số.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

__all__ = ["RandomBaseline", "expected_random_pr_auc", "multi_seed_scores", "STREAM_TRAIN", "STREAM_EVAL"]

STREAM_TRAIN = 0
STREAM_EVAL = 1


class RandomBaseline:
    """Điểm ngẫu nhiên đều, tái lập theo ``(seed, stream)``."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)

    def fit(self, n_train_rows: int) -> "RandomBaseline":
        """Không học gì; giữ API fit/score giống các baseline khác."""
        self.n_train_rows_ = int(n_train_rows)
        return self

    def score(self, n_rows: int, stream: int = STREAM_EVAL) -> np.ndarray:
        if n_rows < 0:
            raise ValueError(f"n_rows phải >= 0, nhận được {n_rows}.")
        return np.random.default_rng([self.seed, int(stream)]).random(int(n_rows))

    def get_metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "seed": self.seed}


def expected_random_pr_auc(labels: Any) -> float:
    """
    PR-AUC (average precision) kỳ vọng của bộ chấm điểm ngẫu nhiên = tỷ lệ dương tính của tập đánh giá.

    ``labels`` là nhãn 0/1 của tập đánh giá SAU khi đã loại ``eval_exclude`` (đúng tập mà PR-AUC dùng).
    Với mẫu hữu hạn, AP ngẫu nhiên lệch dương nhẹ so với π (cỡ O(1/n_pos)) — vì vậy runner đối chiếu
    với trung bình nhiều seed thay vì một seed.
    """
    arr = np.asarray(labels).ravel()
    if arr.size == 0:
        raise ValueError("labels rỗng.")
    return float(np.count_nonzero(arr) / arr.size)


def multi_seed_scores(seeds: Sequence[int], n_rows: int, stream: int = STREAM_EVAL) -> Dict[int, np.ndarray]:
    """Điểm ngẫu nhiên cho từng seed (để tính PR-AUC thực nghiệm nhiều seed)."""
    return {int(s): RandomBaseline(s).score(n_rows, stream) for s in seeds}
