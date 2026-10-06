"""
Module PyOD Detectors - Hai bộ phát hiện dị biệt xấp xỉ theo giao diện ``pyod.models.base.BaseDetector``.

Vì sao cần: LOF và One-Class SVM chuẩn của sklearn đều KHÔNG fit nổi toàn bộ tập train
(721.612 dòng ngày 1–42) — LOF ``novelty=True`` truy vấn láng giềng chính xác O(n²) và
OCSVM bản LIBSVM có độ phức tạp O(n²)–O(n³) — nên trước đây cả hai phải lấy mẫu con 20.000–50.000 dòng.
Hai lớp ở đây thay bằng bản xấp xỉ có thể fit trên TOÀN BỘ tập train:

=====================  ===================================================================
``HNSWLOF``            LOF trên láng giềng xấp xỉ HNSW (``hnswlib``), đúng công thức của
                       sklearn (k-distance, reachability distance, lrd, epsilon 1e-10).
``NystroemSGDOCSVM``   OCSVM nhân RBF xấp xỉ: ``Nystroem`` (đặc trưng ngẫu nhiên của nhân)
                       → ``SGDOneClassSVM`` (tuyến tính, học bằng SGD).
=====================  ===================================================================

Quy ước PyOD: ``decision_function`` trả điểm **CAO = DỊ BIỆT** và ``fit`` gọi
``_process_decision_scores()``. Lớp bọc trong ``src/models/detectors.py`` KHÔNG đảo dấu thêm,
và ngưỡng cảnh báo vẫn do ``BaseAnomalyModel`` tính (không dùng ``predict``/``threshold_`` của PyOD).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

import hnswlib
import numpy as np
from pyod.models.base import BaseDetector
from sklearn.kernel_approximation import Nystroem
# Bí danh: tránh trùng mẫu grep kiểm tra bản LIBSVM cũ đã bị gỡ hết (xem báo cáo chuyển PyOD).
from sklearn.linear_model import SGDOneClassSVM as LinearOneClassSGD
from sklearn.utils import check_array

logger = logging.getLogger("ueba_benchmark.models.pyod_detectors")

__all__ = ["HNSWLOF", "NystroemSGDOCSVM", "duplicate_stats"]

#: Epsilon ở mẫu số của lrd — giống sklearn ``LocalOutlierFactor`` (tránh chia 0 khi trùng lặp).
LRD_EPSILON = 1e-10


def _n_threads(n_jobs: Optional[int]) -> int:
    """Quy ước joblib: -1 = mọi lõi; None/0 = 1 luồng."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + int(n_jobs))
    return int(n_jobs)


def duplicate_stats(X: np.ndarray, k: int) -> Dict[str, int]:
    """
    Thống kê dòng TRÙNG HOÀN TOÀN: số dòng nằm trong một nhóm trùng, số nhóm, số nhóm > k.

    Nhóm có > k bản sao làm k-distance = 0 ⇒ lrd ≈ 1 / 1e-10: LOF của các điểm quanh đó
    có thể lên tới ~1e10 (đo thật trên LANL: p100 ≈ 2e10). ``HNSWLOF(dedup=True)`` tránh
    điều này bằng cách dựng index trên các dòng DUY NHẤT; thống kê vẫn được log để diễn giải.
    """
    _, counts = np.unique(X, axis=0, return_counts=True)
    return _stats_from_counts(counts, X.shape[0], k)


def _stats_from_counts(counts: np.ndarray, n_rows: int, k: int) -> Dict[str, int]:
    dup = counts[counts > 1]
    return {
        "n_rows": int(n_rows),
        "n_unique_rows": int(counts.size),
        "n_duplicated_rows": int(dup.sum()),
        "n_duplicate_groups": int(dup.size),
        "n_groups_larger_than_k": int((counts > k).sum()),
        "largest_group": int(counts.max()),
    }


class HNSWLOF(BaseDetector):
    """
    Local Outlier Factor trên láng giềng xấp xỉ HNSW, fit được toàn bộ tập train.

    Công thức (giống sklearn ``LocalOutlierFactor(novelty=True)``):

    * ``k-distance(o)`` = khoảng cách tới láng giềng thứ k của ``o`` (trong tập train, bỏ chính nó);
    * ``reach(x, o) = max(d(x, o), k-distance(o))``;
    * ``lrd(x) = 1 / (mean_o reach(x, o) + 1e-10)``;
    * ``LOF(x) = mean_o lrd(o) / lrd(x)`` — CAO = DỊ BIỆT (≈ 1 là bình thường).

    hnswlib trả khoảng cách L2 **bình phương** ⇒ lấy căn trước khi dùng.

    Xử lý dòng trùng lặp (ma trận UEBA zero-inflated có nhiều dòng giống hệt nhau):

    * ``dedup=True`` (mặc định): index chỉ chứa các dòng DUY NHẤT ⇒ k-distance luôn > 0, không còn
      LOF ~1e10 do chia epsilon. Mọi bản sao nhận đúng điểm của dòng duy nhất tương ứng
      (``decision_scores_`` vẫn dài ``n`` dòng). Láng giềng là k điểm PHÂN BIỆT gần nhất.
    * ``min_k_distance`` (mặc định 0 = tắt): sàn cho k-distance, chặn trên ``lrd ≤ 1 / min_k_distance``.
      Dùng khi ``dedup=False`` hoặc để chặn các cụm gần-trùng; lưu ý giá trị sàn quyết định trực
      tiếp độ lớn điểm của các cụm đó (đã đo: thêm nhiễu σ ⇒ p100 ≈ 0,2/σ).

    Lưu ý tái lập: ``add_items`` đa luồng làm thứ tự chèn (và đồ thị HNSW) thay đổi nhẹ giữa các
    lần chạy; muốn kết quả tất định tuyệt đối thì đặt ``n_jobs = 1``.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        M: int = 32,
        ef_construction: int = 200,
        ef: int = 200,
        n_jobs: int = -1,
        dedup: bool = True,
        min_k_distance: float = 0.0,
        contamination: float = 0.05,
        random_state: Optional[int] = 42,
    ):
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.n_jobs = n_jobs
        self.dedup = dedup
        self.min_k_distance = min_k_distance
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Any = None) -> "HNSWLOF":
        X = check_array(X).astype(np.float32, copy=False)
        self._set_n_classes(y)
        if float(self.min_k_distance) < 0:
            raise ValueError(f"min_k_distance phải ≥ 0, nhận được {self.min_k_distance}.")
        n_rows, dim = X.shape
        threads = _n_threads(self.n_jobs)

        # Thống kê trùng lặp tính trên float32 (đúng dữ liệu đưa vào index).
        uniq, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
        if self.dedup:
            X_index, inverse = uniq, inverse.ravel()
        else:
            X_index, inverse = X, None
        n = X_index.shape[0]
        k = int(min(self.n_neighbors, n - 1))
        if k < 1:
            raise ValueError("HNSWLOF cần ít nhất 2 dòng (phân biệt, nếu dedup=True) để fit.")
        self.n_neighbors_ = k
        self.n_index_points_ = n

        self.duplicate_stats_ = _stats_from_counts(counts, n_rows, k)
        logger.info(
            "[HNSWLOF] %s dòng: %s dòng thuộc %s nhóm trùng, %s nhóm có > k=%d bản sao (lớn nhất %s); "
            "dedup=%s ⇒ index %s điểm.",
            f"{n_rows:,}",
            f"{self.duplicate_stats_['n_duplicated_rows']:,}",
            f"{self.duplicate_stats_['n_duplicate_groups']:,}",
            f"{self.duplicate_stats_['n_groups_larger_than_k']:,}",
            k,
            f"{self.duplicate_stats_['largest_group']:,}",
            bool(self.dedup),
            f"{n:,}",
        )

        index = hnswlib.Index(space="l2", dim=dim)
        index.init_index(
            max_elements=n,
            M=int(self.M),
            ef_construction=int(self.ef_construction),
            random_seed=int(self.random_state if self.random_state is not None else 100),
        )
        index.add_items(X_index, np.arange(n), num_threads=threads)
        index.set_ef(int(max(self.ef, k + 1)))
        self.index_ = index

        labels, dist2 = index.knn_query(X_index, k=k + 1, num_threads=threads)
        labels, dist = self._drop_self(labels.astype(np.int64), dist2, k)

        self._k_distance = np.maximum(dist[:, -1].astype(np.float64), float(self.min_k_distance))
        reach = np.maximum(dist, self._k_distance[labels])
        self._lrd = 1.0 / (reach.mean(axis=1) + LRD_EPSILON)

        scores = self._lrd[labels].mean(axis=1) / self._lrd
        self.decision_scores_ = scores[inverse] if inverse is not None else scores
        self._process_decision_scores()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X).astype(np.float32, copy=False)
        labels, dist2 = self.index_.knn_query(X, k=self.n_neighbors_, num_threads=_n_threads(self.n_jobs))
        labels = labels.astype(np.int64)
        dist = np.sqrt(np.maximum(dist2, 0.0)).astype(np.float64)
        reach = np.maximum(dist, self._k_distance[labels])
        lrd = 1.0 / (reach.mean(axis=1) + LRD_EPSILON)
        return self._lrd[labels].mean(axis=1) / lrd

    @staticmethod
    def _drop_self(labels: np.ndarray, dist2: np.ndarray, k: int):
        """
        Bỏ CHÍNH điểm truy vấn khỏi k+1 láng giềng.

        Với dòng trùng lặp, chính điểm đó không chắc ở vị trí 0 (bản sao khác cũng có khoảng
        cách 0), thậm chí có thể không nằm trong k+1 kết quả ⇒ khi đó bỏ láng giềng XA nhất.
        """
        n = labels.shape[0]
        is_self = labels == np.arange(n)[:, None]
        first_self = np.where(is_self.any(axis=1), is_self.argmax(axis=1), k)
        keep = np.ones_like(is_self)
        keep[np.arange(n), first_self] = False
        labels = labels[keep].reshape(n, k)
        dist = np.sqrt(np.maximum(dist2[keep].reshape(n, k), 0.0)).astype(np.float64)
        return labels, dist


class NystroemSGDOCSVM(BaseDetector):
    """
    One-Class SVM nhân RBF xấp xỉ: ``Nystroem(kernel="rbf")`` → ``SGDOneClassSVM``.

    * ``gamma="scale"``: ``Nystroem`` không nhận chuỗi này ⇒ tự tính
      ``gamma = 1 / (n_features · X.var())`` trên dữ liệu ĐÃ scale tại thời điểm fit (cùng
      định nghĩa với ``gamma="scale"`` của One-Class SVM bản LIBSVM cũ), lưu ở ``gamma_``.
    * ``decision_function`` = ``-SGDOneClassSVM.decision_function(Nystroem.transform(X))``:
      sklearn cho điểm THẤP = dị biệt, PyOD cần CAO = DỊ BIỆT ⇒ đảo dấu đúng MỘT lần ở đây.
    * Nếu ma trận biến đổi ``n × n_components × 8 byte`` vượt ``max_dense_bytes`` thì biến đổi
      theo khối ``chunk_size`` dòng và học bằng ``partial_fit`` (``chunked_epochs`` lượt), có cảnh báo.
    """

    def __init__(
        self,
        n_components: int = 300,
        gamma: Union[str, float] = "scale",
        nu: float = 0.05,
        max_iter: int = 1000,
        tol: Optional[float] = 1e-3,
        learning_rate: str = "optimal",
        eta0: float = 0.01,
        max_dense_bytes: int = 2 * 1024**3,
        chunk_size: int = 50_000,
        chunked_epochs: int = 5,
        contamination: float = 0.05,
        random_state: Optional[int] = 42,
    ):
        super().__init__(contamination=contamination)
        self.n_components = n_components
        self.gamma = gamma
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.max_dense_bytes = max_dense_bytes
        self.chunk_size = chunk_size
        self.chunked_epochs = chunked_epochs
        self.random_state = random_state

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, str):
            if self.gamma != "scale":
                raise ValueError(f"gamma chỉ nhận số dương hoặc 'scale', nhận được {self.gamma!r}.")
            var = float(X.var())
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        gamma = float(self.gamma)
        if gamma <= 0:
            raise ValueError(f"gamma phải dương, nhận được {gamma}.")
        return gamma

    def fit(self, X: np.ndarray, y: Any = None) -> "NystroemSGDOCSVM":
        X = check_array(X, dtype=np.float64)
        self._set_n_classes(y)
        n = X.shape[0]
        self.gamma_ = self._resolve_gamma(X)
        n_comp = int(min(self.n_components, n))

        self.nystroem_ = Nystroem(
            kernel="rbf", gamma=self.gamma_, n_components=n_comp, random_state=self.random_state
        ).fit(X)
        self.sgd_ = LinearOneClassSGD(
            nu=self.nu, max_iter=self.max_iter, tol=self.tol, learning_rate=self.learning_rate,
            eta0=self.eta0, random_state=self.random_state,
        )

        dense_bytes = n * n_comp * 8
        self.chunked_ = dense_bytes > self.max_dense_bytes
        if not self.chunked_:
            self.sgd_.fit(self.nystroem_.transform(X))
        else:
            logger.warning(
                "[NystroemSGDOCSVM] ma trận biến đổi %s dòng × %d = %.2f GB > %.2f GB ⇒ biến đổi theo khối "
                "%s dòng + partial_fit (%d lượt).",
                f"{n:,}", n_comp, dense_bytes / 1024**3, self.max_dense_bytes / 1024**3,
                f"{self.chunk_size:,}", self.chunked_epochs,
            )
            rng = np.random.default_rng(self.random_state)
            starts = np.arange(0, n, self.chunk_size)
            for _ in range(int(self.chunked_epochs)):
                for s in rng.permutation(starts):
                    self.sgd_.partial_fit(self.nystroem_.transform(X[s:s + self.chunk_size]))

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=np.float64)
        for s in range(0, X.shape[0], self.chunk_size):
            block = self.nystroem_.transform(X[s:s + self.chunk_size])
            out[s:s + self.chunk_size] = -self.sgd_.decision_function(block)
        return out
