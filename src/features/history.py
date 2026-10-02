"""
Đặc trưng LỊCH SỬ (cross-day) cho ma trận Tài khoản × Ngày — nhóm Tier A của schema v3.0.

Bất biến của module này: **mọi giá trị tại ngày t chỉ được suy ra từ dữ liệu ≤ t−1**.

Vì sao cần một lượt tính RIÊNG (không nằm trong ``extract_features_single_day``):
``extract_features_single_day`` chỉ thấy **một ngày** nên không thể biết "Source này đã từng
xuất hiện chưa", "lần cuối thấy là ngày nào", "7 ngày trước khối lượng là bao nhiêu".
Các câu hỏi đó thuộc phạm vi **liên-ngày**, nên phải tính sau khi ghép toàn bộ dải ngày.

Hai quyết định thiết kế được ghi rõ (và được kiểm bằng test):

1. **Cửa sổ là NGÀY LỊCH, không phải "số dòng trước đó".** Ngày không có sự kiện không sinh
   dòng trong panel thưa, nên "7 ngày trước" được suy ra từ *ngày* thật (``prev_day < t − 7``),
   và baseline khối lượng dùng **lưới dày NỘI BỘ** với ngày trống = khối lượng 0
   (không thêm dòng nào vào ma trận ⇒ không đổi quy mô/ranh giới train-test).
2. **Bất định nghĩa ⇒ NULL, không bịa số.** ``source_recency`` NULL khi mọi Source trong ngày
   đều mới; ``volume_robust_z_7d`` NULL khi chưa đủ 7 ngày lịch sử (warm-up) — tầng model
   impute bằng median fit trên train (``src/models/pipeline.py``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import polars as pl

logger = logging.getLogger("ueba_benchmark.features.history")

__all__ = [
    "HISTORY_V3_FEATURES",
    "HISTORY_V3_REJECTED",
    "HISTORY_KEYS",
    "HISTORY_WINDOW_DAYS",
    "add_history_features",
    "entities_from_matrix",
]

#: 4 đặc trưng lịch sử ĐÃ QUA CỔNG kiểm định trên dữ liệu thật (10 ngày, 180.606 dòng).
HISTORY_V3_FEATURES: List[str] = [
    "new_source_count_7d",
    "new_host_count_7d",
    "days_since_last_activity",
    "volume_robust_z_7d",
]

#: 4 đặc trưng lịch sử đã CÀI + ĐO rồi BÁC BỎ (kèm ρ đo được) — xem `removed:` trong schema:
#:   * ``new_source_count``      ρ = 0,9999 với ``new_source_count_7d`` (trùng hoàn toàn trên LANL:
#:                               lần xuất hiện ĐẦU TIÊN của một Source gần như luôn nằm ngoài cửa sổ 7 ngày)
#:   * ``new_host_count``        ρ = 0,9999 với ``new_host_count_7d`` (cùng cơ chế)
#:   * ``source_recency``        ρ = 0,9238 với ``days_since_last_activity``
#:   * ``source_host_pair_novelty`` ρ = 0,9052 với ``new_host_count`` (một LogHost mới ⇒ một cặp mới)
#: Ở bản 28 biến, 4 cặp này đẩy VIF lên 6.306–8.197 (ngưỡng nghiêm trọng là 10).
HISTORY_V3_REJECTED: List[str] = [
    "new_source_count",
    "new_host_count",
    "source_recency",
    "source_host_pair_novelty",
]

HISTORY_KEYS: List[str] = ["DomainName", "UserName"]

#: Cửa sổ "gần đây" tính bằng NGÀY LỊCH (khớp tên biến ``*_7d``).
HISTORY_WINDOW_DAYS: int = 7


def _entity_novelty(
    entities: pl.DataFrame,
    keys: List[str],
    kind: str,
    name: str,
    window: int,
    day_col: str,
) -> pl.DataFrame:
    """
    Số thực thể mới của một chiều phân loại (``Source`` hoặc ``LogHost``) theo từng ngày.

    ``entities`` = khung hiện diện dạng dài ``(keys, day, kind, entity)`` (do extractor dựng).

    ``prev_day`` = ngày gần nhất **trước t** mà (tài khoản, thực thể) xuất hiện ⇒ hai biến:
    ``new_<name>_count`` (chưa từng thấy trong toàn bộ lịch sử) và
    ``new_<name>_count_{window}d`` (không thấy trong ``window`` ngày lịch gần nhất).
    ``_min_<name>_gap`` = khoảng cách ngày tới lần xuất hiện gần nhất (NULL nếu mọi thực thể đều mới).
    """
    suffix = name
    presences = (
        entities.filter(pl.col("kind") == kind)
        .select(keys + [day_col, pl.col("entity")])
        .unique()
        .sort(keys + ["entity", day_col])
        .with_columns(
            pl.col(day_col).shift(1).over(keys + ["entity"]).alias("_prev_day")
        )
    )
    return presences.group_by(keys + [day_col]).agg([
        pl.col("_prev_day").is_null().sum().cast(pl.Int32).alias(f"new_{suffix}_count"),
        (
            pl.col("_prev_day").is_null()
            | (pl.col("_prev_day") < pl.col(day_col) - window)
        ).sum().cast(pl.Int32).alias(f"new_{suffix}_count_{window}d"),
    ])


def _pair_novelty_REMOVED_DO_NOT_USE(*args, **kwargs):  # pragma: no cover
    """ĐÃ BÁC BỎ — xem ``HISTORY_V3_REJECTED``.

    ``source_host_pair_novelty`` (tỷ lệ cặp (Source, LogHost) chưa từng thấy) đo được
    ρ = 0,9052 với ``new_host_count`` và 0,8004 với ``new_source_count`` trên dữ liệu thật:
    một LogHost mới gần như luôn kéo theo một cặp mới ⇒ hai đặc trưng cùng một trục.
    Giữ hàm này rỗng để không ai "vô tình" dùng lại mà quên lý do.
    """
    raise NotImplementedError("source_host_pair_novelty đã bị bác bỏ (ρ = 0,9052 với new_host_count)")


def _days_since_last_activity(df: pl.DataFrame, keys: List[str], day_col: str) -> pl.DataFrame:
    """Số ngày kể từ phiên hoạt động trước đó của tài khoản (NULL ở dòng hoạt động đầu tiên)."""
    activity = df.select(keys + [day_col]).unique().sort(keys + [day_col])
    return activity.with_columns(
        (pl.col(day_col) - pl.col(day_col).shift(1).over(keys))
        .cast(pl.Int32)
        .alias("days_since_last_activity")
    )



def _volume_robust_z(
    df: pl.DataFrame,
    keys: List[str],
    window: int,
    day_col: str,
) -> pl.DataFrame:
    """
    Độ lệch chuẩn hoá BỀN VỮNG của khối lượng so với ``window`` ngày lịch trước đó.

    Cách làm: dựng **lưới dày nội bộ** (mọi tài khoản × mọi ngày trong dải quan sát, ngày trống
    = ``log1p(0) = 0``), tính median/IQR/std cửa sổ ``[t−window, t−1]``, rồi **chỉ giữ lại các
    dòng ứng với ngày thật có hoạt động** ⇒ ma trận đầu ra không đổi số dòng.

    Thang bền vững dùng chuỗi fallback **giống ``ZScoreBaseline`` của repo**:
    ``IQR/1.349`` → nếu 0 thì ``std`` → nếu vẫn 0 thì ``z = 0`` (baseline không có độ tán xạ
    ⇒ không có gì để gọi là "lệch"). NULL khi chưa đủ ``window`` ngày (warm-up).
    """
    volume = df.select(
        keys + [day_col, pl.col("total_logons").cast(pl.Float64).log1p().alias("_v")]
    )
    day_min, day_max = int(df[day_col].min()), int(df[day_col].max())

    # 1. Lưới dày nội bộ (chỉ để tính baseline, KHÔNG ghi ra ma trận)
    grid = (
        df.select(keys).unique()
        .with_columns(pl.int_ranges(day_min, day_max + 1).alias(day_col))
        .explode(day_col)
        .sort(keys + [day_col])
    )
    dense = (
        grid.join(volume, on=keys + [day_col], how="left")
        .with_columns(pl.col("_v").fill_null(0.0))
        .sort(keys + [day_col])
    )

    # 2. Cửa sổ trượt trên các ngày TRƯỚC t (shift 1 ⇒ không bao giờ chạm chính ngày t)
    prev = pl.col("_v").shift(1).over(keys)
    dense = dense.with_columns([
        prev.rolling_median(window_size=window, min_samples=window).over(keys).alias("_med"),
        prev.rolling_quantile(0.25, interpolation="linear", window_size=window, min_samples=window)
        .over(keys)
        .alias("_q25"),
        prev.rolling_quantile(0.75, interpolation="linear", window_size=window, min_samples=window)
        .over(keys)
        .alias("_q75"),
        prev.rolling_std(window_size=window, min_samples=window).over(keys).alias("_std"),
    ])

    centered = pl.col("_v") - pl.col("_med")
    # Hai chốt bảo vệ đã phải thêm SAU KHI ĐO trên dữ liệu thật:
    #   1. Dung sai scale (1e-9): nếu không, sai số dấu phẩy động khiến (v−med)/std bùng nổ.
    #   2. Winsorize ±10: baseline cực ổn định làm IQR ≈ 0 nên z hợp lệ về toán nhưng cực lớn
    #      (ví dụ thật: User718489 giữ 5.966–6.058 sự kiện/ngày suốt 7 ngày rồi tụt còn 472
    #      ⇒ IQR ≈ 0,005 ⇒ z = −976). Cắt ở ±10 giữ nguyên THỨ TỰ bất thường nhưng không để
    #      một đặc trưng lấn át khoảng cách của IForest/LOF/OCSVM.
    min_scale = 1e-9
    z_expr = (
        pl.when(pl.col("_med").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(
            pl.when((pl.col("_q75") - pl.col("_q25")) / 1.349 > min_scale)
            .then(centered / ((pl.col("_q75") - pl.col("_q25")) / 1.349))
            .when(pl.col("_std").fill_null(0.0) > min_scale)
            .then(centered / pl.col("_std"))
            .otherwise(0.0)
            .clip(-10.0, 10.0)
        )
        .alias("volume_robust_z_7d")
    )
    z = dense.with_columns(z_expr).select(keys + [day_col, "volume_robust_z_7d"])

    # 3. Chỉ trả về các (tài khoản, ngày) thật sự có trong ma trận
    return z.join(df.select(keys + [day_col]).unique(), on=keys + [day_col], how="semi")


def entities_from_matrix(df: pl.DataFrame, keys: List[str], day_col: str = "day") -> pl.DataFrame:
    """Dựng khung hiện diện ``(keys, day, kind, entity)`` từ một ma trận CÓ cột ``Source``/``LogHost``.

    Chỉ dùng cho kiểm thử / tương thích: trong pipeline thật, extractor đã dựng khung này
    (``_presence_frames``) vì ma trận thô không giữ danh tính Source/LogHost.
    """
    valid_source = pl.col("Source").is_not_null() & (pl.col("Source").str.strip_chars() != "")
    valid_host = pl.col("LogHost").is_not_null() & (pl.col("LogHost").str.strip_chars() != "")
    return pl.concat([
        df.filter(valid_source)
        .select(keys + [day_col, pl.col("Source").str.strip_chars().alias("entity")])
        .unique()
        .with_columns(pl.lit("Source").alias("kind")),
        df.filter(valid_host)
        .select(keys + [day_col, pl.col("LogHost").str.strip_chars().alias("entity")])
        .unique()
        .with_columns(pl.lit("LogHost").alias("kind")),
    ])


def add_history_features(
    df: pl.DataFrame,
    entities: Optional[pl.DataFrame] = None,
    keys: Optional[List[str]] = None,
    window: int = HISTORY_WINDOW_DAYS,
    day_col: str = "day",
) -> pl.DataFrame:
    """
    Bổ sung **4 đặc trưng lịch sử** (Tier A đã qua cổng, schema v3.0) vào ma trận Tài khoản × Ngày.

    Đầu vào: ma trận **thô** đã ghép toàn bộ dải ngày (cần ``day`` và ``total_logons``), kèm
    khung vật liệu lịch sử ``entities`` = ``(DomainName, UserName, day, kind, entity)``
    (hiện diện ``Source``/``LogHost``) do extractor dựng.

    Nếu không truyền ``entities``, hàm tự suy ra từ cột ``Source``/``LogHost`` của ``df``
    (chỉ hợp lệ khi ma trận còn giữ hai cột này — dùng cho kiểm thử).

    Đầu ra: chính ma trận đó + 4 cột, **không đổi số dòng**.

    An toàn rò rỉ: mọi phép tính dùng ``shift``/``min``/``max`` trên các ngày **trước t**; thêm
    ngày tương lai vào đầu vào **không** làm đổi giá trị của các dòng cũ (kiểm bằng test
    ``test_history_features_are_causal``).
    """
    keys = list(keys or HISTORY_KEYS)
    if day_col not in df.columns or "total_logons" not in df.columns:
        raise ValueError(
            f"add_history_features: ma trận thiếu cột "
            f"{[c for c in [day_col, 'total_logons'] if c not in df.columns]}"
        )
    if entities is None:
        if "Source" not in df.columns or "LogHost" not in df.columns:
            raise ValueError(
                "add_history_features: cần truyền 'entities' (từ extractor) hoặc ma trận "
                "phải còn cột 'Source'/'LogHost'."
            )
        entities = entities_from_matrix(df, keys, day_col)

    if df.height == 0:
        schema: Dict[str, Any] = {c: pl.Float64 for c in HISTORY_V3_FEATURES}
        schema.update({
            "new_source_count_7d": pl.Int32, "new_host_count_7d": pl.Int32,
            "days_since_last_activity": pl.Int32,
        })
        return df.hstack(pl.DataFrame(schema=schema).select(HISTORY_V3_FEATURES))

    source = _entity_novelty(entities, keys, "Source", "source", window, day_col).select(
        keys + [day_col, f"new_source_count_{window}d"]
    )
    host = _entity_novelty(entities, keys, "LogHost", "host", window, day_col).select(
        keys + [day_col, f"new_host_count_{window}d"]
    )
    activity = _days_since_last_activity(df, keys, day_col)
    volume_z = _volume_robust_z(df, keys, window, day_col)

    join_on = keys + [day_col]
    out = (
        df.join(source, on=join_on, how="left")
        .join(host, on=join_on, how="left")
        .join(activity, on=join_on, how="left")
        .join(volume_z, on=join_on, how="left")
    )
    # Ngày không có Source/LogHost hợp lệ ⇒ đếm = 0
    return out.with_columns([
        pl.col("new_source_count_7d").fill_null(0),
        pl.col("new_host_count_7d").fill_null(0),
    ])
