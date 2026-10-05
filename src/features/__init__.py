"""
Features Package.
Quản lý việc trích xuất và chuẩn hóa ma trận đặc trưng hành vi UEBA.
"""

from src.features.schema import FeatureSchema
from src.features.extractor import (
    build_account_day_matrix,
    extract_features_single_day,
    available_days,
    intraday_behavior_features,
    INTRADAY_V3_FEATURES,
    INTRADAY_V3_REJECTED,
    MATRIX_ORDERED_COLS,
    RAW_ORDERED_COLS,
    TEMPLATE_V4_FEATURES,
)
from src.features.history import (
    HISTORY_V3_FEATURES,
    add_history_features,
)
from src.features.preprocessor import (
    normalize_features,
    prepare_processed_dataset,
)

__all__ = [
    "FeatureSchema",
    "build_account_day_matrix",
    "extract_features_single_day",
    "available_days",
    "intraday_behavior_features",
    "INTRADAY_V3_FEATURES",
    "INTRADAY_V3_REJECTED",
    "MATRIX_ORDERED_COLS",
    "HISTORY_V3_FEATURES",
    "add_history_features",
    "RAW_ORDERED_COLS",
    "TEMPLATE_V4_FEATURES",
    "normalize_features",
    "prepare_processed_dataset",
]
