"""
Module Registry - Danh mục 6 mô hình canonical, alias tương thích ngược, và factory.

Vì sao cần registry:
  * ``experiments/logs/experiment_log.csv`` lịch sử ghi tên mô hình theo kiểu CamelCase
    (``IsolationForest``...) lẫn snake_case -> cần alias để đọc lại log cũ mà không phải sửa tay;
  * mọi nơi tạo mô hình (CLI, benchmark, test) phải đi qua **một** factory để tham số YAML
    được kiểm tra whitelist, ``contamination``/``random_state`` lấy từ khối ``defaults``,
    và baseline luật luôn nhận đúng tên đặc trưng.

Cấu trúc ``configs/model_params.yaml`` được registry hiểu như sau::

    defaults:        { contamination, random_state }          # dùng chung cho mọi mô hình
    training:        { max_train_samples }                    # ghi đè ngân sách fit (null = mặc định lớp)
    preprocessing:   { imputer, scaler, feature_set }         # cho AnomalyPipeline
    <model_name>:    { ...tham số gốc của sklearn/luật... }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

import yaml

from src.models.base import BaseAnomalyModel
from src.models.baselines import RandomBaseline, RuleThresholdBaseline, ZScoreBaseline
from src.models.detectors import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
)
from src.models.pipeline import AnomalyPipeline

logger = logging.getLogger("ueba_benchmark.models.registry")

__all__ = [
    "DEFAULT_MODEL_NAMES",
    "MODEL_CLASSES",
    "MODEL_REGISTRY",
    "available_models",
    "create_model",
    "create_pipeline",
    "load_params",
    "resolve_model",
]

#: 6 mô hình của benchmark, đúng thứ tự xuất hiện trong bảng xếp hạng.
#: ``RandomBaseline`` đặt CUỐI vì nó là **mốc dưới** (đoán mò) để đối chiếu mọi chỉ số.
MODEL_CLASSES = (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
    ZScoreBaseline,
    RuleThresholdBaseline,
    RandomBaseline,
)

#: Tên canonical dùng cho CLI ``--models`` và tên cột trong artifact.
DEFAULT_MODEL_NAMES: List[str] = [cls.name for cls in MODEL_CLASSES]


def _build_registry() -> Dict[str, Type[BaseAnomalyModel]]:
    registry: Dict[str, Type[BaseAnomalyModel]] = {}
    for cls in MODEL_CLASSES:
        for key in (cls.name, *cls.aliases, cls.__name__):
            normalized = str(key).lower()
            existing = registry.get(normalized)
            if existing is not None and existing is not cls:
                raise ValueError(f"Xung đột tên mô hình '{key}': {existing.__name__} vs {cls.__name__}.")
            registry[normalized] = cls
    return registry


#: Bảng tra ``tên/alias (lowercase) -> lớp mô hình``.
MODEL_REGISTRY: Dict[str, Type[BaseAnomalyModel]] = _build_registry()


def available_models() -> List[str]:
    """Danh sách 5 tên canonical của benchmark."""
    return list(DEFAULT_MODEL_NAMES)


def resolve_model(name: str) -> Type[BaseAnomalyModel]:
    """Đổi tên/alias thành lớp mô hình; tên lạ -> ``KeyError`` kèm danh sách hợp lệ."""
    key = str(name).strip().lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Mô hình không tồn tại: {name!r}. Tên hợp lệ: {sorted(set(MODEL_REGISTRY))}."
        )
    return MODEL_REGISTRY[key]


def load_params(params_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Đọc ``configs/model_params.yaml`` (trả {} nếu file không tồn tại để dùng mặc định)."""
    if not params_path:
        return {}
    path = Path(params_path)
    if not path.is_file():
        logger.warning("Không tìm thấy file tham số '%s', dùng mặc định của từng lớp mô hình.", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_model(
    name: str,
    params: Optional[Dict[str, Any]] = None,
    contamination: Optional[float] = None,
    random_state: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> BaseAnomalyModel:
    """
    Tạo mô hình từ tên + cấu hình YAML (đã tách khối ``defaults``/``training``).

    Tham số truyền trực tiếp (khác None) luôn thắng cấu hình -> CLI có thể ghi đè khi tái lập thí nghiệm.
    """
    params = params or {}
    cls = resolve_model(name)
    defaults = params.get("defaults") or {}
    model_kwargs = dict(params.get(cls.name) or {})

    # Khối YAML của từng mô hình có thể chứa cả tham số giao diện (contamination/random_state/
    # max_train_samples) — tách ra để tránh truyền trùng, và để tham số tường minh luôn thắng.
    config_contamination = model_kwargs.pop("contamination", None)
    config_random_state = model_kwargs.pop("random_state", None)
    config_max_train_samples = model_kwargs.pop("max_train_samples", None)
    # Baseline luật biên dịch luật theo TÊN đặc trưng -> truyền riêng qua tham số ``feature_names``.
    model_kwargs.pop("feature_names", None)

    if contamination is None:
        contamination = config_contamination if config_contamination is not None else defaults.get("contamination", 0.05)
    if random_state is None:
        random_state = config_random_state if config_random_state is not None else defaults.get("random_state", 42)
    if max_train_samples is None:
        max_train_samples = (
            config_max_train_samples
            if config_max_train_samples is not None
            else (params.get("training") or {}).get("max_train_samples")
        )

    if cls is RuleThresholdBaseline and feature_names is not None:
        model_kwargs["feature_names"] = list(feature_names)

    model = cls(
        contamination=float(contamination),
        random_state=int(random_state),
        max_train_samples=max_train_samples,
        **model_kwargs,
    )
    logger.debug("Đã tạo mô hình '%s' (%s).", cls.name, type(model).__name__)
    return model


def create_pipeline(
    name: str,
    params: Optional[Dict[str, Any]] = None,
    feature_names: Optional[Sequence[str]] = None,
    contamination: Optional[float] = None,
    random_state: int = 42,
    schema_path: Optional[Path | str] = None,
    max_train_samples: Optional[int] = None,
) -> AnomalyPipeline:
    """
    Tạo ``AnomalyPipeline`` hoàn chỉnh (mô hình + imputer + scaler) từ cùng một cấu hình YAML.

    ``max_train_samples`` (tuỳ chọn) ghi đè ngân sách lấy mẫu con khi fit — dùng cho thí nghiệm
    so sánh ngân sách fit (ví dụ LOF/OCSVM 50.000 dòng trong khi Isolation Forest fit toàn bộ).
    """
    params = params or {}
    preprocessing = params.get("preprocessing") or {}

    model = create_model(
        name,
        params=params,
        contamination=contamination,
        random_state=random_state,
        max_train_samples=max_train_samples,
        feature_names=feature_names,
    )
    return AnomalyPipeline(
        model,
        scaler=preprocessing.get("scaler", "robust"),
        imputer=preprocessing.get("imputer", "median"),
        feature_names=feature_names,
        random_state=random_state,
        schema_path=schema_path,
    )
