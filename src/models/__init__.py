"""
Module Models - Giao diện chung + các mô hình phát hiện dị biệt (UEBA).

Điểm vào công khai:
  * ``BaseAnomalyModel`` / ``AnomalyPipeline`` — API chung;
  * ``IsolationForestDetector``, ``LocalOutlierFactorDetector``, ``OneClassSVMDetector`` — 3 thuật toán
    qua giao diện PyOD (IForest, LOF trên HNSW, OCSVM xấp xỉ Nystroem + SGD);
  * ``ZScoreBaseline``, ``RuleThresholdBaseline``, ``RandomBaseline`` — 3 baseline label-free
    (baseline ngẫu nhiên là **mốc dưới**: ROC-AUC kỳ vọng 0,5);
  * ``resolve_model`` / ``create_pipeline`` / ``available_models`` — registry & factory;
  * ``run_model_benchmark`` — chạy benchmark đầy đủ và xuất artifact.
"""

from src.models.base import BaseAnomalyModel
from src.models.baselines import RuleThresholdBaseline, ZScoreBaseline
from src.models.benchmark import run_model_benchmark
from src.models.detectors import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
)
from src.models.pipeline import AnomalyPipeline
from src.models.registry import (
    DEFAULT_MODEL_NAMES,
    create_model,
    create_pipeline,
    available_models,
    load_params,
    resolve_model,
)

__all__ = [
    "BaseAnomalyModel",
    "AnomalyPipeline",
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "OneClassSVMDetector",
    "ZScoreBaseline",
    "RuleThresholdBaseline",
    "DEFAULT_MODEL_NAMES",
    "available_models",
    "resolve_model",
    "create_model",
    "create_pipeline",
    "load_params",
    "run_model_benchmark",
]
