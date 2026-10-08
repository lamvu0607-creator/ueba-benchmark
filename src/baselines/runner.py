"""
Chạy 3 baseline (5 cột điểm) end-to-end: ``python main.py --stage baselines``.

Luồng (cùng quy ước với benchmark ML):
  ma trận processed -> chia theo thời gian (train = day <= split_day) -> thêm 6 thống kê luật ->
  với mỗi phân khúc (Machine / User): fit TRÊN TRAIN, chấm train + đánh giá, ngưỡng = phân vị điểm train
  -> ``baseline_scores.parquet`` (+ ``thresholds.json``); có nhãn thì thêm ``baseline_metrics.csv``
  và ``baseline_pr_auc_by_scenario.csv``.

Ranh giới chống rò rỉ: mọi ``fit`` nhận ``train`` (đã lọc day <= split_day); nhãn chỉ được nạp SAU khi
mọi điểm và ngưỡng đã tính xong, và chỉ đi vào ``src.evaluation.labeled_eval``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import yaml

from src.baselines._common import ROW_KEYS, logger
from src.baselines.random_baseline import STREAM_EVAL, STREAM_TRAIN, RandomBaseline, expected_random_pr_auc
from src.baselines.rule_baseline import EcdfRuleBaseline, attach_rule_statistics
from src.baselines.rule_stats import compute_event_rule_stats
from src.baselines.thresholding import fit_budget_threshold
from src.baselines.zscore_baseline import AccountRobustZScore, GlobalRobustZScore

__all__ = ["load_baselines_config", "load_or_compute_event_stats", "fit_and_score_segment", "run_baselines"]

ALL_SEGMENT = "all"


def load_baselines_config(path: Path | str = "configs/baselines.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_or_compute_event_stats(
    rules_cfg: Dict[str, Any], days: List[int], train_last_day: int, cache_path: Optional[Path]
) -> tuple[pl.DataFrame, Dict[str, Any]]:
    """Đọc cache R2/R4/R6 nếu có (chỉ khi KHÔNG dùng log đã tiêm), ngược lại tính từ log và ghi cache."""
    meta_path = cache_path.with_suffix(".json") if cache_path else None
    if cache_path and cache_path.is_file() and meta_path and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("train_last_day") == int(train_last_day) and meta.get("injected_events_dir") == rules_cfg.get("injected_events_dir"):
            logger.info("[baselines] dùng cache thống kê log '%s'.", cache_path)
            return pl.read_parquet(cache_path), meta["lockout"]
    stats, lockout = compute_event_rule_stats(
        days, rules_cfg.get("interim_dir", "data/interim"), train_last_day, rules_cfg.get("injected_events_dir")
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stats.write_parquet(cache_path)
        meta_path.write_text(json.dumps({
            "train_last_day": int(train_last_day),
            "injected_events_dir": rules_cfg.get("injected_events_dir"),
            "lockout": lockout,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats, lockout


def fit_and_score_segment(
    train: pl.DataFrame,
    evaluation: pl.DataFrame,
    features: List[str],
    cfg: Dict[str, Any],
    external_thresholds: Optional[Dict[str, float]] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Fit mọi baseline trên ``train`` (không nhãn) rồi chấm cả hai tập.

    Trả ``(train_scores, eval_scores, meta)`` với khoá là tên phương pháp.
    """
    zcfg = cfg.get("zscore", {}) or {}
    q = float((zcfg.get("mad_floor") or {}).get("quantile", 0.01))
    min_days = int((zcfg.get("per_account") or {}).get("min_train_days", 7))
    tol = float(zcfg.get("zero_tolerance", 1e-9))
    seed = int(((cfg.get("random") or {}).get("seeds") or [42])[0])

    random_model = RandomBaseline(seed).fit(train.height)
    models: Dict[str, Any] = {
        "zscore_global": GlobalRobustZScore(features, q, tol),
        "zscore_account": AccountRobustZScore(features, q, min_days, zero_tolerance=tol),
        "rule_ecdf": EcdfRuleBaseline(),
    }
    if external_thresholds:
        models["rule_external"] = EcdfRuleBaseline(external_thresholds)

    train_scores = {"random": random_model.score(train.height, STREAM_TRAIN)}
    eval_scores = {"random": random_model.score(evaluation.height, STREAM_EVAL)}
    meta: Dict[str, Any] = {"random": random_model.get_metadata()}
    for name, model in models.items():
        model.fit(train)
        train_scores[name] = model.score(train)
        eval_scores[name] = model.score(evaluation)
        meta[name] = model.get_metadata()
    return train_scores, eval_scores, meta


def run_baselines(
    system_config_path: Path | str = "configs/system_config.yaml",
    baselines_config_path: Path | str = "configs/baselines.yaml",
    params_path: Path | str = "configs/model_params.yaml",
    data_path: Path | str = "data/processed/feature_matrix_processed.parquet",
    labels_path: Optional[Path | str] = None,
    injected_events_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    use_segments: Optional[bool] = None,
) -> Dict[str, Any]:
    from src.evaluation.split import resolve_split_day, time_split
    from src.features.schema import FeatureSchema
    from src.models.benchmark import SegmentConfig, load_feature_matrix, split_segments
    from src.models.registry import load_params

    with open(system_config_path, "r", encoding="utf-8") as f:
        sys_cfg = yaml.safe_load(f) or {}
    cfg = load_baselines_config(baselines_config_path)
    common = cfg.get("common", {}) or {}
    rules_cfg = dict(cfg.get("rules", {}) or {})
    if injected_events_dir is not None:
        rules_cfg["injected_events_dir"] = str(injected_events_dir)
    eval_cfg = sys_cfg.get("evaluation", {}) or {}
    budget_ratio = float(common.get("budget_ratio", 0.01))
    out_dir = Path(output_dir or common.get("output_dir", "experiments/results/baselines"))
    out_dir.mkdir(parents=True, exist_ok=True)

    features = list(FeatureSchema().core_features)
    df = load_feature_matrix(data_path)
    split_day = eval_cfg.get("split_day")
    if split_day is None:
        split_day = resolve_split_day(df, test_split_ratio=float(eval_cfg.get("test_split_ratio", 0.30)))
    days = sorted(int(d) for d in df["day"].unique().to_list())

    # Log đã tiêm thay đổi R2/R4/R6 của ngày test -> cache riêng trong thư mục kết quả của lần chạy.
    if rules_cfg.get("injected_events_dir"):
        cache_path: Optional[Path] = out_dir / "rule_event_stats.parquet"
    else:
        cache_path = Path(rules_cfg["event_stats_cache"]) if rules_cfg.get("event_stats_cache") else None
    event_stats, lockout = load_or_compute_event_stats(rules_cfg, days, int(split_day), cache_path)
    df = attach_rule_statistics(df, event_stats)

    train_all, eval_all, split_info = time_split(df, split_day=int(split_day))

    ext_cfg = rules_cfg.get("external", {}) or {}
    external = None
    if ext_cfg.get("enabled", False):
        l_cfg = ext_cfg.get("r1_lockout_threshold", "auto")
        external = {
            "r1_failure_count": float(lockout["lockout_threshold"] if str(l_cfg) == "auto" else l_cfg),
            "r2_spray_accounts": float(ext_cfg.get("r2_distinct_accounts", 30)),
        }

    seg_cfg = SegmentConfig.from_params(load_params(params_path)) if (
        common.get("use_segments", True) if use_segments is None else use_segments
    ) else None
    if seg_cfg:
        train_parts, _ = split_segments(train_all, seg_cfg, "train")
        eval_parts, _ = split_segments(eval_all, seg_cfg, "đánh giá")
    else:
        train_parts, eval_parts = {ALL_SEGMENT: train_all}, {ALL_SEGMENT: eval_all}

    score_frames: List[pl.DataFrame] = []
    thresholds: Dict[str, Dict[str, Any]] = {}
    metadata: Dict[str, Any] = {}
    eval_scores_by_seg: Dict[str, Dict[str, np.ndarray]] = {}
    for segment, train in train_parts.items():
        evaluation = eval_parts[segment]
        tr_scores, ev_scores, meta = fit_and_score_segment(train, evaluation, features, cfg, external)
        metadata[segment] = meta
        eval_scores_by_seg[segment] = ev_scores
        thresholds[segment] = {m: fit_budget_threshold(s, budget_ratio).to_dict() for m, s in tr_scores.items()}
        for split_name, part, scores in (("train", train, tr_scores), ("eval", evaluation, ev_scores)):
            score_frames.append(
                part.select(ROW_KEYS).with_columns(pl.lit(segment).alias("segment"), pl.lit(split_name).alias("split"))
                .with_columns([pl.Series(m, s) for m, s in scores.items()])
            )

    scores_frame = pl.concat(score_frames, how="vertical")
    scores_path = out_dir / "baseline_scores.parquet"
    scores_frame.write_parquet(scores_path)
    (out_dir / "thresholds.json").write_text(json.dumps({
        "budget_ratio": budget_ratio,
        "split": split_info.to_dict(),
        "lockout_estimate": lockout,
        "external_thresholds": external,
        "thresholds": thresholds,
        "models": metadata,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("[baselines] đã ghi '%s' (%s dòng) và thresholds.json.", scores_path, f"{scores_frame.height:,}")

    result: Dict[str, Any] = {"scores_path": str(scores_path), "thresholds": thresholds, "lockout": lockout}
    if labels_path:
        result.update(_evaluate(labels_path, eval_parts, eval_scores_by_seg, thresholds, cfg, eval_cfg, out_dir))
    return result


def _evaluate(
    labels_path: Path | str,
    eval_parts: Dict[str, pl.DataFrame],
    eval_scores_by_seg: Dict[str, Dict[str, np.ndarray]],
    thresholds: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """Chỉ chạy SAU khi điểm + ngưỡng đã cố định: nạp nhãn và tính bộ chỉ số chung."""
    from sklearn.metrics import average_precision_score

    from src.evaluation.labeled_eval import align_eval_labels, evaluate_detector, load_eval_labels

    labels = load_eval_labels(labels_path)
    daily_budgets = eval_cfg.get("daily_budgets") or [10, 50, 100]
    seeds = [int(s) for s in ((cfg.get("random") or {}).get("seeds") or [42])]
    rows, subset_rows = [], []
    for segment, evaluation in eval_parts.items():
        aligned = align_eval_labels(evaluation, labels)
        keep = ~aligned["eval_exclude"].to_numpy().astype(bool)
        lab = aligned["is_anomaly"].to_numpy()[keep]
        for method, scores in eval_scores_by_seg[segment].items():
            res = evaluate_detector(scores, thresholds[segment][method]["threshold"], aligned, daily_budgets)
            rows.append({"segment": segment, "method": method, **res["summary"]})
            for subset, cell in res["subsets"].items():
                subset_rows.append({"segment": segment, "method": method, "subset": subset, **cell})
        if lab.any() and not lab.all():
            ap = [average_precision_score(lab, RandomBaseline(s).score(evaluation.height, STREAM_EVAL)[keep]) for s in seeds]
            logger.info(
                "[baselines][%s] PR-AUC ngẫu nhiên: kỳ vọng %.5f | thực nghiệm %d seed %.5f ± %.5f",
                segment, expected_random_pr_auc(lab), len(seeds), float(np.mean(ap)), float(np.std(ap)),
            )
            rows.append({"segment": segment, "method": "random_multi_seed_mean", "pr_auc": float(np.mean(ap)),
                         "random_pr_auc": expected_random_pr_auc(lab), "n_eval": float(lab.size)})
    metrics = pl.DataFrame(rows, infer_schema_length=None)
    subsets = pl.DataFrame(subset_rows, infer_schema_length=None)
    metrics.write_csv(out_dir / "baseline_metrics.csv")
    subsets.write_csv(out_dir / "baseline_pr_auc_by_scenario.csv")
    return {"metrics": metrics, "subsets": subsets}
