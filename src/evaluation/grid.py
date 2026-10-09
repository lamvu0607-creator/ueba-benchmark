"""
Lưới thí nghiệm tuần 4: 3 mô hình ML × nhiều cấu hình × nhiều seed trên các run tiêm (configs/benchmark_grid.yaml).

Một "seed" là một cặp (run tiêm, random_state mô hình) -> độ lệch chuẩn gồm cả biến động dữ liệu tiêm lẫn mô hình.
Mỗi lần fit: imputer/scaler/mô hình chỉ học trên TRAIN (day ≤ split_day) của phân khúc; chấm đúng các ngày
đánh giá đóng băng trong ``run_config.json`` của run; ngưỡng cảnh báo = phân vị (1 − contamination) của điểm
train. Chỉ số dùng chung định nghĩa với benchmark (``src/evaluation/metrics.py``).

Đầu ra (``output_dir``):
  * ``grid_runs.csv`` / ``grid_summary.csv``             — mỗi (mô hình, cấu hình, seed) / trung bình ± std,
  * ``grid_scenarios.csv``                               — PR-AUC & ROC-AUC theo kịch bản, mỗi seed,
  * ``contamination_runs.csv`` / ``contamination_summary.csv`` — độ nhạy contamination của cấu hình tốt nhất,
  * ``pr_curves.csv``                                    — đường PR của seed đầu (cấu hình tốt nhất + baseline),
  * ``figures/*.png|pdf``, ``grid_manifest.json``.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.metrics import precision_recall_curve, roc_auc_score

from src.evaluation.labeled_eval import align_eval_labels, load_eval_labels
from src.evaluation.metrics import labeled_metrics, operating_point
from src.models.pipeline import AnomalyPipeline
from src.models.registry import create_model, load_params

logger = logging.getLogger("ueba_benchmark.grid")

__all__ = ["load_grid_config", "load_run_data", "fit_and_score", "run_grid", "summarize", "select_best"]

METRICS = [
    "pr_auc", "roc_auc", "precision_at_10", "precision_at_50", "precision_at_100", "recall_at_budget",
    "recall_at_10_per_day", "recall_at_50_per_day", "recall_at_100_per_day",
    "precision_at_10_per_day", "precision_at_50_per_day", "precision_at_100_per_day",
    "alert_rate", "precision", "recall", "f1", "fit_seconds", "score_seconds",
]


def load_grid_config(path: Path | str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    runs, seeds = cfg.get("runs") or [], cfg.get("model_seeds") or []
    if not runs or len(runs) != len(seeds):
        raise ValueError(f"'runs' ({len(runs)}) và 'model_seeds' ({len(seeds)}) phải cùng độ dài và khác rỗng.")
    ids = [c["id"] for confs in (cfg.get("models") or {}).values() for c in confs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"id cấu hình bị trùng: {sorted(i for i in ids if ids.count(i) > 1)}.")
    return cfg


def run_split_day(run_dir: Path | str) -> int:
    """``split_day`` đóng băng trong ``run_config.json`` của run (dev 36–42 dùng 35, test 43–60 dùng 42)."""
    return int(json.loads((Path(run_dir) / "run_config.json").read_text(encoding="utf-8"))["split_day"])


def load_run_data(run_dir: Path | str, segment: Optional[str], split_day: Optional[int] = None,
                  segment_col: str = "entity_type") -> Dict[str, Any]:
    """
    Ma trận + nhãn của MỘT run: train (day ≤ split_day) và eval (đúng ``eval_days`` của run).

    ``split_day`` = None -> dùng mốc của run; có giá trị -> phải khớp mốc của run (chống fit nhầm ngày tiêm).
    """
    root = Path(run_dir)
    meta = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    eval_days = sorted(int(d) for d in meta["eval_days"])
    if split_day is None:
        split_day = int(meta["split_day"])
    if int(meta["split_day"]) != int(split_day):
        raise ValueError(f"{root}: split_day của run ({meta['split_day']}) khác cấu hình ({split_day}).")
    if min(eval_days) <= int(split_day):
        raise ValueError(f"{root}: ngày đánh giá {eval_days} chạm train (≤ {split_day}).")
    df = pl.read_parquet(root / "processed" / "feature_matrix_processed.parquet")
    if segment:
        df = df.filter(pl.col(segment_col) == segment)
    train = df.filter(pl.col("day") <= split_day)
    ev = df.filter(pl.col("day").is_in(eval_days))
    aligned = align_eval_labels(ev.select("DomainName", "UserName", "day"), load_eval_labels(root / "labels.parquet"))
    keep = ~aligned["eval_exclude"].to_numpy().astype(bool)
    ev, aligned = ev.filter(pl.Series(keep)), aligned.filter(pl.Series(keep))
    return {
        "run_id": root.name, "root": root, "eval_days": eval_days, "train": train.to_pandas(),
        "eval": ev.to_pandas(), "y": aligned["is_anomaly"].to_numpy().astype(np.int8),
        "days": aligned["day"].to_numpy(), "scenario": aligned["scenario"].to_numpy(),
        "seed": meta.get("seed"), "n_injected": (meta.get("target_rate") or {}).get("n_injected"),
    }


def _params_with(base: Dict[str, Any], model: str, override: Dict[str, Any]) -> Dict[str, Any]:
    params = copy.deepcopy(base)
    params[model] = {**(params.get(model) or {}), **(override or {})}
    return params


def fit_and_score(data: Dict[str, Any], model: str, params: Dict[str, Any], scaler: Optional[str],
                  random_state: int, contamination: float) -> Dict[str, Any]:
    """Fit pipeline trên train của run rồi chấm eval -> điểm, cờ (ngưỡng train) và thời gian."""
    if model == "one_class_svm":  # nu là ngân sách cảnh báo của OCSVM -> đồng bộ với contamination
        params = _params_with(params, model, {"nu": float(contamination)})
    est = create_model(model, params=params, contamination=contamination, random_state=random_state)
    pipe = AnomalyPipeline(est, scaler=scaler, imputer=(params.get("preprocessing") or {}).get("imputer", "median"),
                           random_state=random_state)
    t0 = time.perf_counter()
    pipe.fit(data["train"])
    t1 = time.perf_counter()
    scores = np.asarray(pipe.score(data["eval"]), dtype=np.float64)
    t2 = time.perf_counter()
    flags = scores >= float(pipe.model.threshold_)  # cùng quy ước với BaseAnomalyModel.predict
    return {"scores": scores, "flags": flags, "fit_seconds": t1 - t0, "score_seconds": t2 - t1,
            "threshold": float(pipe.model.threshold_), "n_fit": int(pipe.n_train_rows_),
            "n_features": len(pipe.feature_names)}


def _log_row(data: Dict[str, Any], res: Dict[str, Any], m: Dict[str, float], *, experiment: str, model: str,
             config_id: str, scaler: Optional[str], params: str, seed: int, contamination: float, split_day: int,
             segment: Optional[str], commit: Optional[str]) -> Dict[str, Any]:
    """Một dòng của nhật ký thí nghiệm chung (mục 5.6: thời điểm, commit, cấu hình, seed, kết quả)."""
    import sklearn

    from src.evaluation.experiment_log import build_log_row

    summary = {"n_fit": res["n_fit"], "n_train_partition": len(data["train"]), "n_eval": int(data["y"].size),
               "fit_seconds": res["fit_seconds"], "score_seconds": res["score_seconds"],
               "contamination": contamination, "n_features": res["n_features"], "imputer": "median",
               "scaler": scaler or "none", "threshold": res["threshold"],
               "alert_rate_pct": 100.0 * float(m["alert_rate"]), "sklearn_version": sklearn.__version__}
    row = build_log_row(model, res["scores"], summary, {"strategy": "time", "split_day": split_day}, seed=seed,
                        git_commit=commit, segment=segment or "all")
    row.update(experiment=experiment, config_id=config_id, run_id=data["run_id"], params=params,
               pr_auc=round(float(m["pr_auc"]), 6), roc_auc=round(float(m["roc_auc"]), 6))
    return row


def _metrics(data: Dict[str, Any], res: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    out = labeled_metrics(data["y"], res["scores"], data["days"], ks=cfg.get("precision_ks", [10, 50, 100]),
                          budget_ratio=float(cfg.get("contamination", 0.05)),
                          daily_budgets=cfg.get("daily_budgets", [10, 50, 100]))
    op = operating_point(data["y"], res["flags"])
    p, r = op["precision"], op["recall"]
    out.update(alert_rate=op["alert_rate"], precision=p, recall=r,
               f1=(2 * p * r / (p + r)) if p == p and r == r and (p + r) > 0 else float("nan"),
               fit_seconds=res["fit_seconds"], score_seconds=res["score_seconds"])
    return out


def _scenario_rows(data: Dict[str, Any], scores: np.ndarray, base: Dict[str, Any]) -> List[Dict[str, Any]]:
    from sklearn.metrics import average_precision_score

    neg = data["y"] == 0
    rows = []
    for sc in sorted({s for s in data["scenario"] if s is not None}):
        pos = data["scenario"] == sc
        m = pos | neg
        y = pos[m].astype(int)
        rows.append({**base, "scenario": sc, "n_positive": int(pos.sum()),
                     "pr_auc": float(average_precision_score(y, scores[m])),
                     "roc_auc": float(roc_auc_score(y, scores[m]))})
    return rows


def summarize(runs: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str] = METRICS) -> pd.DataFrame:
    """Trung bình, độ lệch chuẩn mẫu (ddof = 1) và số seed hợp lệ của từng chỉ số."""
    cols = [m for m in metrics if m in runs.columns]
    g = runs.groupby(list(keys), sort=False)[cols]
    mean, std, n = g.mean().add_suffix("_mean"), g.std(ddof=1).add_suffix("_std"), g.count().add_suffix("_n")
    out = pd.concat([mean, std, n], axis=1)
    order = [f"{m}_{s}" for m in cols for s in ("mean", "std", "n")]
    return out[order].reset_index()


def select_best(summary: pd.DataFrame, metric: str = "pr_auc_mean") -> Dict[str, str]:
    """Cấu hình có ``metric`` trung bình cao nhất của mỗi mô hình ML (quy tắc chọn ghi vào manifest)."""
    ml = summary[summary["kind"] == "ml"]
    return {m: str(g.sort_values(metric, ascending=False).iloc[0]["config_id"]) for m, g in ml.groupby("model")}


def run_grid(config_path: Path | str = "configs/benchmark_grid.yaml",
             params_path: Path | str = "configs/model_params.yaml",
             system_config_path: Path | str = "configs/system_config.yaml",
             output_dir: Optional[Path | str] = None, formats: Sequence[str] = ("png", "pdf"),
             experiment_log: Optional[Path | str] = "experiments/logs/experiment_log.csv") -> Dict[str, Any]:
    from src.evaluation.manifest import file_sha256, git_state, library_versions

    cfg = load_grid_config(config_path)
    params = load_params(params_path)
    # Mốc chia = mốc đóng băng của các run (mọi run của một lưới phải chung mốc); ``split_day`` trong cấu hình
    # lưới (tuỳ chọn) chỉ để kiểm tra. system config không còn quyết định: khối dev nằm trong train (mốc 35).
    splits = {run_split_day(r) for r in cfg["runs"]}
    if len(splits) != 1:
        raise ValueError(f"Các run của lưới có mốc chia khác nhau: {sorted(splits)}.")
    split_day = splits.pop()
    if cfg.get("split_day") is not None and int(cfg["split_day"]) != split_day:
        raise ValueError(f"split_day của lưới ({cfg['split_day']}) khác mốc của run ({split_day}).")
    out = Path(output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    c0 = float(cfg.get("contamination", 0.05))
    commit = (git_state() or {}).get("commit")
    exp_name = f"grid_{cfg.get('block', 'block')}"
    log_rows: List[Dict[str, Any]] = []

    rows: List[Dict[str, Any]] = []
    scen_rows: List[Dict[str, Any]] = []
    pr_scores: Dict[str, np.ndarray] = {}
    inputs = []
    ref = None
    for i, (run_dir, mseed) in enumerate(zip(cfg["runs"], cfg["model_seeds"])):
        data = load_run_data(run_dir, cfg.get("segment"), split_day)
        ref = ref or data
        inputs.append({"run": str(run_dir), "injection_seed": data["seed"], "model_seed": int(mseed),
                       "n_eval": int(data["y"].size), "n_positive": int(data["y"].sum()),
                       "matrix_sha256": file_sha256(Path(run_dir) / "processed" / "feature_matrix_processed.parquet")})
        logger.info("[grid] seed %d/%d: run %s (tiêm %s), model seed %s, %s dòng eval, %d dương.",
                    i + 1, len(cfg["runs"]), data["run_id"], data["seed"], mseed, f"{data['y'].size:,}",
                    int(data["y"].sum()))
        jobs = [("ml", m, c["id"], c.get("scaler", "robust"), c.get("params") or {})
                for m, confs in cfg["models"].items() for c in confs]
        jobs += [("baseline", b, "default", None, {}) for b in cfg.get("baselines") or []]
        for kind, model, cid, scaler, override in jobs:
            res = fit_and_score(data, model, _params_with(params, model, override), scaler, int(mseed), c0)
            base = {"seed_index": i, "run_id": data["run_id"], "injection_seed": data["seed"],
                    "model_seed": int(mseed), "kind": kind, "model": model, "config_id": cid,
                    "scaler": scaler or "none", "params": json.dumps(override, sort_keys=True)}
            m = _metrics(data, res, cfg)
            rows.append({**base, "n_eval": int(data["y"].size), "n_positive": int(data["y"].sum()), **m})
            log_rows.append(_log_row(data, res, m, experiment=exp_name, model=model, config_id=cid, scaler=scaler,
                                     params=base["params"], seed=int(mseed), contamination=c0, split_day=split_day,
                                     segment=cfg.get("segment"), commit=commit))
            scen_rows += _scenario_rows(data, res["scores"], base)
            if i == 0:
                pr_scores[f"{model}|{cid}"] = res["scores"]
            logger.info("[grid]   %-24s %-22s PR-AUC %.4f  ROC %.3f  (%.0fs)", model, cid, rows[-1]["pr_auc"],
                        rows[-1]["roc_auc"], res["fit_seconds"] + res["score_seconds"])

    runs_df = pd.DataFrame(rows)
    keys = ["kind", "model", "config_id", "scaler", "params"]
    summary = summarize(runs_df, keys)
    best = select_best(summary)
    scen_df = pd.DataFrame(scen_rows)
    scen_summary = summarize(scen_df, ["kind", "model", "config_id", "scenario"], ["pr_auc", "roc_auc"])

    # ---- độ nhạy contamination: cấu hình tốt nhất của mỗi mô hình ML
    conf_by_id = {c["id"]: (m, c) for m, confs in cfg["models"].items() for c in confs}
    c_rows = []
    for i, (run_dir, mseed) in enumerate(zip(cfg["runs"], cfg["model_seeds"])):
        data = ref if i == 0 else load_run_data(run_dir, cfg.get("segment"), split_day)
        for model, cid in best.items():
            _, c = conf_by_id[cid]
            for cont in cfg.get("contamination_sweep") or []:
                res = fit_and_score(data, model, _params_with(params, model, c.get("params") or {}),
                                    c.get("scaler", "robust"), int(mseed), float(cont))
                m = _metrics(data, res, cfg)
                c_rows.append({"seed_index": i, "run_id": data["run_id"], "model": model, "config_id": cid,
                               "contamination": float(cont), **m})
                log_rows.append(_log_row(data, res, m, experiment=f"{exp_name}_contamination", model=model,
                                         config_id=cid, scaler=c.get("scaler", "robust"),
                                         params=json.dumps(c.get("params") or {}, sort_keys=True), seed=int(mseed),
                                         contamination=float(cont), split_day=split_day,
                                         segment=cfg.get("segment"), commit=commit))
                logger.info("[grid] contamination %-22s c=%.2f  alert %.3f  P %.3f  R %.3f  PR-AUC %.4f",
                            cid, cont, m["alert_rate"], m["precision"], m["recall"], m["pr_auc"])
    cont_df = pd.DataFrame(c_rows)
    cont_summary = summarize(cont_df, ["model", "config_id", "contamination"])

    # ---- đường PR (seed đầu, không chọn seed đẹp nhất)
    pr_rows = []
    curve_keys = [f"{m}|{cid}" for m, cid in best.items()] + [f"{b}|default" for b in cfg.get("baselines") or []]
    for key in curve_keys:
        p, r, _ = precision_recall_curve(ref["y"], pr_scores[key], drop_intermediate=True)
        model, cid = key.split("|")
        pr_rows += [{"model": model, "config_id": cid, "recall": float(rr), "precision": float(pp)}
                    for pp, rr in zip(p, r)]
    pr_df = pd.DataFrame(pr_rows)

    artifacts = {
        "grid_runs": out / "grid_runs.csv", "grid_summary": out / "grid_summary.csv",
        "grid_scenarios": out / "grid_scenarios.csv", "grid_scenarios_summary": out / "grid_scenarios_summary.csv",
        "contamination_runs": out / "contamination_runs.csv",
        "contamination_summary": out / "contamination_summary.csv", "pr_curves": out / "pr_curves.csv",
    }
    for df, key in [(runs_df, "grid_runs"), (summary, "grid_summary"), (scen_df, "grid_scenarios"),
                    (scen_summary, "grid_scenarios_summary"), (cont_df, "contamination_runs"),
                    (cont_summary, "contamination_summary"), (pr_df, "pr_curves")]:
        df.to_csv(artifacts[key], index=False, float_format="%.6g")

    from src.evaluation.grid_plots import plot_grid

    figures = plot_grid(summary, scen_summary, cont_summary, pr_df, best, out / "figures",
                        positive_rate=float(ref["y"].mean()), formats=formats)
    manifest = {
        "format": "ueba-grid-manifest/1", "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "git": git_state(), "libraries": library_versions(),
        "config_path": str(config_path), "config": cfg, "params_path": str(params_path),
        "split_day": split_day, "segment": cfg.get("segment"), "inputs": inputs,
        "selection_rule": "best config per ML model = highest mean pr_auc over seeds on this block",
        "best_configs": best, "artifacts": {k: str(v) for k, v in artifacts.items()},
        "figures": [str(f) for f in figures],
    }
    (out / "grid_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                            encoding="utf-8")
    if experiment_log:
        from src.evaluation.experiment_log import append_experiment_log

        append_experiment_log(experiment_log, log_rows)
    return {"summary": summary, "best": best, "contamination": cont_summary, "output_dir": out,
            "figures": figures}
