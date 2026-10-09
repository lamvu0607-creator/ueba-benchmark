"""Week 4 figures from saved evaluation scores; never fit a model here.

Each input directory is one complete benchmark run. Metrics are computed per run
and per segment, then aggregated; raw scores from different runs are never pooled.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.evaluation.labeled_eval import LABEL_KEYS, align_eval_labels, load_eval_labels
from src.evaluation.metrics import daily_budget_flags, precision_at_k

BASELINE_METHODS = ("random", "zscore_global", "zscore_account", "rule_ecdf", "rule_external")
NAMES = {
    "isolation_forest": "Isolation Forest", "local_outlier_factor": "LOF",
    "one_class_svm": "One-Class SVM", "zscore_baseline": "Z-score (global)",
    "rule_threshold_baseline": "Rule (registry)", "random_baseline": "Random (registry)",
    "baseline:random": "Random", "baseline:zscore_global": "Z-score global",
    "baseline:zscore_account": "Z-score account", "baseline:rule_ecdf": "Rule ECDF",
    "baseline:rule_external": "Rule external",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _validate_keys(frame: pl.DataFrame, context: str) -> pl.DataFrame:
    missing = set(LABEL_KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"{context}: missing keys {sorted(missing)}.")
    if "segment" not in frame.columns:
        frame = frame.with_columns(pl.lit("all").alias("segment"))
    frame = frame.with_columns(
        pl.col("DomainName", "UserName", "segment").cast(pl.String), pl.col("day").cast(pl.Int64)
    )
    keys = ["segment", *LABEL_KEYS]
    if frame.select(keys).null_count().row(0) != (0, 0, 0, 0):
        raise ValueError(f"{context}: null identity keys.")
    if frame.select(keys).unique().height != frame.height:
        raise ValueError(f"{context}: duplicate account-day keys within a segment.")
    if frame.is_empty():
        raise ValueError(f"{context}: no evaluation rows remain.")
    return frame


def _attach_labels(frame: pl.DataFrame, labels: Optional[pl.DataFrame]) -> pl.DataFrame:
    if labels is not None:
        aligned = align_eval_labels(frame, labels)
        for name in ("label", "is_anomaly"):
            if name in frame.columns and not np.array_equal(frame[name].to_numpy(), aligned["is_anomaly"].to_numpy()):
                raise ValueError("Saved score labels disagree with the supplied label file.")
        frame = frame.drop([c for c in ("label", "is_anomaly", "eval_exclude", "scenario") if c in frame.columns])
        frame = frame.with_columns(
            aligned["is_anomaly"].alias("label"), aligned["eval_exclude"], aligned["scenario"]
        )
    elif "label" not in frame.columns:
        if "is_anomaly" in frame.columns:
            frame = frame.rename({"is_anomaly": "label"})
        else:
            raise ValueError("Week 4 plots need labels. Pass --labels or run a labeled benchmark first.")
    values = frame["label"].to_numpy()
    if frame["label"].null_count() or not set(values.tolist()) <= {0, 1}:
        raise ValueError("Score labels must contain only 0/1, without nulls.")
    if "eval_exclude" in frame.columns:
        frame = frame.filter(~pl.col("eval_exclude").cast(pl.Boolean).fill_null(False))
    if "scenario" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, pl.String).alias("scenario"))
    return _validate_keys(frame, "labeled scores")


def _signature(manifest: dict) -> dict:
    """Reject parameter sweeps accidentally passed as repeated-seed experiments."""
    config = manifest.get("config", {})
    models = []
    for item in manifest.get("models", []):
        model = item.get("model", {})
        native = {k: v for k, v in model.get("native_params", {}).items() if k not in ("random_state", "seed", "n_jobs")}
        models.append({
            "segment": item.get("segment"), "name": model.get("model_name"),
            "native": native, "contamination": model.get("contamination"),
            "max_train_samples": model.get("max_train_samples"),
            "imputer": item.get("imputer"), "scaler": item.get("scaler"),
        })
    return {"split_day": manifest.get("split", {}).get("split_day"),
            "features": config.get("feature_names"), "feature_set": config.get("feature_set"),
            "models": sorted(models, key=lambda x: (str(x["segment"]), str(x["name"])))}


def load_plot_run(results_dir: Path | str, labels_path: Optional[Path | str] = None,
                  include_baselines: bool = True) -> dict:
    root = Path(results_dir).resolve()
    score_path = root / "anomaly_scores.parquet"
    frame = _validate_keys(pl.read_parquet(score_path), str(score_path))
    manifest = _read_json(root / "run_manifest.json")
    split = manifest.get("split") or _read_json(root / "split_info.json")
    if split.get("split_day") is not None and frame["day"].min() <= int(split["split_day"]):
        raise ValueError("Score artifact contains training rows; plots require evaluation scores only.")
    eval_days = manifest.get("extra", {}).get("eval_days")
    if eval_days is not None and not set(frame["day"].to_list()).issubset(eval_days):
        raise ValueError("Saved scores fall outside the frozen evaluation block.")
    if labels_path is None and (root.parent / "labels.parquet").is_file():
        labels_path = root.parent / "labels.parquet"
    labels = load_eval_labels(labels_path) if labels_path is not None else None
    frame = _attach_labels(frame, labels)
    score_cols = [c for c in frame.columns if c.endswith("_score")]
    if not score_cols:
        raise ValueError("No *_score columns in anomaly_scores.parquet.")
    model_cols = {c[:-6]: c for c in score_cols}
    summary_path = root / "benchmark_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    if not summary.empty:
        if "segment" not in summary:
            summary["segment"] = "all"
        if summary.duplicated(["segment", "model"]).any():
            raise ValueError("benchmark_summary.csv must have one row per segment/model.")
    baseline_path = root / "baselines" / "baseline_scores.parquet"
    if include_baselines and baseline_path.is_file():
        baseline = pl.read_parquet(baseline_path)
        if "split" not in baseline.columns:
            raise ValueError("Baseline artifact needs split=train/eval; cannot safely identify evaluation rows.")
        baseline = _validate_keys(baseline.filter(pl.col("split") == "eval"), "baseline scores")
        if labels is not None:
            baseline = _attach_labels(baseline, labels)
        else:
            baseline = baseline.join(frame.select(["segment", *LABEL_KEYS, "label", "scenario"]),
                                     on=["segment", *LABEL_KEYS], how="left", maintain_order="left")
            if baseline["label"].null_count():
                raise ValueError("Baseline rows differ from ML evaluation rows; supply the original --labels file.")
        keys = ["segment", *LABEL_KEYS]
        if baseline.height != frame.height or not baseline.select(keys).sort(keys).equals(frame.select(keys).sort(keys)):
            raise ValueError("Baseline and ML must evaluate exactly the same account-day rows and segments.")
        methods = [m for m in BASELINE_METHODS if m in baseline.columns]
        renamed = {m: f"baseline:{m}_score" for m in methods}
        frame = frame.join(baseline.select(keys + methods).rename(renamed), on=keys,
                           how="left", maintain_order="left")
        model_cols.update({f"baseline:{m}": renamed[m] for m in methods})
    for model, column in model_cols.items():
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{model}: scores must be finite, with no null values.")
    digest = hashlib.sha256()
    for column in ["segment", *LABEL_KEYS, "label", "scenario", *sorted(model_cols.values())]:
        digest.update(str(frame[column].to_list()).encode("utf-8"))
    return {"root": root, "frame": frame, "models": model_cols, "summary": summary,
            "manifest": manifest, "labels_path": str(labels_path) if labels_path else None,
            "fingerprint": digest.hexdigest()}


def compute_plot_tables(runs: list, precision_ks: Sequence[int], daily_budgets: Sequence[int]) -> dict:
    rows, curves, budgets, scenarios = [], [], [], []
    for index, run in enumerate(runs):
        for segment in run["frame"]["segment"].unique(maintain_order=True).to_list():
            part = run["frame"].filter(pl.col("segment") == segment)
            labels = part["label"].to_numpy().astype(np.int8)
            days = part["day"].to_numpy()
            n_pos = int(labels.sum())
            for model, column in run["models"].items():
                scores = part[column].to_numpy()
                valid_pr = 0 < n_pos < labels.size
                ap = float(average_precision_score(labels, scores)) if valid_pr else float("nan")
                row = {"run": index + 1, "results_dir": str(run["root"]), "segment": segment,
                       "model": model, "n_eval": labels.size, "n_positive": n_pos,
                       "positive_rate": n_pos / labels.size, "ap": ap,
                       "fit_seconds": float("nan"), "score_seconds": float("nan"), "seed": None}
                summary = run["summary"]
                if not summary.empty:
                    match = summary[(summary["segment"] == segment) & (summary["model"] == model)]
                    if not match.empty:
                        for key in ("fit_seconds", "score_seconds", "seed"):
                            if key in match:
                                row[key] = match.iloc[0][key]
                for k in precision_ks:
                    row[f"precision_at_{k}"] = (
                        precision_at_k(labels, scores, k) if n_pos else 0.0
                    ) if k <= labels.size else float("nan")
                rows.append(row)
                # Show a stated reference run, never select the best curve or pool runs.
                if index == 0 and valid_pr:
                    # Drop only redundant intermediate points; preserve PR geometry.
                    precision, recall, _ = precision_recall_curve(labels, scores, drop_intermediate=True)
                    curves.append(pd.DataFrame({"segment": segment, "model": model,
                                                "recall": recall, "precision": precision}))
                for n in daily_budgets:
                    flags = daily_budget_flags(scores, days, n).astype(bool)
                    recall = float(labels[flags].sum() / n_pos) if n_pos else float("nan")
                    budgets.append({"run": index + 1, "segment": segment, "model": model,
                                    "alerts_per_day": n, "recall": recall,
                                    "n_alerts_actual": int(flags.sum())})
                # Fixed daily budget makes scenario recall comparable across scenarios.
                flags = daily_budget_flags(scores, days, daily_budgets[len(daily_budgets) // 2]).astype(bool)
                for scenario in part["scenario"].drop_nulls().unique().to_list():
                    mask = (part["scenario"].fill_null("").to_numpy() == scenario) & (labels == 1)
                    if mask.any():
                        scenarios.append({"run": index + 1, "segment": segment, "model": model,
                                          "scenario": scenario, "n_positive": int(mask.sum()),
                                          "recall": float(flags[mask].mean())})
    return {"metrics": pd.DataFrame(rows), "pr_curves": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
            "daily_recall": pd.DataFrame(budgets), "scenario_recall": pd.DataFrame(scenarios)}


def _aggregate(frame: pd.DataFrame, keys: list, columns: list) -> pd.DataFrame:
    out = frame.groupby(keys, sort=False)[columns].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def _render(tables: dict, output: Path, formats: Sequence[str], precision_ks: list,
            daily_budgets: list, dpi: int, n_runs: int) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    files = []
    metrics = tables["metrics"]
    model_order = metrics["model"].drop_duplicates().tolist()
    colors = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(model_order)}
    label = lambda m: NAMES.get(m, m)

    def save(fig, stem):
        fig.tight_layout()
        for fmt in formats:
            path = output / f"{stem}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            files.append(str(path))
        plt.close(fig)

    def rate_axis(ax, ylabel):
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)

    def errorbars(ax, x, mean, std, **kwargs):
        # No uncertainty estimate is drawn when only one valid run exists.
        x, mean, std = np.asarray(x), np.asarray(mean), np.asarray(std)
        if kwargs.pop("fmt", None) != "none":
            ax.plot(x, mean, **kwargs)
        valid = np.isfinite(mean) & np.isfinite(std)
        if valid.any():
            ax.errorbar(x[valid], mean[valid], yerr=std[valid], fmt="none", capsize=3,
                        ecolor=kwargs.get("ecolor", kwargs.get("color", "#333333")))

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.titleweight": "bold"}):
        for segment, part in metrics.groupby("segment", sort=False):
            prefix = _safe_name(segment)
            names = part["model"].drop_duplicates().tolist()
            subtitle = f"{segment} · {n_runs} lần chạy đầy đủ"
            fig, ax = plt.subplots(figsize=(10, 6))
            curves = tables["pr_curves"]
            if not curves.empty:
                for model in names:
                    curve = curves[(curves.segment == segment) & (curves.model == model)]
                    if not curve.empty:
                        ap = part[(part.run == 1) & (part.model == model)].iloc[0].ap
                        ax.step(curve.recall, curve.precision, where="post", color=colors[model],
                                label=f"{label(model)} (AP={ap:.3f})", linewidth=1.8)
            if ax.lines:
                prevalence = part[part.run == 1].iloc[0].positive_rate
                ax.axhline(prevalence, color="#777777", linestyle="--", label=f"Mốc ngẫu nhiên ≈ {prevalence:.2%}")
                ax.legend(fontsize=8, loc="best")
            else:
                ax.text(.5, .5, "Không có đủ hai lớp nhãn để vẽ PR", ha="center", transform=ax.transAxes)
            ax.set(xlabel="Recall", xlim=(0, 1), title=f"Precision–Recall · {segment}\nLần chạy tham chiếu: đầu vào thứ nhất")
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            rate_axis(ax, "Precision")
            save(fig, f"{prefix}_pr_curve")

            stats = tables["metrics_summary"].query("segment == @segment").set_index("model").reindex(names)
            x = np.arange(len(names))
            fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.15), 6))
            ax.bar(x, stats.ap_mean, color=[colors[m] for m in names], alpha=.85)
            errorbars(ax, x, stats.ap_mean, stats.ap_std, fmt="none", ecolor="#333333")
            for i, model in enumerate(names):
                val, count = stats.loc[model, ["ap_mean", "ap_count"]]
                if np.isfinite(val):
                    ax.text(i, val + .035, f"{val:.3f}\nn={int(count)}", ha="center", fontsize=8)
                else:
                    ax.text(i, .03, "N/A\nn=0", ha="center", fontsize=8)
            ax.set_xticks(x, [label(m) for m in names], rotation=25, ha="right")
            ax.set_title(f"Average Precision (AP) · {subtitle}\nTrung bình ± độ lệch chuẩn mẫu; n=1 không có ước lượng độ lệch")
            rate_axis(ax, "AP")
            save(fig, f"{prefix}_average_precision")

            fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.3), 6))
            width = .8 / len(precision_ks)
            for j, k in enumerate(precision_ks):
                offset = x + (j - (len(precision_ks) - 1) / 2) * width
                mean, std = stats[f"precision_at_{k}_mean"], stats[f"precision_at_{k}_std"]
                ax.bar(offset, mean, width=width, label=f"Top {k}")
                errorbars(ax, offset, mean, std, fmt="none", ecolor="#333333")
            ax.set_xticks(x, [label(m) for m in names], rotation=25, ha="right")
            ax.set_title(f"Precision@k · {subtitle}\nTop-k trên toàn bộ khối đánh giá của từng phân khúc")
            ax.legend()
            rate_axis(ax, "Precision@k")
            save(fig, f"{prefix}_precision_at_k")

            fig, ax = plt.subplots(figsize=(10, 6))
            recall_stats = tables["daily_recall_summary"].query("segment == @segment")
            for model in names:
                line = recall_stats[recall_stats.model == model].sort_values("alerts_per_day")
                errorbars(ax, line.alerts_per_day, line.recall_mean, line.recall_std,
                          marker="o", color=colors[model], label=label(model))
            ax.set(xlabel="Tối đa số cảnh báo / ngày / phân khúc",
                   title=f"Recall theo ngân sách hằng ngày · {subtitle}")
            ax.set_xticks(daily_budgets)
            rate_axis(ax, "Recall")
            ax.legend(fontsize=8)
            save(fig, f"{prefix}_recall_daily_budget")

            fig, axes = plt.subplots(1, 2, figsize=(max(12, len(names) * 1.8), 6))
            for ax, field, title in zip(axes, ("fit_seconds", "score_seconds"), ("Huấn luyện", "Chấm điểm")):
                means, std = stats[f"{field}_mean"], stats[f"{field}_std"]
                valid = np.isfinite(means.to_numpy())
                ax.bar(x[valid], means[valid], color=[colors[m] for m, keep in zip(names, valid) if keep])
                errorbars(ax, x[valid], means[valid], std[valid], fmt="none", ecolor="#333333")
                ax.set_xticks(x, [label(m) for m in names], rotation=30, ha="right")
                ax.set(ylabel="Giây", title=title, ylim=(0, None))
                ax.grid(axis="y", alpha=.22)
                if not valid.all():
                    ax.set_title(f"{title}\nCột trống: chưa ghi nhận thời gian", fontsize=11)
            fig.suptitle(f"Thời gian chạy · {subtitle}", fontweight="bold")
            save(fig, f"{prefix}_runtime")

            scenario = tables["scenario_recall"]
            if not scenario.empty:
                cells = scenario.query("segment == @segment").groupby(["model", "scenario"]).recall.mean().unstack()
                if not cells.empty:
                    cells = cells.reindex(names)
                    fig, ax = plt.subplots(figsize=(max(8, len(cells.columns) * 1.4), max(4, len(names) * .55 + 2)))
                    image = ax.imshow(cells, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
                    for i in range(len(cells)):
                        for j in range(len(cells.columns)):
                            val = cells.iloc[i, j]
                            if np.isfinite(val):
                                ax.text(j, i, f"{val:.0%}", ha="center", va="center", color="white" if val > .6 else "black")
                    ax.set_xticks(range(len(cells.columns)), cells.columns, rotation=25, ha="right")
                    ax.set_yticks(range(len(names)), [label(m) for m in names])
                    n = daily_budgets[len(daily_budgets) // 2]
                    ax.set_title(f"Recall theo kịch bản · {segment}\nNgân sách {n} cảnh báo / ngày / phân khúc")
                    fig.colorbar(image, ax=ax, format=PercentFormatter(1))
                    save(fig, f"{prefix}_scenario_recall")
    return files


def plot_benchmark_results(results_dirs: Sequence[Path | str], *,
                           labels_paths: Optional[Sequence[Optional[Path | str]]] = None,
                           output_dir: Optional[Path | str] = None, include_baselines: bool = True,
                           precision_ks: Sequence[int] = (10, 50, 100),
                           daily_budgets: Sequence[int] = (10, 20, 50, 100),
                           formats: Sequence[str] = ("png", "pdf"), dpi: int = 180) -> dict:
    """Produce figures, numeric CSVs and provenance from one or more saved runs."""
    roots = [Path(p).resolve() for p in results_dirs]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Provide at least one results directory, without duplicates.")
    labels_paths = list(labels_paths) if labels_paths is not None else [None] * len(roots)
    if len(labels_paths) != len(roots):
        raise ValueError("Provide one label path per results directory.")
    ks, budgets = sorted(set(precision_ks)), sorted(set(daily_budgets))
    if not ks or not budgets or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in ks + budgets):
        raise ValueError("K and daily budgets must be positive integers.")
    if not formats or set(formats) - {"png", "pdf", "svg"} or dpi <= 0:
        raise ValueError("Formats must be png/pdf/svg and DPI must be positive.")
    runs = [load_plot_run(root, lab, include_baselines) for root, lab in zip(roots, labels_paths)]
    if len(runs) > 1:
        if any(not run["manifest"] for run in runs):
            raise ValueError("Multiple runs require run_manifest.json to verify experiment configuration.")
        reference = _signature(runs[0]["manifest"])
        days = sorted(runs[0]["frame"]["day"].unique().to_list())
        segments = set(runs[0]["frame"]["segment"].to_list())
        for run in runs[1:]:
            if _signature(run["manifest"]) != reference:
                raise ValueError("Runs use different model parameters/features/splits. Plot each configuration separately.")
            if (sorted(run["frame"]["day"].unique().to_list()) != days
                    or set(run["frame"]["segment"].to_list()) != segments
                    or set(run["models"]) != set(runs[0]["models"])):
                raise ValueError("Runs must have the same evaluation days, segments and methods; do not mix dev/test.")
        # Deterministic models can legitimately return identical scores for different seeds.
        identities = {(run["fingerprint"], run["manifest"].get("config", {}).get("seed"),
                       _read_json(run["root"].parent / "run_config.json").get("seed")) for run in runs}
        if len(identities) != len(runs):
            raise ValueError("Duplicate score/label runs cannot be counted as independent experiments.")
    tables = compute_plot_tables(runs, ks, budgets)
    columns = ["ap", "positive_rate", "fit_seconds", "score_seconds", *[f"precision_at_{k}" for k in ks]]
    tables["metrics_summary"] = _aggregate(tables["metrics"], ["segment", "model"], columns)
    tables["daily_recall_summary"] = _aggregate(
        tables["daily_recall"], ["segment", "model", "alerts_per_day"], ["recall"]
    )
    output = Path(output_dir) if output_dir else roots[0] / "figures" / "week4"
    output.mkdir(parents=True, exist_ok=True)
    files = _render(tables, output, formats, ks, budgets, dpi, len(runs))
    for name, frame in tables.items():
        if not frame.empty:
            frame.to_csv(output / f"{name}.csv", index=False)
    metadata = {"format": "ueba-week4-plots/1", "n_runs": len(runs),
                "reference_pr_run": str(roots[0]), "precision_scope": "whole evaluation block per segment",
                "recall_budget_scope": "per day per segment", "precision_ks": ks, "daily_budgets": budgets,
                "scenario_budget": budgets[len(budgets) // 2], "std_ddof": 1,
                "single_run_std": "undefined (no error estimate)",
                "metric": "Average Precision, not trapezoidal PR area",
                "score_direction": "higher = more anomalous", "figures": files,
                "inputs": [{"results_dir": str(r["root"]), "labels_path": r["labels_path"],
                            "fingerprint": r["fingerprint"], "config": r["manifest"].get("config"),
                            "git": r["manifest"].get("git")} for r in runs],
                "notes": ["Uninjected rows are negatives under the synthetic-label convention.",
                          "Metrics are recomputed from saved scores; old rounded artifacts may differ from original summaries.",
                          "AP/PR require both label classes; recall requires positives; K > n_eval is undefined.",
                          "Runtime is read from benchmark_summary.csv; missing baseline timing is not estimated.",
                          "Stability sample seeds are not full benchmark repetitions."]}
    (output / "plot_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(output), "figures": files, "tables": tables, "manifest": metadata}
