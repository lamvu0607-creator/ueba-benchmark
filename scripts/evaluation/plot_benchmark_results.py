"""Render Week 4 evaluation figures without fitting models."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.plots import plot_benchmark_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 4 plots from saved benchmark scores")
    parser.add_argument("--results-dir", nargs="+", required=True, help="One result directory per full run")
    parser.add_argument("--labels", nargs="+", help="One label path per result directory; auto-detect run labels by default")
    parser.add_argument("--output-dir", help="Default: first result directory / figures / week4")
    parser.add_argument("--precision-ks", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--daily-budgets", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-baselines", action="store_true", help="Ignore results/baselines/baseline_scores.parquet")
    args = parser.parse_args()
    try:
        result = plot_benchmark_results(args.results_dir, labels_paths=args.labels, output_dir=args.output_dir,
                                        include_baselines=not args.no_baselines, precision_ks=args.precision_ks,
                                        daily_budgets=args.daily_budgets, formats=args.formats, dpi=args.dpi)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Created {len(result['figures'])} figures: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
