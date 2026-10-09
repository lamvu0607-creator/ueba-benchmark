"""Chạy lưới thí nghiệm tuần 4: python scripts/evaluation/run_grid.py [--config configs/benchmark_grid.yaml]."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.grid import run_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="3 ML models x configs x seeds grid on injection runs")
    parser.add_argument("--config", default="configs/benchmark_grid.yaml")
    parser.add_argument("--model-params", default="configs/model_params.yaml")
    parser.add_argument("--system-config", default="configs/system_config.yaml")
    parser.add_argument("--output-dir", default=None, help="Default: output_dir of the grid config")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    parser.add_argument("--experiment-log", default="experiments/logs/experiment_log.csv",
                        help="Shared experiment log (spec 5.6); pass '' to skip")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
    result = run_grid(args.config, args.model_params, args.system_config, args.output_dir, args.formats,
                      experiment_log=args.experiment_log or None)
    logging.getLogger("ueba_benchmark.grid").info("Cấu hình tốt nhất: %s -> %s", result["best"], result["output_dir"])


if __name__ == "__main__":
    main()
