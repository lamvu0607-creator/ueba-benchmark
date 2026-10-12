"""
Thí nghiệm độ dài cửa sổ lịch sử: tính lại ma trận của MỘT run tiêm với cửa sổ ``7d`` = N ngày
(tên cột giữ ``*_7d``), rồi chạy lưới test với cấu hình đã chốt từ dev.

    python scripts/experiments/window_ablation.py --run data/injection_runs/test_seed20261061 --windows 7 14 21 28

Ma trận: ``data/window_ablation/w{N}/<run_id>/processed/`` (không đè bản 7 ngày của run).
Kết quả: ``experiments/grid/window_ablation/w{N}/`` + bảng gộp ``window_comparison.csv``.
N = 7 là đối chứng: sha256 ma trận phải trùng ``matrix_sha256`` của run trong grid_manifest test.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.grid import run_grid  # noqa: E402
from src.evaluation.manifest import file_sha256  # noqa: E402
from src.features import build_account_day_matrix, prepare_processed_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("ueba_benchmark.window_ablation")


def ensure_injected(run: Path) -> Path:
    """Tiêm lại cùng seed nếu ``events_injected/`` đã bị xoá (cùng seed -> cùng dữ liệu)."""
    events = run / "events_injected"
    if events.is_dir() and any(events.rglob("*.parquet")):
        return events
    meta = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    logger.info("Thiếu %s -> tiêm lại block=%s seed=%s", events, meta["block"], meta["seed"])
    subprocess.run([sys.executable, "main.py", "--stage", "inject", "--block", meta["block"],
                    "--injection-seed", str(meta["seed"])], cwd=ROOT, check=True)
    return events


def sort_rows(path: Path) -> None:
    """Thứ tự dòng polars không tất định giữa các lần tính, mà IF (lấy mẫu con) và OCSVM (SGD) phụ thuộc
    thứ tự dòng -> sắp cố định để mọi cửa sổ chỉ khác nhau ở giá trị đặc trưng."""
    keys = ["day", "DomainName", "UserName"]
    df = pl.read_parquet(path)
    s = df.sort(keys)
    if not df.select(keys).equals(s.select(keys)):
        s.write_parquet(path, compression="snappy")


def build_matrix(run: Path, events: Path, window: int, out_root: Path, sys_cfg: dict) -> Path:
    dst = out_root / f"w{window}" / run.name
    processed = dst / "processed" / "feature_matrix_processed.parquet"
    if processed.is_file():
        logger.info("Đã có %s, bỏ qua tính đặc trưng.", processed)
        sort_rows(processed)
        return dst
    raw_dir = dst / "features" / "raw"
    build_account_day_matrix(output_dir=raw_dir, config=sys_cfg, events_dir=events, history_window_days=window,
                             interim_dir=sys_cfg.get("paths", {}).get("interim_data_dir"))
    prepare_processed_dataset(raw_feature_path=raw_dir / "feature_matrix_raw.parquet", output_path=processed)
    shutil.rmtree(raw_dir / "daily", ignore_errors=True)
    sort_rows(processed)
    for name in ("labels.parquet", "run_config.json"):
        shutil.copy2(run / name, dst / name)
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run", required=True)
    p.add_argument("--windows", type=int, nargs="+", default=[7, 14, 21, 28])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[42, 7, 2024, 1, 2],
                   help="Nhiều seed mô hình trên CÙNG run tiêm -> trung bình ± std của nhiễu mô hình")
    p.add_argument("--grid-config", default="configs/benchmark_grid_test.yaml")
    p.add_argument("--data-out", default="data/window_ablation")
    p.add_argument("--out", default="experiments/grid/window_ablation")
    args = p.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys_cfg = yaml.safe_load(Path("configs/system_config.yaml").read_text(encoding="utf-8")) or {}
    base_cfg = yaml.safe_load(Path(args.grid_config).read_text(encoding="utf-8"))
    events = ensure_injected(run)

    rows, scen = [], []
    for w in args.windows:
        dst = build_matrix(run, events, w, Path(args.data_out), sys_cfg)
        sha = file_sha256(dst / "processed" / "feature_matrix_processed.parquet")
        cfg = {**base_cfg, "block": f"window_w{w}", "runs": [dst.as_posix()] * len(args.model_seeds), "model_seeds": list(args.model_seeds),
               "output_dir": (out / f"w{w}").as_posix(),
               # quét contamination (~120 lần fit/cửa sổ) không cần cho câu hỏi cửa sổ -> chỉ giữ mức chính
               "contamination_sweep": [base_cfg.get("contamination", 0.05)]}
        cfg_path = out / f"grid_w{w}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if not (out / f"w{w}" / "grid_summary.csv").is_file():
            run_grid(cfg_path, experiment_log=None, formats=["png"])
        s = pd.read_csv(out / f"w{w}" / "grid_summary.csv")
        s.insert(0, "window_days", w)
        s.insert(1, "matrix_sha256", sha)
        rows.append(s)
        sc = pd.read_csv(out / f"w{w}" / "grid_scenarios_summary.csv")
        sc.insert(0, "window_days", w)
        scen.append(sc)
        logger.info("w=%d xong (sha256 %s)", w, sha[:12])

    pd.concat(rows).to_csv(out / "window_comparison.csv", index=False)
    pd.concat(scen).to_csv(out / "window_scenarios.csv", index=False)
    logger.info("Đã ghi %s", out / "window_comparison.csv")


if __name__ == "__main__":
    main()
