"""
Chốt cấu hình cho khối TEST từ kết quả lưới DEV (không chọn lại gì trên test).

Đọc ``best_configs`` trong ``<dev output>/grid_manifest.json`` (PR-AUC trung bình cao nhất trên dev), lấy đúng
mục cấu hình đó từ lưới dev, rồi ghi lưới test (cùng baseline, ngân sách, mức quét contamination).

    python scripts/evaluation/freeze_test_grid.py --runs data/injection_runs/test_seed20261061 ...
"""

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-config", default="configs/benchmark_grid.yaml")
    ap.add_argument("--out", default="configs/benchmark_grid_test.yaml")
    ap.add_argument("--output-dir", default="experiments/grid/test")
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    dev = yaml.safe_load(Path(args.dev_config).read_text(encoding="utf-8"))
    manifest_path = Path(dev["output_dir"]) / "grid_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best = manifest["best_configs"]
    if len(args.runs) != len(dev["model_seeds"]):
        raise SystemExit(f"Cần {len(dev['model_seeds'])} run test (một run cho mỗi seed mô hình).")
    models = {m: [c for c in dev["models"][m] if c["id"] == cid] for m, cid in best.items()}
    test = {
        "block": "test", "segment": dev["segment"], "output_dir": args.output_dir,
        "runs": list(args.runs), "model_seeds": dev["model_seeds"],
        "contamination": dev["contamination"], "precision_ks": dev["precision_ks"],
        "daily_budgets": dev["daily_budgets"], "models": models, "baselines": dev["baselines"],
        "contamination_sweep": dev["contamination_sweep"],
    }
    header = (
        "# =============================================================================\n"
        "# Khối TEST — SINH TỰ ĐỘNG bởi scripts/evaluation/freeze_test_grid.py, không sửa tay.\n"
        f"# Cấu hình chốt trên dev: {manifest_path} (best_configs, commit {manifest['git'].get('commit')}).\n"
        "# Một cấu hình / mô hình -> select_best không còn lựa chọn; KHÔNG chọn lại trên test.\n"
        "# =============================================================================\n"
    )
    Path(args.out).write_text(header + yaml.safe_dump(test, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Đã ghi {args.out}: {best}")


if __name__ == "__main__":
    main()
