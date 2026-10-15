"""
Tái lập benchmark cuối bằng MỘT lệnh (tiêu chí 1, đề cương mục 4.1).

    python scripts/reproduce.py                       # đầy đủ: test -> features -> tiêm 5 run test -> lưới -> đối chiếu
    python scripts/reproduce.py --skip-existing       # bỏ qua bước đã có đầu ra (ma trận gốc, run tiêm đã có processed/)
    python scripts/reproduce.py --compare-only        # chỉ đối chiếu một kết quả lưới đã có với số đã báo cáo

Điều kiện: đã cài môi trường theo README §2 và có ``data/interim/`` (log 4624/4625 ngày 1–60, README §6).

Các bước (đúng thứ tự README §2 và §3):
1. ``pytest``;
2. ma trận đặc trưng của log gốc (``main.py --stage features``);
3. hồ sơ train cho bộ tiêm (``scripts/injection_profiles_summary.py``) và baselines trên log gốc (rule-skip);
4. tiêm 5 run khối test (seed 20261061–20261065), mỗi run tính lại ma trận đặc trưng;
5. lưới test chính thức (mặc định ``configs/benchmark_grid_test_22f.yaml``) vào thư mục riêng ``--out``;
6. đối chiếu ``grid_summary.csv`` mới với bản đã báo cáo (``--reference``): sai lệch tương đối ≤ ``--tolerance``.

Mọi bước ngẫu nhiên có seed cố định (seed tiêm trong lệnh, seed mô hình trong cấu hình lưới). Ngoại lệ đã biết:
LOF dựng chỉ mục HNSW đa luồng (``n_jobs: -1`` trong configs/model_params.yaml) nên không tất định; fit lại trên
test lệch khoảng ±0,003 recall top-100/ngày (khoảng 2% tương đối). Bước 6 báo LOF riêng với ngưỡng ``--lof-tolerance``;
P@50 của LOF dao động rất mạnh giữa các lần fit (0,196 ± 0,111) nên chỉ in ra để tham khảo, không tính vào kết quả.

Mã thoát 0 nếu mọi bước chạy được và đối chiếu đạt; 1 nếu ngược lại.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_SEEDS = (20261061, 20261062, 20261063, 20261064, 20261065)
METRICS = ("recall_at_100_per_day", "pr_auc", "roc_auc", "precision_at_50")
ML = ("one_class_svm", "isolation_forest", "local_outlier_factor")
INTERIM_HASHES = ROOT / "configs/interim_sha256.txt"


def verify_interim(interim_dir: Path, manifest: Path = INTERIM_HASHES) -> None:
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    actual = {p.relative_to(interim_dir).as_posix() for event in (4624, 4625)
              for p in (interim_dir / f"event_{event}").glob("*.parquet")}
    if actual != set(expected):
        raise ValueError(f"Kho interim khác danh sách đã chốt: thiếu {sorted(set(expected) - actual)}, dư {sorted(actual - set(expected))}")
    for name, digest in expected.items():
        with (interim_dir / name).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError(f"Hash interim không khớp: {name}")
    print(f"Đã xác minh SHA-256 của {len(expected)} file interim.", flush=True)


def step(title: str, cmd: list) -> None:
    print(f"\n=== {title}\n$ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def compare(new_path: Path, ref_path: Path, tol: float, lof_tol: float) -> bool:
    new = pd.read_csv(new_path).set_index("model")
    ref = pd.read_csv(ref_path).set_index("model")
    if (not new.index.is_unique or not ref.index.is_unique or set(ref.index) - set(new.index)
            or any(f"{m}_mean" not in new.columns or f"{m}_mean" not in ref.columns for m in METRICS)):
        raise ValueError("Bảng đối chiếu thiếu model/metric hoặc có model trùng.")
    rows, ok = [], True
    for model in ref.index:
        limit = lof_tol if model == "local_outlier_factor" else tol
        for m in METRICS:
            col = f"{m}_mean"
            a, b = float(ref.loc[model, col]), float(new.loc[model, col])
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"Metric không hữu hạn: {model}/{m}")
            rel = abs(b - a) / abs(a) if a else abs(b - a)
            counted = not (model == "local_outlier_factor" and m == "precision_at_50")
            passed = rel <= limit
            ok &= passed or not counted
            rows.append({"model": model, "metric": m, "reported": round(a, 4), "reproduced": round(b, 4),
                         "rel_diff": round(rel, 4), "limit": limit if counted else "tham khảo", "ok": passed})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(new_path.parent / "reproduce_check.csv", index=False, encoding="utf-8-sig")
    print(f"\nĐối chiếu với {ref_path}: {'ĐẠT' if ok else 'KHÔNG ĐẠT'} "
          f"(ngưỡng {tol:.0%}, LOF {lof_tol:.0%}) -> {new_path.parent / 'reproduce_check.csv'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--grid", default="configs/benchmark_grid_test_22f.yaml")
    p.add_argument("--out", default="experiments/reproduce/test_22f")
    p.add_argument("--reference", default="experiments/grid/test_22f/grid_summary.csv")
    p.add_argument("--tolerance", type=float, default=0.01, help="sai lệch tương đối tối đa (OCSVM, IF, baseline)")
    p.add_argument("--lof-tolerance", type=float, default=0.05, help="sai lệch tương đối tối đa cho LOF")
    p.add_argument("--skip-existing", action="store_true", help="bỏ qua bước đã có đầu ra")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--compare-only", action="store_true")
    args = p.parse_args()
    py = sys.executable
    out = ROOT / args.out

    if not args.compare_only:
        if not (ROOT / "data/interim/event_4624").exists():
            print("Thiếu data/interim/event_4624 — tải dữ liệu interim theo README §6 trước.", file=sys.stderr)
            return 1
        verify_interim(ROOT / "data/interim")
        if not args.skip_tests:
            step("1. Kiểm thử", [py, "-m", "pytest", "tests", "-q"])
        if not (args.skip_existing and (ROOT / "data/processed/feature_matrix_processed.parquet").exists()):
            step("2. Ma trận đặc trưng log gốc", [py, "main.py", "--stage", "features", "--interim-only"])
        if not (args.skip_existing and (ROOT / "data/features/train_profiles").exists()):
            step("3a. Hồ sơ train", [py, "scripts/injection_profiles_summary.py"])
        if not (args.skip_existing and (ROOT / "experiments/results/baselines").exists()):
            step("3b. Baselines trên log gốc", [py, "main.py", "--stage", "baselines"])
        for seed in TEST_SEEDS:
            run = ROOT / "data/injection_runs" / f"test_seed{seed}"
            if args.skip_existing and (run / "processed/feature_matrix_processed.parquet").exists():
                print(f"\n=== 4. Run test_seed{seed}: đã có, bỏ qua")
                continue
            step(f"4. Tiêm run test_seed{seed}", [py, "main.py", "--stage", "inject", "--block", "test",
                                                  "--injection-seed", seed, "--run-id", f"test_seed{seed}"])
            step(f"4. Đặc trưng run test_seed{seed}", [py, "main.py", "--stage", "features",
                                                       "--events-dir", run / "events_injected"])
        step("5. Lưới test chính thức", [py, "scripts/evaluation/run_grid.py", "--config", args.grid,
                                          "--output-dir", out])

    return 0 if compare(out / "grid_summary.csv", ROOT / args.reference, args.tolerance, args.lof_tolerance) else 1


if __name__ == "__main__":
    sys.exit(main())
