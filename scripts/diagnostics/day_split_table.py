"""
Chẩn đoán: phân bố số dòng theo NGÀY và các phương án ``split_day`` cho time-based split.

Chạy: ``python scripts/diagnostics/day_split_table.py``

Căn cứ chọn mặc định của dự án: ``train = ngày 1..42``, ``test = ngày 43..60``
(tương ứng ``split_day = 42`` trong ``configs/system_config.yaml``). Script in bảng để bất kỳ ai
cũng kiểm chứng lại được con số này thay vì tin vào một hằng số không nguồn gốc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_PATH = Path("data/processed/feature_matrix_processed.parquet")
CANDIDATE_SPLIT_DAYS = (38, 39, 40, 41, 42, 43, 44, 45)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - chỉ ảnh hưởng hiển thị console
        pass


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit(f"Không tìm thấy '{DATA_PATH}'. Hãy chạy stage features trước.")

    days = pl.read_parquet(DATA_PATH, columns=["day"])
    total = days.height
    per_day = {int(row["day"]): int(row["len"]) for row in days.group_by("day").len().sort("day").iter_rows(named=True)}

    print(f"Dữ liệu: {DATA_PATH} — {total:,} dòng, {len(per_day)} ngày (day {min(per_day)}..{max(per_day)})")
    print()
    print("Phân bố theo ngày (10 ngày đầu/cuối):")
    ordered = sorted(per_day)
    for d in ordered[:5] + ordered[-5:]:
        print(f"  day {d:>3}: {per_day[d]:>9,} dòng")

    print()
    print(f"{'split_day':>9} {'n_train':>12} {'n_eval':>11} {'eval_ratio':>11}  ghi chú")
    for split_day in CANDIDATE_SPLIT_DAYS:
        n_eval = sum(v for k, v in per_day.items() if k > split_day)
        n_train = total - n_eval
        note = "<- mặc định của dự án (train 1-42, test 43-60)" if split_day == 42 else ""
        print(f"{split_day:>9} {n_train:>12,} {n_eval:>11,} {n_eval / total:>10.2%}  {note}")


if __name__ == "__main__":
    main()
