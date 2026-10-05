"""
plot_score_distribution.py

Vẽ **phân bố điểm dị biệt** của các thuật toán trên tập test (ngày 43–60) và in bảng thống kê
đi kèm. Mục đích chính: trả lời câu hỏi "nếu LOF / One-Class SVM chỉ fit trên **N mẫu con**
(mặc định 50.000) còn Isolation Forest fit **toàn bộ** tập train thì phân bố điểm khác nhau thế nào?"

Cách dùng::

    python scripts/diagnostics/plot_score_distribution.py
    python scripts/diagnostics/plot_score_distribution.py --subsample 50000 --models isolation_forest local_outlier_factor one_class_svm
    python scripts/diagnostics/plot_score_distribution.py --split-day -1 --split-ratio 0.30   # suy ranh giới từ tỷ lệ

Đầu ra:
    * `reports/figures/score_distribution_<N>k_subset.png` — lưới 2 hàng × k cột:
        - Hàng 1 (**ECDF**, trục x = thang ``symlog``): thấy được *cả* khối chính *lẫn* đuôi cực nặng
          trên cùng một trục; có đường ngân sách `1 − contamination` và dấu × đỏ cho các dòng thuộc
          tầng `System`/`Service` (nơi có tài khoản `Anonymous`) — chính là "đuôi vô hình" của LOF
          khi vẽ histogram thang gốc.
        - Hàng 2 (**histogram ZOOM [p1, p99]** + hộp ghi chú): hình dạng khối chính, kèm số dòng
          nằm NGOÀI khung để không "giấu" phần đuôi.
    * `reports/score_distribution_stats.csv` — bằng chứng số: phân vị điểm fit/test, alert rate,
      drift ngân sách, độ lệch (skew), độ nhọn (kurtosis), thống kê KS giữa fit và test, thời gian.

Lưu ý diễn giải:
    * Đường ngưỡng là **phân vị trên tập FIT** (contamination), nên alert rate trên test có thể
      lệch — cột `alert_rate_drift_pp` trong CSV đo đúng độ lệch đó.
    * Isolation Forest mặc định fit **toàn bộ** train; LOF/One-Class SVM dùng `--subsample`
      (mặc định 50.000) theo thiết kế `default_max_train_samples` của repo.
    * LOF tính **tỷ số mật độ cục bộ** nên KHÔNG có chặn trên: một tầng hiếm nằm xa đám đông (ví dụ
      tài khoản `Anonymous` của tầng `System`) sẽ nổ điểm và kéo dài trục. Vì thế hình dùng ECDF trên
      trục symlog thay cho histogram thang gốc, và đánh dấu riêng các tầng đó.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")

from matplotlib.ticker import FuncFormatter  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as sps
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.split import time_split  # noqa: E402
from src.features.schema import FeatureSchema  # noqa: E402
from src.models.registry import create_pipeline, load_params  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "feature_matrix_processed.parquet"
DEFAULT_PARAMS = REPO_ROOT / "configs" / "model_params.yaml"
DEFAULT_MODELS = ["isolation_forest", "local_outlier_factor", "one_class_svm"]
DEFAULT_FIG_DIR = REPO_ROOT / "reports" / "figures"
DEFAULT_STATS_OUT = REPO_ROOT / "reports" / "score_distribution_stats.csv"

#: Mô hình LUÔN fit toàn bộ train (không áp ``--subsample``) — đúng thiết kế repo.
FULL_FIT_MODELS = ("isolation_forest",)


def _quantiles(values: np.ndarray, probs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    return {f"p{int(p * 100):02d}": float(np.quantile(values, p)) for p in probs}


def _symlog_formatter(linthresh: float) -> FuncFormatter:
    """Nhãn trục cho ``symlog``: dùng ``%g`` ở vùng |x| < 1e4 (tránh kiểu ``−2.99×10⁻¹``) và
    ký hiệu khoa học cho phần đuôi lớn."""

    def _fmt(value, _pos):
        if value == 0:
            return "0"
        return f"{value:g}" if abs(value) < 1e4 else f"{value:.0e}"

    return FuncFormatter(_fmt)


def run(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"[LỖI] Không tìm thấy ma trận đặc trưng: {data_path}")
        return 1

    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    df = pl.read_parquet(data_path)
    core = list(FeatureSchema().core_features)
    split_day = None if args.split_day is not None and args.split_day < 0 else args.split_day
    train, test, split_info = time_split(df, day_col="day", split_day=split_day,
                                        test_split_ratio=args.split_ratio)
    params = load_params(DEFAULT_PARAMS)

    print("=" * 108)
    print("PHÂN BỐ ĐIỂM DỊ BIỆT — LOF/OCSVM fit mẫu con vs Isolation Forest fit toàn bộ")
    print("=" * 108)
    print(f"Ma trận      : {data_path.name} ({df.height:,} dòng × {df.width} cột | {len(core)} đặc trưng core)")
    print(f"Chia tập     : train = day <= {split_info.split_day} ({train.height:,} dòng) | "
          f"test = {test.height:,} dòng | seed={args.seed} | ngân sách {args.budget_ratio * 100:.1f}%")
    print("-" * 108)
    print(f"{'mo hinh':<24}{'n_fit':>9}{'alert%':>8}{'drift_pp':>9}{'thresh':>11}{'fit_s':>7}{'score_s':>8}")
    print("-" * 108)

    rows, collected = [], []
    for name in args.models:
        max_train = None if name in FULL_FIT_MODELS else args.subsample
        pipeline = create_pipeline(
            name,
            params=params,
            feature_names=core,
            contamination=float(args.budget_ratio),
            random_state=int(args.seed),
            max_train_samples=max_train,
        )
        t0 = time.perf_counter()
        pipeline.fit(train)
        fit_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        test_scores = pipeline.score(test)
        score_seconds = time.perf_counter() - t1
        fit_scores = np.asarray(pipeline.model.fit_scores_, dtype=np.float64)
        alert_rate = float((test_scores >= float(pipeline.model.threshold_)).mean() * 100.0)

        stats = {
            "model": name,
            "n_train_partition": int(train.height),
            "n_fit": int(pipeline.model.n_fit_),
            "n_eval": int(test.height),
            "threshold": float(pipeline.model.threshold_),
            "alert_rate_pct": alert_rate,
            "alert_rate_drift_pp": alert_rate - float(args.budget_ratio) * 100.0,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "skew_test": float(sps.skew(test_scores)),
            "kurtosis_test": float(sps.kurtosis(test_scores)),
            "ks_fit_vs_test": float(sps.ks_2samp(fit_scores, test_scores).statistic),
            "n_unique_fit": int(np.unique(fit_scores).size),
            "n_unique_test": int(np.unique(test_scores).size),
        }
        mode_val, mode_cnt = (lambda v, c: (float(v[c.argmax()]), int(c.max())))(
            *np.unique(fit_scores, return_counts=True)
        )
        stats["fit_ties_share_pct"] = 100.0 * mode_cnt / fit_scores.size
        stats["fit_mode_value"] = mode_val
        for key, value in _quantiles(fit_scores).items():
            stats[f"fit_{key}"] = value
        for key, value in _quantiles(test_scores).items():
            stats[f"test_{key}"] = value
        rows.append(stats)
        # Thông tin ĐUÔI (ngoài p99) để chú thích trên hình: đuôi LOF = tài khoản System/Anonymous.
        p99 = float(np.quantile(test_scores, 0.99))
        tail_mask = test_scores > p99
        etypes = test["entity_type"].to_numpy()
        users = test["UserName"].to_numpy()
        targets = [str(v) for v in getattr(args, "highlight_entities", ["System", "Service"])]
        sys_mask = np.isin(etypes, targets)
        extra = {
            "tail_n": int(tail_mask.sum()),
            "tail_max": float(test_scores.max()),
            "tail_by_entity": Counter(etypes[tail_mask]).most_common(3),
            "tail_top_users": Counter(users[tail_mask]).most_common(3),
            "sys_mask": sys_mask,
            "sys_name": "/".join(targets),
        }
        collected.append((name, fit_scores, test_scores, stats, extra))

        print(f"{name:<24}{int(pipeline.model.n_fit_):>9,}{alert_rate:>8.2f}"
              f"{stats['alert_rate_drift_pp']:>9.2f}{pipeline.model.threshold_:>11.4g}"
              f"{fit_seconds:>7.2f}{score_seconds:>8.2f}")

    return _draw(args, collected, rows)



def _draw(args: argparse.Namespace, collected, rows) -> int:
    """
    Vẽ lưới 2 hàng × k cột:

    * **Hàng 1 — ECDF (hàm phân phối tích luỹ) trên trục ``symlog``**: đọc được cả khối chính lẫn
      đuôi cực nặng trong CÙNG một trục (đây là lý do đổi khỏi histogram thang gốc: với LOF, trục
      phải trải tới ~165 nên histogram nén mọi dòng thành một vạch). Đánh dấu riêng dòng thuộc tầng
      ``System`` (nơi có ``Anonymous``) bằng dấu × đỏ và vẽ đường ngân sách ``1 − contamination``.
    * **Hàng 2 — histogram ZOOM [p1, p99]** + hộp ghi chú đuôi (số dòng ngoài p99, max, tài khoản
      đứng đầu) để không "giấu" phần đuôi như trước.
    """
    n_models = len(collected)
    fig, axes = plt.subplots(2, n_models, figsize=(5.8 * n_models, 8.8), squeeze=False)

    for idx, (name, fit_scores, test_scores, stats, extra) in enumerate(collected):
        ax_ecdf, ax_zoom = axes[0][idx], axes[1][idx]
        title = (f"{name}\nfit {stats['n_fit']:,} mẫu · alert {stats['alert_rate_pct']:.2f}% "
                 f"(lệch {stats['alert_rate_drift_pp']:+.2f} pp)")

        # ---------------- Hàng 1: ECDF trên trục symlog ----------------
        for scores, color, label in ((fit_scores, "#8c8c8c", "FIT (train)"),
                                     (test_scores, "#1f77b4", "TEST (ngày 43–60)")):
            xs = np.sort(scores)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax_ecdf.step(xs, ys, where="post", color=color, linewidth=1.6, label=f"ECDF {label}")

        # Điểm của tầng System (nơi có tài khoản Anonymous) — chính là đuôi của LOF
        sys_mask = extra["sys_mask"]
        if sys_mask.any():
            sx = test_scores[sys_mask]
            order = np.searchsorted(np.sort(test_scores), sx, side="right")
            sy = order / test_scores.size
            ax_ecdf.scatter(sx, sy, marker="x", s=46, color="#d62728", zorder=5,
                            label=f"tầng {extra['sys_name']} ({int(sys_mask.sum())} dòng)")

        ax_ecdf.axvline(stats["threshold"], color="#d62728", linestyle="--", linewidth=1.4,
                        label=f"ngưỡng = p{100 * (1 - args.budget_ratio):.0f} của FIT")
        ax_ecdf.axhline(1 - args.budget_ratio, color="#ff7f0e", linestyle=":", linewidth=1.4,
                        label=f"1 − ngân sách = {1 - args.budget_ratio:.2f}")
        # Trục x: chỉ dùng symlog khi khoảng dữ liệu VƯỢT ngưỡng tuyến tính (nếu không, symlog làm
        # nhãn trục thành kiểu "−2.99×10⁻¹" mà chẳng ích gì — ví dụ Isolation Forest).
        linthresh = float(getattr(args, "linthresh", 1.0))
        lo = float(min(fit_scores.min(), test_scores.min()))
        hi = float(max(fit_scores.max(), test_scores.max()))
        if max(abs(lo), abs(hi)) > linthresh:
            ax_ecdf.set_xscale("symlog", linthresh=linthresh)
            ax_ecdf.xaxis.set_major_formatter(_symlog_formatter(linthresh))
            margin = 1.35
            ax_ecdf.set_xlim(lo * margin if lo < 0 else lo - abs(lo) * 0.35 - 0.05,
                             hi * margin if hi > 0 else hi + abs(hi) * 0.35 + 0.05)
            ax_ecdf.set_xlabel("điểm dị biệt (CAO = dị biệt) — thang symlog")
        else:
            ax_ecdf.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
            ax_ecdf.set_xlabel("điểm dị biệt (CAO = dị biệt)")
        ax_ecdf.set_ylim(-0.02, 1.02)
        ax_ecdf.set_title(f"{title}\nECDF ({'trục symlog' if max(abs(lo), abs(hi)) > linthresh else 'thang gốc'})",
                          fontsize=10)
        ax_ecdf.set_ylabel("tỷ lệ dòng ≤ x")
        top_users = ", ".join(f"{u} ({c})" for u, c in extra["tail_top_users"]) or "—"
        by_entity = ", ".join(f"{e}:{c}" for e, c in extra["tail_by_entity"]) or "—"
        ax_ecdf.text(0.02, 0.04,
                     f"đuôi (score > p99): {extra['tail_n']} dòng · max = {extra['tail_max']:.1f}\n"
                     f"theo tầng: {by_entity}\ntài khoản: {top_users}",
                     transform=ax_ecdf.transAxes, fontsize=8, va="bottom",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax_ecdf.legend(fontsize=7, loc="lower right")

        # ---------------- Hàng 2: histogram zoom [p1, p99] ----------------
        # PHẢI truyền bin tường minh trong cửa sổ zoom: nếu để matplotlib tự chia bin trên toàn bộ
        # khoảng dữ liệu ([−0,58; 164,7] với LOF) thì mỗi bin rộng ~1,0 và 99,8% số dòng dồn vào bin
        # đầu tiên ⇒ biểu đồ chỉ còn một "khối" phẳng, không thấy hình dạng (đây chính là lý do bản
        # vẽ trước trông như "một gai + đuôi vô hình").
        lo_z = min(stats["test_p01"], stats["fit_p01"])
        hi_z = max(stats["test_p99"], stats["fit_p99"])
        pad_z = 0.05 * (hi_z - lo_z) if hi_z > lo_z else 1.0
        lo_z, hi_z = lo_z - pad_z, hi_z + pad_z
        edges = np.linspace(lo_z, hi_z, 121)

        ax_zoom.hist(fit_scores, bins=edges, density=True, alpha=0.45, color="#8c8c8c",
                     label="điểm FIT (train)")
        ax_zoom.hist(test_scores, bins=edges, density=True, alpha=0.55, color="#1f77b4",
                     label="điểm TEST (ngày 43–60)")
        ax_zoom.axvline(stats["threshold"], color="#d62728", linestyle="--", linewidth=1.6,
                        label=f"ngưỡng = p{100 * (1 - args.budget_ratio):.0f} của FIT")
        ax_zoom.axvline(stats["test_p50"], color="#2ca02c", linestyle=":", linewidth=1.3,
                        label="median TEST")
        ax_zoom.set_xlim(lo_z, hi_z)
        ax_zoom.set_title(f"{name} — histogram ZOOM [p1, p99]", fontsize=10)
        ax_zoom.set_xlabel("điểm dị biệt (CAO = dị biệt)")
        ax_zoom.set_ylabel("mật độ")
        hi_test = float(test_scores.max())
        outside = int(((test_scores < lo_z) | (test_scores > hi_z)).sum())
        ax_zoom.text(0.02, 0.95,
                     f"ngoài khung: {outside:,}/{test_scores.size:,} dòng "
                     f"(TEST max = {hi_test:.1f})\n"
                     f"giá trị điểm phân biệt: FIT {stats['n_unique_fit']:,} · TEST {stats['n_unique_test']:,}\n"
                     f"điểm trùng nhiều nhất (FIT): {stats['fit_mode_value']:.3g} ({stats['fit_ties_share_pct']:.1f}%)\n"
                     f"skew={stats['skew_test']:.2f} · kurtosis={stats['kurtosis_test']:.1f}\n"
                     f"KS(fit,test)={stats['ks_fit_vs_test']:.3f}\n"
                     f"fit {stats['fit_seconds']:.2f}s · score {stats['score_seconds']:.2f}s",
                     transform=ax_zoom.transAxes, va="top", fontsize=8,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax_zoom.legend(fontsize=7, loc="upper right")

    fig.suptitle("Phân bố điểm dị biệt — LOF/One-Class SVM fit mẫu con vs Isolation Forest fit toàn bộ",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    out_csv = Path(args.stats_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("-" * 108)
    print(f"[+] hình    : {out_png}")
    print(f"[+] số liệu : {out_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vẽ phân bố điểm dị biệt của các thuật toán")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Ma trận đặc trưng parquet")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help=f"Mặc định: {DEFAULT_MODELS}")
    parser.add_argument("--subsample", type=int, default=50_000,
                        help="Ngân sách fit cho LOF/OCSVM (Isolation Forest luôn fit toàn bộ)")
    parser.add_argument("--split-day", type=int, default=42,
                        help="train = day <= split_day; -1 để suy từ --split-ratio")
    parser.add_argument("--split-ratio", type=float, default=0.30, help="Chỉ dùng khi --split-day = -1")
    parser.add_argument("--budget-ratio", type=float, default=0.05, help="Ngân sách cảnh báo (contamination)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--linthresh", type=float, default=1.0,
                        help="Ngưỡng tuyến tính của trục symlog ở hàng ECDF (mặc định 1.0)")
    parser.add_argument("--highlight-entities", nargs="+", default=["System", "Service"],
                        metavar="TẦNG", help="Các tầng được đánh dấu riêng (mặc định: System Service)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_FIG_DIR / "score_distribution_subsample.png"))
    parser.add_argument("--stats-out", type=str, default=str(DEFAULT_STATS_OUT))
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))

