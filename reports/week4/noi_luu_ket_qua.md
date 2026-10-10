# Nơi lưu kết quả Tuần 4

Bản đồ thư mục cho toàn bộ đầu ra tuần 4 (bản chạy lại: dev ngày 36–42, test ngày 43–60, schema v4.1, 41 đặc trưng).
Đường dẫn tính từ gốc repo. Cột **Git** cho biết file có được commit hay chỉ nằm trên máy chạy.

## 1. Báo cáo và slide

| Đường dẫn | Nội dung | Git |
|---|---|---|
| `reports/week4/tom_tat_tuan4.md` | Báo cáo tóm tắt tuần 4 (bản đầy đủ, nguồn số liệu cho slide) | ✔ |
| `reports/week4/UEBA_Tuan4_Bo_sinh_bat_thuong.tex` | Slide Beamer: bộ sinh bất thường + benchmark (biên dịch XeLaTeX) | ✔ |
| `reports/week4/hinh/` | Hình dùng trong slide, chép từ `experiments/grid/{dev,test}/figures/` | ✔ |
| `reports/week4/kich_ban_tiem_slide.md` | Mô tả 6 kịch bản tiêm (bản markdown cho slide) | ✔ |
| `reports/week4/baselines_report.md` | Báo cáo 3 baseline train-only | ✔ |
| `docs/week4_benchmark_grid.md` | Hướng dẫn chạy lại toàn bộ chuỗi tiêm → đặc trưng → lưới | ✔ |
| `docs/week4_result_plots.md` | Giải thích các biểu đồ kết quả | ✔ |

## 2. Kết quả benchmark (lưới mô hình)

Hai thư mục cùng cấu trúc: `experiments/grid/dev/` (chọn cấu hình) và `experiments/grid/test/` (chạy một lần với cấu hình chốt). Tất cả đều được commit.

| File | Nội dung |
|---|---|
| `grid_summary.csv` | **Bảng chính:** mỗi (mô hình, cấu hình) một dòng, trung bình và độ lệch chuẩn 5 seed của PR-AUC, ROC-AUC, P@10/50/100 (toàn khối và theo ngày), recall@5%, recall top-N/ngày, tỷ lệ cảnh báo, thời gian |
| `grid_runs.csv` | Số liệu từng seed (chưa lấy trung bình) |
| `grid_scenarios_summary.csv` / `grid_scenarios.csv` | ROC-AUC theo từng kịch bản tấn công (trung bình / từng seed) |
| `contamination_summary.csv` / `contamination_runs.csv` | Quét contamination 0,1–20%: precision, recall, tỷ lệ cảnh báo |
| `pr_curves.csv` | Dữ liệu đường Precision–Recall (trung bình các seed trên lưới recall chung) |
| `grid_manifest.json` | Commit, phiên bản thư viện, cấu hình đã chạy, `best_configs` (cấu hình chốt từ dev) |
| `figures/grid_pr_auc.{png,pdf}` | PR-AUC theo cấu hình |
| `figures/pr_curves.{png,pdf}` | Đường PR ± 1 độ lệch chuẩn |
| `figures/contamination_sensitivity.{png,pdf}` | Độ nhạy contamination |
| `figures/scenario_roc_auc.{png,pdf}` | ROC-AUC theo kịch bản |

**Nhật ký thí nghiệm chung:** `experiments/logs/experiment_log.csv` (mục 5.6). Mỗi lần fit của lưới và quét contamination
là một dòng: thời điểm, commit, experiment (`grid_dev` / `grid_test`), config_id, run_id, seed, tham số, PR-AUC, ROC-AUC.

## 3. Các run tiêm

| Đường dẫn | Nội dung | Git |
|---|---|---|
| `experiments/injection_runs/dev_seed20261036` … `dev_seed20261040` | 5 run dev (ngày 36–42): `labels.parquet` (nhãn theo tài khoản/ngày, kịch bản) và `run_config.json` (seed, mốc chia, lượng tiêm, tham số) | ✔ |
| `experiments/injection_runs/test_seed20261061` … `test_seed20261065` | 5 run test (ngày 43–60), cùng cấu trúc | ✔ |
| `data/injection_runs/<run_id>/` | Dữ liệu đầy đủ của run: `injected_events.parquet` (sự kiện log đã tiêm), `injection_manifest.csv`, `processed/feature_matrix_processed.parquet` (ma trận 41 đặc trưng tính lại sau tiêm) | ✘ (gitignore, khoảng 5,4 GB) |
| `data/injection_runs/_template_cache/` | Kho khuôn sự kiện dùng chung giữa các run | ✘ |

Muốn chạy lại lưới cần `data/injection_runs/<run_id>/processed/`. Nếu không có, sinh lại bằng các lệnh trong `docs/week4_benchmark_grid.md`
(cùng seed cho ra cùng dữ liệu).

**Run cũ, không dùng trong báo cáo hiện tại:** `experiments/injection_runs/dev_seed20261043*` và
`data/injection_runs/{dev_seed2026104[3-7],test_seed2026105[2-6]}` thuộc cách chia cũ (dev 43–51 / test 52–60) và bộ tiêm trước khi sửa.
Giữ lại để đối chiếu, có thể xoá phần trong `data/` nếu thiếu dung lượng.

## 4. Đầu vào dựng từ train

| Đường dẫn | Nội dung | Git |
|---|---|---|
| `data/features/train_profiles/` | Hồ sơ hành vi train (ngày ≤ 42) cho khối test | ✘ |
| `data/features/train_profiles_d35/` | Hồ sơ hành vi train (ngày ≤ 35) cho khối dev | ✘ |
| `experiments/results/baselines/` | Baseline luật/ECDF mốc 42: điểm, ngưỡng, thống kê sự kiện (dùng để loại trừ khi tiêm khối test) | ✘ |
| `experiments/results/baselines_d35/` | Như trên, mốc 35 (khối dev) | ✘ |

## 5. Cấu hình và mã

| Đường dẫn | Nội dung |
|---|---|
| `configs/injection.yaml` | Bộ tiêm: 6 kịch bản, tỷ lệ 1%, khối dev/test (ngày, seed, mốc chia) |
| `configs/benchmark_grid.yaml` | Lưới dev: 5 run, 13 cấu hình mô hình, 4 baseline, 8 mức contamination |
| `configs/benchmark_grid_test.yaml` | Lưới test, **sinh tự động** từ `best_configs` của dev, không sửa tay |
| `configs/model_params.yaml` | Tham số mặc định mô hình và baseline (gồm `failure_count_baseline`) |
| `configs/feature_schema.yaml` | Schema v4.1, 41 đặc trưng core |
| `src/injection/` | Mã bộ tiêm |
| `src/evaluation/grid.py`, `src/evaluation/metrics.py` | Chạy lưới, tính chỉ số |
| `scripts/evaluation/run_grid.py` | Chạy một lưới: `python scripts/evaluation/run_grid.py --config configs/benchmark_grid.yaml` |
| `scripts/evaluation/freeze_test_grid.py` | Chốt cấu hình dev thành `benchmark_grid_test.yaml` |
| `scripts/evaluation/rebuild_pr_curves.py` | Dựng lại đường PR của một lưới đã chạy |
