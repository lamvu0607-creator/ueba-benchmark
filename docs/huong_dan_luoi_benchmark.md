# Lưới thí nghiệm tuần 4: 3 mô hình × cấu hình × 5 seed

Sản phẩm: bảng benchmark đầy đủ **3 mô hình ML × ≥ 3 cấu hình**, **đường PR**, **độ nhạy contamination**
và **kết quả lặp trên 5 seed kèm độ lệch chuẩn**. Cấu hình lưới: `configs/benchmark_grid.yaml`.

## Cách chạy

Chia tập (đề cương 5.1): **test = ngày 43–60** (30% thời gian, mô hình fit trên 1–42); **dev = ngày 36–42 nằm trong train**
(fit trên 1–35). Mọi thứ khối dev dùng (hồ sơ hành vi, kho khuôn, luật loại trừ) đều dựng từ ngày ≤ 35: khối `dev` trong
`configs/injection.yaml` ghi đè `common.split_day` và đường dẫn.

```bash
# 0. Ma trận gốc (schema v4.1, 41 đặc trưng) — luật loại trừ của mốc 35 đọc ma trận này
python main.py --stage features
# 1. Hồ sơ train + luật loại trừ cho khối dev (mốc 35); mốc 42 dùng data/features/train_profiles và experiments/results/baselines
python scripts/injection_profiles_summary.py --train-end-day 35 --out data/features/train_profiles_d35
python main.py --stage baselines --split-day 35          # -> experiments/results/baselines_d35/
# 2. Năm run dev (seed tiêm 20261036..40), mỗi run: inject (~2 phút) + features (~21 phút)
python main.py --stage inject --block dev --injection-seed 20261036
python main.py --stage features --events-dir data/injection_runs/dev_seed20261036/events_injected
# 3. Lưới dev -> chọn cấu hình tốt nhất
python scripts/evaluation/run_grid.py --config configs/benchmark_grid.yaml
# 4. Năm run test (seed tiêm 20261061..65, ~25 phút mỗi run), chốt cấu hình từ dev, lưới test (chạy một lần)
python main.py --stage inject --block test --injection-seed 20261061
python main.py --stage features --events-dir data/injection_runs/test_seed20261061/events_injected
python scripts/evaluation/freeze_test_grid.py --runs data/injection_runs/test_seed2026106{1,2,3,4,5}
python scripts/evaluation/run_grid.py --config configs/benchmark_grid_test.yaml
```

Toàn bộ chuỗi mất khoảng 6 giờ trên máy 24 GB RAM. Kết quả nằm ở `experiments/grid/{dev,test}/` (được commit); mọi lần
fit cũng được ghi vào `experiments/logs/experiment_log.csv` (bỏ qua bằng `--experiment-log ""`).

## Thiết kế

- **Một seed = một cặp (run tiêm, `random_state` mô hình)**: run thứ *i* dùng `model_seeds[i]`. Độ lệch chuẩn
  (mẫu, `ddof = 1`) vì vậy gồm cả biến động của dữ liệu tiêm (nạn nhân, sự kiện) lẫn của mô hình.
- **Chỉ phân khúc User** được đánh giá: nhãn dương chỉ tiêm vào User (mẫu số 1% = dòng User), nên Machine
  không có dương để tính PR-AUC.
- **Train / eval**: imputer, scaler và mô hình fit trên train (ngày ≤ mốc chia của run: 35 với dev, 42 với test);
  chấm đúng `eval_days` đóng băng trong `run_config.json`. Ngưỡng cảnh báo = phân vị (1 − contamination) của điểm
  train; cờ = điểm ≥ ngưỡng, giống `BaseAnomalyModel.predict`.
- **Cấu hình**: scaler (`robust` / `standard` / `quantile`) là một chiều của lưới cho LOF và OCSVM. Isolation
  Forest không phụ thuộc phép scale đơn điệu theo cột nên chỉ đổi siêu tham số.
- **Mốc tham chiếu**: z-score toàn cục, luật 6 điều kiện, **luật số lần thất bại ≥ 5** (ngưỡng cố định, mục 5.3), ngẫu nhiên.
- **Chọn cấu hình tốt nhất**: PR-AUC trung bình cao nhất qua 5 seed, trên khối **dev**. `freeze_test_grid.py` sinh lưới test
  từ `best_configs` của dev; test chỉ chạy một lần.
- **Độ nhạy contamination**: chạy lại cấu hình tốt nhất của mỗi mô hình với contamination 0,1–5% (mục 5.6) và 10/20%.
  IF/LOF chỉ đổi ngưỡng; OCSVM đổi cả mô hình vì `nu` = contamination.
- **P@k**: có cả bản trên toàn khối (`precision_at_k`) và bản theo ngày của mục 5.4 (`precision_at_k_per_day`).

## Vì sao scaler là một chiều của lưới

`RobustScaler` chia mỗi đặc trưng cho IQR train. Trên dữ liệu này:

- đặc trưng có IQR rất nhỏ bị phóng đại: `failure_ratio` (IQR 0,0007) lên tới 1.418 đơn vị,
  `delta_mean_share_fail_7d` tới 1.646;
- 9 đặc trưng có IQR = 0 được sklearn giữ đơn vị gốc, gồm chính các đặc trưng tách nạn nhân tốt nhất
  (`failure_locked_out_share`, `new_source_count_7d`, `days_since_last_activity`).

LOF và OCSVM dựa trên khoảng cách nên chỉ "thấy" vài đặc trưng bị phóng đại. Ví dụ trên run dev đầu tiên,
OCSVM robust có ROC-AUC 0,47 và xếp `new_workstation_burst` ngược (AUC 0,21); chỉ đổi sang `standard`
thì PR-AUC tăng 0,024 → 0,092 (bộ tiêm và cách chia trước 2026-10-09; số mới ở reports/week4/tom_tat_tuan4.md §6.2). Kết quả robust được giữ trong lưới làm bằng chứng.

## Đầu ra

| File | Nội dung |
|---|---|
| `grid_runs.csv` | Mỗi (mô hình, cấu hình, seed): PR-AUC, ROC-AUC, P@k, recall theo ngân sách, điểm vận hành tại ngưỡng, thời gian. |
| `grid_summary.csv` | Trung bình, độ lệch chuẩn, số seed của từng chỉ số. |
| `grid_scenarios.csv`, `grid_scenarios_summary.csv` | PR-AUC / ROC-AUC theo kịch bản (dương của kịch bản so với mọi âm). |
| `contamination_runs.csv`, `contamination_summary.csv` | Độ nhạy contamination của cấu hình tốt nhất. |
| `pr_curves.csv` | Toạ độ đường PR của seed đầu (không chọn seed đẹp nhất). |
| `figures/` | `grid_pr_auc`, `pr_curves`, `contamination_sensitivity`, `scenario_roc_auc` (PNG + PDF). |
| `grid_manifest.json` | Commit, thư viện, cấu hình, sha256 ma trận từng run, quy tắc chọn, cấu hình tốt nhất. |

Nhãn là nhãn tổng hợp nên mọi chỉ số là **cận trên lạc quan** (mục 5.4).
