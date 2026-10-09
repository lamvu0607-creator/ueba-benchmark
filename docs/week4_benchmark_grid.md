# Lưới thí nghiệm tuần 4: 3 mô hình × cấu hình × 5 seed

Sản phẩm: bảng benchmark đầy đủ **3 mô hình ML × ≥ 3 cấu hình**, **đường PR**, **độ nhạy contamination**
và **kết quả lặp trên 5 seed kèm độ lệch chuẩn**. Cấu hình lưới: `configs/benchmark_grid.yaml`.

## Cách chạy

```bash
# 1. Bốn run dev bổ sung (seed tiêm khác nhau), mỗi run: inject (~2 phút nhờ cache khuôn) + features (~22 phút)
python main.py --stage inject --block dev --injection-seed 20261044
python main.py --stage features --events-dir data/injection_runs/dev_seed20261044/events_injected
#    ... lặp lại với 20261045, 20261046, 20261047 (run gốc dev_seed20261043 đã có)

# 2. Lưới (mọi fit dùng lại ma trận đã có, không trích lại đặc trưng)
python scripts/evaluation/run_grid.py --config configs/benchmark_grid.yaml
```

Kết quả nằm ở `experiments/grid/dev/` (được commit).

## Thiết kế

- **Một seed = một cặp (run tiêm, `random_state` mô hình)**: run thứ *i* dùng `model_seeds[i]`. Độ lệch chuẩn
  (mẫu, `ddof = 1`) vì vậy gồm cả biến động của dữ liệu tiêm (nạn nhân, sự kiện) lẫn của mô hình.
- **Chỉ phân khúc User** được đánh giá: nhãn dương chỉ tiêm vào User (mẫu số 1% = dòng User), nên Machine
  không có dương để tính PR-AUC.
- **Train / eval**: imputer, scaler và mô hình fit trên train (ngày ≤ 42) của phân khúc; chấm đúng
  `eval_days` đóng băng trong `run_config.json` của run (dev 43–51). Ngưỡng cảnh báo = phân vị
  (1 − contamination) của điểm train; cờ = điểm ≥ ngưỡng, giống `BaseAnomalyModel.predict`.
- **Cấu hình**: scaler (`robust` / `standard` / `quantile`) là một chiều của lưới cho LOF và OCSVM. Isolation
  Forest không phụ thuộc phép scale đơn điệu theo cột nên chỉ đổi siêu tham số.
- **Chọn cấu hình tốt nhất**: PR-AUC trung bình cao nhất qua 5 seed, trên khối **dev**. Khối test chỉ chạy
  một lần với cấu hình đã chốt. Vì việc chọn dùng nhãn dev, con số dev của cấu hình được chọn là lạc quan.
- **Độ nhạy contamination**: chạy lại cấu hình tốt nhất của mỗi mô hình với contamination 0,01–0,20.
  IF/LOF chỉ đổi ngưỡng; OCSVM đổi cả mô hình vì `nu` = contamination.

## Vì sao scaler là một chiều của lưới

`RobustScaler` chia mỗi đặc trưng cho IQR train. Trên dữ liệu này:

- đặc trưng có IQR rất nhỏ bị phóng đại: `failure_ratio` (IQR 0,0007) lên tới 1.418 đơn vị,
  `delta_mean_share_fail_7d` tới 1.646;
- 9 đặc trưng có IQR = 0 được sklearn giữ đơn vị gốc, gồm chính các đặc trưng tách nạn nhân tốt nhất
  (`failure_locked_out_share`, `new_source_count_7d`, `days_since_last_activity`).

LOF và OCSVM dựa trên khoảng cách nên chỉ "thấy" vài đặc trưng bị phóng đại. Ví dụ trên run dev đầu tiên,
OCSVM robust có ROC-AUC 0,47 và xếp `new_workstation_burst` ngược (AUC 0,21); chỉ đổi sang `standard`
thì PR-AUC tăng 0,024 → 0,092. Kết quả robust được giữ trong lưới làm bằng chứng.

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
