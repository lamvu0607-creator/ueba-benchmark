# Tuần 3 — Benchmark phát hiện dị biệt label-free (giao diện chung + time-based split)

| Hạng mục | Nội dung |
|---|---|
| Nhánh | `feature/model-benchmark-framework` |
| Dữ liệu | `data/processed/feature_matrix_processed.parquet` — 1.055.283 dòng × 23 cột, 60 ngày |
| Chia tập | train = ngày 1–42 (**721.612** dòng) / test = ngày 43–60 (**333.671** dòng, 31,62%) |
| Đặc trưng | **16 đặc trưng core** theo `configs/feature_schema.yaml` |
| Tiền xử lý | imputer median (fit train) → RobustScaler (fit train) → mô hình |
| Mô hình | 3 thuật toán sklearn + **2 baseline label-free** |
| Ngân sách cảnh báo | `contamination = 5%` (ngưỡng = phân vị `1 - contamination` trên tập train) |
| Kết quả thô | `experiments/results/` (đã commit) + `experiments/logs/experiment_log.csv` |
| Kiểm thử | **31/31 pass** (`pytest tests/ -q`) |
| Thời gian chạy đầy đủ | **168 s** cho cả 5 mô hình (kể cả 3 seed kiểm tra ổn định) |

---

## 1. Phạm vi Tuần 3

Tuần 3 **chỉ chạy label-free** (không dùng nhãn): mục tiêu là dựng đúng hạ tầng đo lường —
giao diện mô hình thống nhất, chia tập theo thời gian, chỉ số không cần nhãn, log + manifest tái lập.
Nhãn tấn công thật (`redteam.txt` của LANL) và module tiêm bất thường (`src/evaluation/injector.py`)
**thuộc Tuần 4**; các hàm chỉ số cần nhãn (`precision_at_k`, `recall_at_budget`, `roc_auc`,
`average_precision`) đã được cài đặt + kiểm thử nhưng **chưa** dùng trên dữ liệu thật.

---

## 2. Ba lỗi của bản benchmark cũ (đã sửa)

| # | Lỗi | Bằng chứng | Cách sửa ở Tuần 3 |
|---|---|---|---|
| 1 | **Rò rỉ dữ liệu**: fit và chấm điểm trên cùng một ma trận | `model.fit(X)` rồi `model.score(X)` với `X` = toàn bộ 1.055.283 dòng; mọi con số "5,00% dị biệt" chỉ là hệ quả của `contamination` | `time_split`: train/test là hai khoảng thời gian tách biệt; imputer/scaler chỉ fit trên train |
| 2 | **Sai tập đặc trưng**: dùng *mọi* cột số của parquet | `NON_FEATURE_COLS` loại 6 cột nhưng vẫn còn **18 cột số**, gồm biến thể thô trùng lặp `total_logons`, `distinct_hosts`, `rare_logon_type_count` (trùng hạng ρ = 1,0000 với bản log theo `feature_schema.yaml`) | `AnomalyPipeline` lấy đúng **16 đặc trưng core** theo schema; test `test_core_features_only` chứng minh cột ngoài core không ảnh hưởng điểm |
| 3 | **Ngưỡng cảnh báo mù**: dùng `predict()` của sklearn | One-Class SVM với `nu = 0.05` cho **~8%** cảnh báo (ngưỡng offset nội bộ khác phân vị mong muốn) | `threshold_ = quantile(score_train, 1 - contamination)`; alert rate trên test được **báo cáo trung thực** bằng cột `alert_rate_drift_pp` |

Ngoài ra 3 lỗi phụ đã sửa: (a) `fill_null(0.0)` mù quáng cho NULL hợp lệ (`failure_locked_out_share`,
`interarrival_dt_mean`, `delta_t_cv`) → nay impute median fit trên train; (b) không scale đặc trưng
(`interarrival_dt_mean` tới 86.297 giây nằm cạnh các ratio trong [0, 1]) → nay RobustScaler;
(c) khoá cấu hình chết `evaluation.*` và `system.random_seed` không được đọc → nay `main.py` đọc thật.

---

## 3. Thiết kế đã mất được phục hồi (bằng chứng)

`src/models/model_factory.py` và ~10 test của bản trước **chưa từng được commit** nên không thể lấy lại từ Git.
Thiết kế được suy ra từ 2 dấu vết còn lại và đã được **phục hồi chính xác**:

| Nguồn | Nội dung đọc được | Xác nhận |
|---|---|---|
| `.pytest_cache/v/cache/nodeids` | 12 tên test của tầng model/eval (`test_model_factory`, `test_time_based_split`, `test_evaluation_metrics`, `test_max_train_samples_subsampling`, 3× `*_pipeline`, …) | 12/12 tên này nay **đã có thật** và xanh (Phụ lục A) |
| `experiments/logs/experiment_log.csv` | **18 cột**, 8 dòng, 5 tên mô hình (`IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`, `ZScoreBaseline`, `RuleThresholdBaseline`), `train_partition_size = 721612`, `test_samples = 333671`, `train_samples = 5000/10000` | Khớp từng bước: split 721.612/333.671 (`split_info.json`), registry có đủ 5 mô hình + alias CamelCase, `max_train_samples` 20k là biến thể của 5k/10k |
| Giá trị điểm trong log cũ | `OneClassSVM` có `score_min = -2,29`, `LocalOutlierFactor` max 597,7 (dấu âm cho mẫu bình thường) | Xác nhận quy ước **`score = -decision_function`** (CAO = DỊ BIỆT) — nay là bất biến số 1 của `BaseAnomalyModel` |

> Log này từng nằm trong `.gitignore`. Từ Tuần 3, **log và mọi artifact kết quả được commit** (`.gitignore`
> đã được sửa), nên bài học "mất thiết kế vì artifact không nằm trong git" không lặp lại.

---

## 4. Kiến trúc: một giao diện, năm mô hình

`BaseAnomalyModel` (ABC) khoá 3 bất biến — mỗi bất biến đều có test canh:

1. **CAO = DỊ BIỆT**: `score(X)` trả `-decision_function` (sklearn vốn ngược dấu) → `test_score_direction`
   kiểm tra `argmax` điểm đúng là dòng dị biệt được tiêm cho **cả 3 thuật toán**.
2. **Ngân sách cảnh báo kiểm soát được**: `predict(X) = score(X) >= threshold_` với
   `threshold_ = quantile(score_train, 1 - contamination)` → alert rate của 5 mô hình so sánh được.
3. **Tái lập**: `get_metadata()` trả tên mô hình, tham số, ngưỡng, `n_fit`, phiên bản sklearn;
   `save`/`load` lưu cả pipeline (imputer + scaler + mô hình) vào một file `.joblib`.

Ngoài ra bổ sung `score_rank_pct(X, reference_scores)` — percentile hạng trong [0, 1] theo tập tham chiếu,
để so sánh **liên mô hình** dù thang đo điểm rất khác nhau (LOF max 4,4·10¹⁰ vs Z-score max 953).
Test `test_score_rank_pct_matches_score_order` kiểm tra Spearman(score, pct) = 1,0 và pct ∈ [0, 1].

### 4.1. Chia tập theo thời gian (đo lại trên dữ liệu thật)

`python scripts/diagnostics/day_split_table.py`:

| split_day | n_train | n_eval | tỷ lệ test | ghi chú |
|---:|---:|---:|---:|---|
| 41 | 710.424 | 344.859 | 32,68% | |
| **42** | **721.612** | **333.671** | **31,62%** | **mặc định — khớp đúng `train_partition_size`/`test_samples` của log cũ** |
| 43 | 741.838 | 313.445 | 29,70% | gần `test_split_ratio = 0,30` nhất |

`evaluation.split_day = 42` được chốt **tường minh** trong `configs/system_config.yaml` (kèm `test_split_ratio: 0.30`
chỉ dùng khi `split_day: null`), để không ai phải đoán lại con số này.

### 4.2. Tiền xử lý

`AnomalyPipeline`: **16 core features → SimpleImputer(median, fit train) → RobustScaler(fit train) → mô hình**.

* Baseline luật **cố tình không bị scale** vì ngưỡng luật nằm ở đơn vị gốc (`failure_ratio >= 0,5`).
* `imputer_`/`scaler_` chỉ `transform` trên test — không rò rỉ phân phối.
* Test `test_core_features_only` chứng minh: nhân `total_logons` lên 1000× hoặc thêm cột lạ **không đổi điểm**.

---

## 5. Kết quả (seed 42, 16 đặc trưng, ngân sách 5%, K = 20)

| Mô hình | n_fit | n_eval | Tỷ lệ cảnh báo | Lệch ngân sách | Fit (s) | Chấm điểm (s) | Đặc điểm |
|:---|---:|---:|---:|---:|---:|---:|---|
| **Isolation Forest** | 721.612 | 333.671 | **5,93%** | +0,93 pp | 11,43 | 1,97 | fit toàn bộ train |
| **One-Class SVM** | 20.000 | 333.671 | **6,39%** | +1,39 pp | 3,36 | 16,57 | fit mẫu con |
| **Z-score Baseline** | 721.612 | 333.671 | **5,85%** | +0,85 pp | 2,63 | 0,12 | label-free |
| **Local Outlier Factor** | 20.000 | 333.671 | **9,24%** | +4,24 pp | 2,23 | 4,91 | fit mẫu con |
| **Rule-Threshold Baseline** | 721.612 | 333.671 | **20,35%** | +15,35 pp | 1,15 | 0,09 | 6 luật thủ công |

**Cách đọc bảng này (quan trọng, tránh kết luận sai):**

* Cột "Lệch ngân sách" KHÔNG phải chỉ số chất lượng — nó là **độ trôi hiệu chuẩn**: ngưỡng được fit trên train,
  nên nếu phân phối điểm của test khác train thì tỷ lệ cảnh báo lệch khỏi 5%.
* Vì vậy bảng chỉ so sánh **chi phí và độ ổn định**, chưa nói mô hình nào "tốt" hơn: **thiếu nhãn thật**.
* So sánh công bằng giữa các mô hình trong Tuần 3 dùng **xếp hạng trong chính tập đánh giá** (ngân sách 5%
  theo phân vị ⇒ đúng 5,00% theo thiết kế; xem `budget_flags` trong `src/evaluation/metrics.py`).

---

## 6. Phát hiện (kèm bằng chứng đo được)

### 6.1. Độ trôi alert rate là hiện tượng **mẫu nhỏ**, không phải lỗi giao diện

`python scripts/diagnostics/alert_rate_calibration.py` (dữ liệu tổng hợp 16 chiều, cùng thiết kế pipeline):

| Cấu hình train | Isolation Forest | Local Outlier Factor | One-Class SVM |
|---|---:|---:|---:|
| 200 dòng | 19,1% | 10,2% | 31,7% |
| 5.000 dòng | 5,3% | 6,9% | 6,1% |

Trên dữ liệu thật cả 3 thuật toán fit ≥ 20.000 dòng nên alert rate về sát ngân sách (5,9% / 6,4% / 9,2%).
Ngoại lệ là **baseline luật (20,35%)**: điểm của nó chỉ nhận 7 giá trị rời rạc (0, 1/6, …, 1) nên ngưỡng
phân vị 95% rơi đúng vào mức 0,5 ⇒ mọi dòng thoả ≥ 3 luật đều bị cảnh báo. Đây là **hạn chế bản chất của
điểm rời rạc**, đã ghi rõ trong docstring của `RuleThresholdBaseline` và trong báo cáo thay vì che đi.

### 6.2. 7/16 đặc trưng có MAD = 0 trên tập train

`python scripts/diagnostics/feature_variance_check.py` → **0 đặc trưng hằng số** nhưng **7 đặc trưng có MAD = 0**
(hơn nửa số dòng bằng đúng median: các ratio vốn bằng 0 như `failure_ratio`, `same_second_share`,
`custom_proc_share`, `missing_source_ratio`, …). Vì vậy `ZScoreBaseline` dùng chuỗi fallback
`MAD → IQR/1,349 → std → 1,0`; nếu đặt scale = 1 cho các cột đó thì đơn vị thô của
`interarrival_dt_mean` (max 86.297 giây) sẽ lấn át toàn bộ điểm z-score.

### 6.3. Các mô hình gần như **không đồng thuận** ở Top-20

`experiments/results/model_topk_overlap.csv` (tỷ lệ trùng nhau của 20 dòng điểm cao nhất):

| | IF | LOF | OCSVM | Z-score | Rule |
|---|---:|---:|---:|---:|---:|
| **Isolation Forest** | 1,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| **Local Outlier Factor** | 0,00 | 1,00 | 0,00 | 0,00 | 0,00 |
| **One-Class SVM** | 0,00 | 0,00 | 1,00 | **0,10** | 0,00 |
| **Z-score** | 0,00 | 0,00 | 0,10 | 1,00 | 0,00 |
| **Rule-Threshold** | 0,00 | 0,00 | 0,00 | 0,00 | 1,00 |

Hệ quả trực tiếp: **không thể xếp hạng mô hình bằng chỉ số label-free**. Việc Tuần 4 chấm bằng nhãn thật
(`redteam.txt`) là bước bắt buộc, không phải tuỳ chọn.

### 6.4. Độ ổn định đa seed (3 seed: 42, 7, 2024 — mẫu 50.000 dòng của tập test, K = 20)

| Mô hình | Spearman trung bình | Spearman bé nhất | Trùng Top-20 trung bình | Trùng Top-20 bé nhất |
|---|---:|---:|---:|---:|
| Isolation Forest | 0,991 | 0,990 | 0,68 | 0,65 |
| One-Class SVM | 0,975 | 0,955 | 0,70 | 0,65 |
| Z-score Baseline | 1,000 | 1,000 | 1,00 | 1,00 |
| Rule-Threshold Baseline | 1,000 | 1,000 | 1,00 | 1,00 |
| **Local Outlier Factor** | 0,876 | 0,863 | **0,03** | **0,00** |

* Thứ hạng tổng thể khá ổn định (Spearman ≥ 0,88) nhưng **Top-20 của LOF gần như đổi hoàn toàn giữa các seed**
  (trùng 0–3%): LOF chỉ fit trên 20.000 dòng lấy ngẫu nhiên từ 721.612 dòng, nên cấu trúc mật độ cục bộ
  quanh 20 điểm cao nhất thay đổi mạnh. Kết luận thực tiễn: **không nên công bố "LOF tìm ra 20 tài khoản
  đáng ngờ nhất"** khi chưa đo lại trên toàn bộ tập train (hoặc tăng ngân sách lấy mẫu).
* Z-score và Rule cho kết quả tất định (1,000) — đúng như thiết kế (không dùng ngẫu nhiên).

### 6.5. Chi phí chạy

* Toàn bộ 5 mô hình + 3 seed kiểm tra ổn định, trên 721.612 dòng train và 333.671 dòng test: **168 s**.
* Đắt nhất là **chấm điểm** One-Class SVM (16,57 s cho 333.671 dòng × ~1.000 support vector) và fit
  Isolation Forest (11,43 s cho 150 cây trên toàn bộ train).

---

## 7. Việc còn lại cho Tuần 4 (đã khoá chữ ký hàm)

1. **Nhãn tấn công thật** từ `redteam.txt` (LANL) → khớp theo `(DomainName, UserName, day)`, gán nhãn
   0/1 cho tập test ngày 43–60.
2. **Module tiêm bất thường** `src/evaluation/injector.py` (chưa có) — khi đó mới đo được Recall@budget
   một cách có kiểm soát.
3. **Bật các chỉ số cần nhãn** đã cài sẵn: `precision_at_k`, `recall_at_budget`, `roc_auc`,
   `average_precision` (hiện chưa gọi trên dữ liệu thật).
4. **Bổ sung cột nhãn vào `anomaly_scores.parquet`** và bảng xếp hạng có `precision@K`, `recall@budget`.
5. **Cân nhắc lại ngân sách fit của LOF**: mục 6.4 cho thấy Top-20 của LOF gần như phụ thuộc seed khi
   chỉ fit trên 20.000 dòng; cần thử `max_train_samples = 100.000` hoặc fit toàn bộ để xem chênh lệch.
6. **Xử lý điểm rời rạc của baseline luật**: thêm tie-break có căn cứ (ví dụ tổng z-score) hoặc báo cáo
   riêng theo ngân sách thay vì theo ngưỡng.

---

## Phụ lục A — Đối chiếu 12 nodeid của bản benchmark đã mất

Nguồn: `.pytest_cache/v/cache/nodeids` (23 nodeid, gồm cả tầng cleaner/configs/features).

| Nodeid trong cache (bản cũ) | Trạng thái hiện tại (Tuần 3) |
|---|---|
| `tests/test_models.py::test_model_factory` | ✅ có, xanh |
| `tests/test_models.py::test_model_interfaces` | ✅ có (API chung của 5 mô hình) |
| `tests/test_models.py::test_time_based_split` | ✅ có — **đặt ở `tests/test_evaluation.py`** (tách tầng đánh giá) + test đọc parquet thật |
| `tests/test_models.py::test_evaluation_metrics` | ✅ có — đặt ở `tests/test_evaluation.py` |
| `tests/test_models.py::test_baselines` | ✅ có (+ `test_baselines_config_integration`) |
| `tests/test_models.py::test_max_train_samples_subsampling` | ✅ có |
| `tests/test_models.py::test_model_save_and_load` | ✅ có (model + roundtrip pipeline) |
| `tests/test_models.py::test_input_sanitization` | ✅ có |
| `tests/test_models.py::test_isolation_forest_pipeline` | ✅ có |
| `tests/test_models.py::test_local_outlier_factor_pipeline` | ✅ có |
| `tests/test_models.py::test_one_class_svm_pipeline` | ✅ có |
| `tests/test_features.py::test_model_interfaces` | ✅ **phục hồi** (hợp đồng feature ↔ model) |
| `tests/test_features.py::test_feature_extraction_columns_and_counts` | ✅ **phục hồi** (hợp đồng cột của extractor) |

Test **mới** của Tuần 3 (không có trong cache): `test_score_direction`, `test_params_whitelist`,
`test_core_features_only`, `test_baselines_config_integration`, `test_score_rank_pct_matches_score_order`,
`test_experiment_log_and_manifest`, `test_time_based_split_on_real_matrix`.

---

## Phụ lục B — Lệnh tái lập

```bash
# Chạy toàn bộ benchmark label-free (5 mô hình, seed 42, chia tại ngày 42)
python main.py --stage benchmark

# Chạy nhanh một/nhiều mô hình, đổi seed và ngân sách
python main.py --stage benchmark --models isolation_forest zscore_baseline --seed 7 --budget-ratio 0.10

# Suy ra ranh giới chia tập từ tỷ lệ thay vì dùng split_day cố định
python main.py --stage benchmark --split-day -1 --split-ratio 0.30

# Bằng chứng cho báo cáo
python scripts/diagnostics/day_split_table.py
python scripts/diagnostics/alert_rate_calibration.py
python scripts/diagnostics/feature_variance_check.py

# Kiểm thử
python -m pytest tests/ -q          # 31 passed
```

Tất cả artifact nằm trong `experiments/results/` và `experiments/logs/experiment_log.csv` — **đã commit vào git**
(`run_manifest.json` chứa commit hash, `sha256` của parquet và phiên bản thư viện để chứng minh tái lập).



