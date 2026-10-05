# UEBA Benchmark: Windows Security Event Log (4624 / 4625)

Hệ thống pipeline benchmark các mô hình phát hiện bất thường (Anomaly Detection) cho bài toán **Phân tích hành vi người dùng (UEBA)**, khai thác các sự kiện đăng nhập thành công (`EventID 4624`) và thất bại (`EventID 4625`) trên hệ điều hành Windows.

---

## 1. Cấu trúc thư mục dự án

```text
ueba-benchmark/
├── .gitignore                      # Chặn data thô/trung gian/features nặng, checkpoints, cache
├── README.md                       # Tài liệu hướng dẫn sử dụng & kiến trúc
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── pyproject.toml                  # Khai báo package src (hỗ trợ pip install -e .)
├── main.py                         # Entrypoint điều phối toàn bộ pipeline
├── configs/
│   ├── system_config.yaml          # Cấu hình đường dẫn 4 tầng data, seed, bộ lọc account
│   ├── feature_schema.yaml         # Hợp đồng đặc trưng v4 (39 core features, transforms)
│   └── model_params.yaml           # Siêu tham số cho IForest, LOF, OCSVM & Baselines
├── data/                           # Quản lý 4 tầng dữ liệu theo chuẩn Data Lakehouse
│   ├── raw/                        # Tầng 1: Log sự kiện gốc (EVTX / CSV / Parquet / JSON)
│   ├── interim/                    # Tầng 2: Log đã làm sạch & chuẩn hóa theo ngày (.parquet)
│   ├── features/                   # Tầng 3: Ma trận đặc trưng thô (Account x Day) (.parquet)
│   └── processed/                  # Tầng 4: Ma trận đặc trưng đã chuẩn hóa sẵn sàng train (.parquet)
├── src/                            # Mã nguồn lõi (Python Package)
│   ├── __init__.py
│   ├── data/                       # Tầng ingest & làm sạch dữ liệu
│   │   ├── raw_to_interim.py       # Chuyển đổi log thô sang interim Parquet
│   │   ├── quality_survey.py       # Khảo sát chất lượng dữ liệu đa ngày
│   │   └── validate_interim.py     # Kiểm định tính toàn vẹn của interim
│   ├── features/                   # Tầng trích xuất & tiền xử lý đặc trưng
│   │   ├── schema.py               # Quản lý và kiểm định hợp đồng Feature Schema v3
│   │   ├── extractor.py            # Trích xuất ma trận Tài khoản x Ngày bằng Polars
│   │   ├── history.py              # Trích xuất đặc trưng lịch sử 7 ngày (Novelty / Rolling)
│   │   └── preprocessor.py         # Chuẩn hóa log1p, xử lý NULL có kiểm soát
│   ├── models/                     # Tầng mô hình bất thường (Phase 3: IF, LOF, OCSVM, Baselines)
│   └── evaluation/                 # Tầng đánh giá time-split, metrics, manifest & logging
├── scripts/                        # Scripts phân tích chuyên sâu & công cụ bổ trợ
│   ├── convert_raw_to_interim.py
│   ├── survey_multi_days_quality.py
│   ├── validate_interim.py
│   └── feature_engineering/        # Bộ công cụ EDA, kiểm tra tương quan, đa cộng tuyến
├── notebooks/                      # Jupyter Notebooks nghiên cứu & EDA
├── docs/                           # Tài liệu kỹ thuật, kiến trúc & báo cáo đa cộng tuyến
├── reports/                        # Báo cáo kết quả khảo sát & biểu đồ trực quan
├── experiments/                    # Nhật ký thực nghiệm, model weights & metrics
└── tests/                          # Bộ kiểm thử tự động (36 Unit Tests)
    ├── test_cleaner.py             # Test chuẩn hóa dữ liệu thô và sửa lỗi destination
    ├── test_configs.py             # Test cấu hình YAML & tính hợp lệ của Feature Schema
    ├── test_features.py            # Test logic phân loại thực thể & trích xuất đặc trưng
    ├── test_models.py              # Test giao diện detector, baselines, pipeline & registry
    └── test_evaluation.py          # Test time split, chỉ số đánh giá, log & manifest
```

---

## 2. Thiết lập môi trường

Bạn có thể sử dụng môi trường ảo tiêu chuẩn (`venv`) hoặc Conda:

### Lựa chọn 1: Sử dụng Conda
Nếu sử dụng Conda, tên môi trường quy ước là `ueba-benchmark`:

```bash
# Kích hoạt môi trường conda
conda activate ueba-benchmark

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# Cài đặt src dưới dạng editable package (tránh lỗi import)
pip install -e .
```

### Lựa chọn 2: Sử dụng Python Virtualenv tiêu chuẩn (Không dùng Conda)
```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv

# Kích hoạt môi trường:
# - Trên Windows:
.venv\Scripts\activate
# - Trên Linux / macOS:
source .venv/bin/activate

# Cài đặt dependencies và editable package
pip install -r requirements.txt
pip install -e .
```

---

## 3. Chạy Pipeline điều phối (`main.py`)

Hệ thống cung cấp entrypoint tập trung `main.py` để điều phối pipeline từ làm sạch dữ liệu đến trích xuất đặc trưng và đánh giá mô hình.

### Chạy toàn bộ quy trình:

```bash
python main.py --stage all
```

> **Lưu ý**: Đảm bảo đã đặt file log Windows Security Event 4624/4625 vào thư mục `data/raw/` trước khi chạy stage `clean`. Nếu đã có dữ liệu làm sạch tại `data/interim/`, bạn có thể chạy thẳng từ stage `features`.

### Chạy theo từng giai đoạn (Stage):

```bash
# 1. Chỉ làm sạch dữ liệu thô (raw -> interim)
python main.py --stage clean

# 2. Trích xuất và chuẩn hóa 39 đặc trưng hành vi cốt lõi (interim -> features -> processed)
python main.py --stage features

# 2b. Trích xuất đặc trưng cho dải ngày cụ thể (ví dụ: Day 1 đến Day 10)
python main.py --stage features --start-day 1 --end-day 10

# 3. Chạy benchmark các mô hình bất thường (chọn mô hình hoặc chạy toàn bộ 6 mô hình)
python main.py --stage benchmark --models isolation_forest local_outlier_factor one_class_svm
# Hoặc chạy toàn bộ 6 mô hình mặc định:
python main.py --stage benchmark
```

---

## 4. Danh sách 24 Đặc trưng hành vi cốt lõi (Feature Schema v3.0)

Bộ đặc trưng được chuẩn hóa theo hợp đồng [`configs/feature_schema.yaml`](configs/feature_schema.yaml) sau quá trình phân tích tương quan Spearman và kiểm soát đa cộng tuyến (VIF). **v3.0** bổ sung 8 đặc trưng chỉ dùng dữ liệu ≤ t−1 (4 đặc trưng nhóm 9 trong-ngày + 4 đặc trưng nhóm 10 lịch sử), **đã đo trên dữ liệu thật 10 ngày (180.606 dòng): 0 cặp \|ρ\| ≥ 0,85, VIF max 7,31**; 6 ứng viên khác bị bác bỏ sau khi đo (xem mục `removed:` của schema):

| STT | Tên đặc trưng | Nhóm | Mô tả ý nghĩa an toàn thông tin |
| :---: | :--- | :--- | :--- |
| 1 | `log_total_logons` | Volume | $\log(1 + \text{tổng sự kiện trong ngày})$, phản ánh cường độ hoạt động tổng thể |
| 2 | `failure_ratio` | Failure | Tỷ lệ đăng nhập thất bại ($\text{failures} / \text{total}$), chỉ báo brute force/dò mật khẩu |
| 3 | `failure_locked_out_share` | Failure | Tỷ trọng sự kiện khóa tài khoản trong tổng số lần thất bại (tín hiệu lockout) |
| 4 | `off_hours_ratio` | Time | Tỷ lệ đăng nhập ngoài giờ hành chính (cửa sổ 18h - 7h sáng) |
| 5 | `interarrival_dt_mean` | Rhythm | Khoảng cách thời gian trung bình (giây) giữa 2 sự kiện liên tiếp của tài khoản |
| 6 | `delta_t_cv` | Rhythm | Hệ số biến thiên khoảng cách thời gian ($\text{std} / \text{mean}$), đo tính tuần hoàn/đều đặn |
| 7 | `same_second_share` | Rhythm | Tỷ lệ sự kiện diễn ra cùng giây ($\Delta t = 0$), phát hiện đăng nhập tự động dạng script/bot |
| 8 | `is_single_event` | Rhythm | Cờ nhị phân đánh dấu tài khoản chỉ có duy nhất 1 sự kiện trong ngày |
| 9 | `interactive_ratio` | Logon Type | Tỷ lệ đăng nhập tương tác cục bộ (`LogonType 2`) |
| 10 | `rare_logon_type_count_log` | Logon Type | $\log(1 + \text{số sự kiện có kiểu hiếm})$, bao gồm Remote Desktop (Type 10), RunAs (Type 9) |
| 11 | `ntlm_ratio` | Auth | Tỷ lệ sử dụng giao thức xác thực NTLM (thường tương quan cao với các bất thường) |
| 12 | `log_distinct_hosts` | Fan-out | $\log(1 + \text{số máy trạm đích duy nhất})$, phát hiện dò quét mạng (lateral movement) |
| 13 | `distinct_sources_count` | Fan-out | Số lượng máy nguồn duy nhất thực hiện đăng nhập vào tài khoản |
| 14 | `missing_source_ratio` | Context | Tỷ lệ sự kiện khuyết thông tin Source (chỉ báo chất lượng dữ liệu / nguồn ẩn danh) |
| 15 | `remote_logon_ratio` | Context | Tỷ lệ đăng nhập từ xa ($\text{Source} \neq \text{LogHost}$) |
| 16 | `custom_proc_share` | Context | Tỷ lệ đăng nhập phát sinh từ tiến trình ẩn danh / ứng dụng tùy biến |
| **17** | **`activity_peak_hour_sin`** | **Rhythm (v3.0)** | $\sin(2\pi \cdot h_{\text{peak}} / 24)$ — giờ cao điểm mã hoá chu kỳ |
| **18** | **`activity_peak_hour_cos`** | **Rhythm (v3.0)** | $\cos(2\pi \cdot h_{\text{peak}} / 24)$ — cặp với #17 (sin/cos phải đi cùng nhau) |
| **19** | **`hour_entropy`** | **Rhythm (v3.0)** | Pielou evenness $H / \ln(S)$ của phân bố sự kiện trên các khung giờ có mặt (độ trải đều nhịp trong ngày) |
| **20** | **`dst_host_entropy`** | **Fan-out (v3.0)** | Pielou evenness của phân bố sự kiện theo `LogHost` (độ tập trung/trải đều đích kết nối) |
| **21** | **`new_source_count_7d`** | **Novelty (v3.0)** | Số máy nguồn của ngày t **không thấy trong 7 ngày lịch gần nhất** (cửa sổ là NGÀY LỊCH, không phải "7 dòng trước đó" của panel thưa) |
| **22** | **`new_host_count_7d`** | **Novelty (v3.0)** | Số máy đích của ngày t không thấy trong 7 ngày lịch gần nhất |
| **23** | **`days_since_last_activity`** | **Novelty (v3.0)** | Số ngày kể từ phiên hoạt động trước đó của tài khoản (NULL ở dòng đầu tiên) — bắt tài khoản "ngủ đông" quay lại |
| **24** | **`volume_robust_z_7d`** | **Deviation (v3.0)** | z bền vững của $\ln(1+\text{volume})$ so với 7 ngày lịch trước đó (median + IQR/1.349 → std → 0; **winsorize ±10**; ngày trống = volume 0) |

> **Ghi chú về thiết kế:** Các đặc trưng dạng đếm thô (`failure_count`, `off_hours_count`, `burst_logon_count`) và các biến phụ thuộc tuyệt đối (`work_hours_ratio`) đã được loại bỏ để tránh hiện tượng đa cộng tuyến nghiêm trọng (VIF > 1000) và hiện tượng sụp đổ thứ hạng trong các thuật toán dựa trên khoảng cách. Sáu đặc trưng của v3.0 **đã được cài đặt, đo trên dữ liệu thật rồi bác bỏ**: `max_failure_streak` (ρ = 0,9973), `success_after_failure_ratio` (0,9180), `new_source_count` (0,9999 với bản `_7d`), `new_host_count` (0,9999), `source_recency` (0,9238), `source_host_pair_novelty` (0,9052) — xem mục `removed:` của schema và [`docs/reports/bao_cao_bo_dac_trung_v3.md`](docs/reports/bao_cao_bo_dac_trung_v3.md).

---

## 5. Kiểm thử tự động (Unit Tests)

Bộ kiểm thử đảm bảo tính toàn vẹn của cấu hình hệ thống, hợp đồng đặc trưng v3, giao diện mô hình
(`BaseAnomalyModel` + `AnomalyPipeline`), time-based split và các chỉ số đánh giá:

```bash
conda run -n ueba-benchmark python -m pytest tests/ -q
```
hoặc chạy qua môi trường ảo:
```bash
.venv\Scripts\python -m pytest tests/
```

> Trên Windows, `conda run` có thể lỗi khi bị pipe; khi đó gọi trực tiếp python của env:
> `& "$env:CONDA_PREFIX\python.exe" -m pytest tests/ -q`

---

## 6. Dữ liệu đã xử lý & Tải về từ Google Drive

Do tập dữ liệu 60 ngày (Windows Event 4624/4625) có dung lượng rất lớn (>22 GB) nên được lưu trữ ngoài Git trên Google Drive:

### Liên kết tải dữ liệu:
* **Gói dữ liệu 1 (Google Drive Part 1):** [Tải xuống tại đây](https://drive.google.com/file/d/1aMz0oPItXDYn7u326Pl_jTnmdqFvsCmE/view?usp=sharing)
* **Gói dữ liệu 2 (Google Drive Part 2):** [Tải xuống tại đây](https://drive.google.com/file/d/1FjXHtZAjKQxYtN2sv7OO6YMkNbqs00ek/view?usp=sharing)

### Hướng dẫn sử dụng:
1. Tải 2 gói dữ liệu từ link trên.
2. Giải nén vào thư mục `data/` trong dự án (`data/cleaned/` hoặc `data/interim/`).
3. Chạy trực tiếp benchmark:
   ```bash
   python main.py --stage benchmark
   ```

### Kết quả Benchmark label-free (Tuần 3) — chia tập theo THỜI GIAN

> ⚠️ **Số liệu cũ đã được sửa.** Bảng "1,055,283 mẫu đánh giá" trước đây là **artifact của rò rỉ dữ liệu**:
> mô hình được fit và chấm điểm trên **cùng một ma trận** 60 ngày (mọi điểm số đều "đã thấy" chính nó).
> Từ Tuần 3, benchmark dùng time-based split: **train = ngày 1–42 (721.612 dòng)**,
> **test = ngày 43–60 (333.671 dòng)**; imputer/scaler chỉ được fit trên train.

Tái lập: `python main.py --stage benchmark` (seed 42, K=20, ngân sách cảnh báo 5%).

Bảng kết quả chính thức được chạy trên ma trận **24 đặc trưng cốt lõi (Schema v3.0)** với đầy đủ **6 mô hình canonical** trong registry:

| Mô hình | n_fit | n_eval | Tỷ lệ cảnh báo (%) | Lệch ngân sách (pp) | Fit (s) | Chấm điểm (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Isolation Forest** | 721.612 | 333.671 | 5,71 | +0,71 | 12,42 | 2,19 |
| **One-Class SVM** | 20.000 | 333.671 | 5,83 | +0,83 | 5,41 | 19,76 |
| **Local Outlier Factor** | 20.000 | 333.671 | 8,09 | +3,09 | 3,26 | 5,77 |
| **Z-score Baseline** | 721.612 | 333.671 | 9,58 | +4,58 | 4,71 | 0,19 |
| **Rule-Threshold Baseline** | 721.612 | 333.671 | 20,35 | +15,35 | 1,78 | 0,16 |
| **Random Baseline** | 721.612 | 333.671 | 4,96 | −0,04 | 2,47 | 0,50 |

> **Ghi chú về Random Baseline:** Đây là mốc dưới (lower bound) sử dụng thuật toán băm ngẫu nhiên tất định theo dòng (ROC-AUC kỳ vọng ≈ 0,5, alert rate ≈ ngân sách 5%) dùng để đối chứng, kiểm tra xem các mô hình học máy và baseline luật có thực sự vượt qua mức "đoán mò" hay không.

* **Bảng xếp hạng chi tiết**: `experiments/results/benchmark_summary.csv`
* **Điểm số dị biệt** (mỗi mô hình 3 cột `_score`/`_pct`/`_anomaly`): `experiments/results/anomaly_scores.parquet`
* **Độ ổn định đa seed** (3 seed, mẫu 50.000 dòng): `experiments/results/model_stability.csv`
* **Trùng nhau Top-20 giữa các mô hình**: `experiments/results/model_topk_overlap.csv` — các mô hình gần như **không đồng thuận** (đây là lý do Tuần 4 cần nhãn thật để xếp hạng)
* **Manifest tái lập** (commit, sha256 dữ liệu, phiên bản thư viện): `experiments/results/run_manifest.json`
* **Nhật ký thí nghiệm** (tương thích 18 cột cũ + cột truy vết mới): `experiments/logs/experiment_log.csv`

Tài liệu tham khảo:
* **Từ điển dữ liệu 21 trường log LANL**: [`docs/data_dictionary.md`](docs/data_dictionary.md)
* **Báo cáo tóm tắt Tuần 3**: [`reports/week3/tom_tat_tuan3.md`](reports/week3/tom_tat_tuan3.md)
* **Phân tích chi tiết & bằng chứng**: [`reports/week3/label_free_benchmark.md`](reports/week3/label_free_benchmark.md)
* **Báo cáo bộ đặc trưng v3.0**: [`docs/reports/bao_cao_bo_dac_trung_v3.md`](docs/reports/bao_cao_bo_dac_trung_v3.md)