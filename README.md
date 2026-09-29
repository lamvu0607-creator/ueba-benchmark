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
│   ├── feature_schema.yaml         # Hợp đồng đặc trưng v2 (16 core features, transforms)
│   └── model_params.yaml           # Siêu tham số cho IForest, LOF, OCSVM
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
│   │   ├── schema.py               # Quản lý và kiểm định hợp đồng Feature Schema v2
│   │   ├── extractor.py            # Trích xuất ma trận Tài khoản x Ngày bằng Polars
│   │   └── preprocessor.py         # Chuẩn hóa log1p, xử lý NULL có kiểm soát
│   ├── models/                     # Tầng mô hình bất thường (Phase 3)
│   └── evaluation/                 # Tầng đánh giá & tiêm bất thường (Phase 3)
├── scripts/                        # Scripts phân tích chuyên sâu & công cụ bổ trợ
│   ├── convert_raw_to_interim.py
│   ├── survey_multi_days_quality.py
│   ├── validate_interim.py
│   └── feature_engineering/        # Bộ công cụ EDA, kiểm tra tương quan, đa cộng tuyến
├── notebooks/                      # Jupyter Notebooks nghiên cứu & EDA
├── docs/                           # Tài liệu kỹ thuật, kiến trúc & báo cáo đa cộng tuyến
├── reports/                        # Báo cáo kết quả khảo sát & biểu đồ trực quan
├── experiments/                    # Nhật ký thực nghiệm, model weights & metrics
└── tests/                          # Bộ kiểm thử tự động (Unit Tests)
    ├── test_configs.py             # Test cấu hình YAML & tính hợp lệ của Feature Schema
    └── test_features.py            # Test logic phân loại thực thể & tiền xử lý đặc trưng
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

# 2. Trích xuất và chuẩn hóa 16 đặc trưng hành vi (interim -> features -> processed)
python main.py --stage features

# 2b. Trích xuất đặc trưng cho dải ngày cụ thể (ví dụ: Day 1 đến Day 10)
python main.py --stage features --start-day 1 --end-day 10

# 3. Chạy benchmark các mô hình bất thường
python main.py --stage benchmark --models isolation_forest local_outlier_factor one_class_svm
```

---

## 4. Danh sách 16 Đặc trưng hành vi cốt lõi (Feature Schema v2)

Bộ đặc trưng được chuẩn hóa theo hợp đồng [`configs/feature_schema.yaml`](configs/feature_schema.yaml) sau quá trình phân tích tương quan Spearman và kiểm soát đa cộng tuyến (VIF):

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

> **Ghi chú về thiết kế:** Các đặc trưng dạng đếm thô (`failure_count`, `off_hours_count`, `burst_logon_count`) và các biến phụ thuộc tuyệt đối (`work_hours_ratio`) đã được loại bỏ để tránh hiện tượng đa cộng tuyến nghiêm trọng (VIF > 1000) và hiện tượng sụp đổ thứ bậc trong các thuật toán dựa trên khoảng cách.

---

## 5. Kiểm thử tự động (Unit Tests)

Bộ kiểm thử đảm bảo tính toàn vẹn của cấu hình hệ thống, hợp đồng đặc trưng v2 và logic phân loại thực thể:

```bash
pytest
```
hoặc chạy qua môi trường ảo:
```bash
.venv\Scripts\python -m pytest tests/
```

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

### Kết quả Benchmark Leaderboard 60 ngày (1,055,283 Tài khoản):
| Mô hình | Số mẫu đánh giá | Số dị biệt phát hiện | Tỷ lệ dị biệt (%) | Thời gian train (s) | File Model |
|:---|:---:|:---:|:---:|:---:|:---|
| **Isolation Forest** | 1,055,283 | 52,765 | 5.00% | 9.55s | `experiments/models/isolation_forest.joblib` |
| **Local Outlier Factor** | 1,055,283 | 54,192 | 5.14% | 2.97s | `experiments/models/local_outlier_factor.joblib` |
| **One-Class SVM** | 1,055,283 | 49,984 | 4.74% | 9.88s | `experiments/models/one_class_svm.joblib` |