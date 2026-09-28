# Kiến Trúc & Cấu Trúc Dự Án (Project Structure)

Tài liệu này mô tả chi tiết tổ chức thư mục, vai trò kỹ thuật của từng thành phần trong hệ thống **UEBA Benchmark** cho bài toán phát hiện bất thường từ Windows Event Log (4624 & 4625).

---

## 1. Cây thư mục tổng quan

```text
ueba-benchmark/
├── .gitignore                      # Cấu hình loại trừ file nhạy cảm, cache, data lớn
├── README.md                       # Hướng dẫn nhanh cho người dùng mới
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── pyproject.toml                  # Khai báo metadata package Python chuẩn
├── main.py                         # File thực thi điều phối toàn bộ pipeline (Entrypoint)
├── configs/                        # Cấu hình tập trung (Tránh hard-code)
│   ├── system_config.yaml          # Thiết lập đường dẫn dữ liệu 4 tầng, seed, bộ lọc tài khoản
│   ├── feature_schema.yaml         # Hợp đồng đặc trưng v2 (16 core features, transformations)
│   └── model_params.yaml           # Siêu tham số (Hyperparameters) cho từng thuật toán
├── data/                           # Quản lý vòng đời dữ liệu theo 4 tầng (Data Lakehouse)
│   ├── raw/                        # Tầng 1: Log sự kiện thô (CSV / Parquet / JSON / EVTX)
│   ├── interim/                    # Tầng 2: Log đã chuẩn hóa schema và làm sạch (.parquet)
│   ├── features/                   # Tầng 3: Ma trận đặc trưng thô Account x Day (.parquet)
│   └── processed/                  # Tầng 4: Ma trận đặc trưng đã chuẩn hóa/scaled sẵn sàng train
├── docs/                           # Tài liệu kỹ thuật chi tiết
│   ├── project_structure.md        # Cấu trúc và giải thích thành phần dự án
│   ├── feature_engineering/        # Phân tích tương quan Spearman & loại bỏ đa cộng tuyến
│   └── pipeline_flow.md            # Luồng hoạt động end-to-end kèm minh họa
├── notebooks/                      # Nghiên cứu và phân tích thăm dò (EDA)
├── src/                            # Mã nguồn lõi (Python Package)
│   ├── __init__.py
│   ├── data/                       # Tầng ingest & làm sạch dữ liệu
│   │   ├── raw_to_interim.py       # Chuyển đổi log thô sang Parquet chuẩn
│   │   ├── quality_survey.py       # Khảo sát chất lượng dữ liệu đa ngày
│   │   └── validate_interim.py     # Kiểm định dữ liệu interim
│   ├── features/                   # Tầng trích xuất & tiền xử lý đặc trưng
│   │   ├── schema.py               # Quản lý & kiểm định hợp đồng Feature Schema v2
│   │   ├── extractor.py            # Trích xuất ma trận Tài khoản x Ngày bằng Polars
│   │   └── preprocessor.py         # Chuẩn hóa log1p, xử lý NULL có kiểm soát
│   ├── models/                     # Tầng mô hình học máy (Phase 3)
│   └── evaluation/                 # Tầng đánh giá & tiêm bất thường (Phase 3)
├── scripts/                        # Scripts phân tích & công cụ bổ trợ
│   ├── convert_raw_to_interim.py
│   ├── survey_multi_days_quality.py
│   ├── validate_interim.py
│   └── feature_engineering/        # Scripts EDA phân phối, nhịp tuần, tương quan
├── reports/                        # Báo cáo kết quả phân tích & biểu đồ trực quan
├── experiments/                    # Lưu trữ kết quả nghiên cứu & mô hình
│   ├── logs/                       # Nhật ký thí nghiệm (experiment_log.csv)
│   ├── models/                     # Checkpoints trọng số mô hình (.joblib)
│   └── figures/                    # Biểu đồ đánh giá (ROC, PR Curve, Score Distribution)
└── tests/                          # Kiểm thử tự động (Unit Tests)
    ├── test_configs.py             # Test cấu hình YAML & tính hợp lệ của Feature Schema
    └── test_features.py            # Test logic phân loại thực thể & tiền xử lý đặc trưng
```

---

## 2. Giải thích chi tiết từng thành phần

### 2.1. File cấu hình & Quản lý môi trường
- **[pyproject.toml](file:///d:/Github%20Repo/ueba-benchmark/pyproject.toml)**: Định nghĩa package `ueba-benchmark` theo chuẩn PEP 621. Giúp cài đặt package nội bộ qua lệnh `pip install -e .`. Nhờ đó, bất kỳ file nào trong `notebooks/` hay `tests/` đều có thể `import src...` mà không lo lỗi đường dẫn `ModuleNotFoundError`.
- **[requirements.txt](file:///d:/Github%20Repo/ueba-benchmark/requirements.txt)**: Khai báo phiên bản thư viện cần thiết (`pandas`, `scikit-learn`, `pyarrow`, `pytest`, `pyyaml`...).
- **[.gitignore](file:///d:/Github%20Repo/ueba-benchmark/.gitignore)**: Bảo vệ an toàn dữ liệu. Không đẩy các file log thô, model checkpoints nặng hàng GB hoặc cache lên GitHub, chỉ giữ lại khung thư mục qua `.gitkeep`.

### 2.2. Thư mục `configs/` (Loại bỏ Hard-code)
- **[system_config.yaml](file:///d:/Github%20Repo/ueba-benchmark/configs/system_config.yaml)**:
  - Khai báo đường dẫn tương đối cho các tầng data.
  - Danh sách tài khoản hệ thống cần bỏ qua (`SYSTEM`, `LOCAL SERVICE`, `NETWORK SERVICE`, `ANONYMOUS LOGON`).
  - Quy ước giờ ngoài hành chính (`off_hours_start: 18`, `off_hours_end: 7`), ngày cuối tuần.
  - Tỷ lệ dữ liệu kiểm thử và ngưỡng đánh giá $K$ cho Precision@K.
- **[model_params.yaml](file:///d:/Github%20Repo/ueba-benchmark/configs/model_params.yaml)**:
  - Lưu toàn bộ siêu tham số của các thuật toán: Isolation Forest (`n_estimators`, `contamination`), LOF (`n_neighbors`, `novelty=True`), One-Class SVM (`kernel`, `nu`, `gamma`).
  - Khi cần tinh chỉnh mô hình, chỉ cần sửa file YAML này mà **không cần đụng vào code**.

### 2.3. Thư mục `data/` (Kiến trúc 3 tầng tối ưu I/O)
- **`data/raw/`**: Lưu trữ dữ liệu gốc tải về (EVTX / CSV / JSON). Dữ liệu ở đây là Read-Only, không bao giờ được ghi đè.
- **`data/interim/`**: Lưu log đã làm sạch dưới định dạng **Apache Parquet**. Giữ nguyên kiểu dữ liệu chuẩn (`datetime64`, `int`), nén nhỏ gọn gấp 4–5 lần so với CSV, tốc độ đọc nhanh gấp 10 lần.
- **`data/processed/`**: Lưu ma trận đặc trưng hành vi cấp độ **User x Day** (mỗi hàng đại diện cho hành vi của 1 người dùng trong 1 ngày cụ thể).

### 2.4. Thư mục `src/` (Trọng tâm mã nguồn)

#### A. `src/data/` - [make_dataset.py](file:///d:/Github%20Repo/ueba-benchmark/src/data/make_dataset.py)
- Chịu trách nhiệm nạp dữ liệu thô từ nhiều định dạng.
- Lọc bỏ các tài khoản dịch vụ hệ thống không mang tính chất hành vi cá nhân.
- Lọc bỏ tài khoản máy trạm kết thúc bằng ký tự `$` (ví dụ `DESKTOP-ABC$`).
- Chuẩn hóa tên cột thống nhất: `timestamp`, `user`, `event_id`, `logon_type`, `ip_address`, `workstation`.

#### B. `src/features/` - [build_features.py](file:///d:/Github%20Repo/ueba-benchmark/src/features/build_features.py)
- Trích xuất **16 đặc trưng hành vi** bao trùm các khía cạnh an toàn thông tin:
  1. Tần suất hoạt động: `total_logons`, `successful_logons`, `failed_logons`.
  2. Bất thường đăng nhập hỏng: `failed_logon_ratio`, `max_consecutive_failures`.
  3. Bất thường thời gian: `off_hours_logons`, `off_hours_ratio`, `weekend_logons`, `activity_span_hours`.
  4. Đa dạng mạng & thiết bị: `distinct_ips`, `distinct_workstations`.
  5. Phương thức đăng nhập: `distinct_logon_types`, `logon_type_2_count` (Cục bộ), `logon_type_3_count` (Qua mạng), `logon_type_10_count` (RDP), `other_logon_type_count`.

#### C. `src/models/` - [base.py](file:///d:/Github%20Repo/ueba-benchmark/src/models/base.py) & [anomaly_models.py](file:///d:/Github%20Repo/ueba-benchmark/src/models/anomaly_models.py)
- **Thiết kế hướng đối tượng (OOP) & Factory Pattern**:
  - `BaseAnomalyModel`: Lớp cơ sở trừu tượng quy định hợp đồng chung cho mọi mô hình: `fit()`, `predict()`, `get_anomaly_scores()`, `save()`, `load()`.
  - Chuẩn hóa điểm số: Trong Scikit-Learn, hàm `decision_function` của mỗi mô hình trả về giá trị khác nhau (âm/dương trái ngược nhau). Hệ thống này chuẩn hóa quy ước: **Điểm số càng cao $\rightarrow$ Nguy cơ bất thường càng lớn**.
  - `MODEL_REGISTRY` & `get_anomaly_model()`: Giúp bạn thêm bất kỳ thuật toán mới nào (Autoencoder, HBOS, EIF) vào dự án trong tương lai mà không cần sửa code điều phối.

#### D. `src/evaluation/` - [injector.py](file:///d:/Github%20Repo/ueba-benchmark/src/evaluation/injector.py) & [metrics.py](file:///d:/Github%20Repo/ueba-benchmark/src/evaluation/metrics.py)
- `injector.py`: Giải quyết bài toán log thực tế không có nhãn ground-truth bằng cách tiêm các kịch bản tấn công thực tế:
  - **Brute Force**: Tăng đột biến số lần thất bại liên tiếp, `failed_logon_ratio` xấp xỉ 100%.
  - **Off-hours Compromise**: Đăng nhập RDP lúc nửa đêm/cuối tuần từ IP bất thường.
  - **Lateral Movement**: Người dùng đột ngột quét qua hàng loạt máy trạm và IP khác nhau qua Network Logon (Type 3).
- `metrics.py`: Tính toán bộ chỉ số thực tế trong vận hành SOC:
  - **ROC-AUC & PR-AUC**: Đánh giá khả năng phân loại tổng thể trên dữ liệu mất cân bằng nghiêm trọng.
  - **Precision@K**: Tỷ lệ cảnh báo đúng trong top $K$ cảnh báo rủi ro cao nhất mỗi ngày (mô phỏng năng lực điều tra thực tế của chuyên viên bảo mật).
  - **False Alarm Rate (FPR)**: Tỷ lệ báo động giả gây mệt mỏi cảnh báo (Alert Fatigue).

### 2.5. File điều phối - [main.py](file:///d:/Github%20Repo/ueba-benchmark/main.py)
- Điểm vào duy nhất (CLI Entrypoint) kết nối toàn bộ quy trình:
  - Nạp config $\rightarrow$ Làm sạch $\rightarrow$ Trích xuất đặc trưng $\rightarrow$ Tiêm nhãn kiểm thử $\rightarrow$ Huấn luyện & Chấm điểm $\rightarrow$ Xuất Leaderboard $\rightarrow$ Ghi log thí nghiệm.
- Hỗ trợ chạy từng công đoạn độc lập qua tham số `--stage [all|clean|features|benchmark]`.

### 2.6. Thư mục `experiments/` & `tests/`
- **[experiments/logs/experiment_log.csv](file:///d:/Github%20Repo/ueba-benchmark/experiments/logs/experiment_log.csv)**: Bảng theo dõi lịch sử huấn luyện (Model, Tham số, ROC-AUC, PR-AUC, F1, Precision@K).
- **`tests/`**: Thư mục quản lý unit test, sẵn sàng viết bộ test case kiểm thử trên đặc trưng và dữ liệu thật ở Tuần 2.
