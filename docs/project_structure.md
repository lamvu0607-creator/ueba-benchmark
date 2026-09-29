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
│   └── reports/                    # Báo cáo kỹ thuật tổng kết các giai đoạn
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
│   ├── models/                     # Tầng mô hình học máy (Phase 3 - Đang phát triển)
│   └── evaluation/                 # Tầng đánh giá & tiêm bất thường (Phase 3 - Đang phát triển)
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
- **Môi trường Python**: Bạn có thể sử dụng môi trường ảo chuẩn (`python -m venv .venv`) hoặc Conda. **Nếu sử dụng Conda, tên môi trường quy ước là `ueba-benchmark`**.
- **[pyproject.toml](file:///d:/Github%20Repo/ueba-benchmark/pyproject.toml)**: Định nghĩa package `ueba-benchmark` theo chuẩn PEP 621. Giúp cài đặt package nội bộ qua lệnh `pip install -e .`. Nhờ đó, bất kỳ script hay test nào đều có thể `import src...` mà không lo lỗi đường dẫn.
- **[requirements.txt](file:///d:/Github%20Repo/ueba-benchmark/requirements.txt)**: Khai báo phiên bản thư viện cần thiết (`polars`, `pandas`, `scikit-learn`, `scipy`, `pyarrow`, `pytest`, `pyyaml`...).
- **[.gitignore](file:///d:/Github%20Repo/ueba-benchmark/.gitignore)**: Bảo vệ an toàn dữ liệu. Chặn toàn bộ dữ liệu 4 tầng trong `data/`, các định dạng nén (`*.zip`, `*.rar`), model checkpoints và cache, chỉ giữ lại khung thư mục rỗng qua `.gitkeep`.

### 2.2. Thư mục `configs/` (Cấu hình tập trung, tránh Hard-code)
- **[system_config.yaml](file:///d:/Github%20Repo/ueba-benchmark/configs/system_config.yaml)**:
  - Khai báo đường dẫn tương đối cho 4 tầng dữ liệu (`raw`, `interim`, `features`, `processed`).
  - Danh sách tài khoản hệ thống cần phân loại/bỏ qua (`system`, `local service`, `network service`, `anonymous`, `anonymous logon`).
  - Quy ước giờ ngoài hành chính (`off_hours_start: 18`, `off_hours_end: 7`), giờ làm việc (`work_hours_start: 8`, `work_hours_end: 17`) phủ kín 24h.
  - Tỷ lệ dữ liệu kiểm thử và ngưỡng đánh giá $K$ cho Precision@K.
- **[feature_schema.yaml](file:///d:/Github%20Repo/ueba-benchmark/configs/feature_schema.yaml)**:
  - Hợp đồng đặc trưng chuẩn (Feature Contract v2.0): Định nghĩa 16 đặc trưng cốt lõi (Core Features), phương thức tiền xử lý (`log1p`, rank-Gaussian), cột nullable và danh sách các biến đã bị loại bỏ kèm lý do định lượng (VIF cao, đa cộng tuyến).
- **[model_params.yaml](file:///d:/Github%20Repo/ueba-benchmark/configs/model_params.yaml)**:
  - Lưu toàn bộ siêu tham số của các thuật toán: Isolation Forest (`n_estimators`, `contamination`), LOF (`n_neighbors`, `novelty=True`), One-Class SVM (`kernel`, `nu`, `gamma`).

### 2.3. Thư mục `data/` (Kiến trúc 4 tầng Data Lakehouse)
- **`data/raw/`**: Lưu trữ dữ liệu log sự kiện gốc (EVTX / CSV / JSON / BZ2). Dữ liệu ở đây là Read-Only.
- **`data/interim/`**: Lưu log đã làm sạch và chuẩn hóa schema theo ngày dưới định dạng **Apache Parquet** (`event_4624_day-XX.parquet`, `event_4625_day-XX.parquet`).
- **`data/features/`**: Lưu ma trận đặc trưng hành vi dạng thô (Raw Account x Day matrix: `feature_matrix_raw.parquet` và `daily/`) trích xuất trực tiếp từ log interim bằng Polars.
- **`data/processed/`**: Lưu ma trận đặc trưng hoàn chỉnh đã qua chuẩn hóa (`log1p`, xử lý NULL) sẵn sàng nạp trực tiếp vào các mô hình học máy.

### 2.4. Thư mục `src/` (Trọng tâm mã nguồn)

#### A. `src/data/` (Tầng Ingest & Chuẩn hóa Log)
- **[raw_to_interim.py](file:///d:/Github%20Repo/ueba-benchmark/src/data/raw_to_interim.py)**: Chịu trách nhiệm nạp dữ liệu thô, làm sạch, chuẩn hóa tên cột thống nhất và xuất ra interim Parquet.
- **[quality_survey.py](file:///d:/Github%20Repo/ueba-benchmark/src/data/quality_survey.py)**: Khảo sát chất lượng dữ liệu đa ngày theo cơ chế streaming tránh tràn RAM.
- **[validate_interim.py](file:///d:/Github%20Repo/ueba-benchmark/src/data/validate_interim.py)**: Kiểm tra tính toàn vẹn và độ đầy đủ của dữ liệu interim trước khi trích xuất đặc trưng.

#### B. `src/features/` (Tầng Trích xuất & Tiền xử lý Đặc trưng)
- **[schema.py](file:///d:/Github%20Repo/ueba-benchmark/src/features/schema.py)**: Nạp và kiểm định ma trận đặc trưng theo hợp đồng `feature_schema.yaml`.
- **[extractor.py](file:///d:/Github%20Repo/ueba-benchmark/src/features/extractor.py)**: Trích xuất ma trận hành vi Tài khoản × Ngày bằng Polars, phân loại thực thể tường minh (`Admin`, `System`, `Machine`, `User`, `Service`, `Other`).
- **[preprocessor.py](file:///d:/Github%20Repo/ueba-benchmark/src/features/preprocessor.py)**: Thực hiện các phép biến đổi chuẩn hóa (`log1p` cho volume/counts, xử lý missing values) sang tầng `processed`.

#### C. `src/models/` (Tầng Mô hình học máy - Phase 3)
- Lớp cơ sở trừu tượng `BaseAnomalyModel` và Factory pattern hỗ trợ 3 mô hình cốt lõi: Isolation Forest, Local Outlier Factor, One-Class SVM.

#### D. `src/evaluation/` (Tầng Đánh giá Mô hình - Phase 3)
- Module tiêm bất thường giả lập (`injector.py`) và bộ tính toán chỉ số an ninh thông tin (`metrics.py`: PR-AUC, ROC-AUC, Precision@K, FPR).

### 2.5. File điều phối - [main.py](file:///d:/Github%20Repo/ueba-benchmark/main.py)
- CLI Entrypoint duy nhất điều phối toàn bộ pipeline qua tham số `--stage [all|clean|features|benchmark]`.
- Hỗ trợ truyền tham số dải ngày (`--start-day`, `--end-day`) và danh sách mô hình (`--models`).

### 2.6. Thư mục `experiments/` & `tests/`
- **`experiments/`**: Quản lý nhật ký thực nghiệm (`logs/experiment_log.csv`), checkpoints mô hình (`models/`) và biểu đồ đánh giá (`figures/`).
- **`tests/`**: Bộ kiểm thử tự động gồm [test_configs.py](file:///d:/Github%20Repo/ueba-benchmark/tests/test_configs.py) và [test_features.py](file:///d:/Github%20Repo/ueba-benchmark/tests/test_features.py), chạy qua lệnh `pytest` hoặc `conda run -n ueba-benchmark pytest`.
