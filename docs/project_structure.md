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
│   ├── feature_schema.yaml         # Hợp đồng đặc trưng v4 (39 core features, transformations)
│   └── model_params.yaml           # Siêu tham số (Hyperparameters) cho từng thuật toán
├── data/                           # Quản lý vòng đời dữ liệu theo 4 tầng (Data Lakehouse)
│   ├── raw/                        # Tầng 1: Log sự kiện thô (CSV / Parquet / JSON / EVTX)
│   ├── interim/                    # Tầng 2: Log đã chuẩn hóa schema và làm sạch (.parquet)
│   ├── features/                   # Tầng 3: Ma trận đặc trưng thô Account x Day (.parquet)
│   └── processed/                  # Tầng 4: Ma trận đặc trưng đã chuẩn hóa/scaled sẵn sàng train
├── docs/                           # Tài liệu kỹ thuật chi tiết
│   ├── project_structure.md        # Cấu trúc và giải thích thành phần dự án
│   ├── data_dictionary.md          # Từ điển giải thích ý nghĩa 21 trường log Windows LANL
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
│   │   ├── schema.py               # Quản lý & kiểm định hợp đồng Feature Schema v3
│   │   ├── extractor.py            # Trích xuất ma trận Tài khoản x Ngày bằng Polars
│   │   ├── history.py              # Trích xuất đặc trưng lịch sử 7 ngày (Novelty / Rolling)
│   │   └── preprocessor.py         # Chuẩn hóa log1p, xử lý NULL có kiểm soát
│   ├── models/                     # Tầng mô hình học máy (Phase 3: IF, LOF, OCSVM, Baselines)
│   └── evaluation/                 # Tầng đánh giá time-split, metrics, manifest & logging
├── scripts/                        # Scripts phân tích & công cụ bổ trợ
│   ├── convert_raw_to_interim.py
│   ├── survey_multi_days_quality.py
│   ├── validate_interim.py
│   └── feature_engineering/        # Scripts EDA phân phối, nhịp tuần, tương quan
├── reports/                        # Báo cáo kết quả phân tích & biểu đồ trực quan
├── experiments/                    # Lưu trữ kết quả nghiên cứu & mô hình
│   ├── logs/                       # Nhật ký thí nghiệm (experiment_log.csv)
│   ├── models/                     # Checkpoints trọng số mô hình (.joblib)
│   ├── figures/                    # Biểu đồ đánh giá (ROC, PR Curve, Score Distribution)
│   └── results/                    # Bảng xếp hạng benchmark_summary.csv, anomaly_scores.parquet
└── tests/                          # Kiểm thử tự động (36 Unit Tests)
    ├── test_cleaner.py             # Test làm sạch dữ liệu và sửa lỗi destination
    ├── test_configs.py             # Test cấu hình YAML & tính hợp lệ của Feature Schema
    ├── test_features.py            # Test logic phân loại thực thể & trích xuất đặc trưng
    ├── test_models.py              # Test giao diện detector, baselines, pipeline & registry
    └── test_evaluation.py          # Test time split, chỉ số đánh giá, log & manifest
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
  - Hợp đồng đặc trưng chuẩn (Feature Contract v4.0): Định nghĩa 39 đặc trưng cốt lõi (Core Features) — 15 đặc trưng nhóm 11 sinh từ Combinatorial Template (xem docs/reports/bao_cao_bo_dac_trung_v4.md) + 16 của v2.0 + 4 đặc trưng nhóm 9 (giờ cao điểm sin/cos, entropy khung giờ, entropy máy đích) + 4 đặc trưng nhóm 10 (novelty cửa sổ 7 ngày, ngày kể từ hoạt động cuối, độ lệch khối lượng 7 ngày) — phương thức tiền xử lý (`log1p`, rank-Gaussian), cột nullable và danh sách các biến đã bị loại bỏ kèm lý do định lượng (VIF cao, đa cộng tuyến, đo rồi bác bỏ).
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
- **[history.py](file:///d:/Github%20Repo/ueba-benchmark/src/features/history.py)**: Tính toán các đặc trưng lịch sử và độ lệch theo cửa sổ 7 ngày (`new_source_count_7d`, `new_host_count_7d`, `days_since_last_activity`, `volume_robust_z_7d`).
- **[preprocessor.py](file:///d:/Github%20Repo/ueba-benchmark/src/features/preprocessor.py)**: Thực hiện các phép biến đổi chuẩn hóa (`log1p` cho volume/counts, xử lý missing values) sang tầng `processed`.

#### C. `src/models/` (Tầng Mô hình học máy - Phase 3)
- **[base.py](file:///d:/Github%20Repo/ueba-benchmark/src/models/base.py)**: hợp đồng giao diện chung `BaseAnomalyModel` (ABC): `fit` / `score` (**CAO = DỊ BIỆT**) / `predict` (ngưỡng phân vị fit trên train) / `score_rank_pct` / `get_metadata` / `save` / `load`, kèm whitelist tham số sklearn và kiểm tra đầu vào (NaN/Inf, sai số đặc trưng).
- **[detectors.py](file:///d:/Github%20Repo/ueba-benchmark/src/models/detectors.py)**: 3 thuật toán qua giao diện PyOD, cả 3 fit toàn bộ train — **Isolation Forest** (`pyod IForest`), **Local Outlier Factor** (`HNSWLOF`, láng giềng xấp xỉ hnswlib) và **One-Class SVM** (`NystroemSGDOCSVM`, nhân RBF xấp xỉ Nystroem + SGD); hai lớp mới nằm ở `src/models/pyod_detectors.py`.
- **[baselines.py](file:///d:/Github%20Repo/ueba-benchmark/src/models/baselines.py)**: 3 baseline label-free — `ZScoreBaseline` (median + độ tán xạ MAD→IQR→std theo từng đặc trưng, agg max), `RuleThresholdBaseline` (6 luật nghiệp vụ trong `model_params.yaml`) và `RandomBaseline` (**mốc dưới**: điểm băm ngẫu nhiên tất định theo nội dung dòng, ROC-AUC kỳ vọng 0,5).
- **[pipeline.py](file:///d:/Github%20Repo/ueba-benchmark/src/models/pipeline.py)**: `AnomalyPipeline` = **39 đặc trưng core → imputer median → RobustScaler → mô hình**; imputer/scaler chỉ fit trên train (không rò rỉ), baseline luật cố tình không bị scale để giữ ngữ nghĩa ngưỡng.
- **[registry.py](file:///d:/Github%20Repo/ueba-benchmark/src/models/registry.py)**: registry 6 mô hình (tên canonical snake_case + alias CamelCase của log cũ) và factory `create_model` / `create_pipeline` đọc 3 khối cấu hình `defaults` / `training` / `preprocessing`.
- **[benchmark.py](file:///d:/Github%20Repo/ueba-benchmark/src/models/benchmark.py)**: điều phối benchmark label-free — chia theo thời gian, **tách phân khúc** theo khối `segments` của `model_params.yaml` (Machine và User fit imputer/scaler/mô hình riêng; Admin/Service/System/Other bị loại khỏi benchmark ML và được log số dòng/%), chạy 6 mô hình, xuất `benchmark_summary.csv`, `anomaly_scores.parquet`, `model_topk_overlap.csv`, `model_stability.csv`, `split_info.json`, `run_manifest.json` (mọi artifact có cột `segment`; mô hình lưu ở `experiments/models/<segment>/`, run gộp ở `experiments/models/all/`) và ghi `experiment_log.csv`. `--no-segments` quay về một mô hình chung. Hàm `train_and_evaluate_model` được giữ làm shim tương thích ngược (**deprecated**: fit và chấm trên cùng tập = rò rỉ dữ liệu).

#### D. `src/evaluation/` (Tầng Đánh giá Mô hình - Phase 3)
- **[split.py](file:///d:/Github%20Repo/ueba-benchmark/src/evaluation/split.py)**: `time_split` chia train/test theo **NGÀY** (mặc định `day ≤ 42` là train) + `SplitInfo` ghi lại ranh giới và số dòng từng phần.
- **[metrics.py](file:///d:/Github%20Repo/ueba-benchmark/src/evaluation/metrics.py)**: chỉ số label-free (`alert_rate`, ngân sách cảnh báo, Top-K có tie-break ổn định, trùng nhau Top-K, Spearman, phân vị điểm, độ ổn định đa seed, đo thời gian) và chỉ số cần nhãn dùng từ Tuần 4 (`precision_at_k`, `recall_at_budget`, `roc_auc`, `average_precision`).
- **[experiment_log.py](file:///d:/Github%20Repo/ueba-benchmark/src/evaluation/experiment_log.py)**: ghi `experiments/logs/experiment_log.csv` tương thích **18 cột cũ** (tên + thứ tự) rồi tới cột truy vết mới, có chống ghi trùng theo `(model, seed, split_day, n_fit, n_eval, commit)`.
- **[manifest.py](file:///d:/Github%20Repo/ueba-benchmark/src/evaluation/manifest.py)**: `run_manifest.json` — commit/branch/dirty, `sha256` của parquet, cấu hình run, phiên bản Python + sklearn/scipy/polars/numpy/pandas/joblib.
- `injector.py` (kịch bản tiêm bất thường) **chưa có** — thuộc phạm vi Tuần 4, cùng với nhãn tấn công thật `redteam.txt`.

### 2.5. File điều phối - [main.py](file:///d:/Github%20Repo/ueba-benchmark/main.py)
- CLI Entrypoint duy nhất điều phối toàn bộ pipeline qua tham số `--stage [all|clean|features|benchmark]`.
- Hỗ trợ truyền tham số dải ngày (`--start-day`, `--end-day`) và danh sách mô hình (`--models`).

### 2.6. Thư mục `experiments/`, `scripts/` & `tests/`
- **`experiments/results/`**: artifact KẾT QUẢ **được commit vào git** (bài học Tuần 2: log bị ignore nên 8 dòng kết quả suýt mất): `benchmark_summary.csv`, `anomaly_scores.parquet`, `model_topk_overlap.csv`, `model_stability.csv`, `split_info.json`, `run_manifest.json`.
- **`experiments/logs/experiment_log.csv`**: nhật ký thí nghiệm (18 cột cũ + cột truy vết mới) — cũng được commit.
- **`experiments/models/`**: pipeline `.joblib` đã fit (bỏ qua trong git vì là nhị phân lớn).
- **`scripts/diagnostics/`**: script tạo bằng chứng cho báo cáo — `day_split_table.py` (bảng chia theo ngày), `alert_rate_calibration.py` (độ trôi alert rate), `feature_variance_check.py` (đặc trưng hằng số / MAD = 0).
- **`tests/`**: [conftest.py](file:///d:/Github%20Repo/ueba-benchmark/tests/conftest.py) (fixture thư mục tạm ghi được), [test_cleaner.py](file:///d:/Github%20Repo/ueba-benchmark/tests/test_cleaner.py) (làm sạch dữ liệu), [test_configs.py](file:///d:/Github%20Repo/ueba-benchmark/tests/test_configs.py), [test_features.py](file:///d:/Github%20Repo/ueba-benchmark/tests/test_features.py), [test_models.py](file:///d:/Github%20Repo/ueba-benchmark/tests/test_models.py) (giao diện/detector/baseline/pipeline/registry), [test_evaluation.py](file:///d:/Github%20Repo/ueba-benchmark/tests/test_evaluation.py) (split/chỉ số/log/manifest), [test_segments.py](file:///d:/Github%20Repo/ueba-benchmark/tests/test_segments.py) (benchmark phân khúc). Chạy: `pytest tests/ -q` (36 test passed).
