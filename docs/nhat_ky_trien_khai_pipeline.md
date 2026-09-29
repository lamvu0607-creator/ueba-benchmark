# Nhật Ký Triển Khai: Tự Động Hóa Pipeline Làm Sạch, Trích Xuất & Benchmark Mô Hình UEBA

> **Dự án:** UEBA Anomaly Detection Benchmark (Windows Event Logs 4624 & 4625 — LANL Unified Host Dataset)  
> **Nhánh triển khai:** `feature/data-cleaning-pipeline`  
> **Ngày thực hiện:** 29/09/2026  
> **Căn cứ kỹ thuật:** Báo cáo khảo sát khuyết thiếu dữ liệu (Day 16) & Yêu cầu tách rời khâu làm sạch  

---

## 1. Mục tiêu và Bối cảnh

### 1.1. Vấn đề trước khi thực hiện
* File trích xuất ma trận `Tài khoản × Ngày` (`extract_account_day_matrix.py` và `src/features/extractor.py`) bị **ôm đồm quá nhiều chức năng**: vừa load log, vừa lọc rác, vừa điền khuyết, vừa tính toán 16 đặc trưng hành vi.
* Khâu làm sạch chưa được tách rời độc lập, dẫn đến việc khó kiểm soát chất lượng dữ liệu sạch trước khi đưa vào mô hình hóa.
* Dự án chưa có module chạy và so sánh tự động các mô hình phát hiện dị biệt (Baseline: Isolation Forest, Local Outlier Factor, One-Class SVM).

### 1.2. Mục tiêu đạt được
1. **Tách rời 100% khâu Làm sạch dữ liệu (Data Cleaning)** thành module độc lập (`src/data/cleaner.py`), tuân thủ triệt để 5 quyết định trong file PDF khảo sát khuyết thiếu.
2. **Chuẩn hóa khâu Trích xuất đặc trưng (Feature Extraction)** (`src/features/extractor.py`), chỉ nhận dữ liệu đã làm sạch để tính toán ma trận đặc trưng theo `configs/feature_schema.yaml`.
3. **Xây dựng module Huấn luyện & Đánh giá mô hình (Model Benchmark)** (`src/models/benchmark.py`), tự động huấn luyện 3 thuật toán Baseline, lưu file model `.joblib` và bảng điểm dị biệt `anomaly_scores`.
4. **Hợp nhất bộ điều phối tự động (Pipeline Orchestrator)** trong `main.py`, cho phép chạy toàn bộ từ A đến Z hoặc chạy độc lập từng chặng.

---

## 2. Chi tiết 5 Quyết Định Làm Sạch Dữ Liệu (Theo File PDF)

| STT | Vấn đề | Quyết định xử lý | Hiện thực trong `src/data/cleaner.py` |
|:---:|:---|:---|:---|
| **1** | `LogonTypeDescription` bị thiếu 3 bản ghi (ở 4624) do lỗi tác giả thu thập | Ánh xạ 1-1 từ `LogonType` theo tài liệu chuẩn của Microsoft Security Auditing | Dùng từ điển `LOGON_TYPE_MAP` (ví dụ `2 -> Interactive`, `3 -> Network`, `5 -> Service`, `9 -> NewCredentials`...) để điền dứt điểm các dòng null/rỗng. |
| **2** | Bộ ba `Subject*` và bộ đôi `Process*` thiếu >92% - 100% | Xóa bỏ hoàn toàn các cột này khỏi dữ liệu do không đem lại giá trị hữu ích | Loại bỏ các cột: `SubjectUserName`, `SubjectDomainName`, `SubjectLogonID`, `ProcessName`, `ProcessID`, `ParentProcessName`, `ParentProcessID`, `Status`. Bảng sạch giảm từ 21 cột xuống còn **11 cột thiết yếu**. |
| **3** | Các trường `Source`, `LogonID`, `DomainName` bị khuyết thiếu | Điền trực tiếp giá trị khuyết thành nhãn `"Unknown"` | Giữ nguyên toàn bộ số dòng log và mốc thời gian, không làm mất mát dữ liệu chuỗi thời gian. Chuẩn hóa `DomainName` về chữ thường. |
| **4** | Bản ghi trùng lặp hoàn toàn (`Exact Duplicates`) | Rà soát và loại bỏ các dòng trùng nhau 100% | Sử dụng `df.unique()` trong Polars để khử trùng lặp hoàn toàn, tránh phóng đại số lượng sự kiện. |
| **5** | Lệch giờ mùa hè (Daylight Saving Time - DST) | Bù 1 tiếng cho các ngày chuyển giờ mùa hè (từ Day 42) | Cấu hình tham số `adjust_dst=True`, nếu `day >= 42` thì hiệu chỉnh `Time = Time - 3600` giây để đồng bộ toàn bộ 60 ngày về cùng trục 24h chuẩn. |

---

## 3. Kiến Trúc Pipeline 3 Chặng Tinh Gọn

```text
[Dữ liệu log gốc theo ngày] (data/interim/)
             │
             ▼
   ┌────────────────────────────────────────┐
   │        CHẶNG 1: LÀM SẠCH DỮ LIỆU       │
   │           src/data/cleaner.py          │
   └────────────────────────────────────────┘
             │
             ▼
   [Dữ liệu log SẠCH 11 cột] (data/cleaned/cleaned_day-XX.parquet)
             │
             ▼
   ┌────────────────────────────────────────┐
   │     CHẶNG 2: TRÍCH XUẤT ĐẶC TRƯNG      │
   │        src/features/extractor.py       │
   │       src/features/preprocessor.py     │
   └────────────────────────────────────────┘
             │
             ▼
   [Ma trận Tài khoản × Ngày] (data/processed/feature_matrix_processed.parquet)
             │
             ▼
   ┌────────────────────────────────────────┐
   │        CHẶNG 3: BENCHMARK MÔ HÌNH      │
   │        src/models/benchmark.py         │
   └────────────────────────────────────────┘
             │
             ▼
   [Leaderboard & Điểm Dị Biệt] (experiments/results/ & experiments/models/)
```

---

## 4. Các File Đã Xây Dựng và Chỉnh Sửa

1. **`src/data/cleaner.py` (Mới)**:
   * Chứa hàm `clean_single_day()` và `clean_dataset()`.
   * Tối ưu LazyFrame streaming, xử lý hàng triệu bản ghi trong chưa đầy 1-2 phút với Polars.
   * Lưu log sạch ra `data/cleaned/cleaned_day-{day:02d}.parquet`.

2. **`src/features/extractor.py` (Cập nhật)**:
   * Ưu tiên đọc dữ liệu sạch từ `data/cleaned/`.
   * Loại bỏ toàn bộ logic làm sạch dư thừa, chỉ tập trung tính 16 đặc trưng hành vi (khối lượng, tỷ lệ thất bại, off-hours, khoảng cách thời gian giữa các lần đăng nhập, cùng giây, ntlm, entropy máy trạm/nguồn...).
   * Vẫn giữ cơ chế fallback tự động đọc `data/interim/` nếu chưa chạy khâu clean.

3. **`src/models/benchmark.py` (Mới)**:
   * Đọc ma trận đặc trưng đã chuẩn hóa `feature_matrix_processed.parquet`.
   * Nạp siêu tham số từ `configs/model_params.yaml`.
   * Huấn luyện 3 mô hình Baseline: **Isolation Forest**, **Local Outlier Factor (LOF)**, **One-Class SVM**.
   * Xuất mô hình vào `experiments/models/{model_name}.joblib`.
   * Xuất Leaderboard vào `experiments/results/benchmark_summary.csv` và bảng chi tiết điểm dị biệt vào `experiments/results/anomaly_scores.parquet`.

4. **`main.py` (Cập nhật)**:
   * Hợp nhất 3 chặng vào 1 file điều phối duy nhất.
   * Hỗ trợ cờ `--stage`: `all`, `clean`, `features`, `benchmark`.
   * Hỗ trợ chọn dải ngày: `--start-day`, `--end-day`.

5. **`configs/system_config.yaml`**:
   * Bổ sung đường dẫn `cleaned_data_dir: "data/cleaned"` và `results_dir: "experiments/results"`.

6. **`.gitignore`**:
   * Thêm rule bỏ qua các file dữ liệu sạch nặng: `data/cleaned/*`, chỉ giữ `.gitkeep`.

7. **Bộ Unit Test mới**:
   * `tests/test_cleaner.py`: Kiểm tra bảng ánh xạ LogonType, đảm bảo không có cột `Subject*`/`Process*`, kiểm tra hàm điền `Unknown`.
   * `tests/test_models.py`: Kiểm tra quá trình huấn luyện và tính điểm dị biệt của các thuật toán.

---

## 5. Kết Quả Kiểm Thử Thực Nghiệm (Validation Results)

### 5.1. Khâu Làm Sạch (Day 01)
* **Số dòng log gốc ban đầu (4624 + 4625)**: `14,445,486` dòng (21 cột).
* **Số dòng trùng lặp hoàn toàn đã xóa**: `306,043` dòng trùng (2.12%).
* **Số dòng log sạch đầu ra**: `14,139,443` dòng (**11 cột**).
* **Dung lượng file**: `144.64 MB` (giảm đáng kể so với tổng dung lượng ban đầu).

### 5.2. Khâu Trích Xuất Đặc Trưng (Day 01)
* **Số thực thể (Tài khoản × Ngày)**: `17,910` thực thể.
* **Số cột ma trận sau chuẩn hóa**: `23` cột (4 khoá định danh + 16 đặc trưng cốt lõi + 3 cột bổ trợ log).
* **Kiểm định Schema (`FeatureSchema`)**: **HỢP LỆ 100%** (`is_valid: True`, không có missing key, không có giá trị tỷ lệ vượt ngưỡng).

### 5.3. Khâu Huấn Luyện Mô Hình & Benchmark (Thử Nghiệm Nhanh Day 01)

| Mô hình | Thuật toán | Số mẫu | Số dị biệt phát hiện | Tỷ lệ dị biệt (%) | Thời gian train (s) | File Model |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Isolation Forest** | Tree Ensemble | 17,910 | 896 | 5.00% | 2.62s | `isolation_forest.joblib` |
| **Local Outlier Factor** | Density-based | 17,910 | 796 | 4.44% | 1.08s | `local_outlier_factor.joblib` |
| **One-Class SVM** | Kernel SVM | 17,910 | 895 | 5.00% | 3.28s | `one_class_svm.joblib` |

### 5.4. Kết Quả Thực Thi Toàn Bộ 60 Ngày (Toàn Bộ Tập Dữ Liệu)
* **Quy mô ma trận (Tài khoản × Ngày)**: **1,055,283 dòng × 23 cột** (tương đương hơn 1 triệu thực thể quan trắc qua 60 ngày).
* **Kết quả Benchmark Leaderboard trên 1,055,283 thực thể**:

| Mô hình | Số mẫu đánh giá | Số dị biệt phát hiện | Tỷ lệ dị biệt (%) | Thời gian huấn luyện (s) | Vị trí lưu trữ Model |
|:---|:---:|:---:|:---:|:---:|:---|
| **Isolation Forest** | 1,055,283 | 52,765 | 5.00% | 9.55s | `experiments/models/isolation_forest.joblib` |
| **Local Outlier Factor** | 1,055,283 | 54,192 | 5.14% | 2.97s | `experiments/models/local_outlier_factor.joblib` |
| **One-Class SVM** | 1,055,283 | 49,984 | 4.74% | 9.88s | `experiments/models/one_class_svm.joblib` |

* **Đầu ra kết quả**:
  * Bảng tổng hợp Leaderboard: `experiments/results/benchmark_summary.csv`
  * Chi tiết điểm số dị biệt của 1,055,283 tài khoản: `experiments/results/anomaly_scores.parquet`

### 5.5. Kết Quả Unit Tests
* Tổng số test: **11/11 tests passed** (100% pass trong bộ kiểm thử pytest).

---

## 6. Hướng Dẫn Sử Dụng Pipeline

Đứng tại thư mục gốc dự án `D:\AITHUCCHIEN\VINSOC`, sử dụng môi trường `.venv`:

### Cách 1: Chạy tự động trọn gói từ A -> Z (Khuyên Dùng)
```bash
python main.py --stage all --start-day 1 --end-day 3
```

### Cách 2: Chạy độc lập từng chặng khi cần thử nghiệm
```bash
# Chặng 1: Chỉ làm sạch log từ interim -> cleaned
python main.py --stage clean --start-day 1 --end-day 3

# Chặng 2: Chỉ trích xuất ma trận đặc trưng từ log sạch -> features
python main.py --stage features --start-day 1 --end-day 3

# Chặng 3: Chỉ huấn luyện mô hình và xếp hạng leaderboard
python main.py --stage benchmark
```
