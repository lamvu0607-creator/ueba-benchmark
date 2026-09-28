# BÁO CÁO KỸ THUẬT: GIAI ĐOẠN KHỞI TẠO & SETUP HẠ TẦNG (TUẦN 0.5)

* **Dự án**: Benchmark Isolation Forest / LOF / One-Class SVM trên log xác thực Windows (4624 / 4625)
* **Chương trình**: Thực chiến AI – Xây dựng Baseline UEBA có thể tái lập
* **Môi trường thực thi**: Conda Environment `chord-app` (Python 3.10)
* **Thời điểm báo cáo**: 13/09/2026
* **Tài liệu tham chiếu**: [Tong quan de tai ueba.md](file:///d:/Github%20Repo/ueba-benchmark/docs/Tong%20quan%20de%20tai%20ueba.md)

---

## 1. TỔNG QUAN MỤC TIÊU GIAI ĐOẠN SETUP (TUẦN 0.5)

Căn cứ theo đề cương đề tài tại **Mục 3 (Kế hoạch theo tuần)**, **Mục 6.4 (Phân công theo tuần)** và **Mục 4.1 (Tiêu chí bắt buộc)**:
- **Phạm vi tuần 0.5**: Setup môi trường, cấu hình máy, khởi tạo cấu trúc thư mục, đóng băng phiên bản thư viện trong `requirements.txt`, thiết lập file cấu hình tách biệt và chuẩn bị khung dự án ban đầu.
- **Nguyên tắc phân công**: Toàn bộ mã nguồn xử lý chi tiết (làm sạch log, tính 15+ đặc trưng, cài đặt thuật toán IForest/LOF/OCSVM, sinh bất thường tổng hợp) **chưa được viết trước**, để đảm bảo đúng tiến độ và phân chia công việc độc lập giữa **Thực tập sinh A** và **Thực tập sinh B** từ Tuần 1 đến Tuần 5.

---

## 2. CÁC HẠNG MỤC HẠ TẦNG ĐÃ HOÀN THÀNH

### 2.1. Chuẩn hóa Môi trường & Quản lý Phụ thuộc
1. **Đóng băng phiên bản thư viện ([requirements.txt](file:///d:/Github%20Repo/ueba-benchmark/requirements.txt))**:
   - Cố định phiên bản các gói phần mềm lõi trong môi trường `chord-app`:
     - Math & Data: `numpy==1.26.4`, `pandas==2.2.2`, `pyarrow==15.0.2`
     - Machine Learning: `scikit-learn==1.4.2`, `joblib==1.4.2`
     - Visualization: `matplotlib==3.8.4`, `seaborn==0.13.2`
     - Testing & Quality: `pytest==8.2.0`, `pyyaml==6.0.1`, `tqdm==4.66.4`
     - Notebook: `ipykernel==6.29.4`, `jupyterlab==4.2.0`
2. **Cấu hình Package chuẩn ([pyproject.toml](file:///d:/Github%20Repo/ueba-benchmark/pyproject.toml))**:
   - Cài đặt package `ueba-benchmark` ở chế độ editable (`pip install -e .`), giúp import trực tiếp các module trong `src` từ bất kỳ thư mục nào mà không bị lỗi đường dẫn.
3. **Chính sách bảo mật Git ([.gitignore](file:///d:/Github%20Repo/ueba-benchmark/.gitignore))**:
   - Chặn tuyệt đối việc đẩy log thô (`data/raw/`), dữ liệu trung gian (`data/interim/`, `data/processed/`), cache và model checkpoints lên kho mã nguồn.

---

### 2.2. Cấu trúc Thư mục Dự án Chuẩn hóa (Clean Architecture)
```text
ueba-benchmark/
├── .gitignore                      # Chặn data, cache, model checkpoints
├── README.md                       # Tài liệu hướng dẫn thiết lập & lệnh chạy
├── requirements.txt                # Đóng băng phiên bản thư viện
├── pyproject.toml                  # Khai báo package nội bộ (pip install -e .)
├── main.py                         # Khung orchestrator điều phối pipeline (CLI)
├── configs/                        # Cấu hình tập trung (Tránh hard-code)
│   ├── system_config.yaml          # Đường dẫn, seed, danh sách tài khoản hệ thống
│   └── model_params.yaml           # Siêu tham số cho IForest, LOF, OCSVM
├── data/                           # Quản lý vòng đời dữ liệu 3 tầng (Sẵn sàng tiếp nhận log thật)
│   ├── raw/                        # Tầng 1: Log thô 4624/4625 gốc (Read-only)
│   ├── interim/                    # Tầng 2: Log sau khi làm sạch & chuẩn hóa (.parquet)
│   └── processed/                  # Tầng 3: Ma trận đặc trưng User x Day (.parquet)
├── docs/                           # Tài liệu kỹ thuật
│   ├── Tong quan de tai ueba.md    # Đề cương chi tiết đề tài thực tập
│   ├── project_structure.md        # Kiến trúc cấu trúc dự án
│   ├── pipeline_flow.md            # Luồng hoạt động hệ thống
│   └── reports/
│       └── bao_cao_giai_doan_setup.md # Báo cáo hoàn thành tuần 0.5
├── notebooks/                      # Nghiên cứu & Phân tích thăm dò
│   └── 01_eda_event_logs.ipynb     # Notebook phục vụ EDA Tuần 1
├── src/                            # Mã nguồn phân tầng (Sẵn sàng cho TTS A & TTS B thực thi)
│   ├── __init__.py
│   ├── data/
│   │   └── __init__.py             # Module làm sạch log (Phân công: TTS B - Tuần 2)
│   ├── features/
│   │   └── __init__.py             # Module trích xuất đặc trưng (Phân công: TTS A & B - Tuần 2)
│   ├── models/
│   │   └── __init__.py             # Module mô hình Anomaly Detection (Pair & TTS A/B - Tuần 3)
│   └── evaluation/
│       └── __init__.py             # Module tiêm nhãn & bộ đo (Phân công: TTS A & B - Tuần 4)
├── experiments/                    # Lưu trữ kết quả nghiên cứu
│   ├── logs/experiment_log.csv     # Bảng nhật ký thí nghiệm (đã reset sạch)
│   ├── models/                     # Thư mục lưu checkpoint trọng số mô hình
│   └── figures/                    # Thư mục lưu biểu đồ kết quả
└── tests/                          # Bộ kiểm thử tự động
    └── __init__.py                 # Sẵn sàng viết unit tests ở Tuần 2
```

---

### 2.3. Tách Biệt Tham Số & Cấu Hình Tập Trung
- [configs/system_config.yaml](file:///d:/Github%20Repo/ueba-benchmark/configs/system_config.yaml):
  - Định nghĩa đường dẫn tương đối cho các tầng data.
  - Cố định `random_seed: 42` để đảm bảo tính tái lập 100%.
  - Cấu hình danh sách tài khoản dịch vụ hệ thống cần lọc (`SYSTEM`, `LOCAL SERVICE`, `NETWORK SERVICE`, `ANONYMOUS LOGON`).
  - Khung giờ ngoài hành chính (`off_hours_start: 18`, `off_hours_end: 7`), ngày cuối tuần.
- [configs/model_params.yaml](file:///d:/Github%20Repo/ueba-benchmark/configs/model_params.yaml):
  - Mẫu siêu tham số chuẩn bị sẵn cho Isolation Forest, Local Outlier Factor (`novelty: true`), và One-Class SVM.

---

### 2.4. Trạng thái Mã nguồn & Vệ sinh Dữ liệu
1. **Mã nguồn `src/`**:
   - Đã xóa toàn bộ mã nguồn cài đặt trước hạn.
   - Các gói con (`src/data/`, `src/features/`, `src/models/`, `src/evaluation/`) hiện chỉ giữ file `__init__.py` định danh vai trò và ghi chú phân công công việc.
2. **Dữ liệu**:
   - Không chứa bất kỳ file dữ liệu giả lập (mock data) nào.
   - Thư mục [data/raw/](file:///d:/Github%20Repo/ueba-benchmark/data/raw/) trống và sẵn sàng tiếp nhận file log 4624/4625 thực tế.
3. **File điều phối [main.py](file:///d:/Github%20Repo/ueba-benchmark/main.py)**:
   - Được thiết kế dưới dạng khung CLI tối giản; khi chạy sẽ kiểm tra sự hiện diện của dữ liệu trong `data/raw/` và nhắc nhở đưa log thật vào thay vì tự động sinh dữ liệu giả.
4. **Nhật ký thí nghiệm**:
   - [experiments/logs/experiment_log.csv](file:///d:/Github%20Repo/ueba-benchmark/experiments/logs/experiment_log.csv) ở trạng thái sạch, chỉ giữ lại dòng tiêu đề cột.

---

## 3. ĐỐI CHIẾU TIÊU CHÍ HOÀN THÀNH TUẦN 0.5 (MỤC 6.4)

| Hạng mục | Người chịu trách nhiệm | Trạng thái | Ghi chú |
| :--- | :---: | :---: | :--- |
| Setup máy & môi trường Conda `chord-app` | TTS A & B | **HOÀN THÀNH** | Python 3.10, đầy đủ thư viện yêu cầu |
| Đóng băng `requirements.txt` cố định version | TTS A (Release Eng.) | **HOÀN THÀNH** | Khớp với file yêu cầu của đề tài |
| Khởi tạo repository & cấu trúc thư mục sạch | TTS A | **HOÀN THÀNH** | Đúng chuẩn module hóa, không chứa code sớm |
| Khung nhật ký thí nghiệm `experiment_log.csv` | TTS B | **HOÀN THÀNH** | Sẵn sàng lưu trữ metric |
| Thiết lập `.gitignore` bảo mật | TTS B | **HOÀN THÀNH** | Chặn data thô và checkpoints |

---

## 4. KẾ HOẠCH BẮT ĐẦU TUẦN 1 (HIỂU DỮ LIỆU - EDA)

Theo đúng lộ trình phân công tại **Mục 6.4**:
1. **Tiếp nhận log thật**: Đưa file log Windows Security 4624 và 4625 nội bộ (đã ẩn danh) vào `data/raw/`.
2. **Phân công nhiệm vụ EDA**:
   - **TTS A (Chủ trì 4624)**: Mở notebook [notebooks/01_eda_event_logs.ipynb](file:///d:/Github%20Repo/ueba-benchmark/notebooks/01_eda_event_logs.ipynb), phân tích sự kiện đăng nhập thành công theo giờ, theo tài khoản, phân biệt tài khoản máy (`*$`) với tài khoản người dùng, lập từ điển trường dữ liệu.
   - **TTS B (Chủ trì 4625)**: Phân tích sự kiện đăng nhập thất bại theo các mã lỗi `Status` / `SubStatus`, phát hiện dữ liệu bẩn (lệch múi giờ, bản ghi trùng lặp, thiếu trường).
3. **Đầu ra cuối Tuần 1**: Hoàn thiện notebook EDA và chốt lược đồ (schema) file đặc trưng làm "hợp đồng kỹ thuật" giữa 2 bên trước khi bước sang Tuần 2.
