# Sổ Tay Hướng Dẫn Kiểm Tra & Thẩm Định Nghiệm Thu Tuần 3

> **Dự án:** UEBA Anomaly Detection Benchmark (Windows Event Logs 4624 & 4625 — LANL Dataset)  
> **Nhánh thực hiện:** `feature/model-benchmark-framework`  
> **Mục đích:** Cung cấp hướng dẫn từng bước kèm câu lệnh chạy thực tế để bạn tự kiểm chứng, đối soát độc lập toàn bộ các kết quả, mã nguồn, bài toán chống rò rỉ dữ liệu và các artifacts của **Tuần 3** theo đúng đề cương đề tài [`docs/Tong quan de tai ueba.md`](../Tong%20quan%20de%20tai%20ueba.md).

---

## 1. Chuẩn bị môi trường

Mở terminal (PowerShell trên Windows hoặc Bash trên Linux/macOS) tại thư mục gốc repository:
```powershell
cd "d:\Github Repo\ueba-benchmark"

# Kích hoạt môi trường conda của dự án
conda activate ueba-benchmark

# Thiết lập UTF-8 trên Windows để không bị lỗi mã hóa tiếng Việt
$env:PYTHONIOENCODING="utf-8"
```

Kiểm tra branch hiện tại:
```powershell
git branch --show-current
# Kết quả mong đợi: feature/model-benchmark-framework
```

---

## 2. Bước 1: Kiểm tra tính đúng đắn toàn diện của mã nguồn (Unit Tests)

Bộ kiểm thử đảm bảo mọi thành phần (cấu hình, làm sạch dữ liệu, trích xuất đặc trưng, pipeline mô hình, time-split, checkpoint) hoạt động ổn định và không có lỗi logic.

### Câu lệnh chạy:
```powershell
python -m pytest tests/ -q
# Hoặc chạy thông qua conda:
conda run -n ueba-benchmark python -m pytest tests/ -q
```

### Kết quả chuẩn cần đối chiếu:
* **Tổng số test:** `31 passed` (100% pass, khoảng 4.4 giây).
* **Phân bố các file test:**
  * `tests/test_configs.py` (2 tests): Kiểm định cấu hình hệ thống `system_config.yaml` và hợp đồng `feature_schema.yaml`.
  * `tests/test_cleaner.py` (3 tests): Kiểm định 11 cột sạch, loại bỏ cột rác `Subject*`/`Process*`, ánh xạ LogonType.
  * `tests/test_features.py` (4 tests): Kiểm định logic phân loại thực thể và trích xuất đặc trưng.
  * `tests/test_models.py` (22 tests): Kiểm định `BaseAnomalyModel`, `AnomalyPipeline`, 5 mô hình (Isolation Forest, LOF, OCSVM, Z-score, Rule-threshold), cơ chế cắt nhãn, độ lệch chuẩn MAD, time-split và lưu/nạp checkpoint `joblib`.

---

## 3. Bước 2: Kiểm tra Time-based Split & Chống rò rỉ dữ liệu (Mục 5.1 Đề cương)

Đề cương yêu cầu: **Chia tập theo chuỗi thời gian, tuyệt đối không xáo trộn ngẫu nhiên; imputer/scaler chỉ được fit trên Train**.

### 2.1. Kiểm tra ranh giới phân chia ngày:
```powershell
python scripts/diagnostics/day_split_table.py
```
**Kết quả hiển thị đối chiếu:**
* Tổng dữ liệu: **1.055.283 dòng** qua 60 ngày.
* Điểm chia: `split_day = 42`.
  * **Tập Train (Ngày 1 đến Ngày 42):** `721.612 dòng` (chiếm **68,38%** — khớp tỷ lệ ~70% của đề cương).
  * **Tập Test (Ngày 43 đến Ngày 60):** `333.671 dòng` (chiếm **31,62%**).

### 2.2. Kiểm tra phương sai & vấn đề MAD = 0 của đặc trưng:
```powershell
python scripts/diagnostics/feature_variance_check.py
```
**Ý nghĩa kiểm tra:**
* Xác nhận cả **16 đặc trưng core** đều có dữ liệu hợp lệ trên tập train (0 cột hằng số).
* Xác nhận thực tế có **7/16 đặc trưng có MAD = 0** do tỷ lệ zero-inflation cao (như `failure_ratio`, `failure_locked_out_share`).
* Kiểm chứng cơ chế fallback trong `ZScoreBaseline`: `MAD ➔ IQR/1.349 ➔ std ➔ 1.0` để bảo đảm điểm Z-score không bị bùng nổ vô tận.

---

## 4. Bước 3: Tái lập chạy thực nghiệm toàn bộ Pipeline Benchmark

Lệnh này sẽ thực thi pipeline trọn vẹn: chia tập tại ngày 42, fit 5 mô hình trên tập Train, chấm điểm trên 333.671 dòng Test, xuất bảng Leaderboard và ghi nhật ký thí nghiệm.

### Câu lệnh chạy:
```powershell
python main.py --stage benchmark
```
*(Thời gian chạy: khoảng 2,5 đến 3 phút tùy phần cứng máy)*

### Kết quả hiển thị và đối soát:
Xem bảng tổng kết in ra màn hình hoặc mở file [`experiments/results/benchmark_summary.csv`](../../experiments/results/benchmark_summary.csv):

| Mô hình | n_fit (Số mẫu train) | n_eval (Số mẫu test) | Tỷ lệ cảnh báo (%) | Thời gian fit (s) | Thời gian chấm điểm (s) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Isolation Forest** | 721.612 | 333.671 | ~5,93% | ~11,4s | ~2,0s |
| **One-Class SVM** | 20.000 | 333.671 | ~6,39% | ~3,4s | ~16,6s |
| **Z-score Baseline** | 721.612 | 333.671 | ~5,85% | ~2,6s | ~0,1s |
| **Local Outlier Factor** | 20.000 | 333.671 | ~9,24% | ~2,2s | ~4,9s |
| **Rule-Threshold Baseline** | 721.612 | 333.671 | ~20,35% | ~1,1s | ~0,1s |

---

## 5. Bước 4: Kiểm tra tính tái lập và các Artifacts đã xuất bản

Mở và kiểm tra các file artifact đã được sinh ra và quản lý trong thư mục `experiments/`:

### 5.1. File Manifest tái lập (`experiments/results/run_manifest.json`)
Chạy lệnh kiểm tra tính toàn vẹn dữ liệu:
```powershell
Get-Content experiments/results/run_manifest.json | Select-String -Pattern "commit_hash", "data_sha256", "python_version"
```
* **Ý nghĩa:** Chứa mã SHA-256 của file `feature_matrix_processed.parquet`, mã commit Git, phiên bản scikit-learn, polars... đảm bảo bất kỳ ai chạy lại trên máy khác cũng ra cùng một kết quả.

### 5.2. File ma trận độ đồng thuận Top-20 (`experiments/results/model_topk_overlap.csv`)
Chạy lệnh xem ma trận overlap:
```powershell
Get-Content experiments/results/model_topk_overlap.csv
```
* **Phát hiện quan trọng:** Độ trùng lặp Top-20 giữa các mô hình hầu hết bằng `0.00` (cao nhất giữa OCSVM và Z-score là `0.10`).
* **Kết luận khoa học:** Các thuật toán unsupervised không đồng thuận với nhau khi chưa có nhãn. Đây là luận cứ bắt buộc để bước sang Tuần 4 tiêm nhãn `redteam.txt`.

### 5.3. File độ ổn định đa seed (`experiments/results/model_stability.csv`)
Chạy lệnh xem độ ổn định khi đổi seed (seeds: 42, 7, 2024):
```powershell
Get-Content experiments/results/model_stability.csv
```
* **Điểm cần lưu ý:** Isolation Forest và OCSVM có tính ổn định cao. LOF khi fit trên mẫu con 20.000 có độ trùng Top-20 giữa các seed chỉ đạt `0.03` (rất nhạy cảm với mẫu train).

### 5.4. Nhật ký thí nghiệm (`experiments/logs/experiment_log.csv`)
Chạy lệnh xem 5 dòng log gần nhất:
```powershell
Get-Content experiments/logs/experiment_log.csv | Select-Object -Last 6
```
* **Kiểm tra:** Đảm bảo đủ các cột phân phối điểm số (`score_mean`, `score_std`, `score_p95`, `score_max`, `fit_time_sec`, `inference_time_sec`).

---

## 6. Bước 5: Đọc các tài liệu báo cáo kỹ thuật chuyên sâu

Toàn bộ phân tích lý thuyết, công thức toán học và bằng chứng thực nghiệm của Tuần 3 đã được biên soạn tại:
1. **Báo cáo tóm tắt:** [`reports/week3/tom_tat_tuan3.md`](../../reports/week3/tom_tat_tuan3.md) — Tổng kết 5 dòng, bảng kết quả, quyết định thiết kế đã khóa.
2. **Báo cáo chuyên sâu kèm biểu đồ:** [`reports/week3/label_free_benchmark.md`](../../reports/week3/label_free_benchmark.md) — Bằng chứng về độ lệch ngân sách, hiện tượng rời rạc của Rule-threshold, phân tích độ nhạy seed.
3. **Báo cáo đề xuất bộ đặc trưng mở rộng v3.0:** [`docs/reports/bao_cao_bo_dac_trung_v3.md`](bao_cao_bo_dac_trung_v3.md) — Phản biện cấu trúc đặc trưng mở rộng cho các giai đoạn tiếp theo.

---

## 7. Bảng Tiêu Chí Nghiệm Thu Nhanh (Checklist)

| STT | Tiêu chí nghiệm thu | Lệnh kiểm tra | Kết quả đạt yêu cầu | Trạng thái |
|:---:|:---|:---|:---|:---:|
| 1 | Unit tests toàn bộ hệ thống | `python -m pytest tests/ -q` | `31 passed` | [x] |
| 2 | Time-based split không rò rỉ dữ liệu | `python scripts/diagnostics/day_split_table.py` | Train 1–42 (721k), Test 43–60 (333k) | [x] |
| 3 | Xử lý đặc trưng MAD = 0 | `python scripts/diagnostics/feature_variance_check.py` | 7 đặc trưng MAD=0 được fallback an toàn | [x] |
| 4 | Chạy benchmark 5 mô hình đầu-cuối | `python main.py --stage benchmark` | Hoàn thành trong ~168s không lỗi | [x] |
| 5 | Artifacts tái lập đầy đủ | `Test-Path experiments/results/run_manifest.json` | `True` (có sha256 + commit) | [x] |
| 6 | Đầy đủ 3 mô hình ML + 2 baseline | Mở `experiments/results/benchmark_summary.csv` | Đủ iForest, LOF, OCSVM, Z-score, Rule-based | [x] |
| 7 | Cập nhật tài liệu README | Mở `README.md` phần mục 6 | Hiển thị bảng số liệu time-based split | [x] |
