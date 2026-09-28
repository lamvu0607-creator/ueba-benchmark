# Kế hoạch Thực hiện Kiểm định Chất lượng Dữ liệu (Quality Check) 2 Cấp độ

> **Dự án:** UEBA Anomaly Detection Benchmark (Windows Event 4624 & 4625)  
> **Tài liệu:** Kế hoạch triển khai Quality Check (Data Quality Assurance Plan)  
> **Vị trí lưu trữ:** `docs/plans/quality_check_plan.md`

---

## 1. Bối cảnh & Mục tiêu

Sau khi chuyển đổi toàn bộ 60 ngày log raw `.bz2` sang 120 file `.parquet` trong thư mục `data/interim/`, chúng ta có khoảng **~943 triệu dòng dữ liệu**. Bước tiếp theo trong **Tuần 1 (EDA & Data Quality Check)** là đánh giá thực chất chất lượng dữ liệu, nhận diện các quy luật thời gian và thực thể để chuẩn bị cho **Tuần 2 (Feature Engineering: Ma trận User $\times$ Day)**.

Mục tiêu là xây dựng công cụ Quality Check ở **2 cấp độ bổ trợ lẫn nhau**:

```
                              KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
   [CẤP ĐỘ 1: DEEP-DIVE ĐƠN NGÀY]                              [CẤP ĐỘ 2: KHẢO SÁT VĨ MÔ ĐA NGÀY]
   Tool: Jupyter Notebook                                      Tool: Python CLI Script (Polars)
   Target: 1 ngày bất kỳ (tham số hóa)                         Target: 60 ngày (Test trước 2 ngày)
   Mục đích: Trực quan hóa, đi sâu chi tiết                    Mục đích: Xu hướng, phát hiện dị thường
   thực thể, phân bố giờ, lý do fail...                        toàn cục, nhịp điệu tuần, tự động hóa
```

---

## 2. Phản biện kỹ thuật & Điểm cần lưu ý (Critical Insights)

### 2.1. Phản biện về "Lỗi" trong dữ liệu Security Log
* **Không nhầm lẫn giữa Missing Value nghiệp vụ và Lỗi kỹ thuật:**
  * Các trường như `Status`, `ServiceName`, `Destination`, `ParentProcessName` có tỷ lệ null = 100% trong log LANL 4624/4625 (như đã quy định trong [data_dictionary.md](file:///d:/Github%20Repo/ueba-benchmark/data_dictionary.md#L469)). Nếu công cụ kiểm tra đánh dấu đây là "Lỗi" sẽ dẫn đến kết luận sai.
  * **Giải pháp:** Phải phân tách rõ ràng **Trường cốt lõi bắt buộc (Critical Fields)** (`Time`, `EventID`, `UserName`, `LogHost`, `LogonType`) với **Trường mở rộng phụ thuộc ngữ cảnh (Optional/Contextual Fields)**.
* **Đặc thù trường `Time`:**
  * Trong tập LANL, `Time` là số giây trong ngày $[1 \rightarrow 86,400]$. Nếu cố gắng parse sang `datetime.fromtimestamp(Time)` theo Unix epoch chuẩn, kết quả sẽ rơi vào ngày 01/01/1970.
  * **Giải pháp:** Giờ trong ngày được tính chính xác bằng công thức: $\text{Hour} = \lfloor \text{Time} / 3600 \rfloor \pmod{24}$.

### 2.2. Phản biện về việc dùng Polars trên 60 ngày (~11.2 GB, ~943M rows)
* **Nguy cơ tràn RAM (OOM Crash):**
  * Dù Polars rất nhanh, nhưng nếu gọi `pl.read_parquet("data/interim/*/*.parquet")` (Eager mode) gom cả 60 ngày cùng lúc, lượng RAM tiêu thụ sẽ vượt quá 25–30 GB, gây sập tiến trình trên máy cá nhân.
* **Giải pháp kiến trúc không hard-code & tiết kiệm tài nguyên:**
  * Dùng **Polars Streaming / Batch Per-Day**: Đọc và tính toán tổng hợp (aggregation) cho từng ngày, chỉ giữ lại 1 dòng thống kê (summary row) cho mỗi ngày.
  * Nhờ đó, RAM chỉ tốn **< 300MB**, cho phép chạy khảo sát 60 ngày cực kỳ mượt mà trên bất kỳ máy tính nào.

---

## 3. Thiết kế chi tiết Cấp độ 1: Interactive Single-Day Deep Dive (Notebook)

* **Tệp triển khai:** `notebooks/01_data_quality_single_day.ipynb`
* **Môi trường:** Conda `ueba-benchmark` (ipykernel, polars, seaborn, matplotlib).
* **Thiết kế tham số:** Ô code đầu tiên khai báo biến cấu hình:
  ```python
  TARGET_DAY = "day-01"  # Người dùng có thể đổi thành bất kỳ ngày nào: "day-02", "day-15", ...
  INTERIM_DIR = "../data/interim"
  CONFIG_PATH = "../configs/system_config.yaml"
  ```

### Các phân đoạn nội dung trong Notebook:
1. **Khởi tạo & Nạp dữ liệu tối ưu (Polars Fast Ingestion):**
   * Đọc `event_4624_{TARGET_DAY}.parquet` và `event_4625_{TARGET_DAY}.parquet`.
   * Chỉ chọn các cột cần thiết cho EDA để giải phóng RAM tối đa.
2. **Kiểm tra tính toàn vẹn (Integrity & Missingness Audit):**
   * Bảng tỷ lệ Missing/Null từng cột.
   * Xác nhận tính toàn vẹn của Critical Fields (`Time`, `EventID`, `UserName`).
   * Kiểm tra trùng lặp bản ghi (Duplicates).
3. **Phân tích nhịp điệu 24 giờ (Hourly Event Density & Diurnal Rhythm):**
   * Biểu đồ đường/cột thể hiện số lượng logon theo 24 giờ trong ngày.
   * Tính tỷ lệ sự kiện ngoài giờ (**Off-hours ratio**: 18:00 – 07:00 theo cấu hình YAML).
4. **Phân loại thực thể (Entity Segmentation):**
   * Tỷ lệ tài khoản máy (`.*\$`) vs tài khoản người dùng (`User.*`) vs tài khoản dịch vụ (`SYSTEM`, `LOCAL SERVICE`).
   * Top 10 tài khoản hoạt động nhiều nhất ngày.
5. **Phân tích cơ chế & loại xác thực (LogonType & Auth Package):**
   * Phân bố các loại logon: Interactive (2), Network (3), Batch (4), Service (5), RemoteDesktop (10).
   * Phân bố `AuthenticationPackage` (Negotiate, Kerberos, NTLM).
6. **Phân tích hành vi thất bại (Failed Logon Deep-dive):**
   * Tính tỷ lệ thất bại tự nhiên: $\frac{N_{4625}}{N_{4624} + N_{4625}}$.
   * Thống kê các nguyên nhân thất bại hàng đầu (`FailureReason`).
   * Danh sách các tài khoản bị fail nhiều nhất (nghi vấn dò quét mật khẩu).
7. **Bảng kết luận chất lượng dữ liệu của ngày (Daily Quality Scorecard):**
   * Tự động xuất bảng tổng kết: PASS / WARNING kèm giải thích định lượng.

---

## 4. Thiết kế chi tiết Cấp độ 2: Multi-Day Batch Quality Survey (Python CLI)

* **Tệp điều phối CLI:** `scripts/survey_multi_days_quality.py`
* **Tệp engine nghiệp vụ:** `src/data/quality_survey.py`
* **Công nghệ:** `polars` (Lazy / Per-Day Batch Aggregation), `matplotlib`/`seaborn` (xuất chart tự động).

### 4.1. Tham số dòng lệnh (CLI Arguments)
```bash
python scripts/survey_multi_days_quality.py \
    --interim-dir data/interim \
    --config configs/system_config.yaml \
    --days day-01,day-02 \    # Mặc định là 'all' (toàn bộ 60 ngày)
    --output-csv reports/quality_survey_summary.csv \
    --output-plot reports/figures/multi_day_trend.png
```

### 4.2. Các chỉ số khảo sát vĩ mô cho mỗi ngày ($i = 1 \dots N$)
Với mỗi ngày, script sử dụng Polars để tính toán các chỉ số vĩ mô sau:

| Nhóm chỉ số | Tên chỉ số | Ý nghĩa UEBA |
| :--- | :--- | :--- |
| **Volume** | `count_4624`, `count_4625` | Tổng số lượng sự kiện thành công & thất bại |
| **Failure Rate** | `failure_rate_pct` | Tỷ lệ % thất bại hàng ngày $\frac{N_{4625}}{N_{4624} + N_{4625}}$ |
| **Entity** | `distinct_users`, `distinct_computers` | Số lượng tài khoản và máy trạm hoạt động trong ngày |
| **Entity Ratio** | `machine_account_pct` | Tỷ lệ tài khoản máy tính (kết thúc bằng `$`) |
| **Temporal** | `off_hours_pct` | Tỷ lệ sự kiện diễn ra ngoài giờ (18h tối đến 7h sáng) |
| **Integrity** | `null_critical_count` | Số lượng null bất thường ở các trường bắt buộc |
| **Detection** | `is_weekend` | Tự động phát hiện ngày cuối tuần dựa trên mức sụt giảm volume |
| **Outlier Day** | `is_volume_spike` | Đánh dấu ngày có volume vượt quá ngưỡng $Q_3 + 1.5 \times IQR$ |

### 4.3. Đầu ra của Script khảo sát
1. **Console Output:** Bảng định dạng trực quan tóm tắt các ngày được khảo sát.
2. **File CSV tổng hợp (`reports/quality_survey_summary.csv`):** Dùng làm bảng tham chiếu nhanh cho cả nhóm nghiên cứu mà không cần scan lại dữ liệu gốc.
3. **Biểu đồ xu hướng (`reports/figures/multi_day_trend.png`):**
   * Subplot 1: Khối lượng sự kiện qua các ngày (đánh dấu ngày thường vs ngày cuối tuần).
   * Subplot 2: Biến động tỷ lệ thất bại (`failure_rate_pct`) theo thời gian.
   * Subplot 3: Số lượng `distinct_users` hoạt động mỗi ngày.
   * Subplot 4: Tỷ lệ hoạt động ngoài giờ (`off_hours_pct`).

---

## 5. Kế hoạch Thực thi & Nghiệm thu (Verification Strategy)

Theo đúng yêu cầu của bạn, trong phiên làm việc này tôi **chỉ cần chạy kiểm chứng thành công trên 2 ngày (`day-01` và `day-02`)**:

1. **Bước 1:** Viết module `src/data/quality_survey.py` và script `scripts/survey_multi_days_quality.py`.
2. **Bước 2:** Viết notebook `notebooks/01_data_quality_single_day.ipynb` (mặc định trỏ `day-01`, có tham số dễ đổi).
3. **Bước 3:** Chạy thử nghiệm CLI trên 2 ngày trong môi trường conda `ueba-benchmark`:
   ```bash
   python scripts/survey_multi_days_quality.py --days day-01,day-02
   ```
4. **Bước 4:** Chạy kiểm thử notebook thông qua lệnh nbconvert (hoặc chạy test ô mã nguồn) để đảm bảo không có bất kỳ lỗi Runtime/Syntax nào.
5. **Bước 5:** Bàn giao mã nguồn và hướng dẫn bạn cách tự chạy quét toàn bộ 60 ngày khi cần.
