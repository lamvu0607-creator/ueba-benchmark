# Demo UEBA Log Analyzer (local)

Demo một trang dùng ba mô hình Isolation Forest, LOF và One-Class SVM của benchmark tuần 5, đúng 22 đặc trưng của segment **User** (cấu hình `configs/benchmark_grid_test_22f.yaml`). Các loại tài khoản khác vẫn được đếm nhưng không chấm điểm.

## Chuẩn bị

```bash
python -m pip install -r requirements.txt
python main.py --stage features
python -m demo.prepare
uvicorn demo.backend.main:app --host 127.0.0.1 --port 8000
```

Mở <http://localhost:8000>. `demo.prepare` cần `data/processed/feature_matrix_processed.parquet` và kho log `data/interim/` (`event_4624/`, `event_4625/`) đủ ngày liên tục từ Day 1. Script fit ba pipeline trên User Day 1–42 theo seed 42, rồi lưu model, ngưỡng, metadata và lịch sử hiện diện ở `experiments/models/demo_22f_user/`. Nếu artifact đã tồn tại, script dừng để tránh vô tình ghi đè. API health cho biết khi artifact thiếu hoặc sai schema.

Nếu `/var/tmp` không ghi được trong môi trường chạy, dùng thư mục spill có quyền ghi:

```bash
mkdir -p .demo-tmp
POLARS_TEMP_DIR="$PWD/.demo-tmp" \
POLARS_TEMP_DIR_BASE_PATH="$PWD/.demo-tmp" \
POLARS_OOC_SPILL_DIR="$PWD/.demo-tmp" \
POLARS_POLARS_OOC_SPILL_DIR="$PWD/.demo-tmp" \
POLARS_MAX_THREADS=4 python main.py --stage features
```

Có thể xóa `.demo-tmp` sau khi feature stage hoàn tất.
Trên máy có nhiều lõi CPU, có thể chạy `PYTHON_CPU_COUNT=4 python -m demo.prepare` để giới hạn số luồng HNSW khi fit; tham số và dữ liệu train vẫn theo cấu hình đã chốt.

## File upload

Gửi một file Parquet tối đa **512 MB** qua giao diện hoặc `POST /api/analyze` với multipart field `file`. File phải có đủ 21 cột và kiểu của `INTERIM_SCHEMA` trong `src/data/raw_to_interim.py`; các cột thêm được bỏ qua. `Time` là số giây tính theo ngày LANL: `Day = Time // 86400 + 1`. Giao diện hiển thị **Day index**, không suy diễn ngày lịch. `EventID` chỉ được là 4624 hoặc 4625; file phải chứa đầy đủ log của các ngày liên tục cần phân tích, bắt đầu sau Day 42.

Ví dụ API: `curl -F "file=@batch.parquet" http://127.0.0.1:8000/api/analyze`.

| Kiểu Parquet | Cột bắt buộc |
| --- | --- |
| `int64` | `Time` |
| `int32` | `EventID`, `LogonType` |
| `string` | `LogHost`, `LogonTypeDescription`, `UserName`, `DomainName`, `LogonID`, `SubjectUserName`, `SubjectDomainName`, `SubjectLogonID`, `Status`, `Source`, `ServiceName`, `Destination`, `AuthenticationPackage`, `FailureReason`, `ProcessName`, `ProcessID`, `ParentProcessName`, `ParentProcessID` |

Server cần đủ log interim (cả 4624 và 4625) của bảy ngày ngay trước batch trong `data/interim/`; batch được trích đặc trưng bằng đúng đường tính của ma trận train (`build_account_day_matrix` với `events_dir`). File upload không được tự thêm vào kho. Vì vậy batch tiếp theo sẽ báo `missing_context` nếu chưa chuẩn bị lại kho. Mỗi request đọc và xử lý lại bảy ngày context; với dữ liệu đầy đủ, bước này có thể tốn thời gian và bộ nhớ. Server chỉ dành cho máy local; không nhận model/artifact từ upload.

## Đọc kết quả

Điểm thô của từng model có thang riêng; **cao hơn nghĩa là bất thường hơn**. Cảnh báo khi `score >= threshold`, ngưỡng lấy phân vị 95% của điểm train. Alert rate lấy số User account-day được chấm làm mẫu số. Percentile là thứ hạng điểm so với phân bố train để sắp Top 50, **không phải xác suất tấn công**. Top 50 ưu tiên số model cảnh báo, tiếp đến tổng percentile, cuối cùng là khóa ngày/domain/user. Không có metric cần nhãn như PR-AUC trong demo.

`GET /api/health` trả trạng thái model, phiên bản artifact và schema. Lỗi phân tích có HTTP status và JSON `{ "status": "error", "error": "...", "message": "..." }`. Server chỉ chạy một batch tại một thời điểm và trả 409 khi bận.
