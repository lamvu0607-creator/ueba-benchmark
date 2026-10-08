# Quy trình một lần tiêm log (run) — từ log gốc đến bảng kết quả

Tài liệu mô tả **luồng dữ liệu và các hợp đồng** của một run tiêm: đầu vào, đầu ra từng bước, lệnh chạy,
và kiểm tra bắt buộc. Cách dựng sự kiện của từng kịch bản thuộc module bộ tiêm (`operations.py`,
`scenarios.py`, `inject.py`) — **hiện chưa có**, xem `kiem_tra_trang_thai_bo_tiem.md`. Các bước khác đã chạy được.

```
data/interim (60 ngày, log gốc)
   │
   ├─(1) Hồ sơ TỪ TRAIN (ngày 1..42) ──────────────► data/features/train_profiles/
   │
   ├─(2) Danh sách tài khoản-ngày bị loại (luật ECDF) ◄─ experiments/results/baselines/
   │
   ├─(3) Bộ tiêm [CHƯA CÓ] ─► runs/<run_id>/events_injected/ + injected_events.parquet + injection_manifest.csv
   │
   ├─(4) Kiểm tra run
   ├─(5) Nhãn ──────────────────────────────────────► runs/<run_id>/labels.parquet
   ├─(6) Đặc trưng trên log đè ─────────────────────► runs/<run_id>/features/raw/, processed/
   ├─(7) Baseline + ML trên ma trận của run ────────► kết quả có nhãn
   └─(8) Oracle độ khó (tùy chọn) ──────────────────► runs/<run_id>/difficulty.parquet
```

---

## 0. Quy ước nền

| Quy ước | Giá trị | Nguồn |
|---|---|---|
| Tầng log | interim, 21 cột: `Time, EventID, LogHost, LogonType, LogonTypeDescription, UserName, DomainName, LogonID, SubjectUserName, SubjectDomainName, SubjectLogonID, Status, Source, ServiceName, Destination, AuthenticationPackage, FailureReason, ProcessName, ProcessID, ParentProcessName, ParentProcessID` | `data/interim/event_462x/` |
| "day" | số thứ tự file; sự kiện thuộc ngày d ⇔ `Time ∈ [(d−1)·86400, d·86400)` | `layout.day_window` |
| Giờ trong ngày | `(Time % 86400) // 3600`; ngoài giờ = giờ ≥ 18 hoặc ≤ 7 | extractor |
| Khoá dòng | `(DomainName chuẩn hoá, UserName, day)`; DomainName → lowercase/strip, rỗng → "Unknown" | `load_day_events` |
| Train / test | train = ngày ≤ 42; dev = 43–51; test = 52–60 | `system_config.yaml`, quyết định 2026-10-08 |
| Phân khúc | chỉ tài khoản `entity_type == "User"` làm nạn nhân | `entity_type_expr` |

**Không dùng tầng cleaned cho run tiêm:** cleaned trừ 3600s (DST) cho ngày ≥ 42, cửa sổ ngày sẽ lệch.

---

## 1. Dựng hồ sơ từ train

```
python scripts/injection_profiles_summary.py          # dựng + in tóm tắt
python scripts/injection_profiles_summary.py --load   # chỉ in lại hồ sơ đã lưu
```
- Chỉ đọc ngày 1..42. Đầu ra `data/features/train_profiles/`: `accounts`, `account_sources`,
  `account_loghosts`, `account_logon_types`, `account_gaps`, `daily_counts`, `hosts`, `lockout_episodes`,
  `network.json`.
- Kiểm tra trong tóm tắt: `days_outside_window` phải rỗng; ngưỡng khoá L (hiện = 5, khớp L của baseline luật).
- Hồ sơ dùng chung cho cả run dev và run test — không bao giờ dựng lại từ ngày > 42.

## 2. Lấy danh sách tài khoản-ngày bị loại

```
python main.py --stage baselines            # chạy trên log GỐC (không --events-dir)
```
- Dùng `experiments/results/baselines/baseline_scores.parquet`: dòng `segment = User`, `split = eval`,
  `rule_ecdf > thresholds.json[User][rule_ecdf][threshold]` là tài khoản-ngày luật đã cảnh báo ⇒ không chọn.
- Xác nhận đây là lần chạy trên log gốc: `rule_event_stats.json` có `injected_events_dir: null`.

## 3. Bộ tiêm *(chưa có — chỉ ghi hợp đồng)*

Đầu vào: hồ sơ (bước 1), danh sách loại (bước 2), `configs/injection.yaml` (khối `dev` hoặc `test`), log
interim của các ngày trong khối. Một lần chạy = một khối = một `run_id`.

Đầu ra trong `runs/<run_id>/` (đường dẫn lấy từ `RunLayout.for_run(runs_dir, run_id)`):

| File | Nội dung | Ràng buộc |
|---|---|---|
| `events_injected/event_4624/event_4624_day-NN.parquet`, `.../event_4625/...` | log đầy đủ của ngày NN sau khi tiêm | chỉ ngày của khối (> 42); ngày nào có thì đủ CẢ HAI file; đúng 21 cột và dtype interim |
| `injected_events.parquet` | các sự kiện đã thêm + `inj_id` | log chính không có cột đánh dấu |
| `injection_manifest.csv` | mỗi lần tiêm một dòng | `inj_id, scenario, DomainName, UserName, day, campaign_id, split, seed`, tham số thực tế, số sự kiện |

## 4. Kiểm tra run (bắt buộc trước bước 5)

| Kiểm tra | Cách làm |
|---|---|
| Không đụng train | `min(overlay_days(events_dir)) > 42` |
| Không vượt nửa đêm | với mỗi `inj_id`: `assert_events_within_day(events_of_inj, day)` |
| Bảng phụ khớp log | số dòng `injected_events.parquet` = (số dòng log đè − số dòng log gốc) cộng theo ngày |
| Schema | tên + dtype cột file đè = file interim cùng ngày |
| Mỗi tài khoản một lần | `labels_from_manifest` báo lỗi nếu khoá trùng |

## 5. Sinh nhãn

```python
from src.injection import RunLayout, write_run_labels
write_run_labels(RunLayout.for_run("<runs_dir>", "<run_id>"))
```
- Ra `labels.parquet`: `DomainName, UserName, day, is_anomaly=1, eval_exclude=False, scenario, campaign_id`.
- Dòng không có trong file nhãn được coi là âm tính khi đánh giá.

## 6. Tính lại đặc trưng trên log đè

```
python main.py --stage features --events-dir <runs_dir>/<run_id>/events_injected
```
- Ngày có file đè đọc từ run, ngày khác đọc `data/interim`; luôn ở tầng interim.
- Ghi `<run>/features/raw/feature_matrix_raw.parquet` và `<run>/processed/feature_matrix_processed.parquet`.
- Đặc trưng lịch sử (`*_7d`, `days_since_last_activity`, …) tự cập nhật vì ma trận được tính lại toàn dải ngày.

## 7. Đánh giá

```
python main.py --stage baselines --events-dir <run>/events_injected --labels <run>/labels.parquet
```
- Ba baseline (random, z-score, luật ECDF/ngưỡng ngoài), ngưỡng = phân vị train budget 1%.
- **Lưu ý:** đổi `common.output_dir` trong `configs/baselines.yaml` theo run, nếu không run sau đè run trước.
- Stage benchmark ML **chưa** nhận ma trận của run (đọc cứng `data/processed/...`) — cần bổ sung trước khi
  so ML với baseline trên cùng run.

## 8. Oracle độ khó (tùy chọn)

```python
from src.injection import RunLayout, score_difficulty
score_difficulty(RunLayout.for_run("<runs_dir>", "<run_id>"), oracle="none")
```
- Hiện chỉ có `NullOracle` (độ khó NULL). Oracle thật: lớp có `name`, `score(features, labels)`,
  `describe()`, đăng ký vào `ORACLES` trong `src/injection/difficulty.py`.
- Chỉ dùng để phân tầng kết quả; không dùng để fit hay chọn ngưỡng.

---

## Kỷ luật dev / test

1. Mọi chỉnh tham số kịch bản làm trên khối **dev** (ngày 43–51, seed dev).
2. Khi đã chốt tham số, chạy khối **test** (ngày 52–60, seed khác) **một lần**, báo cáo kết quả đó.
3. Không chọn tham số dựa trên kết quả của khối test.
