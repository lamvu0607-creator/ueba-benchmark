# Kiểm tra trạng thái bộ tiêm log (src/injection) — 2026-10-08

Mục đích: liệt kê khâu nào của bộ tiêm đã có, khâu nào còn thiếu, và những điểm cần kiểm tra trước khi chạy
một run thật. Đối chiếu với đặc tả gốc (6 kịch bản, nhân bản sự kiện thật, chỉ tiêm test, hồ sơ chỉ từ train,
config dev/test, hook oracle) và các quyết định đã chốt (xem mục 5).

Trạng thái kiểm thử tại thời điểm viết: `python -m pytest tests -q` → **122 passed** (gồm 19 test
`test_injection_infra.py` và 18 test `test_injection_profiles.py`).

---

## 1. Tổng quan

| Khâu | File | Trạng thái | Test |
|---|---|---|---|
| Bố cục run, quy ước ngày, kiểm tra "không vượt nửa đêm" | `src/injection/layout.py` | **Có** | Có (`test_injection_infra.py`) |
| Hồ sơ TỪ TRAIN (tài khoản + toàn mạng, ngưỡng khoá L) | `src/injection/profiles.py` | **Có** (do nhóm viết) | Có (`test_injection_profiles.py`) |
| Script in thống kê hồ sơ | `scripts/injection_profiles_summary.py` | **Có**; hồ sơ đã dựng ở `data/features/train_profiles/` | — |
| Nhãn từ manifest → `labels.parquet` | `src/injection/labels.py` | **Có** | Có |
| Hook oracle độ khó (placeholder) | `src/injection/difficulty.py` | **Có** — chỉ `NullOracle` | Có |
| Stage features đọc log đè (`--events-dir`) | `src/features/extractor.py`, `main.py` | **Có** | Có (`resolve_day_source`) |
| Stage baselines đọc log đè | `src/baselines/rule_stats.py`, `runner.py` | **Có** (từ trước) | Có |
| Thao tác cơ bản + nhân bản khuôn + sinh thời gian | `src/injection/operations.py` | **THIẾU** | THIẾU |
| 6 kịch bản | `src/injection/scenarios.py` | **THIẾU** | THIẾU |
| Điều phối: đọc config, chọn nạn nhân, ghi đầu ra | `src/injection/inject.py` | **THIẾU** | THIẾU |
| Config tham số kịch bản, khối `dev` / `test` | `configs/injection.yaml` | **THIẾU** | — (chưa có trong `test_configs.py`) |
| Stage benchmark ML đọc ma trận của run | `main.py --stage benchmark` | **THIẾU** (đọc cứng `data/processed/...`) | — |

---

## 2. Các khâu đã có — chi tiết và điểm cần kiểm tra

### 2.1 `layout.py`
- `day` = số thứ tự file `event_462x_day-NN.parquet`; cửa sổ hợp lệ `[(d−1)·86400, d·86400)` ở tầng interim.
- `events_outside_day` / `assert_events_within_day`: đầu cửa sổ tính vào ngày, 00:00 hôm sau và `Time` null
  là vi phạm.
- `RunLayout`: đường dẫn mọi artifact của một run (`events_injected/`, `injected_events.parquet`,
  `injection_manifest.csv`, `labels.parquet`, `difficulty.parquet`, `features/raw/`, `processed/`,
  `results/`, `models/`).
- **Cần lưu ý:** tầng cleaned trừ 3600s (DST) cho ngày ≥ 42 ⇒ mọi bước đọc log đè phải dùng interim. Stage
  features đã tự bỏ qua `data/cleaned` khi có `--events-dir`.

### 2.2 `profiles.py` + script tóm tắt
- Đọc interim qua `load_day_events` của extractor (cùng chuẩn hoá khoá `DomainName` lowercase /
  "Unknown"), đối chiếu `Time` với `day_window` từng ngày (`days_outside_window` = `{}` trên 42 ngày train).
- Hồ sơ đã dựng: 29.830 tài khoản, 15.946 máy, ngày 1..42.
- Ngưỡng khoá: `L = 5` (mode của streak, support 33,7%, 1.861 lần khoá / 446 tài khoản), **trùng** với
  `L = 5` của baseline luật (`rule_stats.estimate_lockout_threshold`, ceil(trung vị), 673 ca) dù hai cách
  ước lượng khác nhau. Nên ghi cả hai con số vào báo cáo để người đọc thấy chúng khớp.
- **Cần kiểm tra khi viết các khâu sau:** hồ sơ có cả tài khoản Machine/Admin; việc lọc chỉ tài khoản
  `entity_type == "User"` phải làm ở bước chọn nạn nhân.

### 2.3 `labels.py`
- Manifest (`inj_id, scenario, DomainName, UserName, day[, campaign_id][, days]`) → 7 cột đúng hợp đồng
  `load_eval_labels`; khoá trùng → lỗi; cột `days` ("44;45") đã chừa cho tiêm nhiều ngày.

### 2.4 `difficulty.py`
- Interface `DifficultyOracle.score(features, labels) -> (DomainName, UserName, day, difficulty)`;
  `validate_difficulty` chặn thiếu/thừa/trùng khoá; `score_difficulty(layout)` ghi `difficulty.parquet` + `.json`.
- **Chưa có oracle thật** — `NullOracle` trả NULL. Đây là chủ đích của đặc tả, không phải lỗi.

### 2.5 Stage features / baselines với log đè
- `python main.py --stage features --events-dir <run>/events_injected` → ma trận ghi vào
  `<run>/features/raw/` và `<run>/processed/`, không đè `data/`.
- `python main.py --stage baselines --events-dir ... --labels <run>/labels.parquet` → R2/R4/R6 tính lại
  trên log đè, cache riêng trong thư mục kết quả.
- **Lưu ý:** stage baselines mặc định ghi vào `experiments/results/baselines` (theo `configs/baselines.yaml`).
  Chạy nhiều run cần đổi `common.output_dir`, nếu không kết quả run sau đè run trước.

---

## 3. Các khâu còn thiếu

### 3.1 `operations.py` — thao tác cơ bản *(thiếu)*
Theo đặc tả cần có: hàm nhân bản khuôn từ sự kiện train, các thao tác thêm/dời/sửa trường, và hàm sinh thời
gian nhận một "profile thời gian" trừu tượng (mặc định 1 ngày). Hợp đồng đầu ra mà các khâu đã có đang chờ:
- sự kiện ra phải đủ **21 cột interim, đúng dtype** (`Time` Int64, `EventID`/`LogonType` Int32, còn lại String);
- mỗi sự kiện phải qua `assert_events_within_day` với ngày dự định.

Test bắt buộc theo đặc tả (chưa có): nhân bản khuôn; mỗi thao tác chỉ đổi đúng trường dự kiến.

### 3.2 `scenarios.py` — 6 kịch bản *(thiếu)*
Chưa có code. Trước khi viết kịch bản 5 và 6, đặc tả yêu cầu **in số tài khoản ứng viên** — việc này chưa làm.

### 3.3 `inject.py` — điều phối *(thiếu)*
Những gì khâu này phải tạo ra để các khâu đã có dùng được:

| Đầu ra | Bên tiêu thụ đã có | Ràng buộc |
|---|---|---|
| `events_injected/event_462x/event_462x_day-NN.parquet` | `overlay_days`, stage features, `rule_stats` | chỉ ngày > `split_day`; ngày nào có file thì **phải đủ cả 4624 và 4625**; cùng schema interim |
| `injected_events.parquet` | (kiểm tra) | mỗi sự kiện tiêm → `inj_id`; log chính không có cột đánh dấu |
| `injection_manifest.csv` | `write_run_labels` | cột tối thiểu `inj_id, scenario, DomainName, UserName, day`; thêm `campaign_id, split, seed`, tham số thực tế, số sự kiện |

Các kiểm tra cần chạy trước khi ghi (chưa có): không đụng ngày train; mọi sự kiện trong ngày dự định;
số dòng bảng phụ = số sự kiện thêm; schema file ra = schema interim; mỗi tài khoản tối đa một lần/run.

Đầu vào cho bước "bỏ tài khoản-ngày luật đã cảnh báo" (quyết định 2026-10-08: baseline luật ECDF, ngưỡng
budget 1%): đã có sẵn ở `experiments/results/baselines/baseline_scores.parquet` (cột `rule_ecdf`, segment
`User`, split `eval`) và ngưỡng ở `thresholds.json` (`User.rule_ecdf.threshold = 0.99814`). Cần đảm bảo hai
file này là của lần chạy **trên log gốc** (`rule_event_stats.json` → `injected_events_dir: null`).

### 3.4 `configs/injection.yaml` *(thiếu)*
Cần hai khối tách biệt `dev` (ngày 43–51) và `test` (ngày 52–60), khác seed; tham số từng kịch bản; đường
dẫn hồ sơ (`data/features/train_profiles/`) và thư mục runs. Nên thêm kiểm tra vào `tests/test_configs.py`
(hai khối không chồng ngày, mọi ngày > `split_day`, seed khác nhau).

### 3.5 Hạ tầng phụ còn thiếu
- **Stage benchmark ML** chưa nhận đường dẫn ma trận của run → chưa so được ML với baseline trên log đè.
- **Lệnh chạy** cho bước tiêm và bước oracle chưa có trong `main.py` (chưa có stage `inject` / `difficulty`).
- **`.gitignore`** chưa có thư mục runs (mỗi run ghi lại ~200 MB/ngày đè; 9 ngày ≈ 2 GB). Nên đặt runs dưới
  một thư mục đã bị ignore hoặc thêm dòng ignore.
- **Tài liệu:** `README.md` và `docs/project_structure.md` chưa nhắc tới `src/injection/`.

---

## 4. Thứ tự đề xuất để hoàn tất

1. `configs/injection.yaml` + test config.
2. `operations.py` + 2 test bắt buộc (nhân bản khuôn, chỉ đổi đúng trường).
3. Đếm ứng viên cho kịch bản 5 và 6 trên hồ sơ đã dựng, in ra trước khi viết hai kịch bản đó.
4. `scenarios.py`, rồi `inject.py` với các kiểm tra ở 3.3.
5. Tham số đường dẫn cho stage benchmark; stage `inject` / `difficulty` trong `main.py`; `.gitignore`.
6. Chạy thử khối `dev` → features → baselines → benchmark; chỉ chạy khối `test` một lần ở cuối.

## 5. Quyết định đã chốt (để không phải hỏi lại)

- Tiêm vào tầng interim, chỉ ngày test; train giữ nguyên tuyệt đối.
- "Luật đã cảnh báo sẵn" = baseline luật ECDF ở ngưỡng train budget 1% (6 luật cố định của
  `model_params.yaml` cờ 95% dòng User test nên không dùng).
- "Đủ 7 ngày lịch sử" = tài khoản có ≥ 7 ngày hoạt động trong train.
- Kịch bản ngoài giờ lấy chuỗi sự kiện từ một ngày **train** của nạn nhân.
- Kịch bản ngủ đông được xem **sự hiện diện** (có/không sự kiện) ở các ngày test ≤ t; ngưỡng ngủ đông từ train.
- Hiện tiêm 1 ngày; thiết kế phải mở cho nhiều ngày. Oracle độ khó chỉ là hook.
