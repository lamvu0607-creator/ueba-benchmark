# Hướng dẫn chạy pipeline trích xuất đặc trưng ở máy khác

> Áp dụng cho: `extract_account_day_matrix.py` (bản 2.0)
> Căn cứ thiết kế: [`docs/feature_engineering/feature_correlation_review.md`](../../docs/feature_engineering/feature_correlation_review.md) §11–§12

---

## 1. Cần gì để chạy

| Thành phần | Bắt buộc? | Ghi chú |
|---|---|---|
| Python **≥ 3.9** | ✅ | Đã kiểm chứng trên Python 3.14.6 |
| `polars` | ✅ | Đã kiểm chứng với polars 1.44.2 (`pip install "polars>=1.0"`) |
| `pyyaml` | ⬜ không bắt buộc | Chỉ dùng khi truyền `--config`. **Không có PyYAML thì script vẫn chạy** với cấu hình mặc định built-in (giá trị giống hệt `configs/system_config.yaml`) |
| `pandas`, `numpy` | ⬜ | Chỉ script kiểm tra `check_feature_matrix.py` cần |

Không cần `scikit-learn`, `matplotlib`, `seaborn` để chạy bước trích xuất đặc trưng.

### Kiểm tra nhanh môi trường

```bash
python -c "import sys, polars; print('python', sys.version.split()[0], '| polars', polars.__version__)"
python -c "import yaml; print('pyyaml', yaml.__version__)"   # nếu lỗi -> vẫn chạy được, xem §5
```

---

## 2. Cần copy những file nào

**Tối thiểu 1 file — chạy được ngay:**

```text
extract_account_day_matrix.py
```

Script **tự dò** dữ liệu, config và nơi ghi kết quả, **không** import từ `src/`, không đọc file nào khác
ngoài dữ liệu đầu vào. Cấu hình mặc định đã có sẵn bên trong file.

Nên copy thêm (không bắt buộc):

```text
system_config.yaml                # nếu muốn đổi cửa sổ giờ / danh sách tài khoản bằng YAML
feature_schema.yaml               # hợp đồng schema (chỉ là tài liệu, không cần lúc chạy)
check_feature_matrix.py           # kiểm tra ma trận đầu ra (9 kiểm tra)
HUONG_DAN_CHAY.md                 # file này
```

### Thứ tự tự dò đường dẫn

| Cần tìm | Thứ tự dò (dừng ở cái đầu tiên tồn tại) |
|---|---|
| Thư mục dữ liệu `interim` | `--interim-dir` → `./data/interim` (thư mục làm việc) → `<cạnh script>/data/interim` → `<cạnh script>/interim` → `scripts/data/interim` → `<repo>/data/interim` → các cấp trên nữa của script |
| File config | `--config` → `./configs/system_config.yaml` → `./system_config.yaml` → `<cạnh script>/system_config.yaml` → `<repo>/configs/system_config.yaml` |
| Nơi ghi kết quả | `--output-dir` → `<gốc dữ liệu>/data/features` (tự đặt cạnh dữ liệu) |

> Điều kiện nhận diện thư mục `interim`: phải có **cả** `event_4624/` và `event_4625/`.
> Với các vị trí "đi ngược lên từ script", script chỉ nhận nếu thư mục gốc thật sự giống gốc dự án
> (có `scripts/`, `configs/` hoặc `.git/`) — tránh nhặt nhầm dữ liệu ở nơi khác trên cùng ổ đĩa.

---

## 3. Dữ liệu đầu vào: cấu trúc thư mục bắt buộc

```text
<THU_MUC_CHAY>/
└── data/
    └── interim/
        ├── event_4624/
        │   ├── event_4624_day-01.parquet
        │   ├── event_4624_day-02.parquet
        │   └── ...                       # tên file PHẢI đúng dạng event_4624_day-XX.parquet
        └── event_4625/
            ├── event_4625_day-01.parquet
            └── ...
```

- Chỉ cần các cột: `Time, EventID, UserName, LogHost, LogonType, AuthenticationPackage, ProcessName, Source, DomainName` (4625 thêm `FailureReason`). Các cột khác không bị đọc.
- `Time` là **số giây trong ngày** (1–86 400), không phải Unix epoch.
- Tên file quyết định số ngày xử lý: `--start-day 1 --end-day 30` ⇒ cần `day-01` … `day-30` trong **cả hai** thư mục. Thiếu ngày nào thì ngày đó bị bỏ qua kèm cảnh báo.

---

## 4. Chạy

### Cách A — chỉ cần chạy file (không truyền gì)

```bash
python extract_account_day_matrix.py
```

Script tự dò dữ liệu, config và nơi ghi. Chạy mặc định 3 ngày đầu (day 1–3).
Chạy được **từ bất kỳ thư mục nào** (không cần `cd` vào đâu): nếu đứng ngoài, script dò tiếp
theo vị trí của chính file script.

### Cách B — chọn khoảng ngày

```bash
python extract_account_day_matrix.py --start-day 1 --end-day 30
```

### Cách C — ghi đè đường dẫn (khi dữ liệu nằm ở nơi khác)

```bash
python extract_account_day_matrix.py ^
    --interim-dir "D:\du_lieu\ueba\data\interim" ^
    --output-dir  "D:\ket_qua\ueba_features" ^
    --config "D:\du_lieu\ueba\configs\system_config.yaml" ^
    --start-day 1 --end-day 30
```

### Cách D — máy không có PyYAML / không có file config

Vẫn chạy được như Cách A/B: script in `Config: không có file -> dùng cấu hình mặc định built-in`
(off 18h–7h, work 8h–17h, luật phân loại tài khoản). Kết quả **không đổi** so với dùng config hiện tại.

### Kết quả ghi ở đâu

Mặc định `<gốc dữ liệu>/data/features/`: nếu dữ liệu ở `<X>/data/interim` thì output ở `<X>/data/features`:

```text
<gốc dữ liệu>/data/features/
├── daily/user_features_day-01.parquet      # từng ngày
├── daily/user_features_day-02.parquet
├── daily/user_features_day-03.parquet
├── account_day_matrix.parquet              # MA TRẬN TỔNG HỢP (dùng cho bước sau)
└── pivot_account_day_logons.parquet        # bảng 2D  tài khoản × ngày (volume)
```

### Tham số đầy đủ

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--start-day` | `1` | Ngày bắt đầu (số trong tên file) |
| `--end-day` | `3` | Ngày kết thúc |
| `--interim-dir` | *tự dò* | Ghi đè thư mục chứa `event_4624/`, `event_4625/` |
| `--output-dir` | *tự dò* | Ghi đè thư mục ghi kết quả |
| `--config` | *tự dò* | Ghi đè file YAML cấu hình (không có ⇒ mặc định built-in) |

---

## 5. Kiểm tra kết quả sau khi chạy

```bash
python check_feature_matrix.py --matrix data/features/account_day_matrix.parquet --schema configs/feature_schema.yaml
```

Script kiểm tra 9 điều kiện (khoá duy nhất, không cột hằng số, 0 cặp |ρ| ≥ 0,85, VIF < 10,
không còn artefact 86 400, ratio ∈ [0,1], NULL của `failure_*_share` đúng chỗ,
`entity_type` phân loại hết, ma trận khớp hợp đồng schema) và **trả exit code 1 nếu có mục FAIL**
(dùng được trong CI).

Kết quả mong đợi trên dữ liệu hiện có (3 ngày): **57.485 dòng × 21 cột**, `0` cặp |ρ| ≥ 0,85, `VIF max ≈ 6,9`.

---

## 6. Xử lý sự cố thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `[LỖI] Không tìm thấy dữ liệu interim (cần có cả event_4624/ và event_4625/)` + danh sách vị trí đã dò | Chưa có dữ liệu ở bất kỳ vị trí tự dò nào | (1) đặt `data/interim` cạnh file script, hoặc (2) chạy từ thư mục gốc repo, hoặc (3) truyền `--interim-dir "D:\...\data\interim"` |
| `[CẢNH BÁO] Không tìm thấy dữ liệu cho Day 03, bỏ qua.` | Đã tìm thấy `interim` nhưng thiếu file của ngày đó | Kiểm tra tên file đúng dạng `event_4624_day-XX.parquet` (2 chữ số) trong **cả** hai thư mục `event_4624/`, `event_4625/` |
| `ModuleNotFoundError: No module named 'polars'` | Thiếu thư viện | `pip install "polars>=1.0"` |
| `Config: không có file -> dùng cấu hình mặc định built-in` | Không tìm thấy file YAML | **Bình thường — không cần làm gì**; muốn đổi cấu hình thì copy `system_config.yaml` cạnh script hoặc truyền `--config` |
| `[CẢNH BÁO] Chưa cài PyYAML ...` | Chưa cài `pyyaml` | Không sao — kết quả không đổi; hoặc `pip install pyyaml` |
| Lỡ chạy `extract_account_day_matrix_v1.bak.py` | Đó là **file lưu trữ** của bản cũ | Dùng `extract_account_day_matrix.py` (bản 2.0). Nếu đã lỡ chạy, chạy lại bản 2.0 để ghi đè ma trận |
| `[!] Day XX: ... entity_type='Other'` | Có tài khoản mới chưa khai báo | Bổ sung tên vào `features.entity_type.service_accounts`/`admin_accounts` trong `system_config.yaml` |
| Chạy rất chậm / hết RAM | Đọc quá nhiều ngày một lần ở máy ít RAM | Chạy theo lô: `--start-day 1 --end-day 10`, rồi `11–20`… và ghép kết quả; hoặc tăng `--end-day` dần để đo |
| Muốn **lọc** tài khoản máy/hệ thống | Mặc định không lọc | Sửa `features.filter.drop_machine_accounts: true` / `drop_system_accounts: true` (⚠️ lọc machine sẽ bỏ ~45% khối lượng sự kiện) |

---

## 7. Định mức thời gian & dung lượng (đo trên máy hiện tại)

| Hạng mục | Giá trị |
|---|---|
| 1 ngày (4624 + 4625 ≈ 17,9 triệu dòng) | **9–14 giây** |
| 3 ngày | **≈ 33 giây** (RAM đỉnh vài GB/ngày) |
| Ma trận 3 ngày | 57.485 dòng × 21 cột ≈ **3,3 MB** parquet |

Ước lượng 30 ngày: ~6–8 phút tuần tự, ma trận ≈ 570 nghìn dòng.

---

## 8. Bộ đặc trưng đầu ra (16 đặc trưng + 4 khoá/nhãn + 1 cột hiển thị)

```text
Khoá/nhãn : UserName, DomainName, day, entity_type (Machine|User|Service|System|Admin)
Hiển thị   : total_logons        (CHỈ để đọc/pivot — không dùng cho model/tương quan)
Đặc trưng  : log_total_logons, failure_ratio, failure_locked_out_share,
             off_hours_ratio, interarrival_dt_mean, delta_t_cv, same_second_share,
             is_single_event, interactive_ratio, rare_logon_type_count_log,
             ntlm_ratio, log_distinct_hosts, distinct_sources_count,
             missing_source_ratio, remote_logon_ratio, custom_proc_share
```

Định nghĩa công thức & lý do loại bỏ từng đặc trưng cũ: xem [`configs/feature_schema.yaml`](../../configs/feature_schema.yaml).

⚠️ **Lưu ý khi dùng tiếp cho model:** các cột `failure_locked_out_share`, `interarrival_dt_mean`,
`delta_t_cv` có **NULL hợp lệ** (khi tài khoản không có thất bại / chỉ có 1 sự kiện trong ngày).
Không nên `fillna(0)` một cách mù quáng — hãy dùng `is_single_event` để nhận biết nhóm này.
