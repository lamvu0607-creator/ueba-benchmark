# Báo cáo: Sửa các bất thường của bộ đặc trưng (Feature Set v1.0 → v2.0)

> **Dự án:** UEBA Anomaly Detection Benchmark (Windows Event 4624 & 4625 — LANL Unified Host and Network Dataset)  
> **Ngày thực hiện:** 2026-09-25  
> **Người thực hiện:** Data Scientist (review + sửa trực tiếp)  
> **Căn cứ:** [`docs/feature_engineering/feature_correlation_review.md`](../../docs/feature_engineering/feature_correlation_review.md) §7, §11, §12  
> **Phạm vi:** **chỉ xử lý các bất thường của tập đặc trưng** (§11.1 #1–#5, §11.2 #6–#10). Các mục hạ tầng/tài liệu (§11.3 #11–#16) **không** thuộc phạm vi báo cáo này.  
> **Vị trí lưu trữ:** `reports/archive/bao_cao_sua_bo_dac_trung.md` (Lưu trữ — đã kế thừa bởi `docs/reports/bao_cao_bo_dac_trung_v3.md`)

---

## 1. Tóm tắt điều hành

Tập đặc trưng bản 1.0 (32 đặc trưng, sau khi script phân tích loại 2 cột hằng số còn 30) có **13 cặp tương quan |ρ| ≥ 0,85**, **VIF lớn nhất 4041,96** và **2 đặc trưng chết**. Báo cáo trước đã chỉ ra phần lớn các tương quan mạnh đó **không phải tín hiệu an ninh** mà đến từ: (a) bản sao qua biến đổi đơn điệu, (b) trộn quần thể Machine/User/Service/Admin, và (c) **missingness có cấu trúc** của log LANL bị biến thành đặc trưng hành vi.

Sau khi sửa (bản 2.0), đo lại trên cùng dữ liệu 3 ngày:

| Chỉ số chất lượng bộ đặc trưng | Trước (v1.0) | Sau (v2.0) | Cải thiện |
|---|---:|---:|---|
| Số đặc trưng đưa vào model | 30 | **16** | −47% |
| Cặp \|ρ\| ≥ 0,85 | **13** | **0** | hết đa cộng tuyến theo ngưỡng |
| VIF lớn nhất | **4041,96** | **6,90** | −99,8% |
| Cột hằng số / đặc trưng chết | 2 | **0** | — |
| Khoá danh tính | `UserName` (trộn domain) | **`(DomainName, UserName)`** | sửa lỗi danh tính |
| Cửa sổ giờ phủ 24h | ✗ trống 21,20% / 25,26% sự kiện | ✓ 18h–7h + 8h–17h | — |
| Artefact `interarrival_dt_mean = 86400` | 0,841% dòng | **0** | sửa giá trị giả |
| Nhãn thực thể | `Administrator` (1,28M sự kiện) bị gán `Service` | **`Admin`** (nhãn riêng) | sửa phân loại sai |

**8/8 điều kiện nghiệm thu (Definition of Done §11.4) PASS.**

---

## 2. Bối cảnh và căn cứ

### 2.1. Vấn đề gốc

| Cặp trong bộ v1.0 | Hệ số | Bản chất |
|---|---:|---|
| `total_logons ↔ log_total_logons` | ρ = **1,0000** | cùng một biến (biến đổi log đơn điệu) |
| `distinct_hosts_count ↔ log_distinct_hosts` | ρ = **1,0000** | cùng một biến |
| `failure_count ↔ failure_ratio` | ρ = **0,9969** | hai cột cùng đo "hôm nay có thất bại hay không" |
| `has_local_logon ↔ has_process_info` | ρ = **0,9738** | cùng dẫn xuất từ việc thiếu `ProcessName`/`Source` |
| `network_ratio ↔ has_process_info` | ρ = **−0,9431** | cùng một trục "network vs non-network" |
| `has_system_logon_id ↔ failure_count` | ρ = **0,9104** | `LogonID = 0x3e7` là proxy của thất bại |

### 2.2. Nguyên nhân dữ liệu (đã kiểm chứng trên raw ngày 16)

- `Status` **null 100%** ở cả 4624 và 4625 ⇒ `wrong_password_count`, `unknown_user_count` **luôn = 0**.
- `ProcessName` null **92,37%** ở 4624 (nhưng chỉ 49,16% ở 4625) ⇒ các cờ từ `ProcessName` chủ yếu đo *chế độ ghi log*.
- Trong 4625, `ProcessName` null và `LogonID` null là **đúng cùng một tập dòng** (49,16%) ⇒ missingness kết cụm theo loại sự kiện.
- `Source` null 24,28% (4624) / 32,09% (4625) và **có cấu trúc theo LogonType** (type 5 = 99,92%, type 9/0 = 100%, type 3 = 23,02%).
- **70,57%** sự kiện 4624 có `delta_t = 0` (LANL chỉ có độ phân giải giây) ⇒ `burst_logon_count` thực chất là proxy của volume.
- `LogonID == 0x3e7` chiếm **45,92%** sự kiện 4625 nhưng chỉ 2,09% ở 4624.

---

## 3. Phương pháp kiểm chứng

| Bước | Cách làm | Kết quả |
|---|---|---|
| Tái lập bảng tương quan cũ | tính lại Pearson/Spearman từ `account_day_matrix.parquet` | khớp tới **4,99e-05** ⇒ bảng cũ đáng tin |
| Truy vết ngữ nghĩa trên raw | quét cả hai file của ngày 16 (`event_4624_day-16.parquet` + `event_4625_day-16.parquet`, ~17,9 triệu dòng) | xác định nguyên nhân missingness/artifact |
| Sinh ma trận mới | `extract_account_day_matrix.py` (bản 2.0) | 57.485 dòng × 21 cột trong **32,8 giây** |
| Nghiệm thu | `scripts/feature_engineering/check_feature_matrix.py` | **8/8 PASS** |

---

## 4. Những gì đã sửa (10 mục)

| # | Bất thường (v1.0) | Cách sửa (v2.0) | Bằng chứng |
|---|---|---|---|
| 1 | Khoá danh tính chỉ có `UserName` ⇒ trộn `Domain001` với `nt authority`… | `group_by(["DomainName", "UserName"])`; `DomainName` thiếu (4 + 33 dòng) → `unknown`; chuẩn hoá `strip + lowercase` | 0 dòng trùng khoá; 21.454 danh tính (trước 20.878) |
| 2 | `entity_type` theo luật "còn lại → Service" ⇒ `Administrator` (1.275.672 sự kiện/ngày 16) bị gán `Service` | Luật **tường minh** từ config: `Admin → System → Machine (*$) → User (User*) → Service → Other` | 5 nhãn: Machine 31.166 · User 26.167 · **Admin 116** · Service 23 · System 13; không còn `Other` |
| 3 | `ignore_system_accounts` trong config **không được đọc**; giá trị không khớp dữ liệu (`ANONYMOUS LOGON` vs `Anonymous`) | Danh sách sửa theo giá trị thật, được **đọc** để gán nhãn `System`; thêm cờ lọc `features.filter.drop_*` (mặc định **tắt**) | 4 tài khoản `system/local service/network service/anonymous` → nhóm `System` (13 dòng) |
| 4 | `wrong_password_count`, `unknown_user_count` luôn = 0 (`Status` null 100%) | Bỏ 2 cột; thêm `failure_locked_out_share` từ `FailureReason` (4625 có 100% giá trị) | Cột mới: 14,36% dòng = 0, 85,46% NULL hợp lệ |
| 5 | 3 định nghĩa cửa sổ giờ khác nhau; vùng trống 21,20% / 25,26% sự kiện | Lấy cửa sổ **từ config**: off 18h–7h + work 8h–17h = **phủ đủ 24h**; self-check mỗi ngày | Không còn cảnh báo `off_hours + work_hours != total_logons` |
| 6 | Script hard-code tham số, không đọc config | Thêm `--config`; đọc `off_hours_*`, `work_hours_*`, `entity_type.*`, `filter.*`; in phiên bản python/polars + nguồn config | Header chạy in `Config: ... (ĐÃ NẠP)` |
| 7 | `burst_logon_count` ≈ đếm sự kiện trùng giây (70,57% `delta_t = 0`), ρ = 0,9102 với volume | Bỏ; thay bằng `same_second_share` (tỷ trọng thay vì count) | VIF cột mới = 2,05 |
| 8 | `fill_null(86400.0)` tạo đỉnh giả 86.400 ở 0,841% dòng | Bỏ fill; giữ **NULL** cho `interarrival_dt_mean` khi chỉ có 1 sự kiện; thêm `is_single_event`, `delta_t_cv` | 806 dòng NULL (1,402%) — khớp chính xác số dòng 1 sự kiện; 0 dòng ≥ 86.399 |
| 9 | 32,39% dòng có tổng 5 ratio logon-type < 1 (type 7/8/9/10/11 không được biểu diễn) | Thêm `rare_logon_type_count_log` (RDP type 10 và runas type 9 vẫn xuất hiện) | 65,46% dòng = 0 (không có type hiếm) |
| 10 | 12 đặc trưng trùng lặp / near-constant / heuristic (§7) | Rút gọn còn 16 đặc trưng; bỏ cả 3 ratio logon-type gần hằng số và 3 cờ `has_*` từ `ProcessName` | Xem §5 |

---

## 5. Kết quả đo được (Definition of Done §11.4)

| # | Điều kiện nghiệm thu | v1.0 | v2.0 |
|---|---|---|---|
| 1 | Khoá `(DomainName, UserName, day)` duy nhất | ✗ | ✅ 0 dòng trùng |
| 2 | Không còn cột hằng số | ✗ 2 cột | ✅ 0 cột |
| 3 | Cặp \|ρ\| ≥ 0,85 trong bộ model | ✗ 13 cặp | ✅ **0 cặp** |
| 3b | VIF lớn nhất | ✗ 4041,96 | ✅ **6,90** (`remote_logon_ratio`) |
| 4 | Không còn artefact `fill_null(86400)` | ✗ 0,841% dòng | ✅ 0 dòng; NULL chỉ tại dòng 1 sự kiện |
| 5 | Bộc lộ được type hiếm (RDP/runas) | ✗ 32,39% dòng thiếu | ✅ `rare_logon_type_count_log` |
| 6 | Cửa sổ giờ phủ 24h | ✗ | ✅ |
| 7 | `entity_type` tường minh, `Administrator` không thuộc `Service` | ✗ | ✅ `Administrator → Admin` |
| 8 | `failure_*_share` nhất quán với `failure_ratio` | — | ✅ 0 dòng vi phạm |
| 9 | Ma trận khớp hợp đồng `configs/feature_schema.yaml` | — | ✅ 16 = 16, không thừa/thiếu |

VIF sau khi sửa (top 8): `remote_logon_ratio` 6,90 · `missing_source_ratio` 3,55 · `custom_proc_share` 3,15 · `log_total_logons` 2,32 · `log_distinct_hosts` 2,26 · `same_second_share` 2,05 · `rare_logon_type_count_log` 2,02 · `ntlm_ratio` 1,68.

File `docs/feature_engineering/tables/multicollinear_pairs.csv` **chỉ còn dòng tiêu đề** (không còn cặp nào vượt ngưỡng).

---

## 6. Hai phát hiện phát sinh khi sửa (đã xử lý, ghi lại để không lặp lại)

1. **Không được "đóng ràng buộc tổng" bằng ratio bù trừ.** Phương án `other_logon_type_ratio = 1 − network − interactive` tạo **phụ thuộc tuyến tính tuyệt đối** ⇒ VIF ≈ **5,2e7** và ρ(network, other) = **0,9415**. Vì mô hình mục tiêu (IForest/LOF/OCSVM) dựa trên khoảng cách, ma trận hiệp phương sai suy biến là lỗi nghiêm trọng ⇒ đã chuyển sang `rare_logon_type_count_log`, và cuối cùng bỏ luôn `network_ratio` (gần hằng số: median = 1,000, chỉ 2,46% dòng = 0; ρ = 0,9031 với cột rare).
2. **"Share of failures" vẫn trùng với cường độ thất bại.** `failure_bad_password_share` có ρ = **0,9220** với `failure_ratio` — vì mọi lý do thất bại đều là bằng chứng "có thất bại". Chỉ giữ `failure_locked_out_share` (tín hiệu lockout, không vượt ngưỡng).

→ Bài học tổng quát: **một đặc trưng thay thế có thể tái tạo đúng vấn đề trùng lặp mà nó định sửa**, nên sau mỗi thay đổi phải **đo lại** hệ số tương quan và VIF, không suy luận bằng trực giác.

---

## 7. Bộ đặc trưng v2.0 (16 đặc trưng)

| # | Đặc trưng | Nhóm | Định nghĩa | Ghi chú |
|---|---|---|---|---|
| 1 | `log_total_logons` | volume | `log1p(số sự kiện 4624+4625 trong ngày)` | bản log; bản thô `total_logons` chỉ để hiển thị |
| 2 | `failure_ratio` | failure | `failure_count / total_logons` | đặc trưng cường độ thất bại **duy nhất** |
| 3 | `failure_locked_out_share` | failure | `#(FailureReason = 'Account locked out') / failure_count` | NULL khi không có thất bại |
| 4 | `off_hours_ratio` | time | tỷ lệ sự kiện có `hour ≥ 18` hoặc `hour ≤ 7` | `work_hours_ratio` đã bỏ (off + work = 1 tuyệt đối) |
| 5 | `interarrival_dt_mean` | rhythm | trung bình `delta_t` (giây) giữa 2 sự kiện liên tiếp | NULL khi chỉ có 1 sự kiện |
| 6 | `delta_t_cv` | rhythm | `std(delta_t) / mean(delta_t)` | thay `interarrival_dt_std` |
| 7 | `same_second_share` | rhythm | tỷ lệ sự kiện có `delta_t = 0` | thay `burst_logon_count` |
| 8 | `is_single_event` | rhythm | 1 nếu khoá chỉ có đúng 1 sự kiện | cờ giải thích NULL |
| 9 | `interactive_ratio` | logon_type | `#(LogonType = 2) / total_logons` | |
| 10 | `rare_logon_type_count_log` | logon_type | `log1p(#(LogonType ∉ {2,3}))` | gồm RDP (10), runas (9), unlock (7)… |
| 11 | `ntlm_ratio` | authentication | tỷ lệ gói xác thực NTLM | là proxy của chế độ thất bại |
| 12 | `log_distinct_hosts` | fanout | `log1p(#LogHost duy nhất)` | |
| 13 | `distinct_sources_count` | fanout | `#Source` duy nhất (bỏ null) | skew 140 ⇒ nên rank-Gaussian/winsorize |
| 14 | `missing_source_ratio` | context | tỷ lệ sự kiện thiếu `Source` | **là chỉ báo chất lượng dữ liệu** |
| 15 | `remote_logon_ratio` | context | tỷ lệ `Source ≠ LogHost` | |
| 16 | `custom_proc_share` | context | tỷ lệ `ProcessName` bắt đầu bằng `proc` | bản tỷ trọng của cờ `has_custom_proc` cũ |

Khoá/nhãn: `UserName`, `DomainName`, `day`, `entity_type` ∈ {Machine, User, Service, System, Admin}.
Cột hiển thị: `total_logons` — **không** dùng cho model/tương quan (đã loại trong `plot_feature_correlation.py`).

Hợp đồng đầy đủ (gồm 25 đặc trưng đã loại kèm lý do): [`configs/feature_schema.yaml`](../../configs/feature_schema.yaml).

⚠️ **Khi đưa vào model:** `failure_locked_out_share`, `interarrival_dt_mean`, `delta_t_cv` có **NULL hợp lệ**
(không có thất bại / chỉ có 1 sự kiện). Dùng `is_single_event` để nhận biết, tránh `fillna(0)` mù quáng.

---

## 8. Sản phẩm bàn giao

| Loại | Đường dẫn | Ghi chú |
|---|---|---|
| Mã nguồn (bản 2.0) | [`scripts/feature_engineering/extract_account_day_matrix.py`](../../scripts/feature_engineering/extract_account_day_matrix.py) | sửa cả 10 mục; thêm `--config` và self-check |
| Hợp đồng schema | [`configs/feature_schema.yaml`](../../configs/feature_schema.yaml) | **mới** |
| Cấu hình | [`configs/system_config.yaml`](../../configs/system_config.yaml) | sửa list tài khoản + thêm `work_hours_*`, `features.entity_type.*`, `features.filter.*` |
| Script kiểm tra | [`scripts/feature_engineering/check_feature_matrix.py`](../../scripts/feature_engineering/check_feature_matrix.py) | **mới**, 9 kiểm tra + exit code cho CI |
| Hướng dẫn chạy ở máy khác | [`scripts/feature_engineering/HUONG_DAN_CHAY.md`](../../scripts/feature_engineering/HUONG_DAN_CHAY.md) | **mới** |
| Ma trận đặc trưng | `data/features/account_day_matrix.parquet` | 57.485 dòng × 21 cột (3,3 MB) |
| Ma trận 2D volume | `data/features/pivot_account_day_logons.parquet` | khoá `domain\account` × ngày |
| Bảng/biểu đồ phân tích | `docs/feature_engineering/tables/*.csv`, `figures/*.png` | đã sinh lại theo ma trận mới |
| Bản cũ để đối chiếu | `scripts/feature_engineering/extract_account_day_matrix_v1.bak.py` | đã gắn cảnh báo **"KHÔNG DÙNG ĐỂ CHẠY"** ở đầu file; xoá được khi không cần |

---

## 9. Cách chạy lại và mang sang máy khác

### 9.1. Chạy chỉ bằng 1 file

```bash
python extract_account_day_matrix.py
```

Không cần truyền tham số, không cần đứng ở thư mục nào cụ thể, không cần file config.
`extract_account_day_matrix.py` **tự dò** (theo thứ tự, dừng ở cái đầu tiên tồn tại):

| Cần tìm | Thứ tự dò |
|---|---|
| Thư mục dữ liệu `interim` | `--interim-dir` → `./data/interim` → `<cạnh script>/data/interim` → `<cạnh script>/interim` → `scripts/data/interim` → `<repo>/data/interim` → các cấp trên của script |
| File config | `--config` → `./configs/system_config.yaml` → `./system_config.yaml` → `<cạnh script>/system_config.yaml` → `<repo>/configs/system_config.yaml` |
| Nơi ghi kết quả | `--output-dir` → `<gốc dữ liệu>/data/features` |

Điều kiện nhận diện `interim`: phải có **cả** `event_4624/` và `event_4625/`. Các vị trí "đi ngược lên từ script"
chỉ được nhận nếu thư mục gốc thật sự giống gốc dự án (có `scripts/`, `configs/` hoặc `.git/`).
Nếu không tìm thấy dữ liệu, script in rõ **danh sách vị trí đã dò** kèm 3 cách xử lý.

**Yêu cầu duy nhất:** Python ≥ 3.9 + `polars` (đã kiểm chứng: Python 3.14.6, polars 1.44.2). PyYAML là tuỳ chọn —
thiếu PyYAML hoặc thiếu file config thì script dùng cấu hình mặc định built-in và **kết quả không đổi**
(vì mặc định trùng với `configs/system_config.yaml`).

**Cấu trúc dữ liệu bắt buộc:**

```text
<THU_MUC_CHAY>/data/interim/event_4624/event_4624_day-XX.parquet
<THU_MUC_CHAY>/data/interim/event_4625/event_4625_day-XX.parquet
```

**Lệnh chạy (trỏ đường dẫn tuyệt đối, chạy ở đâu cũng được):**

```bash
python extract_account_day_matrix.py ^
    --interim-dir "D:\du_lieu\ueba\data\interim" ^
    --output-dir  "D:\ket_qua\ueba_features" ^
    --config "D:\du_lieu\ueba\configs\system_config.yaml" ^
    --start-day 1 --end-day 30
```

**Kiểm tra kết quả:**

```bash
python check_feature_matrix.py --matrix "D:\ket_qua\ueba_features\account_day_matrix.parquet" --schema "D:\du_lieu\ueba\configs\feature_schema.yaml"
```

Chi tiết đầy đủ (tham số, sự cố thường gặp, định mức thời gian): xem [`HUONG_DAN_CHAY.md`](../../scripts/feature_engineering/HUONG_DAN_CHAY.md).

**Gói mang đi:** `ueba_feature_pipeline_v2.zip` ở thư mục gốc repo (chứa script + 2 config + script kiểm tra + hướng dẫn).

**Đã kiểm chứng thực tế (2026-09-25):**

| Test | Điều kiện | Kết quả |
|---|---|---|
| A | Đứng ở gốc repo, chỉ truyền `--start-day/--end-day` | ✅ tự dò `data/interim` + `configs/system_config.yaml` theo CWD → 19.795 dòng, 10,4s |
| B | Đứng ở `C:\Users\ADMIN` (khác hoàn toàn), script nằm trong repo | ✅ tự dò theo **vị trí script** → 19.795 dòng, 11,6s |
| C | Chỉ có **1 file script** ở `D:\ueba_solo\`, không có dữ liệu | ✅ in `[LỖI]` + danh sách vị trí đã dò + 3 cách xử lý (không crash mờ mịt) |
| D | Gốc repo, lệnh tối giản `--start-day 1 --end-day 3`, **không tham số đường dẫn** | ✅ 57.485 dòng × 21 cột, 30,5s → ghi đúng `data/features` |
| E | Giải nén `ueba_feature_pipeline_v2.zip` ra thư mục khác rồi chạy `check_feature_matrix.py` | ✅ 9/9 PASS |

Trong quá trình test đã phát hiện và sửa **3 lỗi thật**:
1. Console Windows dùng cp1252 → script kiểm tra không in được tiếng Việt (`UnicodeEncodeError`).
2. `pl.Expr.n_unique(drop_nulls=...)` không được hỗ trợ trên polars 1.44.
3. Hàm `build_account_day_matrix` dùng biến `interim_dir` (tham số) thay vì `interim` (đã dò) ⇒ lỗi khi không truyền tham số.

---

## 10. Rủi ro & giới hạn cần biết

| Điểm | Mô tả | Ảnh hưởng |
|---|---|---|
| **Chỉ 3 ngày dữ liệu** | Ma trận hiện tại gồm day 1–3 (57.485 dòng) | Chưa ước lượng được baseline ổn định, chưa tách weekday/weekend. **Nên chạy lại ≥ 14–30 ngày** trước khi huấn luyện |
| `missing_source_ratio` | Là **chỉ báo chất lượng dữ liệu** (thiếu `Source` tập trung ở Network+Kerberos) | Không được diễn giải là tấn công; nên dùng làm covariate khi đọc kết quả |
| `ntlm_ratio` | NTLM chiếm 48,18% sự kiện 4625 so với 17,35% ở 4624 | Là proxy của chế độ thất bại, không thuần là "phương thức xác thực" |
| `custom_proc_share` | Hệ quả của sơ đồ ẩn danh `Proc<6 số>.exe` | Ý nghĩa phụ thuộc cách ẩn danh của tập dữ liệu |
| NULL hợp lệ | 3 đặc trưng có NULL (`failure_locked_out_share` 85,46%; `delta_t_cv` 2,22%; `interarrival_dt_mean` 1,40%) | Bước model phải xử lý NULL có chủ đích (không `fillna(0)` mù quáng) |
| `is_single_event` | 98,60% dòng = 0 (gần hằng số) | Giữ có chủ ý để giải thích NULL; có thể bỏ nếu muốn tối giản |

---

## 11. Việc còn lại (ngoài phạm vi báo cáo này)

- §11.3 #11–#16: `main.py` thiếu stage `features`/`benchmark`; tên cột trong `system_config.yaml:13-18` sai schema interim
  (`TimeCreated/TargetUserName/IpAddress/WorkstationName` → thực tế `Time/UserName/Source/LogHost`); chốt đường dẫn output;
  `README.md` vẫn mô tả danh sách "16 đặc trưng" cũ; `tests/` chưa có unit test; script tương quan chưa thêm
  within-account / VIF / Cramér's V.
- Đặc trưng bổ sung §7.3 (novelty, sequence, burst theo cửa sổ) chưa thêm — **đã có đề xuất phân tầng
  kèm bằng chứng loại trừ tại [`bao_cao_bo_dac_trung_v3.md`](../../docs/reports/bao_cao_bo_dac_trung_v3.md)** (Tier A 30 biến
  model + Tier B 30 biến thử nghiệm + Tier C 7 biến chẩn đoán).
- Rà soát `src/evaluation/injector.py`: kịch bản "Off-hours Compromise" dựa vào RDP nhưng RDP chỉ chiếm 0,013% sự kiện thật
  (nay tín hiệu sẽ hiện ở `rare_logon_type_count_log`, cần xác nhận khi chạy benchmark).

---

## 12. Phụ lục — lệnh tái lập

```bash
# 1. Sinh ma trận đặc trưng (3 ngày)
python scripts/feature_engineering/extract_account_day_matrix.py \
    --config configs/system_config.yaml --start-day 1 --end-day 3 --output-dir data/features

# 2. Nghiệm thu schema (kỳ vọng: PASS toàn bộ, exit code 0)
python scripts/feature_engineering/check_feature_matrix.py \
    --matrix data/features/account_day_matrix.parquet --schema configs/feature_schema.yaml

# 3. Sinh lại bảng tương quan & biểu đồ phân phối
python scripts/feature_engineering/plot_feature_correlation.py
python scripts/feature_engineering/plot_feature_distribution.py
```

**Kết quả mong đợi ở bước 1 (máy hiện tại):**

```text
Tổng số bản ghi: 57,485  |  Tổng số cột: 21 (4 khoá/nhãn + 16 đặc trưng + 1 cột hiển thị)
Dung lượng: ~3.3 MB     |  Thời gian: ~33 giây
```

**Kết quả mong đợi ở bước 2:** 8/8 PASS, `0` cặp |ρ| ≥ 0,85, `VIF max = 6.90`.

### Nguồn số liệu

| Số liệu | Nguồn |
|---|---|
| 57.485 × 21, thời gian 32,8s | header pipeline khi chạy `--start-day 1 --end-day 3` |
| 0 cặp \|ρ\| ≥ 0,85, VIF max 6,90 | `check_feature_matrix.py` + `docs/feature_engineering/tables/multicollinear_pairs.csv` |
| 806 dòng NULL (1,402%), 0 dòng ≥ 86.399 | kiểm tra #4 trong `check_feature_matrix.py` |
| Phân bố `entity_type` (31.166 / 26.167 / 116 / 23 / 13) | kiểm tra #7 |
| Số liệu v1.0 (13 cặp, VIF 4041,96, 2 cột hằng số…) | [`docs/feature_engineering/feature_correlation_review.md`](../../docs/feature_engineering/feature_correlation_review.md) §1–§9 |
