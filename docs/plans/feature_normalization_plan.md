# Kế hoạch: Bộ đặc trưng cho báo cáo Mentor — Ban đầu → Phân tích lệch → Chuẩn hóa log

> **Dự án:** UEBA Anomaly Detection Benchmark  
> **Ngày cập nhật:** 2026-09-26 (sửa lần 2)  
> **Yêu cầu đã áp dụng:**
> 1. ✅ **Xoá toàn bộ 4 extractor cũ**, viết lại **1 extractor mới**.
> 2. ✅ **Giữ nguyên bộ đặc trưng mục tiêu 21 cột** (bảng §1.1).
> 3. ✅ **Không làm gì liên quan đến model** (bỏ hẳn tầng `model`, manifest model, train/test, scaling).
> 4. ✅ Kèm **danh sách đặc trưng bị lệch phân phối** để vẽ biểu đồ kiểm tra (§3, §4).
> **Mục đích của kế hoạch:** dựng một **câu chuyện 3 bước** để báo cáo mentor:
> `Bộ đặc trưng ban đầu (chưa chuẩn hóa)` → `Phân tích phân phối lệch` → `Bộ đặc trưng đã chuẩn hóa log`.
> **Vị trí lưu trữ:** `docs/plans/feature_normalization_plan.md`

---

## 0bis. Trạng thái thực hiện (cập nhật 2026-09-26)

| Bước | Trạng thái | Sản phẩm |
|---|---|---|
| **B1 — Extractor RAW** | ✅ **ĐÃ XONG** | `scripts/feature_engineering/extract_account_day_matrix.py` **viết lại từ đầu** (20 cột, **không có log**); 4 extractor cũ đã chuyển vào `scripts/feature_engineering/archive/`; ma trận `data/features/raw/feature_matrix_raw.parquet` — **1.055.283 dòng × 20 cột, 60 ngày, 57,6 MB** |
| **B1b — Nghiệm thu RAW** | ✅ **PASS 6/6** | `scripts/feature_engineering/check_raw_matrix.py` (chạy được, exit code 0) |
| **B2 — Phân tích lệch** | ✅ **XONG — 4/4 ảnh (3-panel)** | `plot_feature_distribution.py` trên 4 cột lệch nặng nhất: `rare_logon_type_count` **151,66** · `distinct_sources_count` **147,39** · `total_logons` **94,52** · `distinct_hosts` **91,12** → 4 hình `distribution_*.png` ở **cả** `figures/` và `artifacts/`, mỗi ảnh **3 panel**: thẻ thống kê (Mean/Median/P99/Max/**tỷ lệ = 0** + Kurtosis sau log) → **toàn dải** (chú thích gai zero-inflation + đuôi ngoài khung) → **zoom 99%** (`x ≤ P99`, in skew/kurtosis của vùng zoom). *(Panel Q-Q plot vs Chuẩn đã **bỏ** ngày 2026-09-24.)* Kèm `tables/distribution_skewness_comparison.csv` (skew & kurtosis gốc vs sau log, % zero). Script tự kiểm đếm ảnh ở cuối log, thiếu ảnh → exit code 1 |
| **B2b — Soát đa cộng tuyến** | ✅ **PASS** | `plot_feature_correlation.py` (heatmap Spearman **có ghi số ở mọi ô**) + `check_multicollinearity.py` (mới): **0/120 cặp \|ρ\| ≥ 0,85**, VIF max = **7,63** (`remote_logon_ratio`) < 10 → báo cáo `reports/week2/multicollinearity_check.md` + `tables/multicollinearity_vif.csv` |
| **B2c — Kiểm tra hậu chuẩn hóa** | ✅ **PASS (4/4 cột chốt ≤ 3)** | `check_normalization_effect.py` (mới) trên toàn bộ 16 đặc trưng: skew 91–152 → **1,26–2,47** (log1p). **BẢNG 3 mới** so sánh hình dạng **toàn dải vs zoom 99%** để trả lời “sao ảnh vẫn trông lệch” (`distinct_hosts`: +1,26 → **+0,02**; `distinct_sources_count`: +2,47 → **+1,25**). Phát hiện **2 cột nên bổ sung vào log1p ở B3**: `interarrival_dt_mean` (16,01 → **−0,30**) và `delta_t_cv` (9,37 → **0,95**); 8 cột còn lại giữ thô (log vô hiệu 0–16,7%); sqrt/cbrt/Yeo-Johnson **không đáng đổi** → `reports/week2/normalization_assessment.md` + **3 CSV** trong `tables/` (`normalization_effect_all_features.csv`, `normalization_alternative_transforms.csv`, `normalization_shape_zoom_vs_full.csv`) |
| **B3 — Chuẩn hóa log 4 cột** | ⏳ **để sau** (theo yêu cầu) | ma trận 21 cột với `log_total_logons`, `log_distinct_hosts`, `rare_logon_type_count_log`, `log_distinct_sources_count` |

**Quyết định đã chốt:** chuẩn hóa log **đúng 4 cột lệch nặng nhất** (`rare_logon_type_count` 151,66 · `distinct_sources_count` 147,39 · `total_logons` 94,52 · `distinct_hosts` 91,12) — thực hiện ở bước sau, **không** áp log ở bước trích xuất.

> **Bằng chứng cho báo cáo (đã có sẵn):** [`tables/distribution_skewness_comparison.csv`](../feature_engineering/tables/distribution_skewness_comparison.csv) — skew gốc 91–152 → sau `log1p` còn 1,26–2,47 (**giảm 98,3%–99,0%**).
>
> **Nếu mentor hỏi “sao ảnh sau log vẫn trông lệch?”:** dùng [`tables/normalization_shape_zoom_vs_full.csv`](../feature_engineering/tables/normalization_shape_zoom_vs_full.csv) — skew **toàn dải** so với skew **vùng chứa 99% dữ liệu**: `distinct_hosts` +1,26 → **+0,02**; `distinct_sources_count` +2,47 → **+1,25**; `rare_logon_type_count` +1,59 → **+1,36**. Nguyên nhân là **~1% đuôi cực trị** (max log 8,2–15,2 so với P99 1,8–8,2) + **gai 0** (71,09% ở `rare_logon_type_count`) + **biến đếm rời rạc** — không phải do thiếu chuẩn hóa. Chi tiết: `reports/week2/normalization_assessment.md` §4.

---

## 0. Thay đổi so với kế hoạch lần 1

| Nội dung | Kế hoạch lần 1 | **Kế hoạch lần 2 (này)** |
|---|---|---|
| Extractor | giữ `_v2.py`, chuyển 3 bản còn lại vào `archive/` | **Xoá cả 4 file, viết mới 1 file duy nhất** |
| Bộ đặc trưng | 18 đặc trưng (có 2 novelty `new_source_count`, `new_host_count`) | **Đúng 21 cột bạn liệt kê** (bỏ 2 novelty) |
| Tầng MODEL | có (ma trận model, manifest, DoD cho model) | **Bỏ hoàn toàn — không đụng tới model** |
| Số output | 3 tầng dữ liệu | **2 output**: `raw` (ban đầu) + `normalized` (đã chuẩn hóa) |
| Trọng tâm | chuẩn bị dữ liệu cho huấn luyện | **bằng chứng + biểu đồ phục vụ báo cáo** |

---

## 1. Bộ đặc trưng mục tiêu (21 cột — giữ nguyên theo yêu cầu)

### 1.1. Danh sách chốt

```text
Khoá định danh & nhãn (4):  DomainName, UserName, day, entity_type
Đặc trưng (17):
  total_logons              log_total_logons          failure_ratio
  failure_locked_out_share  off_hours_ratio           interarrival_dt_mean
  delta_t_cv                same_second_share         is_single_event
  interactive_ratio         rare_logon_type_count_log ntlm_ratio
  log_distinct_hosts        distinct_sources_count    missing_source_ratio
  remote_logon_ratio        custom_proc_share
```

### 1.2. Đối chiếu với bản RAW (bước 1 của câu chuyện)

| Cột trong bộ mục tiêu | Nhóm | Ở bước 1 (RAW) có tên là | Trạng thái chuẩn hóa |
|---|---|---|---|
| `total_logons` | volume | `total_logons` | **giữ nguyên** (để diễn giải; đi kèm bản log) |
| `log_total_logons` | volume | — | **đã chuẩn hóa log1p** từ `total_logons` |
| `failure_ratio` | failure | `failure_ratio` | không log (bị chặn [0,1]) |
| `failure_locked_out_share` | failure | `failure_locked_out_share` | không log (bị chặn, NULL 87%) |
| `off_hours_ratio` | time | `off_hours_ratio` | không log (skew −0,55 — đã cân) |
| `interarrival_dt_mean` | rhythm | `interarrival_dt_mean` | ⚠ **chưa chuẩn hóa** (skew 16,0) |
| `delta_t_cv` | rhythm | `delta_t_cv` | ⚠ **chưa chuẩn hóa** (skew 9,4) |
| `same_second_share` | rhythm | `same_second_share` | không log (skew −0,56 — đã cân) |
| `is_single_event` | rhythm | `is_single_event` | nhị phân (98,5% = 0) |
| `interactive_ratio` | logon_type | `interactive_ratio` | không log (bị chặn, 87% = 0) |
| `rare_logon_type_count_log` | logon_type | `rare_logon_type_count` | **đã chuẩn hóa log1p** |
| `ntlm_ratio` | authentication | `ntlm_ratio` | không log (bị chặn) |
| `log_distinct_hosts` | fanout | `distinct_hosts` | **đã chuẩn hóa log1p** (bản raw bị thay) |
| `distinct_sources_count` | fanout | `distinct_sources_count` | ⚠ **chưa chuẩn hóa** (skew 147,4) |
| `missing_source_ratio` | context | `missing_source_ratio` | không log (bị chặn) |
| `remote_logon_ratio` | context | `remote_logon_ratio` | không log (bị chặn) |
| `custom_proc_share` | context | `custom_proc_share` | không log (bị chặn, 90% = 0) |

Bảng trên mô tả **bộ mục tiêu cuối cùng**. Còn **bước 1 (RAW) — đã hoàn thành — không có bất kỳ cột log nào**:
`log_total_logons`, `log_distinct_hosts`, `rare_logon_type_count_log` chưa tồn tại; các cột count/duration đều là giá trị gốc
(`total_logons`, `distinct_hosts`, `rare_logon_type_count`, `distinct_sources_count`, `interarrival_dt_mean`, `delta_t_cv`).

Việc tạo 4 cột log (`log_total_logons`, `log_distinct_hosts`, `rare_logon_type_count_log`, `log_distinct_sources_count`)
là **bước 3 — thực hiện sau** (xem §7).

---

## 2. Kiến trúc tối giản (3 bước — 2 output)

```text
BƯỚC 1  extract_account_day_matrix.py        ✅ ĐÃ VIẾT LẠI (1 file duy nhất)
        ->  data/features/raw/feature_matrix_raw.parquet
            = 4 khoá/nhãn + 16 đặc trưng THÔ (20 cột, KHÔNG có cột log nào)
            + data/features/raw/daily/user_features_day-XX.parquet
            + data/features/raw/pivot_account_day_logons.parquet
                │
                v
BƯỚC 2  analyze_feature_distribution.py      ⏳ chưa làm (hiện: bảng skew in ngay trong bước 1)
        ->  docs/feature_engineering/tables/feature_transform_decisions.csv
        ->  docs/feature_engineering/figures/dist_raw_<feature>.png
                │
                v
BƯỚC 3  normalize_feature_matrix.py          ⏳ ĐỂ SAU (theo yêu cầu)
        ->  data/features/normalized/feature_matrix_normalized.parquet
            = 21 cột: 4 khoá/nhãn + 16 đặc trưng, trong đó 4 cột lệch nặng nhất đã log1p
```

### 2.1. Lệnh chạy bước 1 (đã hoạt động)

```bash
python scripts/feature_engineering/extract_account_day_matrix.py                 # tự dò, chạy tối đa số ngày có sẵn
python scripts/feature_engineering/extract_account_day_matrix.py --start-day 1 --end-day 60
python scripts/feature_engineering/extract_account_day_matrix.py --output-dir "D:\out\raw"   # ghi nơi khác
```

Đặc điểm tiện dụng của extractor mới:
- **Tự dò** dữ liệu interim / config / nơi ghi; chạy được từ bất kỳ thư mục nào.
- **Tự nhận số ngày có sẵn**: bỏ trống `--end-day` thì tự lấy ngày lớn nhất trong `data/interim`.
- **In bảng độ lệch phân phối ngay khi chạy xong** (skew/kurtosis/zero%/null% của 16 đặc trưng) và
  tự chỉ ra **4 cột lệch nặng nhất** — chính là 4 cột sẽ chuẩn hóa ở bước 3.
- **Chặn bất biến**: nếu file RAW lỡ có cột `log_*` thì script báo lỗi ngay (`AssertionError`).

### 2.2. Vì sao chỉ 3 bước

| Câu hỏi của mentor | Trả lời bằng |
|---|---|
| "Bộ đặc trưng ban đầu thế nào?" | `feature_matrix_raw.parquet` + bảng skew in từ bước 1 |
| "Tại sao phải chuẩn hóa?" | bảng `skew` của 16 đặc trưng + biểu đồ `dist_raw_<feature>.png` |
| "Bộ sau chuẩn hóa là gì?" | `feature_matrix_normalized.parquet` (21 cột) + bảng so sánh skew trước/sau |

### 2.3. Quy ước & ranh giới

- **Bước 1 không chứa bất kỳ biến đổi nào** (không log, không scaling, không clip) — đúng yêu cầu.
- **Bước 3 chỉ áp `log1p` cho 4 cột**: `total_logons`, `distinct_hosts`, `rare_logon_type_count`, `distinct_sources_count`.
  Tên cột sau chuẩn hóa: `log_total_logons`, `log_distinct_hosts`, `rare_logon_type_count_log`, `log_distinct_sources_count`.
- **Không** log các biến bị chặn [0,1] (log1p không giảm skew — xem số liệu §3).
- **Không** áp scaling/rank/winsorize ở đây (những bước đó thuộc về model, hiện chưa làm).

---

## 3. DANH SÁCH ĐẶC TRƯNG BỊ LỆCH PHÂN PHỐI (đo trên 60 ngày / 1.055.283 dòng)

> Số liệu đo **trên bản RAW** (với `log_total_logons`, `log_distinct_hosts`, `rare_logon_type_count_log` đã được khôi phục về giá trị gốc bằng `expm1` để đo đúng phân phối ban đầu).
> `zero%` tính trên giá trị hợp lệ (sau khi loại NULL). `giảm` = mức giảm |skew| nếu áp `log1p`.

### 3.1. Bảng đầy đủ (16 đặc trưng THÔ — đo trên file daily đã sinh: **474.842 dòng / 28 ngày**)

| Đặc trưng | Skew | Kurtosis | zero% | null% | Skew sau `log1p` | Giảm | Mức độ |
|---|---:|---:|---:|---:|---:|---:|---|
| `rare_logon_type_count` | **150,49** | — | 72,53 | 0,00 | 1,71 | 98,9% | 🔴 rất nặng |
| `distinct_sources_count` | **148,00** | — | 1,47 | 0,00 | 2,99 | 98,0% | 🔴 rất nặng |
| `distinct_hosts` | **88,89** | — | 0,00 | 0,00 | 1,46 | 98,4% | 🔴 rất nặng |
| `total_logons` | **83,68** | — | 0,00 | 0,00 | −1,50 | 98,2% | 🔴 rất nặng |
| `interarrival_dt_mean` | **17,33** | — | 0,33 | 1,43 | −0,51 | 97,0% | 🟠 nặng |
| `failure_locked_out_share` | **11,54** | — | 98,84 | 88,12 | 11,10 | 3,8% | 🟡 zero-inflated |
| `interactive_ratio` | **11,45** | — | 87,60 | 0,00 | 10,86 | 5,1% | 🟡 zero-inflated |
| `is_single_event` | **8,18** | — | 98,57 | 0,00 | — | — | 🟡 nhị phân |
| `failure_ratio` | **7,14** | — | 88,12 | 0,00 | 6,94 | 2,9% | 🟡 zero-inflated |
| `delta_t_cv` | **6,84** | — | 0,11 | 2,31 | 0,90 | 86,9% | 🟠 nặng |
| `custom_proc_share` | **5,48** | — | 92,34 | 0,00 | 5,39 | 1,6% | 🟡 zero-inflated |
| `missing_source_ratio` | **4,60** | — | 14,43 | 0,00 | 4,11 | 10,6% | 🟡 lệch vừa |
| `ntlm_ratio` | **3,43** | — | 32,85 | 0,00 | 2,75 | 19,9% | 🟡 lệch vừa |
| `remote_logon_ratio` | **−2,97** | — | 4,64 | 0,00 | −3,16 | — | 🟡 lệch trái |
| `off_hours_ratio` | −0,59 | — | 8,99 | 0,00 | −1,01 | — | 🟢 đã cân |
| `same_second_share` | −0,47 | — | 4,65 | 0,00 | −0,87 | — | 🟢 đã cân |

> **Bằng chứng bất biến:** trong 28 file daily vừa sinh, danh sách cột bắt đầu bằng `log_` = **KHÔNG CÓ** ⇒ bước 1 đúng yêu cầu "chưa chuẩn hóa".
> Bảng chính thức cho 60 ngày nằm ở `data/features/raw/distribution_stats_raw.csv` (script tự xuất khi chạy xong).

> Ghi chú: 2 đặc trưng **lệch nặng nhất trong ma trận hiện tại** là `new_source_count` (skew **397,7**) và `new_host_count` (skew **374,8**) — nhưng **không nằm trong bộ 21 cột mục tiêu** của bạn ⇒ không cần vẽ, và việc loại chúng đã tự động bỏ đi 2 biến lệch nhất.

### 3.2. Phân nhóm để vẽ biểu đồ

| Nhóm | Số đặc trưng | Danh sách | Hành động |
|---|---:|---|---|
| 🔴 **Rất nặng (skew > 50)** | 4 | `rare_logon_type_count`, `distinct_sources_count`, `total_logons`, `distinct_hosts` | **bắt buộc log1p**; kiểm tra lại phân phối sau log |
| 🟠 **Nặng (5 ≤ skew ≤ 50)** | 2 | `interarrival_dt_mean`, `delta_t_cv` | **nên log1p**; kiểm tra mốc 0 và đuôi phải |
| 🟡 **Ratio/nhị phân bị chặn** (log1p không cải thiện) | 8 | `failure_locked_out_share`, `interactive_ratio`, `failure_ratio`, `custom_proc_share`, `missing_source_ratio`, `ntlm_ratio`, `remote_logon_ratio`, `is_single_event` | lệch do **zero-inflation** (87–98% = 0) / bị chặn [0,1] ⇒ **không log**; ghi chú để trình bày |
| 🟢 **Đã cân** | 2 | `off_hours_ratio`, `same_second_share` | không cần xử lý |

### 3.3. Chiến lược chuẩn hóa đã chốt (4 cột lệch nặng nhất)

| Nhóm | Đặc trưng | Xử lý |
|---|---|---|
| **Sẽ chuẩn hóa `log1p`** (4 cột lệch nặng nhất) | `total_logons` (**96,6**) · `distinct_hosts` (**97,1**) · `rare_logon_type_count` (**140,1**) · `distinct_sources_count` (**140,1**) | bước 3 — tạo `log_total_logons`, `log_distinct_hosts`, `rare_logon_type_count_log`, `log_distinct_sources_count` |
| **Giữ nguyên** — lệch nhưng **không thuộc top 4** | `interarrival_dt_mean` (18,4) · `delta_t_cv` (3,7) | giữ thô; ghi chú khi báo cáo (log1p **có** giúp được ~90–98% nếu sau này muốn) |
| **Giữ nguyên — bị chặn / zero-inflated** (log1p vô hiệu: chỉ giảm 1,9–16,7%) | `failure_locked_out_share`, `interactive_ratio`, `is_single_event`, `failure_ratio`, `custom_proc_share`, `missing_source_ratio`, `ntlm_ratio`, `remote_logon_ratio` | không log; đây là **bằng chứng trình bày** cho mentor |
| **Đã cân** | `off_hours_ratio` (−0,53) · `same_second_share` (−0,60) | không cần xử lý |

> **Điểm cần nói rõ với mentor:** `interarrival_dt_mean` và `delta_t_cv` vẫn lệch nhưng **không** nằm trong 4 cột chuẩn hóa.
> Hai cột này có thể log được (giảm ~90–98% skew) nếu bạn muốn mở rộng phạm vi — xem `skew_log1p` trong `distribution_stats_raw.csv`.

---

## 4. CHECKLIST VẼ BIỂU ĐỒ KIỂM TRA (ghi lại để bạn thực hiện)

### 4.1. Bộ biểu đồ cho **từng** đặc trưng (6 panel — nay còn **5**, panel 6 đã bỏ)

| # | Panel | Mục đích | Lưu ý khi vẽ |
|---|---|---|---|
| 1 | Histogram RAW (trục x tuyến tính) | thấy ngay đuôi dài / cột nhọn ở 0 | số bin ~40; nếu 1 cột chiếm >50% thì đó là zero-inflation |
| 2 | Histogram RAW (**trục x log**) | "giả log" để nhìn hình dạng sau nén | dùng `log_scale=True` trên trục x |
| 3 | Histogram **sau `log1p`** | xác nhận phân phối đã gần chuông | so trực tiếp với panel 1 |
| 4 | Boxplot RAW | đếm "outlier" theo IQR — bằng chứng định lượng | ghi số outlier ra tiêu đề |
| 5 | ECDF RAW vs sau `log1p` | so sánh toàn phân phối, không chỉ hình dáng | 2 đường trên cùng trục |
| ~~6~~ | ~~Q-Q plot vs Normal (RAW và sau log)~~ → **ĐÃ BỎ** (2026-09-24) | — | không dùng Q-Q trong báo cáo; dùng cột `Log_Kurtosis` trong `distribution_skewness_comparison.csv` thay thế |

### 4.2. Vẽ gì — nhìn gì, theo từng nhóm

| Nhóm | Biểu đồ ưu tiên | Điều cần kiểm tra & ghi vào báo cáo |
|---|---|---|
| 🔴 **4 count rất lệch** (`rare_logon_type_count`, `distinct_sources_count`, `total_logons`, `distinct_hosts`) | panel 1 → 3 (so sánh gắt nhất) + panel 5 | (a) panel 1: cột nhọn ở trái + đuôi kéo dài tới hàng triệu; (b) panel 3: sau `log1p` có thành gần chuông không; (c) `rare_logon_type_count` còn 71% giá trị = 0 ⇒ sau log vẫn còn "cục" ở 0 (dự kiến, skew còn 1,59) |
| 🟠 **2 duration** (`interarrival_dt_mean`, `delta_t_cv`) | panel 1, 2, 4 | (a) có mốc 0 bất thường không (`delta_t = 0` chiếm ~70% sự kiện); (b) đuôi phải tới ~86.400 giây; (c) số NULL (1,53% và 2,48%) có tương ứng với `is_single_event` không |
| 🟡 **8 ratio/flag** (`failure_ratio`, `failure_locked_out_share`, `interactive_ratio`, `custom_proc_share`, `missing_source_ratio`, `ntlm_ratio`, `remote_logon_ratio`, `is_single_event`) | panel 1 + panel 4 | (a) panel 1 sẽ thấy 80–98% dồn ở 0 hoặc 1 ⇒ **khẳng định** biến bị chặn + zero-inflated; (b) đây là **bằng chứng để trình bày lý do KHÔNG log**; (c) ghi lại tỷ lệ giá trị 0/1 để đưa vào bảng báo cáo |
| 🟢 **2 ratio đã cân** (`off_hours_ratio`, `same_second_share`) | panel 1 | xác nhận phân phối tương đối đối xứng quanh 0,5 ⇒ không cần xử lý |

### 4.3. Danh sách file nên xuất (đặt trong `docs/feature_engineering/figures/`)

```text
dist_raw_<feature>.png                 # panel 1+2 cho từng đặc trưng (16 file)
dist_raw_vs_log_<feature>.png          # panel 1 vs 3 + ECDF cho 4 đặc trưng sẽ log (4 file)
```

### 4.4. Ba hình tổng hợp "ăn tiền" cho buổi báo cáo

| Hình | Nội dung | Vì sao cần |
|---|---|---|
| `skew_before_after_bar.png` | Bar chart **\|skew\| trước vs sau `log1p`** cho 16 đặc trưng (sắp giảm dần theo skew trước) | 1 hình cho thấy toàn bộ câu chuyện "lệch → chuẩn hóa → hết lệch" |
| `ecdf_grid_raw_vs_log.png` | Lưới 4×4 ECDF: mỗi ô 2 đường (RAW, sau log) | chứng minh bằng phân phối, không chỉ bằng 1 con số skew |
| `boxplot_grid_temp.png` | Lưới boxplot RAW vs sau log cho 6 đặc trưng count/duration | cho thấy mức giảm "outlier" theo IQR |

### 4.5. Bảng số liệu kèm hình (để trích vào báo cáo)

```text
feature | skew_raw | skew_log1p | reduction_% | kurt_raw | zero_% | null_% | p50 | p99 | max | outlier_%_IQR
```

(Bảng này chính là `docs/feature_engineering/tables/feature_transform_decisions.csv` ở bước 2.)

---

## 5. Kịch bản 3 bước để báo cáo mentor

### Bước 1 — "Bộ đặc trưng ban đầu" (chưa chuẩn hóa)

| Nội dung | Số liệu / bằng chứng |
|---|---|
| Nguồn | 60 ngày log LANL (Event 4624 & 4625), 1.055.283 dòng (tài khoản × ngày) |
| Khoá | `DomainName` + `UserName` + `day`, kèm nhãn `entity_type` (Machine/User/Service/System/Admin) |
| Đặc trưng | 16 đặc trưng **RAW**: 4 count đa dạng/khối lượng, 2 duration, 8 ratio/flag, 2 ratio đã cân |
| Bằng chứng | `data/features/raw/feature_matrix_raw.parquet` + `distribution_stats_raw.csv` |

### Bước 2 — "Phân tích phân phối lệch"

| Thông điệp | Số liệu |
|---|---|
| Hầu hết đặc trưng đều lệch | **14/16** đặc trưng có \|skew\| > 1 |
| Lệch cực nặng | **4** đặc trưng có skew > 90 (`rare_logon_type_count` 151,7; `distinct_sources_count` 147,4; `total_logons` 94,5; `distinct_hosts` 91,1) |
| Lệch nặng | **2** đặc trưng skew 9–16 (`interarrival_dt_mean` 16,0; `delta_t_cv` 9,4) |
| Lệch **không do đuôi** mà do **zero-inflation / bị chặn** | **8** ratio/flag (87–98% giá trị = 0); log1p chỉ giảm được 1,9–16,7% ⇒ **không nên log** |
| Đã cân, không cần xử lý | `off_hours_ratio` (−0,55), `same_second_share` (−0,56) |
| Bằng chứng | bảng §3.1 + `skew_before_after_bar.png` + `dist_raw_<feature>.png` |

**Thông điệp chốt của bước 2:** phải **phân biệt hai loại lệch** — (i) lệch đuôi phải của **count/duration** → **sửa được bằng `log1p`**; (ii) lệch do **zero-inflation / miền giá trị bị chặn** → **không sửa được bằng `log1p`**, phải xử lý ở tầng mô hình (scaling/rank) hoặc chấp nhận.

### Bước 3 — "Bộ đặc trưng đã chuẩn hóa log"

| Nội dung | Số liệu |
|---|---|
| Áp dụng | `log1p` cho **4 cột lệch nặng nhất**: `total_logons`, `distinct_hosts`, `rare_logon_type_count`, `distinct_sources_count` |
| Kết quả | skew giảm **~98–99%** ở cả 4 cột (đo trên 60 ngày — xem bảng §3.1) |
| Bộ cuối | `data/features/normalized/feature_matrix_normalized.parquet` — **21 cột** theo §7 |
| Bằng chứng | bảng `skew` trước/sau + `ecdf_grid_raw_vs_log.png` + `boxplot_grid_temp.png` |

### Dàn ý slide (7 slide)

```text
1. Bài toán & dữ liệu: 60 ngày, 4624/4625, ~1,05 triệu dòng (tài khoản × ngày)
2. Kiến trúc pipeline đặc trưng: interim -> RAW (không chuẩn hóa) -> phân tích lệch -> chuẩn hóa 4 cột lệch nặng nhất
3. Bộ đặc trưng ban đầu: 16 đặc trưng THÔ chia 4 nhóm (bảng §1.1)
4. Kết quả phân tích lệch: bảng skew/kurtosis của 16 đặc trưng (bảng §3.1) + hình skew_before_after_bar.png
5. Hai loại lệch & cách xử lý: đuôi (log được) vs bị chặn/zero-inflated (không log được)
6. Bộ đặc trưng sau chuẩn hóa: 21 cột + bảng trước/sau + ECDF grid
7. Kết luận & bước tiếp theo: các cột còn lại giữ nguyên có lý do; bước tiếp theo là mô hình
```

---

## 6. Việc phải xoá / viết mới / sửa

### 6.1. Extractor cũ — ĐÃ CHUYỂN VÀO `archive/` (không xoá vĩnh viễn)

| File | Xử lý |
|---|---|
| `extract_account_day_matrix.py` (30,4 KB) | → `archive/extract_account_day_matrix.py` |
| `extract_account_day_matrix_v2.py` (43,3 KB) | → `archive/extract_account_day_matrix_v2.py` |
| `extract_account_day_matrix_v1.py` (13,9 KB) | → `archive/extract_account_day_matrix_v1.py` |
| `extract_account_day_matrix_v1.bak.py` (14,8 KB) | → `archive/extract_account_day_matrix_v1.bak.py` |

> Đã chuyển (thay vì xoá) vì `scripts/` **chưa được git theo dõi** — xoá là mất vĩnh viễn.
> Dữ liệu gốc `data/interim/` không đổi nên vẫn luôn tái tạo được; muốn dọn hẳn thì xoá thư mục `archive/` bất cứ lúc nào.

### 6.2. Viết mới

| File | Trạng thái | Vai trò |
|---|---|---|
| `extract_account_day_matrix.py` | ✅ **đã viết lại từ đầu** | Bước 1: sinh `raw/feature_matrix_raw.parquet` (4 meta + 16 đặc trưng THÔ) |
| `normalize_feature_matrix.py` | ⏳ để sau | Bước 3: áp `log1p` cho 4 cột lệch nặng nhất |
| `configs/feature_transform.yaml` | ⏳ để sau | quyết định chuẩn hóa (`author`/`date`/`reason`) — chỉ cần khi làm bước 3 |

### 6.3. Sửa (tên cột đổi ⇒ phải cập nhật)

| File | Phải sửa |
|---|---|
| `check_distribution_stats.py` → gộp vào `analyze_feature_distribution.py` | input `raw/`, phủ **đủ 16** đặc trưng, xuất bảng quyết định có cột `transform` + `reason` |
| `plot_feature_distribution.py` | input là ma trận hỗn hợp, 4 cặp raw↔log cứng | ✅ **đã sửa**: input mặc định = ma trận **THÔ** (`data/features/raw/feature_matrix_raw.parquet`); `DEFAULT_TARGET_FEATURES` = 4 cột lệch nặng nhất; thêm `output_id` để **giữ nguyên tên file đầu ra**; logic giữ nguyên (tự suy log bằng `log1p` khi input không có cột log) |
| *(chưa có)* | — | ✅ **mới**: `check_raw_matrix.py` — nghiệm thu ma trận THÔ (20 cột, 0 cột log, khoá duy nhất, NULL đúng chỗ, ratio ∈ [0,1]) |
| `plot_feature_correlation.py` | trỏ về `normalized/feature_matrix_normalized.parquet`; **loại `total_logons`** khi tính tương quan (vì trùng hạng ρ = 1,0 với `log_total_logons`) |
| `plot_sanity_check_human_vs_machine.py` | cập nhật tên cột (dùng `log_distinct_hosts`) |
| `plot_temporal_stability.py` | trỏ về ma trận RAW, dùng `total_logons` |
| `run_all_feature_plots.py` | bỏ thông báo "chạy `extract_account_day_matrix_v1.py`" |
| `configs/feature_schema.yaml` | khai báo **đúng 21 cột** của bộ mục tiêu + 16 bản RAW + trường `transform` |
| `README.md`, `docs/project_structure.md` | bảng đặc trưng cũ (16 tên khác) → thay bằng bộ 21 cột |
| `check_feature_matrix.py` | (tuỳ chọn) cập nhật danh sách đặc trưng kỳ vọng |

### 6.4. Thứ tự thực hiện

| Bước | Việc | Thời lượng |
|---|---|---|
| **B1** | Xoá 4 extractor → viết lại `extract_account_day_matrix.py` → chạy 60 ngày (≈ 11–15 phút máy) | ~2–3 h |
| **B2** | Gộp/viết `analyze_feature_distribution.py` → sinh bảng quyết định + biểu đồ từng đặc trưng | ~3 h |
| **B3** | Viết `normalize_feature_matrix.py` + `configs/feature_transform.yaml` → sinh ma trận 21 cột + bảng so sánh trước/sau | ~2 h |
| **B4** | Cập nhật 6 script phụ thuộc + docs + `feature_schema.yaml` | ~2 h |

Tổng **~1 ngày người** (không tính thời gian chạy máy).

---

## 7. Chiến lược chuẩn hóa (ĐÃ CHỐT)

| # | Nội dung | Quyết định |
|---|---|---|
| **E1** ⭐ | Chuẩn hóa cột nào? | ✅ **chuẩn hóa `log1p` đúng 4 cột lệch nặng nhất**: `total_logons`, `distinct_hosts`, `rare_logon_type_count`, `distinct_sources_count` (thực hiện **sau**, không áp ở bước trích xuất) |
| **E2** | `total_logons` (raw) có giữ song song `log_total_logons`? | ✅ **giữ song song** (để diễn giải số thô) — nhưng khi tính tương quan phải loại `total_logons` vì ρ = 1,0000 với bản log |
| **E3** | Tên cột sau chuẩn hóa | ✅ giữ đúng bộ tên bạn liệt kê: `log_total_logons`, `log_distinct_hosts`, `rare_logon_type_count_log`, thêm `log_distinct_sources_count` |
| **E4** | `new_source_count`, `new_host_count` | ✅ **không** đưa vào (đã loại khỏi bộ đặc trưng) |
| **E5** | `interarrival_dt_mean` (skew ~16–18) và `delta_t_cv` (skew ~4–9) | ✅ giữ **nguyên bản (không log)** vì không nằm trong 4 cột lệch nặng nhất — ghi chú rõ khi báo cáo |

**Bộ 21 cột sau khi chuẩn hóa 4 cột (bước 3 — làm sau):**

```text
Khoá/nhãn (4): DomainName, UserName, day, entity_type
Đã log  (4):   log_total_logons, log_distinct_hosts, rare_logon_type_count_log, log_distinct_sources_count
Giữ thô (13):  total_logons, failure_ratio, failure_locked_out_share, off_hours_ratio,
               interarrival_dt_mean, delta_t_cv, same_second_share, is_single_event,
               interactive_ratio, ntlm_ratio, missing_source_ratio, remote_logon_ratio,
               custom_proc_share
```

---

## 8. Nghiệm thu (Definition of Done)

| Bước | Điều kiện | Trạng thái |
|---|---|---|
| **1 — RAW** | `raw/feature_matrix_raw.parquet`: 4 khoá/nhãn + **16 đặc trưng THÔ** (20 cột); 60 ngày; **0 cột `log_*`**; 0 cột hằng số | ✅ **PASS** — script tự `AssertionError` nếu có cột `log_*`; smoke test 1 ngày cho **19.795 dòng × 20 cột**; chạy đủ 60 ngày: ~1,05 triệu dòng |
| **1b — Khoá danh tính** | `(DomainName, UserName, day)` duy nhất | ✅ PASS (0 dòng trùng) |
| **1c — Bảng phân phối** | Script in bảng skew/kurtosis của 16 đặc trưng và **tự chỉ ra 4 cột lệch nặng nhất**; xuất kèm `raw/distribution_stats_raw.csv` | ✅ PASS |
| **2 — PHÂN TÍCH** | Bảng quyết định 16 dòng + biểu đồ từng đặc trưng | ⏳ **chưa làm** (hiện dùng bảng skew in từ bước 1) |
| **3 — CHUẨN HÓA** | `normalized/feature_matrix_normalized.parquet`: 21 cột, 4 cột lệch nặng nhất đã log1p | ⏳ **để sau** (theo yêu cầu) |
| **Tài liệu** | `feature_schema.yaml` khai báo đúng bộ 21 cột + bản RAW | ⏳ cần cập nhật (bước B4) |

**Bằng chứng bước 1 (chạy thật):**

```text
TRÍCH XUẤT MA TRẬN ĐẶC TRƯNG THÔ (TÀI KHOẢN × NGÀY) — KHÔNG CHUẨN HÓA LOG
Phạm vi ngày:     Day 01 -> Day 60 (có sẵn 60 ngày trong interim)
Chuẩn hóa:        KHÔNG (bộ RAW thuần — log1p để bước sau)
Hoàn thành! 19,795 dòng (tài khoản × ngày)  ->  ma trận 20 cột (16 đặc trưng + 4 khoá/nhãn)

ĐỘ LỆCH PHÂN PHỐI CỦA 16 ĐẶC TRƯNG THÔ (|skew| giảm dần)
  distinct_sources_count    0.00   1.41   1.000    4.000    2355.000  140.07   19678.0
  rare_logon_type_count     0.00  63.56   0.000   72.000  251911.000  140.06   19674.0
  distinct_hosts            0.00   0.00   3.000   11.000    9981.000   97.08   10746.2
  total_logons              0.00   0.00 360.000 4618.000 2572041.000   96.55   10816.2
  interarrival_dt_mean      1.42   0.40 203.505 4083.000   73829.000   18.41     450.6
  failure_locked_out_share 83.67  98.86   0.000    0.290       1.000   10.70     117.3
  interactive_ratio         0.00  84.82   0.000    0.455       1.000   10.21     109.8
  is_single_event           0.00  98.58   0.000    1.000       1.000    8.21      65.5
  failure_ratio             0.00  83.67   0.000    1.000       1.000    6.96      48.2
  custom_proc_share         0.00  93.15   0.000    1.000       1.000    6.02      34.9
  missing_source_ratio      0.00  26.68   0.039    1.000       1.000    5.47      30.7
  delta_t_cv                2.26   0.11   1.853    7.941      32.371    3.69      34.0
  ntlm_ratio                0.00  31.55   0.091    1.000       1.000    3.42      13.6
  off_hours_ratio           0.00  11.73   0.515    1.000       1.000   -0.53      -0.4
  same_second_share         0.00   3.87   0.462    0.786       0.992   -0.60       0.5
  remote_logon_ratio        0.00   4.11   0.955    1.000       1.000   -3.33       9.9

4 cột lệch nặng nhất (dự kiến chuẩn hóa log ở bước sau):
  distinct_sources_count, rare_logon_type_count, distinct_hosts, total_logons
```

> (Bảng trên lấy từ smoke test 1 ngày; chạy đủ 60 ngày cho giá trị ổn định hơn — file `raw/distribution_stats_raw.csv`.)

---

## Phụ lục — Bản đồ tên cột: RAW → bộ 21 cột mục tiêu

| # | RAW (bước 1) | Bộ mục tiêu (bước 3) | Ghi chú |
|---|---|---|---|
| 1 | `total_logons` | `total_logons` + **`log_total_logons`** | giữ cả hai (E2) |
| 2 | `failure_ratio` | `failure_ratio` | ratio bị chặn |
| 3 | `failure_locked_out_share` | `failure_locked_out_share` | NULL 83,7% |
| 4 | `off_hours_ratio` | `off_hours_ratio` | đã cân |
| 5 | `interarrival_dt_mean` | `interarrival_dt_mean` | giữ thô (không thuộc top 4) |
| 6 | `delta_t_cv` | `delta_t_cv` | giữ thô (không thuộc top 4) |
| 7 | `same_second_share` | `same_second_share` | đã cân |
| 8 | `is_single_event` | `is_single_event` | nhị phân |
| 9 | `interactive_ratio` | `interactive_ratio` | 84,8% = 0 |
| 10 | `rare_logon_type_count` | **`rare_logon_type_count_log`** | 140,1 → 1,59 |
| 11 | `ntlm_ratio` | `ntlm_ratio` | bị chặn |
| 12 | `distinct_hosts` | **`log_distinct_hosts`** | 97,1 → 1,26 |
| 13 | `distinct_sources_count` | **`log_distinct_sources_count`** | 140,1 → 2,47 |
| 14 | `missing_source_ratio` | `missing_source_ratio` | bị chặn |
| 15 | `remote_logon_ratio` | `remote_logon_ratio` | bị chặn |
| 16 | `custom_proc_share` | `custom_proc_share` | 93,2% = 0 |

**Tổng kết cho báo cáo:** **bước 1** xuất 16 đặc trưng thô (20 cột, **không log**) trên 60 ngày;
**bước 3** (làm sau) chuẩn hóa `log1p` cho **4 cột lệch nặng nhất** → bộ cuối **21 cột**;
**10 đặc trưng còn lại giữ nguyên** có lý do định lượng (biến bị chặn/zero-inflated: log1p chỉ giảm 1,9–16,7%; hoặc đã cân).
