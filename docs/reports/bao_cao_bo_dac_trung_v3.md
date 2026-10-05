# Báo cáo: Bộ đặc trưng mở rộng v3.0 — Tier A **24 biến model** (16 v2.0 + 8 mới) + 30 biến thử nghiệm + 7 biến chẩn đoán

> **Dự án:** UEBA Anomaly Detection Benchmark (Windows Event 4624 & 4625 — LANL Unified Host and Network Dataset)
> **Ngày thực hiện:** 2026-09-30 · **Cập nhật:** 2026-09-30 (đã triển khai ĐỦ Tier A + đo trên dữ liệu thật)
> **Trạng thái:** ✅ **Tier A đã cài đặt đủ 14 ứng viên** → 8 biến qua cổng (schema lên **24 core**), ❌ 6 biến **đã cài + đo rồi bác bỏ** (xem bảng ngay dưới)
> **Căn cứ:** [`configs/feature_schema.yaml`](../../configs/feature_schema.yaml) · [`docs/feature_engineering/feature_correlation_review.md`](../feature_engineering/feature_correlation_review.md) §7.3, §11 · [`reports/archive/bao_cao_sua_bo_dac_trung.md`](../../reports/archive/bao_cao_sua_bo_dac_trung.md) · [`reports/week2/multicollinearity_check.md`](../../reports/week2/multicollinearity_check.md) · [`reports/week2/normalization_assessment.md`](../../reports/week2/normalization_assessment.md) · [`reports/week3/tom_tat_tuan3.md`](../../reports/week3/tom_tat_tuan3.md)
> **Phạm vi:** đề xuất, biện luận **và** triển khai toàn bộ Tier A. Tier B/C vẫn chỉ là đề xuất (theo thiết kế không nạp vector).
> **Vị trí lưu trữ:** `docs/reports/bao_cao_bo_dac_trung_v3.md`

---

## 1. Tóm tắt điều hành

Hai bộ đặc trưng đang được cân nhắc (43 biến "Core + 25 mới" và 61 biến "9 nhóm") **đều không dùng được nguyên trạng** trong repo này:

1. **Bộ 43** tự mâu thuẫn về số học: khối mã liệt kê **40 biến mới** ⇒ 18 + 40 = **58**, nhưng văn bản chốt **25** rồi **43**; **15 biến bị bỏ khỏi danh sách chốt mà không nêu lý do** (trong đó có cả 5 biến residual và 2 biến calendar). Nó cũng mô tả sai hiện trạng: 3 tên không tồn tại trong code (`is_machine_account`, `new_source_count`, `new_host_count`) và bỏ mất 1 biến đang là core (`custom_proc_share`).
2. **Bộ 61** đúng số học (61 = 16 + 45, không trùng lặp, chứa đủ 16 core) nhưng **17/61 tên đã được đo là chết/gần-hằng, tạo phụ thuộc tuyến tính tuyệt đối hoặc trùng trục thông tin** trên chính dữ liệu LANL (xem §4).

Báo cáo đề xuất bộ **v3.0** gồm 3 tầng tách bạch (không trùng nhau, đã kiểm chứng bằng script):

| Tầng | Số biến | Trạng thái | Có vào vector model? |
|:---|---:|:---|:---:|
| **Tier A — Core model** | **24** (16 v2.0 + 8 mới) | ✅ **đã triển khai đủ**: 8/14 ứng viên qua cổng, 6 bị bác bỏ bằng số đo | ✅ có |
| **Tier B — Experimental** | **30** | đề xuất | ❌ chưa (chỉ khi thắng ablation) |
| **Tier C — Diagnostic / Panel** | **7** | đề xuất | ❌ không (chỉ giải thích & cảnh báo) |

**Đã làm ngay và đo trên dữ liệu thật** — 14 ứng viên được cài đặt, đo trên 10 ngày LANL (180.606 dòng), kết quả **8 biến qua cổng, 6 biến bị loại vì vượt ngưỡng \|ρ\| ≥ 0,85**:

| Bị bác bỏ | ρ đo được | Đối tác |
|:---|---:|:---|
| `max_failure_streak` | **0,9973** | `failure_ratio` |
| `success_after_failure_ratio` | **0,9180** | `failure_ratio` |
| `new_source_count` | **0,9999** | `new_source_count_7d` |
| `new_host_count` | **0,9999** | `new_host_count_7d` |
| `source_recency` | **0,9238** | `days_since_last_activity` |
| `source_host_pair_novelty` | **0,9052** | `new_host_count` |

Bộ cuối cùng (24 core) đạt **0 cặp \|ρ\| ≥ 0,85** và **VIF max 7,31**; 2 lỗi thật được phát hiện nhờ quá trình đo (bug `ProcessName` làm `custom_proc_share` = 0 âm thầm; z khối lượng bùng nổ tới −976 ⇒ phải winsorize ±10) — chi tiết §10.

> **Về con số "18":** nó đến từ khuyến nghị **LỊCH SỬ** của [`feature_correlation_review.md`](../feature_engineering/feature_correlation_review.md) §7.2/§11.4 — "rút **30 → 18**" trên bộ **v1.0 (32 cột)** của Tuần 2; đề xuất đó **chưa bao giờ là schema đang chạy**. Đường đi số lượng thực tế: 32 → 30 (v1.0, lưu trữ) → **16** (v2.0, đã chạy benchmark Tuần 3) → **20** (v3.0 nhóm 9, đã chạy ở commit `1d2209d`) → **24** (v3.0 đủ Tier A, hiện tại).

**Không đưa thẳng 43/61 biến vào vector**, vì: (a) 7/16 biến v2.0 đã có **MAD = 0** và nhiều biến zero-inflated 87–99% ⇒ thêm biến tương tự chỉ làm loãng khoảng cách; (b) 5 mô hình đang dùng đều dựa trên khoảng cách (IForest/LOF/OCSVM) nên 61 chiều cần chọn lọc; (c) **Tuần 3 kết luận không thể xếp hạng bằng chỉ số label-free** (Top-20 overlap ≈ 0) ⇒ chỉ có ablation **có nhãn `redteam.txt` ở Tuần 4** mới được quyền chốt.

---

## 2. Điểm neo: hiện trạng thực của bộ đặc trưng

| Hạng mục | Giá trị đã kiểm chứng | Nguồn |
|:---|:---|:---|
| Số đặc trưng model hiện tại | **16** (schema `core: true`), pin cứng bằng test | `configs/feature_schema.yaml`; `tests/test_configs.py:37` |
| Cột extractor sinh ra | **20** cột (`RAW_ORDERED_COLS`), **không** có `new_source_count`/`new_host_count` | `src/features/extractor.py:36-46` |
| `new_source_count`/`new_host_count` | Chỉ tồn tại ở **bản lưu trữ**, không nằm trong pipeline | `scripts/feature_engineering/archive/extract_account_day_matrix_v2.py:213-245` |
| Chất lượng đa cộng tuyến bộ 16 | Cặp mạnh nhất \|ρ\| = **0,678**; VIF lớn nhất **7,63** (`remote_logon_ratio`) | `reports/week2/multicollinearity_check.md` §3.1–3.2 |
| Zero-inflation (đo trên 1.055.283 dòng) | `is_single_event` 98,47% · `failure_locked_out_share` 98,86% · `custom_proc_share` 90,33% · `failure_ratio` 87,36% · `interactive_ratio` 87,00% · `rare_logon_type_count` 71,09% | `reports/week2/normalization_assessment.md` Bảng 1 |
| NULL hợp lệ | `failure_locked_out_share` 85,46% · `delta_t_cv` 2,22% · `interarrival_dt_mean` 1,40% | `reports/archive/bao_cao_sua_bo_dac_trung.md` §10 |

**Hệ quả trực tiếp:** cả hai bộ đang so sánh đều **tính thiếu chi phí triển khai** — toàn bộ novelty/recency/lag/rolling/peer đều là **code phải viết mới**. Trạng thái dữ liệu tại workspace (đã kiểm lại khi triển khai):

| Đường dẫn | Nội dung | Dùng được cho |
|:---|:---|:---|
| `data/raw/wls_day-01..60.bz2` | 60 ngày log thô LANL | `python main.py --stage clean` |
| `data/interim/event_4624|4625/*.parquet` | 60 ngày mỗi loại, có cả `ProcessName` | `python main.py --stage features` (fallback) |
| `data/processed/feature_matrix_processed.parquet` | **1.055.283 × 23** — artifact của **schema v2.0 (16 core)** | phải **dựng lại** để dùng v3.0 |
| `data/cleaned/` | chỉ có `.gitkeep` | — (đường `cleaned` không dùng được) |

⇒ Có thể dựng lại ma trận và **đo ngay** (đã làm, xem §10); chỉ 8 biến novelty/recency còn lại mới cần Dense Panel.

---

## 3. Kiểm chứng số học hai bộ (chạy tự động, không suy luận tay)

Script kiểm chứng đối chiếu trực tiếp với `FeatureSchema().core_features` (script tạm, đã xoá sau khi lấy kết quả):

| Chỉ số kiểm chứng | Bộ 1 | Bộ 2 |
|:---|---:|---:|
| Số biến khai báo trong văn bản | 18 + 25 = **43** | **61** |
| Số biến thực có trong khối mã | 18 + **40** = **58** | 61 |
| Số biến không trùng lặp | 58 | 61 (0 trùng) |
| Chứa đủ **16 core** hiện có? | ❌ **thiếu `custom_proc_share`**, thêm 3 tên lạ | ✅ đủ 16/16 |
| Biến có trong mã nhưng **không** có trong danh sách chốt | **15** | 0 |
| Biến "lơ lửng" (chỉ nhắc ở tầng chẩn đoán) | **1** (`behavior_shift_score_7d`) | 0 |
| Tự mâu thuẫn tiêu đề vs nội dung | ✅ có (`Nhóm Temporal (7 đặc trưng)` nhưng liệt kê 9) | ❌ không |

**15 biến của Bộ 1 bị rơi khỏi danh sách "43" mà không có lý do kèm theo:**
`interarrival_dt_p95`, `new_source_count_14d`, `new_host_count_14d`, `host_set_novelty`, `volume_rolling_std_7d`, `volume_percentile_14d`, `failure_ratio_rolling_mean_7d`, `failure_ratio_robust_z_7d`, `volume_seasonal_deviation_7d`, `volume_seasonal_zscore_7d`, `volume_residual`, `residual_robust_zscore`, `residual_abs_exceed_3sigma`, `day_of_week`, `is_weekend`.

> Đây **không phải lỗi câu chữ** mà là lỗi quy trình: repo đã thống nhất chuẩn "mọi biến bị loại phải có ρ/VIF/prevalence kèm lý do" (mục `removed:` của `configs/feature_schema.yaml` ghi đủ **25 biến** với số đo). Một danh sách bỏ 15 biến không lý do **không thể đưa vào hợp đồng schema**.

**Hai bộ giao nhau đúng 8 tên** — nghĩa là chúng **không phải hai phiên bản của cùng một bộ**, mà là hai triết kế khác nhau: Bộ 1 nghiêng về *chuỗi thời gian hành vi* (lag/rolling/recency), Bộ 2 nghiêng về *bao phủ chức năng* (logon-type, đồ thị, peer). Vì vậy "chọn bộ nào" thực chất là **chọn tổ hợp theo tầng**, không phải chọn một trong hai.


---

## 4. Danh sách loại trừ vĩnh viễn (23 tên) kèm bằng chứng đo được

Không đưa vào bất kỳ tầng nào (kể cả thử nghiệm), vì lý do **quyết định** — không phải "chưa thử":

| # | Biến | Lý do loại (số đo) | Nguồn |
|---:|:---|:---|:---|
| 1 | `bad_password_ratio` | `Status`/`SubStatus` **NULL 100%** ở 4624 và 4625 ⇒ luôn = 0 | `reports/archive/validation_summary.md` |
| 2 | `user_not_exist_ratio` | cùng lý do trên (đúng 2 cột `wrong_password_count`/`unknown_user_count` đã bị xoá ở v1.0) | `configs/feature_schema.yaml` → `removed` |
| 3 | `logon_type_10_ratio` | Type 10 = **2.230/17.553.517 (0,013%)**; 96,33% dòng = 0 | review Phụ lục A.1; `rdp_ratio` đã bị bỏ |
| 4 | `logon_type_4_ratio` | Type 4 = 0,013% sự kiện; 99,57% dòng = 0 | review Phụ lục A.1; `batch_ratio` đã bị bỏ |
| 5 | `logon_type_5_ratio` | Type 5 = 2,129% sự kiện; 99,73% dòng = 0 | review Phụ lục A.1; `service_ratio` đã bị bỏ |
| 6 | `logon_type_9_ratio` | Type 9 = 0,077% sự kiện (chỉ có ở 4624) | review Phụ lục A.1 |
| 7 | `logon_type_3_ratio` | Type 3 = **95,676%** sự kiện ⇒ gần hằng số; trùng vai trò `network_ratio` (median 1,000, chỉ 2,46% dòng = 0; ρ = 0,9031 với `rare_logon_type_count_log`) | review Phụ lục A.1; `network_ratio` đã bị bỏ |
| 8 | `night_ratio` | `night + morning + afternoon + evening = 1` **tuyệt đối** | bài học VIF ≈ 5,2e7 của `other_logon_type_ratio` |
| 9 | `morning_ratio` | đóng ràng buộc tổng (như #8) | như trên |
| 10 | `afternoon_ratio` | đóng ràng buộc tổng (như #8) | như trên |
| 11 | `evening_ratio` | đóng ràng buộc tổng (như #8) | như trên |
| 12 | `interarrival_autocorr_lag1` | **70,57% sự kiện có Δt = 0** (LANL phân giải giây) + chuỗi trong-ngày rất ngắn ⇒ tự tương quan không ổn định | `configs/feature_schema.yaml` (note `same_second_share`) |
| 13 | `interarrival_dt_p10` | cùng một phân phối Δt ⇒ tương quan cao với p50/mean | `interarrival_dt_std` đã bị bỏ vì ρ = 0,6020 với mean |
| 14 | `interarrival_dt_median` | trùng vai trò `interarrival_dt_p50` | như trên |
| 15 | `interarrival_dt_p90` | cùng một phân phối Δt (chỉ giữ **1 đại diện phân vị** là `interarrival_dt_p50`) | như trên |
| 16 | `interarrival_dt_skewness` | cùng một phân phối Δt + rất nhạy với zero-inflation | như trên |
| 17 | `src_to_dst_cardinality_ratio` | biến thể tỷ số của `distinct_sources_count` / `log_distinct_hosts` (đã có) | `distinct_hosts_count` bị bỏ vì ρ = 1,0000 |
| 18 | `is_weekend` | hàm xác định của `day_of_week` ⇒ cộng tuyến tuyệt đối nếu cùng nạp | nguyên tắc "không đóng ràng buộc" |
| 19 | `is_machine_account` | đã có `entity_type` (nhãn `Machine`) làm biến tầng ⇒ không cần cờ nhị phân vào vector | `src/features/extractor.py:74-86` |
| 20 | `volume_rolling_std_7d` | trùng vai trò với `volume_robust_z_7d` (cùng đo độ tán volume 7 ngày) | — |
| 21 | `volume_percentile_14d` | trùng vai trò với `volume_robust_z_7d` | — |
| 22 | `volume_seasonal_deviation_7d`, `volume_seasonal_zscore_7d` | trùng `volume_residual` / `residual_robust_zscore` (đều là phần dư so với baseline chu kỳ) — chỉ giữ bản có kiểm định | — |

> Lưu ý #22 gộp 2 tên ⇒ tổng biến bị loại là **23 tên**. Hai biến bị loại vì *dữ liệu* (#1, #2) là **điều kiện tiên quyết**: nếu bộ nào vẫn liệt kê chúng thì bộ đó chưa được rà trên dữ liệu thật.


---

## 5. Bộ đặc trưng đề xuất v3.0

### 5.1. Ba nguyên tắc chọn (kế thừa từ §7/§11 của review và bài học Tuần 2–3)

1. **Không nhận biến khi chưa đo:** mỗi biến phải qua cổng \|ρ\| ≥ 0,85 (Spearman/Pearson), VIF > 10, prevalence/MAD (loại nếu gần hằng số hoặc > 95% giá trị bằng 0 mà không mang tín hiệu tấn công), và **không** tham gia ràng buộc tổng = 1.
2. **Không biến nào phụ thuộc tương lai:** mọi biến lịch sử/cửa sổ phải tính trên **dữ liệu ≤ t−1** (nhân quả). Biến novelty phải ghi rõ mốc "chưa từng thấy **trước ngày t**".
3. **Một trục thông tin chỉ giữ 1–2 đại diện:** trục *volume-vs-baseline* giữ 1 biến, trục *Δt phân phối* giữ mean + CV + 1 phân vị, trục *novelty theo cửa sổ* giữ 7d (14d chỉ là thử nghiệm).

### 5.2. Tier A — 24 biến vào vector model (sau khi đo: 16 v2.0 + 8 biến mới qua cổng)

**A.1. 16 biến core hiện có (giữ nguyên, không đổi định nghĩa)**

| # | Biến | Nhóm | Định nghĩa | Nguồn | Transform |
|---:|:---|:---|:---|:---|:---|
| 1 | `log_total_logons` | volume | `log1p(số sự kiện 4624+4625 trong ngày)` | v2.0 | none |
| 2 | `failure_ratio` | failure | `failure_count / total_logons` | v2.0 | none |
| 3 | `failure_locked_out_share` | failure | `#(FailureReason ~ 'account locked out') / failure_count` | v2.0 | none (nullable) |
| 4 | `off_hours_ratio` | time | tỷ lệ sự kiện có `hour ≥ 18` hoặc `hour ≤ 7` | v2.0 | none |
| 5 | `interarrival_dt_mean` | rhythm | trung bình Δt (giây) giữa 2 sự kiện liên tiếp | v2.0 | `log1p` |
| 6 | `delta_t_cv` | rhythm | `std(Δt) / mean(Δt)` | v2.0 | none (nullable) |
| 7 | `same_second_share` | rhythm | tỷ lệ sự kiện có Δt = 0 | v2.0 | none |
| 8 | `is_single_event` | rhythm | 1 nếu ngày chỉ có đúng 1 sự kiện | v2.0 | none |
| 9 | `interactive_ratio` | logon_type | `#(LogonType = 2) / total_logons` | v2.0 | none |
| 10 | `rare_logon_type_count_log` | logon_type | `log1p(#(LogonType ∉ {2,3}))` | v2.0 | `log1p` |
| 11 | `ntlm_ratio` | auth | tỷ lệ gói xác thực NTLM | v2.0 | none |
| 12 | `log_distinct_hosts` | fanout | `log1p(#LogHost duy nhất)` | v2.0 | `log1p` |
| 13 | `distinct_sources_count` | fanout | `#Source` duy nhất (bỏ null) | v2.0 | **rank-Gaussian** |
| 14 | `missing_source_ratio` | context | tỷ lệ sự kiện thiếu `Source` (**covariate chất lượng dữ liệu**) | v2.0 | none |
| 15 | `remote_logon_ratio` | context | tỷ lệ `Source ≠ LogHost` | v2.0 | none |
| 16 | `custom_proc_share` | context | tỷ lệ `ProcessName` bắt đầu bằng `proc` | v2.0 | none |

**A.2. 14 biến mới được chọn**

| # | Biến | Nhóm | Định nghĩa | Nguồn đề xuất | Transform |
|---:|:---|:---|:---|:---|:---|
| 17 | `new_source_count` | novelty | số `Source` mới của tài khoản **chưa từng thấy trước ngày t** | chung (Bộ 1 + Bộ 2) — code đã có ở archive v2 | none |
| 18 | `new_host_count` | novelty | số `LogHost` mới **trước ngày t** | chung — code đã có ở archive v2 | none |
| 19 | `new_source_count_7d` | novelty | số `Source` mới trong ngày chưa thấy ở **7 ngày trước** | Bộ 1 | none |
| 20 | `new_host_count_7d` | novelty | số `LogHost` mới trong **7 ngày trước** | Bộ 1 | none |
| 21 | `source_recency` | recency | số ngày kể từ lần cuối thấy máy nguồn này (min trên các Source của ngày) | Bộ 1 | `log1p` |
| 22 | `source_host_pair_novelty` | novelty | `1 − Jaccard(cặp (Source, LogHost) hôm nay, baseline ≤ t−1)` | chung | none |
| 23 | `days_since_last_activity` | recency | số ngày kể từ phiên hoạt động gần nhất (bắt tài khoản "ngủ đông" thức dậy) | chung | `log1p` |
| 24 | `max_failure_streak` | failure | chuỗi thất bại liên tiếp dài nhất trong ngày | Bộ 1 | `log1p` |
| 25 | `success_after_failure_ratio` | failure | tỷ lệ 4624 xuất hiện **ngay sau** 4625 (dấu hiệu dò mật khẩu thành công) | Bộ 1 | none (nullable) |
| 26 | `activity_peak_hour_sin` | rhythm | `sin(2π·h_peak/24)` | Bộ 2 (Bộ 1 chỉ "khuyến nghị") | none |
| 27 | `activity_peak_hour_cos` | rhythm | `cos(2π·h_peak/24)` | Bộ 2 | none |
| 28 | `hour_entropy` | rhythm | Shannon entropy trên 24 khung giờ trong ngày | Bộ 1 | none |
| 29 | `dst_host_entropy` | fanout | Shannon entropy phân bố lượt truy cập theo `LogHost` (đúng `host_entropy` trong §7.3) | Bộ 2 | none |
| 30 | `volume_robust_z_7d` | deviation | z-score bền vững của volume so với **7 ngày trước** (median/MAD) | chung | none |


> **Trạng thái Tier A (đã triển khai ĐỦ):** cả 12 biến mới của Tier A đã được **cài đặt và đo trên dữ liệu thật**.
> **8 biến được nhận** (#26–#29 nhóm 9 và `new_source_count_7d`, `new_host_count_7d`, `days_since_last_activity`, `volume_robust_z_7d` của nhóm 10 — đã vào schema, core = **24**), **6 biến bị bác bỏ** vì vượt ngưỡng \|ρ\| ≥ 0,85 (#24 `max_failure_streak` 0,9973 · #25 `success_after_failure_ratio` 0,9180 · #17 `new_source_count` 0,9999 · #18 `new_host_count` 0,9999 · #21 `source_recency` 0,9238 · #22 `source_host_pair_novelty` 0,9052).
> **Không cần "thêm dòng" vào ma trận:** cửa sổ nhân quả theo NGÀY LỊCH được tính bằng `shift`/`min`/`max` trên panel thưa + lưới dày NỘI BỘ cho baseline khối lượng, nên số dòng (và do đó ranh giới train/test 721.612/333.671) **giữ nguyên** — xem §10.

### 5.3. Tier B — 30 biến thử nghiệm (chia 6 nhóm để ablation)

| Nhóm | Số biến | Danh sách |
|:---|---:|:---|
| Δt / phân phối | 3 | `interarrival_dt_p50`, `max_interarrival_dt`, `burst_event_count_60s` |
| Lag / xu hướng | 7 | `log_total_logons_lag1`, `log_total_logons_lag7`, `delta_log_total_logons_1d`, `delta_log_total_logons_7d`, `volume_trend_slope_7d`, `ewma_volume_7d`, `volume_rolling_mean_7d` |
| Novelty mở rộng | 5 | `source_set_novelty`, `host_set_novelty`, `new_source_count_14d`, `new_host_count_14d`, `host_recency` |
| **Peer cohort** | 6 | `peer_volume_ratio`, `peer_volume_robust_z`, `peer_failure_ratio_diff`, `peer_distinct_hosts_diff`, `peer_off_hours_diff`, `peer_jaccard_similarity` |
| Robust-z khác | 5 | `failure_ratio_robust_z_7d`, `distinct_hosts_robust_z_7d`, `off_hours_robust_z_7d`, `ntlm_robust_z_7d`, `failure_ratio_lag1` |
| Logon-type / đồ thị chi tiết | 4 | `logon_type_entropy`, `src_host_entropy`, `src_dst_pair_entropy`, `dst_host_concentration_ratio` |

**Vì sao `peer_*` xếp Tier B (chứ không phải Tier A)?** Đây là **nhóm có cơ sở phương pháp luận mạnh nhất trong Bộ 2** và khớp đúng khuyến nghị **P1** "chuẩn hoá theo baseline từng tài khoản" (`feature_correlation_review.md` §11, mục #7); `entity_type` cũng đã được ghi chú là "phục vụ … nhóm đồng đẳng Peer-group" (`reports/week2/feature_dictionary_raw.md`:19). Nhưng repo **chưa có hạ tầng cohort**, và định nghĩa "peer" (theo `entity_type`? theo domain? theo dải volume?) sẽ quyết định kết quả ⇒ phải là biến **A/B**, không được mặc định vào vector.

**Biến Tier B cần cổng kiểm tra riêng (vì gần với biến đã bị loại):**

| Biến Tier B | Phải đối chiếu với | Ngưỡng/ghi chú |
|:---|:---|:---|
| `burst_event_count_60s` | `log_total_logons` | repo đã bỏ `burst_logon_count` vì ρ = **0,9102** với volume ⇒ kiểm lại, nếu vượt thì bỏ tiếp |
| `interarrival_dt_p50` | `interarrival_dt_mean`, `delta_t_cv` | phải giữ \|ρ\| < 0,85 |
| `source_set_novelty`, `host_set_novelty` | `new_source_count_7d`, `new_host_count_7d` | hai biến đo cùng trục ⇒ nhiều khả năng chỉ giữ 1 dạng |
| `logon_type_entropy`, `src_host_entropy`, `src_dst_pair_entropy` | tỷ lệ zero | cả 3 gần zero-inflated (mẫu: `rare_logon_type_count` = 71,09% điểm 0) |
| `ewma_volume_7d`, `volume_rolling_mean_7d`, `log_total_logons_lag1/lag7` | `volume_robust_z_7d` | cùng trục volume-vs-baseline ⇒ giữ tối đa 2 |


### 5.4. Tier C — 7 biến chẩn đoán / chỉ mục (không vào vector)

| Biến | Vai trò |
|:---|:---|
| `volume_residual` | phần dư thô `x_t − x̂_t` so với baseline chu kỳ (hiển thị cho SOC) |
| `residual_robust_zscore` | z-score bền vững của phần dư |
| `residual_abs_exceed_3sigma` | cờ luật cảnh báo (\|z\| ≥ 3) |
| `behavior_shift_score_7d` | điểm trôi đa biến (xếp hạng rủi ro, không dùng làm đặc trưng) |
| `day`, `week` | chỉ mục panel để sắp xếp / cắt cửa sổ trượt |
| `day_of_week` | nhãn ngữ cảnh thứ (dùng để **tách báo cáo**, không nạp vector) |

---

## 6. Vì sao không nạp thẳng 43 hay 61 biến (lý do định lượng)

| # | Lý do | Bằng chứng |
|---:|:---|:---|
| 1 | Vector đã ở vùng zero-inflation cao | `is_single_event` 98,47%, `failure_locked_out_share` 98,86%, `custom_proc_share` 90,33%, `failure_ratio` 87,36% = 0 (`reports/week2/normalization_assessment.md`) |
| 2 | Nhiều biến **MAD = 0** ⇒ z-score không tách được | **7/16** đặc trưng MAD = 0 (`reports/week3/tom_tat_tuan3.md`, phát hiện #3) |
| 3 | Mô hình dựa trên khoảng cách | 3/5 mô hình là IForest/LOF/OCSVM; `interarrival_dt_mean` (tới 86.400 giây) từng lấn át điểm z ⇒ phải thêm RobustScaler |
| 4 | Thêm chiều làm mất khả năng đọc kết quả | Top-20 của 5 mô hình **overlap ≈ 0** (cao nhất 0,10) ⇒ 61–68 chiều sẽ càng làm Top-K nhiễu |
| 5 | Không thể xếp hạng bằng label-free | Kết luận Tuần 3: **bắt buộc** dùng nhãn thật `redteam.txt` (Tuần 4) |
| 6 | Bộ 61 chứa biến chết | **17/61 tên** của Bộ 2 nằm ở danh sách loại trừ §4 (toàn bộ nhóm logon-type ratio + 4 khung giờ + 4 phân vị Δt + 2 biến SubStatus) |
| 7 | Bộ 43 bỏ mất 1 biến đang là core và chứa 3 tên không tồn tại | thiếu `custom_proc_share`; thêm `is_machine_account`, `new_source_count`, `new_host_count` (2 tên sau chỉ có ở bản lưu trữ) |
| 8 | Bộ 61 đưa `behavior_shift_score_7d` vào vector | đây là **điểm trôi tổng hợp**, không phải đặc trưng hành vi ⇒ đề xuất chuyển sang Tier C |


---

## 7. Quy trình nghiệm thu đề xuất (6 bước, có thể chạy lại)

| Bước | Việc | Công cụ/lệnh | Điều kiện PASS |
|---:|:---|:---|:---|
| 1 | Khai báo Tier A vào hợp đồng schema (mỗi biến có `group`, `model_transform`, `nullable`) | sửa `configs/feature_schema.yaml` | `python -c "from src.features.schema import FeatureSchema as F; print(len(F().core_features))"` = **30** |
| 2 | Cập nhật extractor + preprocessor | `src/features/extractor.py` (`RAW_ORDERED_COLS`), `src/features/preprocessor.py` | `python main.py --stage features` chạy xong, self-check ràng buộc ratio ∈ [0,1] |
| 3 | Đo lại đa cộng tuyến trên ma trận mới | `scripts/feature_engineering/check_multicollinearity.py` | **0 cặp \|ρ\| ≥ 0,85; VIF ≤ 10** |
| 4 | Đo lại phân phối / MAD | `scripts/feature_engineering/check_distribution_stats.py`, `scripts/diagnostics/feature_variance_check.py` | biến mới không rơi vào "hằng số" |
| 5 | **Ablation theo nhóm có nhãn** (nội dung chính của Tuần 4) | `python main.py --stage benchmark` sau khi gán `redteam.txt` | `precision_at_k`, `recall_at_budget`, `roc_auc`, `average_precision` không thấp hơn bộ 16 hiện tại |
| 6 | Chốt & công bố | ghi vào `experiments/logs/experiment_log.csv` + `run_manifest.json` | chênh lệch > độ bất ổn đa seed (`model_stability.csv`) |

Các hàm chỉ số cần nhãn đã **cài đặt + kiểm thử sẵn** trong `src/evaluation/metrics.py` (chưa gọi trên dữ liệu thật) ⇒ bước 5 không cần viết thêm mã đo lường.

---

## 8. Điều kiện tiên quyết còn thiếu (phải làm trước bước 2)

1. **Dense Panel**: lưới `(DomainName, UserName) × day` liên tục cho 60 ngày — cần cho 4 biến novelty/lag/rolling của Tier A và cả Tier B. Panel hiện tại **thưa** (chỉ sinh dòng cho ngày có sự kiện): train 721.612 dòng/42 ngày ≈ **17.181 dòng-ngày**, test 333.671 dòng/18 ngày ≈ **18.537 dòng-ngày** — trong khi lưới dày sẽ là `#tài khoản duy nhất × 60 ngày` (chưa đo được vì dữ liệu chưa có trong workspace).
2. **Cửa sổ nhân quả**: mọi `*_7d/*_14d/lag/recency` phải dùng `shift` + `rolling` **trước** khi gộp ngày t; đây là chỗ dễ rò rỉ nhất của cả hai bộ (Bộ 2 mô tả novelty "trong lịch sử quan sát" mà không chốt mốc).
3. **Cohort cho `peer_*`**: chốt định nghĩa nhóm đồng đẳng (khuyến nghị `entity_type` + dải volume quá khứ) và tính **chỉ trên ngày ≤ t−1**.
4. **Nhóm burst theo cửa sổ** (`max_failures_5m`, `max_distinct_hosts_15m`) — cả Bộ 1 và Bộ 2 **đều thiếu**, trong khi §7.3 của review coi đây là nhóm "không trùng nhau về cấu trúc"; với LANL (Δt = 0 chiếm 70,57%) đây có thể là tín hiệu mạnh nhất sau novelty. Đề xuất bổ sung vào **Tier B** ở phiên bản kế tiếp sau khi có Dense Panel.
5. **Dữ liệu để đo**: `data/interim` có đủ 60 ngày ⇒ **đã dựng lại ma trận và đo** (§10). `data/processed/feature_matrix_processed.parquet` hiện vẫn là artifact **v2.0 (16 core)**; muốn dùng v3.0 phải chạy `python main.py --stage features` (~15 phút cho 60 ngày) rồi chạy lại `--stage benchmark`.

---

## 9. Phạm vi, giới hạn & việc không làm trong báo cáo này

* **Không sửa mã/cấu hình**: chưa đổi `configs/feature_schema.yaml` (vẫn 16 core, test `tests/test_configs.py::test_feature_schema_contract` sẽ fail nếu thêm biến mà không cập nhật test — cần cập nhật **cùng lúc**).
* **Số liệu trong báo cáo là số đo có sẵn** trong các artifact đã commit (liệt kê ở phần *Căn cứ*). **Chưa có phép đo mới** trên bộ mở rộng vì dữ liệu chưa có trong workspace.
* **Trạng thái kiểm định:** Tier A/B/C đã được kiểm chứng **tính nhất quán tên biến** (không trùng lặp, không giao nhau, Tier A chứa đủ 16 core v2.0). Riêng **6 biến không cần lịch sử đã được đo trên dữ liệu thật và qua cổng** (§10.2: 0 cặp \|ρ\| ≥ 0,85, VIF max 7,30 trên 10 ngày/180.606 dòng). Trên 60 ngày **vẫn phải đo lại** (ma trận đầy đủ sẽ có thêm mùa vụ cuối tuần và các tài khoản chỉ xuất hiện ở giai đoạn sau).
* Bộ 1 và Bộ 2 **không bị loại bỏ hoàn toàn**: mọi biến của chúng rơi vào Tier A/B/C hoặc danh sách loại trừ §4 đều có lý do kèm theo (đây là yêu cầu bắt buộc của hợp đồng schema).

## 10. Kết quả triển khai (đã chạy thật) + số đo cổng kiểm định

### 10.1. Đã thay đổi gì trong mã/cấu hình

| File | Thay đổi |
|:---|:---|
| `configs/feature_schema.yaml` | **16 → 24 core**: +4 nhóm 9 (`activity_peak_hour_sin/cos`, `hour_entropy`, `dst_host_entropy`) +4 nhóm 10 (`new_source_count_7d`, `new_host_count_7d`, `days_since_last_activity`, `volume_robust_z_7d`); 6 biến bị bác bỏ ghi vào `removed:` kèm ρ; thêm khối `deferred:` (Tier B/C) |
| `src/features/history.py` **(mới)** | Module đặc trưng LIÊN-NGÀY: `add_history_features()`, `HISTORY_V3_FEATURES`, `HISTORY_V3_REJECTED`. Cửa sổ theo **NGÀY LỊCH**; baseline khối lượng dùng **lưới dày nội bộ** (ngày trống = 0) nhưng **không thêm dòng nào** vào ma trận |
| `src/features/extractor.py` | +`intraday_behavior_features()` (4 biến trong-ngày), +`_presence_frames()` (vật liệu Source/LogHost cho nhóm 10 — ma trận thô không giữ danh tính nên extractor phải cấp), `MATRIX_ORDERED_COLS` (28 cột), **fix bug `ProcessName`** |
| `tests/` | Thêm `test_intraday_behavior_features`, `test_history_features_values`, `test_history_features_are_causal` (chống rò rỉ tương lai), `test_volume_robust_z_is_winsorized` — **34 passed, 1 skipped** |
| `scripts/.../check_raw_matrix.py` | Hợp đồng raw **28 cột** |
| `README.md`, `docs/project_structure.md`, `configs/model_params.yaml` | 16 → 24 core |

### 10.2. Số đo trên DỮ LIỆU THẬT — 10 ngày (180.606 dòng), dựng lại từ `data/interim`

| Bộ | Cặp \|ρ\| ≥ 0,85 | VIF max |
|:---|---:|---:|
| 16 biến (v2.0) | 0 | 7,30 |
| 22 biến (6 ứng viên nhóm 9) | **3** ❌ | NaN (cột hằng số) |
| 28 biến (8 nhóm 9 + 8 nhóm 10, thử nghiệm) | **5** ❌ | **8.197** ❌ |
| **24 biến (chốt)** | **0** ✅ | **7,31** ✅ |

**4 biến nhóm 10 được giữ** (max \|ρ\| với 23 biến còn lại): `new_host_count_7d` 0,7301 · `new_source_count_7d` 0,7299 · `days_since_last_activity` 0,6982 · `volume_robust_z_7d` 0,227.
Phân bố: `new_source_count_7d` zero 82,59% (p99 = 2, max 1.770) · `new_host_count_7d` zero 69,07% (p99 = 7, max 9.314) · `days_since_last_activity` NULL 13,32% (chỉ ở dòng đầu của mỗi tài khoản) · `volume_robust_z_7d` NULL 67,13% (**10 ngày thì 7 ngày đầu là warm-up**; trên 60 ngày tỷ lệ này sẽ ≈ 12%).

### 10.3. Hai lỗi thật phát hiện nhờ quá trình đo (đã sửa + đã khoá bằng test)

1. **`custom_proc_share` = 0 âm thầm.** Đường fallback đọc `data/interim` chỉ `select` 8 cột nên thiếu `ProcessName` ⇒ một đặc trưng **core** bị 0 tuyệt đối mà không có cảnh báo. Sau khi sửa: 11.016/180.606 dòng (6,10%) khác 0.
2. **`volume_robust_z_7d` bùng nổ tới −976.** Không phải lỗi số học: tài khoản `User718489` giữ 5.966–6.058 sự kiện/ngày suốt 7 ngày rồi tụt còn 472 ⇒ IQR ≈ 0,005 ⇒ z = −976, đủ để lấn át khoảng cách của IForest/LOF/OCSVM. Đã thêm **winsorize ±10** (kèm dung sai scale 1e-9) và test riêng; sau khi sửa miền giá trị là [−10, 10].

### 10.4. Việc còn lại để dùng v3.0 trong benchmark

```bash
# 1) Dựng lại ma trận 20 core từ interim (~15 phút cho 60 ngày)
python main.py --stage features

# 2) Kiểm định artifact (script, exit code cho CI)
python scripts/feature_engineering/check_raw_matrix.py
python scripts/feature_engineering/check_feature_matrix.py --matrix data/features/raw/feature_matrix_raw.parquet
python scripts/feature_engineering/check_multicollinearity.py
python scripts/diagnostics/feature_variance_check.py

# 3) Chạy lại benchmark rồi mới so sánh với bảng Tuần 3 (bảng cũ là số của 16 core)
python main.py --stage benchmark
```

**Lưu ý bắt buộc:** `data/processed/feature_matrix_processed.parquet` hiện vẫn là artifact **v2.0 (16 core)**
⇒ `tests/test_features.py::test_model_interfaces` sẽ **skip** kèm thông điệp yêu cầu dựng lại (đây là guard có chủ ý,
không phải test hỏng); sau khi chạy bước 1, guard này tự chuyển thành kiểm tra đầy đủ.



### Phụ lục — Lệnh tái lập nhanh

```bash
# Hợp đồng đang hiệu lực (kỳ vọng 20 core)
python -c "from src.features.schema import FeatureSchema as F; s=F(); print(len(s.core_features)); print(s.core_features)"

# 2 biến đã đo và bác bỏ (không được nằm trong core)
python -c "from src.features.extractor import INTRADAY_V3_REJECTED as R; print(R)"

# Bằng chứng đa cộng tuyến / phân phối (sau khi dựng lại matrix)
python scripts/feature_engineering/check_multicollinearity.py
python scripts/diagnostics/feature_variance_check.py
python -m pytest tests/ -q
```

