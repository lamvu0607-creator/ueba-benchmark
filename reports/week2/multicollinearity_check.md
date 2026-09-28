# Kiểm tra Đa cộng tuyến (Multicollinearity) — Bộ đặc trưng UEBA

*Ngày tạo: 2026-09-26 02:14 · Phạm vi: Tuần 2 (Feature Engineering & Validation)*
*Script: `scripts/feature_engineering/check_multicollinearity.py` · Biểu đồ: `scripts/feature_engineering/plot_feature_correlation.py`*

---

## 1. Dữ liệu & phạm vi kiểm tra

| Hạng mục | Giá trị |
| :--- | :--- |
| Ma trận đầu vào | `data/features/raw/feature_matrix_raw.parquet` (tầng **THÔ**, bước 1 — chưa chuẩn hóa) |
| Kích thước | **1.055.283 dòng** (tài khoản × ngày) × **60 ngày** (28/07 → 25/09) |
| Số đặc trưng số | **16** (4 cột khoá/nhãn `DomainName, UserName, day, entity_type` không đưa vào tính tương quan) |
| Cột hằng số | 0 (không có cột nào bị loại) |
| Bộ đặc trưng | `total_logons`, `failure_ratio`, `failure_locked_out_share`, `off_hours_ratio`, `interarrival_dt_mean`, `delta_t_cv`, `same_second_share`, `is_single_event`, `interactive_ratio`, `rare_logon_type_count`, `ntlm_ratio`, `distinct_hosts`, `distinct_sources_count`, `missing_source_ratio`, `remote_logon_ratio`, `custom_proc_share` |

## 2. Phương pháp — 3 lớp kiểm tra

| Lớp | Chỉ số | Ngưỡng coi là đa cộng tuyến | Vì sao dùng |
| :--- | :--- | :---: | :--- |
| 1 | **Spearman ρ** (thứ hạng) | \|ρ\| ≥ 0,85 | Bền với đuôi dài/zero-inflation của log sự kiện; không bị 1 outlier chi phối |
| 2 | **Pearson r** (đối chiếu) | \|r\| ≥ 0,85 | Bắt các quan hệ tuyến tính thật (nền tảng của VIF) |
| 3 | **VIF** (Variance Inflation Factor) | \> 5 (cảnh báo), \> 10 (nghiêm trọng) | Mức độ "phình phương sai" khi hồi quy 1 biến theo 15 biến còn lại |

Bổ sung: **độ liên kết trung bình** `mean |ρ|` của từng biến với các biến còn lại (phát hiện biến dư thừa thông tin) và kiểm chứng **bất biến khi `log1p`** (Spearman không đổi → kết luận dùng được luôn cho ma trận đã chuẩn hóa ở bước 3).

## 3. Kết quả

### 3.1 Tổng hợp

| Lớp kiểm tra | Số cặp vượt ngưỡng | Ngưỡng | Kết luận |
| :--- | :---: | :---: | :---: |
| Spearman ρ (120 cặp) | **0 / 120** | 0,85 | ✅ PASS |
| Pearson r (120 cặp) | **0 / 120** | 0,85 | ✅ PASS |
| VIF | **0 / 16 biến > 10** (1 biến > 5) | 10 | ✅ PASS |

**Cặp mạnh nhất toàn bộ ma trận: ρ = −0,678** (`missing_source_ratio` ↔ `remote_logon_ratio`) — còn cách xa ngưỡng cảnh báo 0,85 và cả ngưỡng "đáng lưu ý" 0,70.

### 3.2 VIF — xếp hạng đầy đủ 16 đặc trưng

| Hạng | Đặc trưng | VIF | Mức độ |
| :---: | :--- | :---: | :--- |
| 1 | `remote_logon_ratio` | **7,63** | 🟠 Cảnh báo (> 5) |
| 2 | `distinct_hosts` | 4,15 | 🟢 Bình thường |
| 3 | `custom_proc_share` | 4,11 | 🟢 Bình thường |
| 4 | `missing_source_ratio` | 4,01 | 🟢 Bình thường |
| 5 | `rare_logon_type_count` | 3,52 | 🟢 Bình thường |
| 6 | `same_second_share` | 1,90 | 🟢 Bình thường |
| 7 | `ntlm_ratio` | 1,65 | 🟢 Bình thường |
| 8 | `distinct_sources_count` | 1,54 | 🟢 Bình thường |
| 9 | `failure_ratio` | 1,43 | 🟢 Bình thường |
| 10 | `delta_t_cv` | 1,40 | 🟢 Bình thường |
| 11 | `is_single_event` | 1,33 | 🟢 Bình thường |
| 12 | `interactive_ratio` | 1,27 | 🟢 Bình thường |
| 13 | `interarrival_dt_mean` | 1,15 | 🟢 Bình thường |
| 14 | `off_hours_ratio` | 1,13 | 🟢 Bình thường |
| 15 | `total_logons` | 1,07 | 🟢 Bình thường |
| 16 | `failure_locked_out_share` | 1,02 | 🟢 Bình thường |

### 3.3 Top 10 cặp tương quan thứ hạng mạnh nhất

| # | Cặp đặc trưng | Spearman ρ | Diễn giải nghiệp vụ |
| :---: | :--- | :---: | :--- |
| 1 | `missing_source_ratio` ↔ `remote_logon_ratio` | **−0,678** | Logon từ xa (Type 10/RDP) hầu như **luôn có** trường `Source` → ít khi vừa remote vừa thiếu Source |
| 2 | `delta_t_cv` ↔ `same_second_share` | +0,645 | Chuỗi sự kiện càng "dồn cục" (nhiều logon cùng giây) thì độ biến thiên khoảng cách thời gian càng cao |
| 3 | `delta_t_cv` ↔ `distinct_hosts` | +0,577 | Tài khoản vận hành nhiều máy → nhịp logon không đều hơn |
| 4 | `total_logons` ↔ `interarrival_dt_mean` | −0,529 | Càng nhiều logon trong ngày thì khoảng cách trung bình giữa 2 logon càng ngắn (quan hệ nghịch, đúng logic) |
| 5 | `rare_logon_type_count` ↔ `distinct_sources_count` | +0,524 | Tài khoản dùng nhiều LogonType hiếm cũng thường đến từ nhiều nguồn khác nhau |
| 6 | `rare_logon_type_count` ↔ `distinct_hosts` | +0,477 | LogonType hiếm (service/batch/network cleartext) gắn với nhiều máy |
| 7 | `same_second_share` ↔ `ntlm_ratio` | −0,467 | NTLM (máy ↔ máy, burst) nghịch với tỷ lệ logon trùng giây của người dùng |
| 8 | `delta_t_cv` ↔ `rare_logon_type_count` | +0,461 | Nhịp bất thường ↔ hành vi dùng loại logon hiếm |
| 9 | `distinct_hosts` ↔ `distinct_sources_count` | +0,460 | Hai góc nhìn "không gian" bổ trợ nhưng **không trùng lặp** (ρ chỉ 0,46) |
| 10 | `interactive_ratio` ↔ `rare_logon_type_count` | +0,440 | Phiên tương tác (Type 2/10) đi kèm hành vi logon đa dạng |

**Độ liên kết trung bình cao nhất:** `rare_logon_type_count` 0,314 · `distinct_hosts` 0,288 · `delta_t_cv` 0,281 · `distinct_sources_count` 0,252 — đều dưới 0,35, tức không biến nào "dư thừa" thông tin của phần còn lại.

### 3.4 Hai cặp không tính được Spearman (không phải lỗi dữ liệu)

| Cặp | Số dòng hợp lệ | Nguyên nhân |
| :--- | :---: | :--- |
| `is_single_event` ↔ `delta_t_cv` | 1.029.109 | Trên tập dòng hợp lệ, `is_single_event` **luôn = 0** (hằng số) → Spearman không xác định |
| `is_single_event` ↔ `interarrival_dt_mean` | 1.039.175 | Tương tự — 2 cột "nhịp thời gian" chỉ có giá trị khi ngày đó có ≥ 2 sự kiện |

→ Đây là **ràng buộc logic theo thiết kế** (`delta_t_cv`, `interarrival_dt_mean` = NULL khi `is_single_event = 1`), không phải lỗi missing. Vì vậy heatmap có 134/136 ô được ghi số (2 ô NaN để trắng).

### 3.5 Kiểm chứng bất biến khi chuẩn hóa `log1p`

Tính lại ma trận Spearman sau khi áp `log1p` cho cả 16 đặc trưng: **sai số tuyệt đối lớn nhất = 1,14 × 10⁻⁷** (≈ 0).
→ Spearman là bất biến với biến đổi đơn điệu, nên **kết luận đa cộng tuyến ở trên hiệu lực nguyên vẹn cho ma trận normalized (bước 3, B3)** — không cần soát lại cặp ρ, chỉ cần soát lại Pearson/VIF.

---

## 4. Kết luận

1. **Không có đa cộng tuyến nghiêm trọng** trong bộ 16 đặc trưng: 0/120 cặp vượt 0,85 (cả Spearman và Pearson), VIF max = **7,63** < 10, chỉ 1/16 biến nằm vùng cảnh báo (> 5).
2. Bộ đặc trưng **giữ nguyên 16 biến**, không phải loại bỏ biến nào vì lý do đa cộng tuyến.
3. Các cặp mạnh nhất đều có **giải thích nghiệp vụ hợp lý** (remote ↔ missing Source; burst ↔ CV khoảng cách; nhiều host ↔ nhiều nguồn) → tương quan phản ánh quy luật hành vi, không phải lỗi sinh đặc trưng.
4. Nguy cơ tồn dư duy nhất: `remote_logon_ratio` (VIF 7,63) khi dùng **mô hình tuyến tính**.

## 5. Khuyến nghị

| # | Khuyến nghị | Lý do |
| :---: | :--- | :--- |
| 1 | Giữ đủ 16 đặc trưng; **không cần lọc biến** | Mọi chỉ số dưới ngưỡng |
| 2 | Nếu dùng hồi quy tuyến tính (Logistic/Ridge): **chuẩn hóa (StandardScaler) + L2 (Ridge)** thay vì loại biến | Trung hòa VIF 7,63 của `remote_logon_ratio` |
| 3 | Với mô hình khoảng cách (LOF/OCSVM): vẫn nên scale chuẩn vì đặc trưng thô lệch nặng (đuôi dài) | Tránh 1 biến đếm chi phối khoảng cách |
| 4 | **Ở bước 3 (B3) không đưa đồng thời `total_logons` và `log_total_logons` vào cùng vector đặc trưng cho model** | Hai cột này có ρ = 1,0 (tuyệt đối trùng thông tin); giữ bản thô chỉ để đối chiếu/báo cáo |
| 5 | Bổ sung kiểm tra định kỳ vào pipeline | Đã gắn `check_multicollinearity.py` làm tác vụ #5 trong `run_all_feature_plots.py` |

## 6. Tái lập kết quả

```bash
# 1) Vẽ heatmap Spearman (có ghi số trong từng ô, 2 chữ số thập phân)
python scripts/feature_engineering/plot_feature_correlation.py

# 2) Soát đa cộng tuyến 3 lớp (Spearman / Pearson / VIF) — exit code 0 = PASS
python scripts/feature_engineering/check_multicollinearity.py
# hoặc tuỳ biến ngưỡng / ma trận:
python scripts/feature_engineering/check_multicollinearity.py --matrix data/features/raw/feature_matrix_raw.parquet --threshold 0.85

# 3) Chạy toàn bộ pipeline biểu đồ tuần 2 (5 tác vụ, gồm cả bước 2)
python scripts/feature_engineering/run_all_feature_plots.py
```

## 7. Sản phẩm kèm theo

| Loại | Đường dẫn |
| :--- | :--- |
| Biểu đồ heatmap (có số ở mọi ô) | `docs/feature_engineering/figures/spearman_correlation_matrix.png` (+ bản sao `artifacts/`) |
| Ma trận Spearman đầy đủ | `docs/feature_engineering/tables/correlation_spearman.csv` |
| Danh sách cặp vượt ngưỡng | `docs/feature_engineering/tables/high_multicollinearity_pairs.csv` (rỗng — 0 cặp) |
| Bảng VIF 16 biến | `docs/feature_engineering/tables/multicollinearity_vif.csv` (mới) |

> **Ghi chú kỹ thuật:** biểu đồ hiển thị nửa dưới + đường chéo (mask tam giác trên) để tránh lặp thông tin và giữ chữ số to, dễ đọc ở cỡ in A4; màu phân kỳ đỏ (ρ → +1) / xanh (ρ → −1), ô trắng = 0 hoặc NaN.
