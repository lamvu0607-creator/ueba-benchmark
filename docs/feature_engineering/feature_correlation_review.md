# Nhận xét Kết quả Phân tích Tương quan Đặc trưng (Feature Correlation Review)

> ⚠️ **Lưu ý lịch sử (cập nhật 2026-09-26):** tài liệu này viết cho **bộ 30 đặc trưng cũ**, dữ liệu đã được chuyển sang `data/features/legacy/` và các bảng `correlation_pearson.csv` / `multicollinear_pairs.csv` / `figures/correlation_side_by_side.png` **không còn được sinh ra**. Bộ đặc trưng hiện hành (16 đặc trưng THÔ + 4 khoá/nhãn) có báo cáo đa cộng tuyến mới tại [`reports/week2/multicollinearity_check.md`](../../reports/week2/multicollinearity_check.md). Phần nội dung bên dưới giữ nguyên như **hồ sơ kiểm chứng độc lập cho bộ cũ**.

> **Dự án:** UEBA Anomaly Detection Benchmark (Windows Event 4624 & 4625 — LANL Unified Host and Network Dataset)  
> **Đối tượng được nhận xét:** [`tables/correlation_pearson.csv`](tables/correlation_pearson.csv), [`tables/correlation_spearman.csv`](tables/correlation_spearman.csv), [`tables/multicollinear_pairs.csv`](tables/multicollinear_pairs.csv), [`figures/correlation_side_by_side.png`](figures/correlation_side_by_side.png)  
> **Đối tượng sinh ra kết quả:** [`scripts/feature_engineering/plot_feature_correlation.py`](../../scripts/feature_engineering/plot_feature_correlation.py) chạy trên `data/features/account_day_matrix.parquet`  
> **Vai trò người nhận xét:** Data Scientist (review độc lập, có kiểm chứng lại số liệu)  
> **Vị trí lưu trữ:** `docs/feature_engineering/feature_correlation_review.md`

---

## 1. Phạm vi và cách kiểm chứng

Bản nhận xét này **không** chỉ đọc 3 file CSV. Toàn bộ số liệu được kiểm chứng độc lập bằng cách đọc lại dữ liệu gốc:

| Nguồn | Kích thước | Mục đích sử dụng |
|---|---|---|
| `data/features/account_day_matrix.parquet` | 56.731 dòng × 35 cột (3 ngày: day 1–3) | Tái lập Pearson/Spearman, thống kê mô tả, VIF, phân tích theo `entity_type`, chuẩn hoá within-account |
| `data/interim/event_4624/event_4624_day-16.parquet` | 17.553.517 dòng | Truy vết nguyên nhân dữ liệu của từng tương quan |
| `data/interim/event_4625/event_4625_day-16.parquet` | 359.897 dòng | Truy vết nguyên nhân dữ liệu của từng tương quan |
| `reports/archive/validation_summary.md`, `reports/day16_missingness.md` | — | Đối chiếu chéo tỷ lệ null và cấu trúc missingness |
| [`tables/distribution_skewness_comparison.csv`](tables/distribution_skewness_comparison.csv), [`figures/distribution_before_after_scaling.png`](figures/distribution_before_after_scaling.png) | 6 đặc trưng volume/fan-out | Đối chiếu chéo mức độ lệch (skewness) và hiệu quả `log1p` — xem §4.3 |

**Kết quả kiểm chứng tính tái lập:**

| Bảng | Cột khớp | Sai số tuyệt đối lớn nhất khi tính lại |
|---|---|---|
| `correlation_pearson.csv` | ✅ | `4.99e-05` |
| `correlation_spearman.csv` | ✅ | `4.99e-05` |

→ Hai bảng tương quan **đáng tin và tái lập được** (sai số chỉ do làm tròn 4 chữ số thập phân). Mọi vấn đề nêu trong tài liệu này nằm ở **tập đặc trưng**, không nằm ở khâu tính toán.

> **Ghi chú môi trường:** kiểm chứng chạy bằng Python của conda env `ueba-benchmark` (`polars`, `pandas`, `numpy`). Mã kiểm chứng là script tạm, đã xóa sau khi hoàn tất; Phụ lục B mô tả cách tái lập.

---

## 2. Kết luận điều hành (Executive Summary)

1. **Pipeline tính tương quan chạy đúng, tái lập được, ngưỡng cảnh báo đa cộng tuyến 0.85 là hợp lý.** Việc tự động loại 2 cột đẳng trị (`wrong_password_count`, `unknown_user_count`) là **quyết định đúng** — nhưng cần xóa hẳn khỏi schema đặc trưng, không chỉ loại khỏi bảng tương quan.
2. **Đa số tương quan mạnh trong bảng không phải tín hiệu an ninh.** Chúng đến từ 3 nguồn: (a) bản sao của nhau qua phép biến đổi đơn điệu, (b) gộp chung quần thể Machine/User/Service vào một panel, (c) **proxy của missingness có cấu trúc** trong log LANL.
3. **Đa cộng tuyến ở mức nghiêm trọng:** VIF lớn nhất **4041,96**; 6/30 đặc trưng có VIF > 10; 13 cặp có |ρ| ≥ 0,85 — nhưng phần lớn là hệ quả của định nghĩa, không phải quy luật dữ liệu.
4. **Nên rút gọn 30 → 18 đặc trưng trước khi huấn luyện**: số cặp |ρ| ≥ 0,85 giảm **13 → 0**, VIF lớn nhất giảm **4041,96 → 5,80**, số đặc trưng VIF > 10 giảm **6 → 0** (chi tiết §7).
5. **Hai vấn đề còn nghiêm trọng hơn cả tương quan**, cần xử lý trước khi sang tuần model: (i) khoá danh tính đang chỉ dùng `UserName` nên **trộn nhiều domain vào cùng một tài khoản**; (ii) 2 đặc trưng "chết" do `Status` null 100%, và tài liệu README/project_structure mô tả một bộ 16 đặc trưng **khác hẳn** bộ 32 đặc trưng đang được code sinh ra.

---

## 3. Đánh giá về chính kết quả tương quan

### 3.1. Điểm mạnh

- **Tái lập 100%** (xem §1) — bảng kết quả đủ tin cậy để dùng làm "hợp đồng kỹ thuật" (feature contract) giữa các thành viên nhóm.
- **Ngưỡng 0,85 hợp lý** cho dữ liệu UEBA, đủ nhạy để bắt các cặp gần trùng mà không báo động giả tràn lan.
- **Việc in song song Pearson và Spearman là quyết định rất tốt.** Chính phần chênh lệch giữa hai hệ số là nơi chứa thông tin chẩn đoán quan trọng nhất (xem §4.3).
- **Tự động loại cột đẳng trị** đúng về mặt kỹ thuật: `wrong_password_count` và `unknown_user_count` có `std < 1e-9` vì `Status` null 100% ở cả hai EventID (17.553.517/17.553.517 ở 4624 và 359.897/359.897 ở 4625, đối chiếu `reports/archive/validation_summary.md`).
- Việc script khớp đúng **30 cột** trong CSV (so với 32 cột số của parquet) là dấu hiệu bộ lọc hằng số hoạt động như thiết kế.

### 3.2. Hai điểm mù kỹ thuật cần sửa trong script

| Vấn đề | Bằng chứng định lượng | Hệ quả & cách sửa |
|---|---|---|
| `multicollinear_pairs.csv` **chỉ lọc theo Spearman** | Có **6** cặp \|Pearson\| ≥ 0,85 nhưng **13** cặp \|Spearman\| ≥ 0,85. Hai cặp *tuyến tính mạnh* bị bỏ sót hoàn toàn: `total_logons ↔ off_hours_count` (\|P\| = 0,9752) và `off_hours_count ↔ burst_logon_count` (\|P\| = 0,9744) | Danh sách cảnh báo thiếu đúng 2 cặp quan trọng. Sửa: lọc theo trị tuyệt đối **lớn hơn** trong hai hệ số (Pearson và Spearman), đồng thời ghi rõ hệ số nào đã kích hoạt cảnh báo |
| `fillna(0.0)` trước khi tính corr | Kiểm tra lại: **32/32 cột không có null** trong parquet ⇒ lệnh này là *no-op ngầm* ở lần chạy này | Hôm nay vô hại, nhưng nếu sau này dữ liệu có null thật thì `fillna(0)` sẽ tạo tương quan giả rất khó phát hiện. Sửa: log rõ số null trước/sau khi fill, hoặc dùng hệ số tính trên cặp quan sát đầy đủ (pairwise complete) |

### 3.3. Một chi tiết dễ bị hiểu nhầm là lỗi — nhưng không phải

Với **mọi** cặp cờ nhị phân (32/36 cặp), Pearson **bằng đúng** Spearman, ví dụ:

| Cặp cờ nhị phân | Pearson | Spearman | \|diff\| |
|---|---:|---:|---:|
| `has_local_logon ~ has_process_info` | 0,9738 | 0,9738 | 0,0000 |
| `has_process_info ~ has_system_process` | 0,9455 | 0,9455 | 0,0000 |
| `has_multi_domains ~ has_local_domain` | 0,4060 | 0,4060 | 0,0000 |
| `has_missing_source ~ has_remote_logon` | 0,1550 | 0,1550 | 0,0000 |

**Giải thích toán học:** hạng của một biến 0/1 là hàm **affine** của chính giá trị đó (quan sát bằng 1 nhận hạng trung bình `1 + (n₀+1)/2` với mọi quan sát cùng giá trị), mà Pearson **bất biến** với biến đổi affine. Do đó Pearson ≡ Spearman cho mọi cặp biến nhị phân — đây là tính chất toán học, **không phải** lỗi copy nhầm bảng giữa hai file CSV.

> **Khuyến nghị:** ghi chú ngắn này vào tài liệu feature contract để người sau không "sửa" nhầm hai bảng tương quan.

---

## 4. Bốn phát hiện lớn nhất

### 4.1. Đa cộng tuyến ở mức "trùng lặp cấu trúc", không chỉ "tương quan cao"

VIF tính lại trên ma trận Pearson của toàn bộ 30 đặc trưng:

| Đặc trưng | VIF |
|---|---:|
| `total_logons` | **4041,96** |
| `burst_logon_count` | **3977,17** |
| `has_process_info` | 51,17 |
| `has_local_logon` | 29,20 |
| `has_system_process` | 24,20 |
| `off_hours_count` | 22,86 |
| `remote_logon_ratio` | 8,77 |
| `network_ratio` | 4,75 |

- **6/30 đặc trưng có VIF > 10**, 7/30 có VIF > 5.
- Có những cặp **trùng lặp tuyệt đối về hạng (ρ = 1,0000)** — tức cùng một thông tin được viết hai lần: `total_logons ↔ log_total_logons` và `distinct_hosts_count ↔ log_distinct_hosts`. Đây không phải "hai đặc trưng tương quan cao" mà là **cùng một đặc trưng**.
- Trong 13 cặp ≥ 0,85 của `multicollinear_pairs.csv`, chỉ khoảng **4 cặp là quy luật hành vi thật**, phần còn lại là hệ quả của cách định nghĩa.

Phân tích "độ liên kết trung bình" cho thấy cụm trung tâm của sự trùng lặp là: `has_process_info` (0,39), `has_local_logon` (0,39), `network_ratio` (0,38), `has_system_process` (0,38) — đúng nhóm bị phát hiện ở §4.2.

### 4.2. Nhóm cờ nhị phân thực chất là *cùng một biến* — và là proxy của missingness

| Cặp | Hệ số |
|---|---:|
| `has_local_logon ~ has_process_info` | **+0,9738** |
| `has_process_info ~ has_system_process` | **+0,9455** |
| `network_ratio ~ has_process_info` | **−0,9431** |
| `has_local_logon ~ has_system_process` | +0,9181 |
| `network_ratio ~ has_local_logon` | −0,9135 |
| `network_ratio ~ has_system_process` | −0,9080 |

Truy vết raw ngày 16 cho thấy đây **không phải hành vi, mà là cơ chế ghi log**:

- `ProcessName` null **92,37%** ở 4624 nhưng chỉ **49,16%** ở 4625.
- Trong 4625, `LogonID` null cũng **49,16%** — *đúng cùng số dòng*: hai trường này **mất cùng nhau theo từng bản ghi**.
- `Source` null 24,28% (4624) / 32,09% (4625) và **có cấu trúc theo LogonType** (xem bảng Phụ lục A.3).
- `Source == LogHost` (điều kiện để `has_local_logon` = 1) chỉ đạt **18,80%** số sự kiện 4624 và **50,89%** ở 4625. Một sự kiện có `Source` null **không thể** được tính là "local" ⇒ `has_local_logon` vừa đo hành vi, vừa đo lỗ hổng ghi log.
- `has_remote_logon` có **prevalence 96,60%** ⇒ gần như hằng số, chỉ phân biệt 3,4% số dòng → gần như vô dụng cho anomaly detection.

**Kết luận:** bộ cờ `has_local_logon / has_process_info / has_system_process / has_remote_logon / has_system_logon_id` không đo 5 khía cạnh hành vi khác nhau, mà chủ yếu đo **"tài khoản này có loại sự kiện nào và có bị thiếu trường nào"**. Đưa cả nhóm vào mô hình sẽ khiến model học *cấu hình thu thập log*, và các đợt outage ghi log (như ngày 16 với `Source` thiếu tăng +8,00 điểm phần trăm ở 4624, theo `reports/day16_missingness.md`) sẽ bị gắn nhãn là "bất thường tấn công".

### 4.3. Pearson lệch Spearman thảm hoạ ⇒ phân phối cực lệch (heavy-tailed)

| Cặp | Pearson | Spearman | Pearson sau `log1p` |
|---|---:|---:|---:|
| `total_logons ~ burst_logon_count` | **0,9998** | 0,9102 | **0,1304** |
| `total_logons ~ log_total_logons` | 0,1401 | **1,0000** | — |
| `failure_count ~ failure_ratio` | 0,1095 | **0,9969** | — |
| `off_hours_count ~ off_hours_ratio` | 0,0301 | 0,7261 | — |
| `network_ratio ~ has_system_process` | −0,1657 | **−0,9080** | — |
| `total_logons ~ off_hours_count` | **0,9752** | 0,7818 | — |

Nguyên nhân định lượng (thống kê mô tả trên 56.731 dòng):

| Đặc trưng | Median | p99 | Max | Skewness | Max/Median |
|---|---:|---:|---:|---:|---:|
| `total_logons` | 350 | 4.500 | **2.572.041** | **90,24** | 7.349 |
| `burst_logon_count` | 202 | 2.699 | 2.572.040 | 90,78 | 12.733 |
| `failure_count` | 0 | 11 | 153.610 | **115,43** | — |
| `distinct_hosts_count` | 3 | 11 | 9.981 | 95,53 | 3.327 |
| `distinct_sources_count` | 1 | 4 | 2.355 | **139,08** | 2.355 |
| `off_hours_count` | 97 | 1.516,7 | 589.492 | 74,35 | 6.077 |

Mức tập trung khối lượng (nguyên nhân trực tiếp):

| Nhóm tài khoản | Số tài khoản | % tổng sự kiện |
|---|---:|---:|
| Top 0,1% | 20 | **43,62%** |
| Top 1% | 208 | 52,12% |
| Top 5% | 1.043 | 62,05% |
| Top 10% | 2.087 | 67,66% |

Hệ số Gini (volume/tài khoản) ≈ **0,730**. Thành phần quần thể của panel (3 ngày, 48.522.345 sự kiện):

| `entity_type` | Số dòng | % số dòng | Tổng sự kiện | % khối lượng | Mean logon/ngày | Max |
|---|---:|---:|---:|---:|---:|---:|
| Machine | 31.162 | 54,93% | 21.439.001 | 44,18% | 687,99 | 2.572.041 |
| User | 25.545 | 45,03% | 18.517.100 | 38,16% | 724,88 | 268.582 |
| Service | **24** | **0,04%** | **8.566.244** | **17,65%** | **356.926,83** | 1.120.172 |

→ **Pearson trên tập này gần như vô nghĩa.** Con số `corr(total_logons, burst_logon_count) = 0,9998` chỉ nói "vài dòng khổng lồ chi phối cả hai cột", không nói gì về tính bùng nổ của đăng nhập. Đây cũng là lý do bắt buộc phải log-transform trước khi dùng bất kỳ mô hình dựa trên khoảng cách nào (IsolationForest/LOF/OCSVM).

> **Đối chiếu chéo (nhất quán):** bảng [`tables/distribution_skewness_comparison.csv`](tables/distribution_skewness_comparison.csv) do nhóm đã tạo cũng ghi nhận đúng các giá trị trên (`total_logons`: median 350, p99 4.500, max 2.572.041, skew 90,24; `distinct_sources_count`: skew 139,07) và cho thấy `log1p` giảm skewness **94,3% – 99,0%** (xem thêm [`figures/distribution_before_after_scaling.png`](figures/distribution_before_after_scaling.png)). Hai nguồn số liệu khớp nhau ⇒ phần chẩn đoán phân phối này là chắc chắn. **Lưu ý bổ sung:** sau `log1p`, `failure_count` vẫn còn skew **6,56** và `distinct_sources_count` còn **3,63** ⇒ với hai đặc trưng này nên dùng thêm **rank-Gaussian / winsorize** thay vì chỉ `log1p`.

### 4.4. `burst_logon_count` không đo "burst"

Trong raw ngày 16:

| Chỉ số | 4624 | 4625 |
|---|---:|---:|
| Tỷ lệ sự kiện có `delta_t = 0` (trùng giây với sự kiện trước của cùng user) | **70,57%** | **67,19%** |
| Số sự kiện có `0 < delta_t ≤ 2` | 2.092.670 | 51.254 |
| Tỷ lệ dòng đầu tiên của mỗi user (delta_t null) | 0,11% | 0,88% |

Vì `Time` của LANL chỉ có **độ phân giải giây**, phần lớn "khoảng cách liên tiếp" bằng 0. Do đó `burst_logon_count` (đếm `delta_t ≤ 2`) thực chất ≈ **số sự kiện trùng giây** ⇒ ρ = 0,91 với volume (và 0,9998 theo Pearson). Nó **không** mang thêm thông tin về "bùng nổ đăng nhập" như tên gọi.

---

## 5. Nhận xét chi tiết từng đặc trưng

Bảng dưới đây đối chiếu từng đặc trưng trong `tables/correlation_*.csv` với (i) hệ số tương quan, (ii) phân phối thực tế, (iii) ngữ nghĩa kiểm chứng trên raw, và (iv) khuyến nghị.

### 5.1. Nhóm Khối lượng & Thất bại

| Đặc trưng | Số liệu chính | Vấn đề | Khuyến nghị |
|---|---|---|---|
| `total_logons` | median 350, p99 4.500, max 2.572.041, skew 90,24 | Cặp **trùng hạng ρ = 1,0000** với `log_total_logons`; là 1 trong 2 biến VIF > 4000 | Giữ **một**. Đề xuất: giữ `log_total_logons` cho model, giữ `total_logons` chỉ để diễn giải/hiển thị |
| `log_total_logons` | min 0,693, max 14,760, skew −1,260 | Chỉ là `log1p` của cột trên | Xem trên |
| `burst_logon_count` | median 202, max 2.572.040; zero-share 2,18% | Xem §4.4: 70,57% sự kiện có `delta_t = 0` ⇒ đây là *đếm sự kiện trùng giây*, không phải burstiness. VIF 3977,17 | **Bỏ.** Nếu muốn giữ ý niệm "bùng nổ", định nghĩa lại: `burst_ratio = burst/total`, hoặc `max_failures_5m`, `max_distinct_hosts_15m` theo [`docs/plan/lanl_eda_implementation_plan.md`](../plan/lanl_eda_implementation_plan.md) §6.5 |
| `failure_count` | 86,04% dòng = 0; median 0; p99 11; max 153.610 | ρ = **0,9969** với `failure_ratio`; trên Pearson chỉ còn 0,1095 → minh chứng rõ nhất cho "đừng dùng Pearson" | Bỏ (giữ `failure_ratio`) — hoặc ngược lại, tuỳ bên nào cần cho báo cáo |
| `failure_ratio` | 86,04% dòng = 0; mean 0,011; p99 0,400 | **Hai cột gần như cùng một chỉ báo nhị phân** "hôm nay có fail hay không". Bị nhiễu theo mẫu số nhỏ (Spearman với `1/total_logons` = −0,0459) | Giữ, nhưng nên bổ sung `max_consecutive_failures` và `failed_then_success_count` (brute-force thành công sau dò mật khẩu) — **hiện chưa có** |
| `wrong_password_count` | **100% giá trị = 0** | `Status` null 100% ở cả 4624 và 4625 | **Xóa khỏi schema** + ghi lý do vào data dictionary |
| `unknown_user_count` | **100% giá trị = 0** | Như trên | **Xóa khỏi schema**; có thể thay bằng `FailureReason` (4625 có 100% giá trị, dùng được ngay) |

### 5.2. Nhóm Thời gian & Nhịp sinh học

| Đặc trưng | Số liệu chính | Vấn đề | Khuyến nghị |
|---|---|---|---|
| `off_hours_count` | 24,14% dòng = 0; skew 74,35 | Pearson 0,9752 với `total_logons` (cặp này **bị thiếu** trong `multicollinear_pairs.csv` vì file chỉ lọc Spearman) | Bỏ |
| `off_hours_ratio` | 24,14% dòng = 0; median 0,284; mean 0,220 | **Định nghĩa không khớp tài liệu**: code dùng 22h–5h, còn `README.md`/plan ghi 18h–7h. Ở mức sự kiện, vùng 6h, 7h, 18h–21h chiếm **21,20%** (4624) / **25,26%** (4625) nhưng **không thuộc đặc trưng nào** | Chốt lại cửa sổ giờ (1 nguồn sự thật), cập nhật README. Cân nhắc thêm `first_event_hour`, `last_event_hour`, `activity_span_hours`, `logon_hour_entropy` |
| `work_hours_ratio` | 1,66% dòng = 0; median 0,466 | ρ = **−0,8050** với `off_hours_ratio`; **88,13%** số dòng có `off + work < 1` (không phải cặp bù trừ hoàn hảo) | Giữ **một** trong hai (đề xuất giữ `off_hours_ratio` + ghi rõ quan hệ bù trừ có chủ ý), hoặc mở rộng định nghĩa để phủ đủ 24 giờ |
| `interarrival_dt_mean` | median 206,5; max 86.400; skew 10,32 | `= 86400` đúng ở **0,841%** số dòng — trùng khớp tuyệt đối với tỷ lệ dòng **chỉ có 1 sự kiện** ⇒ **artifact do `fill_null(86400.0)`**, không phải "nhịp thưa". Kèm `delta_t = 0` chiếm 70% ⇒ giá trị bị chi phối bởi sự kiện trùng giây | Giữ nhưng **ghi rõ artifact**; nên thay bằng cặp `median_delta_t` + `delta_t_cv` và/hoặc thêm `n_distinct_timestamps` |
| `interarrival_dt_std` | median 351,5; max 44.247,9; skew 11,11 | ρ = 0,6020 với `interarrival_dt_mean` (Spearman) | Bỏ; thay bằng **CV = std/mean** |

### 5.3. Nhóm Phương thức truy cập (LogonType) & Gói xác thực

| Đặc trưng | Số liệu chính | Vấn đề | Khuyến nghị |
|---|---|---|---|
| `network_ratio` | median **1,000**; 1,86% dòng = 0; skew −5,73 | Gần như hằng số (vì 4624 có **95,68%** sự kiện là type 3). Là mắt xích trung tâm của cụm cờ nhị phân (§4.2): ρ = −0,9431 với `has_process_info` | Giữ (nó là biến "phân cực" của dữ liệu), nhưng phải hiểu đây là **mô tả chế độ log**, không phải điểm bất thường |
| `interactive_ratio` | **84,44%** dòng = 0; skew 11,83 | Type 2 chỉ chiếm 1,25% sự kiện 4624 | Giữ, nhưng chấp nhận đây là biến gần nhị phân |
| `rdp_ratio` | **96,33%** dòng = 0; skew 19,99 | Type 10 chỉ có **2.230 / 17.553.517 sự kiện (0,013%)**. Đây là **cảnh báo quan trọng**: kịch bản tiêm "Off-hours Compromise" dựa vào RDP sẽ rất khó tạo tín hiệu trên dữ liệu LANL | Bỏ khỏi bộ lõi; **rà soát lại `src/evaluation/injector.py`** để đảm bảo kịch bản tiêm thực sự khác biệt trên các đặc trưng được giữ |
| `service_ratio` | **99,73%** dòng = 0 | Type 5 = 2,13% sự kiện | Bỏ khỏi bộ lõi |
| `batch_ratio` | **99,57%** dòng = 0 | Type 4 = 0,013% sự kiện | Bỏ khỏi bộ lõi |
| 5 ratio hợp lại | Phủ **99,08%** (4624) / **98,80%** (4625) sự kiện | Ở mức dòng, **32,39%** dòng có tổng < 1 vì có sự kiện thuộc type 7/8/9/11 (≈1% khối lượng) **không được biểu diễn** | Thêm 1 đặc trưng `other_logon_type_ratio` (hoặc `distinct_logon_types`) để đóng ràng buộc tổng |
| `ntlm_ratio` | 31,39% dòng = 0; median 0,092 | Đặc trưng ratio **tốt nhất** trong nhóm. Nhưng NTLM chiếm **48,18%** sự kiện 4625 so với **17,35%** ở 4624 ⇒ nó là *proxy của chế độ thất bại*, không thuần là "phương thức xác thực" | Giữ, nhưng diễn giải cẩn thận (không kết luận "bất thường NTLM" khi thực chất là "có nhiều đăng nhập thất bại") |

### 5.4. Nhóm Đa dạng thực thể (Fan-out) & Ngữ cảnh Nguồn (Source)

| Đặc trưng | Số liệu chính | Vấn đề | Khuyến nghị |
|---|---|---|---|
| `distinct_hosts_count` | median 3, p99 11, max 9.981, skew 95,53 | Cặp **trùng hạng ρ = 1,0000** với `log_distinct_hosts`. Rất giàu thông tin cho lateral movement nhưng cực lệch: riêng tài khoản Service có median **1.870,5** | Bỏ bản thô |
| `log_distinct_hosts` | min 0,693, max 9,209, skew 1,298 | Bản log — phân phối tốt hơn hẳn | **Giữ** |
| `distinct_sources_count` | 1,20% dòng = 0; median 1; max 2.355; skew 139,08 | Cần log + chuẩn hoá within-account | Giữ (kèm log transform) |
| `missing_source_ratio` | mean 0,060; median 0,040; 26,23% dòng = 0 | Bản chất là **chỉ báo chất lượng dữ liệu**, không phải tín hiệu tấn công: thiếu `Source` tập trung ở Network+Kerberos (**90,57%** số thiếu của 4624) và Service+Negotiate (gần 100%) | **Giữ nhưng tách vai trò**: dùng làm covariate/chú thích chất lượng, **không** đưa vào điểm bất thường. Tránh kết luận "tấn công" khi thực chất là một đợt outage ghi log |
| `has_missing_source` | 26,23% dòng = 0 | ρ = 0,7689 với `missing_source_ratio` ⇒ chỉ là bản nhị phân hoá | Bỏ |
| `has_remote_logon` | **Prevalence 96,60%** | Gần như hằng số | **Bỏ** |
| `remote_logon_ratio` | mean 0,894; median 0,955; skew −3,57 | Biến "gần hằng số" nhưng vẫn giữ được phần thông tin ở đuôi thấp | Giữ (biến biên nhiều thông tin hơn cờ nhị phân tương ứng) |
| `has_local_logon` | Prevalence 37,67% | ρ = **0,9738** với `has_process_info`; bị nhiễu bởi `Source` null (§4.2) | **Bỏ** |

### 5.5. Nhóm Ngữ cảnh Tiến trình, Phiên LUID, Domain & Khoá định danh

| Đặc trưng | Số liệu chính | Vấn đề | Khuyến nghị |
|---|---|---|---|
| `has_process_info` | Prevalence 38,37% | `ProcessName` null **92,37%** ở 4624 ⇒ cờ này chủ yếu là proxy của "đã có sự kiện không phải Network". VIF 51,17; là mắt xích trung tâm của cụm trùng lặp | **Bỏ** |
| `has_system_process` | Prevalence 35,75% | Tập con của `has_process_info` (ρ = 0,9455); danh sách 4 tên (`services.exe`, `lsass.exe`, `winlogon.exe`, `svchost.exe`) là hard-code tuỳ ý — raw còn nhiều tiến trình hệ thống khác | **Bỏ** (nếu muốn giữ khái niệm, thay bằng `system_proc_ratio` và mở rộng danh sách có căn cứ) |
| `has_custom_proc` | Prevalence 6,26% | Heuristic `.starts_with("proc")`; raw cho thấy tên dạng `Proc415255.exe` chiếm 4,64% sự kiện 4624 nhưng **48,76%** sự kiện 4625 ⇒ cờ này phần lớn là "có sự kiện thất bại", trùng thông tin với `failure_ratio` | Giữ tạm trong nhóm thử nghiệm A/B; tốt hơn là thay bằng `custom_proc_ratio` hoặc `distinct_process_count` |
| `has_system_logon_id` | Prevalence 12,09% | `LogonID == "0x3e7"` chiếm **45,92%** sự kiện 4625 nhưng chỉ **2,09%** sự kiện 4624 (và `LogonID` null 49,16% ở 4625) ⇒ giải thích trực tiếp ρ = **0,9104** với `failure_count`. Đây là **proxy của failure**, không phải "phiên SYSTEM". Thêm nữa `LogonID` unique 16.931.277/17.553.517 ở 4624 (≈1 LUID/sự kiện) ⇒ khái niệm "phiên logon" của LANL rất khác Windows thật | **Bỏ** + ghi chú vào data dictionary |
| `has_multi_domains` | **99,18%** dòng = 0 (prevalence 0,82%) | Gần như không dùng được | Bỏ khỏi bộ lõi |
| `has_local_domain` | **99,13%** dòng = 0 (prevalence 0,87%) | Heuristic `starts_with("comp")` bắt **0,05%** sự kiện 4624 nhưng **9,06%** sự kiện 4625 (4624: `Domain001` 94,5%; 4625: thêm `Domain002` 33%). Đây là lý do Pearson(`has_local_domain`, `failure_ratio`) = **0,3354** ⇒ cờ này đang *đếm thất bại ở domain cục bộ*, không phải "đăng nhập domain nội bộ" | Định nghĩa lại (ví dụ `domain_entropy`, `new_domain_for_user`) hoặc bỏ |
| **Khoá `UserName`** | Panel hiện key theo `UserName` | Cùng một `UserName` xuất hiện ở `Domain001`, `Domain002`, `nt authority`, `EnterpriseAppServer` ⇒ **trộn nhiều danh tính vào cùng một hàng**. `reports/day16_missingness.md` §3 đã khuyến nghị giữ `unknown` khi tạo khoá `DomainName + UserName` | **Ưu tiên số 1**: sửa khoá thành `DomainName + UserName` (giữ nhãn `unknown` cho 4 + 33 dòng thiếu) — nếu không, mọi kết quả phía sau đều lệch |

---

## 6. Phê bình phương pháp luận của phép phân tích tương quan

### 6.1. Trộn quần thể (confounding by `entity_type`) — vấn đề nặng nhất về phương pháp

Panel gộp Machine (54,93% số dòng, 21,4M sự kiện), User (45,03%, 18,5M) và Service (0,04% số dòng nhưng 8,6M sự kiện). Kiểm chứng bằng **chuẩn hoá z-score theo từng tài khoản** (chỉ số đúng cho UEBA, vì anomaly = lệch so với baseline của chính entity):

| Cặp | Spearman pooled | Spearman within-account | Nhận xét |
|---|---:|---:|---|
| `network_ratio ~ has_local_logon` | **−0,914** | −0,362 | Phần lớn quan hệ là *between-entity* |
| `network_ratio ~ has_process_info` | **−0,943** | −0,376 | Như trên |
| `has_missing_source ~ network_ratio` | **+0,383** | **−0,061** | **Quan hệ biến mất hoàn toàn** ⇒ đây là hiện tượng "một số host luôn thiếu Source", không phải đặc điểm hành vi |
| `total_logons ~ off_hours_count` | 0,782 | 0,427 | Giảm ~45% |
| `off_hours_ratio ~ work_hours_ratio` | −0,817 | −0,542 | Giảm ~34% |
| `failure_count ~ failure_ratio` | 0,997 | 0,889 | Quan hệ thật (vẫn rất mạnh) |
| `total_logons ~ burst_logon_count` | 0,910 | 0,825 | Quan hệ thật (nhưng do artifact delta_t, xem §4.4) |

**Số cặp |ρ| ≥ 0,85: 13 (pooled) → 5 (within-account).**

Bổ sung: median đặc trưng lệch hẳn giữa các quần thể —

| `entity_type` | `total_logons` | `network_ratio` | `off_hours_ratio` | `distinct_hosts_count` | `interarrival_dt_mean` | `failure_ratio` (p99) |
|---|---:|---:|---:|---:|---:|---:|
| Machine | 359,0 | 1,000 | 0,316 | 3,0 | 217,0 | 0,004 |
| User | 318,0 | 0,978 | 0,165 | 5,0 | 170,8 | **0,937** |
| Service | 238.392,5 | 0,969 | 0,261 | **1.870,5** | 0,4 | — |

⇒ Kết luận: bảng tương quan pooled **không dùng được để ra quyết định chọn đặc trưng** cho UEBA. Phải báo cáo tối thiểu 3 phiên bản: pooled, within-account, và theo từng `entity_type`.

### 6.2. Ma trận đặc trưng chỉ có 3 ngày (day 1–3)

- 17.759 / 19.465 / 19.507 tài khoản mỗi ngày; **80,13%** tài khoản có mặt đủ 3 ngày; số ngày/tài khoản: mean 2,72, median 3, max 3.
- Không thể ước lượng baseline ổn định, không tách được weekday/weekend, không phát hiện được drift.

⇒ Bảng tương quan này nên được **chạy lại trên ≥ 14–30 ngày** trước khi chốt feature contract.

### 6.3. Hệ số tương quan chưa phù hợp với dữ liệu bị chặn (truncated)

86,04% dòng = 0 ở nhóm failure; 96–99% = 0 ở nhóm RDP/service/batch. Spearman với lượng ties khổng lồ là thống kê kém hiệu lực: `failure_count ~ failure_ratio` = 0,9969 thực chất chỉ nói "hai cột cùng là chỉ báo của *có ít nhất một thất bại*".

⇒ Với cờ nhị phân nên báo cáo thêm **Cramér's V / point-biserial / mutual information**, và với các ước lượng đơn lẻ nên kèm **khoảng tin cậy bootstrap** thay vì 1 điểm.

### 6.4. Ratio bị nhiễu theo mẫu số nhỏ

| Tương quan Spearman với `1/total_logons` | Giá trị |
|---|---:|
| `off_hours_ratio` | **−0,3422** |
| `failure_ratio` | −0,0459 |
| `interactive_ratio` | +0,0617 |
| `missing_source_ratio` | +0,0892 |
| `network_ratio` | −0,0176 |

Phần lớn phương sai của `off_hours_ratio` đến từ **số sự kiện trong ngày** (tài khoản chỉ có 1 logon thì mọi ratio ∈ {0, 1}), không từ hành vi.

⇒ Nên tính `ratio = sum(tử) / sum(mẫu)` trên **cửa sổ trượt** có `min_denominator`, thay vì ratio của từng ngày đơn lẻ.

---

## 7. Bộ đặc trưng đề xuất rút gọn (30 → 18)

### 7.1. Danh sách loại bỏ (12)

| Đặc trưng bỏ | Lý do |
|---|---|
| `total_logons` | Trùng hoàn toàn về hạng với `log_total_logons` (ρ = 1,0000) |
| `off_hours_count` | Trùng với `total_logons` (ρ = 0,7818) và `off_hours_ratio` (ρ = 0,7261) |
| `burst_logon_count` | ρ = 0,9102 với volume; 70,57% `delta_t = 0` ⇒ không đo burstiness (§4.4) |
| `failure_count` | ρ = 0,9969 với `failure_ratio` |
| `distinct_hosts_count` | Trùng hạng với `log_distinct_hosts` (ρ = 1,0000) |
| `has_local_logon` | ρ = 0,9738 với `has_process_info`; bị nhiễu bởi `Source` null |
| `has_process_info` | Trùng nguồn với `has_system_process` / `has_local_logon` |
| `has_system_process` | Tập con của `has_process_info` (ρ = 0,9455) |
| `has_system_logon_id` | Proxy của `failure_count` (ρ = 0,9104) qua `LogonID = 0x3e7` của 4625 |
| `has_remote_logon` | Gần như hằng số (prevalence 96,60%) |
| `has_missing_source` | Bản nhị phân hoá của `missing_source_ratio` (ρ = 0,7689) |
| `interarrival_dt_std` | ρ = 0,6020 với `interarrival_dt_mean`; nên dùng `CV = std/mean` |

### 7.2. Bộ giữ lại (18) và hiệu quả định lượng

`failure_ratio, off_hours_ratio, work_hours_ratio, interarrival_dt_mean, network_ratio, interactive_ratio, rdp_ratio, service_ratio, batch_ratio, ntlm_ratio, log_total_logons, log_distinct_hosts, missing_source_ratio, remote_logon_ratio, distinct_sources_count, has_custom_proc, has_multi_domains, has_local_domain`

| Chỉ số | Bộ 30 hiện tại | Bộ 18 rút gọn |
|---|---:|---:|
| Cặp \|ρ\| ≥ 0,85 | 13 | **0** |
| Cặp \|ρ\| ≥ 0,70 | 18 | **1** |
| VIF lớn nhất | 4041,96 | **5,80** |
| Số đặc trưng VIF > 10 | 6 | **0** |

Cặp còn lại > 0,70 trong bộ rút gọn: `off_hours_ratio ~ work_hours_ratio` = 0,8173 — chấp nhận được nếu ghi rõ đây là cặp "bù trừ có chủ ý", hoặc chỉ giữ 1 trong 2.

> **Lưu ý:** `has_custom_proc`, `has_multi_domains`, `has_local_domain` tuy sống sót qua lọc tương quan nhưng yếu về prevalence và ngữ nghĩa — hãy xếp chúng vào nhóm "thử nghiệm A/B", không đưa vào bộ lõi.

### 7.3. Đặc trưng cần bổ sung (đã có trong plan, chưa được implement)

Theo [`docs/plan/lanl_eda_implementation_plan.md`](../plan/lanl_eda_implementation_plan.md) §6.5:

- **Burst theo cửa sổ thời gian:** `max_failures_5m`, `max_distinct_hosts_15m` (thay cho `burst_logon_count`).
- **Sequence:** `failed_then_success_count` (dấu hiệu dò mật khẩu thành công).
- **Novelty:** `new_source_count`, `new_loghost_count`, `new_logon_type_count` (bản chất UEBA).
- **Distribution:** `logon_hour_entropy`, `host_entropy`.
- **Temporal:** `first_event_second`, `last_event_second`, `activity_span_hours`.

Đây mới là nhóm đặc trưng **không trùng nhau về mặt cấu trúc** như các cờ `has_*` hiện tại.

---

## 8. Checklist việc cần làm trước khi sang tuần Model

| # | Việc | Ưu tiên | Căn cứ |
|---|---|---|---|
| 1 | Sửa khoá danh tính thành `DomainName + UserName` (giữ nhãn `unknown` cho 4 + 33 dòng thiếu) | **P0** | §5.5 — hiện đang trộn nhiều danh tính vào cùng tài khoản |
| 2 | Xóa 2 đặc trưng "chết" (`wrong_password_count`, `unknown_user_count`) khỏi schema + ghi lý do vào `docs/data_dictionary.md`; cân nhắc thay bằng `FailureReason` (4625 có 100% giá trị) | **P0** | §3.1, §5.1 |
| 3 | Đồng bộ tài liệu: `README.md` & `docs/project_structure.md` đang mô tả "16 đặc trưng" với tên khác hẳn bộ 32 đặc trưng code đang sinh (`failed_logons`, `weekend_logons`, `distinct_ips`, `logon_type_2_count`, `max_consecutive_failures`, `activity_span_hours`…) | **P0** | §5 — cần 1 schema canonical trước khi cả nhóm cùng build |
| 4 | Chốt lại cửa sổ giờ (README 18h–7h vs code 22h–5h) và xử lý vùng trống 21,20% / 25,26% sự kiện | **P1** | §5.2 |
| 5 | Áp dụng bộ rút gọn 30 → 18 đặc trưng (§7) | **P1** | §7.2 |
| 6 | Nâng cấp `plot_feature_correlation.py`: (a) lọc theo trị tuyệt đối **lớn hơn** trong hai hệ số (Pearson/Spearman), (b) thêm tương quan **within-account** và theo `entity_type`, (c) in **VIF**, (d) báo cáo **Cramér's V** cho cờ nhị phân; chạy lại trên **≥ 14 ngày** | **P1** | §3.2, §6.1, §6.2, §6.3 |
| 7 | Tiền xử lý cho IForest/LOF/OCSVM: `log1p` cho mọi count, **rank-Gaussian / RobustScaler** cho mọi biến (max/median tới ~7.000 lần), chuẩn hoá theo baseline từng tài khoản; tách/loại nhóm 8 tài khoản Service (0,04% dòng nhưng 17,65% khối lượng) | **P1** | §4.3 |
| 8 | Rà soát lại kịch bản tiêm (`src/evaluation/injector.py`): "Off-hours Compromise" dựa vào RDP nhưng RDP chỉ chiếm 0,013% sự kiện thật; "Brute Force" dựa vào failure ratio nhưng `Status` null 100% nên chỉ còn 4625 count | **P2** | §5.3 |
| 9 | Bổ sung nhóm đặc trưng burst theo cửa sổ / sequence / novelty / entropy (§7.3) | **P2** | §7.3 |

---

## 9. Phụ lục

### Phụ lục A. Số liệu kiểm chứng trên raw

#### A.1. Phân bố `LogonType` (ngày 16)

| LogonType | 4624 (n) | 4624 (%) | 4625 (n) | 4625 (%) |
|---|---:|---:|---:|---:|
| 3 — Network | 16.794.417 | **95,676%** | 334.377 | **92,909%** |
| 5 — Service | 373.737 | 2,129% | 146 | 0,041% |
| 2 — Interactive | 219.613 | 1,251% | 18.872 | 5,244% |
| 7 — Unlock | 78.760 | 0,449% | 3.624 | 1,007% |
| 8 — NetworkCleartext | 62.286 | 0,355% | 117 | 0,033% |
| 9 — NewCredentials | 13.451 | 0,077% | 0 | 0% |
| 0 — System | 5.786 | 0,033% | 0 | 0% |
| 4 — Batch | 2.321 | 0,013% | 1.906 | 0,530% |
| **10 — RemoteInteractive (RDP)** | **2.230** | **0,013%** | 267 | 0,074% |
| 11 — CachedInteractive | 916 | 0,005% | 588 | 0,163% |
| **5 type dùng cho ratio phủ** | 17.392.318 | **99,082%** | 355.568 | **98,797%** |

#### A.2. Cửa sổ giờ (ngày 16)

| Khoảng giờ | 4624 | 4625 |
|---|---:|---:|
| `off_hours` theo code (22h–5h) | 19,599% | 25,863% |
| `work_hours` theo code (8h–17h) | 59,197% | 48,876% |
| **Vùng trống 6h, 7h, 18h–21h** | **21,204%** | **25,261%** |

> Lưu ý: `README.md` mô tả off-hours là 18h–7h, trong khi code dùng 22h–5h → hai vùng "off-hours" rất khác nhau. Cần chốt 1 định nghĩa.

#### A.3. Missingness có cấu trúc (ngày 16)

| Trường | 4624 | 4625 |
|---|---:|---:|
| `Status` null | **100,00%** | **100,00%** |
| `Source` null | 24,279% | 32,085% |
| `ProcessName` null | **92,37%** | 49,16% |
| `LogonID` null | 0,160% | 49,16% |
| `LogonID == 0x3e7` | 2,092% | **45,922%** |
| `LogonID` unique | 16.931.277 | **82** |
| `Source == LogHost` (local) | 18,80% | 50,89% |
| `Source != LogHost` (remote) | 56,92% | 17,03% |
| `DomainName` bắt đầu bằng `comp` | 0,05% | **9,06%** |
| `AuthenticationPackage = NTLM` | 17,354% | **48,175%** |
| `AuthenticationPackage = Kerberos` | 76,813% | 44,618% |

> `ProcessName` null ở 4625 (49,16%) **trùng khớp tuyệt đối** với `LogonID` null (49,16%) ⇒ hai trường mất cùng nhau theo từng bản ghi.

#### A.4. `Source` null theo `LogonType` (ngày 16)

| LogonType | 4624 (n) | 4624 (% null) | 4625 (n) | 4625 (% null) |
|---|---:|---:|---:|---:|
| 3 — Network | 16.794.417 | 23,02% | 334.377 | 34,44% |
| 5 — Service | 373.737 | **99,92%** | 146 | 31,51% |
| 2 — Interactive | 219.613 | 0,99% | 18.872 | 0,01% |
| 9 — NewCredentials | 13.451 | **100,00%** | — | — |
| 0 — System | 5.786 | **100,00%** | — | — |
| 10 — RemoteInteractive | 2.230 | 35,52% | 267 | **100,00%** |

#### A.5. Thống kê mô tả bổ sung trên panel (56.731 dòng)

| Đặc trưng | Mean | Std | Median | p99 | Max | Skew | % giá trị 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `total_logons` | 855,31 | 20.205,16 | 350 | 4.500 | 2.572.041 | 90,24 | 0,00% |
| `failure_count` | 16,27 | 1.073,90 | 0 | 11 | 153.610 | 115,43 | 86,04% |
| `failure_ratio` | 0,011 | 0,087 | 0 | 0,400 | 1,000 | 9,78 | 86,04% |
| `off_hours_count` | 231,46 | 5.201,54 | 97 | 1.516,7 | 589.492 | 74,35 | 24,14% |
| `off_hours_ratio` | 0,220 | 0,161 | 0,284 | 0,697 | 1,000 | 0,46 | 24,14% |
| `work_hours_ratio` | 0,562 | 0,224 | 0,466 | 1,000 | 1,000 | 0,62 | 1,66% |
| `interarrival_dt_mean` | 1.061,99 | 8.009,12 | 206,53 | 18.075,68 | 86.400 | 10,32 | 0,28% |
| `interarrival_dt_std` | 574,84 | 1.139,73 | 351,49 | 5.251,66 | 44.247,91 | 11,11 | 1,56% |
| `burst_logon_count` | 661,34 | 20.158,96 | 202 | 2.699 | 2.572.040 | 90,78 | 2,18% |
| `network_ratio` | 0,961 | 0,150 | **1,000** | 1,000 | 1,000 | −5,73 | 1,86% |
| `interactive_ratio` | 0,008 | 0,068 | 0 | 0,125 | 1,000 | 11,83 | 84,44% |
| `rdp_ratio` | — | 0,047 | 0 | — | 1,000 | 19,99 | **96,33%** |
| `service_ratio` | — | 0,039 | 0 | — | 1,000 | 24,85 | **99,73%** |
| `batch_ratio` | — | 0,036 | 0 | — | 1,000 | 26,52 | **99,57%** |
| `ntlm_ratio` | 0,115 | 0,175 | 0,092 | 1,000 | 1,000 | 3,52 | 31,39% |
| `distinct_hosts_count` | 5,03 | 83,15 | 3 | 11 | 9.981 | 95,53 | 0,00% |
| `distinct_sources_count` | 1,34 | 15,30 | 1 | 4 | 2.355 | 139,08 | 1,20% |
| `missing_source_ratio` | 0,060 | 0,141 | 0,040 | 1,000 | 1,000 | 5,60 | 26,23% |
| `remote_logon_ratio` | 0,894 | 0,210 | 0,955 | 1,000 | 1,000 | −3,57 | 3,40% |

#### A.6. Độ phổ biến (prevalence) của các cờ nhị phân

| Cờ | Prevalence | Ghi chú |
|---|---:|---|
| `has_remote_logon` | **96,60%** | Gần như hằng số |
| `has_missing_source` | 73,77% | Trùng thông tin với `missing_source_ratio` |
| `has_process_info` | 38,37% | Proxy của "có sự kiện không phải Network" |
| `has_local_logon` | 37,67% | ρ = 0,9738 với `has_process_info` |
| `has_system_process` | 35,75% | Tập con của `has_process_info` |
| `has_system_logon_id` | 12,09% | Proxy của failure |
| `has_custom_proc` | 6,26% | Heuristic `starts_with("proc")` |
| `has_local_domain` | 0,87% | Heuristic `starts_with("comp")` |
| `has_multi_domains` | 0,82% | Gần như không dùng được |
| `wrong_password_count` / `unknown_user_count` | **0%** | Đặc trưng chết |

#### A.7. Trùng lặp bản ghi & khoảng cách thời gian (ngày 16)

| Chỉ số | 4624 | 4625 |
|---|---:|---:|
| Sự kiện có `delta_t = 0` | 12.387.169 (**70,568%**) | 241.822 (**67,192%**) |
| Sự kiện có `0 < delta_t ≤ 2` | 2.092.670 | 51.254 |
| Dòng đầu mỗi user (delta_t null) | 18.696 (0,11%) | 3.151 (0,88%) |

### Phụ lục B. Cách tái lập các số liệu trong tài liệu này

Tất cả số liệu được sinh bằng các script Python tạm (đã xóa sau khi hoàn tất) chạy trong conda env `ueba-benchmark`. Để tái lập, cần 3 nhóm phép tính sau:

**B.1. Tái lập & kiểm chứng 2 bảng tương quan**

```python
import numpy as np, pandas as pd, polars as pl

df = pl.read_parquet("data/features/account_day_matrix.parquet").to_pandas()
num = [c for c in df.select_dtypes(include=[np.number]).columns
       if c not in ("day",)]                     # bỏ UserName/day/entity_type
X = df[num].replace([np.inf, -np.inf], np.nan)
X = X.loc[:, X.std() > 1e-9].fillna(0.0)         # loại 2 cột đẳng trị

mine_p = X.corr(method="pearson")
mine_s = X.corr(method="spearman")
csv_p = pd.read_csv("docs/feature_engineering/tables/correlation_pearson.csv", index_col=0)
csv_s = pd.read_csv("docs/feature_engineering/tables/correlation_spearman.csv", index_col=0)
print(np.abs(mine_p - csv_p.loc[mine_p.index, mine_p.columns]).max())  # ~4.99e-05
```

**B.2. VIF và bộ đề xuất rút gọn** — tính `VIF_j = diag(inv(R))_j` từ ma trận Pearson
(hoặc đọc trực tiếp `docs/feature_engineering/tables/correlation_pearson.csv`), sau đó đếm lại số cặp có |ρ| ≥ 0.85 trên tập con đã rút gọn.

**B.3. Kiểm chứng ngữ nghĩa trên raw** — **quan trọng: dùng MỘT lượt `collect()` cho mỗi EventID**
(scan lại parquet cho từng phép tính sẽ chậm hơn nhiều lần trên 17,5 triệu dòng):

```python
cols = ["EventID", "UserName", "LogHost", "LogonType", "AuthenticationPackage",
        "Status", "ProcessName", "Source", "LogonID", "DomainName", "Time"]
ev = pl.scan_parquet("data/interim/event_4624/event_4624_day-16.parquet").select(cols).collect()
# rồi tính tất cả chỉ số trên `ev` trong bộ nhớ
```

> **Lưu ý hiệu năng:** `pl.struct(pl.all()).n_unique()` trên 17,5 triệu dòng rất tốn thời gian; nếu chỉ cần thống kê missingness/LogonType thì **không** cần phép tính này.

---

## 10. Kết luận

Kết quả tương quan bạn đưa là **một phân tích làm đúng về mặt kỹ thuật và có tính tái lập cao** (kiểm chứng độc lập: sai số 4,99e-05). Quan trọng hơn, nó đã làm đúng việc của nó: **nó tố cáo tập đặc trưng hiện tại**.

Đọc đúng thì thấy **đa số tương quan mạnh trong bảng không phải tín hiệu an ninh**, mà là:

| Nguồn gây tương quan | Ví dụ điển hình | Xử lý |
|---|---|---|
| **Bản sao qua biến đổi đơn điệu** | `total_logons ↔ log_total_logons` = 1,0000; `distinct_hosts_count ↔ log_distinct_hosts` = 1,0000 | Giữ 1 trong 2 |
| **Trùng nguồn dữ liệu (cùng dẫn xuất từ 1 cột raw)** | `has_process_info / has_system_process / has_custom_proc` đều từ `ProcessName`; `has_local_logon / has_remote_logon` đều từ `Source` | Giữ tối đa 1 đại diện mỗi nguồn |
| **Trộn quần thể (Simpson)** | `network_ratio ~ has_local_logon`: −0,914 pooled → −0,362 within-account | Báo cáo within-account + theo `entity_type` |
| **Missingness có cấu trúc của LANL** | `Status` null 100%; `ProcessName` null 92,37% (4624); `Source` null 24,28%/32,09%; `LogonID` null = `ProcessName` null ở 4625 (49,16%) | Không biến missingness thành đặc trưng hành vi |
| **Chế độ thất bại (failure regime)** | `has_system_logon_id ↔ failure_count` = 0,9104; NTLM 48,18% ở 4625 vs 17,35% ở 4624; `has_custom_proc` 48,76% ở 4625 | Diễn giải là "có thất bại", không phải "bất thường giao thức" |

**Ba việc phải làm trước khi huấn luyện:** (1) sửa khoá danh tính thành `DomainName + UserName`; (2) xóa 2 đặc trưng chết và đồng bộ lại schema canonical giữa code – README – data dictionary; (3) rút gọn 30 → 18 đặc trưng (đưa số cặp |ρ| ≥ 0,85 từ 13 về 0 và VIF lớn nhất từ 4042 về 5,8), kèm `log1p` + chuẩn hoá theo baseline từng tài khoản cho các mô hình dựa trên khoảng cách.

---

## 11. Kế hoạch sửa (bám theo file:line)

> Mục tiêu: đưa tập đặc trưng về trạng thái **dùng được cho benchmark** — không còn đặc trưng chết, không còn trùng lặp cấu trúc, không còn trộn danh tính. Mọi con số dẫn chứng lấy từ §1–§9.

### 11.1. P0 — Phải sửa trước khi chạy model (5 việc)

| # | Việc | Vị trí | Hiện trạng (đo được) | Cần sửa thành |
|---|---|---|---|---|
| 1 | **Khoá danh tính** | `scripts/feature_engineering/extract_account_day_matrix.py:86` (`group_by("UserName")`) và `:176-213` (`ordered_cols`) | Panel key chỉ có `UserName`; raw có **206** `DomainName`; cùng `UserName` xuất hiện ở `Domain001` và ở `nt authority` ⇒ trộn nhiều danh tính | `group_by(["DomainName", "UserName"])`, thêm `DomainName` vào `ordered_cols`; cân nhắc `user_key = DomainName + "\\" + UserName`; giữ nhãn `unknown` khi thiếu (4 + 33 dòng) |
| 2 | **Sửa luật phân loại `entity_type`** | Same file `:148-152` (`endswith("$")` → Machine, `startsWith("User")` → User, **còn lại** → Service) | Nhóm "Service" thực tế chỉ 8 tài khoản: `AppService`, `Scanner`, `Administrator` (**1.275.672 sự kiện — là tài khoản admin người**), `Anonymous`, `system`, `winservice`, `network service`, `local service`. Luật "còn lại" gom tài khoản hệ thống + tài khoản admin vào cùng một nhãn | Tách 4 nhóm tường minh: `Machine` (`*$`), `User` (`User*`), `Service` (`AppService`, `Scanner`, `winservice`), `System` (`system`, `network service`, `local service`, `Anonymous`, `-`); đưa danh sách vào `configs/system_config.yaml` |
| 3 | **Thực thi `ignore_system_accounts`** | `configs/system_config.yaml:22-27` — **không nơi nào trong code đọc nó** | Config ghi `SYSTEM / LOCAL SERVICE / NETWORK SERVICE / ANONYMOUS LOGON / -` nhưng dữ liệu ghi `system / network service / local service / Anonymous`. Nếu áp nguyên văn (case-insensitive): lọc được **379.700 dòng (2,16% sự kiện 4624)** nhưng **bỏ sót `Anonymous` (379.421 dòng = 2,16%)** | Chuẩn hoá so khớp (`strip()` + `casefold()`), sửa list theo giá trị thật (`anonymous`, `-`), và **áp dụng thật** luật lọc/nhãn trong `extract_*`. Quyết định rõ: **lọc bỏ** hay **giữ nhưng gán nhãn `System`** |
| 4 | **Xoá 2 đặc trưng chết** | Same file `:107-108` (định nghĩa) và `:195-196` (`ordered_cols`) | `Status` null **100%** ở cả 4624 và 4625 ⇒ 2 cột **luôn = 0** (đã bị `plot_feature_correlation.py:41-45` loại âm thầm) | Xoá khỏi `agg` + `ordered_cols`; thay bằng count theo **`FailureReason`** của 4625 (trường này có 100% giá trị) |
| 5 | **Chốt cửa sổ giờ** | Code `:92-93` (off 22h–5h, work 8h–17h) vs `configs/system_config.yaml:32-33` (off 18h–7h) vs `README.md:101` (18h–7h) | **3 định nghĩa khác nhau**; **21,20%** (4624) và **25,26%** (4625) sự kiện rơi vào vùng 6h, 7h, 18h–21h không thuộc đặc trưng nào | Chọn 1 nguồn sự thật (khuyến nghị: `system_config.yaml`), cho script đọc `--config`; đảm bảo `off_hours + work_hours` phủ đủ 24h **hoặc** ghi rõ vùng bỏ trống trong tài liệu đặc trưng |

### 11.2. P1 — Sửa để đúng phương pháp & hết mâu thuẫn code/doc (5 việc)

| # | Việc | Vị trí | Hiện trạng (đo được) | Cần sửa thành |
|---|---|---|---|---|
| 6 | **Cho extractor đọc config** | `extract_account_day_matrix.py:295-309` (`argparse` chỉ có `--start-day/--end-day/--interim-dir/--output-dir`) | Các tham số nghiệp vụ (`off_hours_start/end`, `weekend_days`, `filter_machine_accounts`) nằm trong YAML nhưng code hard-code ⇒ sửa YAML không có tác dụng | Thêm `--config configs/system_config.yaml`; dùng `features.off_hours_start/end`; bổ sung `is_weekend`, `weekend_logon_count` (hiện **chưa có** dù config đã khai báo `weekend_days`) |
| 7 | **Định nghĩa lại burst** | Same file `:96` (`delta_t ≤ 2`) | 70,57% (4624) / 67,19% (4625) sự kiện có `delta_t = 0` ⇒ `burst_logon_count` ≈ đếm sự kiện trùng giây; ρ = 0,91 với volume, VIF 3977 | Thay bằng burst **theo cửa sổ thời gian**: `max_failures_5m`, `max_distinct_hosts_15m` (theo [`plan §6.5`](../plan/lanl_eda_implementation_plan.md)); nếu giữ thì chỉ dùng dạng `burst_ratio = burst/total` |
| 8 | **Bỏ artefact `fill_null(86400.0)`** | Same file `:145` | `interarrival_dt_mean = 86400` xuất hiện đúng ở **0,841%** số dòng — trùng khớp tuyệt đối tỷ lệ dòng chỉ có 1 sự kiện ⇒ đỉnh giả, không phải "nhịp thưa" | Bỏ fill cứng; thêm `n_distinct_timestamps`, `median_delta_t`, `delta_t_cv = std/mean` |
| 9 | **Đóng ràng buộc tổng của ratio** | Same file `:98-103` (agg 5 type) và `:137-142` (ratio) | Type 7/8/9/11 (≈ 1% khối lượng) **không được biểu diễn**; **32,39%** số dòng có tổng 5 ratio < 1; `off + work` cũng chỉ phủ 78,19% (mean) | Thêm `other_logon_type_ratio` và `distinct_logon_types` để tổng ratio = 1 |
| 10 | **Rút gọn 30 → 18 đặc trưng** | Same file `:176-213` (`ordered_cols`) | 13 cặp \|ρ\| ≥ 0,85; VIF lớn nhất 4041,96; 6 đặc trưng VIF > 10 (chi tiết §7) | Áp theo §7; nên tách `configs/feature_schema.yaml` có cột `is_core: true/false` để vẫn giữ được bảng đầy đủ cho báo cáo nhưng model chỉ dùng bộ lõi |

### 11.3. P2 — Hạ tầng & tài liệu (6 việc)

| # | Việc | Vị trí | Hiện trạng | Cần sửa thành |
|---|---|---|---|---|
| 11 | **`main.py` thiếu stage** | `main.py:54-64` (in ra "sẽ được triển khai theo tiến độ"), `:81-87` (`--stage` choices) | `README.md:85-89` hướng dẫn `python main.py --stage features` và `--stage benchmark --models ...` nhưng code **không implement** và **không có** tham số `--models` | Implement `features` → gọi `build_account_day_matrix`; thêm `--models`; hoặc sửa README cho khớp thực tế |
| 12 | **Tên cột trong config sai schema** | `configs/system_config.yaml:13-18` | Config ghi `TimeCreated / TargetUserName / IpAddress / WorkstationName`, interim thực tế là `Time / UserName / Source / LogHost` | Cập nhật theo schema interim (1 nguồn sự thật cho tên cột) |
| 13 | **Chốt đường dẫn output** | `system_config.yaml` (`processed_data_dir`) vs `extract_*.py:300` (`--output-dir data/features`) vs `docs/project_structure.md:74` | 3 nơi ghi 2 đường dẫn khác nhau cho cùng ma trận User × Day | Chốt 1 (khuyến nghị `data/features/` như script đang chạy) và sửa lại config + docs |
| 14 | **README "16 đặc trưng"** | `README.md:93-113`, `docs/project_structure.md:84-90` | Liệt kê 16 tên khác hẳn bộ 32 cột thực tế (`failed_logons`, `weekend_logons`, `distinct_ips`, `logon_type_2_count`, `max_consecutive_failures`, `activity_span_hours`…) | Cập nhật theo schema canonical (bộ 18 lõi + phần mở rộng), kèm công thức & artifact đã biết |
| 15 | **Viết unit test cho đặc trưng** | `tests/` hiện chỉ có `__init__.py` | `README.md:116-122` quảng cáo `pytest` kiểm tra "hàm tính toán đặc trưng" nhưng **không có test nào** | Thêm test cho: cửa sổ giờ (biên 5/6/8/17/18/21/22h), `delta_t` & burst, khoá `(DomainName, UserName)`, luật `entity_type`, ràng buộc `sum(ratio) ≤ 1` |
| 16 | **Nâng cấp script phân tích** | `plot_feature_correlation.py:134-145` (chỉ lọc Spearman), `:35-46` (fillna) | Bỏ sót 2 cặp chỉ mạnh ở Pearson; không có VIF / within-account / theo `entity_type` | Lọc theo trị tuyệt đối lớn hơn trong hai hệ số; thêm within-account, theo `entity_type`, VIF, Cramér's V (chi tiết §3.2, §6) |

### 11.4. Thứ tự thực hiện & Definition of Done

**Thứ tự (không nên đảo):**

```text
B1. Sửa code P0 (#1..#5)  ->  B2. Sửa code P1 (#6..#10)
B3. Chạy lại extractor trên >= 14-30 ngày (khuyến nghị):
      python scripts/feature_engineering/extract_account_day_matrix.py ^
          --config configs/system_config.yaml --start-day 1 --end-day 30 ^
          --output-dir data/features
B4. pytest
B5. Chạy lại 2 script phân tích:
      python scripts/feature_engineering/plot_feature_correlation.py
      python scripts/feature_engineering/plot_feature_distribution.py
B6. Cập nhật README/docs + đối chiếu lại tài liệu này (ghi rõ "số liệu trước/sau khi sửa")
B7. Sang bước model (IsolationForest / LOF / OneClassSVM)
```

> `scripts/validate_interim.py` **không cần** chạy lại vì tầng `data/interim/` không bị thay đổi — các lỗi ở §3–§6 nằm ở tầng feature, không ở tầng clean.

**Definition of Done (kiểm tra được bằng script):**

| # | Điều kiện nghiệm thu | Cách kiểm |
|---|---|---|
| 1 | Panel key duy nhất theo `(DomainName, UserName, day)` | `df.select([...]).is_duplicated().sum() == 0` |
| 2 | Không còn cột hằng số (không cột nào bị `plot_feature_correlation.py` loại âm thầm) | stdout không còn dòng `[WARN] Bỏ qua các đặc trưng bất biến` |
| 3 | Bộ lõi 18 đặc trưng: **0** cặp \|ρ\| ≥ 0,85 và VIF lớn nhất < 10 | Đọc lại `tables/multicollinear_pairs.csv` + bảng VIF |
| 4 | Không còn `interarrival_dt_mean == 86400` do `fill_null` (có cột cờ `is_single_event` thay thế) | `(df.interarrival_dt_mean >= 86399).mean() == 0` hoặc chỉ xuất hiện khi `n_events == 1` |
| 5 | Tổng ratio nhóm logon-type = 1 trên mọi dòng | `sum(5 ratio + other) == 1` cho 100% dòng |
| 6 | `off_hours_ratio + work_hours_ratio` phủ 24h (hoặc có tài liệu ghi rõ vùng bỏ trống) | `(off + work == 1).all()` hoặc ghi chú trong tài liệu đặc trưng |
| 7 | `entity_type` có 4 nhóm và `Administrator` **không** nằm trong `Service` | `df.filter(UserName == "Administrator").entity_type` không trả về `Service` |
| 8 | `pytest` xanh với ≥ 1 test cho mỗi nhóm đặc trưng | `pytest` |
| 9 | README / config / docs khớp code: **1** danh sách đặc trưng, **1** định nghĩa cửa sổ giờ, **1** đường dẫn output | Đọc chéo 3 file |

**Nếu chưa có thời gian làm hết:** 3 việc nhanh nhất nhưng đổi chất lượng nhiều nhất là **#4** (xoá 2 đặc trưng chết — 5 phút), **#10** (áp bộ rút gọn 18 đặc trưng — 30 phút), **#14** (sửa README cho khớp schema — 15 phút).

---

## 12. Nhật ký sửa bộ đặc trưng (v1.0 → v2.0)

> Thực hiện ngày 2026-09-25. Phạm vi: **chỉ các bất thường của bộ đặc trưng** (§11.1 #1–#5 và §11.2 #6–#10). Các mục hạ tầng/tài liệu (§11.3 #11–#16) **chưa** thực hiện.

### 12.1. File đã thay đổi

| File | Thay đổi |
|---|---|
| [`scripts/feature_engineering/extract_account_day_matrix.py`](../../scripts/feature_engineering/extract_account_day_matrix.py) | Bản 2.0: khoá `(DomainName, UserName)`, luật `entity_type` tường minh, cửa sổ giờ từ config, bỏ 2 đặc trưng chết + thêm đặc trưng từ `FailureReason`, bỏ `fill_null(86400)`, bỏ `burst_logon_count`, rút gọn schema, thêm `--config` + self-check ràng buộc |
| [`configs/system_config.yaml`](../../configs/system_config.yaml) | `ignore_system_accounts` sửa theo giá trị thật (`anonymous` thay `ANONYMOUS LOGON`); thêm `work_hours_start/end`, khối `features.entity_type.*` và `features.filter.*` |
| [`configs/feature_schema.yaml`](../../configs/feature_schema.yaml) | **Mới**: hợp đồng schema 16 đặc trưng + 25 đặc trưng đã loại kèm lý do |
| [`scripts/feature_engineering/plot_feature_correlation.py`](../../scripts/feature_engineering/plot_feature_correlation.py) | Thêm `DomainName`, `total_logons` vào `exclude_cols` (cột hiển thị, không phải đặc trưng) |
| [`scripts/feature_engineering/plot_feature_distribution.py`](../../scripts/feature_engineering/plot_feature_distribution.py) | Cập nhật `DEFAULT_COUNT_FEATURES` theo schema mới |
| `data/features/account_day_matrix.parquet` | Sinh lại: 57.485 dòng × 21 cột (trước: 56.731 × 35) |
| `docs/feature_engineering/tables/*.csv`, `figures/*.png` | Sinh lại theo ma trận mới |

Bản 1.0 được giữ tại `scripts/feature_engineering/extract_account_day_matrix_v1.bak.py` để đối chiếu (đã gắn cảnh báo **"KHÔNG DÙNG ĐỂ CHẠY"** ở đầu file; có thể xoá khi không cần).

### 12.2. Kết quả đo được (Definition of Done §11.4)

| # | Điều kiện | Trước (v1.0) | Sau (v2.0) |
|---|---|---|---|
| 1 | Khoá danh tính duy nhất theo `(DomainName, UserName, day)` | ✗ chỉ `UserName` (trộn domain) | ✅ 0 dòng trùng khoá; 21.454 danh tính (trước: 20.878) |
| 2 | Không còn cột hằng số | ✗ 2 (`wrong_password_count`, `unknown_user_count`) | ✅ 0 |
| 3 | Cặp \|ρ\| ≥ 0,85 trong bộ đặc trưng model | ✗ 13 | ✅ **0** |
| 3b | VIF lớn nhất | ✗ 4041,96 | ✅ **6,90** |
| 4 | Không còn `fill_null(86400)` | ✗ 0,841% dòng bị gán 86.400 | ✅ 806 dòng NULL (1,402%) đúng bằng số dòng 1 sự kiện |
| 5 | Bộc lộ được type hiếm (RDP/runas…) | ✗ 32,39% dòng không phủ hết sự kiện | ✅ `rare_logon_type_count_log` |
| 6 | Cửa sổ giờ phủ 24h | ✗ vùng trống 21,20% / 25,26% sự kiện | ✅ off 18h–7h + work 8h–17h = 24h |
| 7 | `entity_type` tường minh | ✗ `Administrator` (1,28M sự kiện) bị gán `Service` | ✅ 5 nhãn: Machine 31.166 · User 26.167 · Admin 116 (1 tài khoản) · Service 23 (3) · System 13 (4) |
| 8 | `failure_*_share` nhất quán với `failure_ratio` | — | ✅ 0 dòng vi phạm |

### 12.3. Hai phát hiện mới phát sinh trong lúc sửa (đã xử lý)

1. **Không được "đóng ràng buộc tổng" bằng ratio bù trừ.** Phương án `other_logon_type_ratio = 1 − network − interactive` tạo **phụ thuộc tuyến tính tuyệt đối** ⇒ VIF ≈ 5,2e7, ρ(network, other) = 0,9415. Đã thay bằng `rare_logon_type_count_log = log1p(số sự kiện type ∉ {2,3})`, và cuối cùng bỏ luôn `network_ratio` (gần hằng số, ρ = 0,9031 với cột rare).
2. **Tỷ trọng lý do thất bại dạng "share of failures" vẫn bị trùng với cường độ thất bại.** `failure_bad_password_share` có ρ = 0,9220 với `failure_ratio` (mọi lý do thất bại đều là bằng chứng "có thất bại") ⇒ chỉ giữ `failure_locked_out_share` (tín hiệu lockout, không vượt ngưỡng).

### 12.4. Bộ đặc trưng v2.0 (16 đặc trưng cho model)

`log_total_logons` · `failure_ratio` · `failure_locked_out_share` · `off_hours_ratio` · `interarrival_dt_mean` · `delta_t_cv` · `same_second_share` · `is_single_event` · `interactive_ratio` · `rare_logon_type_count_log` · `ntlm_ratio` · `log_distinct_hosts` · `distinct_sources_count` · `missing_source_ratio` · `remote_logon_ratio` · `custom_proc_share`

Khoá/nhãn: `UserName`, `DomainName`, `day`, `entity_type`; cột hiển thị (không dùng cho model/tương quan): `total_logons`.

### 12.5. Việc còn lại (chưa làm trong phạm vi này)

- §11.3 #11–#16: `main.py` thiếu stage `features`/`benchmark`; tên cột trong `system_config.yaml:13-18` sai schema interim; chốt đường dẫn output; README "16 đặc trưng" (danh sách cũ) chưa cập nhật; `tests/` chưa có unit test; script tương quan chưa thêm within-account/VIF/Cramér's V.
- Ma trận mới vẫn chỉ **3 ngày** → cần chạy lại trên ≥ 14–30 ngày (đã có `--config`, chỉ cần đổi `--end-day`).
- Đặc trưng bổ sung §7.3 (novelty/sequence/burst theo cửa sổ) chưa thêm.


