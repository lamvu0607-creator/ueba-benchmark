# Báo cáo: Bộ đặc trưng v4.0 — xây bằng Combinatorial Template + phễu lọc 2 vòng (24 → **39 core**)

> **Dự án:** UEBA Anomaly Detection Benchmark (Windows Event 4624 & 4625 — LANL)
> **Ngày thực hiện:** 2026-10-04 (Tuần 4) · **Trạng thái:** ✅ đã cài đặt, đo trên dữ liệu thật, chốt vào `configs/feature_schema.yaml`
> **Căn cứ:** [`docs/Tong quan de tai ueba.md`](../Tong%20quan%20de%20tai%20ueba.md) §5.2 · [`bao_cao_bo_dac_trung_v3.md`](bao_cao_bo_dac_trung_v3.md) · bảng số đo ở [`docs/feature_engineering/tables/v4_*.csv`](../feature_engineering/tables/)
> **Lệnh tái lập:** `python scripts/feature_engineering/run_feature_funnel_v4.py` (≈ 12 phút, 21 ngày)

---

## 1. Tóm tắt

Báo cáo này mô tả **quy trình** xây bộ đặc trưng, không chỉ danh sách biến. Mỗi đặc trưng mới là một
tổ hợp của 4 trục (*phép đo × đối tượng sự kiện × chiều thực thể × cửa sổ*), đi qua một phễu có ghi
lý do loại ở từng tầng:

| Tầng | Còn lại | Bị loại | Lý do chính |
|:---|---:|---:|:---|
| Bước 0 — sinh toàn bộ tổ hợp 4 trục | **3.136** | — | 14 phép đo × 4 đối tượng × 8 chiều × 7 cửa sổ |
| Bước 0 — ngữ pháp | 471 | 2.665 | tổ hợp vô nghĩa về định nghĩa (vd. "đếm sự kiện trong 7 ngày trước" không phải đặc trưng của ngày t) |
| Vòng 1 — R1 đã có / đã đo và loại | 451 | 20 | trùng 1 core v3.0 hoặc 1 biến đã bị bác bỏ có số đo |
| Vòng 1 — R2 không gắn kịch bản §5.2 | 242 | 209 | không đo đúng cơ chế của kịch bản nào |
| Vòng 1 — R3 khó diễn giải | 186 | 56 | cần > 3 "đơn vị giải thích" |
| Vòng 1 — R4 biến thể dư cùng họ | **78** | 108 | 14d khi đã có 7d, khung 5m/60m khi đã có 15m, … |
| Vòng 2 — cổng E (rò rỉ thời gian) | 78 | 0 | — |
| Vòng 2 — cổng B (có thông tin) | 68 | 10 | NULL > 50% (họ "thất bại so với lịch sử") |
| Vòng 2 — cổng A (không trùng, \|ρ\| < 0,85) | 29 | 39 | chủ yếu trùng `failure_ratio` / `new_*_count_7d` |
| Vòng 2 — cổng C (phân tách kịch bản tiêm) | **26** | 3 | AUC theo đúng chiều < 0,70 |
| Chốt quy mô (~15, VIF ≤ 10) | **15 core + 11 dự bị** | — | xoay vòng theo ưu tiên kịch bản |

**Kết quả:** schema lên **39 core** (24 v3.0 + 15 v4.0): **max \|ρ\| = 0,8488**, **VIF max = 8,52**
(`remote_logon_ratio`, một biến cũ). Cả 7 kịch bản đều có ít nhất một biến mới với AUC ≥ 0,98.

> ⚠️ **Phát hiện cần nói thẳng:** 24 core v3.0 **cũng đã** phân tách được cả 7 kịch bản tiêm (AUC tốt nhất
> từ 0,985 đến 1,000; §8). Bất thường tổng hợp dễ phát hiện, nên cổng C chỉ chứng minh biến mới
> **phản ứng đúng chiều với đúng kịch bản**. Nó **chưa** chứng minh biến mới làm mô hình tốt hơn.
> Câu hỏi đó thuộc khâu ablation đã hẹn làm sau (§10).

---

## 2. Quyết định đầu vào (thực tập sinh chốt ngày 2026-10-04)

| # | Hạng mục | Quyết định |
|---:|:---|:---|
| 1 | Mục tiêu | Thể hiện **quy trình** (Combinatorial Template + detection engineering) và lý do giữ/loại từng biến |
| 2 | Kịch bản | Bao quát cả 6 loại ở §5.2; ưu tiên **A** brute-force/spraying → **B** chiếm đoạt tài khoản (nguồn/giờ lạ) → **C** di chuyển ngang |
| 3 | Thời gian | ≈ 2 ngày trong Tuần 4 |
| 4 | Diễn giải | Mỗi biến đọc được bằng một câu và gắn ≥ 1 kịch bản §5.2 |
| 5 | Quy trình | Vòng 1 lọc bằng lập luận, vòng 2 tự động cài đặt và lọc bằng số liệu theo thứ tự **E → B → A → C** |
| 6 | Quy mô | Khoảng **15** biến mới; khâu kiểm tra đóng góp (ablation) và loại bớt làm sau |
| 7 | Đầu ra | Schema v4 (`feature_schema.yaml` + extractor) và báo cáo phễu này |

---

## 3. Combinatorial Template

Mã nguồn: [`src/features/templates.py`](../../src/features/templates.py).

| Trục | Giá trị |
|:---|:---|
| **Phép đo** (14) | `count`, `distinct`, `share`, `evenness`, `fanout` · `novelty`, `novelty_share`, `jaccard`, `dist_shift` · `deviation`, `delta_mean`, `peer_z` · `active_days`, `recency` |
| **Đối tượng** (4) | `all` (mọi sự kiện), `fail` (4625), `success` (4624), `night` (ngoài giờ 18h–7h) |
| **Chiều** (8) | `event` (không phân loại), `Source`, `LogHost`, `pair` (Source, LogHost), `hour`, `logon_type`, `auth`, `fail_reason` |
| **Cửa sổ** (7) | `1d`; `5m`/`15m`/`60m` (khung cố định trong ngày, lấy khung đông nhất); `7d`/`14d` (ngày lịch **trước** t); `hist` |

**Tên và câu diễn giải được sinh tự động từ tổ hợp.** Vì vậy yêu cầu "đọc được bằng lời thường" được bảo
đảm ngay từ cách sinh, không phụ thuộc vào việc đặt tên tay. Ví dụ:

| Tổ hợp | Tên | Câu sinh tự động |
|:---|:---|:---|
| `jaccard × all × Source × 7d` | `jaccard_source_7d` | độ trùng (Jaccard) giữa tập máy nguồn hôm nay và tập máy nguồn của 7 ngày lịch trước |
| `delta_mean × fail × event × 7d` | `delta_mean_share_fail_7d` | tỷ lệ đăng nhập thất bại hôm nay trừ trung bình của 7 ngày lịch trước |
| `peer_z × night × event × 1d` | `peer_z_share_night` | z bền vững của tỷ lệ sự kiện ngoài giờ so với các tài khoản cùng loại trong cùng ngày |

**Engine tính tự động** ([`src/features/template_engine.py`](../../src/features/template_engine.py)): mọi tổ hợp
được tính bằng **một đường mã chung** chia hai tầng. Tầng `compute_day` chỉ thấy một ngày và dựng hồ sơ
`(tài khoản, giá trị, số sự kiện)`. Tầng `compute_history` chỉ dùng các ngày ≤ t−1, riêng `peer_z` lấy cắt
ngang trong ngày t. Pipeline chính dùng **cùng engine này** cho 15 biến đã chốt, nên số đo trong phễu đúng
là số của biến chạy thật.

Quy ước giá trị (chốt trước khi đo):
- Biến dạng đếm trả về `log1p`.
- Biến dạng tỷ lệ nằm trong [0, 1].
- `peer_z` cắt ở ±10, giống `volume_robust_z_7d`.
- NULL nghĩa là "không xác định" (chưa có lịch sử, hoặc không có sự kiện của đối tượng đó), không bịa thành 0.

---

## 4. Vòng 1 — lọc bằng lập luận (detection engineering)

Bốn luật áp theo thứ tự. Mỗi tổ hợp được ghi **luật đầu tiên mà nó vi phạm** (cột `reason` trong
`v4_template_space.csv`).

| Luật | Câu hỏi | Bị loại | Ví dụ |
|:---|:---|---:|:---|
| **R1** | Đã có trong core v3.0, hoặc đã được đo và bác bỏ? | 20 | `share_success` = 1 − `failure_ratio`; `novelty_pair_7d` ≡ `source_host_pair_novelty` (ρ = 0,9052 ở v3) |
| **R2** | Đo đúng **cơ chế** của một kịch bản §5.2? | 209 | `distinct_night_pair_*` đo *cái gì* xảy ra ban đêm chứ không đo *dịch chuyển* sang ban đêm; `novelty_auth_7d` (pass-the-hash không thuộc §5.2) |
| **R3** | Có đọc được bằng một câu ngắn không? (> 3 đơn vị giải thích thì loại) | 56 | `deviation_distinct_success_pair_14d` cần 5 đơn vị |
| **R4** | Có phải biến thể dư của một ứng viên khác? | 108 | bản 14d khi đã có bản 7d (v3 đo được ρ = 0,9999 giữa hai cửa sổ novelty); khung 5m/60m khi đã giữ 15m; đối tượng `success` (87,36% dòng không có thất bại nên gần trùng `all`) |

**Đơn vị giải thích (R3):** mỗi phép đo có giá cơ bản (đếm = 1; Jaccard/TV/z/peer = 2). Cộng thêm 1 nếu
đối tượng không phải `all`, 1 nếu chiều là `pair`/`auth`/`fail_reason`, và 1 nếu cửa sổ là 14d/hist.

**Bảng ánh xạ kịch bản (R2)** — cách "detection engineering" được mã hoá thành luật:

| Kịch bản §5.2 | Cơ chế | Tổ hợp được gắn |
|:---|:---|:---|
| Brute-force / spraying | cường độ và độ tập trung thất bại theo nguồn/đích | `obj = fail` × {event, Source, LogHost, pair}; `fail_reason` chỉ gắn brute-force ("Account locked out") |
| Hoạt động ngoài giờ | **dịch chuyển** sang ban đêm so với hồ sơ | `obj = night` × event; `hour` × phép đo so với lịch sử |
| Bùng nổ máy trạm mới | nhiều nguồn và nguồn mới | `Source` × (mọi phép đo trừ evenness/fanout) |
| Di chuyển ngang | chạm nhiều máy đích mới | `LogHost`, `pair` |
| Ngủ đông thức dậy | so với lịch sử **của chính** tài khoản | `event` × {delta_mean, deviation, active_days, recency} |
| Đổi loại logon | thay đổi cơ cấu loại logon | `logon_type` × phép đo so với lịch sử |

---

## 5. Bộ tiêm bất thường (bằng chứng cho cổng C)

Mã nguồn: [`src/evaluation/injection.py`](../../src/evaluation/injection.py). Bộ tiêm can thiệp vào **luồng sự
kiện**, rồi chạy đúng đường tính của pipeline. Nhờ vậy cả 24 biến cũ lẫn 78 ứng viên đều được thử trên
cùng một bằng chứng. Extractor được tách thành `load_day_events` + `features_from_events` để làm việc này.

| Kịch bản | Cách mô phỏng (ngày tiêm = 21) | Nạn nhân |
|:---|:---|---:|
| `brute_force` | 100–300 lần 4625 trong 20 phút, từ **một** nguồn mới, vào máy đích quen | 50 |
| `password_spraying` | **một** nguồn chung; mỗi nạn nhân 3–8 lần 4625 trong 10:00–10:30 | 50 |
| `off_hours` | nén cả ngày hoạt động vào 01:00–06:00 (giữ thứ tự) | 49 |
| `new_workstation_burst` | 10–30 lần 4624 thành công, mỗi lần từ một máy nguồn chưa từng thấy | 50 |
| `dormant_wakeup` | xoá hoạt động ngày 4–20, ngày 21 khối lượng ×5 | 50 |
| `logon_type_switch` | mọi sự kiện ngày 21 đổi sang LogonType 5 (tài khoản chưa từng dùng loại hiếm) | 50 |
| `lateral_fanout` | 15–40 lần 4624 type 3 tới máy đích thật chưa chạm, trong 14:00–15:00 | 50 |

Nạn nhân đều là tài khoản `User`, có ≥ 4/7 ngày hoạt động trước ngày tiêm, được chọn tất định
(seed 42) và không trùng nhau giữa các nhóm.

---

## 6. Vòng 2 — 4 cổng số liệu (21 ngày LANL, 353.730 dòng; cổng B/A đo trên ngày 8–21 = 232.496 dòng)

| Cổng | Định nghĩa | Ngưỡng | Kết quả |
|:---|:---|:---|:---|
| **E** rò rỉ | Tính lại tầng lịch sử khi **bỏ ngày 21**: giá trị ngày 1–20 phải y hệt | Δ = 0 | 78/78 qua (Δmax = 0) |
| **B** thông tin | Trên ngày sau warm-up | NULL ≤ 50%, giá trị phổ biến nhất < 99% | 10 loại: toàn bộ họ "thất bại so với lịch sử thất bại" (`jaccard_fail_*`, `dist_shift_fail_*`, `novelty_share_fail_*`, `evenness_fail_*`). Lý do: chỉ 12,04% dòng có thất bại nên NULL 87–91% |
| **A** trùng lặp | max(\|Spearman\|, \|Pearson\|) với 24 core **và** với ứng viên đã nhận trước | < 0,85 | 39 loại (bảng dưới) |
| **C** phân tách | AUC trên ngày tiêm, **theo chiều bất thường đã khai báo**, ít nhất một kịch bản mục tiêu | ≥ 0,70 | 3 loại |

**Thứ tự xét ở cổng A:** ưu tiên kịch bản (A → B → C) → số đơn vị giải thích → tên. Khi hai ứng viên
trùng trục, biến thuộc kịch bản ưu tiên cao hơn và dễ đọc hơn được giữ.

**Ba cơ chế chiếm phần lớn 39 lần loại ở cổng A:**

1. **Zero-inflation của thất bại.** Đây cũng là cơ chế đã loại `failure_count` ở v3. 13 biến "đếm thất
   bại" (`count_fail_15m`, `distinct_fail_*`, `fanout_fail_source`, `peer_z_*_fail*`) có ρ = 0,945–0,997
   với `failure_ratio`, vì chúng chung chỉ báo "ngày có thất bại hay không". Biến thất bại sống sót đều là
   biến **so với lịch sử của chính tài khoản** (`delta_mean_*`, `novelty_fail_host_7d`).
2. **Cùng trục với novelty v3.** `novelty_share_source_7d` (ρ = 0,998 với `new_source_count_7d`),
   `novelty_share_host_7d` (0,987), cùng các bản `success`. Kết quả này **xác nhận lại** luật R4(c) bằng số đo.
3. **Hai cách chuẩn hoá của cùng một đại lượng.** `novelty_share_logontype_7d` và
   `novelty_logontype_7d` có ρ = 0,9993.

**Ba biến bị loại ở cổng C** đều cho thấy **chiều khai báo sai so với thực tế**:
- `delta_mean_distinct_hour_7d` (AUC = 0,095): khi nén hoạt động vào 5 giờ đêm, số khung giờ *giảm* chứ không tăng.
- `count_night` (0,243): nạn nhân là tài khoản người có khối lượng nhỏ, trong khi quần thể có máy chủ chạy ban đêm hàng triệu sự kiện.
- `delta_mean_distinct_logontype_7d` (0,635).

Cổng C được thiết kế để bắt đúng loại sai này.

---

## 7. Bước chốt quy mô: 26 biến qua cổng → 15 core + 11 dự bị

Phễu 4 cổng cho ra 26 biến. Con số này vượt mục tiêu khoảng 15, và nếu nạp cả 26 thì
`active_days_7d` đẩy VIF lên 12,9, vượt ngưỡng 10 mà repo đang dùng
(`check_multicollinearity.py`, báo cáo v3 §7). Bước chốt chạy tất định như sau:

> Xoay vòng các kịch bản theo thứ tự ưu tiên. Mỗi lượt, mỗi kịch bản lấy biến có AUC cao nhất trên
> kịch bản đó; hoà thì lấy biến có max |ρ| nhỏ hơn. Biến chỉ được nhận nếu VIF lớn nhất của bộ
> 24 + đã chọn + biến đó vẫn ≤ 10. Dừng khi đủ 15.

Vì nhóm A (brute/spray) đứng đầu vòng, cả 4 biến thất bại qua cổng đều vào core, kể cả biến yếu nhất
`novelty_fail_failreason_7d` (AUC 0,728). Đây là hệ quả trực tiếp của thứ tự ưu tiên đã chốt.

### 7.1. 15 biến vào core (nhóm 11 trong schema)

| # | Tên | Đọc là | Kịch bản (AUC tốt nhất) | AUC | max \|ρ\| (với) | NULL |
|---:|:---|:---|:---|---:|:---|---:|
| 1 | `delta_mean_share_fail_7d` | tỷ lệ đăng nhập thất bại hôm nay trừ trung bình 7 ngày lịch trước | brute_force | 0,999 | 0,821 (`delta_mean_distinct_fail_host_7d`) | 2,2% |
| 2 | `delta_mean_distinct_fail_host_7d` | log số máy đích bị đăng nhập thất bại hôm nay trừ trung bình 7 ngày trước | password_spraying | 0,990 | 0,660 (`failure_ratio`) | 0% |
| 3 | `novelty_fail_host_7d` | số máy đích bị đăng nhập thất bại hôm nay mà 7 ngày trước chưa từng có | password_spraying | 0,987 | 0,635 | 0% |
| 4 | `novelty_fail_failreason_7d` | số lý do thất bại hôm nay chưa xuất hiện trong 7 ngày trước | brute_force | 0,728 | 0,837 (`novelty_fail_host_7d`) | 0% |
| 5 | `delta_mean_distinct_source_7d` | log số máy nguồn khác nhau hôm nay trừ trung bình 7 ngày trước | new_workstation_burst | 1,000 | 0,445 | 0% |
| 6 | `jaccard_source_7d` | độ trùng tập máy nguồn hôm nay với tập máy nguồn 7 ngày trước | new_workstation_burst | 0,995 | 0,712 | 3,6% |
| 7 | `peer_z_distinct_source` | số máy nguồn khác nhau so với các tài khoản cùng loại trong ngày (z bền vững) | new_workstation_burst | 1,000 | 0,831 (`distinct_sources_count`) | 0% |
| 8 | `dist_shift_hour_7d` | phân bố hoạt động theo giờ hôm nay khác 7 ngày trước bao nhiêu (total variation) | off_hours | 0,987 | 0,571 | 2,2% |
| 9 | `jaccard_hour_7d` | độ trùng tập khung giờ hoạt động hôm nay với 7 ngày trước | off_hours | 0,984 | 0,692 | 2,2% |
| 10 | `peer_z_share_night` | tỷ lệ ngoài giờ so với các tài khoản cùng loại trong ngày (z bền vững) | off_hours | 0,980 | 0,746 (`off_hours_ratio`) | 0% |
| 11 | `delta_mean_distinct_pair_7d` | log số cặp (nguồn, đích) hôm nay trừ trung bình 7 ngày trước | lateral_fanout | 1,000 | 0,780 | 0% |
| 12 | `novelty_share_pair_7d` | tỷ lệ sự kiện hôm nay đi qua cặp (nguồn, đích) chưa thấy trong 7 ngày | lateral_fanout | 0,990 | 0,756 (`new_host_count_7d`) | 1,5% |
| 13 | `delta_mean_count_7d` | log khối lượng hôm nay trừ trung bình 7 ngày trước (ngày trống = 0) | dormant_wakeup | 0,999 | 0,796 | 0% |
| 14 | `dist_shift_logontype_7d` | cơ cấu loại logon hôm nay khác 7 ngày trước bao nhiêu (total variation) | logon_type_switch | 1,000 | 0,698 | 2,2% |
| 15 | `jaccard_logontype_7d` | độ trùng tập loại logon hôm nay với 7 ngày trước | logon_type_switch | 1,000 | 0,669 | 2,2% |

### 7.2. 11 biến dự bị (mục `reserve:` trong schema, extractor chưa tính)

Đây là các biến đã qua đủ 4 cổng. Chúng là **ứng viên đầu tiên** cho khâu ablation:
`novelty_hour_7d`, `count_night_15m`, `delta_mean_share_night_7d`, `dist_shift_source_7d`,
`dist_shift_pair_7d`, `jaccard_pair_7d`, `peer_z_distinct_pair`, `dist_shift_host_7d`, `jaccard_host_7d`,
`active_days_7d` (bị chặn vì VIF lên 11,72), `novelty_logontype_7d`.

---

## 8. Bao phủ kịch bản: core v3.0 so với 15 biến v4.0

AUC tốt nhất trên mỗi kịch bản (chi tiết theo từng biến ở `v4_scenario_auc.csv`; với core v3 tính
theo hai chiều vì chưa có chiều khai báo):

| Kịch bản | 24 core v3.0 (biến tốt nhất) | 15 biến v4.0 (biến tốt nhất) | Số biến v3 / v4 có AUC ≥ 0,70 |
|:---|:---|:---|:---:|
| brute_force | 0,993 (`new_source_count_7d`) | 0,999 (`delta_mean_share_fail_7d`) | 16 / 11 |
| password_spraying | 0,992 (`new_source_count_7d`) | 0,990 (`delta_mean_distinct_fail_host_7d`) | 8 / 10 |
| off_hours | 0,985 (`off_hours_ratio`) | 0,987 (`dist_shift_hour_7d`) | 13 / 6 |
| new_workstation_burst | 1,000 (`new_source_count_7d`) | 1,000 (`delta_mean_distinct_source_7d`) | 4 / 7 |
| dormant_wakeup | 1,000 (`days_since_last_activity`) | 0,999 (`delta_mean_count_7d`) | 8 / 4 |
| logon_type_switch | 0,989 (`rare_logon_type_count_log`) | 1,000 (`dist_shift_logontype_7d`) | **1** / 2 |
| lateral_fanout | 1,000 (`new_host_count_7d`) | 1,000 (`delta_mean_distinct_pair_7d`) | 7 / 8 |

Cách đọc bảng:
- **Biến mới không mở ra kịch bản nào mà bộ cũ bỏ sót.** Với dữ liệu tiêm, bộ cũ đã chạm trần.
- **Giá trị thật của bộ mới nằm ở chỗ khác:**
  1. **Dự phòng.** `logon_type_switch` trước chỉ dựa vào đúng một biến (`rare_logon_type_count_log`).
  2. **Đúng cơ chế.** Ở bộ cũ, brute-force và spraying được "bắt" chủ yếu nhờ `new_source_count_7d`. Đó là vì
     nguồn tiêm có tên mới, tức là một artifact của cách tiêm, chứ không phải tín hiệu thất bại. Bộ mới bắt
     brute-force/spraying bằng đúng tín hiệu thất bại so với lịch sử.
  3. **Giải thích cảnh báo.** Mỗi biến tự nói "lệch so với 7 ngày trước của chính tài khoản" hay "lệch so với
     các tài khoản cùng loại".
- **Những điều trên mới chỉ là lập luận.** Biến mới có giúp mô hình (IForest/LOF/OCSVM) xếp hạng tốt hơn
  không thì phải đo bằng ablation.

---

## 9. Hạn chế

1. **Bất thường tổng hợp là cận trên lạc quan** (đề cương §5.2). Đặc biệt, nguồn tiêm brute/spray có tên
   mới nên mọi biến novelty Source đều phản ứng. Thực tế kẻ tấn công có thể dùng máy đã quen.
2. **Ngưỡng cổng là lựa chọn đã ghi rõ**, không phải chân lý:
   - |ρ| < 0,85 và VIF ≤ 10 kế thừa từ v3.
   - NULL ≤ 50%, giá trị phổ biến nhất < 99% và AUC ≥ 0,70 được chốt cho vòng này.
   - Chỉ cần đổi hằng số ở đầu `run_feature_funnel_v4.py` rồi chạy lại là có kết quả mới.
3. **Đo trên 21 ngày** (sau warm-up còn 14 ngày, 232.496 dòng). Ma trận 60 ngày **chưa** được dựng lại
   theo schema v4 (lần chạy đầu bị dừng ở ngày 17 vì máy thiếu RAM), nên chưa đo lại tương quan trên 60 ngày.
4. **`peer_z` dùng nhóm = `entity_type`.** Đây là định nghĩa đơn giản nhất. Báo cáo v3 đã cảnh báo rằng định
   nghĩa nhóm đồng đẳng ảnh hưởng tới kết quả.
5. **Thứ tự cổng E → B → A → C** nghĩa là khi hai ứng viên trùng trục, cổng A phân xử bằng ưu tiên kịch bản
   chứ không bằng AUC. Có thể có biến AUC cao hơn bị loại vì "đến sau" (vd. `novelty_fail_source_7d`, AUC
   0,998, có ρ = 0,898 với `novelty_fail_host_7d`).
6. **Lateral fan-out** chọn máy đích "chưa chạm hôm nay" chứ không kiểm cả 20 ngày trước. Xác suất máy đó đã
   được chạm trước đó là rất nhỏ nhưng không bằng 0.

---

## 10. Việc tiếp theo (khâu "kiểm tra và loại bớt" đã hẹn)

1. Chạy lại benchmark với 39 core: `python main.py --stage benchmark`.
2. **Ablation theo nhóm**: bỏ lần lượt từng kịch bản của nhóm 11, và thử hoán đổi với các biến dự bị. Đo bằng
   bộ chỉ số có nhãn (`average_precision`, `recall_at_budget`) trên nhãn tiêm, và nếu có thì trên nhãn
   `redteam.txt`.
3. Biến nào không làm giảm chỉ số khi bị bỏ thì chuyển sang `reserve`. Biến dự bị nào làm tăng chỉ số thì đưa
   vào core.

---

## 11. Tái lập và bản đồ mã

```bash
# Phễu đầy đủ (dựng 2 ma trận sạch/tiêm 21 ngày + 4 cổng + bước chốt) ≈ 12 phút
python scripts/feature_engineering/run_feature_funnel_v4.py
# Chỉ chạy lại cổng B/A/C và bước chốt trên ma trận đã lưu (vài phút) — dùng khi đổi ngưỡng
python scripts/feature_engineering/run_feature_funnel_v4.py --from-cache
# Dựng lại ma trận 60 ngày theo schema v4 (39 core)
python main.py --stage features
# Kiểm thử
python -m pytest -q
```

| File | Vai trò |
|:---|:---|
| `src/features/templates.py` | 4 trục, ngữ pháp (bước 0), luật R1–R4 (vòng 1), sinh tên và câu diễn giải |
| `src/features/template_engine.py` | engine tính mọi tổ hợp; dùng chung cho phễu và pipeline |
| `src/evaluation/injection.py` | tiêm 7 kịch bản ở mức sự kiện |
| `src/features/extractor.py` | tách `load_day_events` / `features_from_events`; `TEMPLATE_V4_FEATURES` (15 biến) |
| `scripts/feature_engineering/run_feature_funnel_v4.py` | phễu vòng 2 và bước chốt |
| `configs/feature_schema.yaml` | nhóm 11 (15 core có khoá `template:`), `reserve:` (11), `removed:` (+52 kèm cổng và số đo) |
| `tests/test_template_features.py` | template, engine (giá trị và tính nhân quả), bộ tiêm |
| `docs/feature_engineering/tables/v4_*.csv` | toàn bộ số đo của báo cáo này |
