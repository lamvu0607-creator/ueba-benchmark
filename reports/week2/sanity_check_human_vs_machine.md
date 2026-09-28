# Kiểm định tính hợp lý nghiệp vụ (Sanity Check: Human vs. Machine) — bằng BOXPLOT

Sinh bởi `scripts/feature_engineering/plot_sanity_check_human_vs_machine.py` — mọi số liệu dưới đây được script tính lại từ dữ liệu mỗi lần chạy (không viết tay).

| Tham số | Giá trị |
|---|---|
| Dữ liệu đầu vào | `D:\Github Repo\ueba-benchmark\data\features\raw\feature_matrix_raw.parquet` |
| Số bản ghi (khoá User x Day) | 1,055,283 |
| Nhóm Người dùng (Human) | 470,284 bản ghi (44.56%) |
| Nhóm Tài khoản Máy (Machine) | 584,999 bản ghi (55.44%) |
| Luật gán nhóm | `entity_type == 'Machine'` (nhãn do extractor sinh theo configs/system_config.yaml); nhóm còn lại = User / Admin / Service / System (người + tài khoản hệ thống) |
| Số đặc trưng kiểm định bằng boxplot | 16 (lưới 4x4) |
| Thang hiển thị log1p | delta_t_cv, rare_logon_type_count, distinct_sources_count, distinct_hosts, total_logons, interarrival_dt_mean |
| Ngưỡng hiệu ứng Cliff's delta | nhỏ ≥ 0.147 · mạnh ≥ 0.330 · rất mạnh ≥ 0.474 (Romano et al.) |

## 1. Kết luận nhanh

1. **Bộ đặc trưng TỰ TÁCH được Human vs Machine — ĐẠT**: **13/16** đặc trưng có |Cliff's delta| ≥ 0.147 (trong đó 4 mức *large*, 1 mức *medium*, 8 mức *small*). Như vậy chỉ bằng **boxplot** (không dùng model, không dùng nhãn) đã nhìn thấy 2 quần thể khác bản chất vật lý.

2. **Ba đặc trưng tách mạnh nhất** (đọc chỉ số ở lưới §4): `delta_t_cv` |δ| = 0.676 (P50 Người 1.334 vs Máy 0.938, Người > Máy); `rare_logon_type_count` |δ| = 0.640 (P50 Người 1.946 vs Máy 0.000, Người > Máy); `same_second_share` |δ| = 0.598 (P50 Người 0.632 vs Máy 0.419, Người > Máy).

3. **Đặc trưng CHƯA tách được (3/16)** — không phải lỗi dữ liệu, mà do bản chất phân phối; phải đọc kèm %NULL/%ZERO chứ không kết luận qua hình dạng hộp:

   - `interarrival_dt_mean` (|δ| = 0.057): trung vị 2 nhóm gần như trùng nhau (5.290 vs 5.372)
   - `is_single_event` (|δ| = 0.029): trung vị 2 nhóm gần như trùng nhau (0.000 vs 0.000); zero-inflated (0: Người 96.88% / Máy 99.75%)
   - `failure_locked_out_share` (|δ| = 0.012): trung vị 2 nhóm gần như trùng nhau (0.000 vs 0.000); NULL-heavy (NULL: Người 73.14% / Máy 98.79%); zero-inflated (0: Người 98.80% / Máy 99.99%)

   → Nhóm này **không dùng để phân biệt loại thực thể**, nhưng KHÔNG nên loại bỏ: chúng vẫn hữu ích cho bài toán *so sánh một khoá với chính nó theo thời gian* (baseline per-entity), nơi Human–Machine không còn là biến phân biệt.

4. **Đối chiếu kỳ vọng nghiệp vụ trên 3 trục hành vi**: 3/3 trục **KHỚP KỲ VỌNG** (chi tiết ở §5).

5. **Hệ quả cho pipeline**: nhãn `entity_type` chỉ được dùng để **kiểm định**, **không** đưa vào vector đặc trưng — nếu đưa vào, model sẽ học đường tắt phân loại thực thể thay vì học bất thường hành vi. Bộ đặc trưng hiện tại đã đủ nhạy để tự phân biệt.

## 2. Phương pháp kiểm định (boxplot là công cụ chính)

| Lớp bằng chứng | Cách làm | Cách đọc / ngưỡng |
|---|---|---|
| ① Boxplot 3 trục hành vi | `sns.boxplot` cho từng nhóm trên cùng một trục: hộp = Q1–Q3, vạch đậm giữa hộp = trung vị (P50), râu = 1,5 × IQR, chấm vàng = trung bình; điểm ngoại lai ẩn (N ≈ 1 triệu nên vẽ ra chỉ thành khối đen) | Hai hộp tách rời theo trục tung ⇒ đặc trưng tự phân biệt được bản chất thực thể |
| ② Lưới boxplot 4×4 | Vẽ TOÀN BỘ 16 đặc trưng, mỗi ô 1 cặp hộp Người–Máy; 6 cột lệch mạnh (|skew| > 3) hiển thị trên thang log1p; sắp giảm dần theo |Cliff's delta| | Màu tiêu đề ô = mức tách (large/medium/small/xám); ô hộp bẹp phải đọc thêm %ZERO / %NULL in ở góc phải |
| ③ Định lượng mức tách | Cliff's delta = 2U / (n_người × n_máy) − 1, với U là thống kê Mann-Whitney (kiểm định 2 phía, có hiệu chỉnh đồng hạng) | |δ| ≥ 0.147 small · ≥ 0.330 medium · ≥ 0.474 large (Romano et al. 2006); δ > 0 ⇒ nhóm NGƯỜI lớn hơn, δ < 0 ⇒ nhóm MÁY lớn hơn |

> **Vì sao boxplot + Cliff's delta?** Cả hai đều **phi tham số**: không giả định phân phối chuẩn, chịu được đuôi dài, giá trị chặn [0,1] và lượng điểm 0 lớn của bộ đặc trưng này. p-value Mann-Whitney chỉ để tham chiếu: với N ≈ 1 triệu thì p gần như luôn ≈ 0, nên **không** dùng p để kết luận mức tách mà dùng |δ|.

## 3. Hình ảnh bằng chứng

| Hình | Cần nhìn gì |
|---|---|
| `docs/feature_engineering/figures/sanity_check_human_vs_machine.png` | 3 boxplot theo trục hành vi (thời gian – phương thức – không gian). Mỗi ô có hộp kỳ vọng nghiệp vụ kèm P50, IQR, Cliff's δ, p-value và kết quả đối chiếu kỳ vọng ⇒ vừa thấy hình dạng, vừa thấy con số. |
| `docs/feature_engineering/figures/sanity_check_human_vs_machine_boxplot_grid.png` | Lưới 4×4 cho TOÀN BỘ 16 đặc trưng, sắp giảm dần theo |δ|. Trả lời trực tiếp câu hỏi *“bộ đặc trưng có tự tách được Human vs Machine mà chưa cần model không?”* |

## 4. Bảng 1 — 16 đặc trưng, sắp giảm dần theo mức tự tách (Hình: lưới boxplot 4×4)

> P50/IQR tính trên thang hiển thị ghi ở cột *Thang* (log1p cho 6 cột lệch mạnh).
> `%NULL` và `%ZERO` là tỷ lệ **trên toàn bộ dòng của nhóm**, không phải trên dòng khả dụng.

| # | Đặc trưng | Nhóm | Thang | P50 Người | P50 Máy | IQR Người | IQR Máy | NULL% N/M | ZERO% N/M | Cliff's δ | Mức tách |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `delta_t_cv` | rhythm | log1p | 1.334 | 0.938 | 0.267 | 0.124 | 5.1 / 0.3 | 0.2 / 0.0 | +0.676 | TÁCH RẤT MẠNH (large) |
| 2 | `rare_logon_type_count` | logon_type | log1p | 1.946 | 0.000 | 2.773 | 0.000 | 0.0 / 0.0 | 35.8 / 99.5 | +0.640 | TÁCH RẤT MẠNH (large) |
| 3 | `same_second_share` | rhythm | thô | 0.632 | 0.419 | 0.200 | 0.097 | 0.0 / 0.0 | 10.5 / 0.9 | +0.598 | TÁCH RẤT MẠNH (large) |
| 4 | `ntlm_ratio` | logon_type | thô | 0.000 | 0.123 | 0.025 | 0.083 | 0.0 / 0.0 | 58.8 / 10.9 | -0.567 | TÁCH RẤT MẠNH (large) |
| 5 | `distinct_sources_count` | context | log1p | 0.693 | 0.693 | 0.406 | 0.000 | 0.0 / 0.0 | 1.6 / 1.4 | +0.453 | TÁCH MẠNH (medium) |
| 6 | `off_hours_ratio` | time | thô | 0.471 | 0.568 | 0.349 | 0.085 | 0.0 / 0.0 | 13.2 / 4.2 | -0.314 | TÁCH RÕ (small) |
| 7 | `distinct_hosts` | context | log1p | 1.609 | 1.386 | 0.847 | 0.511 | 0.0 / 0.0 | 0.0 / 0.0 | +0.310 | TÁCH RÕ (small) |
| 8 | `missing_source_ratio` | context | thô | 0.030 | 0.051 | 0.081 | 0.037 | 0.0 / 0.0 | 40.2 / 3.7 | -0.296 | TÁCH RÕ (small) |
| 9 | `interactive_ratio` | logon_type | thô | 0.000 | 0.000 | 0.002 | 0.000 | 0.0 / 0.0 | 70.9 / 100.0 | +0.291 | TÁCH RÕ (small) |
| 10 | `failure_ratio` | failure | thô | 0.000 | 0.000 | 0.001 | 0.000 | 0.0 / 0.0 | 73.1 / 98.8 | +0.256 | TÁCH RÕ (small) |
| 11 | `custom_proc_share` | context | thô | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 / 0.0 | 79.2 / 99.3 | +0.201 | TÁCH RÕ (small) |
| 12 | `total_logons` | volume *(cột hiển thị)* | log1p | 5.656 | 5.924 | 1.507 | 0.384 | 0.0 / 0.0 | 0.0 / 0.0 | -0.177 | TÁCH RÕ (small) |
| 13 | `remote_logon_ratio` | context | thô | 0.915 | 0.948 | 0.126 | 0.037 | 0.0 / 0.0 | 9.0 / 1.6 | -0.152 | TÁCH RÕ (small) |
| 14 | `interarrival_dt_mean` | rhythm | log1p | 5.290 | 5.372 | 1.056 | 0.331 | 3.1 / 0.2 | 0.8 / 0.0 | -0.057 | CHƯA TÁCH ĐƯỢC (negligible) |
| 15 | `is_single_event` | rhythm | thô | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 / 0.0 | 96.9 / 99.8 | +0.029 | CHƯA TÁCH ĐƯỢC (negligible) |
| 16 | `failure_locked_out_share` | failure | thô | 0.000 | 0.000 | 0.000 | 0.000 | 73.1 / 98.8 | 98.8 / 100.0 | +0.012 | CHƯA TÁCH ĐƯỢC (negligible) |

## 5. Bảng 2 — 3 trục hành vi vật lý (Hình: `sanity_check_human_vs_machine.png`)

| Trục | Đặc trưng | Thang | P50 Người | P50 Máy | IQR Người | IQR Máy | Cliff's δ | Mức tách | Đối chiếu kỳ vọng |
|---|---|---|---|---|---|---|---|---|---|
| Trục 1 — THỜI GIAN | `off_hours_ratio` | thô | 0.471 | 0.568 | 0.349 | 0.085 | -0.314 | TÁCH RÕ (small) | KHỚP KỲ VỌNG |
| Trục 2 — PHƯƠNG THỨC | `interactive_ratio` | thô | 0.000 | 0.000 | 0.002 | 0.000 | +0.291 | TÁCH RÕ (small) | KHỚP KỲ VỌNG |
| Trục 3 — KHÔNG GIAN | `distinct_hosts` | log1p | 1.609 | 1.386 | 0.847 | 0.511 | +0.310 | TÁCH RÕ (small) | KHỚP KỲ VỌNG |

### 5.1. Kỳ vọng nghiệp vụ vs thực nghiệm

| Trục | Kỳ vọng cho nhóm Người | Kỳ vọng cho nhóm Máy | Kết quả |
|---|---|---|---|
| **Trục 1 — THỜI GIAN** | Người: THẤP — chủ yếu làm việc trong giờ hành chính 8h-18h | Máy: CAO & ổn định — tác vụ chạy ngầm 24/7 | KHỚP KỲ VỌNG |
| **Trục 2 — PHƯƠNG THỨC** | Người: > 0 — người ngồi tại bàn phím, có phiên đăng nhập trực tiếp | Máy: ≈ 0 — xác thực qua mạng (Network/Service), không có phiên bàn phím | KHỚP KỲ VỌNG |
| **Trục 3 — KHÔNG GIAN** | Người: PHÂN TÁN RỘNG — người đi lại nhiều máy (roaming) nên hộp cao, đuôi dài | Máy: RẤT HẸP — tài khoản máy gắn chặt với máy của nó, quanh 1-3 máy | KHỚP KỲ VỌNG |

> Quy tắc đối chiếu được script tự kiểm tra: trục thời gian — giá trị nhóm Máy phải CAO hơn; trục phương thức — giá trị nhóm Người phải CAO hơn; trục không gian — IQR nhóm Người phải RỘNG hơn (người đi lại nhiều máy, tài khoản máy gắn chặt với máy của nó).
>
> Với đặc trưng zero-inflated (trung vị 2 nhóm đều = 0, ví dụ `interactive_ratio` có ~87% số 0 ở nhóm Người), script tự chuyển sang đối chiếu theo **trung bình** rồi mới kết luận Khớp/Lệch — nếu chỉ so trung vị sẽ kết luận sai là “lệch kỳ vọng”.

## 6. Giới hạn & lưu ý khi trích dẫn

1. **Nhãn nhóm chỉ là heuristic đặt tên tài khoản**: `entity_type == 'Machine'` (nhãn do extractor sinh theo configs/system_config.yaml); nhóm còn lại = User / Admin / Service / System (người + tài khoản hệ thống). Tài khoản dịch vụ/hệ thống (Service, System) nằm trong nhóm “không phải Machine”, nên phần nào làm dịu mức tách thật.
2. **Boxplot không thấy tính đa đỉnh**: hộp chỉ tổng hợp Q1–Q3 và trung vị; hai phân phối rất khác nhau vẫn có thể cho 2 hộp giống nhau ⇒ phải đọc kèm histogram (`distribution_*.png`) hoặc ECDF khi cần kết luận mạnh.
3. **Điểm ngoại lai đã bị ẩn trên hình** (N ≈ 1 triệu, vẽ ra chỉ thành khối đen) nên đuôi dài không hiện — điều này KHÔNG ảnh hưởng Cliff's delta (tính trên toàn bộ dữ liệu).
4. **Đặc trưng NULL-heavy** (`failure_locked_out_share`): hộp và δ chỉ tính trên dòng KHẢ DỤNG nên n của nhóm Máy nhỏ hơn hẳn; phải đọc kèm cột NULL% ở Bảng 1 chứ không kết luận “không tách được” chỉ từ hình dạng hộp.
5. **p-value ≈ 0 luôn đúng khi N ≈ 1 triệu** ⇒ không dùng p để xếp hạng mức tách, chỉ dùng |δ|.
6. **Tách được Human/Machine KHÔNG đồng nghĩa phát hiện xâm nhập**: đây là kiểm định độ nhạy của bộ đặc trưng với bản chất vật lý của thực thể, không phải nhãn bất thường. Nhãn tấn công thật vẫn lấy từ ground truth (redteam) ở bước benchmark.
7. **Thang log1p chỉ để hiển thị**: 6 cột lệch mạnh được vẽ trên log1p để hộp khỏi bị bóp dẹt; Cliff's delta và các phân vị trong bảng vẫn tính trên giá trị gốc của ma trận — không thay đổi dữ liệu đưa vào model.

## 7. Sản phẩm sinh ra

| Loại | Đường dẫn |
|---|---|
| Hình A — 3 trục hành vi (boxplot ghép) | `D:\Github Repo\ueba-benchmark\docs\feature_engineering\figures\sanity_check_human_vs_machine.png` |
| Hình A — bản sao artifacts | `D:\Github Repo\ueba-benchmark\artifacts\sanity_check_human_vs_machine.png` |
| Hình A1 — Trục 1 (Thời gian) | `D:\Github Repo\ueba-benchmark\docs\feature_engineering\figures\sanity_check_axis_temporal_off_hours_ratio.png` |
| Hình A2 — Trục 2 (Phương thức) | `D:\Github Repo\ueba-benchmark\docs\feature_engineering\figures\sanity_check_axis_logon_mechanism_interactive_ratio.png` |
| Hình A3 — Trục 3 (Không gian) | `D:\Github Repo\ueba-benchmark\docs\feature_engineering\figures\sanity_check_axis_spatial_fanout_distinct_hosts.png` |
| Hình B — lưới boxplot 16 đặc trưng | `D:\Github Repo\ueba-benchmark\docs\feature_engineering\figures\sanity_check_human_vs_machine_boxplot_grid.png` |
| Hình B — bản sao artifacts | `D:\Github Repo\ueba-benchmark\artifacts\sanity_check_human_vs_machine_boxplot_grid.png` |
| Bảng 1 — 16 đặc trưng + hiệu ứng | `D:\Github Repo\ueba-benchmark\docs\feature_engineering\tables\sanity_check_human_vs_machine.csv` |
| Bảng 2 — 3 trục hành vi | `D:\Github Repo\ueba-benchmark\docs\feature_engineering\tables\sanity_check_human_vs_machine_axes.csv` |
| Báo cáo Markdown | `D:\Github Repo\ueba-benchmark\reports\week2\sanity_check_human_vs_machine.md` |
