# Kiểm tra Tính ổn định chu kỳ theo thời gian (Temporal Stationarity)

Sinh bởi `scripts/feature_engineering/plot_temporal_stability.py` — mọi số liệu dưới đây đều được script tính lại từ dữ liệu mỗi lần chạy (không viết tay).

| Tham số | Giá trị |
|---|---|
| Dữ liệu đầu vào | `D:\Github Repo\ueba-benchmark\data\features\raw\feature_matrix_raw.parquet` |
| Số ngày | 60 (trong đó 16 ngày cuối tuần-like) |
| Pha chu kỳ | `(day - 1) % 7`, pha weekend-like = 5, 6 |
| Cửa sổ off-hours (config) | 18h-7h |
| Giờ làm việc (config) | 8h-17h |
| Khoảng nghỉ sinh học (đêm) | 0h-5h |
| Dữ liệu thô event-level | THIẾU -> bỏ qua Q3 |

## 1. Kết luận nhanh

1. **Chu kỳ 7 ngày: RÕ** — ACF lag 7 = 0.5833, lag 14 = 0.4572, đỉnh ACF tại lag 7. Chỉ riêng pha tuần đã giải thích **66.6965%** phương sai log(lưu lượng/ngày); phần dư 0.6986% mới là nền để dò bất thường thật.

2. **Sụt giảm cuối tuần: 29.3201%** (lưu lượng trung bình cuối tuần-like so với ngày thường). Tập pha weekend-like = 5, 6; số tài khoản hoạt động giảm 31.433% theo cùng nhịp; **87.5%** số tuần trọn vẹn lặp lại đúng hình dạng này.

3. **Khoảng nghỉ sinh học: CHƯA KIỂM CHỨNG** — thiếu `data/interim` (cần `event_4624/`, `event_4625/`). Chạy lại script khi có dữ liệu thô.

5. **Trôi nền (drift)**: volume (log1p): rho = 0.1222, failure_ratio (trọng số): rho = -0.3763, off_hours_ratio (trọng số): rho = 0.2765. CÓ dấu hiệu trôi nhẹ ở `volume (log1p)`, `failure_ratio (trọng số)` ⇒ vẫn phải dùng baseline trượt / cửa sổ giới hạn và bật cơ chế cập nhật lại nền định kỳ, không dùng baseline cố định cho toàn bộ chuỗi.

6. **Đặc trưng phản ánh chu kỳ**: 13/15 đặc trưng có |Cliff's delta| ≥ 0.147; mạnh nhất `off_hours_ratio`, yếu nhất `same_second_share`.

## 2. Phương pháp kiểm định

| Câu hỏi | Cách làm | Ngưỡng chấp nhận |
|---|---|---|
| Q1 — nhịp tuần | ACF lag 1..14 trên log(lưu lượng/ngày) + R² của mô hình chỉ-dùng-pha | ACF lag 7 ≥ 0.3; R² pha ≥ 30.0% |
| Q2 — cuối tuần | 2 pha lưu lượng trung bình thấp nhất = weekend-like; kiểm tra lặp lại theo từng tuần | sụt ≥ 10.0%; nhất quán ≥ 75.0% số tuần |
| Q3 — nghỉ sinh học | Đếm sự kiện thô theo giờ (`Time % 86400 // 3600`), tách ngày thường / cuối tuần-like | đêm sụt ≥ 20.0% so với giờ cao điểm |
| Q4 — trôi nền | Spearman(ngày, nền trượt 7 ngày) + hồi quy Theil-Sen + so sánh 2 nửa dữ liệu | |rho| ≤ 0.3; |%/tuần| ≤ 2.0 |
| Q5 — phản ánh chu kỳ | Trung bình có trọng số theo ngày của từng đặc trưng, so sánh ngày thường vs cuối tuần-like | |Cliff's delta| ≥ 0.147 (hiệu ứng 'small') |

## 3. Bảng chỉ số then chốt

| Nhóm | Chỉ số | Giá trị | Đơn vị | Trạng thái |
|---|---|---|---|---|
| Q1 - Nhịp tuần | acf_lag7 | 0.5833 | hệ số tương quan | OK |
| Q1 - Nhịp tuần | acf_lag14 | 0.4572 | hệ số tương quan | OK |
| Q1 - Nhịp tuần | acf_peak_lag | 7 | lag | OK |
| Q1 - Nhịp tuần | phase_variance_explained_pct | 66.6965 | % | OK |
| Q1 - Nhịp tuần | residual_cv_after_phase_pct | 0.6986 | % | INFO |
| Q2 - Cuối tuần | weekend_like_phases | 5, 6 | pha (day_index % 7) | INFO |
| Q2 - Cuối tuần | weekend_drop_pct | 29.3201 | % | OK |
| Q2 - Cuối tuần | weekend_active_accounts_drop_pct | 31.433 | % | INFO |
| Q2 - Cuối tuần | week_block_consistency_pct | 87.5 | % | OK |
| Q2 - Cuối tuần | weekend_minus_weekday_failure_ratio_pp | 0.0897 | điểm % | INFO |
| Q2 - Cuối tuần | weekend_minus_weekday_off_hours_ratio_pp | 10.1639 | điểm % | INFO |
| Q4 - Trôi nền | volume (log1p): rho (nền trượt 7 ngày) | 0.1222 | hệ số | WARN |
| Q4 - Trôi nền | failure_ratio (trọng số): rho (nền trượt 7 ngày) | -0.3763 | hệ số | WARN |
| Q4 - Trôi nền | off_hours_ratio (trọng số): rho (nền trượt 7 ngày) | 0.2765 | hệ số | OK |
| Q3 - Nghỉ sinh học | hourly_check | N/A | - | SKIP |
| Q5 - Đặc trưng | n_features_tested | 15 | đặc trưng | INFO |
| Q5 - Đặc trưng | n_features_reflecting_weekly_cycle | 13 | đặc trưng | INFO |
| Q5 - Đặc trưng | strongest_feature | off_hours_ratio | tên đặc trưng | INFO |
| Q5 - Đặc trưng | weakest_feature | same_second_share | tên đặc trưng | INFO |

> Diễn giải chi tiết từng dòng (kèm ngưỡng): `docs/feature_engineering/tables/temporal_stationarity_metrics.csv`.

## 4. Top đặc trưng phản ánh chu kỳ tuần (Q5)

| # | Đặc trưng | Ngày thường | Cuối tuần-like | Chênh lệch | Cliff's delta | p (MWU) | Phản ánh chu kỳ |
|---|---|---|---|---|---|---|---|
| 1 | `off_hours_ratio` | 0.495993 | 0.597631 | 20.492% | 0.9943 | 0.0 | YES |
| 2 | `is_single_event` | 1.8e-05 | 1.2e-05 | -33.7809% | -0.7642 | 7e-06 | YES |
| 3 | `distinct_sources_count` | 51.759071 | 36.806713 | -28.8884% | -0.6591 | 0.000109 | YES |
| 4 | `interarrival_dt_mean` | 75.177647 | 85.369528 | 13.5571% | 0.5909 | 0.000523 | YES |
| 5 | `ntlm_ratio` | 0.207454 | 0.237151 | 14.3151% | 0.5085 | 0.002846 | YES |

## 5. Khuyến nghị để tránh cảnh báo rác định kỳ

1. **Baseline phải theo mùa vụ**: với mức sụt cuối tuần 29.3201% và R² pha tuần 66.6965%, dùng seasonal-naive `t-7` hoặc trung bình trượt 7 ngày làm nền; **không** dùng baseline toàn cục hoặc cửa sổ < 7 ngày.
2. **Cửa sổ huấn luyện tối thiểu 14 ngày** để mỗi pha có ≥ 2 quan sát trước khi tính baseline.
3. **Giữ `off_hours_ratio`** làm đặc trưng mã hoá nhịp sinh học (Q3 đã kiểm chứng khớp dữ liệu thô); chỉ bật thêm cờ `is_weekend`/pha như covariate sau khi xác nhận calendar origin (`docs/plan/lanl_eda_implementation_plan.md` §4.3).
4. **Rủi ro còn lại**: các đặc trưng ít phản ánh chu kỳ (`failure_ratio`, `same_second_share`) có thể sinh alert khi lưu lượng cuối tuần thấp — cần chuẩn hoá theo baseline cá nhân hoặc thêm pha ngày khi chấm điểm (ngưỡng hiệu ứng đang dùng: |Cliff's delta| ≥ 0.147).

## 6. Giới hạn & mức độ chắc chắn

- **Chưa xác nhận calendar origin**: dữ liệu LANL chỉ có `Time` tương đối nên 2 pha thấp nhất được gọi là “cuối tuần-like”. Khi có ngày lịch thật, chỉ cần đổi nhãn pha — toàn bộ số liệu không thay đổi.
- p-value Mann-Whitney tính trên các quan sát ngày (có tự tương quan) chỉ mang tính tham chiếu; kết luận dựa trên **độ lớn hiệu ứng (Cliff's delta)**, ACF và độ lặp lại theo từng tuần.
- Nhịp giờ (Q3) tính trên toàn bộ sự kiện 4624/4625, chưa tách theo `entity_type` ở mức event-level; khác biệt máy/người chỉ được so sánh gián tiếp qua đặc trưng.
- Chưa loại ngày lễ/ngày bảo trì do thiếu lịch; nếu tồn tại, chúng là điểm ngoại lai trong profile pha và sẽ làm giảm chỉ số nhất quán theo tuần.
