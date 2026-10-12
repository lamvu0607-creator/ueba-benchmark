# Tuần 5: SHAP và LIME — đặc trưng nào đẩy điểm dị biệt lên (chuẩn bị ablation)

## Thiết lập

- **Dữ liệu:** run **dev** `dev_seed20261036`, phân khúc User.
  - Mô hình fit trên ngày 1–35 và chấm ngày 36–42.
  - Chọn ablation dựa trên kết quả giải thích, nên **không dùng khối test** (tránh rò rỉ).
- **Mô hình:** cấu hình chốt (OCSVM standard 1000 thành phần, IF 300 cây, LOF robust k=20), seed 42.
- **Hàm được giải thích:** điểm dị biệt của pipeline (scaler → mô hình), CAO = DỊ BIỆT.
  - Giải thích trên **41 đặc trưng core sau imputer, trước scaler**, tức theo đơn vị gốc.
  - Cả ba mô hình dùng chung imputer, nên chung không gian giải thích.
- **Dòng được giải thích:**
  - 30 nạn nhân mỗi kịch bản (180 dòng, dùng chung cho cả ba mô hình);
  - 30 báo nhầm điểm cao nhất của từng mô hình.
- **SHAP (KernelSHAP):**
  - nền 50 dòng train, 1.000 mẫu liên minh;
  - sai số cộng tính ≤ 3e-14, nghĩa là Σφ + giá trị nền khớp đúng điểm.
- **LIME:**
  - một nửa số dòng trên (105 dòng mỗi mô hình), 2.000 mẫu nhiễu;
  - rời rạc hoá tứ phân vị, mẫu tham chiếu là 20.000 dòng train.
- **So giữa các mô hình:** mỗi dòng chuẩn hoá thành **tỷ phần |đóng góp|** = |φⱼ| / Σ|φ|, vì thang điểm của ba mô hình khác nhau.
- **Nhóm ablation:** schema gộp 17 đặc trưng template vào một nhóm `template_v4`, nên mình chia lại 41 đặc trưng thành **8 nhóm theo nghĩa**:

| Nhóm | Số đặc trưng | Thành phần |
|---|---|---|
| volume | 3 | log_total_logons, volume_robust_z_7d, delta_mean_count_7d |
| failure | 6 | failure_ratio, failure_locked_out_share, 4 đặc trưng template thất bại |
| time | 8 | off_hours_ratio, giờ đỉnh sin/cos, hour_entropy, dist_shift/jaccard giờ, 2 đặc trưng đêm |
| rhythm | 4 | interarrival_dt_mean, delta_t_cv, same_second_share, is_single_event |
| logon_type | 6 | interactive_ratio, rare_logon_type_count_log, ntlm_ratio, 3 đặc trưng template LogonType |
| fanout | 6 | log_distinct_hosts, distinct_sources_count, dst_host_entropy, 3 đặc trưng template source/pair |
| novelty | 5 | new_source/host_count_7d, days_since_last_activity, jaccard_source_7d, novelty_share_pair_7d |
| context | 3 | missing_source_ratio, remote_logon_ratio, custom_proc_share |

Chạy lại (khoảng 10 phút):

```
python scripts/evaluation/explain_models.py
```

Kết quả nằm ở `experiments/analysis/explain/`:

| File | Nội dung |
|---|---|
| `shap_values.parquet`, `lime_weights.parquet` | đóng góp từng (dòng, đặc trưng) |
| `feature_importance.csv` | tỷ phần theo đặc trưng: nạn nhân / báo nhầm, SHAP / LIME, hạng |
| `group_importance.csv` | tỷ phần theo (mô hình, kịch bản, nhóm) |
| `ablation_plan.csv` | mỗi nhóm: tỷ phần ở nạn nhân, kịch bản phụ thuộc nhiều nhất, tỷ phần ở báo nhầm |
| `origin_share.csv` | đặc trưng gốc v3 so với đặc trưng template v4 |
| `shap_lime_agreement.csv` | mức khớp giữa SHAP và LIME |
| `figures/shap_group_by_scenario.png` | tỷ phần nhóm × kịch bản cho từng mô hình |
| `figures/top_features_shap_lime.png` | 15 đặc trưng đóng góp nhiều nhất của mỗi mô hình |

## 1. Nhóm nào đẩy điểm lên, theo kịch bản (SHAP, tỷ phần |φ| trung bình)

| Kịch bản | Nhóm chính (OCSVM / IF / LOF) |
|---|---|
| brute_force | failure 0,76 / 0,51 / 0,60 |
| password_spraying | failure 0,33 / 0,30 / 0,50, sau đó là time |
| off_hours | time 0,52 / 0,50 / 0,29 |
| logon_type_switch | logon_type 0,44 / 0,38 / 0,22, sau đó là fanout và novelty |
| new_workstation_burst | novelty 0,44 / 0,34 / 0,39 và fanout 0,35 / 0,36 / 0,18 |
| dormant_wakeup | novelty 0,49 / 0,40 / 0,30, chủ yếu là `days_since_last_activity` |
| **Báo nhầm** | OCSVM: **rhythm 0,27**, novelty 0,18, failure 0,18 · IF: failure 0,26, logon_type 0,18 · LOF: failure 0,29, logon_type 0,20 |

Tỷ phần trên toàn bộ nạn nhân, so với mức "chia đều" (số đặc trưng / 41):

| Nhóm | Chia đều | OCSVM | IF | LOF |
|---|---|---|---|---|
| novelty | 0,12 | **0,22** | **0,22** | **0,23** |
| failure | 0,15 | 0,21 | 0,17 | **0,26** |
| time | 0,20 | 0,16 | 0,19 | 0,14 |
| fanout | 0,15 | 0,15 | 0,15 | 0,10 |
| logon_type | 0,15 | 0,14 | 0,15 | 0,13 |
| rhythm | 0,10 | 0,04 | 0,05 | 0,06 |
| volume | 0,07 | 0,05 | 0,04 | 0,04 |
| context | 0,07 | 0,03 | 0,04 | 0,04 |

**Nhận xét:**

1. **Mỗi kịch bản được giải thích bằng đúng nhóm đặc trưng thiết kế cho nó.** Bộ đặc trưng hoạt động đúng ý đồ, và các nhóm ít trùng nhau. Điều này đúng ở cả ba mô hình.
2. **novelty và failure là hai nhóm gánh nhiều nhất**, cao hơn hẳn mức chia đều. Đặc trưng đơn lẻ mạnh nhất:
   - `new_source_count_7d`: hạng 1 ở OCSVM và IF;
   - `failure_locked_out_share`: OCSVM;
   - `delta_mean_share_fail_7d`: LOF;
   - `days_since_last_activity`: dormant_wakeup.
3. **rhythm, volume và context đóng góp dưới mức chia đều ở mọi kịch bản.** Riêng với OCSVM, rhythm lại là nguồn chính của **báo nhầm**: `delta_t_cv` chiếm 0,22 tỷ phần ở báo nhầm. Đây là ứng viên ablation số 1, vì bỏ đi có thể giảm nhiễu mà ít mất recall.
4. **off_hours: nhóm time giải thích điểm của nạn nhân (0,52), nhưng điểm tuyệt đối vẫn thấp**, nên vẫn không lọt top. Đây là điều tuần 5 đã nói: 8 chiều thời gian bị pha loãng trong 41 chiều.
   - Tỷ phần cho biết **cái gì** đẩy điểm lên, không cho biết điểm có **đủ cao** hay không.
   - Ablation sẽ trả lời câu hỏi thứ hai.
5. **Đặc trưng template v4 gánh 44–51% tỷ phần ở nạn nhân** (OCSVM 0,44; IF 0,51; LOF 0,46), trong khi chiếm 17/41 = 41% số chiều. Bản v4 có đóng góp thật; mức đóng góp cụ thể cần đo bằng ablation "bỏ cả khối v4".
6. **Đặc trưng ít được dùng nhất** (tỷ phần cao nhất qua ba mô hình chỉ khoảng 0,011–0,016, dưới mức chia đều 0,024):
   - `is_single_event`, `activity_peak_hour_sin`, `log_distinct_hosts`, `missing_source_ratio`, `log_total_logons`;
   - `dst_host_entropy`, `delta_t_cv` (ở nạn nhân), `volume_robust_z_7d`, `peer_z_share_night`, `same_second_share`.

   Đây là ứng viên cắt tỉa. Tỷ phần thấp **chưa chứng minh** đặc trưng vô dụng, vì thông tin có thể đã trùng với đặc trưng khác. Phải xác nhận bằng ablation.

## 2. SHAP và LIME có khớp nhau không?

| Mô hình | Spearman hạng toàn cục | Trùng top-5 trên từng dòng | R² cục bộ của LIME |
|---|---|---|---|
| OCSVM | 0,44 | 0,22 | 0,35 |
| IF | 0,37 | 0,17 | 0,54 |
| LOF | 0,23 | 0,32 | **0,06** |

- **Khớp ở mức nhóm và đặc trưng mạnh nhất.** Ví dụ `failure_locked_out_share` hạng 1 theo LIME của OCSVM, `delta_mean_share_fail_7d` hạng 1 theo cả SHAP lẫn LIME của LOF.
- **Khớp kém ở từng dòng.** Mô hình tuyến tính cục bộ của LIME giải thích được rất ít điểm của LOF (R² 0,06), vì LOF phụ thuộc mật độ hàng xóm và rất phi tuyến.
- LIME còn rời rạc hoá theo tứ phân vị. Với các đặc trưng lệch nặng (đa số bằng 0, như `novelty_*`, `new_*_count`), một bin chứa cả bình thường lẫn bất thường, nên LIME đánh giá thấp nhóm novelty.
- **Kết luận phương pháp:**
  - dùng **SHAP làm căn cứ chính** cho ablation (chính xác, cộng tính, so được giữa mô hình);
  - **LIME chỉ để minh hoạ** từng cảnh báo và đối chiếu, không dùng để xếp hạng đặc trưng.

## 3. Đề xuất thứ tự ablation

Đo bằng lưới dev (5 run), so PR-AUC và recall top-100/ngày theo kịch bản với bản đủ 41 đặc trưng:

| # | Thí nghiệm | Giả thuyết từ SHAP |
|---|---|---|
| 1 | Bỏ nhóm **rhythm** (4) | Mất ít recall; giảm báo nhầm của OCSVM do `delta_t_cv` |
| 2 | Bỏ **volume + context** (6) | Đóng góp thấp nhất; kiểm tra xem có giữ được dormant_wakeup không (volume 0,09 ở kịch bản này) |
| 3 | Bỏ khối **template_v4** (17) | Đo giá trị của vòng v4: dự kiến tụt mạnh ở logon_type_switch và brute_force |
| 4 | Bỏ lần lượt **failure / time / logon_type / fanout / novelty** | Mỗi nhóm chỉ hại kịch bản "của nó" (bảng mục 1). Nếu bỏ time mà off_hours không đổi, nhóm time hiện chưa đủ mạnh |
| 5 | Cắt 10 đặc trưng ít dùng nhất (mục 1, nhận xét 6) | Bộ gọn 31 đặc trưng giữ được hiệu năng |
| 6 | **Tăng trọng số** nhóm time (nhân hệ số sau scaler) | Thử chữa điểm mù off_hours mà không cần đặc trưng mới |

## Hạn chế

- Chỉ một run dev, một seed, 30 nạn nhân mỗi kịch bản. Thứ hạng nhóm rõ ràng, còn thứ hạng từng đặc trưng ở giữa bảng có thể đổi giữa các run. Muốn chắc hơn thì chạy thêm `--run` với các run dev khác.
- KernelSHAP giả định đặc trưng độc lập khi thay bằng giá trị nền. Các đặc trưng tương quan, ví dụ `failure_ratio` và `delta_mean_share_fail_7d`, có thể chia nhau phần đóng góp một cách tuỳ ý. Vì vậy nên đọc ở **mức nhóm**.
- Tỷ phần là tầm quan trọng **tương đối** trong điểm của một dòng, không phải đóng góp vào khả năng phát hiện. Ablation mới đo được đóng góp thật.
