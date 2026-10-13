# Tuần 5: Ablation theo nhóm đặc trưng

## Thiết lập

- **Dữ liệu:** khối **dev**, 5 run (`dev_seed20261036`–`040`), phân khúc User. Fit ngày 1–35, chấm 36–42. Model seed giống lưới dev (42, 7, 2024, 1, 2).
- **Mô hình:** cấu hình đã chốt (OCSVM standard 1000 thành phần, IF 300 cây, LOF robust k=20). Không chỉnh lại tham số khi bỏ đặc trưng. Dòng được sắp cố định trước khi fit.
- **Cách bỏ:** mỗi cấu hình bỏ một nhóm trong 8 nhóm theo nghĩa của `src/evaluation/feature_groups.py`, cùng với một số cấu hình gộp, rồi so với bản đủ 41 đặc trưng.
- **Cách so:** dùng **hiệu ghép cặp** trên cùng run và seed, rồi lấy TB ± std của hiệu qua 5 run. Dấu * nghĩa là |Δ TB| > 1 std.
- **Đối chứng nhiễu `control_shuffle_rows`:** giữ đủ 41 đặc trưng, chỉ xáo thứ tự dòng train. Kết quả cho thấy:
  - **OCSVM nhiễu đáng kể** chỉ vì thứ tự dòng (do SGD): PR-AUC −0,020 ± 0,019, và recall của new_workstation_burst **−0,16**;
  - IF và LOF gần như không đổi (≤ 0,008).

  Vì vậy với OCSVM, một thay đổi chỉ được tính là thật khi **vượt cả mức đối chứng này**, không chỉ vượt 1 std.
- **Chạy lại:** `python scripts/evaluation/feature_ablation.py`, mất khoảng 1 giờ cho 12 cấu hình × 3 mô hình × 5 run, và tự bỏ qua phần đã chạy.
- **Kết quả** nằm ở `experiments/ablation/feature_groups/`:
  - `delta_overall.csv`: hiệu trên toàn bộ;
  - `delta_scenario.csv`: hiệu theo kịch bản;
  - `summary.csv`, `scenario_summary.csv`: giá trị tuyệt đối;
  - `figures/delta_recall_by_scenario.png`: hình hiệu theo kịch bản.

## 1. Kết quả tổng thể (Δ so với đủ 41 đặc trưng, TB ± std qua 5 run dev)

Mốc đủ 41 đặc trưng:

| Mô hình | PR-AUC | Recall top-100/ngày |
|---|---|---|
| OCSVM | 0,167 | 0,317 |
| IF | 0,062 | 0,139 |
| LOF | 0,055 | 0,164 |

| Cấu hình (số đặc trưng còn lại) | ΔPR-AUC OCSVM | ΔPR-AUC IF | ΔPR-AUC LOF | ΔRecall top-100/ngày OCSVM / IF / LOF | ΔP@50 OCSVM |
|---|---|---|---|---|---|
| **bỏ rhythm + 10 đặc trưng SHAP thấp (30)** | **+0,064 ± 0,014** | **+0,034 ± 0,012** | **+0,037 ± 0,019** | **+0,06 / +0,09 / +0,05** | **+0,18** |
| bỏ rhythm (37) | +0,050 ± 0,013 | +0,019 ± 0,006 | +0,029 ± 0,012 | +0,05 / +0,06 / +0,03 | +0,15 |
| bỏ 10 đặc trưng SHAP thấp (31) | +0,042 ± 0,010 | +0,028 ± 0,006 | −0,005 ± 0,004 | +0,03 / +0,08 / 0,00 | +0,12 |
| bỏ volume + context (35) | +0,014 ± 0,003 | +0,038 ± 0,010 | +0,003 ± 0,011 | +0,01 / +0,08 / +0,01 | +0,02 |
| bỏ time (33) | −0,002 ± 0,003 | +0,012 ± 0,002 | −0,012 ± 0,005 | +0,01 / +0,07 / −0,02 | +0,01 |
| bỏ logon_type (35) | −0,011 ± 0,005 | +0,004 ± 0,006 | −0,012 ± 0,012 | −0,01 / −0,01 / −0,02 | −0,09 |
| bỏ fanout (35) | −0,014 ± 0,003 | −0,012 ± 0,002 | +0,002 ± 0,014 | −0,03 / −0,03 / 0,00 | −0,08 |
| bỏ novelty (36) | −0,039 ± 0,016 | −0,018 ± 0,006 | −0,028 ± 0,006 | −0,03 / −0,04 / −0,10 | −0,26 |
| bỏ failure (35) | **−0,056 ± 0,018** | −0,007 ± 0,005 | +0,002 ± 0,006 | −0,08 / 0,00 / 0,00 | +0,02 |
| bỏ template_v4 (24) | −0,048 ± 0,014 | −0,030 ± 0,005 | −0,020 ± 0,004 | −0,04 / −0,07 / −0,08 | −0,14 |
| *đối chứng: xáo dòng (41)* | *−0,020 ± 0,019* | *0,000 ± 0,001* | *−0,005 ± 0,008* | *−0,02 / 0,00 / 0,00* | *−0,09* |

## 2. Theo kịch bản (Δ recall top-100/ngày, xem hình)

| Cấu hình | Thay đổi rõ nhất |
|---|---|
| bỏ **failure** | OCSVM: brute_force **−0,57**, spraying −0,27, nhưng burst +0,17 và logon_type_switch +0,14. IF: brute_force −0,15, spraying −0,20, logon_type_switch +0,29 |
| bỏ **novelty** | OCSVM: burst **−0,32**, dormant −0,09, nhưng brute_force +0,24. LOF: logon_type_switch **−0,50** |
| bỏ **template_v4** | spraying −0,28 (OCSVM), −0,25 (IF). logon_type_switch −0,53 (LOF), −0,17 (IF) |
| bỏ **logon_type** | logon_type_switch −0,14 (OCSVM), −0,17 (IF), −0,28 (LOF) |
| bỏ **fanout** | burst −0,14 (OCSVM), logon_type_switch −0,12 (IF) |
| bỏ **time** | **off_hours gần như không đổi ở cả 3 mô hình** (−0,02 đến 0,00) |
| bỏ **rhythm** | không kịch bản nào tụt rõ. Tăng brute_force +0,13 (OCSVM), logon_type_switch +0,27 (IF) và +0,13 (LOF) |
| bỏ rhythm + SHAP thấp | brute_force +0,18 (OCSVM), logon_type_switch **+0,38** (IF), off_hours +0,08 (LOF). Không kịch bản nào tụt quá 0,01 |

## 3. Nhận xét

1. **Bỏ nhóm rhythm giúp cả ba mô hình, đúng như SHAP dự đoán.**
   - Ở OCSVM, rhythm là nguồn chính của báo nhầm (`delta_t_cv`). Bỏ đi thì P@50 tăng 0,15 và PR-AUC tăng 0,050, gấp 2,5 lần mức nhiễu đối chứng.
   - Bỏ thêm 10 đặc trưng SHAP thấp (tổng 11 đặc trưng, còn 30) cho kết quả tốt nhất ở **cả ba mô hình**:

     | Mô hình | PR-AUC | Recall top-100/ngày |
     |---|---|---|
     | OCSVM | 0,167 → 0,231 | 0,32 → 0,38 |
     | IF | 0,062 → 0,096 | 0,14 → 0,23 |
     | LOF | 0,055 → 0,092 | 0,16 → 0,21 |

   - **Không kịch bản nào bị hại.**
2. **Các nhóm "thiết kế cho kịch bản nào thì cứu kịch bản đó" đều được xác nhận:**
   - failure → brute_force và spraying;
   - novelty → burst và dormant;
   - logon_type → logon_type_switch;
   - fanout → burst.

   Bỏ các nhóm này hại đúng kịch bản của chúng, khớp với bảng SHAP. Nên **giữ** failure, novelty, logon_type và fanout.
3. **Khối template v4 có giá trị thật.** Bỏ 17 đặc trưng v4 làm cả ba mô hình tụt (PR-AUC −0,02 đến −0,05), nặng nhất ở spraying và logon_type_switch. Đây là bằng chứng ablation mà vòng v4 còn thiếu.
4. **Nhóm time không giúp gì cho off_hours.** Bỏ cả 8 đặc trưng thời gian mà off_hours không đổi.
   - SHAP cho thấy time chiếm tỷ phần lớn trong điểm của nạn nhân off_hours, nhưng điểm vẫn không đủ cao để lọt top, nên nhóm này hiện không tạo khác biệt.
   - Cần đặc trưng giờ mới, so với lịch sử giờ của chính tài khoản, thay vì chỉnh trọng số.
5. **Đánh đổi giữa các nhóm qua khoảng cách.** Bỏ failure làm burst và logon_type_switch *tăng*; bỏ novelty làm brute_force *tăng*.
   - Lý do: các chiều lớn cạnh tranh nhau trong khoảng cách hoặc trong việc chọn chiều để cắt của IF.
   - IF đặc biệt nhạy với số chiều: bỏ bất kỳ nhóm "ít dùng" nào cũng làm logon_type_switch tăng 0,27–0,38. Ít chiều nhiễu hơn thì cây chọn đúng chiều LogonType thường hơn.
6. **OCSVM nhiễu theo thứ tự dòng:** đối chứng cho std khoảng 0,02 PR-AUC, và tới 0,16 recall ở burst. Các hiệu nhỏ của OCSVM theo kịch bản (dưới khoảng 0,15) cần đọc thận trọng.

## 4. Đề xuất

- **Bộ ứng viên 30 đặc trưng** (`drop_rhythm_low_shap10`): bỏ 11 đặc trưng sau, đây cũng là `ablation_configs.json` → `drop_rhythm_low_shap10`.
  - `interarrival_dt_mean`, `delta_t_cv`, `same_second_share`, `is_single_event`;
  - `activity_peak_hour_sin`, `log_distinct_hosts`, `missing_source_ratio`, `log_total_logons`;
  - `dst_host_entropy`, `volume_robust_z_7d`, `peer_z_share_night`.
- **Đã kiểm tra trên khối test một lần, xem mục 5.** Mức tăng được xác nhận. Chưa đổi `feature_schema.yaml`.
- **Điểm mù không giải được bằng cách bớt đặc trưng:** off_hours (cần đặc trưng giờ mới) và password_spraying (cần đặc trưng fan-out theo nguồn). Hai việc này giữ trong đề xuất cho tuần sau.

## 5. Kiểm tra một lần trên khối test

- **Thiết lập:** 5 run test (`test_seed20261061`–`065`), fit ngày 1–42, chấm 43–60, cùng model seed và cấu hình chốt.
- **Chỉ chạy 3 cấu hình:** đủ 41 đặc trưng, bộ 30 đặc trưng, và đối chứng xáo dòng. Không chọn lại gì trên test.
- **Chạy lại:**

  ```
  python scripts/evaluation/feature_ablation.py --dev-grid configs/benchmark_grid_test.yaml --only full drop_rhythm_low_shap10 control_shuffle_rows --out experiments/ablation/feature_groups_test
  ```

| Mô hình | PR-AUC 41 → 30 | ΔPR-AUC (đối chứng) | Recall top-100/ngày 41 → 30 | ΔRecall (đối chứng) | P@50 41 → 30 |
|---|---|---|---|---|---|
| OCSVM | 0,170 → **0,192** | +0,023 ± 0,006 (+0,005 ± 0,015) | 0,32 → **0,38** | +0,055 ± 0,020 (+0,019 ± 0,029) | 0,20 → 0,26 |
| IF | 0,066 → **0,089** | +0,023 ± 0,005 (−0,001 ± 0,007) | 0,16 → **0,21** | +0,047 ± 0,010 (−0,003 ± 0,013) | 0,17 → 0,20 |
| LOF | 0,048 → **0,064** | +0,017 ± 0,006 (0,000 ± 0,002) | 0,13 → **0,15** | +0,021 ± 0,009 (0,000 ± 0,003) | 0,09 → 0,12 |

Δ recall top-100/ngày theo kịch bản trên test:

| Mô hình | brute_force | logon_type_switch | new_workstation_burst | dormant_wakeup | off_hours | password_spraying |
|---|---|---|---|---|---|---|
| OCSVM | **+0,17** (0,68 → 0,85) | +0,08 | +0,07 | +0,01 | +0,01 | 0,00 |
| IF | +0,03 | **+0,20** (0,42 → 0,61) | +0,02 | +0,03 | +0,02 | −0,01 |
| LOF | 0,00 | **+0,15** (0,51 → 0,66) | +0,01 | −0,03 ± 0,03 | −0,01 | +0,01 |

**Kết luận:**

- **Bộ 30 đặc trưng tốt hơn trên test ở cả ba mô hình**, và mức tăng vượt rõ mức nhiễu đối chứng:
  - PR-AUC tăng khoảng 13% ở OCSVM, khoảng 35% ở IF và LOF;
  - recall top-100/ngày tăng 0,02–0,06.
- **Mức tăng của OCSVM trên test nhỏ hơn trên dev** (PR-AUC +0,023 so với +0,064). Đúng như dự đoán: dev hơi lạc quan vì bộ 30 được chọn trên chính dev. Recall thì giữ gần nguyên (+0,055 so với +0,064).
- **Không kịch bản nào tụt rõ.** Lần giảm duy nhất là dormant_wakeup của LOF, −0,03 ± 0,03, nằm trong nhiễu.
- **Đề xuất:** chuyển 11 đặc trưng bị bỏ sang `reserve:` trong `configs/feature_schema.yaml` (schema v4.2, 30 core).
  - Lưới test chính thức với bộ 30 đã chạy: `reports/week5/benchmark_30_dac_trung.md`.
  - Dò lại tham số (k của LOF, số thành phần OCSVM) trên dev nếu còn thời gian.

## Hạn chế

- Mọi chỉ số tính trên nhãn tổng hợp (lớp 1, §5.2 đề cương), nên là **cận trên lạc quan**: bộ 30 đặc trưng được xác nhận tốt hơn với 6 kiểu tấn công tiêm, chưa phải với tấn công thật.
- Bộ 30 đặc trưng được chọn trên chính 5 run dev đã dùng để đo. Mức tăng trên dev có thể hơi lạc quan, nên bắt buộc phải kiểm tra trên test.
- Danh sách 10 đặc trưng SHAP thấp lấy từ một run dev (`dev_seed20261036`), cũng là một trong 5 run ablation.
- Tham số mô hình giữ nguyên theo bản 41 đặc trưng. Với ít chiều hơn, OCSVM (số thành phần Nystroem) và LOF (k) có thể có cấu hình tốt hơn, nhưng chưa dò lại.
