# Tuần 5: Bảng benchmark cuối trên khối test với bộ 30 đặc trưng

> **Đã thay thế:** bộ đặc trưng cuối là **bộ 22** (K = 18), xem báo cáo tuần 5 mục 4–5 và `experiments/grid/test_22f/`. File này giữ lại kết quả bộ 30 để so sánh.

## Thiết lập

- **Lưới:** `configs/benchmark_grid_test_30f.yaml`, bản sao của lưới test tuần 4 (`configs/benchmark_grid_test.yaml`).
  - Cùng 5 run test (fit ngày 1–42, chấm 43–60), cùng model seed và cùng cấu hình chốt từ dev; không chọn lại gì.
- **Hai khác biệt so với lưới tuần 4:**
  1. `drop_features`: mô hình ML bỏ 11 đặc trưng theo ablation trên dev (`reports/week5/ablation_nhom_dac_trung.md`).
     - Bỏ: nhóm rhythm (4) cộng 7 đặc trưng SHAP thấp.
     - Baseline giữ nguyên đặc trưng, nên số của baseline gần như không đổi.
  2. `sort_rows: true`: sắp dòng cố định trước khi fit. Lưới tuần 4 chưa sắp.
- **Chưa đổi `configs/feature_schema.yaml`.** Bộ 30 chỉ được chọn qua cấu hình lưới.
- **Chạy lại:**

  ```
  python scripts/evaluation/run_grid.py --config configs/benchmark_grid_test_30f.yaml
  ```

- **Kết quả** nằm ở `experiments/grid/test_30f/`: `grid_summary.csv`, `grid_scenarios_summary.csv`, `contamination_summary.csv`, `pr_curves.csv` và `figures/`. Lưới tuần 4 ở `experiments/grid/test/` giữ nguyên.

## 1. Bảng benchmark (TB ± std qua 5 run test)

| Mô hình | Số đặc trưng | PR-AUC | ROC-AUC | P@100 | Recall top-100/ngày | P@100/ngày | Recall tại ngưỡng train (5%) | Fit + chấm (s) |
|---|---|---|---|---|---|---|---|---|
| **OCSVM** | 30 | **0,192 ± 0,016** | 0,911 | 0,234 | **0,376** | **0,323** | 0,659 | 48 |
| IF | 30 | 0,089 ± 0,006 | 0,913 | 0,166 | 0,209 | 0,179 | 0,400 | 14 |
| LOF | 30 | 0,062 ± 0,004 | 0,701 | 0,088 | 0,147 | 0,126 | 0,349 | 11 |
| z-score | baseline | 0,032 ± 0,003 | 0,743 | 0,228 | 0,075 | 0,064 | 0,154 | 2 |
| luật thất bại ≥ 5 | baseline | 0,025 ± 0,000 | 0,623 | 0,008 | 0,098 | 0,084 | 0,198 | 1 |
| luật 6 điều kiện | baseline | 0,015 ± 0,001 | 0,536 | 0,042 | 0,073 | 0,062 | 0,124 | 1 |
| ngẫu nhiên | baseline | 0,010 ± 0,000 | 0,503 | 0,018 | 0,014 | 0,012 | 0,052 | 2 |

Tỷ lệ dương ≈ 1% (1.543 nạn nhân mỗi run).

Xếp hạng không đổi so với tuần 4: **OCSVM > IF > LOF > mọi baseline**.

## 2. So với bộ 41 đặc trưng

Có hai cách so, cần đọc cả hai:

| Mô hình | PR-AUC 41 đặc trưng, lưới tuần 4 (chưa sắp dòng) | PR-AUC 41 đặc trưng, sắp dòng (ablation trên test) | **PR-AUC 30 đặc trưng** | Recall top-100/ngày 41 (tuần 4) → 30 |
|---|---|---|---|---|
| OCSVM | 0,182 ± 0,023 | 0,170 ± 0,015 | **0,192 ± 0,016** | 0,334 → **0,376** |
| IF | 0,065 ± 0,003 | 0,066 ± 0,006 | **0,089 ± 0,006** | 0,155 → **0,209** |
| LOF | 0,047 ± 0,004 | 0,048 ± 0,003 | **0,062 ± 0,004** | 0,129 → **0,147** |

- **Cột "sắp dòng" là so sánh sạch**, cùng quy trình chỉ khác bộ đặc trưng. Bộ 30 tăng PR-AUC +0,023 (OCSVM), +0,023 (IF), +0,017 (LOF).
- **So với bảng chính thức tuần 4**, mức tăng của OCSVM nhỏ hơn (+0,010) và nằm trong 1 std. Lý do: lưới tuần 4 chưa sắp dòng, và thứ tự dòng tình cờ có lợi cho OCSVM (0,182 so với 0,170 khi sắp).
- IF và LOF không nhạy thứ tự dòng, nên tăng rõ ở cả hai cách so: khoảng +35% PR-AUC.
- **Recall top-100/ngày tăng ở cả ba mô hình theo cả hai cách so.** Đây là chỉ số gần với vận hành nhất (analyst xem 100 cảnh báo mỗi ngày).

## 3. Theo kịch bản (ROC-AUC, 41 tuần 4 → 30)

| Kịch bản | OCSVM | IF | LOF |
|---|---|---|---|
| brute_force | 0,994 → 0,995 | 0,886 → **0,928** | 0,453 → 0,517 |
| logon_type_switch | 0,985 → 0,988 | 0,957 → 0,971 | 0,896 → 0,921 |
| new_workstation_burst | 0,975 → 0,983 | 0,900 → **0,935** | 0,919 → 0,934 |
| dormant_wakeup | 0,947 → 0,940 | 0,911 → 0,936 | 0,604 → **0,521** |
| off_hours | 0,906 → 0,887 | 0,877 → 0,876 | 0,730 → 0,718 |
| password_spraying | 0,696 → 0,675 | 0,797 → **0,832** | 0,529 → 0,597 |

PR-AUC theo kịch bản nằm trong `grid_scenarios_summary.csv`. Đáng chú ý là logon_type_switch của LOF: 0,047 → 0,116.

**Nhận xét:**
- **IF được lợi nhiều nhất và đều nhất.** Ít chiều nhiễu hơn thì cây chọn đúng chiều hơn.
- **LOF giảm ở dormant_wakeup** (ROC-AUC 0,604 → 0,521). Đây là thiệt hại duy nhất đáng kể.
  - LOF vốn đã gần mức ngẫu nhiên ở kịch bản này.
  - Bỏ `log_total_logons` và `volume_robust_z_7d` làm mất tín hiệu "khối lượng bất thường khi thức dậy".
- **OCSVM gần như giữ nguyên ROC-AUC theo kịch bản.** off_hours và spraying giảm nhẹ 0,02, nằm trong nhiễu thứ tự dòng của OCSVM. Mức tăng của OCSVM nằm ở phần đầu bảng xếp hạng (recall top-100/ngày, P@100/ngày), chứ không ở ROC-AUC tổng thể.
- **Hai điểm mù không đổi:** off_hours và password_spraying. Bớt đặc trưng không chữa được; cần đặc trưng mới.

## 4. Độ nhạy contamination (OCSVM, recall / precision tại ngưỡng train)

| contamination | 41 đặc trưng (tuần 4) | 30 đặc trưng |
|---|---|---|
| 0,01 | 0,419 / 0,219 | **0,460** / 0,221 |
| 0,02 | 0,544 / 0,166 | **0,587** / 0,164 |
| 0,05 | 0,711 / 0,102 | 0,719 / 0,098 |
| 0,10 | 0,829 / 0,067 | 0,825 / 0,062 |

- Ở ngân sách cảnh báo chặt (1–2%), bộ 30 bắt thêm khoảng 4 điểm % nạn nhân mà precision không đổi.
- IF ở 5%: recall 0,360 → 0,482, precision 0,057 → 0,074.

Chi tiết ở `contamination_summary.csv` và `figures/contamination_sensitivity.png`.

## 5. Kết luận

1. **Bộ 30 đặc trưng tốt hơn hoặc ngang bộ 41 ở cả ba mô hình**, theo cùng quy trình đánh giá:
   - IF và LOF tăng PR-AUC khoảng 35%;
   - OCSVM tăng recall trong ngân sách cảnh báo.

   Thiệt hại duy nhất đáng kể là dormant_wakeup của LOF.
2. **OCSVM 30 đặc trưng là cấu hình tốt nhất của benchmark:** PR-AUC 0,19, ROC-AUC 0,91, bắt 38% nạn nhân trong 100 cảnh báo mỗi ngày.
3. Theo §5.2 đề cương, mọi số liệu là **cận trên lạc quan** trên nhãn tổng hợp.
4. **Việc còn lại để chốt (chờ quyết định):**
   - chuyển 11 đặc trưng sang `reserve:` (schema v4.2), để mọi pipeline dùng bộ 30 mà không cần `drop_features`;
   - cập nhật bảng benchmark trong báo cáo tổng.
