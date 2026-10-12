# Tuần 5: Phân tích theo loại bất thường — mô hình nào bắt tốt loại nào

**Thiết lập.**
- **Dữ liệu:** 5 run tiêm khối test (ngày 43–60, fit ngày 1–42), phân khúc User, 1.543 nạn nhân mỗi run (7.715 tổng).
- **Mô hình:** cấu hình chốt từ dev (OCSVM standard 1000 thành phần, IF 300 cây, LOF robust k=20) và 4 baseline.
- **Thứ tự dòng:** sắp cố định trước khi fit. Vì vậy số liệu lệch nhẹ so với lưới tuần 4, nhưng nằm trong khoảng 1 độ lệch chuẩn.
- **Chỉ số chính:** **recall trong ngân sách 100 cảnh báo/ngày**, tức analyst xem 100 tài khoản điểm cao nhất mỗi ngày thì thấy bao nhiêu % nạn nhân.

Chạy lại: `python scripts/evaluation/scenario_analysis.py`. Kết quả nằm ở `experiments/analysis/scenario/`:
- `scenario_detection.csv`: mọi chỉ số, trung bình ± std 5 run;
- `complementarity.csv`: tính bổ trợ giữa OCSVM và IF;
- `missed_victims.csv`: nạn nhân bị mọi mô hình bỏ sót;
- `figures/scenario_recall_heatmap.png`, `figures/victim_rank_boxplot.png`.

## 1. Recall trong top-100 cảnh báo/ngày

| Kịch bản | OCSVM | IF | LOF | z-score | Luật thất bại | Luật 6 đk | Tốt nhất |
|---|---|---|---|---|---|---|---|
| brute_force | **0,68** | 0,10 | 0,01 | 0,13 | 0,45 | 0,26 | OCSVM |
| logon_type_switch | **0,58** | 0,42 | 0,51 | 0,02 | 0,00 | 0,00 | OCSVM (LOF sát) |
| new_workstation_burst | **0,29** | 0,07 | 0,10 | 0,02 | 0,01 | 0,00 | OCSVM |
| password_spraying | 0,19 | **0,23** | 0,04 | 0,22 | 0,05 | 0,12 | IF ≈ z-score |
| dormant_wakeup | **0,17** | 0,08 | 0,08 | 0,00 | 0,04 | 0,02 | OCSVM |
| off_hours | 0,03 | **0,08** | 0,05 | 0,06 | 0,04 | 0,03 | không mô hình nào |
| **Tất cả** | **0,32** | 0,16 | 0,13 | 0,08 | 0,10 | 0,07 | OCSVM |

Ngẫu nhiên ≈ 0,014 ở mọi kịch bản.

Hạng trung vị của nạn nhân OCSVM (càng nhỏ càng tốt):
- **brute_force: top 0,8%**;
- logon_type_switch: top 1,2%;
- new_workstation_burst: top 1,8%;
- dormant_wakeup: top 4,0%;
- off_hours: top 7,7%;
- password_spraying: top 26%.

## 2. Nhận xét theo kịch bản

- **brute_force:** OCSVM bắt tốt nhất (0,68; hạng trung vị top 0,8%).
  - Luật thất bại ≥ 5 cờ được **100%** nạn nhân tại ngưỡng của nó, nhưng chỉ 0,45 lọt top-100/ngày. Lý do: nhiều tài khoản thật cũng có ≥ 5 thất bại, nên luật không xếp hạng được.
  - LOF gần như mù (0,01), vì train có khoảng 600 dòng thật giống brute-force (đã giải thích ở tuần 4).
- **logon_type_switch:** cả ba mô hình ML đều bắt khá (0,42–0,58). Đặc trưng `novelty_logontype_7d` và `jaccard_logontype_7d` tách rất rõ (AUC đơn lẻ 0,95–1,00).
- **new_workstation_burst:** chỉ OCSVM bắt được đáng kể (0,29).
  - Nạn nhân bị bỏ sót có ít máy mới hơn (trung vị 6 so với 9 máy) và ít sự kiện hơn (12 so với 18).
  - Bùng nổ nhỏ chìm vào biến động tự nhiên.
- **dormant_wakeup:** yếu ở mọi mô hình (OCSVM 0,17).
  - Ngược trực giác: nạn nhân bị bỏ sót có **nhiều** sự kiện hơn nạn nhân bị bắt (trung vị 150 so với 85).
  - Ngày "thức dậy" được phát lại từ một ngày train của chính tài khoản. Khi khối lượng lớn, nó giống một tài khoản bận bình thường; chỉ `days_since_last_activity` lệch.
- **off_hours:** **không mô hình nào bắt được** (tốt nhất IF 0,08; OCSVM 0,03).
  - Nạn nhân bị bỏ sót có trung vị 107 sự kiện, nạn nhân bị bắt chỉ 9,5. Một ngày bận dời sang đêm vẫn trông như tài khoản bận.
  - Hai đặc trưng đêm (`delta_mean_share_night_7d`, `peer_z_share_night`) chỉ là 2/41 chiều, nên bị các chiều khối lượng lấn át trong khoảng cách của OCSVM.
- **password_spraying:** IF (0,23) ≈ z-score (0,22) > OCSVM (0,19).
  - Nguyên nhân gốc: mỗi nạn nhân chỉ nhận **1–3 lần thử** (trung vị 2). Chiến dịch rải trên nhiều tài khoản, nên ở mức (tài khoản, ngày) gần như không có tín hiệu.
  - Kích thước chiến dịch không khác giữa nạn nhân bị bắt và bị sót (cùng trung vị 8).
  - Không chỉnh mô hình nào giải được: cần đặc trưng **theo nguồn** (một Source thử nhiều tài khoản), đúng kế hoạch fan-out.

## 3. Kết hợp mô hình có đáng không?

Tỷ lệ nạn nhân trong top-100/ngày (`complementarity.csv`):

| Kịch bản | Chỉ OCSVM | Chỉ IF | Cả hai | Không ai (OCSVM, IF) | Bất kỳ mô hình ML |
|---|---|---|---|---|---|
| brute_force | 0,58 | 0,00 | 0,10 | 0,32 | 0,68 |
| logon_type_switch | 0,18 | 0,02 | 0,39 | 0,40 | 0,69 |
| new_workstation_burst | 0,23 | 0,01 | 0,05 | 0,70 | 0,34 |
| password_spraying | 0,01 | 0,05 | 0,19 | 0,76 | 0,25 |
| dormant_wakeup | 0,13 | 0,04 | 0,04 | 0,80 | 0,25 |
| off_hours | 0,01 | 0,06 | 0,02 | 0,92 | 0,12 |
| **Tất cả** | 0,19 | 0,03 | 0,13 | 0,65 | 0,39 |

IF chỉ thêm khoảng 3% nạn nhân mà OCSVM bỏ sót, chủ yếu ở off_hours và spraying. Hợp ba mô hình chỉ nâng recall từ 0,32 lên 0,39, mà phải gánh gấp 2–3 lần cảnh báo. **Ghép mô hình không đáng.** Nên dồn công vào đặc trưng cho các kịch bản đang mù.

## 4. Kết luận

1. **OCSVM là mô hình chính:** tốt nhất ở 4/6 kịch bản (brute-force, đổi LogonType, bùng nổ máy trạm, ngủ đông).
2. **Hai kịch bản mà mọi mô hình đều mù:**
   - **off_hours:** cần đặc trưng tách giờ khỏi khối lượng (ví dụ tỷ lệ đêm có trọng số theo lịch sử giờ của chính tài khoản), hoặc tăng trọng số nhóm đặc trưng thời gian.
   - **password_spraying:** cần đặc trưng theo nguồn (fan-out tài khoản của mỗi Source trong cửa sổ ngắn).
3. Luật thất bại chỉ có giá trị cho brute-force; z-score ngang IF ở spraying. Đây là các mốc đơn giản nên giữ để so.
4. Nhãn tổng hợp cho **cận trên lạc quan**: tấn công thật kín đáo hơn. Phần review thủ công top 50 trên log gốc bổ sung góc nhìn còn thiếu này.
