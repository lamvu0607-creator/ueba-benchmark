# Tuần 5: Tóm tắt những gì cần đọc (13/10/2026)

Tuần 5 gồm sáu việc:
1. thí nghiệm độ dài cửa sổ lịch sử;
2. phân tích theo loại bất thường;
3. review top 50 cảnh báo;
4. giải thích mô hình bằng SHAP/LIME;
5. ablation theo nhóm đặc trưng;
6. bảng benchmark cuối với bộ 30 đặc trưng.

Mỗi mục gồm kết luận chính và file chi tiết cần đọc. Cuối file là bảng đối chiếu với sản phẩm tuần 5 trong đề cương.

Mọi số liệu tính trên nhãn tiêm tổng hợp (lớp 1, §5.2 đề cương), nên là **cận trên lạc quan**. Review top 50 (mục 3) là lớp đánh giá định tính trên log thật.

## 1. Cửa sổ lịch sử 7 / 14 / 21 ngày → giữ 7 ngày

Thí nghiệm chạy trên một run tiêm test (test_seed20261061), mỗi cửa sổ 5 model seed.

| Mô hình | Recall top-100/ngày 7 / 14 / 21 ngày (chỉ số chính) | PR-AUC 7 / 14 / 21 ngày (phụ) |
|---|---|---|
| OCSVM | **0,317** / 0,318 / 0,294 | 0,165 / 0,169 / 0,160 |
| IF | **0,159** / 0,156 / 0,146 | 0,066 / 0,063 / 0,059 |
| LOF | **0,121** / 0,034 / 0,003 | 0,043 / 0,027 / 0,020 |

- **14 ngày:** khác 7 ngày chưa tới 1 độ lệch chuẩn.
- **21 ngày:** giảm recall top-100/ngày của mọi mô hình ML, tốn thời gian gấp ~3 lần và tốn nhiều RAM hơn.
- **Đánh đổi theo kịch bản:** cửa sổ dài giúp password spraying (ROC-AUC 0,69 → 0,77) nhưng hại dormant wakeup.
- **Phát hiện phụ:** thứ tự dòng trong ma trận thay đổi giữa các lần tính và làm lệch kết quả IF/OCSVM tới 0,027 PR-AUC. Các script mới đều sắp dòng cố định.

📄 Đọc: `experiments/grid/window_ablation/README.md`

## 2. Mô hình nào bắt tốt loại nào → OCSVM là mô hình chính, có 2 điểm mù

Chỉ số chính là **recall trong 100 cảnh báo/ngày**. Bảng dưới là 5 run test, cấu hình chốt từ dev, 41 đặc trưng; số với bộ 30 đặc trưng ở mục 6.

| Kịch bản | OCSVM | IF | LOF |
|---|---|---|---|
| brute_force | **0,68** | 0,10 | 0,01 |
| logon_type_switch | **0,58** | 0,42 | 0,51 |
| new_workstation_burst | **0,29** | 0,07 | 0,10 |
| password_spraying | 0,19 | **0,23** | 0,04 |
| dormant_wakeup | **0,17** | 0,08 | 0,08 |
| off_hours | 0,03 | 0,08 | 0,05 |
| Tất cả | **0,32** | 0,16 | 0,13 |

- **OCSVM tốt nhất ở 4/6 kịch bản.**
- **Gộp ba mô hình không đáng:** chỉ tăng recall từ 0,32 lên 0,39 mà phải xem gấp 2–3 lần số cảnh báo.
- **Điểm mù 1, off_hours:** một ngày bận dời sang đêm vẫn trông như tài khoản bận. Cần đặc trưng giờ tách khỏi khối lượng.
- **Điểm mù 2, password_spraying:** mỗi nạn nhân chỉ bị thử 1–3 lần. Cần đặc trưng theo máy nguồn (fan-out).

📄 Đọc: `reports/week5/phan_tich_theo_kich_ban.md`. Số liệu và hình ở `experiments/analysis/scenario/`.

## 3. Review top 50 cảnh báo trên log gốc → mới có bản của Claude, chờ hai bạn chấm

Lấy 50 cảnh báo điểm cao nhất của mỗi mô hình (OCSVM, IF, LOF), hợp lại được 137 cảnh báo. Claude chấm mù 137 cảnh báo này:

| Mô hình | a — đáng ngờ | b — giải thích được | c — nhiễu |
|---|---|---|---|
| OCSVM | 19 | 29 | **2** |
| IF | 19 | 5 | 26 |
| LOF | 10 | 1 | 39 |

- **OCSVM:** gần như không báo nhiễu. Phát hiện đáng giá nhất là User272607: 0,1–1,9 triệu đăng nhập NTLM/ngày tới 600–1.600 máy mới.
- **IF:** tìm được nhiều tài khoản đáng ngờ khác nhau nhất nhưng một nửa danh sách là nhiễu. **LOF:** chủ yếu là nhiễu.
- **Nhiễu lặp lại** đến từ hai nguồn: tỷ lệ tính trên rất ít sự kiện (1–6 sự kiện/ngày), và tài khoản mới chưa có lịch sử.
- Bộ review dùng 41 đặc trưng. Chưa làm lại với bộ 30, để giữ nguyên bộ thẻ cho hai bạn chấm.

📄 Đọc sau (đã lưu lại, chưa cần đọc ngay):
- `reports/week5/review_top50.md`: báo cáo review, bản của Claude (một người chấm).
- `experiments/analysis/review_top50/claude_review.csv`: nhãn và lý do của Claude cho từng cảnh báo.
  ⚠️ **Chỉ mở sau khi hai bạn đã tự chấm xong**, để giữ tính độc lập. Tương tự với `answer_key.csv`.

## 4. SHAP và LIME → mỗi kịch bản được giải thích đúng bằng nhóm đặc trưng thiết kế cho nó

Giải thích điểm của ba mô hình chốt trên một run dev, gồm 30 nạn nhân mỗi kịch bản và 30 báo nhầm của mỗi mô hình. 41 đặc trưng được chia thành 8 nhóm theo nghĩa: volume, failure, time, rhythm, logon_type, fanout, novelty, context.

- **Mỗi nhóm đẩy điểm của đúng kịch bản thiết kế cho nó:**
  - failure → brute_force (tỷ phần 0,51–0,76);
  - time → off_hours;
  - logon_type → logon_type_switch;
  - novelty và fanout → burst;
  - `days_since_last_activity` → dormant_wakeup.
- **novelty và failure gánh nhiều nhất.** rhythm, volume và context thấp hơn mức chia đều ở mọi kịch bản.
- **rhythm là nguồn chính của báo nhầm ở OCSVM:** `delta_t_cv` chiếm 0,22 tỷ phần ở báo nhầm.
- **Đặc trưng template v4 gánh 44–51% tỷ phần ở nạn nhân,** trong khi chỉ chiếm 41% số chiều.
- **SHAP và LIME khớp ở mức nhóm, kém ở từng dòng.** R² cục bộ của LIME với LOF chỉ 0,06. Vì vậy **SHAP là căn cứ chính**, còn LIME chỉ để minh hoạ.
- **Bổ sung theo từng đặc trưng, 3 run dev** (báo cáo tuần 5, mục 3.4):
  - **Thứ hạng từng đặc trưng ổn định gần như nhóm:** Spearman giữa các run đạt 0,93–0,99. Vì vậy đọc kết quả theo từng đặc trưng là hợp lệ.
  - **Mỗi kịch bản do 1–2 đặc trưng dẫn dắt, không phải cả nhóm.** Ví dụ: off_hours gần như chỉ dựa vào `delta_mean_share_night_7d`; dormant dựa vào `days_since_last_activity`; burst dựa vào `new_source_count_7d`.
  - **Nhóm theo nghĩa phần lớn không phải cụm tương quan.** Chỉ 6 cụm (14/41 đặc trưng) có |ρ| ≥ 0,7.

📄 Đọc: `reports/week5/shap_lime.md` và mục 3.4 của `reports/week5/bao_cao_tuan5.md`. Số liệu và hình ở `experiments/analysis/explain/` và `experiments/analysis/explain_stability/`.

## 5. Ablation theo nhóm đặc trưng (dev, 5 run) → bỏ 11 đặc trưng, còn 30

Bỏ từng nhóm rồi so với bản đủ 41 đặc trưng bằng hiệu ghép cặp trên cùng run và seed. Có thêm đối chứng xáo thứ tự dòng: OCSVM nhiễu khoảng 0,02 PR-AUC chỉ vì thứ tự dòng.

| Cấu hình | ΔPR-AUC OCSVM / IF / LOF |
|---|---|
| **bỏ rhythm + 7 đặc trưng SHAP thấp (còn 30)** | **+0,064 / +0,034 / +0,037** |
| bỏ rhythm (còn 37) | +0,050 / +0,019 / +0,029 |
| bỏ failure | −0,056 / −0,007 / +0,002 |
| bỏ novelty | −0,039 / −0,018 / −0,028 |
| bỏ khối template v4 (17) | −0,048 / −0,030 / −0,020 |
| bỏ time | −0,002 / +0,012 / −0,012 |

- **Giữ** failure, novelty, logon_type, fanout và khối template v4. Bỏ nhóm nào thì hại đúng kịch bản của nhóm đó, khớp với SHAP.
- **Bỏ rhythm giúp cả ba mô hình,** đúng như SHAP dự đoán.
- **Bỏ cả 8 đặc trưng time mà off_hours không đổi.** Điểm mù này cần đặc trưng mới, không chữa được bằng cách chỉnh bộ hiện có.
- **Kiểm tra một lần trên test xác nhận bộ 30** ở cả ba mô hình, và mức tăng vượt đối chứng:
  - PR-AUC OCSVM 0,170 → 0,192, IF 0,066 → 0,089, LOF 0,048 → 0,064;
  - không kịch bản nào tụt rõ.
- **Thêm lại từng đặc trưng trong 11 đặc trưng vào bộ 30** (dev, báo cáo tuần 5, mục 4.4):
  - **Không đặc trưng nào bị bỏ oan.** Không lần thêm lại nào cải thiện rõ.
  - `interarrival_dt_mean` có SHAP cao ở LOF nhưng thêm lại làm LOF tệ hơn (PR-AUC −0,039). Như vậy SHAP cao chưa chắc là có ích.
  - **Báo nhầm của OCSVM không chỉ do `delta_t_cv`:** `interarrival_dt_mean` và `is_single_event` cũng gây hại ở mức tương tự. Bỏ cả nhóm rhythm là đúng.
  - **LOF không tất định:** fit lại cùng dữ liệu đã lệch −0,016 ± 0,014 PR-AUC, do chỉ mục HNSW dựng đa luồng.

📄 Đọc: `reports/week5/ablation_nhom_dac_trung.md`. Số liệu ở `experiments/ablation/feature_groups/` (dev) và `experiments/ablation/feature_groups_test/` (test).

## 6. Bảng benchmark cuối (lưới test chính thức, 30 đặc trưng)

5 run test, cấu hình chốt từ dev, dòng sắp cố định. Lưới tuần 4 với 41 đặc trưng giữ nguyên để so sánh.

| Mô hình | PR-AUC | ROC-AUC | Recall top-100/ngày | PR-AUC 41 đặc trưng (tuần 4) |
|---|---|---|---|---|
| **OCSVM** | **0,192 ± 0,016** | 0,911 | **0,376** | 0,182 ± 0,023 |
| IF | 0,089 ± 0,006 | 0,913 | 0,209 | 0,065 ± 0,003 |
| LOF | 0,062 ± 0,004 | 0,701 | 0,147 | 0,047 ± 0,004 |
| z-score (baseline tốt nhất) | 0,032 ± 0,003 | 0,743 | 0,075 | — |
| ngẫu nhiên | 0,010 | 0,503 | 0,014 | — |

- **Thứ hạng giữ nguyên:** OCSVM > IF > LOF > mọi baseline.
- **IF và LOF tăng khoảng 35% PR-AUC.**
- **Mức tăng PR-AUC của OCSVM so với bảng tuần 4 nằm trong 1 std.** Lý do: lưới tuần 4 chưa sắp dòng. So cùng quy trình thì OCSVM tăng 0,170 → 0,192.
- **Recall top-100/ngày tăng ở cả ba mô hình.** Ở ngân sách cảnh báo 1–2%, OCSVM bắt thêm khoảng 4 điểm % nạn nhân mà precision không đổi.
- **Thiệt hại duy nhất đáng kể:** dormant_wakeup của LOF, ROC-AUC 0,60 → 0,52.
- **off_hours và password_spraying vẫn là điểm mù.**
- `configs/feature_schema.yaml` **chưa đổi.** Bộ 30 đặc trưng được chọn qua `drop_features` trong `configs/benchmark_grid_test_30f.yaml`.

📄 Đọc: `reports/week5/benchmark_30_dac_trung.md`. Kết quả ở `experiments/grid/test_30f/`.

## 7. Kiểm tra chạy lại

- **Clone sạch nhánh `feature/target-rate-1pct` vào thư mục mới rồi chạy `pytest`:** 252 passed, 2 skipped.
  - Hai test bị bỏ qua cần `data/processed/feature_matrix_processed.parquet`. File này tải từ Google Drive (README §6), không nằm trong Git.
  - Bản clone không cần file nào ngoài Git để chạy test.
- **Giới hạn:** lần này dùng lại môi trường Python của máy hiện tại, chưa phải máy sạch thật. Cần một bạn cài lại theo README §2 (conda hoặc venv) trên máy khác.
- **README** đã thêm mục "Tuần 5" với lệnh chạy lại từng thí nghiệm.

## Đối chiếu sản phẩm tuần 5 (đề cương, mục 4)

| Sản phẩm | Trạng thái |
|---|---|
| Bảng phân tích theo loại bất thường | ✅ Mục 2 và mục 6 |
| Báo cáo review top 50 | ⏳ Đã có bộ thẻ làm mù và bản chấm của Claude. **Còn thiếu hai bạn chấm độc lập và Cohen's κ** |
| Kết quả ablation | ✅ Mục 4–6, kèm kiểm tra trên test và bảng benchmark cuối |
| Xác nhận chạy lại được trên môi trường mới | ⚠️ Clone sạch chạy được toàn bộ test. Còn thiếu cài trên máy sạch thật |
| Dọn mã, bổ sung README | ⚠️ Đã thêm mục tuần 5 vào README. Phần mô tả đặc trưng trong README vẫn ghi schema v4.0 với 39 đặc trưng, cần cập nhật |

## Việc còn lại

1. **Review top 50:** hai bạn chấm độc lập.
   - Đọc `docs/huong_dan_review_top50.md` và các thẻ trong `experiments/analysis/review_top50/alert_cards.html`.
   - Ghi nhãn vào `review_sheet.csv`.
   - Chạy `python scripts/evaluation/review_agreement.py` để có Cohen's κ, thống nhất các cảnh báo hai bạn chấm khác nhau, rồi cập nhật `review_top50.md`.
2. **Cài lại trên máy sạch** theo README §2, chạy `pytest` và một lệnh thí nghiệm, rồi ghi kết quả vào mục 7.
3. **Chờ quyết định:** chuyển 11 đặc trưng sang `reserve:` trong `feature_schema.yaml` (schema v4.2, 30 core).
   - Khi đó mọi pipeline dùng bộ 30 mà không cần `drop_features`.
   - Cập nhật luôn bảng đặc trưng trong README.
4. **Đề xuất cho tuần 6** (báo cáo kỹ thuật dùng bảng ở mục 6):
   - đặc trưng fan-out theo nguồn cho spraying;
   - đặc trưng giờ so với lịch sử giờ của chính tài khoản, cho off_hours;
   - làm mượt tỷ lệ khi khối lượng nhỏ, để giảm nhiễu trong top cảnh báo;
   - dò lại k của LOF và số thành phần OCSVM với 30 đặc trưng, nếu còn thời gian.
