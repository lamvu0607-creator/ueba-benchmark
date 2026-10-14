# Tuần 5: Tóm tắt những gì cần đọc (13/10/2026)

Tuần 5 gồm sáu việc:
1. thí nghiệm độ dài cửa sổ lịch sử;
2. phân tích theo loại bất thường;
3. review top 50 cảnh báo;
4. giải thích mô hình bằng SHAP;
5. loại bỏ đặc trưng (bộ 30, đường cong K, chốt K = 18);
6. bảng benchmark cuối với bộ 22 đặc trưng.

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

Chỉ số chính là **recall trong 100 cảnh báo/ngày**. Bảng dưới là 5 run test, cấu hình chốt từ dev, 41 đặc trưng; số với bộ 22 đặc trưng ở mục 6.

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
- Bản chấm của Claude ở trên dùng bộ review 41 đặc trưng. Bộ review mới với 22 đặc trưng (142 cảnh báo) ở `experiments/analysis/review_top50_22f/`; hai bạn chấm trên bộ mới.

📄 Đọc sau (đã lưu lại, chưa cần đọc ngay):
- `reports/week5/review_top50.md`: báo cáo review, bản của Claude (một người chấm).
- `experiments/analysis/review_top50/claude_review.csv`: nhãn và lý do của Claude cho từng cảnh báo.
  ⚠️ **Chỉ mở sau khi hai bạn đã tự chấm xong**, để giữ tính độc lập. Tương tự với `answer_key.csv`.

## 4. SHAP → mỗi kịch bản được giải thích đúng bằng nhóm đặc trưng thiết kế cho nó

Giải thích điểm của ba mô hình chốt trên một run dev, gồm 30 nạn nhân mỗi kịch bản và 30 báo nhầm của mỗi mô hình. Báo nhầm không phải kịch bản tiêm: đó là 30 dòng bình thường (không tiêm) có điểm cao nhất, dùng để xem đặc trưng nào làm mô hình báo sai. Luồng phân tích: SHAP theo 8 nhóm theo nghĩa → SHAP theo từng đặc trưng → nhóm rhythm.

- **Mỗi nhóm đẩy điểm của đúng kịch bản thiết kế cho nó, nhất quán ở cả ba mô hình:**
  - failure → brute_force (tỷ phần 0,51–0,76);
  - time → off_hours;
  - logon_type → logon_type_switch;
  - novelty và fanout → burst;
  - `days_since_last_activity` → dormant_wakeup.
- **novelty và failure gánh nhiều nhất.** rhythm, volume và context thấp hơn mức chia đều ở mọi kịch bản.
- **10 ứng viên cắt tỉa theo từng đặc trưng:** lấy tỷ phần cao nhất qua 3 mô hình, 10 đặc trưng thấp nhất chỉ đạt 0,011–0,016, dưới mức chia đều 0,024. Con số 10 là ngưỡng chọn tròn: hạng 10 và 11 chỉ chênh 0,0002.
- **rhythm ít giúp bắt nạn nhân nhưng gây báo nhầm ở OCSVM:** 0,042 tỷ phần ở nạn nhân so với 0,266 ở báo nhầm, gần như toàn bộ do `delta_t_cv`.
- **SHAP là căn cứ chính, LIME chỉ để minh hoạ:** R² cục bộ của LIME với LOF chỉ 0,06.

📄 Đọc: mục 3 của `reports/week5/bao_cao_tuan5.md` và `reports/week5/shap_lime.md`. Số liệu ở `experiments/analysis/explain/`.

## 5. Loại bỏ đặc trưng (dev, 5 run) → chốt K = 18, còn 22 đặc trưng

Bỏ danh sách ứng viên rồi so với bản đủ 41 đặc trưng bằng hiệu ghép cặp trên cùng run và seed. Đối chứng xáo thứ tự dòng: OCSVM nhiễu khoảng 0,02 PR-AUC chỉ vì thứ tự dòng.

| Cấu hình | ΔRecall top-100/ngày OCSVM / IF / LOF | ΔPR-AUC OCSVM / IF / LOF |
|---|---|---|
| bỏ rhythm + 10 SHAP thấp (còn 30) | +0,06 / +0,09 / +0,05 | +0,064 / +0,034 / +0,037 |
| bỏ rhythm (còn 37) | +0,05 / +0,06 / +0,03 | +0,050 / +0,019 / +0,029 |
| bỏ 10 SHAP thấp (còn 31) | +0,03 / +0,08 / 0,00 | +0,042 / +0,028 / −0,005 |

- **Bộ 30** (bước đầu) bỏ 4 đặc trưng rhythm và 7 đặc trưng SHAP thấp; con số 10 là chọn tròn.
- **Đường cong K** (bỏ rhythm + K đặc trưng SHAP thấp nhất, K = 0–20) thay cho con số chọn tròn:
  - OCSVM phẳng từ 33 đến 22 đặc trưng;
  - IF càng bỏ càng tốt đến 22 đặc trưng (recall 0,195 → 0,264);
  - LOF tốt nhất ở 30–33 đặc trưng trên dev.
- **Chốt K = 18 (22 đặc trưng, bỏ 19):** đúng 18 đặc trưng dưới mức chia đều 1/41 ở mọi mô hình và mọi run; K = 20 không tốt hơn ở mô hình nào.
- **Kiểm tra trên test, so với bộ 30** (hiệu ghép cặp, 5 run):
  - OCSVM −0,003 ± 0,005, IF +0,034 ± 0,013, LOF +0,028 ± 0,005 recall;
  - LOF đảo chiều so với dev, nên không dựa vào LOF để chọn K;
  - giảm rõ duy nhất theo kịch bản: OCSVM ở logon_type_switch (0,65 → 0,61), bù bằng brute_force.

📄 Đọc: mục 4–5 của `reports/week5/bao_cao_tuan5.md`, `reports/week5/chon_so_dac_trung_K.md` và `reports/week5/kiem_tra_test_K_va_context.md`. Số liệu ở `experiments/ablation/feature_groups/`, `experiments/ablation/kcurve/` (dev) và `experiments/ablation/kcurve_test/` (test).

## 6. Bảng benchmark cuối (lưới test chính thức, 22 đặc trưng)

5 run test, cấu hình chốt từ dev, dòng sắp cố định. Lưới bộ 30 và lưới tuần 4 với 41 đặc trưng giữ nguyên để so sánh.

| Mô hình | Recall top-100/ngày | PR-AUC | ROC-AUC | Recall / PR-AUC bộ 30 |
|---|---|---|---|---|
| **OCSVM** | **0,373 ± 0,022** | **0,189 ± 0,013** | 0,908 | 0,376 / 0,192 |
| IF | 0,242 ± 0,016 | 0,111 ± 0,011 | 0,921 | 0,209 / 0,089 |
| LOF | 0,176 ± 0,015 | 0,070 ± 0,008 | 0,717 | 0,147 / 0,062 |
| z-score (baseline tốt nhất) | 0,075 | 0,032 ± 0,003 | 0,743 | — |
| ngẫu nhiên | 0,014 | 0,010 | 0,503 | — |

- **Thứ hạng giữ nguyên:** OCSVM > IF > LOF > mọi baseline. IF có ROC-AUC nhỉnh hơn OCSVM nhưng chỉ bắt khoảng hai phần ba số nạn nhân trong 100 cảnh báo/ngày.
- **So với bộ 30:** OCSVM ngang, IF +0,033 recall, LOF +0,029 recall.
- **So với 41 đặc trưng ban đầu:** recall OCSVM 0,32 → 0,37, IF 0,16 → 0,24, LOF 0,13 → 0,18.
- **LOF ở dormant_wakeup vẫn yếu:** ROC-AUC 0,604 (41) → 0,521 (30) → 0,570 (22).
- **off_hours và password_spraying vẫn là điểm mù.**
- `configs/feature_schema.yaml` **chưa đổi.** Bộ 22 đặc trưng được chọn qua `drop_features` trong `configs/benchmark_grid_test_22f.yaml`.

📄 Đọc: mục 5 của `reports/week5/bao_cao_tuan5.md`. Kết quả ở `experiments/grid/test_22f/` (bộ 30: `experiments/grid/test_30f/`).

## 7. Kiểm tra chạy lại

- **Clone sạch nhánh `feature/target-rate-1pct` vào thư mục mới rồi chạy `pytest`:** 252 passed, 2 skipped.
  - Hai test bị bỏ qua cần `data/processed/feature_matrix_processed.parquet`. File này tải từ Google Drive (README §6), không nằm trong Git.
  - Bản clone không cần file nào ngoài Git để chạy test.
- **Giới hạn:** lần này dùng lại môi trường Python của máy hiện tại, chưa phải máy sạch thật. Cần một bạn cài lại theo README §2 (conda hoặc venv) trên máy khác.
- **README** đã thêm mục "Tuần 5" với lệnh chạy lại từng thí nghiệm.
- **Một lệnh tái lập:** `python scripts/reproduce.py` chạy test, đặc trưng, tiêm 5 run test, lưới bộ 22 rồi so từng chỉ số với bảng benchmark (ngưỡng 1%; LOF 5% vì không tất định). Chưa chạy trên máy sạch.

## Đối chiếu sản phẩm tuần 5 (đề cương, mục 4)

| Sản phẩm | Trạng thái |
|---|---|
| Bảng phân tích theo loại bất thường | ✅ Mục 2 và mục 6 |
| Báo cáo review top 50 | ⏳ Đã có bộ thẻ làm mù và bản chấm của Claude. **Còn thiếu hai bạn chấm độc lập và Cohen's κ** |
| Kết quả ablation | ✅ Mục 4–6, kèm kiểm tra trên test và bảng benchmark cuối |
| Xác nhận chạy lại được trên môi trường mới | ⚠️ Clone sạch chạy được toàn bộ test. Còn thiếu cài trên máy sạch thật |
| Dọn mã, bổ sung README | ✅ README có mục tuần 5, lệnh tái lập một lần (`scripts/reproduce.py`) và bảng 41 đặc trưng v4.1 kèm kiểu dữ liệu và cột bộ 22 |

## Việc còn lại

1. **Review top 50:** hai bạn chấm độc lập.
   - Đọc `docs/huong_dan_review_top50.md` và các thẻ trong `experiments/analysis/review_top50_22f/alert_cards.html` (bộ mới, 22 đặc trưng, 142 cảnh báo).
   - Ghi nhãn vào `review_sheet.csv`.
   - Chạy `python scripts/evaluation/review_agreement.py` để có Cohen's κ, thống nhất các cảnh báo hai bạn chấm khác nhau, rồi cập nhật `review_top50.md`.
2. **Cài lại trên máy sạch** theo README §2, chạy `python scripts/reproduce.py`, rồi ghi `reproduce_check.csv` vào mục 7.
3. **Chờ quyết định:** chuyển 19 đặc trưng bỏ ở K = 18 sang `reserve:` trong `feature_schema.yaml` (schema v4.2, 22 core).
   - Khi đó mọi pipeline dùng bộ 22 mà không cần `drop_features`.
   - Cập nhật luôn bảng đặc trưng trong README.
4. **Đề xuất cho tuần 6** (báo cáo kỹ thuật dùng bảng ở mục 6):
   - đặc trưng fan-out theo nguồn cho spraying;
   - đặc trưng giờ so với lịch sử giờ của chính tài khoản, cho off_hours;
   - làm mượt tỷ lệ khi khối lượng nhỏ, để giảm nhiễu trong top cảnh báo;
   - dò lại k của LOF và số thành phần OCSVM với 22 đặc trưng, nếu còn thời gian.
