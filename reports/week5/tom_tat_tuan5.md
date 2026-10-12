# Tuần 5: Tóm tắt những gì cần đọc (12/10/2026)

Ba việc đã làm: thí nghiệm độ dài cửa sổ lịch sử, phân tích theo loại bất thường, review top 50 cảnh báo.
Mỗi mục gồm kết luận chính và file chi tiết cần đọc.

## 1. Cửa sổ lịch sử 7 / 14 / 21 ngày → giữ 7 ngày

Thí nghiệm chạy trên một run tiêm test (test_seed20261061), mỗi cửa sổ 5 model seed.

| Mô hình | PR-AUC 7 / 14 / 21 ngày | P@100/ngày 7 / 14 / 21 |
|---|---|---|
| OCSVM | 0,165 / 0,169 / 0,160 | 0,27 / 0,27 / 0,25 |
| IF | 0,066 / 0,063 / 0,059 | 0,14 / 0,13 / 0,13 |
| LOF | 0,043 / 0,027 / 0,020 | 0,10 / 0,03 / 0,00 |

- **14 ngày:** khác 7 ngày chưa tới 1 độ lệch chuẩn.
- **21 ngày:** giảm P@k của mọi mô hình ML, tốn thời gian gấp ~3 lần và tốn nhiều RAM hơn.
- **Đánh đổi theo kịch bản:** cửa sổ dài giúp password spraying (ROC-AUC 0,69 → 0,77) nhưng hại dormant wakeup.
- **Phát hiện phụ:** thứ tự dòng trong ma trận thay đổi giữa các lần tính và làm lệch kết quả IF/OCSVM tới 0,027 PR-AUC. Các script mới đều sắp dòng cố định.

📄 Đọc: `experiments/grid/window_ablation/README.md`

## 2. Mô hình nào bắt tốt loại nào → OCSVM là mô hình chính, có 2 điểm mù

Chỉ số chính là **recall trong 100 cảnh báo/ngày** (5 run test, cấu hình chốt từ dev):

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

📄 Đọc sau (đã lưu lại, chưa cần đọc ngay):
- `reports/week5/review_top50.md`: báo cáo review, bản của Claude (một người chấm).
- `experiments/analysis/review_top50/claude_review.csv`: nhãn và lý do của Claude cho từng cảnh báo.
  ⚠️ **Chỉ mở sau khi hai bạn đã tự chấm xong**, để giữ tính độc lập. Tương tự với `answer_key.csv`.

## Việc còn lại của tuần 5

1. **Review top 50:** hai bạn chấm độc lập.
   - Đọc `docs/huong_dan_review_top50.md` và các thẻ trong `experiments/analysis/review_top50/alert_cards.html`.
   - Ghi nhãn vào `review_sheet.csv`.
   - Chạy `python scripts/evaluation/review_agreement.py` để có Cohen's κ, thống nhất các cảnh báo hai bạn chấm khác nhau, rồi cập nhật `review_top50.md`.
2. **Ablation theo nhóm đặc trưng** (bỏ từng nhóm, đo đóng góp): chỉ cần 10 ma trận đặc trưng sẵn có.
3. **Dọn mã, bổ sung README, kiểm tra chạy lại trên máy sạch.**
4. **Đề xuất cho tuần sau:**
   - đặc trưng fan-out theo nguồn cho spraying;
   - đặc trưng giờ tách khỏi khối lượng cho off_hours;
   - làm mượt tỷ lệ khi khối lượng nhỏ, để giảm nhiễu trong top cảnh báo.
