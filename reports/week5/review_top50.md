# Tuần 5: Review thủ công top 50 cảnh báo — bản chấm của Claude (người chấm 1)

> **Trạng thái:** đây là kết quả của **một** người chấm (Claude, chấm theo `docs/huong_dan_review_top50.md`).
> Đề cương yêu cầu hai thực tập sinh chấm **độc lập** rồi đo đồng thuận (Cohen's κ). Bản này là mốc đối chiếu.
> - Nhãn của Claude nằm riêng ở `experiments/analysis/review_top50/claude_review.csv`. **Nên chấm xong trước khi mở file này** để giữ tính độc lập.
> - Sau khi có nhãn người, chạy `scripts/evaluation/review_agreement.py` để có κ và thay bảng dưới bằng nhãn thống nhất.

## Thiết lập

- **Dữ liệu và mô hình:** log gốc, không tiêm; phân khúc User. Mô hình fit trên ngày 1–42 với cấu hình chốt từ dev, seed 42, rồi chấm ngày 43–60.
- **Cảnh báo:** top 50 của OCSVM, IF và LOF, hợp lại thành **137 cảnh báo** khác nhau. Độ trùng giữa các mô hình: OCSVM–IF 5, OCSVM–LOF 8, IF–LOF 0.
- **Cách chấm:** chấm mù, không biết mô hình nào đưa ra cảnh báo. Nhãn chỉ ghép với `answer_key.csv` sau khi đã chấm xong.
- **Căn cứ của mỗi thẻ:** log thô ngày đó (số sự kiện, thất bại, Source/LogHost mới so với 7 ngày, LogonType, lý do thất bại, giờ, tiến trình) và đặc trưng lệch nhất so với train.

Ba nhãn:
- **a** — đáng ngờ, cần điều tra;
- **b** — bất thường nhưng giải thích được;
- **c** — nhiễu.

## Kết quả theo mô hình

| Mô hình | a — đáng ngờ | b — giải thích được | c — nhiễu | Số tài khoản khác nhau |
|---|---|---|---|---|
| **OCSVM** | **19** (11 tài khoản) | 29 (13) | **2** | 23 |
| IF | 19 (19) | 5 (5) | 26 | 50 |
| LOF | 10 (3) | 1 (1) | 39 | 43 |
| Tổng 137 cảnh báo | 35 | 35 | 67 | |

Theo nửa danh sách:
- **OCSVM:** hạng 1–25 có 15 a, 8 b, 2 c; hạng 26–50 có 4 a, 21 b, 0 c.
- **IF:** hạng 1–25 có 11 a, 3 b, 11 c; hạng 26–50 có 8 a, 2 b, 15 c.
- **LOF:** hạng 1–25 có 9 a, 0 b, 16 c; hạng 26–50 có 1 a, 1 b, 23 c.

## Nhận xét

1. **OCSVM gần như không báo nhiễu (2/50)**, và đầu danh sách giàu cảnh báo đáng ngờ nhất (15/25).
   - Đổi lại, cảnh báo dồn vào ít tài khoản: 50 cảnh báo chỉ thuộc 23 tài khoản.
   - Riêng User272607 chiếm 9 cảnh báo. Tài khoản này đăng nhập NTLM 0,1–1,9 triệu lần/ngày tới 600–1.600 máy mới, có hàng nghìn "Unknown user name", và ngày 52 xuất hiện lại sau 7 ngày im lặng.
   - Đây là cảnh báo đáng giá nhất của cả bộ: hoặc là máy quét được phép, hoặc là di chuyển ngang diện rộng. Chỉ log thì không phân biệt được.
   - Nửa dưới của OCSVM chủ yếu là b: tài khoản dịch vụ khối lượng lớn và chuỗi khoá tài khoản do mật khẩu cũ lặp lại hằng ngày. Đó là bất thường thật nhưng analyst sẽ nhanh chóng đưa vào danh sách loại trừ.
2. **IF tìm được nhiều tài khoản đáng ngờ khác nhau nhất (19 tài khoản)**, nhưng một nửa danh sách là nhiễu (26/50).
   - IF bắt tốt các vụ đoán mật khẩu lần đầu xuất hiện, ví dụ 19.695 và 7.783 thất bại trong ngày ở tài khoản trước đó gần như im lặng. Vụ 26.766 thất bại thì do OCSVM bắt.
   - Nhiễu đến từ các ngày có 1–6 sự kiện mà phần lớn thất bại: `failure_ratio = 1` đẩy điểm lên.
   - IF cũng hay báo các tài khoản NTLM bình thường chỉ có vài lần gõ sai.
3. **LOF chủ yếu là nhiễu (39/50).**
   - Cảnh báo LOF là các ngày tương tác 1–3 sự kiện của tài khoản mới (cold start).
   - 10 cảnh báo đáng ngờ của LOF chỉ thuộc 3 tài khoản, trong đó 8 là User272607, trùng với OCSVM.
4. **Hai mẫu nhiễu lặp lại nhiều nhất** gợi ý sửa đặc trưng:
   - **Tỷ lệ trên khối lượng rất nhỏ** (`failure_ratio`, `delta_mean_share_fail_7d`, `interactive_ratio` với 1–6 sự kiện). Nên co tỷ lệ về trung bình khi số sự kiện nhỏ (làm mượt kiểu Bayes) hoặc thêm trọng số theo `log_total_logons`.
   - **Cold start:** tài khoản có ≤ 4 ngày lịch sử thì mọi thứ đều "mới". Nên gắn cờ hoặc tách các tài khoản này khỏi top cảnh báo.
5. **Khớp với phân tích trên nhãn tổng hợp:** OCSVM tốt nhất cả trên dữ liệu tiêm (PR-AUC 0,18) lẫn trên log thật (tỷ lệ nhiễu thấp nhất). IF có ích như mô hình phụ vì tìm được nhiều tài khoản khác nhau hơn.

## Hạn chế

- Một người chấm, không phải chuyên gia vận hành của tổ chức. Nhãn b dựa trên mẫu lặp lại trong log, chưa xác minh với chủ tài khoản.
- Ranh giới a/b của chuỗi khoá tài khoản là quy ước: lần đầu xuất hiện thì xếp a, lặp lại nhiều ngày thì xếp b. Hai người chấm có thể khác nhau ở nhóm này.
- Cần nhãn của hai thực tập sinh để tính κ. Đó là đầu ra bắt buộc theo đề cương.

Lý do cho từng cảnh báo nằm ở cột `ly_do_claude` trong `experiments/analysis/review_top50/claude_review.csv`.
