# Hướng dẫn review thủ công top 50 cảnh báo (tuần 5)

Đề cương mục "review top 50": hai thực tập sinh chấm **độc lập** 50 cảnh báo điểm cao nhất của mỗi mô hình,
sau đó đối chiếu và đo mức đồng thuận. Lớp đánh giá này bù cho điểm yếu của nhãn tổng hợp: nó cho biết cảnh báo
mô hình đưa ra trên **log thật** có đáng để analyst mở ra xem hay không.

## Bộ dữ liệu

Tạo bằng `python scripts/evaluation/build_review_set.py` → `experiments/analysis/review_top50/`:

| File | Dùng để |
|---|---|
| `alert_cards.html` | Đọc thẻ từng cảnh báo (mở bằng trình duyệt, tìm theo `alert_id`) |
| `review_sheet.csv` | Ghi nhãn (Excel / Google Sheets). Mỗi người một bản, chỉ điền cột của mình |
| `answer_key.csv` | Cảnh báo nào của mô hình nào, hạng mấy. **Không mở trước khi chấm xong** |

- Log gốc, **không tiêm**; phân khúc User; mô hình fit trên ngày 1–42 với cấu hình chốt từ dev; chấm ngày 43–60.
- Top 50 của OCSVM, IF, LOF được hợp lại (cảnh báo trùng chỉ chấm một lần) và **xáo thứ tự**, ẩn tên mô hình
  và điểm, để người chấm không bị ảnh hưởng bởi "mô hình tốt nhất nói thế".
- Thẻ gồm: số sự kiện hôm nay so với 7 ngày trước và trung vị train, Source/LogHost mới, LogonType, lý do thất bại,
  phân bố theo giờ, tiến trình, 5 đặc trưng lệch nhất so với train (đỏ = vượt phân vị 99,9% train), 12 sự kiện mẫu.

## Ba nhãn

**a — Đáng ngờ, cần điều tra.** Có dấu hiệu khớp một kỹ thuật tấn công và chưa có lời giải thích lành tính:
- thất bại dồn dập (nhiều `Unknown user name or bad password`, khoá tài khoản) rồi có đăng nhập thành công;
- đăng nhập tới nhiều LogHost **mới** trong thời gian ngắn (di chuyển ngang), nhất là LogonType 3/10;
- tài khoản người dùng đột nhiên dùng LogonType lạ với nó (10 RemoteInteractive, 9 NewCredentials, NTLM thay Kerberos);
- hoạt động vào giờ tài khoản chưa từng hoạt động, kèm Source mới;
- tài khoản im lặng lâu rồi hoạt động mạnh từ Source chưa từng thấy.

**b — Bất thường nhưng giải thích được bằng nghiệp vụ.** Lệch khỏi thường lệ nhưng có nguyên nhân lành tính hợp lý:
- hành vi giống tài khoản dịch vụ / tác vụ định kỳ (LogonType 4/5, tiến trình `services.exe`/`svchost.exe`,
  nhịp đều, nhiều LogHost cố định);
- khối lượng lớn nhưng **cùng kiểu** với các ngày trong train (đặc trưng lệch là do mức, không phải do hành vi mới);
- quay lại sau nghỉ với Source/LogHost **quen** (máy cũ của chính người đó);
- vài thất bại lẻ tẻ do gõ sai rồi thành công ngay, cùng máy quen;
- đổi một máy trạm mới đơn lẻ (cấp máy mới), không kèm dấu hiệu khác.

**c — Nhiễu, không mang thông tin.** Điểm cao do tính toán chứ không do hành vi:
- rất ít sự kiện (1–3) làm tỷ lệ cực đoan (`failure_ratio = 1` với 1 sự kiện, entropy = 0…);
- tài khoản mới xuất hiện / ít lịch sử nên mọi thứ đều "mới" (cold start, đặc trưng 7 ngày bị điền median);
- giá trị lỗi/thiếu (Source `Unknown`/null chiếm hết, thời gian trùng giây hàng loạt do lỗi ghi log);
- thẻ không cho thấy điều gì khác thường dù đặc trưng lệch.

Khi phân vân giữa a và b: chọn **a** nếu cần thêm thông tin ngoài log để loại trừ tấn công; chọn **b** nếu chính
log đã đủ cho thấy hành vi lành tính. Ghi lý do ngắn (1 câu) vào cột ghi chú cho **mọi** cảnh báo.

## Quy trình

1. Mỗi người tải `review_sheet.csv` thành bản riêng, chỉ điền `nhan_nguoi_1` / `ghi_chu_1` (hoặc `_2`) bằng `a`, `b`, `c`.
   **Không trao đổi** trong lúc chấm.
2. Gộp hai cột vào một file, chạy
   `python scripts/evaluation/review_agreement.py --sheet <file gộp>` → Cohen's κ, ma trận bất đồng.
3. Ngồi cùng nhau xem lại các cảnh báo bất đồng, điền `nhan_thong_nhat` + `ghi_chu_thong_nhat`.
4. Chạy lại `review_agreement.py` → tỷ lệ a/b/c trong top 50 của từng mô hình (dùng `answer_key.csv`).

Đọc κ: < 0,4 yếu (định nghĩa nhãn chưa rõ → sửa hướng dẫn); 0,4–0,6 trung bình; 0,6–0,8 khá; > 0,8 rất tốt.
