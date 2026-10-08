# Trạng thái bộ tiêm log và đánh giá theo run

Bộ tiêm đã có `operations.py`, `scenarios.py`, `inject.py`, cấu hình dev/test và kiểm thử tự động.

| Thành phần | Trạng thái |
|---|---|
| Hồ sơ train | Chỉ ngày 1–42; giữ nguyên log train |
| Sinh sự kiện | Nhân bản log thật, schema interim 21 cột, kiểm tra timestamp đúng ngày |
| Kịch bản | Brute force, spraying, ngoài giờ, máy nguồn mới, ngủ đông, đổi LogonType |
| Điều phối | Mỗi tài khoản tối đa một lần/run; xuất overlay, bảng phụ, manifest và nhãn |
| Phạm vi đánh giá | Dev 43–51, test 52–60; ML và baseline dùng cùng khối |
| Cấu hình run | `run_config.json` lưu block, seed, split_day và đầy đủ eval_days |
| Nhãn | Đọc chung `is_anomaly`/`label`; khóa trùng báo lỗi; áp dụng `eval_exclude` |
| Ngủ đông | Đọc sự hiện diện mọi ngày sau train đến ngày tiêm, kể cả khối trước |
| Oracle | Chỉ `NullOracle`; chưa chấm độ khó thực tế |

## Cách chạy

Xem `quy_trinh_tiem_log.md`. `--events-dir <run>/events_injected` chọn ma trận, nhãn và đầu ra của run;
không đè artifact log gốc. Features giữ toàn bộ lịch sử; đánh giá chỉ dùng ngày của khối.
Run cũ dùng manifest + cấu hình injection để xác định đầy đủ khối, không suy từ các ngày có tiêm.

## Kiểm chứng và giới hạn

- Kiểm thử hồi quy bao phủ nhãn âm rõ ràng, dòng loại khỏi đánh giá, phạm vi ngày dev/test,
  hoạt động trước khối test làm ngắt ngủ đông, và thiếu log không bị coi là im lặng.
- Chạy `python -m pytest tests -q` để kiểm tra trên môi trường hiện tại. Kết quả test không thay thế
  việc chạy pipeline trên dữ liệu thật và kiểm tra nhãn khớp ma trận.
- Ngoài giờ dùng replay: chuỗi dài hơn cửa sổ sẽ bị bỏ; cần báo cáo số ứng viên bỏ qua.
- Chỉ tiêm User và một ngày/lần; Machine chưa có nhãn tổng hợp từ bộ tiêm này.
- Loại ứng viên đã bị luật ECDF cảnh báo là chọn mẫu có điều kiện theo baseline; cần ghi rõ trong báo cáo.
- Log gốc không có ground truth: nhãn 0 nghĩa là không được tiêm, chưa chứng minh lành tính thật.
- Mọi bước của run dùng tầng interim; so sánh với đối chứng cũng phải dùng cùng tầng, tránh khác biệt DST.
- Tham số chỉ chỉnh trên dev; test dùng cấu hình đã khóa, không chọn tham số theo kết quả test.
