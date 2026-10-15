# Tài liệu dự án

Thư mục `docs/` chứa tài liệu **còn dùng**: đề cương, tham chiếu dữ liệu, hướng dẫn chạy.
Báo cáo theo tuần và kết quả thí nghiệm nằm ở [`reports/`](../reports/README.md).

## Đề cương và kiến trúc

| Tài liệu | Nội dung |
|---|---|
| [`tong_quan_de_tai.md`](tong_quan_de_tai.md) | Đề cương đề tài: mục tiêu, phạm vi, kế hoạch 6 tuần, phương pháp đánh giá |
| [`project_structure.md`](project_structure.md) | Cấu trúc thư mục và vai trò từng module |
| [`data_dictionary.md`](data_dictionary.md) | Từ điển 21 trường log LANL (Event 4624/4625) |

## Hướng dẫn chạy

| Tài liệu | Nội dung |
|---|---|
| [`quy_trinh_tiem_log.md`](quy_trinh_tiem_log.md) | Luồng dữ liệu của một run tiêm, từ log gốc đến bảng kết quả |
| [`kiem_tra_trang_thai_bo_tiem.md`](kiem_tra_trang_thai_bo_tiem.md) | Trạng thái bộ tiêm và các giới hạn còn lại |
| [`huong_dan_tiem_bat_thuong_tuan4.md`](huong_dan_tiem_bat_thuong_tuan4.md) | Sáu kịch bản tiêm, cách chạy và kiểm tra tiến độ (mốc tuần 4) |
| [`huong_dan_luoi_benchmark.md`](huong_dan_luoi_benchmark.md) | Chạy lưới 3 mô hình × cấu hình × 5 seed |
| [`huong_dan_bieu_do_ket_qua.md`](huong_dan_bieu_do_ket_qua.md) | Vẽ và đọc biểu đồ kết quả từ điểm đã lưu |
| [`huong_dan_review_top50.md`](huong_dan_review_top50.md) | Review thủ công top 50 cảnh báo và đo đồng thuận |
| [`../scripts/feature_engineering/HUONG_DAN_CHAY.md`](../scripts/feature_engineering/HUONG_DAN_CHAY.md) | Chạy trích xuất đặc trưng trên máy khác |
| [`../demo/README.md`](../demo/README.md) | Web demo chấm một ngày log mới |

## Phân tích đặc trưng (tuần 2)

[`feature_engineering/feature_correlation_review.md`](feature_engineering/feature_correlation_review.md): nhận xét tương quan
của bộ 30 đặc trưng cũ. Bảng và hình do các script trong `scripts/feature_engineering/` ghi vào
`feature_engineering/tables/` và `feature_engineering/figures/`.
