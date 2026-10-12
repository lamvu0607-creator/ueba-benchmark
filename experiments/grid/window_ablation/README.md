# Thí nghiệm độ dài cửa sổ lịch sử (7 / 14 / 21 ngày)

Câu hỏi: các đặc trưng lịch sử `*_7d` (novelty, jaccard, dist_shift, delta_mean, `new_*_count_7d`,
`volume_robust_z_7d`) so ngày t với 7 ngày trước. Cửa sổ dài hơn có phát hiện tốt hơn không?

## Thiết lập

- **Dữ liệu:** một run tiêm `test_seed20261061` (test, ngày 43–60, fit trên ngày 1–42). Tiêm lại cùng seed vì
  `events_injected/` đã bị xoá; ma trận 7 ngày tính lại khớp bản gốc (sai khác ≤ 3,6·10⁻¹⁴, do làm tròn).
- **Cửa sổ N ngày:** chỉ đổi số ngày, giữ tên cột `*_7d` để schema và lưới không đổi
  (`compute_history(hist_days=N)`, `add_history_features(window=N)`).
- **Mô hình:** cấu hình chốt từ dev (`configs/benchmark_grid_test.yaml`), không chọn lại.
  5 model seed (42, 7, 2024, 1, 2) trên **cùng** run tiêm → std chỉ đo nhiễu mô hình, không gồm nhiễu tiêm.
- **Thứ tự dòng được sắp cố định** theo (day, DomainName, UserName). Polars ghi ma trận với thứ tự dòng khác nhau mỗi lần,
  và IF (lấy mẫu con) / OCSVM (SGD) phụ thuộc thứ tự dòng: cùng dữ liệu nhưng OCSVM seed 42 ra 0,156 thay vì 0,183.
  Vì vậy mốc so sánh là cột 7 ngày dưới đây, **không** phải 0,182 của lưới test chính.
- Quét contamination chỉ giữ mức 5% (bản đầy đủ tốn ~120 lần fit mỗi cửa sổ).

Chạy lại: `python -X utf8 scripts/experiments/window_ablation.py --run data/injection_runs/test_seed20261061 --windows 7 14 21`
(ma trận ở `data/window_ablation/w{N}/`, kết quả ở `w{N}/`, bảng gộp `window_comparison.csv`, `window_scenarios.csv`).

## Kết quả 7 vs 14 ngày (2026-10-12)

| Mô hình | PR-AUC 7 ngày | PR-AUC 14 ngày | ROC-AUC 7 → 14 | P@100/ngày 7 → 14 |
|---|---|---|---|---|
| OCSVM | 0,165 ± 0,011 | 0,169 ± 0,012 | 0,915 → 0,925 | 0,27 → 0,27 |
| IF | 0,066 ± 0,004 | 0,063 ± 0,002 | 0,885 → 0,885 | 0,14 → 0,13 |
| LOF | 0,043 ± 0,002 | 0,027 ± 0,001 | 0,680 → 0,633 | 0,10 → 0,03 |
| z-score | 0,031 | 0,033 | 0,738 → 0,739 | 0,06 → 0,07 |
| Luật thất bại | 0,025 | 0,025 | 0,625 → 0,625 | 0,08 → 0,08 |
| Luật 6 điều kiện | 0,014 | 0,014 | 0,533 → 0,533 | 0,05 → 0,05 |
| Ngẫu nhiên | 0,010 | 0,010 | 0,50 → 0,50 | 0,01 → 0,01 |

ROC-AUC theo kịch bản (7 → 14 ngày):

| Kịch bản | OCSVM | IF | LOF |
|---|---|---|---|
| brute_force | 0,993 → 0,993 | 0,893 → 0,908 | 0,429 → 0,236 |
| password_spraying | 0,690 → **0,739** | 0,784 → 0,794 | 0,526 → 0,463 |
| off_hours | 0,911 → 0,917 | 0,859 → 0,859 | 0,746 → 0,720 |
| new_workstation_burst | 0,974 → 0,978 | 0,902 → 0,907 | 0,905 → 0,910 |
| dormant_wakeup | 0,940 → 0,939 | 0,908 → 0,884 | 0,594 → 0,672 |
| logon_type_switch | 0,985 → 0,983 | 0,960 → 0,958 | 0,883 → 0,797 |

Nhận xét:

1. **OCSVM:** PR-AUC +0,004, nhỏ hơn std giữa các seed (~0,011) → chưa đủ để nói 14 ngày tốt hơn.
   Tăng rõ duy nhất ở password spraying (+0,05 ROC-AUC): máy bị rải mật khẩu "mới" hơn khi so với 14 ngày.
2. **IF:** không đổi. Khớp phân tích tuần 4: IF đo độ hiếm trong phạm vi train, ít nhạy với cách định nghĩa "mới".
3. **LOF:** giảm mạnh (PR-AUC 0,043 → 0,027), nhất là brute_force và logon_type_switch. Giả thuyết (chưa kiểm):
   warm-up dài hơn → ngày train 1–14 có `delta_mean_*`/`active_days` NULL, điền median → cụm dày giả trong train,
   LOF (dựa mật độ) nhạy nhất với kiểu này.
4. Luật và z-score gần như không đổi (luật không dùng đặc trưng lịch sử).
5. Chi phí: bước ghép lịch sử 14 ngày mất ~200 s và tốn RAM gần gấp đôi 7 ngày (bảng `int_ranges` nhân theo N);
   lần đầu chạy 14 ngày bị dừng vì máy thiếu RAM.

**Kết luận tạm:** chưa có lý do đổi 7 → 14 ngày. Lợi ích đáng kể duy nhất là password spraying, nên giải bằng đặc trưng
fan-out theo nguồn (kế hoạch tuần 5). Kết quả chỉ trên 1 run tiêm; muốn khẳng định cần chạy đủ 5 run test.

## Kết quả 7 / 14 / 21 ngày (2026-10-12)

| Mô hình | PR-AUC 7 | PR-AUC 14 | PR-AUC 21 | ROC-AUC 7 / 14 / 21 | P@100/ngày 7 / 14 / 21 |
|---|---|---|---|---|---|
| OCSVM | 0,165 ± 0,011 | **0,169 ± 0,012** | 0,160 ± 0,009 | 0,915 / 0,925 / **0,929** | **0,27** / **0,27** / 0,25 |
| IF | **0,066 ± 0,004** | 0,063 ± 0,002 | 0,059 ± 0,002 | 0,885 / 0,885 / 0,882 | 0,14 / 0,13 / 0,13 |
| LOF | **0,043 ± 0,002** | 0,027 ± 0,001 | 0,020 ± 0,001 | 0,680 / 0,633 / 0,660 | 0,10 / 0,03 / 0,00 |
| z-score | 0,031 | 0,033 | **0,040** | 0,738 / 0,739 / 0,733 | 0,06 / 0,07 / 0,07 |
| Luật thất bại | 0,025 | 0,025 | 0,025 | 0,625 | 0,08 |
| Luật 6 điều kiện | 0,014 | 0,014 | 0,014 | 0,533 | 0,05 |
| Ngẫu nhiên | 0,010 | 0,010 | 0,010 | 0,50 | 0,01 |

ROC-AUC theo kịch bản, OCSVM (7 / 14 / 21): password_spraying 0,690 / 0,739 / **0,770**; off_hours 0,911 / 0,917 / 0,929;
new_workstation_burst 0,974 / 0,978 / 0,979; brute_force 0,993 / 0,993 / 0,991; logon_type_switch 0,985 / 0,983 / 0,979;
dormant_wakeup 0,940 / 0,939 / **0,924**. IF: password_spraying 0,784 → 0,806, dormant_wakeup 0,908 → **0,850**.

Nhận xét:

1. **Đánh đổi theo kịch bản, không có cửa sổ thắng tuyệt đối.** Cửa sổ dài giúp các kịch bản "mới so với lịch sử"
   (password spraying +0,08, ngoài giờ +0,02 ROC-AUC ở OCSVM) nhưng làm hại ngủ đông (OCSVM −0,016, IF −0,058):
   cửa sổ càng dài thì khoảng ngủ đông càng dễ nằm trong cửa sổ, tài khoản "thức dậy" bớt nổi bật.
2. **OCSVM:** ROC-AUC tăng đều (0,915 → 0,929) nhưng PR-AUC và P@k (phần đầu danh sách — thứ analyst thực sự xem)
   đạt đỉnh ở 14 ngày rồi giảm ở 21 ngày. Mọi chênh lệch PR-AUC đều trong khoảng 1 std.
3. **IF và LOF giảm đều theo N**; LOF gần như mất khả năng xếp hạng ở 21 ngày (P@100/ngày 0,00).
   Phù hợp với giả thuyết warm-up: ngày train 1–N không có `delta_mean_*` (điền median) → N càng lớn, phần train
   bị "làm phẳng" càng nhiều (21/42 ngày ở N = 21).
4. **Chi phí tăng nhanh:** bước ghép lịch sử mất ~200 s ở 14 ngày và ~570 s ở 21 ngày, RAM tăng theo N.

**Kết luận:** giữ cửa sổ 7 ngày. 14 ngày không tốt hơn rõ ràng (trong 1 std), 21 ngày làm giảm P@k của mọi mô hình ML
và tốn gấp ~3 lần thời gian. Lợi ích thật của cửa sổ dài chỉ nằm ở password spraying — nên thêm đặc trưng
riêng cho kịch bản này (fan-out theo nguồn, hoặc novelty 21–28 ngày cho riêng thực thể máy đích) thay vì kéo dài
cửa sổ cho cả 41 đặc trưng. Không chạy 28 ngày: xu hướng PR-AUC/P@k đã đi xuống từ 21 ngày, warm-up sẽ chiếm 2/3 train.
