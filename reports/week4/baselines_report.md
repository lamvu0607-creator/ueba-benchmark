# Báo cáo: ba baseline so với IForest / LOF / OCSVM

Ngày chạy: 2026-10-08 · Lệnh: `python main.py --stage baselines` · Cấu hình: `configs/baselines.yaml`

## 1. Nguyên tắc

- **Không gán ngưỡng tùy ý.** Mọi ngưỡng và sàn đều tính từ dữ liệu train theo một quy tắc viết sẵn trong config.
- **Không nhìn nhãn.** Baseline chỉ fit trên train không nhãn, giống ba mô hình ML.
- **Cùng điều kiện với ML:**
  - train = ngày 1–42, đánh giá = ngày 43–60;
  - fit riêng hai phân khúc Machine và User;
  - dùng đúng bộ 39 đặc trưng core;
  - giá trị NULL thay bằng median của train.
- **Mỗi baseline cho một điểm liên tục** để tính PR-AUC giống hệt ML.
- **Điểm vận hành chung:** ngưỡng = phân vị 0,99 của chính điểm số đó trên train (ngân sách 1%). Dòng bị gắn cờ khi điểm lớn hơn hẳn ngưỡng.

## 2. Ba baseline

| Baseline | Điểm của một dòng | Cột kết quả |
|---|---|---|
| Ngẫu nhiên | Số ngẫu nhiên đều, seed cố định. PR-AUC kỳ vọng = tỷ lệ dòng bất thường | `random` |
| Z-score robust | max theo đặc trưng của \|x − median\| / (1,4826·MAD), tính trên 39 đặc trưng core | `zscore_global`, `zscore_account` |
| Luật ngưỡng | max qua 6 luật của tỷ lệ dòng train có giá trị nhỏ hơn hẳn dòng đang xét (ECDF trái) | `rule_ecdf`, `rule_external` |

Sáu luật, mỗi luật ứng với một kịch bản tấn công:

| Luật | Thống kê | Lấy từ đâu |
|---|---|---|
| R1 | Số lần đăng nhập thất bại (`failure_count`) | `total_logons × failure_ratio` |
| R2 | Số tài khoản bị cùng một Source gây thất bại trong ngày (dấu hiệu spraying) | Log 4625; dùng thông tin chéo giữa các tài khoản |
| R3 | Số sự kiện ngoài giờ | `total_logons × off_hours_ratio` |
| R4 | Số Source mới so với lịch sử của tài khoản | Log, so với mọi ngày trước đó |
| R5 | `days_since_last_activity` | Đặc trưng core có sẵn |
| R6 | Số sự kiện có LogonType chưa từng thấy ở tài khoản | Log, so với mọi ngày trước đó |

Các quy tắc đã chốt:

- **Sàn MAD q = 0,25.** Khi MAD của một đặc trưng bằng 0, thay bằng phân vị 25% của các độ lệch khác 0 trên train. Ban đầu dùng q = 0,01, nhưng sàn quá nhỏ (khoảng 1e-5) làm |z| lên tới hàng nghìn và rất nhiều dòng trùng điểm ở đỉnh.
- **Dung sai số học 1e-9.** Các đặc trưng `delta_mean_*` có độ lệch cỡ 1e-16 do sai số làm tròn số thực. Không có dung sai này thì |z| lên tới khoảng 1e15.
- **Z-score theo tài khoản:** cần ít nhất 7 ngày train; tài khoản ít hơn thì dùng thống kê toàn cục.
- **Biến thể ngưỡng ngoài (`rule_external`):**
  - R1 ≥ L, với L = 5 ước lượng từ log train (673 lần khóa tài khoản; trung vị số lần thất bại trước khi bị khóa là 5, khoảng tứ phân vị 4–10);
  - R2 ≥ 30 theo mốc của Splunk.

## 3. Kết quả không cần nhãn (ngân sách 1%)

Tỷ lệ cờ trên train → trên tập đánh giá, kèm số cảnh báo trung bình mỗi ngày (giá trị lớn nhất trong ngoặc).

| Phương pháp | Machine | User |
|---|---|---|
| `random` | 1,00% → 1,00% · 100/ngày (123) | 1,00% → 0,98% · 84/ngày (112) |
| `zscore_global` | 0,84% → 0,75% · 75/ngày (91) | 1,00% → 1,17% · 99/ngày (133) |
| `zscore_account` | 1,00% → 1,01% · 101/ngày (191) | 0,99% → 1,59% · 135/ngày (201) |
| `rule_ecdf` | 1,00% → 1,23% · 123/ngày (229) | 1,00% → 1,28% · 109/ngày (271) |
| `rule_external` | 0,95% → 1,06% · 106/ngày (178) | **0% → 0% · 0/ngày** |

## 4. Phát hiện

1. **`rule_external` không gắn cờ dòng nào ở phân khúc User.**
   - Hơn 1% dòng train của User đã đạt R1 (failure_count ≥ 5) hoặc R2. Các dòng này cùng nhận điểm 1,0, nên ngưỡng train cũng bằng 1,0, và không dòng nào lớn hơn hẳn ngưỡng.
   - Kết luận: ngưỡng khóa L = 5 quá thấp để dùng làm luật cảnh báo cho tài khoản người dùng. Kết quả được giữ nguyên để báo cáo.
2. **`zscore_account` ở User gắn cờ trên tập đánh giá nhiều hơn trên train** (1,59% so với 0,99%). Hành vi của từng tài khoản trong giai đoạn đánh giá lệch khỏi lịch sử train của chính nó.
3. **`rule_ecdf` có số cảnh báo dao động mạnh giữa các ngày** (tối đa 229–271/ngày, so với trung bình khoảng 110–120). Mọi giá trị vượt max train đều nhận điểm 1,0 nên nhiều dòng trùng điểm ở đỉnh.

## 5. Kiểm chứng

- `tests/test_baselines.py`: 22/22 test pass.
- PR-AUC ngẫu nhiên (trung bình 40 seed) khớp tỷ lệ dòng bất thường.
- Không rò rỉ train/test:
  - không hàm `fit` nào nhận tham số nhãn;
  - phân tích mã nguồn xác nhận các module baseline không đọc nhãn;
  - thay đổi tập đánh giá (kể cả cực đoan) không làm đổi tham số, điểm train hay ngưỡng;
  - điểm của một dòng không phụ thuộc các dòng đánh giá khác.

## 6. Việc còn lại

- Chạy lại với `--labels runs/<id>/labels.parquet --events-dir runs/<id>/events_injected` khi đã có nhãn tiêm. Khi đó sẽ có:
  - PR-AUC theo từng kịch bản, dùng chung một tập dòng âm;
  - precision và recall tại ngân sách 1%;
  - recall theo `campaign_id`.
- Đưa ba mô hình ML vào cùng phần đánh giá. Để tính ngưỡng theo ngân sách, benchmark cần lưu thêm điểm train của ML.

Artifact: `experiments/results/baselines/` gồm `baseline_scores.parquet` và `thresholds.json`.
