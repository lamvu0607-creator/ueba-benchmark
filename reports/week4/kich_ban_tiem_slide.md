# Bộ tiêm bất thường tổng hợp: tóm tắt cho slide

Mọi số liệu đo trên run `dev_seed20261043` (ngày 43–51). Cấu hình: `configs/injection.yaml`.

---

## Slide 1: Vì sao phải tiêm, và tiêm thế nào

**Vấn đề:** log LANL không có nhãn tấn công, nên không thể tính PR-AUC hay Recall.
**Giải pháp:** cấy các mẫu tấn công tổng hợp vào phần dữ liệu đánh giá, rồi đo xem mô hình có tìm ra chúng không.

- **Không bịa dữ liệu:** mỗi sự kiện tiêm là bản sao một sự kiện **thật** của train, ưu tiên của chính nạn nhân, nếu thiếu thì của tài khoản cùng loại. Chỉ đổi thời gian, nguồn và máy đích theo kịch bản.
- **Không rò rỉ:** hồ sơ hành vi chỉ học từ train (ngày 1–42), train không bị sửa.
- **Tiêm vào log, không vào đặc trưng:** sau khi tiêm, toàn bộ 39 đặc trưng được tính lại, gồm cả đặc trưng lịch sử 7 ngày.
- **Dev/test tách biệt:** dev (ngày 43–51) để tinh chỉnh, test (ngày 52–60) chỉ chạy một lần.
- **Nạn nhân:** tài khoản người dùng, có ≥ 7 ngày hoạt động trong train, mỗi tài khoản bị tiêm tối đa một lần. Bỏ các (tài khoản, ngày) mà luật ngưỡng đã cảnh báo sẵn trên log gốc, để nhãn không trùng với cảnh báo có sẵn.

---

## Slide 2: Quy trình tiêm (một run = một khối dev hoặc test + một seed)

```text
 Log gốc train (ngày 1–42)          Log gốc khối đánh giá (dev 43–51 / test 52–60)
          │                                         │
   ① Hồ sơ hành vi ───────┐          ② Luật ngưỡng chấm log gốc → bỏ (tài khoản, ngày) đã bị cảnh báo
          │               ▼                         │
   ③ Kho khuôn sự kiện thật ──► ④ Chọn nạn nhân + sinh sự kiện (6 kịch bản, hạn mức 1%)
                                                    │
                    ⑤ Ghi log đã tiêm + nhãn + manifest (kiểm tra tự động)
                                                    │
                    ⑥ Tính lại 39 đặc trưng ──► ⑦ Benchmark: fit trên train, chấm khối đánh giá
```

| Bước | Làm gì | Đầu ra |
|---|---|---|
| ① Hồ sơ | Học từ train: máy nguồn/đích quen, LogonType hay dùng, giờ hoạt động, khoảng nghỉ dài nhất, ngưỡng khoá L = 5 | `data/features/train_profiles/` |
| ② Loại cảnh báo sẵn | Luật ngưỡng chấm log **gốc**; các (tài khoản, ngày) đã bị cờ không được chọn làm nạn nhân | danh sách loại trừ |
| ③ Kho khuôn | Gom sự kiện thật của train làm khuôn để nhân bản (của nạn nhân, hoặc của tài khoản cùng loại) | cache trên đĩa |
| ④ Chọn và sinh | Tính hạn mức (1% → 736 lần), xáo trộn ứng viên theo seed, lần lượt chạy 6 kịch bản. Mỗi tài khoản tối đa 1 lần; ứng viên không dựng được thì bỏ và thử người kế; thiếu thì chia lại cho kịch bản khác | sự kiện tổng hợp |
| ⑤ Ghi run | Gộp sự kiện tiêm vào bản sao log của ngày bị tiêm (log gốc không bị sửa). Kiểm tra: đúng ngày, sau train, không trùng tài khoản, đúng schema | `events_injected/`, `labels.parquet`, `injection_manifest.csv`, `run_config.json` |
| ⑥ Đặc trưng | Trích lại ma trận (tài khoản × ngày) trên log đã tiêm; log **không chứa** cột đánh dấu tiêm, nên mô hình không "nhìn thấy" nhãn | ma trận đặc trưng của run |
| ⑦ Đánh giá | Mô hình chỉ học trên train, chấm khối đánh giá, so với nhãn | `experiments/` |

**Lệnh** (mỗi seed khoảng 25 phút: tiêm 2 phút, đặc trưng 22 phút):

```bash
python main.py --stage inject   --block dev --injection-seed 20261044
python main.py --stage features --events-dir data/injection_runs/dev_seed20261044/events_injected
python scripts/evaluation/run_grid.py   # lưới 3 mô hình × cấu hình × 5 seed
```

---

## Slide 3: Sáu kịch bản (số liệu thực tế)

| Kịch bản (MITRE) | Hành vi được tiêm | Số sự kiện / lần tiêm |
|---|---|---|
| **Brute-force** (T1110.001) | 8–25 lần đăng nhập **thất bại** (4625) từ một máy lạ, dồn trong 5–30 phút. 5 lần sai mật khẩu, sau đó bị **khoá tài khoản** | 8–25 (trung vị 17) |
| **Password spraying** (T1110.003) | Một máy lạ thử 1–3 lần mật khẩu sai trên **5–10 tài khoản** cùng ngày (17 chiến dịch) | 1–3 (trung vị 2) |
| **Đăng nhập ngoài giờ** (T1078) | **Thêm** một chuỗi đăng nhập thật của chính tài khoản vào **0–6 giờ sáng**, chỉ chọn người vốn làm ban ngày (≤ 20% ngoài giờ) | 10–40 (trung vị 24) |
| **Bùng nổ máy trạm mới** (T1078/T1021) | Đăng nhập thành công từ **4–10 máy chưa từng dùng**, trong 0,5–2 giờ, giờ hành chính | 6–23 (trung vị 14) |
| **Ngủ đông thức dậy** (T1078) | Tài khoản **im lặng 7–41 ngày** (trung vị 13), lâu hơn khoảng nghỉ dài nhất của chính nó, bỗng có lại một ngày hoạt động điển hình | 1–31.800 (trung vị 127) |
| **Đổi kiểu đăng nhập** (T1021) | Tài khoản vốn dùng một kiểu (≥ 80%, thường là network) bỗng đăng nhập **RDP (type 10)** hoặc **tại máy (type 2)** chưa từng dùng | 3–10 (trung vị 7) |

4 kịch bản đầu bảng theo thứ tự đề cương là brute-force, ngoài giờ, máy trạm mới, ngủ đông. Spraying và đổi kiểu đăng nhập là **bổ sung**.

---

## Slide 4: Lượng tiêm

| | Giá trị |
|---|---|
| Mục tiêu (mục 5.2 đề cương: 0,5–1%) | **1% số dòng (tài khoản, ngày) User** |
| Mẫu số (dòng User của khối dev trên log gốc) | 72.845 |
| Số lần tiêm = ⌈1% × 72.845 / 0,99⌉ | **736**, chia đều 6 kịch bản (122–123 mỗi kịch bản) |
| Tỷ lệ đạt được | **1,01%**, đủ ở cả 5 run dev (5 seed) |
| Sự kiện log được thêm | 64.428 trên 138,4 triệu sự kiện gốc của 9 ngày (0,05%) |

- **Một nhãn = một dòng (tài khoản, ngày),** không phải một sự kiện. Thêm 25 sự kiện cho một nạn nhân vẫn chỉ là 1 nhãn.
- 88% sự kiện tiêm thuộc kịch bản ngủ đông, vì kịch bản này cấy cả một ngày hoạt động. Spraying chỉ chiếm 0,4%.
- Bản đầu dùng số nạn nhân cố định và chỉ đạt **0,13%** (92 nhãn). Nhóm đã sửa để tiêm theo tỷ lệ.

---

## Slide 5: Giới hạn cần nói rõ

- **Nhãn tổng hợp cho cận trên lạc quan:** tấn công thật kín đáo và đa dạng hơn. Dòng nhãn 0 cũng chưa chắc lành tính.
- **Spraying chưa đồng thời:** một chiến dịch trải 9–23 giờ trong ngày, trong khi spraying thật thường dồn trong vài phút đến vài giờ.
- **Ngủ đông chỉ phát lại một ngày điển hình,** không tạo đợt hoạt động dồn dập. Đây là cách hiểu "thức dậy" khác với đề cương.
- **Ngoài giờ được thêm vào ngày đã có hoạt động ban ngày,** nên tỷ lệ ngoài giờ bị pha loãng. Đây là một lý do khiến không mô hình nào bắt được kịch bản này.
- **Chỉ tiêm vào tài khoản người dùng;** tài khoản máy không được đánh giá.
