# Bộ tiêm bất thường tổng hợp: tóm tắt cho slide

Số liệu đo trên 5 run dev (ngày 36–42) và 5 run test (ngày 43–60), bản 2026-10-09. Cấu hình: `configs/injection.yaml`.

---

## Slide 1: Vì sao phải tiêm, và tiêm thế nào

**Vấn đề:** log LANL không có nhãn tấn công, nên không thể tính PR-AUC hay Recall.
**Giải pháp:** cấy các mẫu tấn công tổng hợp vào phần dữ liệu đánh giá, rồi đo xem mô hình có tìm ra chúng không.

- **Không bịa dữ liệu:** mỗi sự kiện tiêm là bản sao một sự kiện **thật** của train, ưu tiên của chính nạn nhân, nếu thiếu thì của tài khoản cùng loại. Chỉ đổi thời gian, nguồn và máy đích theo kịch bản.
- **Không rò rỉ:** hồ sơ hành vi chỉ học từ train của khối (dev: ngày 1–35, test: ngày 1–42), train không bị sửa.
- **Tiêm vào log, không vào đặc trưng:** sau khi tiêm, toàn bộ 41 đặc trưng được tính lại, gồm cả đặc trưng lịch sử 7 ngày.
- **Dev/test tách biệt:** test = ngày 43–60 (30% thời gian, đề cương 5.1), chỉ chạy một lần; dev để tinh chỉnh nằm trong train (ngày 36–42).
- **Nạn nhân:** tài khoản người dùng, có ≥ 7 ngày hoạt động trong train, mỗi tài khoản bị tiêm tối đa một lần. Bỏ các (tài khoản, ngày) mà luật ngưỡng đã cảnh báo sẵn trên log gốc, để nhãn không trùng với cảnh báo có sẵn.

---

## Slide 2: Quy trình tiêm (một run = một khối dev hoặc test + một seed)

```text
 Log gốc train (dev: 1–35 / test: 1–42)   Log gốc khối đánh giá (dev 36–42 / test 43–60)
          │                                         │
   ① Hồ sơ hành vi ───────┐          ② Luật ngưỡng chấm log gốc → bỏ (tài khoản, ngày) đã bị cảnh báo
          │               ▼                         │
   ③ Kho khuôn sự kiện thật ──► ④ Chọn nạn nhân + sinh sự kiện (6 kịch bản, hạn mức 1%)
                                                    │
                    ⑤ Ghi log đã tiêm + nhãn + manifest (kiểm tra tự động)
                                                    │
                    ⑥ Tính lại 41 đặc trưng ──► ⑦ Benchmark: fit trên train, chấm khối đánh giá
```

| Bước | Làm gì | Đầu ra |
|---|---|---|
| ① Hồ sơ | Học từ train: máy nguồn/đích quen, LogonType hay dùng, giờ hoạt động, khoảng nghỉ dài nhất, ngưỡng khoá L = 5 | `data/features/train_profiles/` |
| ② Loại cảnh báo sẵn | Luật ngưỡng chấm log **gốc**; các (tài khoản, ngày) đã bị cờ không được chọn làm nạn nhân | danh sách loại trừ |
| ③ Kho khuôn | Gom sự kiện thật của train làm khuôn để nhân bản (của nạn nhân, hoặc của tài khoản cùng loại) | cache trên đĩa |
| ④ Chọn và sinh | Tính hạn mức (1% → 553 lần ở dev, 1.543 ở test), xáo trộn ứng viên theo seed, lần lượt chạy 6 kịch bản. Mỗi tài khoản tối đa 1 lần; ứng viên không dựng được thì bỏ và thử người kế; thiếu thì chia lại cho kịch bản khác | sự kiện tổng hợp |
| ⑤ Ghi run | Gộp sự kiện tiêm vào bản sao log của ngày bị tiêm (log gốc không bị sửa). Kiểm tra: đúng ngày, sau train, không trùng tài khoản, đúng schema | `events_injected/`, `labels.parquet`, `injection_manifest.csv`, `run_config.json` |
| ⑥ Đặc trưng | Trích lại ma trận (tài khoản × ngày, 41 đặc trưng) trên log đã tiêm; log **không chứa** cột đánh dấu tiêm, nên mô hình không "nhìn thấy" nhãn | ma trận đặc trưng của run |
| ⑦ Đánh giá | Mô hình chỉ học trên train, chấm khối đánh giá, so với nhãn | `experiments/` |

**Lệnh** (mỗi seed khoảng 25 phút: tiêm 2–4 phút, đặc trưng 21 phút):

```bash
python main.py --stage inject   --block dev --injection-seed 20261037
python main.py --stage features --events-dir data/injection_runs/dev_seed20261037/events_injected
python scripts/evaluation/run_grid.py   # lưới 3 mô hình × cấu hình × 5 seed
```

---

## Slide 3: Sáu kịch bản (số liệu thực tế)

| Kịch bản (MITRE) | Hành vi được tiêm | Số sự kiện / lần tiêm |
|---|---|---|
| **Brute-force** (T1110.001) | 8–25 lần đăng nhập **thất bại** (4625) từ một máy lạ, dồn trong 5–30 phút. 5 lần sai mật khẩu, sau đó bị **khoá tài khoản** | 8–25 (trung vị 18) |
| **Password spraying** (T1110.003) | Một máy lạ thử 1–3 lần mật khẩu sai trên **5–13 tài khoản**, cả chiến dịch dồn trong **30–90 phút** | 1–3 (trung vị 2) |
| **Ngoài giờ** (T1078) | **Dời toàn bộ một ngày hoạt động** điển hình của người làm ban ngày (≤ 20% ngoài giờ) sang **0–7 giờ sáng** | trung vị ~110–150 |
| **Bùng nổ máy trạm mới** (T1078/T1021) | Đăng nhập thành công từ **4–10 máy chưa từng dùng**, trong 0,5–2 giờ, giờ hành chính | 6–23 (trung vị 14) |
| **Ngủ đông thức dậy** (T1078) | Tài khoản **im lặng** lâu hơn khoảng nghỉ dài nhất của chính nó, bỗng có lại một ngày hoạt động điển hình (giữ nguyên giờ) | trung vị ~135 |
| **Đổi kiểu đăng nhập** (T1021) | **Cả một ngày** đăng nhập chuyển sang loại chưa từng dùng: ưu tiên **tại máy (2) → dịch vụ (5)** như đề cương; LANL chỉ có ~35 tài khoản như vậy nên phần lớn là **mạng (3) → RDP (10)** | trung vị ~200 |

4 kịch bản đầu bảng theo thứ tự đề cương là brute-force, ngoài giờ, máy trạm mới, ngủ đông. Spraying và đổi kiểu đăng nhập là **bổ sung**.

---

## Slide 4: Lượng tiêm

| | Giá trị |
|---|---|
| Mục tiêu (mục 5.2 đề cương: 0,5–1%) | **1% số dòng (tài khoản, ngày) User** |
| Mẫu số (dòng User trên log gốc) | dev 54.656 (7 ngày) · test 152.740 (18 ngày) |
| Số lần tiêm = ⌈1% × D / 0,99⌉ | dev **553** (92–93 mỗi kịch bản) · test **1.543** (257–258 mỗi kịch bản) |
| Tỷ lệ đạt được | **1,01%** ở cả 10 run |
| Sự kiện log được thêm | dev 62–130 nghìn · test 157–198 nghìn (dưới 0,1% sự kiện gốc) |

- **Một nhãn = một dòng (tài khoản, ngày),** không phải một sự kiện. Thêm 25 sự kiện cho một nạn nhân vẫn chỉ là 1 nhãn.
- Phần lớn sự kiện tiêm thuộc 3 kịch bản cấy cả một ngày (ngủ đông, ngoài giờ, đổi kiểu đăng nhập). Spraying chỉ chiếm phần rất nhỏ.
- Bản đầu dùng số nạn nhân cố định và chỉ đạt **0,13%** (92 nhãn). Nhóm đã sửa để tiêm theo tỷ lệ.

---

## Slide 5: Giới hạn cần nói rõ

- **Nhãn tổng hợp cho cận trên lạc quan:** tấn công thật kín đáo và đa dạng hơn. Dòng nhãn 0 cũng chưa chắc lành tính.
- **Đã sửa theo đề cương (2026-10-09):** spraying trước trải 9–23 giờ (lỗi code); ngoài giờ trước chỉ thêm vài sự kiện đêm;
  đổi kiểu đăng nhập trước chỉ thêm 3–10 sự kiện và ngược chiều; ngủ đông trước bị dời giờ ngẫu nhiên (lỗi code).
  Sau khi sửa, ROC-AUC của OCSVM: ngoài giờ 0,49 → 0,91, đổi kiểu đăng nhập 0,44 → 0,99.
- **Chiều 2 → 5/3 của đề cương chỉ có 5–8 nạn nhân mỗi run** vì log thiếu tài khoản chủ yếu đăng nhập tại máy.
- **Spraying là kịch bản khó nhất cho mô hình** (OCSVM 0,70): đặc trưng tính theo từng tài khoản không thấy "một nguồn, nhiều tài khoản".
- **Chỉ tiêm vào tài khoản người dùng;** tài khoản máy không được đánh giá.
