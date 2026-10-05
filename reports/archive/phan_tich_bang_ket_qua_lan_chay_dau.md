# Phân tích bảng kết quả **lần chạy đầu tiên** (Tuần 3)

> **Nguồn số liệu:** `experiments/logs/experiment_log.csv` — lần chạy `2026-09-29 23:46:23`, commit `124de89a`,
> `n_features = 16`, `seed = 42`, `split_day = 42`, `contamination = 0,05` (ngân sách 5%),
> `SimpleImputer(median)` + `RobustScaler`, test = 333.671 dòng (ngày 43–60).
> Bảng trích sẵn: [`bang_ket_qua_lan_chay_dau_3_mo_hinh.csv`](../week3/bang_ket_qua_lan_chay_dau_3_mo_hinh.csv).

## 1. Bảng kết quả lần chạy đầu (5 mô hình)

| Mô hình | n_fit | Alert rate | Lệch ngân sách | Fit (s) | Chấm điểm (s) | Dòng/giây khi chấm | Ngưỡng | Đuôi trên (p95 − median) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 721.612 | 5,9349 % | **+0,93 pp** | 11,43 | 1,97 | 169.290 | ≈ 0 (−8,3e−17) | 0,212 |
| **Z-score Baseline** | 721.612 | 5,8537 % | **+0,85 pp** | 2,63 | 0,12 | 2.827.720 | 23,79 | 25,320 |
| One-Class SVM | 20.000 | 6,3880 % | +1,39 pp | 3,36 | **16,57** | **20.137** | +1,46e−4 | 32,898 |
| Local Outlier Factor | 20.000 | 9,2360 % | +4,24 pp | 2,23 | 4,90 | 68.027 | −0,0542 | 0,620 |
| Rule-Threshold | 721.612 | **20,3512 %** | **+15,35 pp** | 1,15 | 0,09 | 3.791.716 | 0,5 | 0,167 |

## 2. Bảy nhận xét định lượng

### 2.1. Xếp theo "bám ngân sách": Z-score ≈ IF < OCSVM < LOF ≪ Rule
Nhưng **đây KHÔNG phải bảng xếp hạng chất lượng**: ở lần chạy đầu, baseline **đơn giản nhất (Z-score)** lại bám ngân sách tốt nhất (+0,85 pp). Đó chính là bằng chứng củng cố kết luận Tuần 3: *không thể xếp hạng mô hình bằng chỉ số label-free*. Cột "lệch ngân sách" đo **độ trôi hiệu chuẩn**, không đo năng lực phát hiện.

### 2.2. Rule-Threshold trùng khít 20,3512% ở **cả hai** lần chạy (16 và 24 đặc trưng)
Điểm luật chỉ có **7 giá trị rời rạc** (0, 1/6, …, 1) ⇒ ngưỡng phân vị 95% buộc rơi vào 0,5 ⇒ mọi dòng thoả ≥3/6 luật đều bị kêu. Con số **không đổi** khi thêm 8 đặc trưng vì `rule_threshold_baseline` vẫn chỉ dùng **6 luật trên 6 đặc trưng cũ** ⇒ bộ đặc trưng mới **chưa được baseline luật khai thác** (việc cần làm: tie-break hoặc bổ sung luật cho đặc trưng mới).

### 2.3. Nút cổ chai hiệu năng là **chấm điểm của One-Class SVM**, không phải fit
`OCSVM`: fit 3,36 s nhưng **chấm điểm 16,57 s** cho 333.671 dòng ≈ **20.137 dòng/s** — chậm hơn IF **8,4×**, chậm hơn Z-score **140×** (RBF SVM phải tính kernel với ~10³ support vector cho **mỗi** dòng). `Isolation Forest` thì ngược lại: fit đắt nhất (11,43 s cho 150 cây trên 721.612 dòng) nhưng chấm điểm rẻ (1,97 s).
⇒ Khi chạy ablation nhiều cấu hình, OCSVM sẽ là nút cổ chai; cân nhắc giảm `nu`/tăng `tol`/giảm mẫu fit.

### 2.4. Ngưỡng **không so sánh được** giữa các mô hình
`IF ≈ 0` · `LOF = −0,0542` · `OCSVM = +1,46e−4` · `Z = 23,79` · `Rule = 0,5` — mỗi mô hình một định nghĩa điểm và một thang đo. Cột `threshold` chỉ để **tái lập**, so sánh ngưỡng giữa mô hình là vô nghĩa (đó là lý do repo có `score_rank_pct`).

### 2.5. Hai "họ" phân phối điểm rất khác nhau
Đuôi trên (`p95 − median`): IF 0,212 · LOF 0,620 · **OCSVM 32,9** · **Z 25,3** (với `p99/median ≈ 34×`).
⇒ IF/LOF cho phân phối "mềm" (ngưỡng phân vị ổn định), còn OCSVM/Z có **đuôi rất nặng** ⇒ alert rate của chúng **nhạy** với vài trăm dòng ở đuôi. Đây giải thích vì sao Z-score là mô hình trôi mạnh nhất khi bộ đặc trưng thay đổi (xem §3).

### 2.6. Chi phí tăng theo bộ đặc trưng là chuyện bình thường, nhưng Z bị ảnh hưởng mạnh nhất
Xem §3: 3 mô hình học máy **bám ngân sách tốt hơn** khi lên 24 đặc trưng, còn Z-score **xấu đi +3,72 pp**.

### 2.7. Giới hạn của mọi kết luận trên
Không có nhãn ⇒ **không** nói được mô hình nào bắt đúng. Các nhận xét trên chỉ thuộc 3 nhóm: **hiệu chuẩn**, **chi phí**, **ổn định**. Muốn kết luận về năng lực phát hiện phải chờ `redteam.txt` (Tuần 4).

## 3. So với lần chạy hiện tại (24 đặc trưng, commit `1d2209d`)

| Mô hình | Alert rate (16) | Alert rate (24) | Chênh lệch | Xu hướng |
|:---|---:|---:|---:|:---|
| Isolation Forest | 5,9349 % | 5,7053 % | **−0,23 pp** | tốt hơn |
| One-Class SVM | 6,3880 % | 5,8345 % | **−0,55 pp** | tốt hơn |
| Local Outlier Factor | 9,2360 % | 8,0936 % | **−1,14 pp** | tốt hơn nhiều nhất |
| **Z-score Baseline** | 5,8537 % | 9,5762 % | **+3,72 pp** | **xấu đi rõ** |
| Rule-Threshold | 20,3512 % | 20,3512 % | 0,00 pp | không đổi (không dùng đặc trưng mới) |

**Đọc ra sao:** 8 đặc trưng mới (đặc biệt `volume_robust_z_7d`, `new_*_count_7d`, `days_since_last_activity`) có phân phối **không dừng** giữa 42 ngày train và 18 ngày test ⇒ ngưỡng phân vị học trên train không chuyển nguyên sang test. Mô hình **đa biến** (IF/LOF/OCSVM) chịu được vì tín hiệu dàn trên 24 chiều; mô hình **đơn biến** (Z-score, `agg = max`) bị đẩy lên ngay khi một chiều dịch chuyển.

## 4. Việc nên làm rút ra từ bảng này

| # | Việc | Lý do (bám vào số nào) |
|:--|:---|:---|
| 1 | **Tie-break cho Rule** (ví dụ cộng z-score) hoặc báo cáo Rule theo ngân sách | p95 = 0,5 trùng ngưỡng ⇒ 20,35% cảnh báo; hiện không so sánh được với mô hình khác |
| 2 | Bổ sung **luật mới** dùng đặc trưng v3.0 (ví dụ `new_host_count_7d ≥ 1`, `days_since_last_activity ≥ 7`) | Rule đứng yên ở cả 2 lần chạy ⇒ chưa khai thác bộ đặc trưng mới |
| 3 | Với Z-score: cân nhắc **rank-Gaussian/quantile** cho cột zero-inflated | Z trôi +0,85 → +4,58 pp; đuôi `p99/median ≈ 34×` |
| 4 | Ghi **chi phí chấm điểm** vào báo cáo như một tiêu chí (không chỉ alert rate) | OCSVM 20.137 dòng/s vs IF 169.290 dòng/s |
| 5 | Luôn nêu rõ "không phải bảng xếp hạng" khi trích bảng này | Z-score (baseline đơn giản) đứng đầu theo lệch ngân sách |
