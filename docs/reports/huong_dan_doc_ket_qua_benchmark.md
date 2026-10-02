# Hướng dẫn đọc kết quả benchmark (dành cho thực tập sinh)

> **Mục đích:** giải thích 6 file trong `experiments/results/` + `experiments/logs/experiment_log.csv` bằng ngôn ngữ đơn giản, kèm 4 phát hiện quan trọng và checklist tự kiểm chứng.
> **Bối cảnh:** kết quả dưới đây là lần chạy trên **schema v3.0 (24 đặc trưng core)**, commit `1d2209d`, ngày chạy `2026-10-02`.

---

## 1. Bức tranh trong 1 phút: benchmark này làm gì?

Hãy tưởng tượng bạn có một cuốn sổ ghi **mỗi dòng = một tài khoản trong một ngày** (gọi là "account-day"), kèm ~24 con số mô tả hành vi của ngày đó (số lần đăng nhập, tỷ lệ thất bại, giờ hoạt động, có dùng nguồn mới không…).

1. **Cắt sổ theo thời gian:** 42 ngày đầu (721.612 dòng) = **tập train**; 18 ngày cuối (333.671 dòng) = **tập test**. Cắt theo *thời gian* chứ không ngẫu nhiên, để mô hình không "nhìn thấy tương lai".
2. **Cho 6 "máy soi bất thường" học quy luật từ train:** Isolation Forest, LOF, One-Class SVM (học máy) + 3 mốc so sánh (Z-score, Luật ngưỡng, **Ngẫu nhiên**).
3. **Chấm điểm từng dòng trong test**, điểm CAO = càng dị thường.
4. **Vì không có nhãn**, quy ước: mỗi mô hình được phép "kêu" đúng **5% số dòng** (= 16.684 dòng) — gọi là **ngân sách cảnh báo**. Ai vượt thì ghi lại mức vượt (`alert_rate_drift_pp`).

> Vì quy ước "kêu 5%" là do ta đặt ra, nên **tỷ lệ cảnh báo 5% KHÔNG có nghĩa là mô hình tốt**. Nó chỉ cho biết mô hình có bám đúng ngân sách hay không. Muốn biết "bắt đúng bao nhiêu" thì **bắt buộc phải có nhãn tấn công thật** (`redteam.txt` — chưa có trong repo).

## 2. File nào nói điều gì?

| File | Trả lời câu hỏi | Đọc ở đâu trước |
|:---|:---|:---|
| `split_info.json` | Chia train/test thế nào? | ✅ Đọc đầu tiên — xác nhận 721.612 / 333.671 dòng, cắt ở ngày 42 |
| `benchmark_summary.csv` | Mỗi mô hình: tỷ lệ cảnh báo, ngưỡng, thời gian chạy, đặc trưng nào, scaler nào | ✅ Bảng chính |
| `anomaly_scores.parquet` | Từng dòng test được chấm bao nhiêu điểm, có bị kêu hay không (5 mô hình × 3 cột `_score`/`_pct`/`_anomaly`) | ✅ Dùng để tự phân tích sâu (theo ngày, theo loại tài khoản…) |
| `model_topk_overlap.csv` | 5 mô hình có "kêu cùng người" không? | Xem khi tự hỏi "nên tin ai" |
| `model_stability.csv` | Đổi seed (ngẫu nhiên) thì kết quả có đổi không? | Xem khi nghi ngờ "kết quả này có phải do may mắn?" |
| `run_manifest.json` | Chạy trên commit nào, dữ liệu có bị đổi không (sha256), thư viện phiên bản nào | Xem khi cần **tái lập** |
| `experiments/logs/experiment_log.csv` | Lịch sử các lần chạy (mỗi dòng = 1 mô hình trong 1 lần chạy) | ⚠️ Đọc kèm cảnh báo ở §6 |

## 3. Từ điển thuật ngữ (đọc 1 lần là hiểu)

| Thuật ngữ | Nghĩa đơn giản | Ví dụ từ chính kết quả này |
|:---|:---|:---|
| **feature (đặc trưng)** | Một con số mô tả hành vi trong ngày, ví dụ "tỷ lệ đăng nhập thất bại" | 24 đặc trưng core |
| **account-day** | Một dòng = 1 tài khoản × 1 ngày | 1.055.283 dòng cho 60 ngày |
| **train / test** | Phần để học / phần để kiểm tra (cắt theo ngày, không ngẫu nhiên) | train 42 ngày, test 18 ngày |
| **contamination / ngân sách cảnh báo** | Tỷ lệ dòng được phép "kêu" | 5% test = 16.684 dòng |
| **alert rate** | Tỷ lệ dòng thực sự bị kêu trong test | IF: 5,71% · LOF: 8,09% · Rule: 20,35% |
| **alert_rate_drift_pp** | Số điểm phần trăm lệch khỏi ngân sách 5% (dương = kêu nhiều hơn cho phép) | IF +0,71 pp · Z-score **+4,58 pp** |
| **imputer** | Cách "lấp chỗ trống" khi một đặc trưng bị thiếu (NULL) | Đang dùng **median** (học từ train) |
| **scaler** | Cách đưa các đặc trưng về cùng thang đo (vì "86.297 giây" và "0,5" không so được với nhau) | Đang dùng **RobustScaler** cho 3 mô hình khoảng cách |
| **Spearman** | Hai bảng xếp hạng giống nhau tới mức nào (1 = giống hệt, 0 = không liên quan, âm = ngược nhau) | IF ↔ Z-score = 0,853 (khá giống) |
| **Top-K overlap** | Trong 20 dòng bị kêu nhiều nhất, hai mô hình trùng bao nhiêu? | LOF ↔ OCSVM = 0,70; **các cặp khác = 0,00** |
| **leakage (rò rỉ dữ liệu)** | Mô hình vô tình "xem trước đáp án" → điểm số đẹp nhưng vô nghĩa | Đã sửa từ Tuần 3 (không fit trên test, scaler chỉ học từ train) |
| **drift (trôi)** | Test khác train tới mức ngưỡng cũ không còn đúng | Xem cột `alert_rate_drift_pp` |
| **precision@K / recall@budget** | Trong K cảnh báo đầu có bao nhiêu đúng / bắt được bao nhiêu phần trăm vụ tấn công thật | **Chưa chạy được vì chưa có nhãn** |
| **ROC-AUC / AP** | Chất lượng xếp hạng tổng thể (AP bền hơn khi dữ liệu cực mất cân bằng) | **Chưa chạy được vì chưa có nhãn** |

## 4. Bốn phát hiện, kể theo cách "đọc bảng ra chuyện gì"

### 4.1. Hạ tầng đo đã đúng (tin được)
`split_info.json` khớp đúng Tuần 3 (721.612 / 333.671) dù đã thêm 8 đặc trưng mới ⇒ chứng tỏ các đặc trưng lịch sử (novelty, 7 ngày, warm-up) **không thêm dòng nào** vào ma trận. Mọi thứ cần để tái lập đều có: ngưỡng, imputer/scaler, seed, phiên bản sklearn, `sha256` dữ liệu.

### 4.2. Các mô hình đang kêu "tài khoản khác loại", không phải "hành vi bất thường"
Số liệu (tôi tính từ `anomaly_scores.parquet`):

| Loại tài khoản | Số dòng | IF | LOF | OCSVM | Z-score |
|:---|---:|---:|---:|---:|---:|
| System | 72 | **100%** | **100%** | **100%** | **100%** |
| Service | 137 | **87,6%** | **83,2%** | **93,4%** | 62,0% |
| Admin | 630 | **87,3%** | 29,4% | 33,8% | 82,2% |
| User | 152.740 | 10,8% | 11,5% | 9,9% | 15,6% |
| Machine | 180.092 | 1,0% | 5,0% | 2,2% | 4,2% |

**Đọc ra sao:** tài khoản System/Service/Admin (chỉ 839 dòng, 0,25%) bị kêu gần như **mọi ngày**, còn tài khoản máy thì hầu như không. Nghĩa là mô hình đang phản ứng với "**loại tài khoản này vốn khác**" chứ chưa hẳn là "ngày này có gì lạ".
**So sánh dễ hiểu:** giống như camera lần đầu thấy xe cứu hỏa thì báo động — vì nó khác xe thường, không phải vì có chuyện gì xảy ra.
**Việc cần làm (đã có trong kế hoạch):** chuẩn hoá theo nhóm đồng đẳng (peer), hoặc tách nhóm tài khoản khi chấm điểm và báo cáo minh bạch.

### 4.3. Ngân sách bị "chiếm chỗ" bởi nhóm tài khoản kêu hoài
Isolation Forest kêu 19.037 dòng nhưng trải trên **8.386 / 27.011 tài khoản (31%)**; **10 tài khoản đứng đầu bị kêu 18/18 ngày test** (mọi ngày!). Với người trực SOC, đây là "tiếng ồn lặp lại" — họ sẽ học cách bỏ qua nó.
**Việc cần làm:** thêm chỉ số vận hành "top-N tài khoản chiếm bao nhiêu % ngân sách" và "số ngày bị kêu liên tiếp".

### 4.4. Chưa thể nói mô hình nào tốt hơn (và đây không phải lỗi của bạn)
* Top-20: chỉ **LOF ↔ OCSVM trùng 0,70**; mọi cặp khác **trùng 0,00** — kể cả IF ↔ Z-score dù Spearman 0,853.
* Có những dòng **cả 5 mô hình cùng kêu**: 462 dòng; nhưng **67% số dòng không mô hình nào kêu** (223.603 dòng).
* Hợp của cả 5 mô hình = **110.068 dòng = 33% test**, trong khi mỗi mô hình chỉ nhắm 5% ⇒ chọn mô hình nào làm khối lượng việc của SOC thay đổi ~6,6 lần.

**Kết luận cần nhớ:** khi không có nhãn, các chỉ số này chỉ nói "các mô hình không đồng ý với nhau", **không** nói mô hình nào bắt đúng. Đây chính là lý do Tuần 4 phải gán nhãn `redteam.txt`.


## 5. Ba cái bẫy hay gặp khi đọc kết quả (đừng mắc)

1. **"Tỷ lệ cảnh báo 5% ⇒ mô hình tốt"** — Sai. 5% là con số **ta tự đặt** (ngân sách). Muốn biết tốt/xấu phải có nhãn.
2. **"Rule baseline chỉ 20,35% cảnh báo nên nó dở"** — Chưa kết luận được. Điểm của luật chỉ có **7 giá trị rời rạc** (`benchmark_summary.csv`: p00=0, p50=0,33, p95=0,5, max=1,0) nên ngưỡng phân vị 95% buộc phải rơi vào 0,5 ⇒ kêu 20,35%. Đây là **hạn chế kỹ thuật đã biết**, không phải bằng chứng về chất lượng phát hiện.
3. **"Mô hình A có Spearman với B = 0,85 nghĩa là hai mô hình giống nhau"** — Ở mức Top-20 thì lại **trùng 0%**. Hai bảng xếp hạng có thể "nhìn chung giống nhau" nhưng **20 người đứng đầu khác hoàn toàn** — mà SOC chỉ đọc 20 dòng đầu.

## 6. ✅ Lỗi "sổ sách" đã sửa (giữ lại để hiểu nguyên nhân)

**Trước khi sửa:**

```text
experiments/results/benchmark_summary.csv  → n_features = 24  (schema v3.0 hiện tại)
experiments/logs/experiment_log.csv        → chỉ có 16 (5 dòng) và 20 (5 dòng), KHÔNG có dòng 24 nào
```

**Vì sao?** Log chống ghi trùng theo khoá `(model_key, seed, split_day, train_samples, test_samples, git_commit)` — **không có `n_features`/`feature_set`**. Lần chạy 24 đặc trưng có cùng commit/seed/split/số dòng với lần 20 đặc trưng ⇒ bị bỏ qua im lặng.

**Đã sửa (2026-10-02):** dựng lại đúng 5 dòng bằng chính hàm `src/evaluation/experiment_log.build_log_row`
(dữ liệu đầu vào lấy từ artifact: `benchmark_summary.csv`, `split_info.json`, `anomaly_scores.parquet`, `run_manifest.json`)
rồi ghi bằng `append_experiment_log(..., dedupe=False)`; `timestamp` của dòng = **thời điểm chạy thật**
(`run_manifest.created_at`), không phải thời điểm vá log.

| Kiểm chứng sau khi sửa | Kết quả |
|:---|:---|
| Số dòng log | **18 → 23** (8 legacy + 5×16 biến + 5×20 biến + **5×24 biến**) |
| Số cột | 30 (giữ nguyên định dạng) |
| Đối chiếu số liệu | phân vị trong log **khớp** phân vị tính lại từ `anomaly_scores.parquet` (sai số < 1e-5) |

**Phòng ngừa tái diễn:** khi đổi bộ đặc trưng mà **không** đổi commit/seed/split, phải gọi
`append_experiment_log(..., dedupe=False)` (hoặc commit mỗi biến thể để `git_commit` khác nhau).
Ngoài ra `main.py` đã sửa dòng log cứng: nay in **số đặc trưng lấy từ schema** + số dòng × số cột,
thay vì chuỗi "16 đặc trưng cốt lõi".


## 7. Checklist tự kiểm chứng (làm được ngay, ~10 phút)

```bash
# 1. Trạng thái chia tập (phải thấy 721612 / 333671, split_day = 42)
type experiments\results\split_info.json

# 2. Bảng chính: 5 mô hình, n_features = 24, alert_rate_pct, alert_rate_drift_pp
type experiments\results\benchmark_summary.csv

# 3. Tự tính tỷ lệ cảnh báo theo loại tài khoản & theo ngày
python scripts\diagnostics\alert_rate_calibration.py
python scripts\diagnostics\day_split_table.py

# 4. Đọc chéo log để đối chiếu 3 bộ đặc trưng (16 / 20 / 24)
type experiments\logs\experiment_log.csv
```

**Ba câu hỏi bạn phải trả lời được sau khi đọc:**
1. Vì sao phải cắt train/test theo **ngày** thay vì ngẫu nhiên? *(vì dữ liệu là chuỗi thời gian; cắt ngẫu nhiên = cho mô hình nhìn thấy tương lai)*
2. Vì sao không thể nói "mô hình nào tốt nhất" từ các file này? *(vì không có nhãn; chỉ số label-free chỉ đo mức đồng thuận/drift, không đo độ đúng)*
3. Nếu chỉ được sửa **một** thứ để kết quả đáng tin hơn, bạn sửa gì? *(gán nhãn `redteam.txt` rồi bật `precision_at_k`, `recall_at_budget`, `roc_auc`, `average_precision`)*

## 8. Nghiệm thu Tuần 3 — 3 tiêu chí (đã kiểm chứng)

### 8.1. Khung benchmark chạy được **đầu cuối** ✅

Lệnh kiểm chứng (chạy trong **sandbox riêng** nên KHÔNG đụng `data/` và `experiments/` thật):

```bash
python main.py --stage all --start-day 1 --end-day 2 --split-day 1 \
  --models isolation_forest local_outlier_factor one_class_svm \
  --config .sandbox/config_sandbox.yaml
```

| Bằng chứng | Kết quả |
|:---|:---|
| Exit code | **0** (không lỗi) |
| Chặng 1 (clean) | `cleaned_day-01.parquet`, `cleaned_day-02.parquet` |
| Chặng 2 (features) | `features/raw/feature_matrix_raw.parquet` (**28 cột**) + `daily/user_features_day-0{1,2}.parquet`; 8 đặc trưng lịch sử tính trong **0,5 s** |
| Chặng 2 (processed) | `processed/feature_matrix_processed.parquet` = **37.690 × 31 cột** |
| Chặng 3 (benchmark) | `results/{benchmark_summary.csv, anomaly_scores.parquet, model_stability.csv, model_topk_overlap.csv, split_info.json, run_manifest.json}` + `models/*.joblib` (3 mô hình) |
| Nhật ký | `logs/experiment_log.csv` (3 dòng mới, đúng 30 cột) |
| Chia tập trong sandbox | train = 17.910 (ngày 1) / test = 19.780 (ngày 2), `split_strategy = time` |
| Impute NULL | **35.820** giá trị = `days_since_last_activity` 17.910 + `volume_robust_z_7d` 17.910 — đúng **toàn bộ** NULL warm-up của bộ 2 ngày |
| Log schema | in số đặc trưng **lấy từ schema** (24) + số dòng × số cột — đã hết chuỗi cứng "16 đặc trưng" |

> ⚠️ **Con số của sandbox KHÔNG dùng để kết luận** (chỉ 2 ngày; ví dụ LOF alert rate 75,70% là hiệu ứng mẫu nhỏ). Sandbox chỉ chứng minh **khung chạy được đầu cuối**.

### 8.2. Bảng kết quả **lần chạy đầu tiên** cho 3 mô hình ✅

Artifact: [`reports/week3/bang_ket_qua_lan_chay_dau_3_mo_hinh.csv`](../../reports/week3/bang_ket_qua_lan_chay_dau_3_mo_hinh.csv) (3 dòng × 20 cột, trích từ `experiment_log.csv` — lần chạy `2026-09-29 23:46:23`, `n_features = 16`):

| Mô hình | n_fit | n_eval | Alert rate | Lệch ngân sách | Fit (s) | Chấm điểm (s) | Ngưỡng |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 721.612 | 333.671 | **5,9349 %** | +0,93 pp | 11,431 | 1,971 | −0,000000 |
| Local Outlier Factor | 20.000 | 333.671 | **9,2360 %** | +4,24 pp | 2,226 | 4,905 | −0,054215 |
| One-Class SVM | 20.000 | 333.671 | **6,3880 %** | +1,39 pp | 3,355 | 16,570 | +0,000146 |

Chung cho cả 3: `seed = 42`, `split_day = 42`, `contamination = 0,05`, `feature_set = core`,
`imputer = SimpleImputer`, `scaler = RobustScaler`, commit `124de89a`.

### 8.3. Nhật ký thí nghiệm ✅

`experiments/logs/experiment_log.csv` — **23 dòng** = 8 legacy + 5 (16 biến) + 5 (20 biến) + 5 (**24 biến**), 30 cột, có cột truy vết (`n_features`, `feature_set`, `imputer`, `scaler`, `threshold`, `alert_rate_pct`, `sklearn_version`, `git_commit`).
Chi tiết lần chạy 24 biến bị khử trùng nay đã được ghi bù: xem §6.

### 8.4. Baseline ngẫu nhiên — mốc dưới (bổ sung sau Tuần 3) ✅

`RandomBaseline` (`--models random_baseline`) chấm điểm bằng **băm ngẫu nhiên tất định**: cùng dòng ⇒ cùng
điểm, đảo thứ tự dòng không đổi kết quả, và **không** có tương quan với dữ liệu. Chạy thật trên ma trận
**24 đặc trưng** (ghi vào thư mục tạm, không đụng artifact):

| Chỉ số | RandomBaseline | Đối chiếu |
|:---|---:|:---|
| Alert rate (test) | **4,96 %** | lệch **−0,04 pp** so ngân sách 5 % |
| Ngưỡng | **0,9502** | ≈ phân vị 95 % của phân phối đều |
| Phân vị điểm (p25 · p50 · p75 · p95 · p99) | 0,250 · 0,500 · 0,750 · 0,950 · 0,990 | khớp **lý thuyết phân phối đều** |
| Fit / chấm điểm | 2,49 s / 0,364 s | rẻ hơn IF (11,1 s / 2,0 s) |
| Scaler | `None` | không cần scale (đúng thiết kế) |

**Ba kết luận rút ra:**

1. **Hạ tầng ngưỡng đúng**: mô hình "đoán mò" cho alert rate gần như chính xác bằng ngân sách ⇒ cơ chế
   `threshold = quantile(score_train, 1 − contamination)` hoạt động như thiết kế.
2. **Alert rate KHÔNG phải chỉ số chất lượng**: trên cùng dữ liệu, RandomBaseline lệch −0,04 pp còn
   Z-score lệch **+4,58 pp** ⇒ nếu xếp hạng theo "độ bám ngân sách" thì mô hình đoán mò sẽ thắng.
3. **Từ Tuần 4 (có nhãn)**, mọi mô hình phải được so với mốc này: kỳ vọng `ROC-AUC ≈ 0,5` và
   `AP ≈ tỷ lệ dương tính`; nếu một mô hình "thật" không vượt được thì kết quả không có ý nghĩa.

## 9. Tóm tắt 5 dòng cho người bận

1. Chia tập và quy trình đã đúng chuẩn (72 vạn train / 33 vạn test, không rò rỉ).
2. Isolation Forest bám ngân sách tốt nhất (5,71% so với mức cho phép 5%).
3. Nhưng cảnh báo đang tập trung vào các tài khoản **User** (86,8% khối lượng của IF) trong khi tài khoản **máy** chỉ nhận 9,3% (lift 0,17× — bị bỏ sót); tài khoản System/Service/Admin bị kêu gần như mọi ngày ⇒ cần chuẩn hoá theo nhóm/chuẩn hoá theo chính tài khoản.
4. 5 mô hình **không đồng thuận** ở mức quan trọng nhất (Top-20) ⇒ chưa chọn được mô hình.
5. Việc tiếp theo bắt buộc: **gán nhãn `redteam.txt`** (Tuần 4); 2 lỗi sổ sách **đã sửa** (§6).

