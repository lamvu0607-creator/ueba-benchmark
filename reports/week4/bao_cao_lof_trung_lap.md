# Báo cáo: LOF — điểm ~1e10 do dòng trùng lặp & alert rate 15,8%

> **Ngày:** 2026-10-05  
> **Phạm vi:** `HNSWLOF` (`src/models/pyod_detectors.py`), mô hình `local_outlier_factor` trong benchmark  
> **Dữ liệu:** `data/processed/feature_matrix_processed.parquet`, 39 đặc trưng core (schema v4.0), train = ngày 1–42 (721.612 dòng), test = ngày 43–60 (333.671 dòng)  
> **Tái lập:** `scripts/diagnostics/lof_duplicate_sweep.py` (mục 2–4), `scripts/diagnostics/lof_alert_rate_diagnosis.py` (mục 6)

## Tóm tắt

| Vấn đề | Nguyên nhân | Trạng thái |
|---|---|---|
| `score_p100` của LOF ≈ **2,0e10** | 94 dòng train có k-distance = 0 (nhóm trùng > k bản sao) ⇒ chia cho epsilon 1e-10 | **Đã sửa** bằng `dedup: true` ⇒ p100 = **12,03** |
| Alert rate test của LOF ≈ **15,8%** (ngân sách 5%) | Không phải do trùng lặp. LOF rất nhạy với **trôi theo thời gian**: láng giềng của một dòng train thường **cùng ngày** với nó, còn dòng test không bao giờ có láng giềng cùng ngày | **Đã xác định nguyên nhân**, chưa sửa (đề xuất ở mục 7) |

---

## 1. Hiện tượng

Benchmark ngày 2026-10-05 (trước khi sửa), `experiments/results/benchmark_summary.csv`:

| model | threshold | alert_rate_pct | score_p50 | score_p99 | score_p100 |
|---|---|---|---|---|---|
| local_outlier_factor | 1,3807 | 15,83 | 1,163 | 2,196 | **2,02e10** |

LOF chuẩn có giá trị ≈ 1 với điểm bình thường; 2e10 ≈ 1 / 1e-10 là dấu hiệu chia cho epsilon.

## 2. Nguyên nhân p100 ≈ 1e10

`HNSWLOF` dùng đúng công thức sklearn: `lrd(x) = 1 / (mean reach(x, o) + 1e-10)`. Với ma trận UEBA
zero-inflated, train có **3.621 dòng thuộc nhóm trùng hoàn toàn**, trong đó **4 nhóm lớn hơn k = 20**
(nhóm lớn nhất 27 bản sao). Mỗi dòng trong nhóm này có 20 láng giềng ở khoảng cách 0 ⇒
k-distance = 0 ⇒ lrd ≈ 1e10 ⇒ mọi điểm (train hoặc test) có láng giềng là các dòng này nhận LOF ~1e9–1e10.

Đây là hành vi đúng của công thức (sklearn cũng vậy), nhưng điểm này **không mang thông tin**: độ lớn
của nó chỉ phản ánh epsilon.

## 3. Thử nghiệm: thêm nhiễu vi mô (jitter)

Thêm nhiễu Gauss σ vào train (sau RobustScaler) trước khi fit:

| σ | k-distance = 0 | p100 train | p100 test | p99 test | alert test | Spearman vs gốc | Trùng Top-5% |
|---|---|---|---|---|---|---|---|
| 0 | 94 | 2,77e10 | 2,02e10 | 2,24 | 15,82% | 1 | 100% |
| 1e-6 | 0 | 2,67e5 | 1,95e5 | 2,25 | 15,79% | 0,994 | 97,8% |
| 1e-5 | 0 | 2,70e4 | 1,97e4 | 2,21 | 15,76% | 0,992 | 97,0% |
| 1e-4 | 0 | 2.700 | 1.966 | 2,23 | 15,83% | 0,992 | 97,2% |
| 1e-3 | 0 | 270,6 | 196,7 | 2,20 | 15,99% | 0,990 | 95,5% |
| 1e-2 | 0 | 24,6 | 18,3 | 2,22 | 15,80% | 0,989 | 94,9% |
| 1e-1 | 0 | 18,9 | 12,0 | 1,78 | 12,70% | 0,812 | 73,5% |

**Kết luận:** σ = 1e-6 đã xoá hết k-distance = 0, nhưng **p100 ≈ 0,2 / σ**: nhiễu không sửa lỗi mà chỉ
thay epsilon bằng σ, và giá trị đó do người chọn σ quyết định. σ = 0,1 thì bắt đầu làm méo mô hình
(Spearman 0,81). Không chọn hướng này.

## 4. Thử nghiệm: sàn k-distance và gộp dòng trùng

| Biến thể | Điểm index | p100 train | p100 test | p99 test | alert test | Spearman vs gốc | Trùng Top-5% |
|---|---|---|---|---|---|---|---|
| gốc (`dedup=False`) | 721.612 | 2,77e10 | 2,02e10 | 2,24 | 15,82% | 1 | 100% |
| sàn `min_k_distance = 1e-3` | 721.612 | 2.774 | 2.020 | 2,23 | 15,84% | 0,992 | 97,4% |
| sàn `min_k_distance = 1e-2` | 721.612 | 314,8 | 202,1 | 2,22 | 15,75% | 0,994 | 97,6% |
| **`dedup`** | **719.235** | **18,98** | **12,03** | 2,19 | 15,97% | 0,984 | 92,2% |
| `dedup` + sàn 1e-2 | 719.235 | 20,32 | 15,31 | 2,20 | 15,85% | 0,984 | 93,4% |

* **Sàn k-distance** có cùng nhược điểm với nhiễu: p100 tỉ lệ nghịch với giá trị sàn (1e-3 → 2.020, 1e-2 → 202),
  và giá trị sàn phụ thuộc thang đo của scaler.
* **Dedup** (index chỉ chứa dòng duy nhất) không có tham số: k-distance luôn > 0 một cách tự nhiên, p100 về
  mức có nghĩa (12–19), và fit nhanh hơn (ít điểm hơn). Thêm sàn lên trên dedup hầu như không đổi gì.
* Thứ hạng gần như giữ nguyên. Một phần chênh lệch Spearman/Top-5% là do `add_items` đa luồng của HNSW
  (chạy lại cùng cấu hình p99 cũng lệch 2,19–2,24).

**Lưu ý ngữ nghĩa của dedup:** k láng giềng là k điểm **phân biệt** gần nhất; một nhóm 27 bản sao được
coi như một điểm. Với UEBA điều này hợp lý: nhiều thực thể-ngày giống hệt nhau là cùng một hành vi, không phải
"mật độ vô hạn".

## 5. Thay đổi đã áp dụng

* `HNSWLOF` thêm 2 tham số:
  * `dedup: bool = True`: `np.unique` trên dữ liệu float32, dựng index trên dòng duy nhất, mọi bản sao nhận
    điểm của dòng duy nhất tương ứng (`decision_scores_` vẫn đủ `n` dòng ⇒ cách đặt ngưỡng không đổi).
    Số điểm index ở `n_index_points_` / `get_metadata()["n_index_points"]`.
  * `min_k_distance: float = 0.0`: sàn k-distance (tắt). Chỉ dùng khi `dedup=False` hoặc khi gặp cụm gần-trùng.
* `configs/model_params.yaml`: `dedup: true`, `min_k_distance: 0.0`.
* Test (`tests/test_models.py`): `test_hnswlof_dedup_removes_epsilon_blowup` (dedup khớp với LOF fit trực tiếp
  trên dòng duy nhất, bản sao cùng điểm, không còn điểm > 100) và `test_hnswlof_min_k_distance_floor`. Test cũ về
  nhóm trùng chạy với `dedup=False`.

Benchmark chạy lại sau khi sửa (cấu hình mặc định):

| | Trước | Sau |
|---|---|---|
| threshold | 1,3807 | 1,3803 |
| alert_rate_pct | 15,83 | 15,85 |
| score_p50 | 1,1634 | 1,1632 |
| score_p99 | 2,196 | 2,240 |
| **score_p100** | **2,02e10** | **12,03** |
| fit_seconds | 50,1 | 40,5 |
| Ổn định 3 seed (Spearman mean / min) | — | 0,992 / 0,990 |

Năm mô hình còn lại cho kết quả không đổi.

---

## 6. Điều tra alert rate 15,8%

Ngưỡng = phân vị 95% điểm train ⇒ kỳ vọng ~5% cảnh báo trên test. IF ra 5,9%, OCSVM 6,5%, nhưng LOF ra 15,8%.

### 6.1. Theo ngày và theo thực thể (`anomaly_scores.parquet`)

| Ngày | 43 | 44 | 45 | 46 | 47 | **48** | **49** | 50 | 51 | 52 | 53 | 54 | **55** | **56** | 57 | 58 | 59 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LOF % | 8,7 | 9,9 | 12,9 | 13,4 | 16,0 | **23,0** | **25,0** | 11,8 | 14,8 | 12,0 | 25,6 | 17,4 | **24,1** | **25,7** | 14,8 | 14,2 | 14,0 | 13,2 |
| IF % | 8,7 | 7,3 | 5,9 | 5,1 | 4,1 | 2,2 | 1,8 | 5,8 | 6,4 | 6,3 | 7,1 | 5,7 | 5,8 | 5,9 | 7,2 | 5,7 | 5,9 | 6,6 |

* LOF đã vượt ngân sách ngay ngày 43 (8,7%), tăng dần theo thời gian, và cao nhất vào cuối tuần (48–49, 55–56),
  trong khi IF lại *giảm* vào cuối tuần.
* Thực thể đã thấy trong train: 15,8% (6.569 dòng thực thể mới: 18,8%) ⇒ **không phải do thực thể mới**.
* Machine 10,7%, User 21,9%.

### 6.2. Thí nghiệm có đối chứng (`lof_alert_rate_diagnosis.py`)

| Thí nghiệm | Fit | Đánh giá | LOF alert | IF alert |
|---|---|---|---|---|
| `random_holdout` (80/20 dòng ngẫu nhiên) | 1–42 | 1–42 | **5,16%** | 4,94% |
| `time_holdout` | 1–35 | 36–42 | **17,84%** | 7,98% |
| `baseline` (= benchmark) | 1–42 | 43–60 | 15,84% | 5,94% |
| `no_warmup` (bỏ 7 ngày đầu) | 8–42 | 43–60 | 15,82% | 5,68% |
| `no_warmup_time` | 8–35 | 36–42 | 19,51% | 7,27% |

Loại trừ được 3 giả thuyết:

1. **Điểm train leave-self-out vs test out-of-sample:** `random_holdout` cho 5,16% ⇒ cách chấm và đặt ngưỡng trên
   train **không sai** khi không có yếu tố thời gian.
2. **NULL của các đặc trưng `*_7d` trong 7 ngày đầu** (16,8% dòng train bị median impute, test 0%): bỏ 7 ngày đầu
   không đổi gì (15,82%).
3. **Trôi phân phối toàn cục:** KS lớn nhất giữa train và test chỉ là 0,24 (`missing_source_ratio`), các đặc trưng
   còn lại < 0,12, và IF (mô hình toàn cục) chỉ lệch ~1 điểm %.

⇒ Nguyên nhân là **trôi thời gian ở mức cục bộ**, và LOF nhạy hơn IF rất nhiều (17,8% vs 8,0% chỉ với 1 tuần).

### 6.3. Cơ chế: láng giềng cùng ngày

Láng giềng (k = 20, leave-self-out) của dòng train, so với láng giềng chọn ngẫu nhiên:

| Khoảng cách ngày | Láng giềng thật | Ngẫu nhiên |
|---|---|---|
| \|Δday\| = 0 (cùng ngày) | **28,0%** | 2,5% |
| \|Δday\| ≤ 1 | 40,4% | 7,2% |
| \|Δday\| ≤ 3 | 50,9% | 16,1% |
| \|Δday\| ≤ 7 | 65,0% | 32,5% |

Các dòng cùng ngày giống nhau hơn hẳn: chúng chia chung "dấu vân tay của ngày" qua các đặc trưng cửa sổ trượt 7 ngày
(`novelty_share_pair_7d`, `new_source_count_7d`, `jaccard_source_7d`…), đặc trưng so với nhóm ngang hàng trong ngày
(`peer_z_*`) và các tỉ lệ có xu hướng theo thời gian (`missing_source_ratio`, `remote_logon_ratio`: |Spearman với day|
≈ 0,2). Vì vậy:

* Dòng train luôn có "anh em cùng ngày" trong index ⇒ trông đặc ⇒ LOF thấp ⇒ **ngưỡng bị kéo thấp**.
* Dòng test không bao giờ có láng giềng cùng ngày trong train ⇒ khoảng cách tới láng giềng gần nhất (trung vị)
  1,48 so với 0,92 của train ⇒ LOF cao hơn có hệ thống.
* `random_holdout` vẫn có anh em cùng ngày trong 80% còn lại ⇒ 5,16%.
* Cuối tuần có dấu vân tay riêng và ít dòng hơn ⇒ xa train hơn ⇒ alert cao nhất.

Giả thuyết "mỗi thực thể tạo một cụm riêng" bị bác bỏ: chỉ 2,8% láng giềng của dòng train là cùng thực thể.

### 6.4. Thử sửa nhanh: chấm train kiểu "leave-day-out"

Truy vấn 120 láng giềng, bỏ những láng giềng cách ≤ w ngày, giữ 20 láng giềng hợp lệ đầu tiên để tính
k-distance / lrd / LOF của train (bản thử nghiệm ngoài repo, `dedup=False`):

| Cách | Ngưỡng | Alert test | Alert theo ngày | Dòng thiếu láng giềng hợp lệ |
|---|---|---|---|---|
| Hiện tại (leave-self-out) | 1,382 | 15,76% | 8,6–25,8% | — |
| A: chỉ đổi ngưỡng, w = 0 | 1,402 | 14,32% | | 8,1% |
| B: mật độ train + ngưỡng leave-day-out, w = 0 | 1,402 | 11,10% | 4,9–21,0% | 8,1% |
| **B: w = 1** | 1,408 | **8,86%** | 4,5–17,3% | 11,3% |
| B: w = 3 | 1,417 | 9,59% | 4,0–18,8% | 17,2% |

Leave-day-out kéo alert rate từ 15,8% xuống ~8,9% ⇒ hiệu ứng cùng ngày giải thích **khoảng 2/3 độ lệch**. Phần còn lại
(~4 điểm %, ngang mức lệch của IF ở `time_holdout`) là trôi thời gian thật. Hạn chế của thử nghiệm: 8–17% dòng không
đủ 20 láng giềng hợp lệ trong 120 kết quả (khi đó dùng tạm láng giềng cùng ngày), và chạy `dedup=False` nên p100 vẫn ~1e10.

## 7. Đề xuất

1. **Giữ `dedup: true`** (đã áp dụng).
2. **Alert rate của LOF**, theo thứ tự ưu tiên:
   * **Đặt ngưỡng trên một cửa sổ kiểm định theo thời gian**: fit ngày 1–35, ngưỡng = phân vị 95% điểm ngày 36–42.
     Cách này khớp đúng điều kiện test và áp dụng chung cho mọi mô hình. Đổi lại, ngưỡng chỉ dựa trên 7 ngày, nên cần
     quyết định có fit lại trên 1–42 hay không.
   * **Leave-day-out trong `HNSWLOF`** (tham số kiểu `exclude_same_group`, truyền cột `day` vào lúc fit): cần truy vấn
     nhiều láng giềng hơn hoặc lọc trong lúc tìm kiếm (hnswlib hỗ trợ `filter`) để không thiếu láng giềng, và phải đưa
     `day` qua `AnomalyPipeline`.
   * **Xem lại đặc trưng mang dấu vân tay ngày** (cửa sổ 7 ngày, `peer_z_*`, đặc trưng có xu hướng theo `day`) trong
     lần cập nhật schema tiếp theo. Cách này cũng giúp các mô hình khác.
3. Khi báo cáo kết quả, nên ghi rõ alert rate thực của LOF là ~16%, không phải 5%: so Top-K giữa các mô hình vẫn công bằng,
   nhưng so precision/recall theo ngưỡng thì LOF đang dùng ngân sách gấp 3 lần.
