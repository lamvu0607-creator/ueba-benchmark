# Báo cáo tóm tắt Tuần 3 — Giao diện mô hình thống nhất & Benchmark label-free

**Trạng thái:** ✅ hoàn thành · nhánh `feature/model-benchmark-framework` @ `fffb9e1` (mốc mã + artifact: `07c85a7`, tag `features-v1`) · **31/31 test pass** · working tree sạch
**Tài liệu chi tiết kèm toàn bộ bằng chứng:** [`label_free_benchmark.md`](label_free_benchmark.md)

---

## 1. TL;DR (5 dòng)

1. Đã dựng **một giao diện duy nhất** (`BaseAnomalyModel` + `AnomalyPipeline`) cho **5 mô hình**: 3 thuật toán sklearn + 2 baseline label-free.
2. Đã **sửa lỗi rò rỉ dữ liệu**: benchmark nay chia theo **thời gian** — train = ngày 1–42 (**721.612** dòng), test = ngày 43–60 (**333.671** dòng); imputer/scaler chỉ fit trên train.
3. Chạy thật 60 ngày trong **168 giây** cho cả 5 mô hình + 3 seed kiểm tra ổn định; alert rate **5,85–9,24%** với 3/5 mô hình (2 baseline có lý do riêng, xem §5).
4. Đã **phục hồi thiết kế bị mất** (`model_factory` + 12 test) từ `.pytest_cache` và `experiment_log.csv`; artifact kết quả **đã được commit vào git** để không mất lần nữa.
5. Kết luận quan trọng nhất: **chỉ số label-free không đủ để xếp hạng mô hình** — Top-20 của 5 mô hình gần như không trùng nhau (overlap ≈ 0) ⇒ Tuần 4 **bắt buộc** dùng nhãn thật.

## 2. Mục tiêu & phạm vi

| Hạng mục | Nội dung |
|---|---|
| Mục tiêu | Hạ tầng đo lường đáng tin: một giao diện, chia tập theo thời gian, chỉ số label-free, log + manifest tái lập |
| Trong phạm vi | 5 mô hình, 16 đặc trưng core, time-based split, chỉ số label-free, artifact + tài liệu |
| **Ngoài phạm vi** (Tuần 4) | Nhãn tấn công thật `redteam.txt`, module tiêm bất thường `injector.py`, chỉ số cần nhãn (đã cài sẵn, chưa dùng) |

## 3. Đã bàn giao

**Mã nguồn mới**

| File | Vai trò |
|---|---|
| `src/models/base.py` | `BaseAnomalyModel` — 3 bất biến: `score` CAO = DỊ BIỆT, ngưỡng = phân vị trên train, metadata tái lập; whitelist tham số sklearn |
| `src/models/detectors.py` | Isolation Forest (fit toàn train), LOF + One-Class SVM (fit mẫu con 20.000 dòng) |
| `src/models/baselines.py` | `ZScoreBaseline` (median + MAD→IQR→std), `RuleThresholdBaseline` (6 luật nghiệp vụ) |
| `src/models/pipeline.py` | `AnomalyPipeline`: **16 core → imputer median → RobustScaler → mô hình** |
| `src/models/registry.py` | 5 tên canonical + alias CamelCase của log cũ; factory đọc `defaults`/`training`/`preprocessing` |
| `src/models/benchmark.py` | Viết lại: điều phối benchmark trên time split (bản cũ giữ làm shim deprecated) |
| `src/evaluation/split.py` | `time_split` + `SplitInfo` (gói `src/evaluation/` **chưa từng tồn tại trong Git**) |
| `src/evaluation/metrics.py` | Chỉ số label-free + chỉ số cần nhãn (sẵn sàng cho Tuần 4) |
| `src/evaluation/experiment_log.py` | Log tương thích **nguyên 18 cột cũ** + cột truy vết, chống ghi trùng |
| `src/evaluation/manifest.py` | `run_manifest.json`: commit, `sha256` dữ liệu, phiên bản thư viện |

**Artifact đã commit (không còn nằm ngoài git)**

`experiments/results/`: `benchmark_summary.csv`, `anomaly_scores.parquet` (333.671 dòng × **19 cột** = 4 cột danh tính + 5 mô hình × 3 cột `_score`/`_pct`/`_anomaly`), `model_topk_overlap.csv`, `model_stability.csv`, `split_info.json`, `run_manifest.json`
`experiments/logs/experiment_log.csv`: 13 dòng (8 dòng lịch sử + 5 dòng Tuần 3), 30 cột

## 4. Kết quả chạy thật (seed 42, ngân sách cảnh báo 5%, K = 20)

| Mô hình | n_fit | Tỷ lệ cảnh báo | Lệch ngân sách | Fit (s) | Chấm điểm (s) |
|:---|---:|---:|---:|---:|---:|
| **Isolation Forest** | 721.612 | 5,93% | +0,93 pp | 11,43 | 1,97 |
| **One-Class SVM** | 20.000 | 6,39% | +1,39 pp | 3,36 | 16,57 |
| **Z-score Baseline** | 721.612 | 5,85% | +0,85 pp | 2,63 | 0,12 |
| **Local Outlier Factor** | 20.000 | 9,24% | +4,24 pp | 2,23 | 4,91 |
| **Rule-Threshold Baseline** | 721.612 | 20,35% | +15,35 pp | 1,15 | 0,09 |

> Cột "Lệch ngân sách" là **độ trôi hiệu chuẩn**, không phải chất lượng mô hình — ngưỡng được fit trên train nên
> nếu phân phối điểm của test khác train thì tỷ lệ cảnh báo trôi. Bảng này chưa nói mô hình nào tốt hơn: thiếu nhãn.

**Phân tích định lượng bảng này** (chi phí chấm điểm, hình dạng phân phối điểm, so sánh 16 → 24 đặc trưng):
xem [`phan_tich_bang_ket_qua_lan_chay_dau.md`](../archive/phan_tich_bang_ket_qua_lan_chay_dau.md) ·
bảng trích sẵn: [`bang_ket_qua_lan_chay_dau_3_mo_hinh.csv`](bang_ket_qua_lan_chay_dau_3_mo_hinh.csv).

## 5. Phát hiện chính (đều có script đo lại được)

| # | Phát hiện | Ý nghĩa / đã xử lý |
|---|---|---|
| 1 | Alert rate trôi là **hiệu ứng mẫu nhỏ**, không phải lỗi: train 200 dòng → IF 19,1% · LOF 10,2% · OCSVM 31,7%; train 5.000 dòng → 5,3% · 6,9% · 6,1% | Trên dữ liệu thật (fit ≥ 20.000 dòng) về sát ngân sách; ngưỡng phân vị trên train là lựa chọn đúng |
| 2 | Baseline luật đạt **20,35%** cảnh báo vì điểm chỉ có 7 giá trị rời rạc → ngưỡng phân vị 95% rơi vào mức 0,5 | Hạn chế bản chất của điểm rời rạc — **ghi rõ trong docstring + báo cáo**, không che; Tuần 4 thêm tie-break có căn cứ |
| 3 | **7/16 đặc trưng có MAD = 0** (quá nửa số dòng bằng median) nhưng 0 đặc trưng hằng số | `ZScoreBaseline` dùng fallback `MAD → IQR/1,349 → std → 1,0`; nếu không, đơn vị thô của `interarrival_dt_mean` (tới 86.297 giây) sẽ lấn át điểm z |
| 4 | **5 mô hình gần như không đồng thuận** ở Top-20 (overlap ≈ 0; cao nhất OCSVM↔Z-score = 0,10) | Kết luận: **không thể xếp hạng bằng label-free** → Tuần 4 chấm bằng nhãn thật là bắt buộc |
| 5 | **Top-20 của LOF phụ thuộc seed** (Spearman 0,876 nhưng overlap Top-20 chỉ 0,03) | LOF chỉ fit 20.000/721.612 dòng ⇒ **không nên công bố "20 tài khoản đáng ngờ nhất" của LOF**; cần thử ngân sách 100k/toàn bộ ở Tuần 4 |

## 6. Quyết định đã khoá (để tái lập không phải đoán)

| Hạng mục | Giá trị | Ghi chú |
|---|---|---|
| Chia tập | `split_day = 42` | train ngày 1–42 = 721.612 · test ngày 43–60 = 333.671 (31,62%) — **khớp đúng** `train_partition_size`/`test_samples` của log cũ |
| Đặc trưng | 16 core theo `configs/feature_schema.yaml` | cột ngoài core (`total_logons`, `distinct_hosts`, `rare_logon_type_count`) không được dùng |
| Tiền xử lý | imputer median (fit train) → RobustScaler (fit train) | baseline luật **không** bị scale để giữ ngữ nghĩa ngưỡng |
| Ngân sách cảnh báo | `contamination = 5%` → ngưỡng = phân vị `1 − contamination` trên train | so sánh công bằng giữa mô hình dùng xếp hạng theo ngân sách trên chính tập test |
| Ngân sách fit | LOF/OCSVM 20.000 dòng; IF + baseline dùng toàn bộ train | log tách bạch `n_fit` và `n_training_partition` |
| Ổn định | 3 seed (42, 7, 2024), mẫu 50.000 dòng test, K = 20 | kích thước mẫu được ghi trong artifact |

## 7. Việc còn lại (Tuần 4)

1. **Nhãn thật** `redteam.txt` → khớp `(DomainName, UserName, day)`, gán nhãn cho 333.671 dòng test.
2. **Bật chỉ số cần nhãn** đã cài + test sẵn: `precision_at_k`, `recall_at_budget`, `roc_auc`, `average_precision`.
3. **`src/evaluation/injector.py`** (kịch bản tiêm bất thường) — chưa có.
4. **Thử lại ngân sách fit của LOF** (100.000 dòng hoặc toàn bộ) để xử lý phát hiện #5.
5. **Tie-break cho baseline luật** (ví dụ tổng z-score) hoặc báo cáo riêng theo ngân sách (phát hiện #2).
6. Thêm cột nhãn vào `anomaly_scores.parquet` + bảng xếp hạng có precision/recall.

## 8. Tái lập trong 4 lệnh

```bash
python main.py --stage benchmark          # 5 mô hình, seed 42, chia tại ngày 42 (~168 s)
python -m pytest tests/ -q                # 31 passed
python scripts/diagnostics/day_split_table.py        # bằng chứng ranh giới chia tập
python scripts/diagnostics/feature_variance_check.py # bằng chứng MAD = 0
```

## 9. Trạng thái kỹ thuật

* **Test:** 31/31 pass (trước Tuần 3: 11/11). Gồm 12 tên test **phục hồi** từ `.pytest_cache` của bản đã mất
  (`test_model_factory`, `test_time_based_split`, `test_evaluation_metrics`, `test_baselines`, 3× `*_pipeline`,
  `test_max_train_samples_subsampling`, `test_model_save_and_load`, `test_input_sanitization`,
  `test_model_interfaces`, `test_feature_extraction_columns_and_counts`) và 7 test mới.
* **Git:** 4 commit trên `feature/model-benchmark-framework` (`e801c67`, `124de89`, `106d396`, `07c85a7`) + tag `features-v1`;
  `run_manifest.json` ghi commit, `sha256` của parquet và phiên bản thư viện (sklearn 1.9.1, scipy 1.18.1, polars 1.44.2, numpy 2.5.3).
* **Chống mất mát:** `.gitignore` đã sửa để `experiments/results/*` và `experiments/logs/experiment_log.csv` **được commit**
  (bài học Tuần 2: log bị ignore nên 8 dòng kết quả + `model_factory` + ~10 test suýt mất hoàn toàn).
* **Tài liệu:** README, `docs/project_structure.md`, `docs/nhat_ky_trien_khai_pipeline.md` đã đính chính — bảng
  "1.055.283 mẫu đánh giá" cũ là **artifact của rò rỉ dữ liệu** (vẫn giữ trong `<details>` để đối chiếu lịch sử).

