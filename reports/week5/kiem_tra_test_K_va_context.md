# Bỏ nhóm context và kiểm tra trên test: bộ 28, K = 15, K = 18

> Tiếp theo [`chon_so_dac_trung_K.md`](chon_so_dac_trung_K.md). Phân khúc User, cấu hình mô hình chốt, model seed 42 / 7 / 2024 / 1 / 2.
> - **Dev:** 5 run `dev_seed20261036`–`040`, fit ngày 1–35, chấm ngày 36–42.
> - **Test:** 5 run `test_seed20261061`–`065`, fit ngày 1–42, chấm ngày 43–60.
>
> Chỉ số chính: **recall top-100/ngày**; phụ: PR-AUC. Mọi Δ là **hiệu ghép cặp** trên cùng run và seed, TB ± std qua 5 run.

## 1. Câu hỏi

1. Sau rhythm, nhóm **context** cũng đóng góp ít. `missing_source_ratio` đã bỏ trong bộ 30; có nên bỏ nốt `remote_logon_ratio` và `custom_proc_share` (còn **28** đặc trưng) không?
2. Đường cong K trên dev cho mỗi mô hình một K tốt nhất: OCSVM phẳng (tốt nhất K = 15, 25 đặc trưng), IF tốt nhất K = 18 (22 đặc trưng), LOF tốt nhất ở bộ 30. Các kết luận này có lặp lại trên test không?

**Quy tắc quyết định (đặt trước khi chạy):** chỉ bỏ đặc trưng nếu không mô hình nào giảm rõ (|Δ TB| > 1 std; với LOF còn phải vượt nhiễu fit lại). Ứng viên chọn trên dev được kiểm tra **một lần** trên test.

**Quan hệ giữa các bộ:** các bộ lồng nhau. Bộ 28 = bộ 30 bỏ thêm 2 đặc trưng context; bộ 22 (K = 18) = bộ 28 bỏ thêm 6 đặc trưng: `delta_mean_distinct_pair_7d`, `new_host_count_7d`, `rare_logon_type_count_log`, `off_hours_ratio`, `delta_mean_distinct_fail_host_7d`, `jaccard_logontype_7d`.

## 2. Bỏ nhóm context trên dev

**Bảng C.1: Δ so với bộ 30, 5 run dev** (mốc bộ 30: recall OCSVM 0,380 / IF 0,228 / LOF 0,200)

| Cấu hình | Còn lại | ΔRecall OCSVM | ΔRecall IF | ΔRecall LOF | ΔPR-AUC OCSVM | ΔPR-AUC IF | ΔPR-AUC LOF |
|---|---|---|---|---|---|---|---|
| bỏ `remote_logon_ratio` | 29 | +0,001 ± 0,003 | **+0,025 ± 0,013** | 0,000 ± 0,013 | +0,001 ± 0,001 | **+0,027 ± 0,013** | +0,002 ± 0,017 |
| bỏ `custom_proc_share` | 29 | **+0,004 ± 0,003** | **+0,016 ± 0,011** | 0,000 ± 0,009 | **+0,004 ± 0,002** | **+0,020 ± 0,013** | +0,002 ± 0,006 |
| **bỏ cả hai (bộ 28)** | 28 | **+0,007 ± 0,006** | **+0,027 ± 0,017** | −0,011 ± 0,010 | **+0,004 ± 0,003** | **+0,036 ± 0,011** | −0,007 ± 0,010 |
| đối chứng: fit lại bộ 30 | 30 | 0 | 0 | +0,004 ± 0,012 | 0 | 0 | +0,004 ± 0,018 |

In đậm: |Δ TB| > 1 std.

**Nhận xét:**
- Bỏ từng đặc trưng riêng lẻ: OCSVM và LOF không đổi, IF tăng rõ. Hai đặc trưng context còn lại không giúp mô hình nào.
- Bỏ cả hai: OCSVM tăng nhẹ nhưng rõ, IF tăng rõ nhất. LOF giảm −0,011, nhỏ hơn nhiễu fit lại của LOF trên dev (±0,012), nên theo quy tắc không tính là giảm rõ. Mức giảm tập trung ở burst (−0,04) và logon_type_switch (−0,02).
- **Kết luận trên dev:** bộ 28 đạt quy tắc, là ứng viên duy nhất đưa lên test.

📁 [`experiments/ablation/context/`](../../experiments/ablation/context/)

## 3. Kiểm tra trên test

### 3.1 Bộ 28 so với bộ 30

**Bảng C.2: Δ so với bộ 30, 5 run test** (mốc bộ 30: recall OCSVM 0,376 / IF 0,209 / LOF 0,145; PR-AUC 0,192 / 0,089 / 0,058)

| Cấu hình | ΔRecall OCSVM | ΔRecall IF | ΔRecall LOF | ΔPR-AUC OCSVM | ΔPR-AUC IF | ΔPR-AUC LOF |
|---|---|---|---|---|---|---|
| **bộ 28** | +0,004 ± 0,005 | **+0,032 ± 0,009** | +0,010 ± 0,010 | **+0,004 ± 0,001** | **+0,031 ± 0,012** | +0,003 ± 0,007 |
| đối chứng: fit lại bộ 30 | 0 | 0 | +0,006 ± 0,003 | 0 | 0 | +0,004 ± 0,003 |

**Δ recall theo kịch bản của bộ 28 trên test:**

| Kịch bản | OCSVM | IF | LOF |
|---|---|---|---|
| brute_force | +0,026 | +0,022 | −0,002 |
| password_spraying | +0,002 | +0,009 | −0,001 |
| off_hours | +0,004 | +0,006 | +0,020 |
| logon_type_switch | **−0,033** | +0,056 | −0,037 |
| new_workstation_burst | +0,020 | +0,058 | +0,056 |
| dormant_wakeup | +0,006 | +0,039 | +0,024 |

**Nhận xét:**
- **Bộ 28 được xác nhận trên test:** không mô hình nào giảm. IF tăng rõ (+0,032 recall, +0,031 PR-AUC), khớp với dev (+0,027). OCSVM tăng nhẹ ở PR-AUC (+0,004 ± 0,001, rõ); recall +0,004 trong nhiễu. LOF +0,010, ngang mức nhiễu fit lại.
- Mức giảm của LOF trên dev (−0,011) không lặp lại trên test (+0,010), nên đó là nhiễu.
- **Điểm cần theo dõi:** logon_type_switch giảm ở OCSVM (−0,033) và LOF (−0,037) trên test; trên dev OCSVM cũng giảm nhẹ (−0,011). `remote_logon_ratio` có thể mang một phần tín hiệu đổi kiểu đăng nhập. Mức giảm nhỏ so với recall của kịch bản này (0,65 ở OCSVM) và được bù ở brute_force và burst.

📁 [`experiments/ablation/context_test/`](../../experiments/ablation/context_test/)

### 3.2 K = 15 và K = 18 so với bộ 30

Chạy theo yêu cầu: K = 15 (tốt nhất cho OCSVM trên dev), K = 18 (tốt nhất cho IF trên dev), bộ 30 (tốt nhất cho LOF trên dev).

**Bảng C.3: Recall top-100/ngày trên test, TB 5 run** (trong ngoặc: hiệu ghép cặp so với bộ 30)

| Cấu hình | Còn lại | OCSVM | IF | LOF |
|---|---|---|---|---|
| đủ 41 | 41 | 0,322 (−0,055 ± 0,021) | 0,162 (−0,047 ± 0,011) | 0,129 (−0,018 ± 0,010) |
| bộ 30 | 30 | **0,376** | 0,209 | 0,146 |
| K = 15 | 25 | 0,375 (−0,002 ± 0,005) | 0,235 (+0,027 ± 0,007) | 0,172 (+0,026 ± 0,012) |
| K = 18 | 22 | 0,373 (−0,003 ± 0,005) | **0,242 (+0,034 ± 0,013)** | **0,174 (+0,028 ± 0,005)** |

**Bảng C.4: PR-AUC trên test, TB 5 run**

| Cấu hình | OCSVM | IF | LOF |
|---|---|---|---|
| đủ 41 | 0,170 | 0,066 | 0,046 |
| bộ 30 | 0,192 | 0,089 | 0,059 |
| K = 15 | **0,194** | 0,111 | 0,070 |
| K = 18 | 0,189 | **0,111** | **0,072** |

**So với dev:**

| Mô hình | Dev | Test | Lặp lại? |
|---|---|---|---|
| OCSVM | phẳng từ 30 đến 22 đặc trưng | phẳng (K = 15: −0,002; K = 18: −0,003) | **Có** |
| IF | càng bỏ càng tốt, tốt nhất 22 | K = 18 tốt nhất (+0,034) | **Có** |
| LOF | tốt nhất bộ 30, K = 15/18 giảm rõ (−0,026 / −0,037) | K = 15/18 **tăng** rõ (+0,026 / +0,028) | **Không** |

**Nhận xét:**
1. **OCSVM và IF lặp lại đúng như dev.** Với OCSVM (mô hình chính), số đặc trưng từ 22 đến 30 không quan trọng. IF được lợi rõ khi bỏ chiều yếu, ở cả dev và test.
2. **LOF đảo chiều giữa dev và test.** Nhiễu fit lại của LOF trên test nhỏ (±0,003–0,005 recall), nên mức tăng +0,026–0,028 trên test là thật *với 5 run test này*. Mức giảm trên dev cũng vượt nhiễu dev. Vậy phản ứng của LOF với việc bỏ 6–9 đặc trưng phụ thuộc vào dữ liệu (dev: 35 ngày fit; test: 42 ngày fit và 18 ngày chấm), không ổn định. Kết luận "LOF tốt nhất ở 30–33 đặc trưng" trong `chon_so_dac_trung_K.md` **không được test xác nhận**.
3. **Nhiễu LOF trên test nhỏ hơn dev:** fit lại bộ 30 cho ±0,003 recall trên test, so với ±0,012 trên dev. Hai lần chạy bộ 30 trên test (ở hai thí nghiệm) chỉ lệch −0,001 ± 0,005 với LOF; OCSVM và IF trùng khớp.

📁 [`experiments/ablation/kcurve_test/`](../../experiments/ablation/kcurve_test/)

## 4. Kết luận và đề xuất

- **Đổi bộ đặc trưng chung từ 30 sang 28 (bỏ nốt nhóm context).** Đây là ứng viên duy nhất chọn từ dev theo quy tắc đặt trước, và test xác nhận: không mô hình nào giảm, IF +0,032 recall, OCSVM +0,004 PR-AUC. Bộ 28 = bộ 30 bỏ thêm `remote_logon_ratio` và `custom_proc_share`.
- **Không đổi sang 22 đặc trưng (K = 18) dựa trên kết quả này,** dù trên test K = 18 tốt hơn bộ 30 ở IF và LOF:
  - K = 18 được chọn trên dev chỉ cho IF; trên dev nó làm LOF giảm rõ. Nếu giờ chọn K = 18 *vì* test tốt, ta đang dùng test để chọn, và test mất vai trò kiểm tra độc lập.
  - Với OCSVM (mô hình chính), K = 18 không hơn bộ 30 hay bộ 28.
  - Nếu muốn dùng 22 đặc trưng, cần một khối dữ liệu **mới chưa dùng** (ví dụ 5 run tiêm mới với seed khác trên ngày 43–60) để kiểm tra lại.
- **Nếu tối ưu riêng IF:** cả dev và test đều cho IF tốt nhất ở khoảng 22 đặc trưng; bộ riêng cho IF có căn cứ ở cả hai khối.
- **Dừng cắt tỉa ở đây.** Dev đã dùng cho 5 vòng chọn (rhythm, 10 SHAP thấp, đường cong K, context) và test đã dùng 3 lần (bộ 30, K = 15/18, bộ 28). Cắt tỉa thêm (ví dụ nhóm volume) sẽ khớp quá mức vào hai khối này. Đặc trưng volume còn lại (`delta_mean_count_7d`) có SHAP hạng 11 ở OCSVM và nằm trong top-5 của dormant_wakeup, nên không phải ứng viên bỏ.

**Việc cần làm nếu chốt bộ 28:**
1. Thêm `remote_logon_ratio`, `custom_proc_share` vào `drop_features` trong `configs/benchmark_grid_test_30f.yaml` (hoặc tạo cấu hình lưới mới cho bộ 28) và chạy lại lưới test chính thức. Không đổi `configs/feature_schema.yaml`.
2. Cập nhật báo cáo tuần 5 (mục 4.2, 4.3) và slide từ "bộ 30" sang "bộ 28".

## 4b. Kiểm tra cuối: nhóm volume và chọn giữa bộ của OCSVM và bộ của IF

**Volume.** Bộ 30 đã bỏ `log_total_logons` và `volume_robust_z_7d`; chỉ còn `delta_mean_count_7d`. Bỏ nốt nó khỏi bộ 28 (còn 27), 5 run dev, hiệu ghép cặp so với bộ 28:

| | OCSVM | IF | LOF |
|---|---|---|---|
| ΔRecall top-100/ngày | −0,001 ± 0,003 | +0,008 ± 0,015 | +0,028 ± 0,024 |
| ΔPR-AUC | +0,000 ± 0,001 | +0,004 ± 0,013 | +0,010 ± 0,021 |
| ΔRecall dormant_wakeup | −0,017 | −0,020 | +0,033 |

Không mô hình nào thay đổi rõ (mọi |Δ| < 1 std; LOF dao động trong nhiễu). Bỏ không có lợi, nhưng làm OCSVM và IF giảm nhẹ ở dormant_wakeup, kịch bản yếu nhất và đúng là kịch bản SHAP cho thấy đặc trưng này đóng góp (top-5 ở cả 3 run). **Giữ `delta_mean_count_7d`.** Không đưa bộ 27 lên test.

**Bộ tốt cho OCSVM hay bộ tốt cho IF?** Recall top-100/ngày (PR-AUC trong ngoặc), TB 5 run:

| Bộ | Khối | OCSVM | IF | LOF |
|---|---|---|---|---|
| 30 | dev | 0,380 (0,230) | 0,228 (0,096) | 0,200 (0,076) |
| **28** | dev | **0,388 (0,235)** | 0,255 (**0,132**) | 0,189 (0,070) |
| 22 (K = 18) | dev | 0,383 (0,227) | **0,264** (0,128) | 0,170 (0,062) |
| 30 | test | 0,376 (0,192) | 0,209 (0,089) | 0,145 (0,058) |
| **28** | test | **0,381 (0,196)** | 0,240 (**0,120**) | 0,155 (0,061) |
| 22 (K = 18) | test | 0,373 (0,189) | **0,242** (0,111) | **0,174 (0,072)** |

Bộ 28 không phải thỏa hiệp: nó **tốt nhất cho OCSVM ở cả dev và test**, và cho IF gần như toàn bộ lợi ích của bộ 22 (test: recall 0,240 so với 0,242; PR-AUC 0,120 so với 0,111). Chỉ LOF trên test thích bộ 22 hơn, nhưng trên dev thì ngược lại. **Chọn bộ 28 làm bộ chung**; không cần bộ riêng cho IF.

📁 Bộ 27: `experiments/ablation/context/` (cấu hình `drop_context_volume`).

## 5. Hạn chế

- Mỗi khối chỉ có 5 run, model seed khác nhau theo run; std của hiệu ghép cặp còn lớn so với các mức Δ nhỏ (±0,005).
- Hiện tượng LOF đảo chiều giữa dev và test chưa được giải thích. Giả thuyết: LOF nhạy với cấu trúc láng giềng, mà cấu trúc này đổi khi tập fit dài hơn (42 so với 35 ngày).
- Tham số mô hình giữ nguyên theo bản 41 đặc trưng, chưa dò lại cho 28 hoặc 22 đặc trưng.
- Chỉ kiểm tra phân khúc User và bộ kịch bản tiêm hiện có.

## 6. Chạy lại

```
python scripts/evaluation/feature_ablation.py --mode context                      # dev, ~25 phút
python scripts/evaluation/feature_ablation.py --mode context --dev-grid configs/benchmark_grid_test.yaml \
    --only base30 drop_context control_refit_base30 --out experiments/ablation/context_test              # test, ~25 phút
python scripts/evaluation/feature_ablation.py --mode kcurve --dev-grid configs/benchmark_grid_test.yaml \
    --only full drop_rhythm_low_shap10 drop_rhythm_k15 drop_rhythm_k18 --out experiments/ablation/kcurve_test   # test, ~30 phút
```
