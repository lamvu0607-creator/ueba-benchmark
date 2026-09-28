# Đánh giá HẬU CHUẨN HÓA (log1p) — “Đã ổn chưa? Có cần chuẩn hóa tiếp không?”

*Ngày tạo: 2026-09-26 · Cập nhật: 2026-09-24 (bổ sung Bảng 3 + 3 panel hình) · Phạm vi: Tuần 2 · Script: `scripts/feature_engineering/check_normalization_effect.py`*
*Nguồn số liệu: `docs/feature_engineering/tables/normalization_effect_all_features.csv`, `normalization_alternative_transforms.csv`, `normalization_shape_zoom_vs_full.csv`, `distribution_skewness_comparison.csv`, `multicollinearity_vif.csv`*

---

## 1. Trả lời ngắn

| Câu hỏi | Trả lời | Bằng chứng |
| :--- | :--- | :--- |
| Chuẩn hóa log1p **đã ổn chưa**? | ✅ **ỔN cho 4 cột đã chốt**: `\|skew\|` 91–152 → **1,26–2,47** (giảm 98,3–99,0%), cả 4 ≤ ngưỡng 3 | Bảng §2 |
| Có cần chuẩn hóa **tiếp** không? | ✅ **CÓ — 2 cột nữa nên áp log1p**: `interarrival_dt_mean` (16,01 → **−0,30**) và `delta_t_cv` (9,37 → **0,95**) vì đang vẫn ở dạng thô với \|skew\| > 3 | Bảng §2 |
| Có nên đổi sang biến đổi mạnh hơn (sqrt / cbrt / Yeo-Johnson)? | ❌ **Không cần**: sqrt / cbrt **tệ hơn hẳn**; Yeo-Johnson chỉ nhích hơn chút nhưng mất khả năng diễn giải/hoàn tác và λ fit trên dữ liệu | Bảng §3 |
| 8 cột còn lại thì sao? | 🟢 **Giữ THÔ** — log1p vô hiệu (chỉ giảm 0–16,7%) vì zero-inflated / bị chặn `[0,1]` | Bảng §2 |
| Nhìn histogram sau log1p **vẫn thấy lệch**? | ✅ **Đúng — nhưng không phải do thiếu chuẩn hóa**: do ~1% đuôi cực trị kéo dài trục + gai 0 + biến đếm rời rạc; vùng chứa 99% dữ liệu có skew **+0,02 … +1,36** | Bảng §4 |
| Bước “chuẩn hóa” còn thiếu? | ⏳ **SCALING** (Standard/Robust) — khác bản chất với biến đổi hình dạng; **chưa làm** vì thuộc tầng model (theo phạm vi đã chốt) | §6 |

## 2. Bảng 1 — 16 đặc trưng: skew gốc → sau log1p (60 ngày, N = 1.055.283)

| Nhóm | Đặc trưng | Zero_% | Skew gốc | Skew sau log1p | Skew sau log (chỉ dòng ≠ 0) | Giảm | Kết luận |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Đã chuẩn hóa (4 cột chốt)** | `rare_logon_type_count` | 71,09 | 151,66 | **+1,59** | 0,88 | 99,0% | ✅ Đạt |
| | `distinct_sources_count` | 1,46 | 147,39 | **+2,47** | 3,28 | 98,3% | ✅ Đạt (đuôi thật còn dài) |
| | `total_logons` | 0,00 | 94,52 | **−1,51** | −1,51 | 98,4% | ✅ Đạt (hơi lệch trái) |
| | `distinct_hosts` | 0,00 | 91,12 | **+1,26** | 1,26 | 98,6% | ✅ Đạt |
| **NÊN BỔ SUNG log1p** | `interarrival_dt_mean` | 0,34 | 16,01 | **−0,30** | 0,08 | 98,2% | ⭐ Hiệu quả rõ rệt |
| | `delta_t_cv` | 0,09 | 9,37 | **+0,95** | 0,99 | 89,8% | ⭐ Hiệu quả rõ rệt |
| **Giữ THÔ (log vô hiệu)** | `failure_locked_out_share` | 98,86 | 11,69 | 11,22 | −0,47 | 4,0% | 🟢 Zero-inflated |
| | `interactive_ratio` | 87,00 | 11,39 | 10,79 | 3,59 | 5,3% | 🟢 Zero-inflated |
| | `is_single_event` | 98,47 | 7,91 | 7,91 | — | 0,0% | 🟢 Nhị phân 0/1 |
| | `failure_ratio` | 87,36 | 6,77 | 6,58 | 1,76 | 2,8% | 🟢 Zero-inflated |
| | `custom_proc_share` | 90,33 | 5,12 | 5,02 | 0,57 | 1,9% | 🟢 Zero-inflated |
| | `missing_source_ratio` | 19,97 | 4,24 | 3,73 | 3,51 | 12,0% | 🟢 Bị chặn [0,1] |
| | `ntlm_ratio` | 32,28 | 3,39 | 2,83 | 2,60 | 16,7% | 🟢 Bị chặn [0,1] |
| | `remote_logon_ratio` | 4,88 | −2,79 | −2,99 | −4,44 | −7,0% | 🟢 Log làm **xấu hơn** |
| **Đã cân sẵn** | `off_hours_ratio` | 8,19 | −0,55 | −0,99 | −0,80 | – | ✅ Không cần |
| | `same_second_share` | 5,17 | −0,56 | −0,98 | −0,51 | – | ✅ Không cần |

### Vì sao độ lệch CÒN LẠI không sửa được bằng biến đổi hàm
- `rare_logon_type_count` (còn 1,59): skew chỉ tính trên dòng **≠ 0** là **0,88** → độ lệch còn lại đến từ **cụm 71,09% điểm 0**. Đây là thông tin nghiệp vụ (phần lớn tài khoản không dùng logon loại hiếm), luôn tồn tại với biến zero-inflated — mọi biến đổi đơn điệu `f(0)=0` đều giữ nguyên gai này.
- `distinct_sources_count` (còn 2,47): ngược lại, skew trên dòng ≠ 0 là **3,28** → đuôi cực trị (max 3.744 → `log1p` ≈ 8,2) là **thật**. Nén tiếp sẽ làm mờ chính tín hiệu bất thường → **không nên**.
- `total_logons` = −1,51: dấu hiệu **nén hơi quá tay** (lệch trái nhẹ) nhưng vẫn trong ngưỡng 3 → chấp nhận, không cần hoàn tác.

## 3. Bảng 2 — So sánh log1p với các biến đổi mạnh hơn (4 cột lệch nặng)

| Đặc trưng | Skew gốc | `log1p` | `sqrt` | `cbrt` | Yeo-Johnson | λ (YJ) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rare_logon_type_count` | 151,66 | **1,59** | 69,69 | 7,11 | 0,99 | −1,123 |
| `distinct_sources_count` | 147,39 | **2,47** | 52,31 | 7,07 | −0,36 | −0,909 |
| `total_logons` | 94,52 | **−1,51** | 32,64 | 10,58 | 0,52 | 0,145 |
| `distinct_hosts` | 91,12 | **1,26** | 45,69 | 16,41 | −0,04 | −0,295 |

→ `sqrt` / `cbrt` **không đủ mạnh** (còn 7–70 skew). Yeo-Johnson nhích hơn một chút (3/4 cột gần 0) nhưng: (i) **không diễn giải được** với mentor/đội dự án, (ii) phải **lưu λ và refit khi dữ liệu đổi ngày**, (iii) nén mạnh hơn ⇒ **giảm biên độ của chính các điểm bất thường**. Vì log1p đã đạt ≤ 3 nên **giữ log1p**.

## 4. Bảng 3 — Hình dạng TOÀN DẢI vs ZOOM 99%: vì sao ảnh histogram vẫn “trông lệch”?

> Nguồn: `tables/normalization_shape_zoom_vs_full.csv` (sinh tự động, BẢNG 3 của script audit).
> “Zoom 99%” = chỉ giữ các dòng có `log1p(x) ≤ P99` — tức vùng chứa **99% dữ liệu**, nơi phản ánh hình dạng thật của khối chính.

| Đặc trưng | Zero_% | P99 (log) | Max log (ngoài khung zoom) | Skew **toàn dải** | Kurt toàn dải | Skew **zoom 99%** | Kurt zoom 99% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rare_logon_type_count` | 71,09 | 4,25 | 12,89 | +1,59 | 1,97 | **+1,36** | 0,24 |
| `distinct_sources_count` | 1,46 | 1,79 | 8,23 | +2,47 | 26,01 | **+1,25** | 3,17 |
| `total_logons` | 0,00 | 8,23 | 15,21 | −1,51 | 4,59 | **−1,99** | 4,28 |
| `distinct_hosts` | 0,00 | 2,40 | 9,24 | +1,26 | 15,62 | **+0,02** | −0,60 |
| `interarrival_dt_mean` | 0,34 | 8,90 | 11,37 | −0,30 | 7,86 | −1,17 | 8,06 |
| `delta_t_cv` | 0,09 | 2,17 | 4,95 | +0,95 | 2,76 | +0,50 | 0,72 |

### Ba nguyên nhân khiến ảnh vẫn “trông lệch” (không phải do chuẩn hóa chưa đủ)
1. **~1% đuôi cực trị kéo dài trục x, khối 99% bị nén vào một góc.** `distinct_sources_count`: 99% dữ liệu nằm trong `x ≤ 1,79` nhưng max là `8,23` → trục x dài gấp **4,6 lần** vùng có dữ liệu; `distinct_hosts`: P99 = 2,40 vs max = 9,24 (gấp **3,9 lần**). Vì vậy chỉ ~6/35 bin histogram có dữ liệu, phần còn lại là “khoảng trắng” ⇒ mắt đọc thành “lệch nặng”.
2. **Gai zero-inflation.** `rare_logon_type_count` có **71,09%** điểm bằng 0 → một cột cao 1,94 (mật độ) che hết phần còn lại. Đây là thông tin nghiệp vụ; **mọi biến đổi đơn điệu `f(0)=0` đều giữ nguyên gai này** (không sửa được bằng toán).
3. **Kurtosis (đuôi dày/đỉnh nhọn) chứ không phải skew.** Sau log1p, kurtosis còn 1,97–26,01 — cảm giác “lệch” của mắt chủ yếu do **đuôi dày** chứ không phải bất đối xứng: `distinct_hosts` sau log có `skew = +1,26` nhưng ở vùng zoom là **+0,02 / kurt −0,60** (gần như **đối xứng chuẩn**).

### Đối chứng định lượng
- `distinct_hosts`: skew toàn dải +1,26 → **+0,02** khi chỉ bỏ top 1% (đuôi hosts > 10). Nghĩa là bất đối xứng **chỉ đến từ ~1% đuôi**, khối chính đã cân.
- `distinct_sources_count`: +2,47 → **+1,25** (zoom), kurt 26,01 → 3,17. Phần còn lại là **bậc thang rời rạc** của biến đếm nguyên (0,1,2,3,4,5 → `log1p` = 0; 0,69; 1,10; 1,39; 1,61; 1,79): histogram của biến đếm luôn “gai” và **không có biến đổi đơn điệu nào** biến nó thành chuẩn.
- `total_logons`: vùng zoom còn lệch trái −1,99 + **hai đỉnh** (một đỉnh phụ ở `log ≈ 0,7` = tài khoản gần như không hoạt động). Đây là **hai nhóm hành vi thật**, không phải lỗi chuẩn hóa → giữ để mô hình bất thường phát hiện.
- `interarrival_dt_mean`: skew toàn dải −0,30 nhưng zoom −1,17 / kurt 8,06 (không có 0) → **đuôi trái** do các cặp sự kiện cách nhau rất ngắn; nằm trong ngưỡng nên vẫn chấp nhận.

### Hình đã được cập nhật để chứng minh điều này
`docs/feature_engineering/figures/distribution_*.png` nay gồm **3 panel**:
1. Thẻ thống kê (Mean/Median/P99/Max/**tỷ lệ = 0**) + bằng chứng giảm lệch + **Kurtosis sau log**;
2. Toàn bộ dải sau `log1p` + chú thích **gai zero-inflation** và **đuôi ngoài khung**;
3. **Zoom vùng 99%** (`x ≤ P99`) kèm skew/kurtosis của chính vùng zoom.

> *(Panel Q-Q plot vs Chuẩn đã **bỏ** khỏi hình ngày 2026-09-24 — thay bằng `Kurtosis sau log` trong thẻ thống kê và cột `Log_Kurtosis` của [`tables/distribution_skewness_comparison.csv`](tables/distribution_skewness_comparison.csv).)*

> **Trả lời nếu mentor hỏi “sao sau log1p vẫn lệch?”** — Đúng là ảnh toàn dải vẫn lệch, nhưng đó là do **~1% đuôi cực trị** + **gai 0** + **biến đếm rời rạc**, không phải do thiếu chuẩn hóa: |skew| đã giảm **98,3–99,0%**, vùng 99% dữ liệu có skew **0,02–1,36**, và các biến đổi mạnh hơn (Yeo-Johnson) chỉ đổi được hình dạng nhưng **nén mất biên độ điểm bất thường** — xem §5.

## 5. Kết luận theo 4 nhóm và việc cần làm

1. **4 cột đã chuẩn hóa** → giữ nguyên, không thêm biến đổi.
2. **Thêm 2 cột** `interarrival_dt_mean`, `delta_t_cv` vào bước log1p ở B3 (chúng có NULL: 1,53% và 2,48% → `log1p` **chỉ áp cho dòng khác NULL**, giữ NULL nguyên trạng).
3. **8 cột giữ thô** → không áp log (không có lợi ích, chỉ mất khả năng diễn giải).
4. **Không** dùng sqrt/cbrt/Yeo-Johnson.
5. Sau log1p, bước còn lại (khi sang model) là **scaling**, không phải biến đổi hình dạng.

## 6. Ranh giới phạm vi (chưa làm)

- Chưa tạo `data/features/normalized/` — hiện log1p mới được tính **trong bộ nhớ** khi vẽ/kiểm tra (B3 vẫn ⏳).
- Chưa chuẩn hóa thang đo (StandardScaler / RobustScaler) — **thuộc tầng model**, ngoài phạm vi đã chốt (không làm gì liên quan đến model).
- Ghi chú tương quan: Spearman **bất biến** với log1p (sai số 1,14×10⁻⁷) nên kết luận “0/120 cặp vượt 0,85” trong [`multicollinearity_check.md`](multicollinearity_check.md) vẫn đúng cho bộ đã log; chỉ Pearson/VIF cần tính lại nếu ma trận normalized được sinh ra.

## 7. Tái lập

```bash
# Bảng 1 + 2 + 3 + kết luận PASS/FAIL, ghi 3 CSV vào docs/feature_engineering/tables/
python scripts/feature_engineering/check_normalization_effect.py

# 4 ảnh 3-panel (thẻ thống kê | toàn dải | zoom 99%) + bảng skew/kurtosis
python scripts/feature_engineering/plot_feature_distribution.py

# Hoặc chạy toàn bộ chuỗi (kiểm tra ma trận thô -> tương quan -> đa cộng tuyến -> hậu chuẩn hóa)
python scripts/feature_engineering/run_all_feature_plots.py
```
