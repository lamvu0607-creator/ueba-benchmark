# UEBA Benchmark: Windows Security Event Log (4624 / 4625)

Benchmark các mô hình phát hiện bất thường không giám sát cho bài toán phân tích hành vi người dùng (UEBA). Dữ liệu là log đăng nhập thành công (`EventID 4624`) và thất bại (`EventID 4625`) của bộ LANL, ngày 1–60.

Thiết lập của benchmark cuối:

- Mỗi dòng là một cặp tài khoản × ngày. Schema có 41 đặc trưng, mô hình dùng 22 đặc trưng (xem [mục 6](#6-bộ-đặc-trưng)).
- Chỉ chấm tài khoản loại User. Train ngày 1–42, đánh giá ngày 43–60.
- Nhãn là bất thường tổng hợp: bộ tiêm nhân bản sự kiện thật theo 6 kịch bản (brute-force, password spraying, ngoài giờ, máy lạ, ngủ đông thức dậy, đổi LogonType). Đây không phải nhãn tấn công thật của LANL.
- Ba mô hình (Isolation Forest, LOF, One-Class SVM) qua giao diện PyOD, so với bốn baseline (z-score, luật ngưỡng, đếm thất bại, ngẫu nhiên).

## 1. Kết quả chính

Trung bình 5 run tiêm test, 22 đặc trưng ([`experiments/grid/test_22f/grid_summary.csv`](experiments/grid/test_22f/grid_summary.csv)). Chỉ số chính là recall khi mỗi ngày chỉ xem 100 cảnh báo đầu.

| Mô hình | Cấu hình | Recall top-100/ngày | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| One-Class SVM | `ocsvm_standard_g01` | 0,336 ± 0,070 | 0,175 ± 0,048 | 0,892 |
| Isolation Forest | `if_n300_ms1024` | 0,242 ± 0,016 | 0,111 ± 0,011 | 0,921 |
| LOF | `lof_k20_robust` | 0,175 ± 0,006 | 0,071 ± 0,013 | 0,717 |
| Baseline đếm thất bại | | 0,098 | 0,025 | 0,623 |
| Baseline z-score | | 0,075 | 0,032 | 0,743 |
| Baseline luật ngưỡng | | 0,073 | 0,015 | 0,536 |
| Baseline ngẫu nhiên | | 0,014 | 0,010 | 0,503 |

Cấu hình mô hình được chọn trên lưới dev 22 đặc trưng. Phân tích theo kịch bản, SHAP và cách chọn 22 đặc trưng ở [tóm tắt tuần 5](reports/week5/tom_tat_tuan5.md).

---

## 2. Thiết lập môi trường

```bash
git clone https://github.com/lamvu0607-creator/ueba-benchmark.git
cd ueba-benchmark
```

`hnswlib` là bắt buộc: LOF dùng nó để tìm láng giềng, và `src/models` import nó ngay khi nạp. Bản trên PyPI phải biên dịch C++, nên trên Windows nên cài qua conda-forge.

### 2.1 Conda (khuyên dùng, kể cả Windows)

File [`environment.yml`](environment.yml) khai báo đủ thư viện, lấy từ conda-forge, Python 3.11. Trên Windows, mở Anaconda Prompt rồi gõ:

```bat
cd /d D:\duong\dan\ueba-benchmark
conda env create -f environment.yml
conda activate ueba
pip install -e . --no-deps
python -c "import hnswlib, pyod, polars; print('ok')"
```

Trên Linux/macOS các lệnh giống hệt, chỉ thay dòng `cd`.

### 2.2 venv

Windows (PowerShell), nên dùng Python 3.11 hoặc 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install chroma-hnswlib        # hnswlib dựng sẵn, vẫn import hnswlib
pip install -r requirements.txt
pip install -e .
```

Nếu pip vẫn cố biên dịch `hnswlib` và báo thiếu Microsoft Visual C++, bỏ tạm dòng `hnswlib>=0.8.0` trong `requirements.txt` rồi cài lại.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 3. Dữ liệu

`data/` không nằm trong Git. Benchmark cần 120 file Parquet interim: `data/interim/event_4624/event_4624_day-01.parquet` đến `day-60`, và bộ `event_4625` tương ứng (khoảng 12 GB).

Tải hai gói từ Google Drive rồi giải nén vào `data/interim/`:

- [Gói 1](https://drive.google.com/file/d/1aMz0oPItXDYn7u326Pl_jTnmdqFvsCmE/view?usp=sharing)
- [Gói 2](https://drive.google.com/file/d/1FjXHtZAjKQxYtN2sv7OO6YMkNbqs00ek/view?usp=sharing)

Kiểm tra đủ file và đúng phiên bản (so SHA-256 với [`configs/interim_sha256.txt`](configs/interim_sha256.txt)):

```bash
python scripts/validate_interim.py --mode fast
python -c "from pathlib import Path; from scripts.reproduce import verify_interim; verify_interim(Path('data/interim'))"
```

Muốn tự dựng từ log gốc LANL (`wls_day-XX.bz2` để trong `data/raw/`), chạy `python scripts/convert_raw_to_interim.py --raw-dir data/raw --interim-dir data/interim`. Chi tiết ở [quy trình từ dữ liệu đến kết quả](docs/chay_tu_du_lieu_den_ket_qua.md).

---

## 4. Chạy

### 4.1 Tái lập bảng kết quả cuối bằng một lệnh

```bash
python scripts/reproduce.py                  # --skip-existing: bỏ qua bước đã có đầu ra
```

Lệnh chạy kiểm thử, kiểm tra hash, tính đặc trưng, hồ sơ train, baseline, tiêm 5 run test (seed 20261061–20261065), lưới 22 đặc trưng, rồi so với bảng đã lưu. Kết quả ở `experiments/reproduce/test_22f/`, file `reproduce_check.csv` cho biết từng chỉ số có lệch quá 1% không. Lệnh mất nhiều giờ.

LOF không tất định vì chỉ mục HNSW dựng đa luồng. Trung bình 5 run vẫn lặp lại được (ngưỡng so 5%), nhưng danh sách cảnh báo của từng lần fit thì không. Muốn LOF tất định, đặt `n_jobs: 1` trong `configs/model_params.yaml`.

Hai con số để đối chiếu với máy khác: hồ sơ train ra 29.830 tài khoản, 15.946 máy, L = 5.

### 4.2 Chạy từng bước bằng `main.py`

```bash
python -m pytest tests -q
python main.py --stage features                # interim -> data/features/raw/, data/processed/
python scripts/injection_profiles_summary.py   # hồ sơ train (chỉ đọc ngày 1–42)
python main.py --stage baselines               # baseline trên log gốc
```

Các stage có sẵn: `clean`, `features`, `baselines`, `benchmark`, `inject`, `difficulty`, `plots`, `all`. `--stage all` là luồng cũ (raw → benchmark không nhãn); bảng cuối dùng `reproduce.py` hoặc `run_grid.py`.

Một run tiêm log ghi vào `data/injection_runs/<run_id>/` và không đè dữ liệu gốc:

```bash
python main.py --stage inject --block dev                                # -> data/injection_runs/dev_seed<seed>/
RUN=data/injection_runs/dev_seed<seed>
python main.py --stage features  --events-dir $RUN/events_injected
python main.py --stage baselines --events-dir $RUN/events_injected
python main.py --stage benchmark --events-dir $RUN/events_injected
python main.py --stage plots     --events-dir $RUN/events_injected       # biểu đồ PR, recall theo ngân sách, heatmap kịch bản
```

Khối `test` chỉ chạy sau khi đã chốt tham số trên `dev`. Xem [hướng dẫn tiêm](docs/huong_dan_tiem_bat_thuong_tuan4.md) và [lưới dev/test](docs/week4_benchmark_grid.md).

### 4.3 Lưới mô hình và phân tích

```bash
python scripts/evaluation/run_grid.py --config configs/benchmark_grid_dev_22f.yaml    # lưới dev, chọn cấu hình
python scripts/evaluation/run_grid.py --config configs/benchmark_grid_test_22f.yaml   # benchmark cuối
python scripts/evaluation/scenario_analysis.py --config configs/benchmark_grid_test_22f.yaml --out experiments/analysis/scenario_22f
python scripts/evaluation/build_review_set.py --config configs/benchmark_grid_test_22f.yaml --out experiments/analysis/review_top50_22f
python scripts/evaluation/review_agreement.py              # Cohen's kappa, sau khi hai người chấm xong
python scripts/evaluation/explain_models.py                # SHAP/LIME -> experiments/analysis/explain/
python scripts/evaluation/shap_feature_stability.py        # xếp hạng đặc trưng theo SHAP qua 3 run dev
python scripts/evaluation/feature_ablation.py --mode kcurve # đường cong K -> experiments/ablation/kcurve/
python scripts/evaluation/plot_kcurve.py
python -X utf8 scripts/experiments/window_ablation.py --run data/injection_runs/test_seed20261061 --windows 7 14 21
```

---

## 5. Demo

Cả hai demo dùng ba mô hình 22 đặc trưng, chỉ chấm tài khoản User, và cần `data/interim/` cho 7 ngày lịch sử.

Demo web (FastAPI, chạy trên máy):

```bash
python main.py --stage features
python -m demo.prepare                       # fit 3 mô hình -> experiments/models/demo_22f_user/
python -m uvicorn demo.backend.main:app --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000> và tải lên một file Parquet interim của các ngày sau ngày 42. Định dạng file và cách đọc kết quả ở [`demo/README.md`](demo/README.md).

Demo dòng lệnh, chấm log một ngày mới và in top-K cảnh báo kèm lý do:

```bash
python scripts/evaluation/train_demo_models.py                                 # lưu mô hình -> experiments/models/User_22f/
python scripts/demo_score_day.py --log path/to/wls_day-61.bz2 --inject         # log thật ngày 61, tiêm 1% User
python scripts/demo_score_day.py --simulate-from-day 47 --day 61 --inject      # diễn tập: dời log ngày 47 sang ngày 61
python scripts/demo_score_day.py --day 60 --check-against-matrix               # kiểm tra 22 đặc trưng khớp ma trận gốc
```

Log đưa vào phải đủ mọi User trong ngày vì `peer_z` so mỗi tài khoản với cả quần thể. Đầu ra ở `experiments/demo/day<t>/`.

---

## 6. Bộ đặc trưng

Schema [`configs/feature_schema.yaml`](configs/feature_schema.yaml) (v4.1) có 41 đặc trưng core, chọn qua phễu tương quan, VIF và khả năng tách kịch bản ([báo cáo bộ đặc trưng v4](docs/reports/bao_cao_bo_dac_trung_v4.md)). Benchmark cuối bỏ 19 đặc trưng có SHAP thấp bằng `drop_features` trong [`configs/benchmark_grid_test_22f.yaml`](configs/benchmark_grid_test_22f.yaml), còn 22 (cột Bộ 22). Cột Nhóm là nhóm dùng trong phân tích SHAP (`src/evaluation/feature_groups.py`).

Đặc trưng lịch sử chỉ dùng dữ liệu đến ngày t−1. Riêng `peer_z` so trong cùng ngày t với các tài khoản cùng loại.

| STT | Tên đặc trưng | Nhóm | Kiểu | Bộ 22 | Mô tả ý nghĩa an toàn thông tin |
| :---: | :--- | :--- | :--- | :---: | :--- |
| 1 | `log_total_logons` | volume | float64 | — | $\log(1 + \text{tổng sự kiện trong ngày})$, phản ánh cường độ hoạt động tổng thể |
| 2 | `failure_ratio` | failure | float64 | ✓ | Tỷ lệ đăng nhập thất bại ($\text{failures} / \text{total}$), chỉ báo brute force/dò mật khẩu |
| 3 | `failure_locked_out_share` | failure | float64 | ✓ | Tỷ trọng sự kiện khóa tài khoản trong tổng số lần thất bại (tín hiệu lockout) |
| 4 | `off_hours_ratio` | time | float64 | — | Tỷ lệ đăng nhập ngoài giờ hành chính (cửa sổ 18h - 7h sáng) |
| 5 | `interarrival_dt_mean` | rhythm | float64 | — | Khoảng cách thời gian trung bình (giây) giữa 2 sự kiện liên tiếp của tài khoản |
| 6 | `delta_t_cv` | rhythm | float64 | — | Hệ số biến thiên khoảng cách thời gian ($\text{std} / \text{mean}$), đo tính tuần hoàn/đều đặn |
| 7 | `same_second_share` | rhythm | float64 | — | Tỷ lệ sự kiện diễn ra cùng giây ($\Delta t = 0$), phát hiện đăng nhập tự động dạng script/bot |
| 8 | `is_single_event` | rhythm | uint8 | — | Cờ nhị phân đánh dấu tài khoản chỉ có duy nhất 1 sự kiện trong ngày |
| 9 | `activity_peak_hour_sin` | time | float64 | — | $\sin(2\pi \cdot h_{\text{peak}} / 24)$ — giờ cao điểm mã hoá chu kỳ |
| 10 | `activity_peak_hour_cos` | time | float64 | ✓ | $\cos(2\pi \cdot h_{\text{peak}} / 24)$ — cặp với #17 (sin/cos phải đi cùng nhau) |
| 11 | `hour_entropy` | time | float64 | ✓ | Pielou evenness $H / \ln(S)$ của phân bố sự kiện trên các khung giờ có mặt (độ trải đều nhịp) |
| 12 | `interactive_ratio` | logon_type | float64 | ✓ | Tỷ lệ đăng nhập tương tác cục bộ (`LogonType 2`) |
| 13 | `rare_logon_type_count_log` | logon_type | float64 | — | $\log(1 + \text{số sự kiện có kiểu hiếm})$, bao gồm Remote Desktop (Type 10), RunAs (Type 9) |
| 14 | `ntlm_ratio` | logon_type | float64 | ✓ | Tỷ lệ sử dụng giao thức xác thực NTLM (thường tương quan cao với các bất thường) |
| 15 | `log_distinct_hosts` | fanout | float64 | — | $\log(1 + \text{số máy trạm đích duy nhất})$, phát hiện dò quét mạng (lateral movement) |
| 16 | `distinct_sources_count` | fanout | uint32 | ✓ | Số lượng máy nguồn duy nhất thực hiện đăng nhập vào tài khoản |
| 17 | `dst_host_entropy` | fanout | float64 | — | Pielou evenness của phân bố sự kiện theo `LogHost` (độ tập trung/trải đều đích kết nối) |
| 18 | `missing_source_ratio` | context | float64 | — | Tỷ lệ sự kiện khuyết thông tin Source (chỉ báo chất lượng dữ liệu / nguồn ẩn danh) |
| 19 | `remote_logon_ratio` | context | float64 | — | Tỷ lệ đăng nhập từ xa ($\text{Source} \neq \text{LogHost}$) |
| 20 | `new_source_count_7d` | novelty | int32 | ✓ | Số máy nguồn ngày t chưa từng thấy trong 7 ngày lịch trước đó |
| 21 | `new_host_count_7d` | novelty | int32 | — | Số máy đích ngày t chưa từng thấy trong 7 ngày lịch trước đó |
| 22 | `days_since_last_activity` | novelty | int32 | ✓ | Số ngày kể từ phiên hoạt động trước đó (bắt tài khoản ngủ đông thức dậy) |
| 23 | `volume_robust_z_7d` | volume | float64 | — | z bền vững của $\ln(1+\text{volume})$ so với 7 ngày trước (winsorize $\pm 10$) |
| 24 | `custom_proc_share` | context | float64 | — | Tỷ lệ đăng nhập phát sinh từ tiến trình ẩn danh / ứng dụng tùy biến |
| 25 | `delta_mean_share_fail_7d` | failure | float64 | ✓ | Tỷ lệ thất bại hôm nay trừ trung bình 7 ngày trước (Brute-force, AUC 0,999) |
| 26 | `delta_mean_distinct_fail_host_7d` | failure | float64 | — | $\log(1 + \text{đích thất bại})$ hôm nay trừ trung bình 7 ngày trước (Spraying, AUC 0,990) |
| 27 | `novelty_fail_host_7d` | failure | float64 | ✓ | Số máy đích bị đăng nhập thất bại mới xuất hiện so với 7 ngày trước (Spraying, AUC 0,987) |
| 28 | `novelty_fail_failreason_7d` | failure | float64 | ✓ | Số lý do thất bại mới xuất hiện so với 7 ngày trước (Brute-force, AUC 0,728) |
| 29 | `delta_mean_distinct_source_7d` | fanout | float64 | ✓ | $\log(1 + \text{số máy nguồn})$ hôm nay trừ trung bình 7 ngày trước (Burst, AUC 1,000) |
| 30 | `jaccard_source_7d` | novelty | float64 | ✓ | Độ trùng Jaccard giữa tập máy nguồn hôm nay và 7 ngày trước (Burst, AUC 0,995) |
| 31 | `peer_z_distinct_source` | fanout | float64 | ✓ | z bền vững số máy nguồn so với các tài khoản cùng loại trong ngày (Burst, AUC 1,000) |
| 32 | `dist_shift_hour_7d` | time | float64 | ✓ | Khoảng cách Total Variation của phân bố giờ hôm nay so với 7 ngày trước (Off-hours, AUC 0,987) |
| 33 | `jaccard_hour_7d` | time | float64 | ✓ | Độ trùng Jaccard giữa tập giờ hoạt động hôm nay và 7 ngày trước (Off-hours, AUC 0,984) |
| 34 | `peer_z_share_night` | time | float64 | — | z bền vững tỷ lệ ngoài giờ so với các tài khoản cùng loại trong ngày (Off-hours, AUC 0,980) |
| 35 | `delta_mean_distinct_pair_7d` | fanout | float64 | — | $\log(1 + \text{số cặp nguồn-đích})$ hôm nay trừ trung bình 7 ngày trước (Lateral fan-out, AUC 1,000) |
| 36 | `novelty_share_pair_7d` | novelty | float64 | ✓ | Tỷ lệ sự kiện đi qua cặp nguồn-đích mới so với 7 ngày trước (Lateral fan-out, AUC 0,990) |
| 37 | `delta_mean_count_7d` | volume | float64 | ✓ | $\log(1 + \text{volume})$ hôm nay trừ trung bình 7 ngày trước (Dormant wakeup, AUC 0,999) |
| 38 | `dist_shift_logontype_7d` | logon_type | float64 | ✓ | Total Variation của cơ cấu loại logon hôm nay so với 7 ngày trước (Logon switch, AUC 1,000) |
| 39 | `jaccard_logontype_7d` | logon_type | float64 | — | Độ trùng Jaccard của tập loại logon hôm nay so với 7 ngày trước (Logon switch, AUC 1,000) |
| 40 | `delta_mean_share_night_7d` | time | float64 | ✓ | Tỷ lệ sự kiện ban đêm hôm nay trừ trung bình 7 ngày trước (Off-hours, v4.1) |
| 41 | `novelty_logontype_7d` | logon_type | float64 | ✓ | Số loại logon mới xuất hiện so với 7 ngày trước (Logon switch, v4.1) |

---

## 7. Kiểm thử

```bash
python -m pytest tests -q
```

Trên Windows, nếu `conda run` lỗi khi bị pipe, bật môi trường bằng `conda activate ueba` rồi chạy lệnh trên.

---

## 8. Cấu trúc thư mục

```text
ueba-benchmark/
├── main.py                 # Điều phối các stage: clean, features, baselines, benchmark, inject, difficulty, plots
├── environment.yml         # Môi trường conda (conda-forge, Python 3.11)
├── requirements.txt        # Thư viện cho pip
├── configs/                # Đường dẫn, schema đặc trưng, tham số mô hình, cấu hình tiêm và lưới
├── data/                   # Không nằm trong Git: raw, interim, features, processed, injection_runs
├── src/
│   ├── data/               # Log thô -> interim, khảo sát và kiểm tra chất lượng
│   ├── features/           # Ma trận tài khoản × ngày (Polars), đặc trưng lịch sử 7 ngày, tiền xử lý
│   ├── injection/          # Bộ tiêm 6 kịch bản, nhãn và manifest
│   ├── models/             # IF, HNSW-LOF, Nystroem + SGD OCSVM trên giao diện PyOD; registry, pipeline
│   ├── baselines/          # z-score, luật ngưỡng, đếm thất bại, ngẫu nhiên
│   └── evaluation/         # Chia train/test theo thời gian, chỉ số có nhãn, lưới, theo kịch bản, biểu đồ
├── scripts/
│   ├── reproduce.py        # Tái lập bảng cuối
│   ├── demo_score_day.py   # Demo dòng lệnh
│   ├── evaluation/         # Lưới, phân tích kịch bản, SHAP, ablation, review top 50
│   ├── experiments/        # So sánh cửa sổ lịch sử 7/14/21 ngày
│   ├── feature_engineering/# Phễu chọn đặc trưng, EDA
│   └── diagnostics/        # Kiểm tra phân bố điểm, độ chính xác HNSW-LOF
├── demo/                   # Demo web: backend FastAPI, frontend tĩnh, prepare.py
├── experiments/            # Kết quả lưới, phân tích, ablation, mô hình đã lưu
├── reports/                # Báo cáo theo tuần (week2–week5) và báo cáo cuối
├── docs/                   # Hướng dẫn, từ điển dữ liệu, báo cáo bộ đặc trưng
└── tests/                  # Kiểm thử pytest
```

---

## 9. Tài liệu

- [Quy trình từ dữ liệu LANL đến kết quả cuối](docs/chay_tu_du_lieu_den_ket_qua.md)
- [Tóm tắt tuần 5](reports/week5/tom_tat_tuan5.md) và [báo cáo tuần 5](reports/week5/bao_cao_tuan5.md)
- [Hướng dẫn tiêm bất thường](docs/huong_dan_tiem_bat_thuong_tuan4.md)
- [Hướng dẫn review top 50](docs/huong_dan_review_top50.md)
- [Báo cáo bộ đặc trưng v4](docs/reports/bao_cao_bo_dac_trung_v4.md)
- [Từ điển 21 trường log LANL](docs/data_dictionary.md)
- [Tóm tắt tuần 3](reports/week3/tom_tat_tuan3.md): benchmark không nhãn, chưa có bộ tiêm
