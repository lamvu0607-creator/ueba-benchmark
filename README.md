# UEBA Benchmark: Windows Security Event Log (4624 / 4625) bu cac

Hệ thống pipeline benchmark các mô hình phát hiện bất thường (Anomaly Detection) cho bài toán **Phân tích hành vi người dùng (UEBA)**, khai thác các sự kiện đăng nhập thành công (`EventID 4624`) và thất bại (`EventID 4625`) trên hệ điều hành Windows.

**Tuần 4:** xem [Hướng dẫn tiêm bất thường và kiểm tra tiến độ](docs/huong_dan_tiem_bat_thuong_tuan4.md) để hiểu 6 kịch bản, chạy từng bước, kiểm tra nhãn/tỷ lệ tiêm và đối chiếu các đầu ra cần nghiệm thu.

---

## 1. Cấu trúc thư mục dự án

```text
ueba-benchmark/
├── .gitignore                      # Chặn data thô/trung gian/features nặng, checkpoints, cache
├── README.md                       # Tài liệu hướng dẫn sử dụng & kiến trúc
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── pyproject.toml                  # Khai báo package src (hỗ trợ pip install -e .)
├── main.py                         # Entrypoint điều phối toàn bộ pipeline
├── configs/
│   ├── system_config.yaml          # Cấu hình đường dẫn 4 tầng data, seed, bộ lọc account
│   ├── feature_schema.yaml         # Hợp đồng đặc trưng v4 (39 core features, transforms)
│   └── model_params.yaml           # Siêu tham số cho IForest, LOF, OCSVM & Baselines
├── data/                           # Quản lý 4 tầng dữ liệu theo chuẩn Data Lakehouse
│   ├── raw/                        # Tầng 1: Log sự kiện gốc (EVTX / CSV / Parquet / JSON)
│   ├── interim/                    # Tầng 2: Log đã làm sạch & chuẩn hóa theo ngày (.parquet)
│   ├── features/                   # Tầng 3: Ma trận đặc trưng thô (Account x Day) (.parquet)
│   └── processed/                  # Tầng 4: Ma trận đặc trưng đã chuẩn hóa sẵn sàng train (.parquet)
├── src/                            # Mã nguồn lõi (Python Package)
│   ├── __init__.py
│   ├── data/                       # Tầng ingest & làm sạch dữ liệu
│   │   ├── raw_to_interim.py       # Chuyển đổi log thô sang interim Parquet
│   │   ├── quality_survey.py       # Khảo sát chất lượng dữ liệu đa ngày
│   │   └── validate_interim.py     # Kiểm định tính toàn vẹn của interim
│   ├── features/                   # Tầng trích xuất & tiền xử lý đặc trưng
│   │   ├── schema.py               # Quản lý và kiểm định hợp đồng Feature Schema v4
│   │   ├── extractor.py            # Trích xuất ma trận Tài khoản x Ngày bằng Polars
│   │   ├── history.py              # Trích xuất đặc trưng lịch sử 7 ngày (Novelty / Rolling)
│   │   ├── templates.py            # Không gian tổ hợp đặc trưng 4 trục (Combinatorial Templates)
│   │   ├── template_engine.py      # Bộ sinh đặc trưng template tự động theo lô
│   │   └── preprocessor.py         # Chuẩn hóa log1p, xử lý NULL có kiểm soát
│   ├── models/                     # Tầng mô hình bất thường (Phase 3: IF, LOF, OCSVM, Baselines)
│   ├── injection/                  # Bộ tiêm log: 6 kịch bản nhân bản sự kiện thật vào ngày test (tầng interim)
│   │   ├── profiles.py             # Hồ sơ TỪ TRAIN (tài khoản, máy, ngưỡng khoá L)
│   │   ├── operations.py           # Kho khuôn, nhân bản sự kiện, sinh thời gian theo profile
│   │   ├── scenarios.py            # 6 kịch bản (brute-force, spraying, ngoài giờ, máy lạ, ngủ đông, đổi LogonType)
│   │   ├── inject.py               # Điều phối một khối dev/test: chọn nạn nhân, ghi log đè + manifest + nhãn
│   │   ├── layout.py / labels.py   # Bố cục thư mục run, nhãn từ manifest
│   │   └── difficulty.py           # Hook oracle độ khó (hiện chỉ có oracle "none")
│   └── evaluation/                 # Tầng đánh giá time-split, metrics, tiêm tấn công & manifest
│       ├── injection.py            # Khung tiêm bất thường tổng hợp (Synthetic Injection)
│       ├── split.py                # Time-based train/test partition (Day 42)
│       └── metrics.py              # Đo lường alert rate, top-K overlap, stability
├── scripts/                        # Scripts phân tích chuyên sâu & công cụ bổ trợ
│   ├── convert_raw_to_interim.py
│   ├── survey_multi_days_quality.py
│   ├── validate_interim.py
│   ├── diagnostics/                # Phân tích phân bố điểm dị biệt, ổn định
│   └── feature_engineering/        # Phễu lọc đặc trưng v4, EDA, kiểm tra đa cộng tuyến
├── notebooks/                      # Jupyter Notebooks nghiên cứu & EDA
├── docs/                           # Tài liệu kỹ thuật, kiến trúc & báo cáo đa cộng tuyến
├── reports/                        # Báo cáo kết quả khảo sát & biểu đồ trực quan
├── experiments/                    # Nhật ký thực nghiệm, model weights & metrics
└── tests/                          # Bộ kiểm thử tự động (43 Unit Tests)
    ├── test_cleaner.py             # Test chuẩn hóa dữ liệu thô và sửa lỗi destination
    ├── test_configs.py             # Test cấu hình YAML & tính hợp lệ của Feature Schema
    ├── test_features.py            # Test logic phân loại thực thể & trích xuất đặc trưng
    ├── test_template_features.py   # Test đặc trưng sinh từ Combinatorial Templates
    ├── test_models.py              # Test giao diện detector, baselines, pipeline & registry
    └── test_evaluation.py          # Test time split, chỉ số đánh giá, log & manifest
```

---

## 2. Thiết lập môi trường

### 2.1 Lấy mã nguồn

```bash
git clone https://github.com/lamvu0607-creator/ueba-benchmark.git
cd ueba-benchmark
# Khi nhánh chưa được merge vào main:
git checkout feature/segmented-benchmark
```

Đã clone từ trước thì chỉ cần `git fetch origin && git checkout feature/segmented-benchmark && git pull`.

> **Lưu ý về `hnswlib`:** thư viện này là bắt buộc. `src/models/__init__.py` → `detectors` →
> `pyod_detectors` có `import hnswlib`, nên thiếu nó thì stage `baselines`, `benchmark` và phần lớn test
> lỗi ngay khi import. Bản `hnswlib` trên PyPI phải biên dịch C++. Trên Windows nên cài qua conda-forge
> hoặc dùng `chroma-hnswlib` (wheel dựng sẵn, vẫn `import hnswlib`).

### 2.2 Lựa chọn 1: Conda

```bash
conda create -n ueba-benchmark python=3.11 -y
conda activate ueba-benchmark
conda install -c conda-forge hnswlib -y   # Windows/Python 3.14: PyPI không có wheel
pip install -r requirements.txt
pip install -e .                          # cài src dưới dạng editable package (tránh lỗi import)
```

### 2.3 Lựa chọn 2: venv (không dùng Conda)

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv            # nên dùng 3.11/3.12, tránh 3.14
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install chroma-hnswlib        # hnswlib dựng sẵn, không cần trình biên dịch
pip install -r requirements.txt
pip install -e .
python -c "import hnswlib; print('ok')"
```

- Nếu `pip install -r requirements.txt` cố biên dịch `hnswlib` và báo
  *"Microsoft Visual C++ 14.0 or greater is required"*, tạm bỏ dòng `hnswlib>=0.8.0` trong
  `requirements.txt` rồi cài lại (đã có `chroma-hnswlib` thay thế).
- Nếu `chroma-hnswlib` không có wheel cho phiên bản Python đang dùng, chọn một trong hai:
  - đổi sang Python 3.11 hoặc 3.12;
  - cài **Visual Studio Build Tools** (workload "Desktop development with C++"), rồi `pip install hnswlib`.

**Linux / macOS** (thường đã có sẵn gcc/clang để biên dịch `hnswlib`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2.4 Dữ liệu và các artifact không nằm trong Git

`data/` và phần lớn `experiments/results/` bị `.gitignore`, nên sau khi clone cần làm thêm hai việc:

1. **Tải dữ liệu interim** (khoảng 22 GB, xem mục 6). Giải nén sao cho có
   `data/interim/event_4624/event_4624_day-01.parquet … day-60`, và tương tự với `event_4625`.
2. **Tạo lại các artifact**, đúng thứ tự (`baselines` đọc `data/processed/` nên phải chạy sau `features`):

```bash
python -m pytest tests -q                      # kỳ vọng: toàn bộ passed
python main.py --stage features                # -> data/features/raw/, data/processed/
python scripts/injection_profiles_summary.py   # -> data/features/train_profiles/ (chỉ đọc ngày 1..42)
python main.py --stage baselines               # -> experiments/results/baselines/ (trên log gốc)
python main.py --stage benchmark               # tuỳ chọn: chạy lại các mô hình ML
```

Hai con số để đối chiếu với máy khác:
- bản tóm tắt hồ sơ train ra **29.830 tài khoản, 15.946 máy, L = 5**;
- `experiments/results/baselines/rule_event_stats.json` có `"injected_events_dir": null`.

---

## 3. Chạy Pipeline điều phối (`main.py`)

Hệ thống cung cấp entrypoint tập trung `main.py` để điều phối pipeline từ làm sạch dữ liệu đến trích xuất đặc trưng và đánh giá mô hình.

### Chạy toàn bộ quy trình:

```bash
python main.py --stage all
```

> **Lưu ý**: Đảm bảo đã đặt file log Windows Security Event 4624/4625 vào thư mục `data/raw/` trước khi chạy stage `clean`. Nếu đã có dữ liệu làm sạch tại `data/interim/`, bạn có thể chạy thẳng từ stage `features`.

### Chạy theo từng giai đoạn (Stage):

```bash
# 1. Chỉ làm sạch dữ liệu thô (raw -> interim)
python main.py --stage clean

# 2. Trích xuất và chuẩn hóa 39 đặc trưng hành vi cốt lõi (interim -> features -> processed)
python main.py --stage features

# 2b. Trích xuất đặc trưng cho dải ngày cụ thể (ví dụ: Day 1 đến Day 10)
python main.py --stage features --start-day 1 --end-day 10

# 3. Chạy benchmark các mô hình bất thường (chọn mô hình hoặc chạy toàn bộ 6 mô hình)
python main.py --stage benchmark --models isolation_forest local_outlier_factor one_class_svm
# Hoặc chạy toàn bộ 6 mô hình mặc định:
python main.py --stage benchmark
```

### Chạy một run tiêm log (`configs/injection.yaml`)

Mỗi run ghi vào `data/injection_runs/<run_id>/` (đã ignore). Sau bước `inject`, mọi stage nhận
`--events-dir <run>/events_injected` sẽ đọc ma trận + nhãn **của run** và ghi kết quả vào chính thư mục run,
không đè `data/processed/` hay `experiments/` của log gốc. Cần có hồ sơ train trước
(`python scripts/injection_profiles_summary.py`) và kết quả baselines trên log gốc (để bỏ tài khoản-ngày luật đã cờ).

```bash
python main.py --stage inject --block dev                                  # -> data/injection_runs/dev_seed<seed>/
RUN=data/injection_runs/dev_seed<seed>
python main.py --stage features   --events-dir $RUN/events_injected       # -> $RUN/features/raw, $RUN/processed
python main.py --stage baselines  --events-dir $RUN/events_injected       # -> $RUN/results/baselines (nhãn $RUN/labels.parquet)
python main.py --stage benchmark  --events-dir $RUN/events_injected       # -> $RUN/results, $RUN/models
python main.py --stage difficulty --events-dir $RUN/events_injected       # -> $RUN/difficulty.parquet (oracle "none")
```

`run_config.json` của run cố định ngày đánh giá: dev 43–51, test 52–60. Features vẫn tính
đầy đủ lịch sử, còn benchmark và baseline chỉ đánh giá khối đã chọn (kể cả ngày không được tiêm).
Run cũ dùng `split` trong manifest và cấu hình injection để xác định khối.

Khối `test` (`--block test`) chỉ chạy **một lần** ở cuối, sau khi đã chốt tham số trên `dev`.

### Biểu đồ kết quả tuần 4

Sau benchmark có nhãn, vẽ từ điểm đã lưu (không huấn luyện lại):

```bash
python main.py --stage plots --events-dir $RUN/events_injected
```

PNG/PDF được ghi vào `$RUN/results/figures/week4/`, tách theo phân khúc: PR, AP,
Precision@k, Recall theo ngân sách ngày, thời gian chạy và heatmap kịch bản nếu có nhãn kịch bản.
Có CSV số liệu và manifest truy vết; hỗ trợ tổng hợp nhiều lần benchmark đầy đủ qua CLI riêng.
Xem [hướng dẫn vẽ biểu đồ tuần 4](docs/week4_result_plots.md).

---

## 4. Danh sách 39 Đặc trưng hành vi cốt lõi (Feature Schema v4.0)

Bộ đặc trưng được chuẩn hóa theo hợp đồng [`configs/feature_schema.yaml`](configs/feature_schema.yaml) sau quá trình phân tích tương quan Spearman, kiểm soát đa cộng tuyến (VIF) và phễu lọc 2 vòng.

**Quy trình sinh đặc trưng v4.0 (Combinatorial Template + Detection Engineering):**
* **Không gian tổ hợp 4 trục** (`src/features/templates.py`): *14 Phép đo × 4 Đối tượng × 8 Chiều × 7 Cửa sổ* = **3.136 tổ hợp** $\to$ lọc ngữ pháp còn 471 tổ hợp hợp lệ.
* **Vòng 1 (Lập luận an toàn thông tin & luật R1–R4):** Lọc bỏ biến trùng lặp, biến không gắn kịch bản tấn công, biến khó diễn giải (> 3 đơn vị giải thích) $\to$ còn **78 ứng viên**.
* **Vòng 2 (4 Cổng số liệu đo trên dữ liệu thật LANL):**
  * **Cổng E (Rò rỉ thời gian):** Đảm bảo $\Delta = 0$ khi cắt dữ liệu lịch sử $\le t-1$.
  * **Cổng B (Thông tin):** Loại bỏ biến có tỷ lệ NULL $> 50\%$ hoặc hằng số $> 99\%$.
  * **Cổng A (Đa cộng tuyến):** Yêu cầu $\max(|\text{Pearson}|, |\text{Spearman}|) < 0,85$ với 24 core và các biến đã chọn.
  * **Cổng C (Phân tách kịch bản):** Kiểm tra khả năng phát hiện trên 7 kịch bản tiêm bất thường ($\text{AUC} \ge 0,70$).
* **Chốt quy mô (VIF $\le 10$):** Chốt **15 đặc trưng mới** vào core (nâng từ 24 lên **39 core features**) và 11 biến vào danh sách dự bị (`reserve:`). Chi tiết phương pháp luận: [`docs/reports/bao_cao_bo_dac_trung_v4.md`](docs/reports/bao_cao_bo_dac_trung_v4.md).

| STT | Tên đặc trưng | Nhóm | Mô tả ý nghĩa an toàn thông tin |
| :---: | :--- | :--- | :--- |
| 1 | `log_total_logons` | Volume | $\log(1 + \text{tổng sự kiện trong ngày})$, phản ánh cường độ hoạt động tổng thể |
| 2 | `failure_ratio` | Failure | Tỷ lệ đăng nhập thất bại ($\text{failures} / \text{total}$), chỉ báo brute force/dò mật khẩu |
| 3 | `failure_locked_out_share` | Failure | Tỷ trọng sự kiện khóa tài khoản trong tổng số lần thất bại (tín hiệu lockout) |
| 4 | `off_hours_ratio` | Time | Tỷ lệ đăng nhập ngoài giờ hành chính (cửa sổ 18h - 7h sáng) |
| 5 | `interarrival_dt_mean` | Rhythm | Khoảng cách thời gian trung bình (giây) giữa 2 sự kiện liên tiếp của tài khoản |
| 6 | `delta_t_cv` | Rhythm | Hệ số biến thiên khoảng cách thời gian ($\text{std} / \text{mean}$), đo tính tuần hoàn/đều đặn |
| 7 | `same_second_share` | Rhythm | Tỷ lệ sự kiện diễn ra cùng giây ($\Delta t = 0$), phát hiện đăng nhập tự động dạng script/bot |
| 8 | `is_single_event` | Rhythm | Cờ nhị phân đánh dấu tài khoản chỉ có duy nhất 1 sự kiện trong ngày |
| 9 | `interactive_ratio` | Logon Type | Tỷ lệ đăng nhập tương tác cục bộ (`LogonType 2`) |
| 10 | `rare_logon_type_count_log` | Logon Type | $\log(1 + \text{số sự kiện có kiểu hiếm})$, bao gồm Remote Desktop (Type 10), RunAs (Type 9) |
| 11 | `ntlm_ratio` | Auth | Tỷ lệ sử dụng giao thức xác thực NTLM (thường tương quan cao với các bất thường) |
| 12 | `log_distinct_hosts` | Fan-out | $\log(1 + \text{số máy trạm đích duy nhất})$, phát hiện dò quét mạng (lateral movement) |
| 13 | `distinct_sources_count` | Fan-out | Số lượng máy nguồn duy nhất thực hiện đăng nhập vào tài khoản |
| 14 | `missing_source_ratio` | Context | Tỷ lệ sự kiện khuyết thông tin Source (chỉ báo chất lượng dữ liệu / nguồn ẩn danh) |
| 15 | `remote_logon_ratio` | Context | Tỷ lệ đăng nhập từ xa ($\text{Source} \neq \text{LogHost}$) |
| 16 | `custom_proc_share` | Context | Tỷ lệ đăng nhập phát sinh từ tiến trình ẩn danh / ứng dụng tùy biến |
| 17 | `activity_peak_hour_sin` | Rhythm (v3.0) | $\sin(2\pi \cdot h_{\text{peak}} / 24)$ — giờ cao điểm mã hoá chu kỳ |
| 18 | `activity_peak_hour_cos` | Rhythm (v3.0) | $\cos(2\pi \cdot h_{\text{peak}} / 24)$ — cặp với #17 (sin/cos phải đi cùng nhau) |
| 19 | `hour_entropy` | Rhythm (v3.0) | Pielou evenness $H / \ln(S)$ của phân bố sự kiện trên các khung giờ có mặt (độ trải đều nhịp) |
| 20 | `dst_host_entropy` | Fan-out (v3.0) | Pielou evenness của phân bố sự kiện theo `LogHost` (độ tập trung/trải đều đích kết nối) |
| 21 | `new_source_count_7d` | Novelty (v3.0) | Số máy nguồn ngày t chưa từng thấy trong 7 ngày lịch trước đó |
| 22 | `new_host_count_7d` | Novelty (v3.0) | Số máy đích ngày t chưa từng thấy trong 7 ngày lịch trước đó |
| 23 | `days_since_last_activity` | Novelty (v3.0) | Số ngày kể từ phiên hoạt động trước đó (bắt tài khoản ngủ đông thức dậy) |
| 24 | `volume_robust_z_7d` | Deviation (v3.0) | z bền vững của $\ln(1+\text{volume})$ so với 7 ngày trước (winsorize $\pm 10$) |
| **25** | **`delta_mean_share_fail_7d`** | **Failure (v4.0)** | Tỷ lệ thất bại hôm nay trừ trung bình 7 ngày trước (Brute-force, AUC 0,999) |
| **26** | **`delta_mean_distinct_fail_host_7d`** | **Failure (v4.0)** | $\log(1 + \text{đích thất bại})$ hôm nay trừ trung bình 7 ngày trước (Spraying, AUC 0,990) |
| **27** | **`novelty_fail_host_7d`** | **Failure (v4.0)** | Số máy đích bị đăng nhập thất bại mới xuất hiện so với 7 ngày trước (Spraying, AUC 0,987) |
| **28** | **`novelty_fail_failreason_7d`** | **Failure (v4.0)** | Số lý do thất bại mới xuất hiện so với 7 ngày trước (Brute-force, AUC 0,728) |
| **29** | **`delta_mean_distinct_source_7d`** | **Fan-out (v4.0)** | $\log(1 + \text{số máy nguồn})$ hôm nay trừ trung bình 7 ngày trước (Burst, AUC 1,000) |
| **30** | **`jaccard_source_7d`** | **Novelty (v4.0)** | Độ trùng Jaccard giữa tập máy nguồn hôm nay và 7 ngày trước (Burst, AUC 0,995) |
| **31** | **`peer_z_distinct_source`** | **Peer (v4.0)** | z bền vững số máy nguồn so với các tài khoản cùng loại trong ngày (Burst, AUC 1,000) |
| **32** | **`dist_shift_hour_7d`** | **Time (v4.0)** | Khoảng cách Total Variation của phân bố giờ hôm nay so với 7 ngày trước (Off-hours, AUC 0,987) |
| **33** | **`jaccard_hour_7d`** | **Time (v4.0)** | Độ trùng Jaccard giữa tập giờ hoạt động hôm nay và 7 ngày trước (Off-hours, AUC 0,984) |
| **34** | **`peer_z_share_night`** | **Peer (v4.0)** | z bền vững tỷ lệ ngoài giờ so với các tài khoản cùng loại trong ngày (Off-hours, AUC 0,980) |
| **35** | **`delta_mean_distinct_pair_7d`** | **Lateral (v4.0)** | $\log(1 + \text{số cặp nguồn-đích})$ hôm nay trừ trung bình 7 ngày trước (Lateral fan-out, AUC 1,000) |
| **36** | **`novelty_share_pair_7d`** | **Lateral (v4.0)** | Tỷ lệ sự kiện đi qua cặp nguồn-đích mới so với 7 ngày trước (Lateral fan-out, AUC 0,990) |
| **37** | **`delta_mean_count_7d`** | **Volume (v4.0)** | $\log(1 + \text{volume})$ hôm nay trừ trung bình 7 ngày trước (Dormant wakeup, AUC 0,999) |
| **38** | **`dist_shift_logontype_7d`** | **Logon Type (v4.0)**| Total Variation của cơ cấu loại logon hôm nay so với 7 ngày trước (Logon switch, AUC 1,000) |
| **39** | **`jaccard_logontype_7d`** | **Logon Type (v4.0)**| Độ trùng Jaccard của tập loại logon hôm nay so với 7 ngày trước (Logon switch, AUC 1,000) |

> **Ghi chú về thiết kế:** Bộ 39 đặc trưng v4.0 bảo đảm tính nhân quả (causality) tuyệt đối khi chỉ sử dụng dữ liệu $\le t-1$ cho các đặc trưng lịch sử (riêng `peer_z` tính lát cắt ngang trong ngày $t$ theo `entity_type`). Toàn bộ 39 đặc trưng đều đạt VIF $\le 8,52$ (không còn đa cộng tuyến), max $|\rho| < 0,85$, và phủ kín 7 kịch bản tấn công UEBA với AUC $\ge 0,98$.

---

## 5. Kiểm thử tự động (Unit Tests)

Bộ kiểm thử đảm bảo tính toàn vẹn của cấu hình hệ thống, hợp đồng đặc trưng v4, giao diện mô hình
(`BaseAnomalyModel` + `AnomalyPipeline`), time-based split và các chỉ số đánh giá:

```bash
conda run -n ueba-benchmark python -m pytest tests/ -q
```
hoặc chạy qua môi trường ảo:
```bash
.venv\Scripts\python -m pytest tests/
```

> Trên Windows, `conda run` có thể lỗi khi bị pipe; khi đó gọi trực tiếp python của env:
> `& "$env:CONDA_PREFIX\python.exe" -m pytest tests/ -q`

---

## 6. Dữ liệu đã xử lý & Tải về từ Google Drive

Do tập dữ liệu 60 ngày (Windows Event 4624/4625) có dung lượng rất lớn (>22 GB) nên được lưu trữ ngoài Git trên Google Drive:

### Liên kết tải dữ liệu:
* **Gói dữ liệu 1 (Google Drive Part 1):** [Tải xuống tại đây](https://drive.google.com/file/d/1aMz0oPItXDYn7u326Pl_jTnmdqFvsCmE/view?usp=sharing)
* **Gói dữ liệu 2 (Google Drive Part 2):** [Tải xuống tại đây](https://drive.google.com/file/d/1FjXHtZAjKQxYtN2sv7OO6YMkNbqs00ek/view?usp=sharing)

### Hướng dẫn sử dụng:
1. Tải 2 gói dữ liệu từ link trên.
2. Giải nén vào thư mục `data/` trong dự án (`data/cleaned/` hoặc `data/interim/`).
3. Chạy trực tiếp benchmark:
   ```bash
   python main.py --stage benchmark
   ```

### Kết quả Benchmark label-free (Tuần 3) — chia tập theo THỜI GIAN

> ⚠️ **Số liệu cũ đã được sửa.** Bảng "1,055,283 mẫu đánh giá" trước đây là **artifact của rò rỉ dữ liệu**:
> mô hình được fit và chấm điểm trên **cùng một ma trận** 60 ngày (mọi điểm số đều "đã thấy" chính nó).
> Từ Tuần 3, benchmark dùng time-based split: **train = ngày 1–42 (721.612 dòng)**,
> **test = ngày 43–60 (333.671 dòng)**; imputer/scaler chỉ được fit trên train.

Tái lập: `python main.py --stage benchmark` (seed 42, K=20, ngân sách cảnh báo 5%).

Bảng kết quả chính thức được chạy trên ma trận **39 đặc trưng cốt lõi (Schema v4.0)** với đầy đủ **6 mô hình canonical** trong registry (mặc định lấy mẫu con 50.000 dòng để so sánh công bằng):

| Mô hình | n_fit | n_eval | Tỷ lệ cảnh báo (%) | Lệch ngân sách (pp) | Fit (s) | Chấm điểm (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Isolation Forest** | 50.000 | 333.671 | 6,06 | +1,06 | 4,83 | 2,32 |
| **One-Class SVM** | 50.000 | 333.671 | 6,64 | +1,64 | 20,13 | 44,31 |
| **Local Outlier Factor** | 50.000 | 333.671 | 8,52 | +3,52 | 8,56 | 13,10 |
| **Z-score Baseline** | 50.000 | 333.671 | 6,80 | +1,80 | 2,86 | 0,28 |
| **Rule-Threshold Baseline** | 50.000 | 333.671 | 20,35 | +15,35 | 2,67 | 0,22 |
| **Random Baseline** | 50.000 | 333.671 | 4,98 | −0,02 | 2,67 | 0,53 |

> **Ghi chú về Random Baseline:** Đây là mốc dưới (lower bound) sử dụng thuật toán băm ngẫu nhiên tất định theo dòng (ROC-AUC kỳ vọng ≈ 0,5, alert rate ≈ ngân sách 5%) dùng để đối chứng, kiểm tra xem các mô hình học máy và baseline luật có thực sự vượt qua mức "đoán mò" hay không.

* **Bảng xếp hạng chi tiết**: `experiments/results/benchmark_summary.csv`
* **Điểm số dị biệt** (mỗi mô hình 3 cột `_score`/`_pct`/`_anomaly`): `experiments/results/anomaly_scores.parquet`
* **Độ ổn định đa seed** (3 seed, mẫu 50.000 dòng): `experiments/results/model_stability.csv`
* **Trùng nhau Top-20 giữa các mô hình**: `experiments/results/model_topk_overlap.csv` — các mô hình gần như **không đồng thuận** (đây là lý do Tuần 4 cần nhãn thật để xếp hạng)
* **Manifest tái lập** (commit, sha256 dữ liệu, phiên bản thư viện): `experiments/results/run_manifest.json`
* **Nhật ký thí nghiệm** (tương thích 18 cột cũ + cột truy vết mới): `experiments/logs/experiment_log.csv`

Tài liệu tham khảo:
* **Báo cáo bộ đặc trưng v4.0 (39 core)**: [`docs/reports/bao_cao_bo_dac_trung_v4.md`](docs/reports/bao_cao_bo_dac_trung_v4.md)
* **Báo cáo bộ đặc trưng v3.0 (24 core)**: [`docs/reports/bao_cao_bo_dac_trung_v3.md`](docs/reports/bao_cao_bo_dac_trung_v3.md)
* **Báo cáo tóm tắt Tuần 3**: [`reports/week3/tom_tat_tuan3.md`](reports/week3/tom_tat_tuan3.md)
* **Phân tích chi tiết & bằng chứng label-free**: [`reports/week3/label_free_benchmark.md`](reports/week3/label_free_benchmark.md)
* **Từ điển dữ liệu 21 trường log LANL**: [`docs/data_dictionary.md`](docs/data_dictionary.md)
