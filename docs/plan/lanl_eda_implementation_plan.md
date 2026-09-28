# Kế hoạch EDA chi tiết cho dữ liệu chuỗi thời gian LANL

## 1. Mục tiêu tài liệu

Tài liệu này mô tả kế hoạch triển khai Exploratory Data Analysis (EDA) cho bộ dữ liệu LANL Windows Security Event Log trong dự án `ueba-benchmark`. EDA không chỉ nhằm mô tả dữ liệu, mà phải tạo được cơ sở định lượng cho:

- lựa chọn đơn vị phân tích phù hợp cho UEBA;
- thiết kế đặc trưng hành vi `User × Day`;
- nhận diện các mẫu brute force, password spraying và lateral movement;
- thiết lập baseline chỉ sử dụng dữ liệu quá khứ;
- chia train/validation/test theo thời gian, tránh data leakage;
- xây dựng pipeline có thể chạy lại, kiểm thử và mở rộng.

Phạm vi chính là hai loại sự kiện:

- `4624`: đăng nhập thành công;
- `4625`: đăng nhập thất bại.

EDA được xem là hoàn thành khi tạo ra các bảng tổng hợp, biểu đồ, báo cáo kết luận và các quyết định feature engineering có thể tái lập bằng CLI; notebook chỉ dùng để trình bày và drill-down, không chứa logic nghiệp vụ duy nhất.

---

## 2. Hiện trạng và các ràng buộc đã biết

### 2.1. Quy mô dữ liệu

Theo kết quả validation hiện có:

| Thành phần | Quy mô |
|---|---:|
| Số ngày | 60 |
| File Parquet interim | 120 |
| Event 4624 | 929.103.392 |
| Event 4625 | 13.823.040 |
| Tổng event | 942.926.432 |
| Dung lượng Parquet | khoảng 11,19 GB |

Hệ quả kỹ thuật:

- không nạp toàn bộ dữ liệu vào pandas;
- không concatenate 120 file ở dạng eager;
- đọc theo ngày và chỉ chọn cột cần thiết;
- aggregate sớm, lưu bảng trung gian nhỏ dưới dạng Parquet;
- chỉ chuyển bảng đã tổng hợp sang pandas để vẽ nếu thực sự cần.

### 2.2. Cấu trúc dữ liệu hiện tại

```text
data/interim/
├── event_4624/event_4624_day-01.parquet ... day-60.parquet
└── event_4625/event_4625_day-01.parquet ... day-60.parquet
```

Schema nghiệp vụ hiện tại sử dụng các cột như:

```text
Time, EventID, UserName, DomainName, LogHost, Source,
LogonType, LogonTypeDescription, AuthenticationPackage,
FailureReason, LogonID, ProcessName, ProcessID, ...
```

### 2.3. Các vấn đề phải xử lý trước khi EDA chính thức

1. `configs/system_config.yaml` đang khai báo `TimeCreated`, `TargetUserName`, `IpAddress`, `WorkstationName`, nhưng schema interim dùng `Time`, `UserName`, `Source`, `LogHost`.
2. `notebooks/01_eda_event_logs.ipynb` đang trỏ đến `data/interim/cleaned_logs.parquet`, file không tồn tại trong kiến trúc partition hiện tại.
3. Kết quả `reports/quality_survey_summary.csv` mới chỉ chứa hai ngày kiểm chứng, chưa đại diện cho toàn bộ 60 ngày.
4. `Time` là số giây tương đối tính từ đầu bộ dữ liệu, không được diễn giải trực tiếp như Unix timestamp.
5. Chưa có ngày lịch gốc được xác nhận, do đó chưa được gán thứ trong tuần theo lịch thật. Có thể phát hiện chu kỳ 7 ngày, nhưng phải ghi rõ đây là suy luận.
6. Missing value phải được đánh giá theo ngữ cảnh EventID. Ví dụ `FailureReason` null với 4624 là hợp lệ, không phải lỗi dữ liệu.

---

## 3. Nguyên tắc thiết kế

### 3.1. Tách logic tính toán khỏi notebook

- Logic đọc, chuẩn hóa, aggregate và kiểm tra đặt trong `src/eda/`.
- CLI trong `scripts/` gọi lại các module này.
- Notebook chỉ đọc artifact đã aggregate hoặc gọi API public của module.
- Mọi chỉ số quan trọng phải có unit test.

Lý do: notebook khó kiểm thử, dễ tạo trạng thái ẩn và khó chạy lại trên toàn bộ 60 ngày.

### 3.2. Một nguồn cấu hình duy nhất

Tạo `configs/eda_config.yaml` để chứa:

- mapping tên cột canonical;
- đường dẫn input/output;
- off-hours;
- kích thước cửa sổ thời gian;
- quy tắc phân loại account;
- số ngày baseline tối thiểu;
- ngưỡng sequence/burst;
- seed và thiết lập biểu đồ.

Không hard-code ngưỡng nghiệp vụ rải rác trong notebook hoặc module.

### 3.3. Pipeline có tính lũy tiến

Mỗi stage tạo artifact độc lập. Nếu stage vẽ biểu đồ lỗi thì không phải scan lại 943 triệu event. Các artifact phải có metadata gồm:

- thời điểm sinh;
- config được sử dụng;
- danh sách ngày đầu vào;
- phiên bản schema;
- row count và checksum hoặc thống kê đối soát tối thiểu.

### 3.4. Không dùng thông tin tương lai

Mọi feature dạng baseline, novelty hoặc rolling score tại ngày `t` chỉ được dùng dữ liệu trước `t`. Đây là yêu cầu bắt buộc vì EDA sẽ định hướng feature cho mô hình anomaly detection.

---

## 4. Mô hình thời gian chuẩn

### 4.1. Các biến thời gian canonical

Với `Time` là số giây tương đối:

```python
dataset_day = (Time // 86_400) + 1
second_of_day = Time % 86_400
hour = (Time // 3_600) % 24
five_minute_bucket = (Time // 300) * 300
```

Trước khi chốt công thức, phải kiểm tra biên `Time = 0`, `Time = 86.400` và sự khớp giữa `dataset_day` với tên partition. Nếu quy ước timestamp là 1-based thì bổ sung test riêng cho biên ngày, không tự điều chỉnh công thức mà không có bằng chứng từ dữ liệu.

### 4.2. Các độ phân giải cần sử dụng

| Độ phân giải | Mục đích |
|---|---|
| 5 phút | burst, brute force, password spraying |
| 15 phút | fan-out ngắn hạn, lateral movement |
| 1 giờ | nhịp ngày, off-hours, heatmap |
| 1 ngày | trend, drift, `User × Day`, train/test |

### 4.3. Chu kỳ tuần

Khi chưa biết ngày lịch bắt đầu:

- dùng `day_index % 7` như một pha chu kỳ, không gọi ngay là Monday/Sunday;
- suy luận hai ngày có volume thấp là “weekend-like”;
- ghi rõ mức độ chắc chắn trong báo cáo;
- chỉ bật feature `is_weekend` sau khi xác nhận calendar origin hoặc có bằng chứng chu kỳ đủ mạnh.

---

## 5. Kiến trúc đầu ra đề xuất

```text
configs/
└── eda_config.yaml

src/eda/
├── __init__.py
├── config.py               # đọc và validate cấu hình
├── schema.py               # canonical columns, data contract
├── io.py                   # khám phá partition, scan cột cần thiết
├── account_classifier.py   # human/machine/service/system/unknown
├── temporal.py             # aggregate 5m/hour/day
├── entity.py               # user/source/host profiles
├── user_day.py             # ma trận đặc trưng User × Day
├── sequences.py            # fail→success, spray, burst
├── graph.py                # cạnh user/source/loghost theo ngày
├── baseline.py             # rolling baseline và novelty
├── validation.py           # reconciliation và invariant checks
├── plotting.py             # biểu đồ từ bảng aggregate
└── reporting.py            # sinh báo cáo Markdown

scripts/
└── run_lanl_eda.py

data/processed/eda/
├── system_daily.parquet
├── system_hourly.parquet
├── categorical_daily.parquet
├── entity_daily.parquet
├── user_day_features.parquet
├── user_host_day_edges.parquet
├── sequence_indicators.parquet
└── rolling_baselines.parquet

reports/eda/
├── figures/
├── tables/
├── run_metadata.json
└── lanl_eda_report.md

notebooks/
└── 02_lanl_eda_report.ipynb

tests/eda/
├── test_schema.py
├── test_temporal.py
├── test_account_classifier.py
├── test_user_day.py
├── test_sequences.py
├── test_baseline.py
└── test_reconciliation.py
```

Các module nhỏ theo trách nhiệm giúp mở rộng thêm EventID hoặc window mới mà không phải sửa một script lớn.

---

## 6. Các bảng dữ liệu cần tạo

### 6.1. `system_daily`

Khóa: `day`.

Các cột tối thiểu:

```text
count_4624, count_4625, total_events, failure_rate,
active_users, active_sources, active_loghosts,
human_event_count, machine_event_count, service_event_count,
off_hours_count, off_hours_ratio,
null_critical_count, exact_duplicate_count
```

### 6.2. `system_hourly`

Khóa: `day, hour, EventID, account_type`.

Các cột:

```text
event_count, distinct_users, distinct_sources, distinct_loghosts,
failure_count, failure_rate
```

### 6.3. `categorical_daily`

Khóa: `day, dimension, value, EventID, account_type`.

`dimension` gồm:

- `LogonType`;
- `AuthenticationPackage`;
- `FailureReason`;
- `DomainName`.

Các cột:

```text
event_count, event_share, distinct_users
```

### 6.4. `entity_daily`

Khóa: `day, entity_type, entity_id` với `entity_type` là `user`, `source` hoặc `loghost`.

Các cột:

```text
event_count, success_count, failure_count, failure_rate,
off_hours_count, distinct_users, distinct_sources,
distinct_loghosts, distinct_logon_types
```

Chỉ các cột phù hợp với từng loại entity mới được sử dụng; cột không áp dụng phải được tài liệu hóa rõ.

### 6.5. `user_day_features`

Khóa đề xuất: `day, DomainName, UserName` hoặc một `user_key` chuẩn hóa từ hai trường này.

Nhóm feature:

| Nhóm | Feature tối thiểu |
|---|---|
| Volume | `total_logons`, `successful_logons`, `failed_logons` |
| Ratio | `failure_ratio`, `off_hours_ratio` |
| Diversity | `distinct_sources`, `distinct_loghosts`, `distinct_logon_types` |
| Logon type | count của type 2, 3, 10 và nhóm khác |
| Temporal | `first_event_second`, `last_event_second`, `activity_span_hours` |
| Burst | `max_failures_5m`, `max_distinct_hosts_15m` |
| Sequence | `failed_then_success_count` |
| Novelty | `new_source_count`, `new_loghost_count`, `new_logon_type_count` |
| Distribution | `logon_hour_entropy`, `host_entropy` |

### 6.6. `user_host_day_edges`

Khóa: `day, user_key, Source, LogHost`.

Các cột:

```text
event_count, success_count, failure_count,
first_seen_time, last_seen_time,
is_new_edge, days_since_last_seen
```

### 6.7. `sequence_indicators`

Khóa tùy pattern, nhưng bắt buộc có `day`, `window_start`, `pattern_type` và các entity liên quan.

Pattern ban đầu:

- nhiều failure rồi success cùng user trong 5/15/30 phút;
- một source fail trên nhiều user trong 5/15 phút;
- một user truy cập nhiều LogHost trong 15/30 phút;
- tăng đột biến failure so với lịch sử gần.

---

## 7. Danh mục phân tích và biểu đồ

### 7.1. Tổng quan hệ thống

| Câu hỏi | Chỉ số | Biểu đồ |
|---|---|---|
| Volume thay đổi ra sao? | event/day theo EventID | line chart, log scale phụ |
| Failure có tăng tương đối không? | failure rate | line chart với median/MAD band |
| Số entity active có drift không? | users/sources/hosts | multi-line chart |
| Có nhịp 7 ngày không? | volume theo `day mod 7` | seasonal plot |

### 7.2. Nhịp trong ngày

- Heatmap `day × hour` cho tổng event.
- Heatmap riêng cho failure rate.
- Đường phân bố 24 giờ theo account type.
- So sánh off-hours ratio giữa human, machine và service account.

### 7.3. Categorical drift

- Stacked share của LogonType theo ngày.
- AuthenticationPackage theo ngày và account type.
- FailureReason theo ngày.
- Jensen-Shannon divergence hoặc Population Stability Index giữa từng ngày và baseline, nếu cần định lượng drift.

### 7.4. Long-tail entity

Không chỉ dùng Top-N. Với event/user/day, host/user/day và user/source/day phải có:

- histogram với `log1p`;
- ECDF;
- các percentile `p50`, `p90`, `p95`, `p99`, `p99.9`, `max`;
- tỷ trọng volume của top 1%, 5% entity;
- Lorenz curve tùy chọn.

### 7.5. UEBA và anomaly-oriented EDA

- Phân bố robust z-score theo từng feature.
- Số user đủ/không đủ lịch sử baseline.
- Tỷ lệ user có source/host mới mỗi ngày.
- Số cạnh user-host mới và tái xuất hiện.
- Scatter `failed_logons` và `distinct_users` theo source để tìm password spraying.
- Scatter `distinct_hosts_15m` và total volume theo user để tìm lateral movement.
- Timeline drill-down cho các case có score cao.

---

## 8. Quy tắc phân loại account

Phân loại theo thứ tự ưu tiên để tránh một account khớp nhiều nhóm:

1. `system`: danh sách explicit như `SYSTEM`, `LOCAL SERVICE`, `NETWORK SERVICE`, `ANONYMOUS LOGON`, `-`.
2. `machine`: username kết thúc bằng `$`.
3. `service`: regex hoặc danh sách được cấu hình và xác nhận từ dữ liệu.
4. `human`: account còn lại có định danh hợp lệ.
5. `unknown`: null, rỗng hoặc không thể phân loại.

EDA phải báo cáo cả:

- toàn bộ population;
- human account riêng;
- machine/service riêng.

Không xóa machine account khỏi artifact gốc. Việc lọc chỉ áp dụng khi tạo view hoặc feature matrix dành cho một thí nghiệm cụ thể.

---

## 9. Baseline, novelty và chống data leakage

### 9.1. Rolling baseline

Tại ngày `t`, baseline chỉ dùng các ngày `< t`, ưu tiên cửa sổ 7, 14 và 28 ngày. Thống kê mặc định:

```text
rolling_median
rolling_mad
rolling_p95
rolling_active_days
```

Robust score:

```text
robust_z = (x_t - rolling_median) / (1.4826 * rolling_mad + epsilon)
```

### 9.2. Cold start

Mỗi feature có baseline phải kèm:

- `history_days`;
- `baseline_available`;
- `cold_start_reason` nếu không đủ lịch sử.

Không gán user mới là anomaly chỉ vì thiếu lịch sử. Có thể so với peer/global baseline, nhưng phải đánh dấu nguồn baseline.

### 9.3. Novelty

`is_new_source`, `is_new_loghost`, `is_new_edge` chỉ được so với tập quan sát ở các ngày trước. Tuyệt đối không xây tập “đã từng thấy” từ toàn bộ 60 ngày rồi quay lại tính feature cho ngày đầu.

### 9.4. Chia dữ liệu

Split mặc định để khảo sát:

```text
Ngày 01–40: train/baseline
Ngày 41–50: validation
Ngày 51–60: test
```

Đây là cấu hình ban đầu và có thể thay đổi sau EDA, nhưng không dùng random split ở cấp record hoặc user-day.

---

## 10. Kế hoạch triển khai theo phase

## Phase 0 — Data contract và cấu hình

### Công việc

- [ ] Xác nhận schema của cả 120 file bằng Parquet metadata.
- [ ] Tạo canonical column mapping.
- [ ] Sửa hoặc tách cấu hình để không còn dùng lẫn schema raw và interim.
- [ ] Xác nhận công thức ngày/giờ tại các timestamp biên.
- [ ] Kiểm tra cặp file 4624/4625 tồn tại đủ cho từng ngày.
- [ ] Xây `src/eda/schema.py`, `config.py`, `io.py`.
- [ ] Viết test cho schema, partition discovery và boundary time.

### Đầu ra

- `configs/eda_config.yaml`;
- data contract có version;
- báo cáo schema và partition inventory.

### Tiêu chí nghiệm thu

- 120/120 file đọc được metadata;
- mọi file đúng EventID của partition;
- `dataset_day` khớp partition hoặc mọi ngoại lệ được giải thích;
- CLI fail-fast với thông báo rõ khi thiếu cột bắt buộc.

## Phase 1 — EDA chất lượng và tổng quan 2 ngày

### Công việc

- [ ] Chạy toàn bộ pipeline trên `day-01,day-02`.
- [ ] Tính row count, missing theo EventID và duplicate.
- [ ] Tạo hourly/daily aggregate.
- [ ] Phân loại account.
- [ ] Đối chiếu với `reports/validation_summary.md` hiện có.

### Đầu ra

- các bảng aggregate thử nghiệm;
- bộ biểu đồ smoke test;
- báo cáo reconciliation.

### Tiêu chí nghiệm thu

- tổng 4624/4625 khớp chính xác với input;
- mọi ratio nằm trong `[0, 1]`;
- tổng các account type bằng tổng event có UserName;
- chạy lại cho kết quả deterministic.

## Phase 2 — Tổng quan toàn bộ 60 ngày

### Công việc

- [ ] Chạy batch per-day trên 60 ngày.
- [ ] Sinh `system_daily`, `system_hourly`, `categorical_daily`.
- [ ] Vẽ trend, heatmap, nhịp 7 ngày và categorical drift.
- [ ] Đánh dấu ngày volume/rate bất thường bằng median/MAD hoặc IQR.

### Đầu ra

- ba bảng aggregate toàn cục;
- bộ biểu đồ temporal/categorical;
- danh sách ngày cần deep-dive.

### Tiêu chí nghiệm thu

- tổng count qua 60 ngày khớp 942.926.432;
- không cần giữ toàn bộ event trong RAM;
- artifact đủ nhỏ để notebook đọc trực tiếp;
- báo cáo phân biệt observation với kết luận an ninh.

## Phase 3 — Entity profile và User × Day

### Công việc

- [ ] Sinh profile theo user, source, LogHost.
- [ ] Sinh `user_day_features` cho feature cơ bản.
- [ ] Phân tích long-tail và account type.
- [ ] Kiểm tra tương quan và feature redundancy.
- [ ] Đánh giá độ thưa: active days/user, rows/user, cold-start rate.

### Đầu ra

- `entity_daily.parquet`;
- phiên bản đầu của `user_day_features.parquet`;
- bảng percentile và correlation;
- khuyến nghị giữ/bỏ/biến đổi feature.

### Tiêu chí nghiệm thu

- `total_logons = successful_logons + failed_logons`;
- các count không âm và các ratio hợp lệ;
- mỗi khóa `day, user_key` là duy nhất;
- machine/service/human có thể lọc mà không scan lại raw event.

## Phase 4 — Sequence và burst analysis

### Công việc

- [ ] Cài đặt cửa sổ 5/15/30 phút.
- [ ] Tìm failure burst theo user và source.
- [ ] Tìm failure→success sequence.
- [ ] Tìm một source tác động nhiều user.
- [ ] Tìm một user fan-out sang nhiều host.
- [ ] Tạo case table có bằng chứng để drill-down.

### Đầu ra

- `sequence_indicators.parquet`;
- bảng top cases theo từng pattern;
- timeline cho một số case đại diện.

### Tiêu chí nghiệm thu

- test bằng synthetic event có thứ tự thời gian biết trước;
- không nối sequence qua ranh giới không hợp lệ;
- xử lý rõ event cùng timestamp;
- định nghĩa pattern và window được ghi vào metadata.

## Phase 5 — Graph và novelty

### Công việc

- [ ] Aggregate cạnh `user/source/loghost` theo ngày.
- [ ] Tính fan-in, fan-out và cạnh mới.
- [ ] Tính `days_since_last_seen`.
- [ ] Phân tích cạnh hiếm, cạnh mới và chuyển community nếu khả thi.
- [ ] Join các đặc trưng graph cần thiết vào user-day.

### Đầu ra

- `user_host_day_edges.parquet`;
- daily graph metrics;
- biểu đồ cạnh mới và degree distribution.

### Tiêu chí nghiệm thu

- `is_new_edge` chỉ nhìn ngày quá khứ;
- count cạnh aggregate đối soát được với event có đủ source/host;
- missing Source được xử lý thành nhóm rõ ràng, không tự coi là cạnh hợp lệ.

## Phase 6 — Rolling baseline và anomaly-oriented EDA

### Công việc

- [ ] Tính rolling median/MAD/p95 bằng dữ liệu lịch sử.
- [ ] Tính robust z-score cho feature quan trọng.
- [ ] Xử lý cold start.
- [ ] So sánh self-baseline, peer baseline và global baseline.
- [ ] Đánh giá stability của feature qua train/validation/test.

### Đầu ra

- `rolling_baselines.parquet`;
- user-day feature matrix hoàn chỉnh;
- danh sách feature có tín hiệu và feature bất ổn;
- đề xuất threshold ban đầu cho rule-based benchmark.

### Tiêu chí nghiệm thu

- test khẳng định không dùng ngày hiện tại/tương lai trong rolling window;
- feature score không chứa `inf`;
- null do thiếu history được giữ và gắn cờ, không âm thầm điền 0;
- split theo thời gian được ghi trong metadata.

## Phase 7 — Báo cáo và bàn giao

### Công việc

- [ ] Sinh báo cáo Markdown tự động từ artifact.
- [ ] Hoàn thiện notebook trình bày và drill-down.
- [ ] Ghi decision log về calendar, account filtering, baseline window và feature set.
- [ ] Cập nhật README với lệnh chạy.
- [ ] Chạy test và pipeline smoke lần cuối trong conda.

### Đầu ra

- `reports/eda/lanl_eda_report.md`;
- `notebooks/02_lanl_eda_report.ipynb`;
- hình và bảng cuối cùng;
- hướng dẫn tái lập kết quả.

### Tiêu chí nghiệm thu

- một lệnh CLI có thể tái tạo artifact và báo cáo;
- notebook chạy từ đầu đến cuối mà không phụ thuộc state cũ;
- kết luận có số liệu và đường dẫn artifact hỗ trợ;
- test pass trong môi trường conda của dự án.

---

## 11. Thiết kế CLI dự kiến

Ví dụ giao diện:

```powershell
conda activate <ten-moi-truong-conda>

python scripts/run_lanl_eda.py `
  --config configs/eda_config.yaml `
  --stages validate,temporal,entity,user-day,sequence,graph,baseline,report `
  --days all
```

Chạy nhanh hai ngày:

```powershell
python scripts/run_lanl_eda.py `
  --config configs/eda_config.yaml `
  --stages validate,temporal,entity,user-day `
  --days day-01,day-02 `
  --force
```

CLI cần hỗ trợ tối thiểu:

```text
--config
--stages
--days
--output-dir
--force
--log-level
```

Mặc định không overwrite artifact nếu config hoặc phạm vi ngày khác; phải tạo metadata mới hoặc yêu cầu `--force`.

---

## 12. Chiến lược kiểm thử và đối soát

### 12.1. Unit test

- chuyển đổi thời gian tại biên ngày;
- account classification và thứ tự ưu tiên;
- failure rate khi mẫu số bằng 0;
- entropy với một/nhiều category;
- max consecutive failure;
- failure→success trong/ngoài window;
- novelty và rolling baseline không nhìn tương lai.

### 12.2. Integration test

Dùng synthetic Parquet nhỏ gồm:

- hai ngày;
- human, machine, service account;
- một brute-force case;
- một password-spraying case;
- một lateral-movement case;
- missing Source;
- timestamp đúng biên ngày.

Chạy toàn pipeline và so sánh artifact với expected table.

### 12.3. Reconciliation bắt buộc

Mỗi stage cần kiểm tra:

```text
sum(daily total_events) == input row count
sum(user-day total_logons) == event count trong population được chọn
success + failure == total
sum(account type counts) == classified event count
0 <= ratio <= 1
unique(output primary key) == output row count
```

Nếu không đối soát được do missing key hoặc filter, báo cáo phải nêu chính xác số record bị loại theo từng lý do.

---

## 13. Quản lý hiệu năng

### Yêu cầu

- dùng projection pushdown;
- predicate pushdown theo ngày/EventID khi có thể;
- batch theo partition ngày;
- tránh `.collect()` trước aggregation;
- tránh Python loop trên từng record;
- dùng Parquet cho bảng lớn, CSV chỉ cho summary nhỏ;
- ghi log thời gian, row count và peak memory theo stage nếu khả thi.

### Ngưỡng vận hành ban đầu

- smoke test luôn chạy trên hai ngày trước;
- full run chỉ bắt đầu khi reconciliation của smoke test pass;
- một lỗi ở ngày cụ thể phải ghi rõ partition và dừng an toàn;
- artifact đã hoàn thành của stage trước không bị xóa khi stage sau lỗi.

Không đặt cam kết thời gian chạy cứng trước khi benchmark thật trên máy hiện tại. Sau smoke test, ghi nhận tốc độ `rows/second` để ước lượng full run.

---

## 14. Các quyết định cần chốt bằng EDA

| Quyết định | Bằng chứng cần có |
|---|---|
| Có loại machine account khỏi model chính không? | volume, off-hours, logon type và failure profile theo account type |
| Off-hours 18:00–07:00 có phù hợp không? | phân bố 24 giờ của human account |
| Window user-day có quá thô không? | burst/sequence ở 5–30 phút |
| Baseline 7, 14 hay 28 ngày? | coverage, stability và cold-start rate |
| Có dùng failure ratio trực tiếp không? | phân bố denominator và zero/small-count cases |
| Feature host/source novelty có ổn định không? | new-edge rate và missing Source theo ngày |
| Có thể suy ra weekend không? | chu kỳ 7 ngày nhất quán trên volume và active entities |
| Feature nào cần `log1p`/robust scaling? | skewness, percentile và outlier profile |

Mọi quyết định được ghi trong phần “Decision log” của báo cáo cuối cùng, kèm số liệu thay vì chỉ dựa trên trực giác.

---

## 15. Rủi ro và biện pháp giảm thiểu

| Rủi ro | Tác động | Biện pháp |
|---|---|---|
| OOM khi đọc toàn bộ dữ liệu | pipeline dừng | batch per-day, lazy scan, projection pushdown |
| Sai nghĩa của Time | sai hour/day/weekend | boundary test và partition reconciliation |
| Service/machine account lấn át | baseline user bị méo | phân nhóm account và báo cáo riêng |
| Top-N tạo kết luận sai | bỏ qua long tail | percentile, ECDF, log-scale |
| Dùng tương lai để tính novelty | metric/model quá lạc quan | lag/shift bắt buộc và leakage test |
| Missing bị coi là lỗi chung | loại sai record hợp lệ | rule theo EventID và data dictionary |
| Notebook chứa logic duy nhất | khó bảo trì/tái lập | module hóa trong `src/eda/` |
| Cấu hình tên cột không khớp | code lỗi hoặc feature sai | canonical schema và fail-fast validation |
| Không có ground truth đầy đủ | khó kết luận anomaly | phân biệt anomaly candidate và confirmed attack |

---

## 16. Thứ tự ưu tiên thực thi

### Mức P0 — bắt buộc trước model

- Data contract và timestamp validation.
- Full 60-day system daily/hourly summary.
- Account segmentation.
- User-day feature cơ bản.
- Temporal split và leakage-safe baseline.
- Reconciliation tests.

### Mức P1 — rất nên có

- Failure→success sequence.
- Source-to-many-users pattern.
- User-to-many-hosts pattern.
- Novel source/host/edge.
- Entity long-tail và feature stability.

### Mức P2 — mở rộng sau baseline

- Community detection trên graph.
- Peer-group discovery tự động.
- PSI/JS divergence nâng cao.
- Interactive dashboard.
- Kết hợp thêm các EventID ngoài 4624/4625.

---

## 17. Definition of Done

EDA LANL được coi là hoàn thành khi đáp ứng đồng thời:

- [ ] Schema/config canonical đã thống nhất với interim data.
- [ ] 60 ngày được aggregate mà không nạp toàn bộ dataset vào RAM.
- [ ] Tổng event ở mọi tầng được đối soát hoặc có giải thích phần bị loại.
- [ ] Có system, temporal, categorical, entity, user-day, sequence và graph artifacts.
- [ ] Machine/service/human account được phân tích riêng.
- [ ] Novelty và rolling baseline đã có test chống leakage.
- [ ] Các biểu đồ chính được sinh tự động từ bảng aggregate.
- [ ] Báo cáo cuối nêu rõ observation, hypothesis và decision; không đồng nhất outlier với attack.
- [ ] Notebook chạy sạch từ đầu đến cuối trong môi trường conda.
- [ ] Unit test và integration test pass.
- [ ] README có lệnh chạy smoke test và full run.

---

## 18. Bước triển khai ngay tiếp theo

Thứ tự công việc cho iteration đầu tiên:

1. Tạo `configs/eda_config.yaml` và canonical schema.
2. Xây `src/eda/io.py`, `schema.py`, `account_classifier.py`.
3. Viết synthetic integration fixture và test boundary time.
4. Xây aggregate `system_daily` và `system_hourly`.
5. Chạy smoke test trên `day-01,day-02` và đối soát với báo cáo hiện có.
6. Chỉ sau khi smoke test pass mới chạy toàn bộ 60 ngày.

Iteration đầu không nên bắt đầu bằng biểu đồ hoặc notebook. Trước tiên phải bảo đảm data contract và các bảng aggregate đúng; biểu đồ khi đó chỉ là lớp trình bày có thể thay đổi dễ dàng.
