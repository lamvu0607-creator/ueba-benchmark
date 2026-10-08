# Biểu đồ kết quả tuần 4

Phần vẽ chỉ đọc điểm đã lưu, không fit lại mô hình. Đơn vị đánh giá là **tài khoản × ngày**;
điểm cao hơn nghĩa là bất thường hơn. Mỗi phân khúc (`User`, `Machine`, hoặc `all`) có bộ hình riêng.

## Chạy trên một run đã có kết quả

Chạy `features`, `benchmark` có nhãn và, nếu cần, `baselines` trước. Sau đó:

```bash
python main.py --stage plots --events-dir data/injection_runs/dev_seed42/events_injected
```

Đường dẫn run chỉ là ví dụ: thay `dev_seed42` bằng run thực tế. Kết quả nằm ở
`<run>/results/figures/week4/`. `plots` là stage riêng, không được gọi ngầm bởi `all`.
Nhãn được lấy từ `<run>/labels.parquet`; nếu dùng dữ liệu ngoài run:

```bash
python scripts/evaluation/plot_benchmark_results.py \
  --results-dir experiments/results \
  --labels path/to/labels.parquet \
  --output-dir reports/week4/figures
```

Nếu `anomaly_scores.parquet` đã có cột `label`, có thể bỏ `--labels`. Tuy nhiên, file nhãn gốc vẫn
cần thiết để có `scenario`, `eval_exclude`, hoặc đối chiếu baseline có các dòng đã bị ML loại.
Nếu cung cấp nhãn mới khác nhãn đã lưu trong điểm, chương trình dừng để tránh ghép nhầm thí nghiệm.
Run không có nhãn sẽ báo lỗi; không tạo số liệu chất lượng phát hiện từ kết quả label-free tuần 3.

## Các biểu đồ

| Tên file (tiền tố là phân khúc) | Nội dung và cách đọc |
|---|---|
| `*_pr_curve` | Precision theo Recall, từ **thư mục kết quả đầu tiên**. Legend ghi AP; đường gạch là tỷ lệ nhãn dương của lần chạy đó, làm mốc ngẫu nhiên gần đúng. |
| `*_average_precision` | AP trung bình qua các lần chạy, thanh độ lệch chuẩn mẫu (`ddof=1`), kèm số lần hợp lệ. Một lần chạy không có ước lượng độ lệch chuẩn. |
| `*_precision_at_k` | Precision@10/50/100 (hoặc K tự chọn) trên **toàn bộ khối đánh giá của từng phân khúc**; không phải Top-K mỗi ngày. |
| `*_recall_daily_budget` | Recall khi lấy tối đa N dòng điểm cao nhất **mỗi ngày, trong mỗi phân khúc**. Mẫu số là tổng nhãn dương của khối. |
| `*_runtime` | Thời gian fit và score ở hai ô riêng; đọc từ `benchmark_summary.csv`. Baseline chưa có thời gian thì không dựng cột giả bằng 0. |
| `*_scenario_recall` | Có khi nhãn có `scenario`: Recall theo kịch bản, cùng ngân sách ngày. Mặc định chọn phần tử giữa của danh sách ngân sách đã sắp tăng (50 với [10,20,50,100]). |

AP là `sklearn.metrics.average_precision_score`, cùng định nghĩa của cột `pr_auc` trong benchmark,
không phải tích phân hình thang của đường PR. Khi vẽ PR chỉ bỏ các điểm trung gian dư thừa bằng
`precision_recall_curve(..., drop_intermediate=True)`; AP luôn tính trực tiếp từ nhãn và điểm.

Chương trình dùng bộ đọc nhãn chung, loại `eval_exclude` trước mọi phép tính; dòng không xuất hiện trong
file nhãn được coi là 0 theo quy ước nhãn tổng hợp. Nhãn 0 chưa chứng minh hành vi lành tính thật.
AP/PR cần cả hai lớp; Recall cần nhãn dương; K lớn hơn số dòng đánh giá có kết quả không xác định (NaN).
Các trường hợp này được ghi rõ trong CSV, không đổi thành kết quả tốt giả tạo.

## Tổng hợp nhiều lần chạy

Mỗi thư mục phải chứa kết quả **benchmark đầy đủ**, không phải chỉ chạy độ ổn định trên mẫu con.
Ví dụ có năm lần benchmark với các seed mô hình khác nhau trên cùng ma trận đã tiêm:

```bash
python scripts/evaluation/plot_benchmark_results.py \
  --results-dir runs/seed42/results runs/seed7/results runs/seed2024/results runs/seed1/results runs/seed2/results \
  --output-dir reports/week4/multi_seed \
  --precision-ks 10 50 100 \
  --daily-budgets 10 20 50 100 \
  --formats png pdf
```

Dùng `--labels` với đúng **một file cho mỗi thư mục** nếu nhãn chưa nằm trong điểm hoặc chưa có
`<results-dir>/../labels.parquet`. Có thể dùng cùng một đường dẫn nhãn nhiều lần khi ma trận tiêm không đổi.
Mỗi benchmark cần ghi vào thư mục riêng; không chạy đè lên cùng thư mục rồi mong tái lập năm kết quả.
Các seed trong `model_stability.csv` không cung cấp đủ điểm/chỉ số benchmark để dựng độ lệch chuẩn AP.

Các chỉ số được tính riêng từng lần rồi lấy trung bình, không nối điểm của nhiều lần thành một tập lớn.
Cần `run_manifest.json` cho chế độ nhiều lần. Chương trình từ chối trộn dev/test, phân khúc/phương pháp
khác nhau, khác đặc trưng/siêu tham số mô hình hoặc các bản sao của cùng điểm, nhãn và seed.
Mô hình tất định cho cùng kết quả ở các seed khác nhau vẫn hợp lệ, với độ lệch chuẩn bằng 0.
So sánh cấu hình khác nhau phải xuất thành bộ hình riêng; không gộp chúng thành các seed của một cấu hình.

Số lần trên hình là **số thư mục kết quả**, không tự diễn giải thành số seed mô hình hay seed tiêm.
Chốt phương án lặp (đổi seed mô hình, seed tiêm, hoặc cả hai) và ghi rõ trong báo cáo.
Các ngân sách trên áp dụng riêng cho mỗi phân khúc: N cho User và N cho Machine có thể tạo tối đa 2N
cảnh báo/ngày. Đây chưa phải ngân sách N chung cho toàn hệ thống.

## So sánh baseline

Nếu tồn tại `<results-dir>/baselines/baseline_scores.parquet`, tự lấy các dòng `split=eval` và ghép
bằng khóa, không dùng dòng train. Baseline và ML phải có **chính xác cùng tập tài khoản × ngày và phân khúc**
sau khi loại `eval_exclude`; sai phạm vi sẽ báo lỗi. Các baseline trong module riêng có tiền tố
`baseline:` để phân biệt với baseline trong registry mô hình. Dùng `--no-baselines` trong CLI riêng để bỏ qua.

## Đầu ra và truy vết

Mặc định xuất PNG (180 DPI) và PDF; có thể chọn SVG và DPI qua CLI riêng.

- `metrics.csv`: AP, Precision@k, thời gian, tỷ lệ dương, seed mô hình và đường dẫn cho từng lần/phân khúc/phương pháp.
- `metrics_summary.csv`: trung bình, độ lệch chuẩn mẫu, số quan sát hợp lệ của từng chỉ số.
- `pr_curves.csv`: tọa độ đường PR của lần tham chiếu, không chọn seed có kết quả tốt nhất.
- `daily_recall.csv`: Recall và số cảnh báo thực tế cho từng ngân sách/ngày/phân khúc/lần chạy.
- `daily_recall_summary.csv`: trung bình, độ lệch chuẩn và số lần hợp lệ cho từng điểm trên đường ngân sách.
- `scenario_recall.csv`: có khi có nhãn kịch bản; số dương và Recall từng lần.
- `plot_manifest.json`: nguồn dữ liệu, fingerprint, config/commit nguồn, số lần, cách tính và danh sách hình.

Với `main.py`, dùng `--plots-output-dir` để đổi thư mục, `--plot-formats png svg` để đổi định dạng.
K và ngân sách đọc từ `evaluation.precision_ks` / `evaluation.daily_budgets` của cấu hình hệ thống.

Điểm mới được lưu đủ độ chính xác float để tái tính đúng xếp hạng. Artifact cũ làm tròn 6 chữ số vẫn
đọc được, nhưng có thể tạo đồng hạng; số liệu vẽ tính lại từ **điểm đã lưu**, có thể lệch nhẹ so với
summary cũ tính từ điểm chưa làm tròn. Không sửa điểm để ép khớp con số cũ.
