# Từ dữ liệu LANL đến benchmark UEBA cuối

Chạy các lệnh từ thư mục gốc `ueba-benchmark/`. Cài Python và thư viện theo [README mục 2](../README.md#2-thiết-lập-môi-trường). Luồng chuẩn để tái lập bảng cuối là **interim → đặc trưng → hồ sơ và baseline → log tiêm có nhãn → lưới 22 đặc trưng → đối chiếu → demo**. `main.py --stage all` là luồng raw/clean/benchmark cũ; với bộ dữ liệu và bảng cuối trong tài liệu này, dùng `scripts/reproduce.py`.

## 1. Dữ liệu từ đâu?

Nguồn gốc là [LANL Unified Host and Network Data Set](https://csr.lanl.gov/data/2017/). LANL công bố host log Windows theo từng ngày dưới tên `wls_day-XX.bz2` (JSON Lines), khoảng 90 ngày. Dự án chỉ dùng **ngày 01–60** và hai EventID: **4624** (đăng nhập thành công), **4625** (thất bại); không dùng Netflow hay các EventID khác. Xem [từ điển 21 trường](data_dictionary.md).

Chọn **một** đường lấy `data/interim/`:

1. **Workspace này:** 120 file Parquet đang ở `../ueba-demo/data/interim/`. Hai liên kết trong `data/interim/` của `ueba-benchmark` đã trỏ tới đó. Nếu dựng lại liên kết trên Linux/macOS:

   ```bash
   ln -s ../../../ueba-demo/data/interim/event_4624 data/interim/event_4624
   ln -s ../../../ueba-demo/data/interim/event_4625 data/interim/event_4625
   ```

2. **Từ bản Parquet nhóm đã chuẩn bị:** tải hai gói Google Drive ở [README mục 6](../README.md#6-dữ-liệu-đã-xử-lý--tải-về-từ-google-drive), giải nén để có `data/interim/event_4624/event_4624_day-01.parquet` đến `day-60.parquet` và bộ `event_4625` tương ứng. Đây là bản sao do dự án cung cấp; dùng hash bên dưới để xác nhận đúng phiên bản.
3. **Từ log gốc LANL:** tải riêng host log `wls_day-01.bz2` đến `wls_day-60.bz2` từ trang LANL vào `data/raw/`, rồi chạy:

   ```bash
   python scripts/convert_raw_to_interim.py --raw-dir data/raw --interim-dir data/interim
   ```

   Script lọc 4624/4625, chuẩn hóa 21 cột và ghi hai file Parquet cho mỗi ngày. Chỉ chạy đường này vào kho `interim` trống; đừng ghi đè các liên kết của workspace đang dùng. Sau đó kiểm tra theo lệnh dưới. Chưa xác minh rằng chuyển lại từ log gốc bằng phiên bản môi trường khác tạo ra Parquet **trùng byte** với bản tham chiếu; nếu hash lệch, không dùng kết quả đó để tuyên bố tái lập bảng cuối.

```bash
python scripts/validate_interim.py --mode fast
python -c "from pathlib import Path; from scripts.reproduce import verify_interim; verify_interim(Path('data/interim'))"
```

Kho đầu vào đúng có **60 ngày × 2 EventID = 120 file**. Lệnh thứ hai so SHA-256 của từng file với [manifest đã chốt](../configs/interim_sha256.txt). `data/cleaned/` và các file `downloads/*.zip` là dữ liệu phụ trợ của các giai đoạn cũ; chúng không thay thế 120 file interim của benchmark cuối.

## 2. Chạy đến bảng kết quả cuối

```bash
python scripts/reproduce.py
```

Lệnh này chạy theo đúng thứ tự sau:

| Bước | Đầu vào → đầu ra | Vì sao cần |
|---|---|---|
| Kiểm thử và hash | Mã + 120 file interim → báo lỗi nếu thiếu/sai | Khóa phiên bản dữ liệu trước khi tính |
| Đặc trưng | Interim → `data/features/raw/feature_matrix_raw.parquet` → `data/processed/feature_matrix_processed.parquet` | Ma trận tài khoản × ngày, đủ 41 đặc trưng core; lệnh ép đọc interim dù `data/cleaned/` tồn tại |
| Hồ sơ, baseline | Ngày train 01–42 → `data/features/train_profiles/`, `experiments/results/baselines/` | Chỉ dùng quá khứ để chọn nạn nhân/khuôn và loại cảnh báo luật đã có |
| Tiêm 5 run test | Interim + seed 20261061–20261065 → `data/injection_runs/test_seed*/` | Tạo sáu kịch bản bất thường trên ngày 43–60 và nhãn tổng hợp; tính lại ma trận cho mỗi run |
| Lưới 22 đặc trưng | 5 run + [cấu hình đã chốt](../configs/benchmark_grid_test_22f.yaml) → `experiments/reproduce/test_22f/grid_summary.csv`, `grid_manifest.json`, các bảng và hình | Chấm IF, LOF, OCSVM và bốn baseline trên User; train ngày 01–42, đánh giá ngày 43–60 |
| Đối chiếu | Bảng mới + `experiments/grid/test_22f/grid_summary.csv` → `experiments/reproduce/test_22f/reproduce_check.csv` | Kiểm tra sai lệch tương đối ≤ 1%; LOF ≤ 5% vì HNSW đa luồng không tất định |

Lệnh trả mã 0 khi mọi bước và phép đối chiếu đạt. Một lần chạy đầy đủ mất nhiều giờ và cần dung lượng cho 5 run tiêm. `--skip-existing` chỉ dành cho artifact đã xác minh nguồn gốc; mặc định chạy lại. Nhãn là **bất thường tổng hợp do bộ tiêm tạo**, không phải nhãn tấn công thật của LANL.

## 3. Từ bảng đến phân tích, báo cáo và demo

Kết quả chuẩn đã lưu trong `experiments/grid/test_22f/`; kết quả chạy lại nằm trong `experiments/reproduce/test_22f/`. Đọc `grid_summary.csv` để xem trung bình/độ lệch theo mô hình, `grid_scenarios_summary.csv` để xem từng kịch bản, `grid_manifest.json` để tra seed, cấu hình và hash ma trận. Diễn giải kết quả ở [báo cáo kỹ thuật cuối](../reports/final/bao_cao_ky_thuat.md) và [tóm tắt tuần 5](../reports/week5/tom_tat_tuan5.md).

Sau khi có 5 run, có thể tạo lại phân tích theo kịch bản và bộ cảnh báo cần người xem:

```bash
python scripts/evaluation/scenario_analysis.py --config configs/benchmark_grid_test_22f.yaml --out experiments/analysis/scenario_22f
python scripts/evaluation/build_review_set.py --config configs/benchmark_grid_test_22f.yaml --out experiments/analysis/review_top50_22f
```

`review_agreement.py` cần **hai người chấm** vào phiếu review trước khi tính Cohen's kappa; bước này không thể tự sinh từ log. Các phân tích chọn đặc trưng SHAP/ablation và lưới dev 22 đặc trưng là giai đoạn nghiên cứu trước khi chốt test, được lưu ở `experiments/analysis/`, `experiments/ablation/`, `experiments/grid/dev_22f/` và giải thích trong [báo cáo tuần 5](../reports/week5/bao_cao_tuan5.md). Lệnh `reproduce.py` tái lập **bảng test cuối**, không chạy lại toàn bộ lịch sử chọn đặc trưng. Nếu cần dựng lại lưới dev, xem [quy trình dev/test](week4_benchmark_grid.md): dev dùng ngày train 01–35, đánh giá 36–42 và cần hồ sơ `train_profiles_d35` cùng baseline `baselines_d35` riêng.

Để chạy demo web bằng mô hình 22 đặc trưng, sau khi có ma trận processed:

```bash
python -m demo.prepare
python -m uvicorn demo.backend.main:app --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000>, kiểm tra `/api/health`, rồi upload một file Parquet interim cho ngày sau 42; định dạng và giới hạn ở [hướng dẫn demo](../demo/README.md). Nếu `experiments/models/demo_22f_user/` đã có, dùng model hiện có vì `demo.prepare` không ghi đè.
