# Tuần 4: quy trình tiêm bất thường và kiểm tra tiến độ

Ngày kiểm tra: **08/10/2026**. Mốc code: `main` tại commit `3f48f037e4fbd8467e1a149ad408b399eaf49e47`.

Tài liệu này giải thích từ mục đích, cách tạo log, cách chạy, đến cách đọc kết quả. Các giá trị mặc định dưới đây lấy từ code và cấu hình tại mốc kiểm tra; thay cấu hình thì phải ghi lại giá trị mới.

## 1. Tuần 4 đã hoàn thành chưa?

> **Cập nhật 09/10/2026** (nhánh `feature/target-rate-1pct`, PR #10). Bảng trạng thái bên dưới là ảnh chụp ngày 08/10 và đã lỗi thời ở các điểm sau:
> - **Tiêm theo tỷ lệ định trước:** đã đáp ứng. `common.target_rate` tiêm đúng 1% số dòng (tài khoản, ngày) User; mỗi run dev đạt 736 / 72.845 = 1,01%.
> - **Bảng 3 mô hình × ≥ 3 cấu hình, 5 seed, đường PR, độ nhạy contamination:** đã có ở `experiments/grid/dev/`. Xem `reports/week4/tom_tat_tuan4.md`.
> - **PR #8 (biểu đồ):** đã hợp nhất vào `main`.
>
> Bản tóm tắt ngắn của bộ tiêm (dùng cho slide): `reports/week4/kich_ban_tiem_slide.md`.

**Chưa đủ bằng chứng để nghiệm thu toàn bộ tuần 4.** Bộ tiêm và phần tính chỉ số đã được triển khai, nhưng đầu ra thực nghiệm và một số yêu cầu còn thiếu.

Đề cương yêu cầu tuần 4 có bộ sinh bất thường, tiêm theo tỷ lệ định trước, tính chỉ số, quét tham số; bàn giao bảng **3 mô hình × ít nhất 3 cấu hình**, đường Precision–Recall, biểu đồ độ nhạy `contamination`, và kết quả **5 seed kèm độ lệch chuẩn**.

| Yêu cầu | Trạng thái tại mốc kiểm tra | Bằng chứng / phần còn cần làm |
| --- | --- | --- |
| Bộ sinh brute-force, ngoài giờ, máy trạm mới, ngủ đông | Có code trên `main` | `src/injection/` có 6 kịch bản, thêm spraying và đổi LogonType. Cách tạo ngoài giờ/ngủ đông có khác mô tả đề cương, xem mục 4. |
| Tiêm vào phần dữ liệu sau train, có nhãn và nhật ký tiêm | Có code | Có log bổ sung, manifest, nhãn, `run_config.json`; tách dev/test; giữ đủ ngày của khối đánh giá. |
| Tiêm theo tỷ lệ định trước | Chưa đáp ứng đầy đủ | Cấu hình dùng `n_victims` và `n_campaigns`, chưa có cơ chế bảo đảm tỷ lệ dương tính mục tiêu; phải đo tỷ lệ thực tế và bổ sung cách chốt ngân sách tiêm. |
| Tính các chỉ số đánh giá | Có code; chưa có đủ kết quả có nhãn để nghiệm thu | AP/PR-AUC, ROC-AUC, Precision@K, Recall theo ngân sách, alert rate, thời gian fit/score đã có. Bảng lưu trên `main` hiện không có các chỉ số cần nhãn. |
| Bảng 3 mô hình × ≥3 cấu hình | Chưa có bằng chứng hoàn thành | `benchmark_summary.csv` lưu một lần chạy seed 42: 3 mô hình ML và baseline, mỗi phương pháp tách User/Machine; đây không phải 3 cấu hình/mô hình. |
| Đường PR và biểu đồ kết quả | Có code ở PR, chưa vào `main` | [PR #8](https://github.com/lamvu0607-creator/ueba-benchmark/pull/8) đang mở tại thời điểm kiểm tra. Cần hợp nhất và chạy trên kết quả có nhãn. |
| Biểu đồ độ nhạy contamination | Chưa có trong bộ vẽ ở PR #8 | Cần chạy các mức contamination và vẽ chỉ số theo từng mức; xem mục 10. |
| 5 seed, mean ± std | Chưa có bằng chứng hoàn thành | Manifest lần chạy lưu 3 seed cho kiểm tra ổn định trên mẫu 50.000 dòng. Đây không phải 5 lần benchmark đầy đủ có nhãn. |

Đã chạy `python -m pytest -q` trên mốc `main` này: **209 passed, 2 skipped**. Hai test bỏ qua cần `data/processed/feature_matrix_processed.parquet`, hiện không có trong workspace kiểm tra. Cũng không có file log interim thật hoặc run tiêm để chạy lại toàn bộ thí nghiệm. Kết luận trên nói về **bằng chứng có thể kiểm tra trong repo/workspace**, không khẳng định các thí nghiệm trên máy khác chưa từng được chạy.

Một điểm cần thống nhất với mentor: đề cương nêu LOF `novelty=True` và One-Class SVM RBF của scikit-learn; repo hiện dùng **HNSWLOF xấp xỉ** và **Nystroem + SGDOneClassSVM xấp xỉ** để xử lý dữ liệu lớn. Isolation Forest dùng PyOD bọc scikit-learn. Phải ghi rõ các triển khai này trong báo cáo, không coi tên thuật toán giống nhau là đủ để chứng minh khớp hoàn toàn với đề cương.

Review top 50, ablation, phân tích lỗi chi tiết và chạy lại trên máy sạch thuộc **tuần 5**; không dùng các mục đó để kết luận tuần 4 thiếu đầu ra.

## 2. Tiêm bất thường để làm gì?

Dữ liệu log chưa có nhãn xác nhận ngày nào là tấn công. Vì vậy, chỉ nhìn điểm bất thường cao chưa biết mô hình phát hiện đúng hay sai. Bộ tiêm tạo những hành vi mà ta biết rõ đã thêm vào, rồi dùng chúng làm các trường hợp dương tính để so sánh mô hình.

Ví dụ: một tài khoản thường đăng nhập ban ngày. Ta lấy chuỗi đăng nhập thật của tài khoản đó trong train, tạo bản sao vào ban đêm ở ngày 45. Sau khi tính lại đặc trưng, dòng của tài khoản ở ngày 45 được đánh dấu `is_anomaly = 1`. Mô hình vẫn học từ các ngày train; nhãn chỉ dùng khi đánh giá.

**Đơn vị đánh giá là tài khoản × ngày.** Thêm 20 sự kiện vào cùng một tài khoản trong một ngày tạo một dòng nhãn dương, không tạo 20 dòng nhãn. Một tài khoản bị tiêm ở ngày 45 không làm mọi ngày của tài khoản đó thành dương tính.

Kết quả phản ánh khả năng bắt **các bất thường tổng hợp theo cách ta đã thiết kế**. Những dòng không được tiêm là âm tính theo quy ước thực nghiệm, vẫn có thể chứa bất thường thật chưa biết. Không diễn giải AP/Recall của thí nghiệm này thành độ chính xác bắt tấn công thực tế đã được xác nhận.

## 3. Dữ liệu, lịch sử và nguyên tắc tách tập

| Phần dữ liệu | Ngày mặc định | Dùng để làm gì? |
| --- | --- | --- |
| Train | 1–42 | Dựng hồ sơ hành vi, lấy khuôn sự kiện, fit bộ điền thiếu/scaler/mô hình và xác định ngưỡng. |
| Dev | 43–51 | Tiêm, kiểm tra ngữ nghĩa, quét cấu hình và chọn phương án. |
| Test cuối | 52–60 | Đánh giá phương án đã chốt; không tiếp tục chỉnh tham số theo kết quả test. |

Code gọi mọi ngày sau mốc train là phần `test` của phép chia thời gian; **dev/test của bộ tiêm** là hai khối nhỏ hơn trong phần đó. Với `--events-dir`, benchmark/baseline lấy danh sách `eval_days` đã lưu của run để chỉ đánh giá đúng khối, kể cả ngày không có lần tiêm nào.

Nguồn tiêm là `data/interim`, gồm hai file cho mỗi ngày:

```text
data/interim/event_4624/event_4624_day-01.parquet
data/interim/event_4625/event_4625_day-01.parquet
... đến ngày 60
```

`4624` là đăng nhập thành công, `4625` là đăng nhập thất bại. `Source` là máy nguồn; `LogHost` là máy ghi nhận sự kiện, không mặc định coi hai trường là cùng một máy.

Ngày `d` ở tầng interim có `Time` trong khoảng `[(d−1)×86400, d×86400)`, lấy giờ bằng `(Time % 86400) // 3600`. Tầng `cleaned` có điều chỉnh DST từ ngày 42, nên so sánh một ma trận từ cleaned với một ma trận từ interim có thể gây khác biệt dù chưa tiêm. Khi có `--events-dir`, extractor chủ động đọc **interim + log của run**. Ma trận đối chứng để kiểm tra train cũng phải được tạo từ interim.

Thông tin về hành vi quen/mới và khuôn log lấy từ train. Tuy nhiên, code có đọc **log gốc sau train** để biết tài khoản có hoạt động ở ngày tiêm không và tính khoảng im lặng đến ngày đó. Với test 52–60, lịch sử im lặng tính cả hoạt động gốc ngày 43–51; không chỉ ghép train với ngày 52. Việc chọn mẫu tổng hợp này phải được mô tả riêng với việc fit mô hình, vốn chỉ dùng train.

## 4. Sáu kịch bản đang được tiêm như thế nào?

**Số nạn nhân:** khi bật `common.target_rate` (mặc định từ 09/10), `n_victims` / `n_campaigns` bị bỏ qua. Mỗi kịch bản nhận 1/6 tổng số lần tiêm (dev: 122–123 lần/kịch bản; spraying khoảng 17 chiến dịch). Xem mục 7.

Các số `[a, b]` của tham số đếm được rút ngẫu nhiên **bao gồm hai đầu**. Cửa sổ giờ `[a, b)` không gồm giờ kết thúc. Mặc định mọi lần tiêm nằm trong một ngày.

| Kịch bản | Hành vi được thêm theo cấu hình hiện tại | Tín hiệu mong đợi sau tính lại đặc trưng |
| --- | --- | --- |
| `brute_force` | Mỗi nạn nhân thêm 8–25 lần 4625, LogonType 3, từ máy nguồn mới so với train; dồn trong 300–1.800 giây. 5 lần đầu là khuôn sai mật khẩu, các lần sau là khuôn bị khoá (luôn xảy ra vì 8 > L = 5). | Tăng thất bại, thay đổi nhịp sự kiện; có thể tăng tỷ lệ thất bại do bị khoá. |
| `password_spraying` | Mở chiến dịch tới khi đủ hạn mức; mỗi chiến dịch chọn 5–10 tài khoản, chung một máy nguồn và một ngày; mỗi nạn nhân thêm 1–3 lần 4625. Có `campaign_id` để đánh giá theo chiến dịch. | Nhiều tài khoản nhận một ít lần thử thất bại từ cùng nguồn; có thể tạo dòng tài khoản-ngày mới. |
| `off_hours` | Chọn tài khoản có tỷ lệ ngoài giờ trong train ≤0,2; lấy chuỗi 10–40 sự kiện 4624 liên tiếp của chính tài khoản trong một ngày train, đưa vào 0–6 giờ, giữ khoảng cách thời gian của chuỗi. | Tăng tỷ lệ ngoài giờ và thay đổi phân bố giờ. |
| `new_workstation_burst` | Mỗi nạn nhân thêm đăng nhập thành công từ 4–10 máy nguồn chưa quen trong train, 1–3 sự kiện/máy; dồn trong 1.800–7.200 giây, khung 8–18 giờ. | Tăng số nguồn khác nhau; có thể tăng novelty nguồn và thay đổi quan hệ nguồn–máy đích. |
| `dormant_wakeup` | Tại ngày đang trống sau khoảng im lặng đủ dài, cấy toàn bộ 4624/4625 của một ngày train gần trung vị số sự kiện nhất, giữ giờ trong ngày. | Tạo hoạt động trở lại và một dòng tài khoản-ngày mới; tác động lên đặc trưng lịch sử. |
| `logon_type_switch` | Chọn tài khoản có một LogonType chiếm ≥80% trong train và có nguồn/máy ghi log quen; thêm 3–10 lần 4624 với type chưa dùng trong train, ưu tiên 10 rồi 2, rải trong 8–18 giờ. | Thay đổi cơ chế đăng nhập, phân bố LogonType và mức độ hiếm của type. |

Các tín hiệu trên là **mong đợi cần kiểm tra**, không phải cam kết rằng mọi đặc trưng sẽ tăng hoặc mọi mô hình sẽ bắt được.

### Những chi tiết dễ hiểu nhầm

**Brute-force:** `L: auto` lấy ngưỡng khoá ước lượng từ train, không luôn bằng 5. Nếu rút 12 lần thử và `L = 5`, chế độ `locked_out` tạo 5 khuôn sai mật khẩu rồi 7 khuôn bị khoá. Lần thứ `L` vẫn là khuôn sai mật khẩu. Chế độ `stop` dừng ở ngưỡng, vì vậy phải đọc số sự kiện thực tế trong manifest thay vì suy từ khoảng cấu hình.

**Spraying:** chú thích cấu hình muốn số lần thử nhỏ hơn `L`, nhưng hàm kịch bản hiện không tự ép điều kiện đó. Cần kiểm tra `L` và chỉnh khoảng `[1, 3]` nếu cần. Các nạn nhân được sinh lịch burst riêng; code bảo đảm cùng nguồn/ngày/chiến dịch, chưa bảo đảm toàn bộ chiến dịch nằm trong một cửa sổ burst chung. Kiểm tra độ trải thời gian của cả `campaign_id` trước khi mô tả là một chiến dịch đồng thời. **Đã đo (09/10, dev_seed20261043):** một chiến dịch trải **9,5–23 giờ** trong ngày (trung vị 17 giờ), nên **không** phải đợt dồn dập đồng thời. Mỗi chiến dịch đúng 1 nguồn, 1 ngày.

**Ngoài giờ:** code **thêm** chuỗi ban đêm vào log gốc. Nó không chuyển toàn bộ hoạt động ban ngày sang ban đêm. Chuỗi không vừa cửa sổ 0–6 giờ hoặc không có đủ khuôn sẽ bị bỏ qua.

**Máy mới:** chọn tên máy có thật trong mạng train, loại các máy đã là Source hoặc LogHost quen của nạn nhân trong train. Nếu bộ lọc độ phổ biến không đủ máy, code có thể mở rộng sang các máy mới còn lại. “Chưa dùng trong train” không bảo đảm “chưa dùng trong 7 ngày trước ngày tiêm”: máy đó có thể xuất hiện trong log gốc sau ngày 42. Vì vậy `new_source_count_7d` không nhất thiết tăng như số máy được chọn.

**Ngủ đông:** số ngày im lặng là số ngày trống giữa ngày hoạt động gần nhất và ngày tiêm. Mặc định cần ≥7 ngày và **lớn hơn khoảng trống nội bộ dài nhất của chính tài khoản trong train** (`gap_rule: account_max`). Code không xoá log để tạo khoảng im lặng và không tạo một đợt khối lượng lớn bất thường: ngày được phát lại có khối lượng điển hình gần trung vị. Nếu cần đúng ý “im lặng nhiều tuần rồi hoạt động dồn dập” của đề cương, phải chốt lại định nghĩa và chỉnh thiết kế; hiện không được báo cáo là đã mô phỏng đúng biến thể đó.

**Đổi LogonType:** không chỉ sửa số `LogonType` trên một log bất kỳ. Code chọn khuôn 4624 thật có đúng type mục tiêu để các trường liên quan như mô tả loại đăng nhập, gói xác thực, tiến trình vẫn phù hợp; sau đó gắn tài khoản, máy quen và thời gian mới.

## 5. Từ log gốc đến một run tiêm

### Bước 1 — Dựng hồ sơ chỉ từ train

`scripts/injection_profiles_summary.py` đọc ngày train và lưu các bảng: tài khoản, Source quen, LogHost quen, LogonType, khoảng im lặng, số sự kiện theo ngày, máy trong mạng và các ca khoá; `network.json` ghi ngày train và thống kê mạng/ngưỡng `L`.

Mặc định nạn nhân phải thuộc nhóm **User**, có ít nhất **7 ngày có hoạt động** trong train. Đây không phải chỉ có tài khoản xuất hiện trong khoảng lịch dài 7 ngày. Tài khoản máy và các nhóm ngoài User không được chọn để tiêm theo cấu hình hiện tại.

### Bước 2 — Chạy baseline trên log gốc

Baseline luật ECDF xác định các tài khoản-ngày đã bị cảnh báo trước khi tiêm. Bộ tiêm tránh chọn những dòng đó làm nạn nhân. Việc này giúp hạn chế chọn một dòng vốn đã dễ bị cảnh báo; đồng thời cũng là một thiên lệch chọn mẫu cần ghi trong báo cáo.

Đây là **loại khỏi tập ứng viên tiêm**, không tự loại các dòng đó khỏi mẫu âm tính khi tính chỉ số. Không nhầm với `eval_exclude` ở mục 7.

Phải có `baseline_scores.parquet`, `thresholds.json` và metadata thống kê luật chứng minh `injected_events_dir: null`. Hiện code chỉ cảnh báo rồi tiếp tục nếu thiếu score/ngưỡng; nếu metadata không tồn tại, cũng chưa chứng minh baseline là của log gốc. Vì vậy, quy trình vận hành phải kiểm tra đủ file trước khi tiêm, không chỉ thấy lệnh kết thúc thành công là coi bước này hợp lệ.

### Bước 3 — Dựng kho khuôn sự kiện

Khuôn là log thật trong train. Ưu tiên log của chính nạn nhân; trường hợp cho phép thì mượn log của tài khoản cùng loại. Các thao tác nhân bản đổi tài khoản/thời gian/nguồn cần thiết, giữ schema interim và dùng khuôn đúng loại thành công/thất bại/LogonType.

Khuôn của nạn nhân được lưu trong cache trên đĩa ở `data/injection_runs/_template_cache/`, đọc theo bucket và cache truy cập gần đây. Khuôn mượn giữ tối đa 200 mẫu mỗi nhóm `(entity_type, EventID, LogonType, fail_kind)` trong RAM. Cách này giảm việc nạp toàn bộ log train vào bộ nhớ; lần dựng cache đầu vẫn cần thời gian và dung lượng đĩa.

### Bước 4 — Chọn nạn nhân và sinh sự kiện

Thứ tự chạy: brute-force → spraying → ngoài giờ → máy trạm mới → ngủ đông → đổi LogonType. Một tài khoản chỉ được tiêm tối đa **một lần trong một run**, nên kịch bản chạy trước có thể dùng mất ứng viên của kịch bản sau.

Brute-force, ngoài giờ, máy trạm mới và đổi LogonType chọn ngày đã có hoạt động gốc; spraying có thể chọn ngày trống; ngủ đông phải chọn ngày trống. Ứng viên không đủ khuôn, không đủ máy mới hoặc không xếp được lịch sẽ bị bỏ qua. Bộ tiêm có thể tạo **ít nạn nhân hơn cấu hình** và ghi cảnh báo; nếu không tạo được lần nào thì báo lỗi.

### Bước 5 — Ghi bản log có tiêm và sinh nhãn

Với mỗi ngày có tiêm, code đọc đầy đủ log gốc của cả 4624 và 4625, nối sự kiện tổng hợp, sắp theo `Time`, rồi ghi **bản riêng trong run**. Đây là bản ngày hoàn chỉnh: khi extractor chọn nó, bản này thay cho file gốc của ngày đó, không được nối với log gốc thêm một lần nữa.

Ngày không có tiêm tiếp tục đọc từ interim gốc. File gốc không bị ghi đè bởi bộ tiêm. Bản log đưa vào feature engineering giữ 21 cột interim, **không có `inj_id`, nhãn hoặc tên kịch bản**. Những trường đánh dấu nằm ở bảng phụ để mô hình không học trực tiếp dấu hiệu “đây là dữ liệu tiêm”.

Code kiểm tra ngày tiêm sau split, timestamp đúng ngày, số dòng thêm bằng bảng phụ, tên/thứ tự cột và không tiêm trùng tài khoản; sau đó sinh nhãn và lưu phạm vi đánh giá. Các kiểm tra này chưa thay thế việc kiểm tra ngữ nghĩa từng kịch bản và tính lại đặc trưng.

## 6. Các file của một run và vai trò của chúng

Ví dụ dùng `run_id = dev_w4_01`:

| File / thư mục trong `data/injection_runs/dev_w4_01/` | Nội dung / mục đích |
| --- | --- |
| `events_injected/event_4624/`, `events_injected/event_4625/` | Bản log đầy đủ của các ngày có tiêm; chỉ chứa schema sự kiện. |
| `injected_events.parquet` | Chỉ các sự kiện tổng hợp, có `inj_id` để truy vết từng lần tiêm. |
| `injection_manifest.csv` | Một dòng/lần tiêm: `inj_id`, kịch bản, domain, user, ngày, campaign, block, seed, số sự kiện và `params` JSON thực tế. |
| `labels.parquet` | Nhãn tài khoản-ngày từ manifest; dùng khi đánh giá. |
| `run_config.json` | `run_id`, block, seed tiêm, split day và toàn bộ `eval_days`. Không lưu toàn bộ YAML tham số. |
| `features/raw/`, `processed/` | Ma trận tính lại từ interim + bản log của run. |

Kết quả benchmark **không** nằm trong `data/injection_runs/<run_id>/` mà ở `experiments/injection_runs/<run_id>/`:

| Thành phần | Ý nghĩa |
|---|---|
| `results/`, `models/` | Kết quả (được commit) và mô hình .joblib (không commit) khi chạy benchmark với `--events-dir` của run. |
| `results/baselines/` | Kết quả baseline trên chính dữ liệu của run. |
| `labels.parquet`, `run_config.json` | Bản sao từ run để thư mục kết quả tự đủ khi vẽ hình. |

Không tái dùng `run_id` để thử cấu hình khác: code chưa có cơ chế đóng băng run hoặc từ chối ghi đè đầy đủ, có nguy cơ giữ lại file ngày cũ. Mỗi phiên bản tiêm dùng tên mới; khi lỗi giữa chừng, không coi thư mục đã xuất hiện là run hoàn chỉnh.

## 7. Nhãn và tỷ lệ tiêm: cách hiểu đúng

Khóa nhãn là `(DomainName, UserName, day)`. Domain được strip và chuyển về chữ thường, rỗng/null thành `Unknown`, giống khóa feature. File nhãn hiện sinh các dòng dương tính, với `is_anomaly = 1`, `eval_exclude = false`, kèm `scenario` và `campaign_id`. Dòng đánh giá không có trong file nhãn được gán 0. Nếu cung cấp nhãn có `eval_exclude = true`, dòng đó bị loại trước khi tính chỉ số.

Vì bộ tiêm mặc định chỉ nhắm User, nên báo cáo tỷ lệ và chỉ số bắt tiêm chính trên **User**. Machine không có nhãn dương được tiêm; AP/ROC/Recall ở phân khúc đó có thể không xác định. Không biến giá trị thiếu thành 0 rồi lấy trung bình cùng User.

```text
Tỷ lệ tiêm thực tế (%) = 100 × số dòng User dương tính khớp ma trận
                              / số dòng User trong khối đánh giá sau eval_exclude
```

Đo trên ma trận **sau tiêm**, vì spraying/ngủ đông có thể tạo khóa tài khoản-ngày mới và thay đổi mẫu số. Nếu báo cáo tỷ lệ trên toàn bộ User + Machine, phải ghi mẫu số đó riêng.

Không nhầm ba số:

| Số | Ý nghĩa |
| --- | --- |
| Tỷ lệ tiêm | Phần trăm dòng tài khoản-ngày được gán dương tính. |
| `contamination` | Tỷ lệ dùng để đặt ngưỡng cảnh báo từ phân bố điểm train của mô hình. Không phải tỷ lệ sự kiện được thêm. |
| Ngân sách cảnh báo | Số hoặc tỷ lệ cảnh báo được phép xét để tính Recall; có thể cố định K/ngày. |

Ví dụ minh họa: 20.000 dòng User sau tiêm, 100 dòng dương tính khớp ma trận → tỷ lệ tiêm 0,5%. Thêm 2.000 sự kiện cho 100 dòng đó vẫn chỉ có 100 dòng dương.

**Cập nhật 09/10:** đã có bộ điều phối `common.target_rate`. Tỷ lệ mục tiêu là 1%, mẫu số D = số dòng (tài khoản, ngày) User của khối trên log **gốc**. Tổng N = ⌈1% · D / (1 − 1%)⌉; chia cho (1 − 1%) vì spraying/ngủ đông có thể tạo dòng mới. N chia đều cho 6 kịch bản; kịch bản thiếu ứng viên thì phần thiếu được chia lại (tối đa 3 vòng). `run_config.json` ghi mẫu số, hạn mức, số đạt được và tỷ lệ đạt được. Dev: D = 72.845, N = 736, đạt 1,01%; trên ma trận sau tiêm là 736 / 72.998 = 1,008%. Mức 1% được chốt theo mục 5.2 đề cương (0,5–1%), không chọn theo AP. Bản cũ (số nạn nhân cố định) chỉ đạt 92 dòng = 0,13%.

## 8. Lệnh chạy từ đầu đến cuối

Chạy ở thư mục gốc repo, trong môi trường đã cài dependency theo README. Các lệnh một dòng dưới đây dùng được cả Bash và PowerShell. Cần có đủ file interim 1–60 cho cả hai EventID; kiểm tra tính đầy đủ trước, vì một số bước dựng train/feature có thể bỏ qua ngày thiếu thay vì luôn báo lỗi.

### 8.1. Tạo cấu hình đối chứng đọc interim

Sao chép `configs/system_config.yaml` thành `configs/week4_system.yaml`, giữ các thiết lập phân loại tài khoản/đặc trưng, và đổi **các mục tương ứng** sau (không thay cả file bằng đoạn rút gọn này):

```yaml
paths:
  interim_data_dir: data/interim
  cleaned_data_dir: data/_no_cleaned_week4
  features_data_dir: data/features/week4_control
  processed_data_dir: data/processed/week4_control
evaluation:
  split_day: 42
  labels_path: null
```

`data/_no_cleaned_week4` phải không chứa file `cleaned_day-*.parquet`; extractor khi đó sẽ dùng interim. Các đường dẫn khác của bản sao giữ nguyên. Đây là cấu hình vận hành cần lưu cùng thí nghiệm, chưa phải một file có sẵn trong repo.

### 8.2. Tạo ma trận gốc, hồ sơ và baseline gốc

```bash
python main.py --stage features --config configs/week4_system.yaml --start-day 1 --end-day 60
python scripts/injection_profiles_summary.py --config configs/week4_system.yaml --out data/features/train_profiles
python main.py --stage baselines --config configs/week4_system.yaml
```

Các lệnh này tạo ma trận đối chứng ở `data/processed/week4_control/`, hồ sơ train và baseline gốc ở đường dẫn mặc định của `configs/baselines.yaml`. Chúng có thể thay artifact cũ ở cùng đường dẫn: lưu bản thí nghiệm cũ trước khi dựng lại. Cache baseline hiện chưa xác minh hash toàn bộ đầu vào; nếu dữ liệu gốc đã thay đổi, không dùng cache thống kê cũ.

Trước khi đi tiếp, kiểm tra:

1. `network.json` của hồ sơ chỉ liệt kê ngày 1–42, đủ dữ liệu theo kế hoạch.
2. Có `experiments/results/baselines/baseline_scores.parquet` và `thresholds.json`.
3. `data/features/baselines/rule_event_stats.json` có `train_last_day = 42`, `injected_events_dir = null`; baseline vừa chạy thực sự từ log gốc và ma trận interim.
4. Khối User và phương pháp `rule_ecdf` có trong score/ngưỡng; không nhầm baseline của run tiêm với baseline gốc.
5. `configs/injection.yaml` trỏ đúng các file đó; đã chốt seed, nạn nhân, cường độ và thời gian. Nếu đổi đường dẫn baseline/cache, sửa cả `common.rule_exclusion` tương ứng.

### 8.3. Tiêm vào dev

```bash
python main.py --stage inject --config configs/week4_system.yaml --injection-config configs/injection.yaml --block dev --run-id dev_w4_01
```

Seed **tiêm** lấy từ `blocks.dev.seed`, mặc định `20261043`. `--seed` là seed **benchmark**, không ghi đè seed tiêm. Không chỉ định `--run-id` thì tên mặc định là `dev_seed20261043` hoặc `test_seed20261052` theo cấu hình hiện tại.

Đọc thông báo số nạn nhân từng kịch bản, ứng viên bị bỏ và manifest thực tế. Phải có đủ `injected_events.parquet`, `injection_manifest.csv`, `labels.parquet`, `run_config.json` trước khi tiếp tục.

### 8.4. Tính lại đặc trưng và đánh giá dev

```bash
python main.py --stage features --config configs/week4_system.yaml --start-day 1 --end-day 60 --events-dir data/injection_runs/dev_w4_01/events_injected
python main.py --stage baselines --config configs/week4_system.yaml --events-dir data/injection_runs/dev_w4_01/events_injected
python main.py --stage benchmark --config configs/week4_system.yaml --model-params configs/model_params.yaml --seed 42 --events-dir data/injection_runs/dev_w4_01/events_injected
```

Tính đặc trưng cả dải 1–60 để giữ train và lịch sử liên tục; chỉ số dev vẫn chỉ tính 43–51 theo metadata run. Nhãn được lấy tự động từ `labels.parquet` cùng run. Các đầu ra của những lệnh có `--events-dir` được ghi trong run, không thay ma trận đối chứng.

Baseline riêng ở stage `baselines` mặc định đặt ngưỡng theo ngân sách **1%**; benchmark ML dùng mặc định **5%**. Không kết luận phương pháp tốt hơn từ Recall/alert rate tại hai ngân sách khác nhau. So sánh AP, Precision@K, Recall@N/ngày ở cùng tập/cùng K/N; nếu so sánh cờ cảnh báo tại ngưỡng, đồng bộ ngân sách và ghi cấu hình. Baseline luật trong registry benchmark cũng không phải đúng cùng triển khai `rule_ecdf` của stage baseline riêng.

### 8.5. Chốt dev rồi chạy test cuối

Chốt thiết kế tiêm, cấu hình mô hình, tỷ lệ/độ mạnh, feature schema, seed và tiêu chí tổng hợp trước khi xem test. Hai block đang dùng chung cấu hình kịch bản qua YAML anchor; sửa kịch bản chung sẽ ảnh hưởng cả dev/test.

```bash
python main.py --stage inject --config configs/week4_system.yaml --injection-config configs/injection.yaml --block test --run-id test_w4_final
python main.py --stage features --config configs/week4_system.yaml --start-day 1 --end-day 60 --events-dir data/injection_runs/test_w4_final/events_injected
python main.py --stage baselines --config configs/week4_system.yaml --events-dir data/injection_runs/test_w4_final/events_injected
python main.py --stage benchmark --config configs/week4_system.yaml --model-params configs/model_params.yaml --seed 42 --events-dir data/injection_runs/test_w4_final/events_injected
```

Đây là một lần chạy chính minh họa; kế hoạch 5 seed cần dùng cách lưu riêng ở mục 10. “Chạy test một lần” nghĩa là một đợt đánh giá với kế hoạch đã đóng băng, có thể gồm 5 seed định trước, không phải mở test để chọn seed tốt nhất. Test là run độc lập từ log gốc, không nối thêm các sự kiện đã tiêm của dev.

### 8.6. Vẽ biểu đồ

Tại mốc kiểm tra, `main` **chưa có** stage `plots`. Sau khi PR #8 được hợp nhất, dùng hướng dẫn `docs/week4_result_plots.md` của PR đó. Lệnh theo giao diện hiện tại của PR:

```bash
python main.py --stage plots --events-dir data/injection_runs/dev_w4_01/events_injected
```

Chạy lại benchmark với code của PR trước khi vẽ để score xuất ra có đầy đủ độ chính xác; score trên `main` hiện được làm tròn 6 chữ số. Không áp lệnh này vào `main` hiện tại rồi coi lỗi “stage không hợp lệ” là lỗi bộ tiêm. PR #8 có đường PR và các biểu đồ so sánh kết quả/thời gian, nhưng chưa có biểu đồ quét contamination.

## 9. Các kiểm tra phải đạt trước khi tin bảng kết quả

| Kiểm tra | Điều kiện cần đạt |
| --- | --- |
| Phạm vi train | Không có log bổ sung ngày ≤42; ma trận train của đối chứng interim và run tiêm giống nhau sau sắp khóa. |
| Ngày/giờ | Mọi `Time` tổng hợp thuộc đúng cửa sổ ngày; sự kiện ngoài giờ, burst, replay phù hợp cấu hình. |
| Số lượng | Tổng `n_events` trong manifest bằng số dòng bảng phụ; số dòng tăng của bản ngày so với gốc đúng với số sự kiện thêm. |
| Schema | Cả hai file 4624/4625 của ngày có tiêm tồn tại; đúng cột và kiểu dữ liệu interim; không có cột đánh dấu đưa vào đặc trưng. |
| Nhãn | Không trùng khóa; mọi nhãn dương trong block khớp một dòng feature; không có nhãn dương trong train. Không bỏ qua cảnh báo unmatched. |
| Kịch bản | Có đủ số lần thành công từng loại; kiểm tra cả trường log và thay đổi đặc trưng, không chỉ số sự kiện. |
| Phân khúc/phạm vi | Chỉ số chính cho User đúng dev hoặc test, không trộn 43–60 hoặc xếp hạng raw score chung User/Machine. |
| Tỷ lệ | Ghi tỷ lệ mục tiêu, thực tế và mẫu số sau tiêm; số sự kiện và số dòng dương ghi riêng. |
| Tính tái lập | Lưu cấu hình thực tế, hash dữ liệu, commit, seed tiêm/model, dependency, cảnh báo và đầu ra. |

Hai đoạn Python sau chạy sau khi có ma trận đối chứng và ma trận run. Có thể lưu tạm thành file `.py` rồi chạy từ gốc repo; tên run trong ví dụ là `dev_w4_01`.

**Đối chiếu train:**

```python
from pathlib import Path
import polars as pl

run = Path("data/injection_runs/dev_w4_01")
base = pl.read_parquet("data/processed/week4_control/feature_matrix_processed.parquet")
injected = pl.read_parquet(run / "processed/feature_matrix_processed.parquet")
keys = ["DomainName", "UserName", "day"]
train_base = base.filter(pl.col("day") <= 42).sort(keys)
train_run = injected.filter(pl.col("day") <= 42).select(train_base.columns).sort(keys)
assert train_base.equals(train_run), "Train khác đối chứng: kiểm tra nguồn log, DST, cấu hình."
print("Train giữ nguyên:", train_run.height, "dòng")
```

**Đối chiếu nhãn và đo tỷ lệ User thực tế:**

```python
import json
from pathlib import Path
import polars as pl
from src.evaluation.labeled_eval import align_eval_labels, load_eval_labels

run = Path("data/injection_runs/dev_w4_01")
meta = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
matrix = pl.read_parquet(run / "processed/feature_matrix_processed.parquet")
labels = load_eval_labels(run / "labels.parquet")
keys = ["DomainName", "UserName", "day"]
assert labels.filter((pl.col("is_anomaly") == 1) & (pl.col("day") <= meta["split_day"])).is_empty()
positive = labels.filter(pl.col("day").is_in(meta["eval_days"]) & (pl.col("is_anomaly") == 1) & ~pl.col("eval_exclude"))
assert labels.filter(~pl.col("day").is_in(meta["eval_days"])).is_empty(), "Nhãn ngoài block."
user = matrix.filter(pl.col("day").is_in(meta["eval_days"]) & (pl.col("entity_type") == "User"))
assert positive.join(user.select(keys), on=keys, how="anti").is_empty(), "Nhãn dương không khớp User."
aligned = align_eval_labels(user, labels).filter(~pl.col("eval_exclude"))
assert aligned.height > 0, "Không có dòng User để đánh giá."
n_positive = int(aligned["is_anomaly"].sum())
assert n_positive == positive.height and n_positive > 0
manifest = pl.read_csv(run / "injection_manifest.csv", schema_overrides={"campaign_id": pl.String})
n_events = pl.scan_parquet(run / "injected_events.parquet").select(pl.len()).collect().item()
assert int(manifest["n_events"].sum()) == n_events
print(manifest.group_by("scenario").agg(pl.len().alias("n_injections"), pl.col("n_events").sum()).sort("scenario"))
print(f"User: {n_positive}/{aligned.height} dương tính = {100*n_positive/aligned.height:.4f}%")
print("Sự kiện được thêm:", n_events)
```

Đoạn trên kiểm tra nhãn/số lượng cơ bản; chưa chứng minh ngữ nghĩa của cả 6 kịch bản. Cần xem các trường hợp mẫu trước/sau, nhất là chuỗi ngoài giờ, lịch chung spraying, nguồn mới trong lịch sử 7 ngày và khoảng im lặng của dormant.

## 10. Hoàn thành phần benchmark, 5 seed và biểu đồ như thế nào?

### Cố định một bộ tiêm khi so sánh mô hình

Seed tiêm quyết định nạn nhân, khuôn và lịch; seed model quyết định ngẫu nhiên lúc fit. Khi so sánh các model/config, dùng cùng một ma trận tiêm, cùng nhãn, cùng tập ngày. Giữ seed tiêm cố định và chạy model với 5 seed định trước, chẳng hạn `42, 7, 2024, 1, 2`. Nếu nghiên cứu thêm nhiều seed tiêm, báo cáo đó là một yếu tố riêng; không trộn hai nguồn biến động vào cùng độ lệch chuẩn mà không giải thích.

Không chạy liên tiếp `main.py --stage benchmark --events-dir ... --seed ...` rồi kỳ vọng giữ đủ 5 bộ kết quả: các file kết quả chính cùng run sẽ bị ghi đè. Lưu từng cấu hình/seed vào thư mục riêng. Có thể gọi API đang có trong repo như ví dụ dưới; tạo các YAML cấu hình trước rồi mới chạy:

```python
from pathlib import Path
from src.injection.layout import RunLayout, load_run_eval_days
from src.models.benchmark import run_model_benchmark

layout = RunLayout.for_run("data/injection_runs", "dev_w4_01")
eval_days = load_run_eval_days(layout)
# Ba file này là bản sao model_params.yaml đã chỉnh và lưu trước thí nghiệm.
configs = {
    "c1": Path("configs/week4_models_c1.yaml"),
    "c2": Path("configs/week4_models_c2.yaml"),
    "c3": Path("configs/week4_models_c3.yaml"),
}
for config_id, params_path in configs.items():
    assert params_path.is_file(), f"Chưa tạo cấu hình: {params_path}"
    for seed in [42, 7, 2024, 1, 2]:
        out = layout.root / "repetitions" / config_id / f"seed_{seed}"
        assert not out.exists(), f"Thư mục kết quả đã tồn tại: {out}"
        run_model_benchmark(
            data_path=layout.processed_dir / "feature_matrix_processed.parquet",
            model_names=["isolation_forest", "local_outlier_factor", "one_class_svm"],
            params_path=params_path,
            output_results_dir=out / "results",
            output_models_dir=out / "models",
            split_day=42, seed=seed, contamination=None,
            stability_seeds=[seed],
            labels_path=layout.labels_path, eval_days=eval_days,
            precision_ks=[10, 50, 100], daily_budgets=[10, 50, 100],
            experiment_log_path=layout.root / "repetitions/experiment_log.csv",
            system_config_path="configs/week4_system.yaml",
        )
```

Ví dụ gọi API này là hướng dẫn vận hành, **không phải bằng chứng các lần chạy đã được thực hiện**. Nó chạy 3 mô hình × 3 cấu hình × 5 seed = 45 trường hợp mô hình; mặc định tách hai phân khúc nên có hai bộ fit cho mỗi trường hợp. Chạy baseline trên cùng ma trận/phạm vi/ngân sách để làm mốc tham chiếu. Test cuối chỉ chạy kế hoạch các cấu hình/seed đã chốt, không dùng kết quả test để chọn cấu hình.

Ba YAML có thể quét `n_estimators` của IF, `n_neighbors` của LOF, `nu`/`gamma` của OCSVM ở mức vừa phải. Chốt rõ từng tổ hợp; “3 cấu hình” nghĩa là ba bộ tham số khác nhau của từng mô hình, không phải ba phân khúc hay ba seed. Nếu gộp nhiều tham số cùng thay đổi, kết quả chỉ so sánh tổ hợp, chưa phân lập ảnh hưởng của từng tham số.

**Lưu ý ghi đè tham số:** `main.py` truyền `evaluation.default_contamination` tường minh vào benchmark, có thể ghi đè contamination trong YAML model. Nếu dùng CLI để quét, sửa cả cấu hình hệ thống tương ứng và kiểm tra manifest thực tế. Ví dụ API ở trên truyền `contamination=None` để lấy giá trị trong YAML model. Riêng OCSVM hiện đồng bộ contamination với `nu`, và `nu` trong model có thể quyết định mức thực tế. Không ghi một mức contamination vào tên file mà bỏ `nu` ở mức khác.

### Tổng hợp và vẽ đúng mục đích

Nhóm theo **mô hình + cấu hình + phân khúc + block + cùng bộ dữ liệu tiêm**. Với từng metric, báo cáo trung bình và độ lệch chuẩn mẫu trên đủ 5 lần chạy (`ddof=1`), kèm số seed hợp lệ. Giữ giá trị thiếu nếu không có nhãn dương/âm cần thiết; không âm thầm lấy trung bình từ ít seed hơn hoặc thay giá trị thiếu bằng 0. `model_stability.csv` đo tương quan/độ chồng lấp thứ hạng trên mẫu, không thay được bảng mean ± std các metric trên toàn khối đánh giá.

| Biểu đồ cần có | Câu hỏi nó trả lời |
| --- | --- |
| Precision–Recall | Khi lấy thêm cảnh báo, mô hình bắt được bao nhiêu trường hợp tiêm và giữ độ chính xác đến đâu? Đường random có mức tham chiếu bằng tỷ lệ dương tính. |
| AP/PR-AUC mean ± std | Mô hình/cấu hình nào xếp hạng tốt hơn và kết quả có ổn định giữa seed không? `pr_auc` trong repo là Average Precision; ghi rõ định nghĩa, không gọi là diện tích hình thang PR. |
| Precision@10/50/100 | Nếu chỉ kiểm tra K cảnh báo cao nhất trong toàn block của một phân khúc, bao nhiêu phần là dương tính? |
| Recall@N/ngày | Với tối đa N cảnh báo mỗi ngày trong một phân khúc, tìm được bao nhiêu dòng dương? Không nhầm với top-K toàn block. |
| Độ nhạy contamination | Tại cùng dữ liệu/seed và các tham số còn lại cố định, mức đặt ngưỡng ảnh hưởng thế nào đến alert rate, Precision/Recall ở ngưỡng? |
| Thời gian fit/score | Chất lượng đạt được đi kèm chi phí tính toán nào? |

Cho phân tích contamination, có thể chốt trước các mức như `0.01, 0.03, 0.05, 0.10`, giữ các tham số khác cố định cho từng đường so sánh và lặp cùng 5 seed. Vẽ trục X là mức thực tế, trục Y là alert rate và Precision/Recall tại ngưỡng, kèm mean ± std. Với IF/LOF hiện tại, thay contamination chủ yếu đổi ngưỡng, nên AP/PR và các metric top-K có thể gần như không đổi; đó không phải lỗi vẽ. OCSVM thay `nu` có thể đổi cả mô hình lẫn ngưỡng, nên phải diễn giải riêng. PR #8 chưa thực hiện biểu đồ này.

## 11. Lưu gì để có thể làm lại và nghiệm thu?

Giữ mỗi run có một bộ cấu hình/đầu vào xác định. Lưu bản YAML hệ thống, injection, model, baseline và feature schema; phiên bản dependency; commit code; seed tiêm/model; hash hoặc định danh phiên bản log/ma trận; manifest, nhãn, metadata, log cảnh báo, kết quả từng seed, bảng tổng hợp và hình vẽ.

`run_config.json` chỉ lưu phạm vi/seed, không thay thế bản sao đầy đủ cấu hình. Cache khuôn được nhận diện bằng đường dẫn/kích thước/mtime và thông tin chọn mẫu, chưa phải checksum nội dung toàn bộ log. Các bản cấu hình và định danh dữ liệu vì vậy cần được lưu thêm. LOF HNSW đa luồng có thể thay thứ tự chèn dù cùng seed; khi cần kiểm tra tái lập chặt, đặt `n_jobs: 1` và ghi rõ điều kiện chạy.

Thư mục `data/injection_runs/` bị ignore trong Git. Nếu chỉ commit code thì người review chưa có bằng chứng thực nghiệm. Giữ dữ liệu/log nặng ở nơi lưu trữ của dự án; đưa bản cấu hình, manifest hoặc bản tóm tắt, bảng có nhãn từng cấu hình/seed, bảng mean ± std, tỷ lệ thực tế, hình PR và độ nhạy vào hồ sơ bàn giao có thể truy cập. Không ghi đè các artifact tuần 3 mà không lưu phiên bản.

### Các việc còn lại để chốt tuần 4

1. Thống nhất với mentor các triển khai xấp xỉ và định nghĩa ngoài giờ/ngủ đông; ghi rõ biến thể thực tế.
2. Bổ sung cơ chế/kế hoạch tiêm theo tỷ lệ định trước, đo và chấp nhận tỷ lệ thực tế theo dung sai đã chốt.
3. Chạy dữ liệu thật qua các bước trên, kiểm tra nhãn, train không đổi, kịch bản và số lần tiêm đạt yêu cầu.
4. Hoàn thành 3 mô hình × ≥3 cấu hình × 5 seed trên dev; lưu riêng từng lần và tổng hợp mean ± std, baseline cùng điều kiện.
5. Hợp nhất phần vẽ PR/kết quả ở PR #8, bổ sung quét và biểu đồ độ nhạy contamination.
6. Đóng băng phương án, chạy đợt test cuối theo kế hoạch, lưu đầy đủ bằng chứng và giới hạn của nhãn tổng hợp.

## 12. Nguồn đối chiếu trong dự án

- Đề cương được cung cấp: `01-Tong_quan_de_tai_UEBA_Benchmark.pdf`, mục 3 (tuần 4), 4 (tiêu chí), 5 (phương pháp); bản đề cương trong repo: [Tổng quan đề tài](<Tong quan de tai ueba.md>).
- Cấu hình: [injection.yaml](../configs/injection.yaml), [system_config.yaml](../configs/system_config.yaml), [model_params.yaml](../configs/model_params.yaml), [baselines.yaml](../configs/baselines.yaml).
- Mã nguồn: [điều phối tiêm](../src/injection/inject.py), [kịch bản](../src/injection/scenarios.py), [thao tác/kho khuôn](../src/injection/operations.py), [layout](../src/injection/layout.py), [nhãn](../src/injection/labels.py), [extractor](../src/features/extractor.py), [benchmark](../src/models/benchmark.py), [đánh giá có nhãn](../src/evaluation/labeled_eval.py).
- Bằng chứng đã lưu: [benchmark_summary.csv](../experiments/results/benchmark_summary.csv), [run_manifest.json](../experiments/results/run_manifest.json), [model_stability.csv](../experiments/results/model_stability.csv).
- Tài liệu kỹ thuật bổ sung: [Quy trình tiêm log](quy_trinh_tiem_log.md). Nếu có khác biệt với hướng dẫn cũ, dùng hành vi code tại commit và cấu hình đã ghi của thí nghiệm để xác định kết quả.
