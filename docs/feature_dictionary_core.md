# Từ điển đặc trưng tài khoản × ngày (schema v4.1)

Nguồn chuẩn về **danh sách và định nghĩa** là [`configs/feature_schema.yaml`](../configs/feature_schema.yaml). Pipeline tạo 41 đặc trưng `core: true` trong `feature_matrix_processed.parquet`; lưới benchmark cuối dùng 30 đặc trưng sau khi loại 11 tên trong [`configs/benchmark_grid_test_30f.yaml`](../configs/benchmark_grid_test_30f.yaml). Cột **30F** cho biết đặc trưng có mặt trong lưới cuối, không phải một schema Parquet khác.

**Đơn vị:** một dòng là `(DomainName, UserName, day)`. `t` là ngày của dòng; `H₇` là 7 **ngày lịch** trước `t` (không phải 7 dòng có hoạt động). `N` là số sự kiện 4624 + 4625 trong ngày; `fail` là sự kiện 4625. `log1p(x) = ln(1+x)`. Với tập A và B, `Jaccard(A,B) = |A∩B|/|A∪B|`; `TV(P,Q) = ½Σ|Pᵢ−Qᵢ|`. `z_bền_vững` dùng median và IQR/1,349, fallback sang độ lệch chuẩn nếu IQR gần 0; kết quả được chặn ở ±10. Với các biến `delta_mean` dạng **đếm**, ngày không hoạt động trong `H₇` góp giá trị 0; dạng **tỷ lệ** chỉ lấy trung bình các ngày có hoạt động.

**Kiểu dữ liệu** là kiểu cột Polars/Parquet của ma trận *processed*, trước khi mô hình chuyển sang mảng số và fit imputer/scaler. `Float64` có thể chứa NULL ở các biến lịch sử khi chưa đủ dữ liệu; riêng `failure_locked_out_share`, `interarrival_dt_mean` và `delta_t_cv` được điền 0 ở bước xử lý, với `is_single_event` giúp diễn giải hai biến nhịp. Các biến `novelty_*` do template engine sinh ra là **Float64 của `log1p(số lượng)`**, không phải số đếm nguyên.

**Lưu ý Source:** `distinct_sources_count` và `remote_logon_ratio` hiện bỏ `NULL`/`Unknown` nhưng chưa bỏ chuỗi rỗng; `missing_source_ratio` vẫn xem chuỗi rỗng là thiếu. Đây là quy tắc của mã đang chạy, cần nhớ khi đọc kết quả trên dữ liệu có Source rỗng.

| # | Đặc trưng | Kiểu | 30F | Công thức / quy tắc tính | Ý nghĩa hành vi |
|---:|---|---|:---:|---|---|
| 1 | `log_total_logons` | Float64 | Không | `log1p(N)` | Cường độ xác thực trong ngày. |
| 2 | `failure_ratio` | Float64 | Có | `số fail / N` | Tỷ trọng đăng nhập thất bại. |
| 3 | `failure_locked_out_share` | Float64 | Có | `số fail có FailureReason chứa "account locked out" / số fail`; bằng 0 nếu không có fail sau xử lý | Dấu hiệu tài khoản bị khóa khi đăng nhập thất bại. |
| 4 | `off_hours_ratio` | Float64 | Có | `số sự kiện trong 18:00–07:59 / N` | Mức hoạt động ngoài giờ cấu hình. |
| 5 | `interarrival_dt_mean` | Float64 | Không | Trung bình khoảng cách giây giữa các sự kiện liên tiếp trong ngày | Nhịp xác thực; chỉ có một sự kiện thì không xác định trước xử lý. |
| 6 | `delta_t_cv` | Float64 | Không | `std(khoảng cách) / mean(khoảng cách)` khi mẫu số > 0 | Độ đều của nhịp xác thực. |
| 7 | `same_second_share` | Float64 | Không | `số sự kiện có khoảng cách đến sự kiện trước = 0 giây / N` | Hoạt động dồn cùng giây. |
| 8 | `is_single_event` | UInt8 | Không | `1` nếu `N = 1`, ngược lại `0` | Cờ giải thích thiếu thông tin về nhịp. |
| 9 | `activity_peak_hour_sin` | Float64 | Không | `sin(2π × giờ đông sự kiện nhất / 24)`; hòa thì chọn giờ nhỏ hơn | Pha chu kỳ của giờ hoạt động cao điểm. |
| 10 | `activity_peak_hour_cos` | Float64 | Có | `cos(2π × giờ đông sự kiện nhất / 24)` | Thành phần còn lại của pha giờ cao điểm. |
| 11 | `hour_entropy` | Float64 | Có | `−Σp_h ln(p_h) / ln(S)` với `S` giờ có sự kiện; bằng 0 nếu `S ≤ 1` | Hoạt động trải đều hay tập trung theo giờ. |
| 12 | `interactive_ratio` | Float64 | Có | `số sự kiện LogonType = 2 / N` | Tỷ trọng đăng nhập tương tác. |
| 13 | `rare_logon_type_count_log` | Float64 | Có | `log1p(số sự kiện LogonType ∉ {2,3})` | Tần suất dùng loại logon ít gặp trong bộ dữ liệu. |
| 14 | `ntlm_ratio` | Float64 | Có | `số sự kiện AuthenticationPackage chứa "ntlm" / N` | Tỷ trọng xác thực NTLM; cũng có thể phản ánh cơ cấu lỗi. |
| 15 | `log_distinct_hosts` | Float64 | Không | `log1p(số LogHost phân biệt)` | Độ rộng máy đích mà tài khoản chạm tới. |
| 16 | `distinct_sources_count` | UInt32 | Có | Số Source phân biệt, bỏ `NULL` và `Unknown` | Độ đa dạng máy nguồn. |
| 17 | `dst_host_entropy` | Float64 | Không | `−Σp_host ln(p_host) / ln(S)` với `S` LogHost; bằng 0 nếu `S ≤ 1` | Hoạt động phân tán hay tập trung ở máy đích. |
| 18 | `missing_source_ratio` | Float64 | Không | `số sự kiện thiếu Source / N` | Chất lượng dữ liệu nguồn; không tự diễn giải là tấn công. |
| 19 | `remote_logon_ratio` | Float64 | Có | `số sự kiện có Source khác NULL/Unknown và Source ≠ LogHost / N` | Tỷ trọng xác thực từ máy nguồn khác máy ghi log. |
| 20 | `new_source_count_7d` | Int32 | Có | Số Source của `t` không có trong `H₇` | Máy nguồn mới so với lịch sử gần. |
| 21 | `new_host_count_7d` | Int32 | Có | Số LogHost của `t` không có trong `H₇` | Máy đích mới so với lịch sử gần. |
| 22 | `days_since_last_activity` | Int32 | Có | `t − ngày hoạt động gần nhất trước t`; NULL ở lần đầu | Khoảng im lặng trước khi tài khoản hoạt động lại. |
| 23 | `volume_robust_z_7d` | Float64 | Không | `z_bền_vững(log1p(N_t), log1p(N) trên H₇)`, ngày trống = 0 | Mức tăng/giảm khối lượng so với chính tài khoản. |
| 24 | `custom_proc_share` | Float64 | Có | `số sự kiện ProcessName bắt đầu "proc" / N` | Tỷ trọng tiến trình được ẩn danh dạng `Proc…`. |
| 25 | `delta_mean_share_fail_7d` | Float64 | Có | `failure_ratio(t) − trung bình failure_ratio trên ngày hoạt động của H₇` | Thay đổi tỷ lệ thất bại so với quá khứ. |
| 26 | `delta_mean_distinct_fail_host_7d` | Float64 | Có | `log1p(số LogHost phân biệt của fail hôm nay) − trung bình giá trị đó trên H₇` | Mở rộng máy đích của các lần thất bại. |
| 27 | `novelty_fail_host_7d` | Float64 | Có | `log1p(số LogHost của fail hôm nay không có trong fail của H₇)` | Máy đích mới trong các lần đăng nhập thất bại. |
| 28 | `novelty_fail_failreason_7d` | Float64 | Có | `log1p(số FailureReason hôm nay không có trong fail của H₇)` | Lý do thất bại mới. |
| 29 | `delta_mean_distinct_source_7d` | Float64 | Có | `log1p(số Source phân biệt hôm nay) − trung bình giá trị đó trên H₇` | Mức tăng độ đa dạng máy nguồn. |
| 30 | `jaccard_source_7d` | Float64 | Có | `Jaccard(tập Source hôm nay, tập Source của H₇)` | Độ giống nhau của tập máy nguồn; thấp là đổi nguồn. |
| 31 | `peer_z_distinct_source` | Float64 | Có | `z_bền_vững(log1p(số Source), các tài khoản cùng entity_type trong ngày t)` | Độ khác biệt về máy nguồn so với nhóm cùng loại. |
| 32 | `dist_shift_hour_7d` | Float64 | Có | `TV(phân bố giờ hôm nay, phân bố giờ cộng dồn của H₇)` | Mức dịch chuyển giờ hoạt động. |
| 33 | `jaccard_hour_7d` | Float64 | Có | `Jaccard(tập giờ hôm nay, tập giờ của H₇)` | Mức thay đổi các khung giờ thường dùng. |
| 34 | `peer_z_share_night` | Float64 | Không | `z_bền_vững(tỷ lệ ngoài giờ, các tài khoản cùng entity_type trong ngày t)` | Mức hoạt động đêm khác nhóm cùng loại. |
| 35 | `delta_mean_distinct_pair_7d` | Float64 | Có | `log1p(số cặp Source–LogHost hôm nay) − trung bình giá trị đó trên H₇` | Mở rộng các đường nguồn → đích. |
| 36 | `novelty_share_pair_7d` | Float64 | Có | `số sự kiện hôm nay trên cặp Source–LogHost mới / số sự kiện có cặp hợp lệ hôm nay` | Tỷ trọng hoạt động trên đường nguồn → đích mới. |
| 37 | `delta_mean_count_7d` | Float64 | Có | `log1p(N_t) − trung bình log1p(N) trên H₇` | Biến động khối lượng, kể cả sau chuỗi ngày im lặng. |
| 38 | `dist_shift_logontype_7d` | Float64 | Có | `TV(phân bố LogonType hôm nay, phân bố LogonType của H₇)` | Chuyển dịch phương thức đăng nhập. |
| 39 | `jaccard_logontype_7d` | Float64 | Có | `Jaccard(tập LogonType hôm nay, tập LogonType của H₇)` | Mức thay đổi các loại logon đã dùng. |
| 40 | `delta_mean_share_night_7d` | Float64 | Có | `off_hours_ratio(t) − trung bình off_hours_ratio trên ngày hoạt động của H₇` | Đột biến tỷ trọng hoạt động ngoài giờ. |
| 41 | `novelty_logontype_7d` | Float64 | Có | `log1p(số LogonType hôm nay chưa thấy trong H₇)` | Loại logon mới so với lịch sử gần. |

**Cột không phải đặc trưng mô hình:** `DomainName`, `UserName`, `entity_type` là `String`; `day` là `Int32`; `total_logons` là số đếm `UInt32` để hiển thị. Giá trị NULL của các biến lịch sử được xử lý bằng imputer fit trên tập train ở tầng mô hình; bảng trên mô tả giá trị lưu trong Parquet.

**Kiểm thử hiện có:** [`tests/test_features.py`](../tests/test_features.py), [`tests/test_template_features.py`](../tests/test_template_features.py) và [`tests/test_configs.py`](../tests/test_configs.py) kiểm tra số lượng/đường sinh đặc trưng, công thức trên dữ liệu tổng hợp, ca biên và hợp đồng schema.
