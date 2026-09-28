# Feature Dictionary — Ma trận đặc trưng thô (Tài khoản × Ngày)

> **File dữ liệu:** `data/features/raw/feature_matrix_raw.parquet`  
> **Script sinh ma trận:** [`scripts/feature_engineering/extract_account_day_matrix.py`](../../scripts/feature_engineering/extract_account_day_matrix.py)  
> **Phiên bản:** 2.0 (Raw Matrix — không áp dụng log, không scale, không clip)  
> **Quy mô ma trận:** 20 cột (4 trường định danh/nhãn + 16 đặc trưng hành vi thô)

---

## 1. Khóa định danh & Nhãn phân loại (4 trường)

Khóa danh tính được thiết kế theo khuyến nghị từ báo cáo phân tích dữ liệu ngày 16, đảm bảo tính duy nhất và tách biệt giữa tài khoản Domain và tài khoản Local trên từng máy trạm.

| Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả & Ý nghĩa an ninh mạng |
| :--- | :---: | :---: | :--- |
| **`DomainName`** | String | `domain001`, `comp...`, `unknown` | Tên miền Active Directory hoặc tên máy tính cục bộ (chuẩn hóa chữ thường). Nếu thiếu trong log thô sẽ gán nhãn `unknown`. |
| **`UserName`** | String | `User...`, `administrator`, `...$` | Tên tài khoản người dùng hoặc tiến trình thực hiện xác thực. |
| **`day`** | Int32 | $1 \to 60$ | Ngày quan sát chuỗi sự kiện. |
| **`entity_type`** | String | `Admin`, `User`, `Machine`, `Service`, `System`, `Other` | Nhãn vai trò thực thể được gán theo luật tường minh từ cấu hình (phục vụ phân tầng mô hình và xây dựng nhóm đồng đẳng Peer-group). |

> **Khóa chính duy nhất (Composite Primary Key):** `(DomainName, UserName, day)`  
> *Lợi ích:* Tránh việc gộp chung các tài khoản cùng tên `Administrator` ở các máy khác nhau vào một dòng.

---

## 2. Danh mục 16 đặc trưng hành vi thô (Raw Features)

### 🔹 Nhóm 1: Khối lượng (Volume)
| Tên đặc trưng | Kiểu | Miền giá trị | Định nghĩa & Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| **`total_logons`** ⚠️ | Int64 | $[1, +\infty)$ | **Tổng số lần đăng nhập (4624 + 4625) trong ngày.**<br>• *Ý nghĩa:* Quy mô hoạt động tổng thể. Biến động tăng đột biến (Spike) là dấu hiệu tài khoản bị chiếm đoạt hoặc brute-force.<br>• *Ghi chú:* Lệch nặng (skew $\sim 84$), dự kiến áp $\log(1+x)$ ở bước sau. |

---

### 🔹 Nhóm 2: Thất bại & Cảnh báo khóa (Failure & Lockout)
| Tên đặc trưng | Kiểu | Miền giá trị | Định nghĩa & Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| **`failure_ratio`** | Float64 | $[0, 1]$ | **Tỷ lệ đăng nhập thất bại:** $\frac{\text{Event 4625}}{\text{total\_logons}}$.<br>• *Ý nghĩa:* Đo cường độ sai mật khẩu. Người dùng bình thường hiếm khi $> 0.1$; tỷ lệ cao cảnh báo tấn công dò mật khẩu (Password Guessing). |
| **`failure_locked_out_share`** | Float64 | $[0, 1]$ *(Nullable)* | **Tỷ lệ lỗi do khóa tài khoản:** $\frac{\text{FailureReason = 'Account locked out'}}{\text{Event 4625}}$.<br>• *Ý nghĩa:* Chỉ dấu tấn công brute-force liên tục làm kích hoạt chính sách khóa tài khoản của Windows.<br>• *Ghi chú:* **NULL** nếu trong ngày tài khoản không có thất bại nào (`failure_count = 0`). |

---

### 🔹 Nhóm 3: Thời gian & Nhịp sinh học (Time & Rhythm)
| Tên đặc trưng | Kiểu | Miền giá trị | Định nghĩa & Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| **`off_hours_ratio`** | Float64 | $[0, 1]$ | **Tỷ lệ hoạt động ngoài giờ hành chính** (18h tối hôm trước đến 7h sáng hôm sau).<br>• *Ý nghĩa:* Phát hiện truy cập trái phép ban đêm hoặc hoạt động tự động hóa bất thường ngoài ca làm việc. |
| **`interarrival_dt_mean`** | Float64 | $[0, 86400]$ *(Nullable)* | **Khoảng cách thời gian trung bình giữa 2 lần đăng nhập liên tiếp** ($\text{giây}$).<br>• *Ý nghĩa:* Nhịp độ đăng nhập. Số giây quá ngắn chứng tỏ có tool/script chạy tự động.<br>• *Ghi chú:* **NULL** nếu tài khoản chỉ đăng nhập đúng 1 lần trong ngày. |
| **`delta_t_cv`** | Float64 | $[0, +\infty)$ *(Nullable)* | **Hệ số biến thiên nhịp đăng nhập:** $\frac{\text{std}(\Delta t)}{\text{mean}(\Delta t)}$.<br>• *Ý nghĩa:* Đo độ phân tán của nhịp. $CV \approx 0$ nghĩa là các lần logon cách nhau cực kỳ đều đặn (dấu hiệu của Cronjob / Malware Beaconing).<br>• *Ghi chú:* **NULL** nếu chỉ có $\le 1$ sự kiện. |
| **`same_second_share`** | Float64 | $[0, 1]$ | **Tỷ lệ đăng nhập dồn dập cùng 1 giây** ($\Delta t = 0$).<br>• *Ý nghĩa:* Tỷ lệ cao khẳng định hành vi dùng tool/script bắn đồng loạt, thao tác người dùng bình thường không thể bấm phím trong cùng giây. |
| **`is_single_event`** | UInt8 | $\{0, 1\}$ | **Cờ báo ngày đó chỉ có đúng 1 sự kiện.**<br>• *Ý nghĩa:* Cờ nhị phân giải thích lý do các trường nhịp thời gian ở trên nhận giá trị NULL hợp lệ (tránh `fillna(0)` sai lệch). |

---

### 🔹 Nhóm 4: Phương thức đăng nhập (Logon Type)
| Tên đặc trưng | Kiểu | Miền giá trị | Định nghĩa & Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| **`interactive_ratio`** | Float64 | $[0, 1]$ | **Tỷ lệ đăng nhập trực tiếp tại máy (LogonType = 2).**<br>• *Ý nghĩa:* Đặc trưng của người dùng ngồi trước bàn phím vật lý; các tài khoản dịch vụ/máy thường có giá trị này bằng 0. |
| **`rare_logon_type_count`** ⚠️ | UInt32 | $[0, +\infty)$ | **Số lần đăng nhập bằng các kiểu hiếm** (khác Type 2 và Type 3).<br>• *Ý nghĩa:* Bao gồm RDP từ xa (Type 10), RunAs quyền khác (Type 9), Service ngầm (Type 5), Batch job (Type 4)...<br>• *Ghi chú:* Lệch nặng nhất bộ dữ liệu (skew $\sim 150$), dự kiến áp $\log(1+x)$ ở bước sau. |

---

### 🔹 Nhóm 5: Gói xác thực (Authentication)
| Tên đặc trưng | Kiểu | Miền giá trị | Định nghĩa & Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| **`ntlm_ratio`** | Float64 | $[0, 1]$ | **Tỷ lệ sử dụng gói xác thực NTLM** (thay vì Kerberos).<br>• *Ý nghĩa:* NTLM là giao thức cũ dễ bị tấn công Pass-the-Hash. Tỷ lệ NTLM tăng bất thường là chỉ báo tấn công hoặc hạ cấp xác thực (Authentication Downgrade). |

---

### 🔹 Nhóm 6: Đa dạng thực thể & Lan tỏa mạng (Fan-out)
| Tên đặc trưng | Kiểu | Miền giá trị | Định nghĩa & Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| **`distinct_hosts`** ⚠️ | UInt32 | $[1, +\infty)$ | **Số lượng máy đích (`LogHost`) phân biệt mà tài khoản này đăng nhập tới.**<br>• *Ý nghĩa:* Phát hiện leo thang đặc quyền hoặc lây nhiễm lan tỏa ngang (Lateral Movement) sang nhiều máy trạm trong mạng.<br>• *Ghi chú:* Lệch nặng (skew $\sim 89$), dự kiến áp $\log(1+x)$ ở bước sau. |
| **`distinct_sources_count`** ⚠️ | UInt32 | $[0, +\infty)$ | **Số lượng máy nguồn (`Source`) phân biệt phát sinh yêu cầu đăng nhập.**<br>• *Ý nghĩa:* Phát hiện chia sẻ tài khoản (Credential Sharing) hoặc một tài khoản bị chiếm đoạt từ nhiều nguồn máy lạ.<br>• *Ghi chú:* Lệch rất nặng (skew $\sim 148$), dự kiến áp $\log(1+x)$ ở bước sau. |

---

### 🔹 Nhóm 7: Ngữ cảnh nguồn & Tiến trình (Context)
| Tên đặc trưng | Kiểu | Miền giá trị | Định nghĩa & Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| **`missing_source_ratio`** | Float64 | $[0, 1]$ | **Tỷ lệ sự kiện bị thiếu trường máy nguồn (`Source` is null).**<br>• *Ý nghĩa:* Biến kiểm soát chất lượng dữ liệu (Covariate). Log LANL thiếu Source có cấu trúc theo giao thức (Service/Kerberos thiếu nhiều), không tự ý gán đây là hành vi hacker. |
| **`remote_logon_ratio`** | Float64 | $[0, 1]$ | **Tỷ lệ đăng nhập từ xa qua mạng:** $\frac{\text{Source } \neq \text{ LogHost}}{\text{total\_logons}}$.<br>• *Ý nghĩa:* Phân định ranh giới giữa làm việc tại chỗ (`Source == LogHost`) và truy cập nhảy cóc qua mạng (`Source != LogHost`). |
| **`custom_proc_share`** | Float64 | $[0, 1]$ | **Tỷ lệ sự kiện khởi tạo bởi tiến trình ẩn danh** (`Proc<6 số>.exe`).<br>• *Ý nghĩa:* Đo mức độ phụ thuộc vào các ứng dụng nghiệp vụ riêng của mạng LANL thay vì các tiến trình hệ điều hành Windows chuẩn (`services.exe`, `lsass.exe`). |

---

## 3. Kế hoạch chuẩn hóa Log(1 + x) cho bước tiếp theo

Có **4 đặc trưng đếm lệch nặng nhất** cần được áp dụng hàm $\log(1 + x)$ trước khi đưa vào các thuật toán Machine Learning dựa trên khoảng cách (Isolation Forest, AutoEncoder, PCA, k-NN):

| Đặc trưng thô | Tên cột sau biến đổi | Độ lệch gốc (Skew) | Độ lệch sau log | Mức giảm lệch |
| :--- | :--- | :---: | :---: | :---: |
| `rare_logon_type_count` | `rare_logon_type_count_log` | **+150.5** | $\sim 2.0$ | **Giảm 98.7%** |
| `distinct_sources_count` | `log_distinct_sources` | **+148.0** | $\sim 3.6$ | **Giảm 97.4%** |
| `distinct_hosts` | `log_distinct_hosts` | **+88.9** | $\sim 1.3$ | **Giảm 98.6%** |
| `total_logons` | `log_total_logons` | **+83.7** | $\sim -1.3$ | **Giảm 98.6%** |

*(12 đặc trưng còn lại đã là tỷ lệ trong đoạn $[0, 1]$ hoặc hệ số biến thiên $CV$, có phân phối ổn định nên không cần log-transform).*
