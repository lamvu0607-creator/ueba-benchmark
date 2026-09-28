# Data Dictionary — LANL Windows Event Logs

## 1. Mục đích

Tài liệu này giải thích ý nghĩa của 21 trường dữ liệu đang được sử dụng trong dự án **UEBA Benchmark** cho Windows Security Event, đặc biệt tập trung vào:

- **EventID 4624** — Successful Logon
- **EventID 4625** — Failed Logon

Các trường được giải thích theo ngữ cảnh của **LANL Unified Host and Network Dataset**. Một số trường là phiên bản đã được chuẩn hóa/rút gọn từ Windows Security Event Log gốc

Tài liệu này phục vụ trước hết cho bước **Data Profiling** và làm nền cho bước **Quality Check**, EDA và Feature Engineering sau này.

---

## 2. Phân nhóm trường

| Nhóm | Các trường |
|---|---|
| Temporal | `Time` |
| Event | `EventID` |
| Identity | `UserName`, `DomainName`, `LogonID`, `SubjectUserName`, `SubjectDomainName`, `SubjectLogonID` |
| Host / Network | `LogHost`, `Source`, `Destination` |
| Authentication | `LogonType`, `LogonTypeDescription`, `Status`, `ServiceName`, `AuthenticationPackage`, `FailureReason` |
| Process | `ProcessName`, `ProcessID`, `ParentProcessName`, `ParentProcessID` |

---

# 3. Giải thích chi tiết từng trường

## 3.1. `Time`

**Nhóm:** Temporal

**Ý nghĩa:**  
Thời điểm event xảy ra.

Trong LANL Unified Host and Network Dataset, `Time` được biểu diễn dưới dạng **epoch time tính bằng giây**.

**Vai trò:**

- xác định thời điểm event;
- sắp xếp event theo thời gian;
- phân tích số lần đăng nhập theo giờ/ngày;
- tính khoảng cách thời gian giữa các event;
- phát hiện chuỗi hành vi theo thời gian.

**Lưu ý:**

Không nên mặc định coi `Time` chỉ là một biến numerical thông thường. Về nghiệp vụ, đây là trường **temporal** và thường cần được chuyển sang datetime hoặc các đặc trưng thời gian phù hợp khi EDA/Feature Engineering.

---

## 3.2. `EventID`

**Nhóm:** Event

**Ý nghĩa:**  
Mã định danh loại Windows Security Event.

Trong phạm vi hiện tại của dự án:

| EventID | Ý nghĩa |
|---:|---|
| `4624` | Đăng nhập thành công — Successful Logon |
| `4625` | Đăng nhập thất bại — Failed Logon |

**Vai trò:**

Dùng để xác định loại sự kiện đang được phân tích.

**Lưu ý:**

`EventID` có thể được lưu dưới dạng integer nhưng về nghiệp vụ nó là **event code / categorical identifier**, không phải continuous numerical feature.

---

## 3.3. `LogHost`

**Nhóm:** Host / Network

**Ý nghĩa:**  
Tên máy tính nơi event được ghi nhận.

Đối với authentication event có hướng, `LogHost` thường là máy nơi quá trình authentication **kết thúc**, tức máy đích của yêu cầu xác thực.

Ví dụ:

```text
Source Computer
      |
      | authentication
      v
   LogHost
```

Nếu user từ máy `CompA` đăng nhập vào `CompB`, thì thông thường:

```text
Source  = CompA
LogHost = CompB
```

**Vai trò:**

- xác định host nhận authentication;
- phân tích hành vi user theo host;
- xác định tài khoản thường đăng nhập vào những máy nào;
- hỗ trợ xây dựng quan hệ User ↔ Host.

---

## 3.4. `LogonType`

**Nhóm:** Authentication

**Ý nghĩa:**  
Mã số mô tả **kiểu đăng nhập**.

Một số giá trị thường gặp:

| LogonType | Ý nghĩa |
|---:|---|
| `2` | Interactive |
| `3` | Network |
| `4` | Batch |
| `5` | Service |
| `7` | Unlock |
| `8` | NetworkCleartext |
| `9` | NewCredentials |
| `10` | RemoteInteractive |
| `11` | CachedInteractive |

**Ví dụ:**

```text
LogonType = 2
```

nghĩa là user đăng nhập trực tiếp vào máy.

```text
LogonType = 3
```

nghĩa là user/computer đăng nhập từ mạng.

**Lưu ý:**

Đây là **categorical code**, không phải numerical feature theo nghĩa khoảng cách số học.

---

## 3.5. `LogonTypeDescription`

**Nhóm:** Authentication

**Ý nghĩa:**  
Tên mô tả tương ứng với `LogonType`.

Ví dụ:

```text
LogonType = 3
LogonTypeDescription = Network
```

**Vai trò:**

Giúp đọc dữ liệu dễ hơn so với chỉ sử dụng mã số.

**Quality Check liên quan:**

Cần kiểm tra mapping:

```text
LogonType
    ↕
LogonTypeDescription
```

Một `LogonType` không nên có nhiều description khác nhau nếu dataset sử dụng mapping nhất quán.

---

## 3.6. `UserName`

**Nhóm:** Identity

**Ý nghĩa:**  
Tài khoản user chính liên quan đến event.

Theo LANL, đây là user account **initiating the event**.

Ví dụ:

```text
UserName = User012345
```

Nếu tên kết thúc bằng `$`, ví dụ:

```text
Comp123456$
```

thì thường đây là **computer account**, không phải user cá nhân.

**Vai trò:**

Đây là một trong những trường quan trọng nhất đối với UEBA vì entity chính thường là:

```text
User
```

Các hành vi sau này có thể được tổng hợp theo `UserName`.

---

## 3.7. `DomainName`

**Nhóm:** Identity

**Ý nghĩa:**  
Domain của `UserName`.

Ví dụ:

```text
UserName   = User001
DomainName = Domain001
```

Hai trường thường nên được xem như một cặp:

```text
DomainName\UserName
```

**Vai trò:**

- phân biệt cùng username ở các domain khác nhau;
- xác định tài khoản local/domain;
- hỗ trợ xây dựng identity đầy đủ.

**Lưu ý:**

Không nên chỉ sử dụng `UserName` nếu cùng một username có thể xuất hiện trong nhiều domain.

---

## 3.8. `LogonID`

**Nhóm:** Identity

**Ý nghĩa:**  
Identifier của một logon session.

Theo LANL, `LogonID` là giá trị **semi-unique**, có thể dùng để liên kết các event thuộc cùng một phiên đăng nhập trên cùng `LogHost`.

Ví dụ:

```text
LogonID = 0x3E7
```

Các event sau đó trong cùng session có thể chứa cùng `LogonID`.

**Vai trò:**

- correlation giữa các event;
- theo dõi lifecycle của một logon session;
- hỗ trợ reconstruction chuỗi hoạt động.

**Lưu ý rất quan trọng:**

Dù `LogonID` có thể được parse thành số, đây là **identifier**, không phải continuous numerical feature.

Không nên tính:

```text
mean(LogonID)
median(LogonID)
```

vì không có ý nghĩa nghiệp vụ.

---

## 3.9. `SubjectUserName`

**Nhóm:** Identity

**Ý nghĩa:**  
Tài khoản phía **Subject** của event.

Trong Windows Security Event, Subject thường biểu diễn account hoặc process trên hệ thống đã yêu cầu, tạo hoặc báo cáo hoạt động authentication.

Trong cách mô tả của LANL, với authentication mapping event, `SubjectUserName` là tài khoản được ánh xạ tới `UserName`.

**Ví dụ conceptual:**

```text
SubjectUserName
       |
       | requests / maps authentication
       v
    UserName
```

**Lưu ý:**

Không nên mặc định:

```text
SubjectUserName == UserName
```

Hai trường có vai trò khác nhau.

---

## 3.10. `SubjectDomainName`

**Nhóm:** Identity

**Ý nghĩa:**  
Domain của `SubjectUserName`.

Nên đọc cùng:

```text
SubjectDomainName + SubjectUserName
```

Ví dụ:

```text
SubjectDomainName = WORKGROUP
SubjectUserName   = SYSTEM
```

**Vai trò:**

Giúp xác định identity đầy đủ của Subject.

---

## 3.11. `SubjectLogonID`

**Nhóm:** Identity

**Ý nghĩa:**  
Logon session identifier của Subject.

Về bản chất giống `LogonID`, nhưng gắn với phía Subject của event.

**Vai trò:**

Có thể hỗ trợ correlation các event liên quan tới cùng security context/session.

**Lưu ý:**

Đây là **identifier**, không phải continuous numerical feature.

---

## 3.12. `Status`

**Nhóm:** Authentication

**Ý nghĩa:**  
Trạng thái của authentication request.

Theo LANL:

```text
0x0
```

nghĩa là authentication thành công.

Các giá trị khác thường biểu diễn failure/error status.

**Ví dụ:**

```text
Status = 0x0
```

→ thành công.

```text
Status != 0x0
```

→ có lỗi/thất bại.

**Vai trò:**

Đặc biệt quan trọng khi phân tích Event 4625.

**Lưu ý:**

`Status` là **status/error code**, không phải numerical measurement.

---

## 3.13. `Source`

**Nhóm:** Host / Network

**Ý nghĩa:**  
Máy tính nơi authentication bắt đầu.

Ví dụ:

```text
Source  = CompA
LogHost = CompB
```

có thể hiểu là authentication bắt đầu từ `CompA` và được ghi nhận/kết thúc tại `CompB`.

Đối với local logon:

```text
Source == LogHost
```

có thể xảy ra.

**Vai trò UEBA:**

Giúp trả lời:

```text
User đăng nhập từ đâu?
```

và xây dựng quan hệ:

```text
User → Source → LogHost
```

---

## 3.14. `ServiceName`

**Nhóm:** Authentication

**Ý nghĩa:**  
Tên account của computer hoặc service mà user đang yêu cầu ticket/truy cập.

Trường này đặc biệt liên quan đến các event xác thực Kerberos hoặc service ticket.

Ví dụ:

```text
ServiceName = Comp001$
```

**Lưu ý:**

Không phải EventID nào cũng sử dụng trường này.

Do đó missing value ở `ServiceName` không nên tự động bị kết luận là lỗi dữ liệu.

---

## 3.15. `Destination`

**Nhóm:** Host / Network

**Ý nghĩa:**  
Server/resource mà mapped credential đang truy cập.

Có thể hiểu ở mức conceptual:

```text
User
 |
 | credential
 v
Destination
```

Trong một số trường hợp, `Destination` có thể là chính máy local.

**Phân biệt với `LogHost`:**

- `LogHost`: máy ghi nhận event;
- `Destination`: server/resource mà credential đang truy cập;
- `Source`: nơi authentication bắt đầu.

Ba trường này không nên được xem là đồng nghĩa.

---

## 3.16. `AuthenticationPackage`

**Nhóm:** Authentication

**Ý nghĩa:**  
Cơ chế/package authentication được sử dụng.

Các giá trị thường gặp:

```text
Kerberos
NTLM
Negotiate
```

`Negotiate` là security package có thể lựa chọn giữa Kerberos và NTLM tùy ngữ cảnh.

**Vai trò UEBA:**

Có thể dùng để phân tích:

- user thường dùng authentication mechanism nào;
- một user/host bất ngờ chuyển sang mechanism khác;
- tỷ lệ NTLM/Kerberos theo user hoặc host.

Ở bước Profiling chỉ mô tả phân phối; chưa kết luận bất thường.

---

## 3.17. `FailureReason`

**Nhóm:** Authentication

**Ý nghĩa:**  
Lý do authentication/logon thất bại.

Trường này đặc biệt có ý nghĩa đối với:

```text
EventID = 4625
```

Ví dụ conceptual:

```text
wrong password
unknown user
account disabled
account locked
```

Giá trị cụ thể phụ thuộc dataset.

**Lưu ý quan trọng:**

Với:

```text
EventID = 4624
```

authentication đã thành công, do đó `FailureReason` không có ý nghĩa nghiệp vụ và có thể missing/not applicable.

Vì vậy:

```text
FailureReason is null
```

không đồng nghĩa với dữ liệu lỗi.

---

## 3.18. `ProcessName`

**Nhóm:** Process

**Ý nghĩa:**  
Tên executable của process liên quan đến event.

Đối với authentication event, đây có thể là process xử lý hoặc yêu cầu authentication.

Ví dụ:

```text
C:\Windows\System32\svchost.exe
```

hoặc tên executable tương ứng trong dữ liệu LANL.

**Vai trò:**

- xác định process tạo ra authentication;
- liên kết user activity với process;
- hỗ trợ phát hiện hành vi không thông thường ở bước EDA/UEBA.

---

## 3.19. `ProcessID`

**Nhóm:** Process

**Ý nghĩa:**  
Process Identifier — PID của process.

PID dùng để nhận diện một process đang chạy trên một host tại một thời điểm nhất định.

Có thể dùng để correlation với các event khác của cùng process.

Ví dụ:

```text
ProcessID = 0x44c
```

**Lưu ý rất quan trọng:**

`ProcessID` là **identifier**, không phải continuous numerical feature.

PID chỉ có ý nghĩa khi kết hợp với các thông tin như:

```text
LogHost
Time
ProcessName
```

Không nên hiểu:

```text
PID 5000 > PID 1000
```

là một quan hệ định lượng có ý nghĩa.

---

## 3.20. `ParentProcessName`

**Nhóm:** Process

**Ý nghĩa:**  
Tên executable của process cha đã tạo process hiện tại.

Quan hệ:

```text
ParentProcess
      |
      | creates
      v
   Process
```

Ví dụ:

```text
ParentProcessName = explorer
ProcessName       = powershell.exe
```

**Vai trò:**

Hữu ích khi dựng process tree hoặc phân tích lineage của process.

---

## 3.21. `ParentProcessID`

**Nhóm:** Process

**Ý nghĩa:**  
PID của process cha đã tạo process hiện tại.

Có thể sử dụng để tìm event tạo process trước đó, ví dụ EventID `4688`, có:

```text
ProcessID == ParentProcessID
```

Từ đó có thể dựng quan hệ:

```text
Parent Process
      |
      v
Child Process
```

**Lưu ý:**

Giống `ProcessID`, `ParentProcessID` là **identifier**, không phải continuous numerical feature.

---

# 4. Các quan hệ quan trọng giữa các trường

## 4.1. User và Domain

Nên đọc theo cặp:

```text
DomainName + UserName
```

và:

```text
SubjectDomainName + SubjectUserName
```

Không nên phân tích username hoàn toàn tách khỏi domain.

---

## 4.2. Logon Type

Hai trường:

```text
LogonType
LogonTypeDescription
```

phải có mapping logic với nhau.

Ví dụ:

```text
3 ↔ Network
2 ↔ Interactive
```

---

## 4.3. Source và LogHost

Quan hệ thường gặp:

```text
Source
   |
   | authentication
   v
LogHost
```

`Source` trả lời:

> Authentication bắt đầu từ đâu?

`LogHost` trả lời:

> Event được ghi nhận ở máy nào?

---

## 4.4. Status và FailureReason

Đặc biệt quan trọng với Event 4625:

```text
Status
   |
   v
FailureReason
```

Một error/status code cần được xem cùng lý do thất bại nếu dataset có cung cấp.

---

## 4.5. Process và Parent Process

Có thể hình dung:

```text
ParentProcessID
ParentProcessName
       |
       | creates
       v
ProcessID
ProcessName
```

Quan hệ này hữu ích cho process lineage.

---

# 5. Cách hiểu 21 trường dưới góc nhìn UEBA

Có thể gom các trường thành các câu hỏi nghiệp vụ:

```text
WHEN
└── Time

WHAT EVENT
└── EventID

WHO
├── UserName
├── DomainName
├── LogonID
├── SubjectUserName
├── SubjectDomainName
└── SubjectLogonID

FROM WHERE
└── Source

TO WHERE
├── LogHost
└── Destination

HOW
├── LogonType
├── LogonTypeDescription
├── AuthenticationPackage
└── ServiceName

RESULT
├── Status
└── FailureReason

BY WHAT PROCESS
├── ProcessName
├── ProcessID
├── ParentProcessName
└── ParentProcessID
```

Đây là cách đọc rất hữu ích trước khi chuyển sang EDA và Feature Engineering.

---

# 6. Phân biệt identifier, category và numeric

Không nên dựa hoàn toàn vào dtype mà Pandas đọc được.

## Identifier

Các trường sau có thể trông giống số nhưng thực chất là identifier:

```text
LogonID
SubjectLogonID
ProcessID
ParentProcessID
```

Không nên tính các statistic như:

```text
mean
median
standard deviation
```

cho các identifier này.

---

## Categorical / Code

Các trường như:

```text
EventID
LogonType
Status
```

có thể được lưu dưới dạng số hoặc hexadecimal code nhưng về mặt nghiệp vụ là **category/code**.

---

## Temporal

```text
Time
```

là trường temporal dù raw dtype có thể là integer.

---

# 7. Bảng tóm tắt 21 trường

| # | Field | Group | Ý nghĩa ngắn |
|---:|---|---|---|
| 1 | `Time` | Temporal | Thời điểm event xảy ra |
| 2 | `EventID` | Event | Mã loại Windows Security Event |
| 3 | `LogHost` | Host / Network | Máy ghi nhận event / đích authentication |
| 4 | `LogonType` | Authentication | Mã kiểu đăng nhập |
| 5 | `LogonTypeDescription` | Authentication | Mô tả kiểu đăng nhập |
| 6 | `UserName` | Identity | User chính liên quan đến event |
| 7 | `DomainName` | Identity | Domain của `UserName` |
| 8 | `LogonID` | Identity | ID của logon session |
| 9 | `SubjectUserName` | Identity | User phía Subject |
| 10 | `SubjectDomainName` | Identity | Domain của Subject |
| 11 | `SubjectLogonID` | Identity | Logon session ID của Subject |
| 12 | `Status` | Authentication | Status/error code của authentication |
| 13 | `Source` | Host / Network | Máy nơi authentication bắt đầu |
| 14 | `ServiceName` | Authentication | Service/computer account được yêu cầu |
| 15 | `Destination` | Host / Network | Server/resource credential truy cập |
| 16 | `AuthenticationPackage` | Authentication | Cơ chế xác thực như Kerberos/NTLM |
| 17 | `FailureReason` | Authentication | Lý do đăng nhập thất bại |
| 18 | `ProcessName` | Process | Tên process liên quan |
| 19 | `ProcessID` | Process | PID của process |
| 20 | `ParentProcessName` | Process | Tên process cha |
| 21 | `ParentProcessID` | Process | PID của process cha |

---

# 8. Gợi ý dùng trong Profiling

Đối với mỗi trường, `data_dictionary.md` sau khi Profiling hoàn tất có thể bổ sung các thông tin:

| Thuộc tính | Ý nghĩa |
|---|---|
| `Field` | Tên trường |
| `Meaning` | Ý nghĩa nghiệp vụ |
| `Field Group` | Nhóm Temporal / Identity / Authentication / ... |
| `Used In Event` | 4624, 4625 hoặc cả hai |
| `Observed dtype` | Kiểu dữ liệu thực tế |
| `Recommended dtype` | Kiểu dữ liệu đề xuất sau QC |
| `Cardinality` | Số giá trị khác nhau |
| `Missing behavior` | Ý nghĩa của missing |
| `Example` | Ví dụ giá trị |
| `Notes` | Lưu ý nghiệp vụ |

Ví dụ:

```text
Field              : LogonID
Meaning            : Identifier của logon session
Field Group        : Identity
Used In Event      : 4624 / 4625
Observed dtype     : object
Recommended dtype  : string
Cardinality        : high
Missing behavior   : cần kiểm tra theo EventID
Example            : 0x3E7
Notes              : identifier, không phải continuous numeric
```

---

# 9. Nguồn tham khảo

- Los Alamos National Laboratory — **Unified Host and Network Data Set**, phần mô tả các event attributes.
- Microsoft Learn — **Event 4624: An account was successfully logged on**, dùng để đối chiếu semantics của Subject, Logon ID, Logon Type, Process và Authentication Package.

---

# 10. Kết luận

21 trường không nên được nhìn đơn thuần như 21 cột dữ liệu độc lập.

Chúng tạo thành các nhóm quan hệ:

```text
User
 ↓
Authentication
 ↓
Source → LogHost / Destination
 ↓
Status / FailureReason
 ↓
Process
 ↓
Time
```

Hiểu đúng semantics của các trường này là bước nền trước khi thực hiện:

```text
Profiling
    ↓
Quality Check
    ↓
EDA
    ↓
Feature Engineering
    ↓
UEBA Benchmark
```
