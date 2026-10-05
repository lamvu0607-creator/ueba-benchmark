# Báo cáo Thống kê Phân loại Tài khoản & Phân tích Khả thi Học máy cho Nhóm Thiểu số

> **Dự án:** UEBA Anomaly Detection Benchmark (Windows Event 4624 & 4625 — LANL Dataset)  
> **Căn cứ dữ liệu:** Ma trận đặc trưng `data/processed/feature_matrix_processed.parquet` (60 ngày, 1.055.283 dòng × 47 cột)  
> **Ngày lập báo cáo:** 2026-10-05  

---

## 1. Tóm tắt & Bối cảnh

Trong bài toán UEBA (User and Entity Behavior Analytics), một sai lầm phổ biến là gom tất cả các loại tài khoản vào cùng một mô hình học máy (gây ra hiện tượng trộn lẫn quần thể / Simpson's Paradox) hoặc chia nhỏ mù quáng mỗi loại tài khoản một mô hình học máy riêng.

Báo cáo này cung cấp:
1. **Bảng thống kê định lượng chi tiết** của 5 nhóm tài khoản (`Machine`, `User`, `Admin`, `Service`, `System`) trên tập **Test** (ngày 43–60), tập **Train** (ngày 1–42) và toàn bộ 60 ngày.
2. **Bóc tách danh tính cụ thể** của các nhóm thiểu số (`Admin`, `Service`, `System`).
3. **Luận cứ toán học & kỹ thuật** chứng minh vì sao các nhóm thiểu số **hoàn toàn không thể và không nên dùng mô hình học máy không giám sát (Unsupervised ML)**.

---

## 2. Bảng thống kê định lượng theo loại tài khoản

Khoá định danh bản ghi là cặp `(DomainName, UserName)`.

### 2.1. Tập TEST (Ngày 43–60: 18 ngày đánh giá nhãn)

Tổng số dòng: **333.671 dòng** | Tổng số sự kiện log: **305.518.131 sự kiện**

| Loại tài khoản | Số dòng (Account-Day) | Tỷ lệ dòng (%) | Số tài khoản duy nhất `(Domain, User)` | Số `UserName` duy nhất | Tổng số sự kiện log | Tỷ lệ log (%) | Số ngày hoạt động / 18 ngày (Mean / Median / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Machine`** | **180.092** | 53,97% | 12.448 | 12.436 | 136.063.943 | 44,54% | 14,5 / 18 / 18 |
| **`User`** | **152.740** | 45,78% | 14.213 | 12.343 | 112.650.229 | 36,87% | 10,7 / 12 / 18 |
| **`Admin`** | **630** | 0,19% | 336 | **1** | 9.714.192 | 3,18% | 1,9 / 1 / 18 |
| **`Service`** | **137** | 0,04% | **10** | **3** | 38.208.063 | 12,51% | 13,7 / 18 / 18 |
| **`System`** | **72** | 0,02% | **4** | **4** | 8.881.704 | 2,91% | **18,0 / 18 / 18** |
| **Tổng cộng** | **333.671** | **100%** | **27.011** | — | **305.518.131** | **100%** | — |

---

### 2.2. Tập TRAIN (Ngày 1–42: 42 ngày huấn luyện mô hình)

Tổng số dòng: **721.612 dòng** | Tổng số sự kiện log: **637.408.261 sự kiện**

| Loại tài khoản | Số dòng | Tỷ lệ dòng (%) | Số tài khoản `(Domain, User)` | Số `UserName` duy nhất | Tổng số sự kiện log | Tỷ lệ log (%) | Số ngày hoạt động / 42 ngày (Mean / Median / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Machine`** | **404.907** | 56,11% | 12.983 | 12.961 | 289.662.606 | 45,44% | 31,2 / 38 / 42 |
| **`User`** | **314.847** | 43,63% | 16.263 | 13.042 | 223.383.892 | 35,05% | 19,4 / 21 / 42 |
| **`Admin`** | **1.371** | 0,19% | 569 | **1** | 21.461.540 | 3,37% | 2,4 / 1 / 42 |
| **`Service`** | **317** | 0,04% | **9** | **3** | 79.601.037 | 12,49% | 35,2 / 42 / 42 |
| **`System`** | **170** | 0,02% | **6** | **4** | 23.299.226 | 3,66% | 28,3 / 42 / 42 |

---

### 2.3. Toàn bộ tập dữ liệu (Ngày 1–60: 60 ngày)

Tổng số dòng: **1.055.283 dòng** | Tổng số sự kiện log: **942.926.392 sự kiện**

| Loại tài khoản | Số dòng | Tỷ lệ dòng (%) | Số tài khoản `(Domain, User)` | Tổng số sự kiện log | Tỷ lệ log (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Machine`** | 584.999 | 55,44% | 13.497 | 425.726.549 | 45,15% |
| **`User`** | 467.587 | 44,31% | 18.003 | 336.034.121 | 35,64% |
| **`Admin`** | 2.001 | 0,19% | 825 | 31.175.732 | 3,31% |
| **`Service`** | 454 | 0,04% | 10 | 117.809.100 | 12,49% |
| **`System`** | 242 | 0,02% | 6 | 32.180.930 | 3,41% |

---

## 3. Bóc tách danh tính chi tiết của các nhóm thiểu số (Test Set)

Ba nhóm `Admin`, `Service`, `System` chỉ chiếm **0,25% số dòng ma trận** (839 / 333.671 dòng), nhưng lại tạo ra tới **18,6% tổng số sự kiện log toàn hệ thống** (hơn 56,8 triệu sự kiện). Khi đi sâu vào dữ liệu:

### 3.1. Nhóm `System` (72 dòng ở Test) $\to$ Thực chất chỉ có đúng **4 tài khoản**
Cả 4 tài khoản đều hoạt động đủ 18/18 ngày ($4 \times 18 = 72$ dòng):
1. `nt authority\Anonymous`: 18 ngày, tổng **5.866.027 sự kiện** (TB 325.890 log/ngày).
2. `nt authority\system`: 18 ngày, tổng **2.956.503 sự kiện** (TB 164.250 log/ngày).
3. `nt authority\network service`: 18 ngày, tổng **29.921 sự kiện** (TB 1.662 log/ngày).
4. `nt authority\local service`: 18 ngày, tổng **29.253 sự kiện** (TB 1.625 log/ngày).

### 3.2. Nhóm `Service` (137 dòng ở Test) $\to$ Thực chất chỉ có đúng **3 tên dịch vụ**
Chỉ có 10 tài khoản trên toàn mạng, xoay quanh 3 tên: `AppService`, `Scanner`, `winservice`:
1. `domain001\AppService`: 18 ngày $\to$ **18.594.639 sự kiện** (hơn 1 triệu log/ngày!).
2. `domain001\Scanner`: 18 ngày $\to$ **15.747.298 sự kiện** (TB 874.850 log/ngày).
3. `enterpriseappserver\AppService`: 18 ngày $\to$ **3.328.426 sự kiện** (TB 184.913 log/ngày).
4. `domain001\winservice`: 18 ngày $\to$ **479.795 sự kiện** (TB 26.655 log/ngày).
5. `domain002\Scanner`: 18 ngày $\to$ **57.085 sự kiện**.
6. 5 tài khoản còn lại (`domain005\AppService`, `domain003\AppService`...) có volume nhỏ (< 1.000 log).

### 3.3. Nhóm `Admin` (630 dòng ở Test) $\to$ Thực chất chỉ có đúng **1 tên UserName (`Administrator`)**
Toàn bộ nhóm Admin đều mang tên `Administrator`, phân bổ thành 2 cực hoàn toàn trái ngược:
* **2 Domain Admins trung tâm** gánh 99,9% khối lượng:
  1. `domain001\Administrator`: Hoạt động 18/18 ngày $\to$ **9.192.151 sự kiện** (TB 510.675 log/ngày).
  2. `domain002\Administrator`: Hoạt động 18/18 ngày $\to$ **511.914 sự kiện** (TB 28.440 log/ngày).
* **334 Local Administrator máy trạm:** Các tài khoản `compXXXXXX\Administrator` trên từng máy cá nhân chỉ xuất hiện lác đác 1–2 ngày (Mean = 1,9 ngày hoạt động) với 1–2 lần logon cục bộ để cài đặt/bảo trì.

---

## 4. Phân tích: Vì sao các nhóm thiểu số KHÔNG DÙNG HỌC MÁY ĐƯỢC?

Nếu cố tình huấn luyện các mô hình Machine Learning riêng biệt cho `System`, `Service` hoặc `Admin`, các thuật toán sẽ sụp đổ vì 3 nguyên nhân kỹ thuật cốt lõi:

### 4.1. Sụp đổ tham số láng giềng của Local Outlier Factor (LOF)
* Thuật toán LOF ước lượng mật độ cục bộ dựa trên $k$ láng giềng gần nhất (cấu hình chuẩn: `n_neighbors = 20`).
* Với nhóm `System`, toàn bộ tập train chỉ có **170 dòng**, sinh ra từ **4–6 tài khoản thực thể**.
* Với nhóm `Service`, tập train chỉ có **317 dòng**, sinh ra từ **9–10 tài khoản thực thể**.
* **Hậu quả toán học:** Số lượng thực thể độc lập nhỏ hơn cả số láng giềng $k=20$. Mô hình buộc phải lấy các ngày khác nhau của **chính cùng một tài khoản** làm láng giềng của nó. Khoảng cách láng giềng bị suy biến (degenerate), tỷ số mật độ cục bộ (LOF score) mất hoàn toàn ý nghĩa và không thể phát hiện bất thường.

### 4.2. Phân cực nhị nguyên cực đoan (Extreme Bimodality) phá vỡ One-Class SVM và Isolation Forest
* Giả định nền tảng của One-Class SVM và Isolation Forest là: *Dữ liệu bình thường tập trung thành một khối mật độ cao liên tục (hoặc vài cụm mượt mà), còn outlier nằm thưa thớt ở vùng biên*.
* Nhìn vào nhóm `Admin`: Có 2 tài khoản chiếm 9,7 triệu sự kiện, và 334 tài khoản chỉ có 1–2 sự kiện. Khoảng cách Euclid đa chiều giữa hai nhóm này tạo thành một "vực thẳm" nhân tạo.
* Thuật toán SVM kernel RBF khi fit trên phân phối này sẽ không thể hội tụ về một siêu phẳng bao quanh mượt mà:
  * Hoặc nó coi 2 Domain Admin là dị biệt cực đoan (False Positive nghiêm trọng).
  * Hoặc nó coi 334 Local Admin là dị biệt (False Positive toàn bộ).

### 4.3. Bản chất xác định (Deterministic) của Dịch vụ & Hệ thống
* Người dùng (`User`) có tâm lý, cảm xúc và hành vi thay đổi linh hoạt $\to$ bắt buộc dùng Học máy để nắm bắt phân phối mờ.
* Ngược lại, `Service` và `System` là các chương trình máy tính chạy tự động theo kịch bản:
  * Giao thức xác thực (Kerberos/NTLM), loại logon (Type 3 Network), IP nguồn/đích của chúng gần như cố định 100%.
  * **Giải pháp tối ưu vượt trội:** Dùng **Luật nghiệp vụ & Danh sách trắng (Rules & Whitelisting)**:
    * *Ví dụ:* `domain001\AppService` chỉ được phép đăng nhập Type 3 từ Server A sang Server B. Nếu một ngày nó xuất hiện đăng nhập Type 2 (Interactive - có người ngồi trước màn hình gõ phím) hoặc Type 10 (RDP) $\to$ **Kích hoạt cảnh báo mức Critical ngay lập tức**, không cần và không nên trông cậy vào xác suất mờ của AI.

---

## 5. Kết luận & Định hướng kiến trúc tối ưu (Architecture Recommendation)

Thống kê cho thấy **`Machine` và `User` chiếm tới 99,75% số dòng trong toàn bộ hệ thống**. Vì vậy, kiến trúc phân luồng chuẩn công nghiệp cho UEBA phải là:

1. **Luồng Học máy Chuyên sâu (`User` - 45,78% số dòng):**
   * Đối tượng duy nhất cần áp dụng toàn diện các mô hình học máy phức tạp (Isolation Forest, LOF, OCSVM xấp xỉ).
   * Sử dụng các đặc trưng chuẩn hóa độ lệch theo lịch sử cá nhân (`delta_mean_7d`, `jaccard_7d`) và theo nhóm đồng đẳng (`peer_z`).
2. **Luồng Học máy Khối lượng lớn (`Machine` - 53,97% số dòng):**
   * Huấn luyện mô hình cây độc lập (Isolation Forest) với bộ đặc trưng tinh gọn chuyên biệt cho máy trạm (Workstation burst, lateral fan-out, port scan).
3. **Luồng Kiểm soát Đặc quyền & Luật tường minh (`Admin`, `Service`, `System` - 0,25% số dòng):**
   * Tuyệt đối không đưa vào mô hình học máy.
   * Áp dụng Rule-based Baselines, Whitelisting tiến trình và giám sát leo thang đặc quyền (Privilege Escalation).
