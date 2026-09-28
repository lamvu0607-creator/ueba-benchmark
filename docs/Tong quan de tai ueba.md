# CHƯƠNG TRÌNH THỰC CHIẾN AI
## TỔNG QUAN ĐỀ TÀI THỰC TẬP

### Benchmark Isolation Forest / LOF / One-Class SVM trên log xác thực
*Xây dựng baseline UEBA có thể tái lập cho bài toán phát hiện tài khoản hành xử bất thường*

| Hạng mục | Nội dung |
| :--- | :--- |
| **Lĩnh vực** | An toàn thông tin – Machine Learning (Unsupervised Anomaly Detection) |
| **Dữ liệu** | Windows Security Event 4624 (đăng nhập thành công) / 4625 (đăng nhập thất bại) nội bộ, đã ẩn danh |
| **Thời lượng** | 6 tuần (0,5 tuần đầu dành cho setup máy móc và môi trường) |
| **Nhân sự** | 02 thực tập sinh + 01 mentor phụ trách review |
| **Sản phẩm bàn giao** | Mã nguồn pipeline tái lập được, bảng benchmark, báo cáo kỹ thuật và buổi demo nghiệm thu |

---

## 1. MỤC TIÊU

### 1.1. Mục tiêu tổng quát
Xây dựng một baseline UEBA (User and Entity Behavior Analytics) có thể tái lập hoàn toàn cho bài toán phát hiện "tài khoản hành xử bất thường", dựa trên dữ liệu log xác thực Windows. Baseline này đóng vai trò mốc tham chiếu để các nghiên cứu và sản phẩm về sau của đơn vị có cơ sở so sánh khách quan, thay vì mỗi lần lại bắt đầu từ con số không.

### 1.2. Mục tiêu cụ thể
* **Chuẩn hóa dữ liệu:** Xây dựng quy trình làm sạch và tổng hợp log 4624/4625 thành ma trận đặc trưng theo đơn vị phân tích (tài khoản × ngày), có tài liệu mô tả rõ ràng từng trường.
* **Thiết kế bộ đặc trưng hành vi:** Trích xuất tối thiểu 15 đặc trưng phản ánh hành vi xác thực của tài khoản (tần suất, khung giờ, số máy trạm, tỷ lệ thất bại, loại logon, v.v.).
* **Benchmark ba thuật toán:** Cài đặt và so sánh Isolation Forest, Local Outlier Factor (LOF) và One-Class SVM trên cùng một bộ dữ liệu, cùng một bộ đặc trưng và cùng một cách chia tập, đảm bảo so sánh công bằng.
* **Định lượng hiệu năng:** Đo lường bằng bộ chỉ số thống nhất (PR-AUC, Precision@k, Recall theo ngân sách cảnh báo, thời gian huấn luyện/suy luận) và phân tích độ nhạy với tham số contamination.
* **Đảm bảo tính tái lập:** Toàn bộ kết quả phải chạy lại được bằng một lệnh duy nhất, với seed cố định, sai lệch chỉ số giữa các lần chạy dưới 1%.
* **Rút ra khuyến nghị triển khai:** Chỉ ra thuật toán nào phù hợp với loại bất thường nào, ngưỡng cảnh báo hợp lý và khối lượng cảnh báo mà đội vận hành có thể xử lý.

### 1.3. Mục tiêu đào tạo đối với thực tập sinh
Ngoài sản phẩm kỹ thuật, đề tài hướng tới việc thực tập sinh nắm được: tư duy đánh giá mô hình khi không có nhãn thật, kỷ luật thực nghiệm (version dữ liệu, cố định seed, ghi nhật ký thí nghiệm), và khả năng trình bày kết quả cho cả đối tượng kỹ thuật lẫn đối tượng vận hành.

---

## 2. PHẠM VI ĐƠN GIẢN HÓA

Để đề tài khả thi trong 6 tuần với hai thực tập sinh, phạm vi được thu hẹp có chủ đích. Những giới hạn dưới đây là quyết định thiết kế, không phải thiếu sót; mọi hạng mục nằm ngoài phạm vi đều được ghi nhận lại để mở rộng ở giai đoạn sau.

### 2.1. Xây dựng dựa trên

| Thành phần | Lựa chọn cố định |
| :--- | :--- |
| **Nguồn dữ liệu** | Chỉ Windows Security Event ID 4624 và 4625, đã ẩn danh. Không dùng 4768/4769/4776, không dùng log VPN, proxy, EDR hay DNS. |
| **Thuật toán** | Đúng ba thuật toán từ scikit-learn: IsolationForest, LocalOutlierFactor (novelty=True), OneClassSVM (kernel RBF). |
| **Chế độ xử lý** | Xử lý theo lô (batch) trên dữ liệu tĩnh đã trích xuất. Không xây dựng luồng streaming, không xử lý thời gian thực. |
| **Sản phẩm đầu ra** | Mã nguồn dạng script/notebook có cấu trúc, file kết quả CSV, biểu đồ và báo cáo. Không xây dựng giao diện người dùng, không tích hợp SIEM, không đóng gói API. |

### 2.2. Nằm ngoài phạm vi
* Tích hợp trực tiếp với SIEM hoặc hệ thống cảnh báo đang vận hành.
* Phát hiện bất thường theo thời gian thực hoặc cập nhật mô hình trực tuyến (online learning).
* Phân tích mối quan hệ giữa các tài khoản, dò tìm đường di chuyển ngang (lateral movement) bằng phương pháp đồ thị.
* Điều tra pháp chứng, truy vết sự cố thật hoặc đưa ra kết luận về nhân sự cụ thể trong dữ liệu.
* So sánh với các sản phẩm UEBA thương mại.

### 2.3. Giả định và ràng buộc
* **Về nhãn:** Dữ liệu nội bộ không có ground truth. Đề tài chấp nhận dùng bất thường tổng hợp được tiêm vào (synthetic injection) làm nhãn đánh giá chính, kết hợp với đánh giá định tính trên cảnh báo có điểm cao nhất.
* **Về dữ liệu:** Giả định khoảng thời gian dùng để huấn luyện là tương đối "sạch" – tức tỷ lệ hành vi độc hại thực sự trong đó là rất nhỏ. Giả định này phải được nêu rõ trong báo cáo như một hạn chế.
* **Về quyền riêng tư:** Chỉ làm việc trên dữ liệu đã ẩn danh. Thực tập sinh không được phép giải ẩn danh, đối chiếu với danh bạ nội bộ hay xuất dữ liệu ra khỏi môi trường được cấp.
* **Về khối lượng:** Dự kiến 6–8 tuần log. Nếu dữ liệu vượt quá khả năng xử lý của máy đơn, được phép lấy mẫu theo tài khoản nhưng phải ghi rõ phương pháp lấy mẫu.

---

## 3. KẾ HOẠCH THEO TUẦN

Tổng thời lượng 6 tuần. Nửa tuần đầu tiên (khoảng 2,5 ngày làm việc) dành riêng cho việc setup máy móc, môi trường và thủ tục cấp quyền truy cập dữ liệu — đây là giai đoạn thường bị đánh giá thấp và gây trễ tiến độ nếu không tách riêng.

| Tuần | Giai đoạn | Nội dung công việc | Đầu ra kiểm chứng được |
| :---: | :--- | :--- | :--- |
| **0,5** | Setup | Nhận và cấu hình máy | |
| **1** | Hiểu dữ liệu (EDA) | Đọc tài liệu về Event 4624/4625 và ý nghĩa các trường (Logon Type, Status/Sub Status, Workstation Name, IP nguồn, Authentication Package). Thống kê mô tả: số sự kiện theo ngày/giờ, phân bố theo tài khoản, tỷ lệ thất bại, tài khoản máy so với tài khoản người dùng. Phát hiện và xử lý dữ liệu bẩn: trùng lặp, lệch múi giờ, trường thiếu. | Notebook EDA có biểu đồ; từ điển dữ liệu (data dictionary) mô tả từng trường; danh sách quyết định làm sạch dữ liệu kèm lý do. |
| **2** | Kỹ thuật đặc trưng | Xây dựng pipeline tổng hợp log thô thành ma trận (tài khoản × ngày). Trích xuất ≥15 đặc trưng: | Module feature engineering có unit test; file đặc trưng dạng Parquet/CSV; bảng mô tả từng đặc trưng kèm công thức tính. |
| **3** | Cài đặt mô hình | Xây dựng khung benchmark thống nhất: một giao diện chung cho cả ba mô hình, chia tập theo thời gian, cố định seed, ghi nhật ký thí nghiệm. Cài đặt Isolation Forest, LOF và One-Class SVM. Chạy lần đầu với tham số mặc định để lấy kết quả sơ bộ. | Khung benchmark chạy được đầu-cuối; bảng kết quả lần chạy đầu tiên cho cả ba mô hình; nhật ký thí nghiệm. |
| **4** | Đánh giá & tinh chỉnh | Xây dựng bộ sinh bất thường tổng hợp (brute-force, đăng nhập ngoài giờ bất thường, bùng nổ số máy trạm mới, tài khoản im lặng lâu ngày hoạt động trở lại). Tiêm vào tập kiểm thử theo tỷ lệ định trước. Tính toàn bộ chỉ số đánh giá. Quét tham số ở mức vừa phải (contamination, n_estimators, n_neighbors, nu, gamma). Phân tích độ nhạy tham số. | Bảng benchmark đầy đủ 3 mô hình × ≥3 cấu hình; biểu đồ PR và độ nhạy contamination; kết quả lặp trên 5 seed kèm độ lệch chuẩn. |
| **5** | Phân tích lỗi & củng cố | Phân tích theo từng loại bất thường: mô hình nào bắt tốt loại nào. Review thủ công top 50 cảnh báo điểm cao nhất, phân loại thành đáng ngờ / lành tính giải thích được / nhiễu. Chạy ablation loại bỏ từng nhóm đặc trưng để đo đóng góp. Dọn mã nguồn, bổ sung README, kiểm tra tính tái lập trên máy sạch. | Bảng phân tích theo loại bất thường; báo cáo review top 50; kết quả ablation; xác nhận chạy lại được trên môi trường mới. |
| **6** | Báo cáo & bàn giao | Hoàn thiện báo cáo kỹ thuật: phương pháp, kết quả, hạn chế, khuyến nghị triển khai. Chuẩn bị slide và demo. Nghiệm thu với mentor, tiếp nhận phản hồi và chỉnh sửa. Bàn giao repository cùng tài liệu hướng dẫn vận hành. | Báo cáo kỹ thuật hoàn chỉnh; bộ slide demo; repository đã bàn giao; biên bản nghiệm thu. |

---

## 4. TIÊU CHÍ HOÀN THÀNH

Đề tài được coi là hoàn thành khi đáp ứng đầy đủ các tiêu chí bắt buộc dưới đây. Mỗi tiêu chí được thiết kế để kiểm chứng khách quan, tránh đánh giá cảm tính.

### 4.1. Tiêu chí bắt buộc

| TT | Hạng mục | Điều kiện đạt |
| :---: | :--- | :--- |
| **1** | Tính tái lập | Người thứ ba chạy được toàn bộ pipeline trên máy sạch bằng một lệnh duy nhất, theo đúng README, và thu được chỉ số sai lệch không quá 1% so với kết quả báo cáo. Seed được cố định ở mọi bước ngẫu nhiên. |
| **2** | Pipeline dữ liệu | Quy trình từ log thô đến ma trận đặc trưng chạy tự động, có xử lý lỗi cơ bản và ghi log. Mọi quyết định làm sạch dữ liệu được ghi lại kèm lý do. |
| **3** | Bộ đặc trưng | Tối thiểu 15 đặc trưng, mỗi đặc trưng có tên, công thức tính, kiểu dữ liệu và diễn giải ý nghĩa hành vi. Có unit test cho các hàm tính đặc trưng chính. |
| **4** | Bảng benchmark | Đủ 3 mô hình × tối thiểu 3 cấu hình tham số, chạy trên 5 seed, báo cáo giá trị trung bình kèm độ lệch chuẩn cho mọi chỉ số. |
| **5** | Bộ chỉ số | Báo cáo đầy đủ: PR-AUC, ROC-AUC, Precision@10/50/100, Recall tại ngân sách cảnh báo, tỷ lệ cảnh báo, thời gian huấn luyện và thời gian suy luận. |
| **6** | So sánh với baseline nền | Kết quả của cả ba mô hình được đặt cạnh hai mốc tham chiếu: chấm điểm ngẫu nhiên và một luật ngưỡng đơn giản dựa trên số lần đăng nhập thất bại. |
| **7** | Phân tích lỗi | Review thủ công tối thiểu 50 cảnh báo điểm cao nhất, có bảng phân loại và nhận xét. Có phân tích hiệu năng theo từng loại bất thường tổng hợp. |
| **8** | Báo cáo kỹ thuật | Tối thiểu 10 trang, gồm: mô tả dữ liệu, phương pháp, kết quả, phân tích, hạn chế đã biết và khuyến nghị triển khai. Viết đủ rõ để người ngoài nhóm hiểu và làm lại được. |
| **9** | Chất lượng mã nguồn | Mã nguồn có cấu trúc thư mục rõ ràng, không hard-code đường dẫn, tham số nằm trong file cấu hình, đã qua review chéo giữa hai thực tập sinh và được mentor duyệt. |
| **10** | Nghiệm thu | Thực hiện demo trực tiếp 20–30 phút, trả lời được câu hỏi về lựa chọn thiết kế và giới hạn của kết quả. |

### 4.2. Tiêu chí mở rộng (không bắt buộc)
* Bổ sung phân tích khả năng giải thích: chỉ ra đặc trưng nào đóng góp nhiều nhất vào điểm bất thường của từng cảnh báo.
* Thử nghiệm thêm một mô hình thứ tư (ví dụ Elliptic Envelope hoặc HBOS) để mở rộng phổ so sánh.
* Đánh giá độ ổn định của mô hình theo thời gian: huấn luyện trên tuần N, kiểm thử trên tuần N+1, N+2 để quan sát hiện tượng trôi dữ liệu (data drift).
* Ước lượng chi phí vận hành: số cảnh báo mỗi ngày và thời gian xử lý ước tính của một analyst.

*Lưu ý: đề tài không đặt ngưỡng tối thiểu cho giá trị PR-AUC. Một kết quả "mô hình hoạt động kém trên dữ liệu này" vẫn là kết quả hợp lệ và có giá trị, miễn là phương pháp đánh giá chặt chẽ và nguyên nhân được phân tích rõ ràng. Tiêu chí đánh giá là chất lượng thực nghiệm, không phải con số đẹp.*

---

## 5. PHƯƠNG PHÁP ĐÁNH GIÁ

Thách thức trung tâm của đề tài là dữ liệu không có nhãn thật. Phương pháp đánh giá vì vậy được thiết kế theo ba lớp bổ trợ nhau: nhãn tổng hợp để định lượng, mốc tham chiếu để đặt kết quả vào bối cảnh, và đánh giá định tính để kiểm tra tính hữu dụng thực tế.

### 5.1. Chia tập dữ liệu
Chia theo thời gian, tuyệt đối không chia ngẫu nhiên. Giai đoạn đầu (khoảng 70% khoảng thời gian) dùng để huấn luyện với giả định tương đối sạch; giai đoạn sau dùng để kiểm thử. Cách chia này mô phỏng đúng điều kiện triển khai thực tế, nơi mô hình học từ quá khứ và chấm điểm cho tương lai.

### 5.2. Lớp 1 – Nhãn tổng hợp (định lượng chính)
Tiêm các mẫu hành vi bất thường được sinh nhân tạo vào tập kiểm thử với tỷ lệ khoảng 0,5–1% số bản ghi. Mỗi mẫu tiêm được gắn nhãn loại để phân tích sau. Các loại bất thường mô phỏng:

| Loại bất thường | Cách mô phỏng |
| :--- | :--- |
| **Dò mật khẩu (brute-force)** | Tăng đột biến số sự kiện 4625 của một tài khoản trong ngày, tỷ lệ thất bại tiến sát 100%. |
| **Rải mật khẩu (password spraying)** | Nhiều tài khoản cùng ghi nhận số lần thất bại vừa phải từ cùng một IP nguồn trong một khung thời gian ngắn. |
| **Hoạt động ngoài giờ** | Dịch chuyển toàn bộ hoạt động của tài khoản sang khung giờ đêm, khác hẳn hồ sơ hành vi lịch sử. |
| **Bùng nổ máy trạm mới** | Tài khoản đột ngột đăng nhập thành công từ nhiều máy trạm chưa từng xuất hiện trước đó. |
| **Tài khoản ngủ đông thức dậy** | Tài khoản không hoạt động nhiều tuần bỗng phát sinh khối lượng đăng nhập lớn. |
| **Đổi loại logon bất thường** | Tài khoản vốn chỉ dùng logon tương tác (Type 2) chuyển sang logon mạng hoặc dịch vụ (Type 3/5). |

*Cảnh báo phương pháp: bất thường tổng hợp có xu hướng dễ phát hiện hơn tấn công thật. Chỉ số thu được là cận trên lạc quan, không phải ước lượng hiệu năng thực tế. Điều này phải được nêu rõ trong báo cáo.*

### 5.3. Lớp 2 – Mốc tham chiếu
* **Ngẫu nhiên:** chấm điểm ngẫu nhiên, dùng để xác định sàn hiệu năng. Mô hình không vượt được mốc này là không có giá trị.
* **Luật ngưỡng đơn giản:** cảnh báo khi số lần đăng nhập thất bại trong ngày vượt một ngưỡng cố định. Đây là thứ đội vận hành có thể làm trong 10 phút mà không cần machine learning — nếu mô hình không vượt được mốc này thì chưa đủ lý do để triển khai.
* **Z-score đơn biến:** chấm điểm dựa trên độ lệch chuẩn của một đặc trưng duy nhất, dùng để đo giá trị gia tăng của việc mô hình hóa đa biến.

### 5.4. Bộ chỉ số

| Chỉ số | Ý nghĩa | Vai trò |
| :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | Diện tích dưới đường Precision–Recall. | Chỉ số chính. Phù hợp với dữ liệu mất cân bằng nặng hơn ROC-AUC. |
| **ROC-AUC** | Khả năng phân tách tổng thể giữa hai lớp. | Chỉ số phụ, để đối chiếu với tài liệu tham khảo. |
| **Precision@k (k = 10, 50, 100)** | Tỷ lệ cảnh báo đúng trong k cảnh báo điểm cao nhất. | Phản ánh trực tiếp trải nghiệm của analyst khi chỉ xử lý được k cảnh báo mỗi ngày. |
| **Recall tại ngân sách** | Tỷ lệ bất thường bắt được khi giới hạn số cảnh báo mỗi ngày. | Gắn kết quả với năng lực xử lý thực tế của đội vận hành. |
| **Tỷ lệ cảnh báo** | Số cảnh báo sinh ra trên tổng số bản ghi. | Đánh giá gánh nặng vận hành. |
| **Thời gian huấn luyện / suy luận** | Thời gian chạy thực tế trên cùng phần cứng. | One-Class SVM có độ phức tạp bậc cao theo số mẫu; đây là yếu tố quyết định khả năng mở rộng. |

### 5.5. Lớp 3 – Đánh giá định tính
Hai thực tập sinh cùng review thủ công 50 cảnh báo có điểm cao nhất của mỗi mô hình, phân loại độc lập thành ba nhóm: (a) đáng ngờ, cần điều tra; (b) bất thường nhưng giải thích được bằng nghiệp vụ, ví dụ tài khoản dịch vụ hoặc nhân sự trực ca; (c) nhiễu, không mang thông tin. Sau đó đối chiếu kết quả và ghi nhận mức độ đồng thuận. Lớp đánh giá này bù đắp cho điểm yếu của nhãn tổng hợp và cho biết mô hình có thực sự hữu ích với người dùng cuối hay không.

### 5.6. Kiểm soát tính vững
* Lặp lại mỗi thí nghiệm với 5 seed khác nhau; báo cáo trung bình kèm độ lệch chuẩn. Một mô hình có phương sai lớn giữa các seed là mô hình không đáng tin để triển khai.
* Phân tích độ nhạy tham số contamination trên dải giá trị từ 0,1% đến 5%, vẽ biểu đồ biến thiên chỉ số theo tham số.
* Ablation theo nhóm đặc trưng: loại bỏ lần lượt từng nhóm (tần suất, thời gian, đa dạng thực thể, độ lệch lịch sử) để đo mức đóng góp.
* Mọi thí nghiệm ghi vào một file nhật ký chung: thời điểm chạy, mã commit, cấu hình, seed và kết quả.

---

## 6. PHÂN CHIA CÔNG VIỆC CHO 2 NHÂN SỰ (GỢI Ý)

### 6.1. Nguyên tắc phân chia
Công việc được chia theo hạng mục bàn giao, không chia theo tầng kiến trúc. mỗi thực tập sinh chỉ tiếp xúc với nửa bài toán và kết thúc kỳ thực tập với năng lực lệch hẳn về một phía. Mô hình dưới đây đảm bảo cả hai đều đi qua trọn vẹn chuỗi dữ liệu → đặc trưng → mô hình → đánh giá.
* **Chủ trì và đối trọng:** mỗi hạng mục bàn giao có đúng một người chủ trì, chịu trách nhiệm cuối cùng, và một người đối trọng, có nhiệm vụ review, chạy lại được sản phẩm trên máy mình và chất vấn các lựa chọn thiết kế.
* **Luân phiên đều ở mọi tầng:** ở mỗi tầng của pipeline, hai người thay nhau giữ vai chủ trì. Không ai chủ trì trọn một tầng, cũng không ai chỉ đóng vai đối trọng ở một tầng nào đó.
* **Không tự phê duyệt:** không ai được hợp nhất mã nguồn của chính mình. Mọi pull request cần phê duyệt của người còn lại.
* **Cặp đôi ở điểm khó:** những hạng mục định hình kiến trúc chung — khung benchmark và giao diện mô hình thống nhất — được làm chung theo hình thức pair programming, để hai người có cùng một hiểu biết nền.

### 6.2. Bản đồ chủ trì – đối trọng

| Tầng | Hạng mục bàn giao | Chủ trì | Đối trọng |
| :--- | :--- | :---: | :---: |
| **Dữ liệu** | EDA và từ điển dữ liệu cho sự kiện 4624 (đăng nhập thành công) | TTS A | TTS B |
| **Dữ liệu** | EDA và từ điển dữ liệu cho sự kiện 4625 (đăng nhập thất bại) | TTS B | TTS A |
| **Dữ liệu** | Pipeline làm sạch và tổng hợp thành ma trận (tài khoản × ngày) | TTS B | TTS A |
| **Đặc trưng** | Nhóm đặc trưng khối lượng và kết quả xác thực (số lần thành công/thất bại, tỷ lệ thất bại, loại logon) | TTS A | TTS B |
| **Đặc trưng** | Nhóm đặc trưng thời gian và đa dạng thực thể (khung giờ, entropy, số máy trạm/IP, thực thể mới) | TTS B | TTS A |
| **Mô hình** | Khung benchmark, giao diện mô hình thống nhất và LOF | Làm chung (pair programming) | Mentor review |
| **Mô hình** | Isolation Forest và mốc tham chiếu Z-score đơn biến | TTS A | TTS B |
| **Mô hình** | One-Class SVM và mốc tham chiếu luật ngưỡng đơn giản | TTS B | TTS A |
| **Đánh giá** | Bộ sinh bất thường tổng hợp và quy trình tiêm nhãn | TTS B | TTS A |
| **Đánh giá** | Cài đặt bộ chỉ số, đường PR và biểu đồ kết quả | TTS A | TTS B |
| **Phân tích** | Phân tích theo loại bất thường và độ nhạy tham số contamination | TTS A | TTS B |
| **Phân tích** | Ablation theo nhóm đặc trưng và kiểm tra tái lập trên máy sạch | TTS B | TTS A |
| **Chung** | Review thủ công top 50 cảnh báo | Cả hai chấm độc lập rồi đối chiếu | — |
| **Chung** | Báo cáo kỹ thuật và buổi demo nghiệm thu | Cả hai | Mentor |

### 6.3. Vai trò luân phiên hàng tuần
Ngoài các hạng mục kỹ thuật, mỗi tuần một người giữ vai "kỹ sư phát hành", luân phiên theo tuần chẵn – tuần lẻ. Người giữ vai này chịu trách nhiệm: giữ nhánh chính luôn chạy được, thực hiện kiểm tra tái lập vào cuối tuần, cập nhật nhật ký thí nghiệm, và chuẩn bị nội dung cho buổi review với mentor. Cơ chế này đảm bảo cả hai đều hình thành thói quen kỷ luật thực nghiệm thay vì phó thác cho một người.

### 6.4. Phân công theo tuần

| Tuần | Thực tập sinh A | Thực tập sinh B |
| :---: | :--- | :--- |
| **0,5** | Setup máy và môi trường; khởi tạo repository, cấu trúc thư mục, file requirements cố định phiên bản. (Kỹ sư phát hành) | Setup máy và môi trường; thiết lập quy ước Git, mẫu pull request và quy trình review; dựng khung nhật ký thí nghiệm. |
| **1** | Chủ trì EDA sự kiện 4624: phân bố theo giờ và theo tài khoản, phân biệt tài khoản máy với tài khoản người dùng, từ điển trường. | Chủ trì EDA sự kiện 4625: phân bố mã Status/Sub Status, phát hiện dữ liệu bẩn, lệch múi giờ, trùng lặp. (Kỹ sư phát hành) |
| **1** | Chung cuối tuần: chốt lược đồ (schema) file đặc trưng và bàn giao file giả lập đúng schema để hai bên làm việc song song. | Chung cuối tuần: chốt lược đồ file đặc trưng và thống nhất cách chia tập theo thời gian. |
| **2** | Chủ trì nhóm đặc trưng khối lượng và kết quả xác thực; viết unit test cho nhóm này. (Kỹ sư phát hành) | Chủ trì pipeline tổng hợp (tài khoản × ngày) và nhóm đặc trưng thời gian – đa dạng thực thể. |
| **3** | Pair programming nửa đầu tuần: cùng dựng khung benchmark, giao diện mô hình thống nhất và cài đặt LOF làm bản tham chiếu. | Pair programming nửa đầu tuần: cùng dựng khung benchmark, giao diện mô hình thống nhất và cài đặt LOF làm bản tham chiếu. (Kỹ sư phát hành) |
| **3** | Nửa sau tuần: cài đặt Isolation Forest và mốc tham chiếu Z-score đơn biến; chạy kết quả sơ bộ. | Nửa sau tuần: cài đặt One-Class SVM và mốc tham chiếu luật ngưỡng; đóng băng bộ đặc trưng phiên bản v1. |
| **4** | Chủ trì cài đặt bộ chỉ số (PR-AUC, Precision@k, Recall theo ngân sách) và biểu đồ kết quả. (Kỹ sư phát hành) | Chủ trì bộ sinh bất thường tổng hợp và quy trình tiêm nhãn vào tập kiểm thử. |
| **4** | Chung: chia đôi khối lượng chạy lưới tham số trên 5 seed, mỗi người phụ trách một nửa cấu hình. | Chung: chia đôi khối lượng chạy lưới tham số trên 5 seed; tổng hợp kết quả vào một bảng duy nhất. |
| **5** | Chủ trì phân tích hiệu năng theo từng loại bất thường và độ nhạy tham số contamination. | Chủ trì ablation theo nhóm đặc trưng; kiểm tra tái lập đầu-cuối trên máy sạch. (Kỹ sư phát hành) |
| **5** | Chung: review độc lập top 50 cảnh báo, sau đó đối chiếu và đo mức độ đồng thuận. | Chung: review độc lập top 50 cảnh báo, sau đó đối chiếu và đo mức độ đồng thuận. |
| **6** | Viết chương phương pháp và kết quả; chuẩn bị nửa đầu bộ slide demo. (Kỹ sư phát hành) | Viết chương hạn chế và khuyến nghị triển khai; hoàn thiện README và hướng dẫn vận hành; chuẩn bị nửa sau bộ slide. |

### 6.5. Quy tắc cộng tác bắt buộc
* Không ai hợp nhất pull request của chính mình; mỗi PR cần ít nhất một phê duyệt từ người còn lại.
* Người đối trọng phải chạy được sản phẩm của người chủ trì trên máy mình trước khi phê duyệt — không chấp nhận review chỉ đọc mã.
* Tối thiểu 3 giờ pair programming mỗi tuần, dành cho hạng mục khó nhất của tuần đó.
* Họp đồng bộ 15 phút mỗi ngày; review 30 phút với mentor vào cuối tuần, do kỹ sư phát hành của tuần đó chuẩn bị nội dung.
* Lược đồ file đặc trưng được chốt cuối tuần 1 và coi như hợp đồng kỹ thuật giữa hai người; bộ đặc trưng đóng băng ở phiên bản v1 cuối tuần 3.
* Mọi thí nghiệm ghi vào nhật ký chung, bất kể ai chạy: thời điểm, mã commit, cấu hình, seed, kết quả.

### 6.6. Rủi ro của mô hình này và cách kiểm soát

| Rủi ro | Biện pháp kiểm soát |
| :--- | :--- |
| **Trách nhiệm bị pha loãng — việc chung hóa việc không của ai** | Mỗi hạng mục có đúng một người chủ trì ghi tên trong bản đồ ở mục 6.2. "Làm chung" chỉ áp dụng cho ba hạng mục được nêu đích danh. |
| **Chi phí chuyển ngữ cảnh do mỗi người làm nhiều lớp** | Trong một tuần, mỗi người chỉ chủ trì tối đa một hạng mục lớn. Việc chuyển tầng diễn ra giữa các tuần, không diễn ra trong ngày. |
| **Một người chờ người kia hoàn thành đặc trưng** | Bàn giao file đặc trưng giả lập đúng lược đồ ngay cuối tuần 1, cho phép công việc mô hình bắt đầu song song. |
| **Chất lượng không đồng đều giữa hai nửa** | Vai trò đối trọng bắt buộc phải chạy lại được sản phẩm; mentor kiểm tra chéo vào các mốc cuối tuần 3 và tuần 5. |

---

## 7. BẢNG ĐÁNH GIÁ NĂNG LỰC THỰC TẬP SINH

| Mã | Tiêu chí | Trọng số | Bằng chứng dùng để chấm |
| :---: | :--- | :---: | :--- |
| **A1** | Xử lý dữ liệu và kỹ thuật đặc trưng | 15% | Chất lượng pipeline và nhóm đặc trưng do mình chủ trì; unit test; xử lý dữ liệu bẩn; không để rò rỉ thông tin tương lai vào đặc trưng. |
| **A2** | Mô hình hóa và thiết kế thực nghiệm | 15% | Mô hình do mình chủ trì; cách chia tập; thiết kế lưới tham số; lựa chọn và cài đặt chỉ số đánh giá. |
| **A3** | Kỷ luật thực nghiệm và tính tái lập | 15% | Cố định seed; nhật ký thí nghiệm đầy đủ; kết quả chạy lại sai lệch dưới 1%; chất lượng các tuần giữ vai kỹ sư phát hành. |
| **A4** | Chất lượng mã nguồn và quy trình Git | 10% | Cấu trúc thư mục; tham số tách khỏi mã; lịch sử commit; chất lượng nhận xét khi đóng vai đối trọng review. |
| **B1** | Hiểu ngữ cảnh an toàn thông tin | 10% | Mức độ nắm ngữ nghĩa Event 4624/4625; tính hợp lý của các mẫu bất thường tổng hợp; chất lượng nhận định khi review top 50 cảnh báo. |
| **B2** | Tư duy phản biện và phân tích kết quả | 15% | Có nhận ra giới hạn của nhãn tổng hợp không; có kết luận quá mạnh từ dữ liệu yếu không; chất lượng phân tích lỗi và ablation. |
| **C1** | Tài liệu và báo cáo | 8% | Từ điển dữ liệu; README; chương báo cáo do mình viết; người ngoài nhóm đọc có hiểu và làm lại được không. |
| **C2** | Cộng tác và tiếp nhận phản hồi | 7% | Tính xây dựng trong review chéo; phản ứng với góp ý; đóng góp trong các buổi pair programming. |
| **C3** | Chủ động và quản lý tiến độ | 5% | Tự nêu vướng mắc sớm thay vì để lộ ra ở buổi review; bám mốc tuần; tự tìm tài liệu khi gặp vấn đề mới. |
| | **TỔNG** | **100%** | |

*— Hết —*