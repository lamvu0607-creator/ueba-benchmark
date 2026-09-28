# Báo cáo Kiểm định Dữ liệu Interim (Event 4624 & 4625)
*Thời điểm tạo: 2026-09-19 18:07:09*

---
## 1. Tổng quan Toàn vẹn File & Schema (Integrity Check)

- **Tổng số file Parquet kiểm tra:** 120
- **Số file hợp lệ (Pass):** 120
- **Số file hỏng/lỗi (Fail):** 0
- **Tổng dung lượng:** 11.19 GB
- **Tổng số records Event 4624:** 929,103,392
- **Tổng số records Event 4625:** 13,823,040

## 3. Kiểm định Chất lượng Dữ liệu & Miền Giá trị (Data Quality)

### File: `event_4624_day-01.parquet` (Event 4624) — ✅ PASS
- **Tổng số dòng:** 14,110,061
- **Dải thời gian (Time range):** `[1 -> 86399]`
- **Tập LogonType:** `[0, 2, 3, 4, 5, 7, 8, 9, 10, 11]`

| Tên trường | Số lượng Null | Tỷ lệ Null (%) | Đánh giá |
| :--- | :---: | :---: | :---: |
| `Time` | 0 | 0.00% | Bình thường |
| `EventID` | 0 | 0.00% | Bình thường |
| `LogHost` | 0 | 0.00% | Bình thường |
| `LogonType` | 0 | 0.00% | Bình thường |
| `LogonTypeDescription` | 0 | 0.00% | Bình thường |
| `UserName` | 0 | 0.00% | Bình thường |
| `DomainName` | 1 | 0.00% | Bình thường |
| `LogonID` | 28,042 | 0.20% | Bình thường |
| `SubjectUserName` | 14,097,579 | 99.91% | Bình thường |
| `SubjectDomainName` | 14,097,579 | 99.91% | Bình thường |
| `SubjectLogonID` | 14,097,579 | 99.91% | Bình thường |
| `Status` | 14,110,061 | 100.00% | Bình thường |
| `Source` | 1,761,899 | 12.49% | Bình thường |
| `ServiceName` | 14,110,061 | 100.00% | Bình thường |
| `Destination` | 14,110,061 | 100.00% | Bình thường |
| `AuthenticationPackage` | 0 | 0.00% | Bình thường |
| `FailureReason` | 14,110,061 | 100.00% | Chuẩn (4624 không có failure) |
| `ProcessName` | 13,517,603 | 95.80% | Bình thường |
| `ProcessID` | 13,517,603 | 95.80% | Bình thường |
| `ParentProcessName` | 14,110,061 | 100.00% | Bình thường |
| `ParentProcessID` | 14,110,061 | 100.00% | Bình thường |

### File: `event_4625_day-01.parquet` (Event 4625) — ✅ PASS
- **Tổng số dòng:** 335,425
- **Dải thời gian (Time range):** `[1 -> 86398]`
- **Tập LogonType:** `[2, 3, 4, 5, 7, 8, 10, 11]`

| Tên trường | Số lượng Null | Tỷ lệ Null (%) | Đánh giá |
| :--- | :---: | :---: | :---: |
| `Time` | 0 | 0.00% | Bình thường |
| `EventID` | 0 | 0.00% | Bình thường |
| `LogHost` | 0 | 0.00% | Bình thường |
| `LogonType` | 0 | 0.00% | Bình thường |
| `LogonTypeDescription` | 0 | 0.00% | Bình thường |
| `UserName` | 0 | 0.00% | Bình thường |
| `DomainName` | 9 | 0.00% | Bình thường |
| `LogonID` | 133,310 | 39.74% | Bình thường |
| `SubjectUserName` | 335,425 | 100.00% | Bình thường |
| `SubjectDomainName` | 335,425 | 100.00% | Bình thường |
| `SubjectLogonID` | 335,425 | 100.00% | Bình thường |
| `Status` | 335,425 | 100.00% | Bình thường |
| `Source` | 83,698 | 24.95% | Bình thường |
| `ServiceName` | 335,425 | 100.00% | Bình thường |
| `Destination` | 335,425 | 100.00% | Bình thường |
| `AuthenticationPackage` | 0 | 0.00% | Bình thường |
| `FailureReason` | 0 | 0.00% | Chuẩn (100% có failure reason) |
| `ProcessName` | 133,310 | 39.74% | Bình thường |
| `ProcessID` | 133,310 | 39.74% | Bình thường |
| `ParentProcessName` | 335,425 | 100.00% | Bình thường |
| `ParentProcessID` | 335,425 | 100.00% | Bình thường |
