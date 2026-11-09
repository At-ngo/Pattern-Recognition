# KẾT QUẢ THỰC NGHIỆM MỤC 2.6 (ĐỊNH DANH) & MỤC 2.7 (ĐA LỚP)

## 1. Mục 2.6: Xử lý Đặc trưng Định danh (Categorical Data)

### 1.1. RandomOverSampler trên dữ liệu hỗn hợp
- **Trước:** `[['abc', 1], ['def', 2], ['ghi', 3]]`, nhãn `[0, 0, 1]`
- **Sau:** `[['abc', 1], ['def', 2], ['ghi', 3], ['ghi', 3]]`, nhãn `[0, 0, 1, 1]`
- *Nhận xét:* ROS sao chép nguyên bản ghi `['ghi', 3]` nên hoạt động tự nhiên với biến chuỗi.

### 1.2. SMOTE-NC (Nominal + Continuous)
- **Trước:** `[['small', 1], ['medium', 2], ['large', 3], ['large', 4], ['large', 5]]`, nhãn `[0, 0, 1, 0, 1]`
- **Sau:** `[['small', 1.0], ['medium', 2.0], ['large', 3.0], ['large', 4.0], ['large', 5.0], ['large', 3.005630378122263]]`, nhãn `[0, 0, 1, 0, 1, 1]`
- *Mẫu tổng hợp mới:* `['large', 3.005630378122263]` -> Cột số nội suy liên tục (`~3.006`), cột chuỗi gán theo majority vote (`'large'`).

### 1.3. SMOTEN (100% Nominal với Value Distance Metric - VDM)
- **Trước:** `['abc', 'def', 'ghi', 'ghi', 'ghi']`, nhãn `[0, 0, 1, 1, 1]`
- **Sau:** `['abc', 'def', 'ghi', 'ghi', 'ghi', 'abc']`, nhãn `[0, 0, 1, 1, 1, 0]`

### 1.4. OrdinalEncoder cho dữ liệu thứ bậc
- Đầu vào `['low', 'medium', 'high', 'high']` -> Mã hóa `[0.0, 1.0, 2.0, 2.0]`

## 2. Mục 2.7: Oversampling trong Bài toán Đa lớp (Multi-class)

| Chiến lược (`sampling_strategy`) | Lớp 0 (Thiểu số) | Lớp 1 (Trung gian) | Lớp 2 (Đa số) | Tổng mẫu |
|:---|:---:|:---:|:---:|:---:|
| `Original` | 10 | 40 | 50 | 100 |
| `minority` | 50 | 40 | 50 | 140 |
| `not minority` | 10 | 50 | 50 | 110 |
| `auto / not majority` | 50 | 50 | 50 | 150 |
| `Custom dict {0:40,1:40,2:50}` | 40 | 40 | 50 | 130 |
