# CHECKLIST THỰC NGHIỆM & TÁI LẬP (REPLICATION CHECKLIST)
## CHAPTER 02: OVERSAMPLING METHODS
**Giáo trình:** *Machine Learning for Imbalanced Data* (Packt Publishing, 2023)  
**Môn học:** Phân tích và Nhận dạng mẫu (Pattern Recognition) - Đại học Sài Gòn (SGU)  
**Nhóm thực hiện:** Ngô Đông Ất, Nguyễn Đăng Khoa, Lê Thanh Phát  

---

## TÓM TẮT TRẠNG THÁI THỰC NGHIỆM HIỆN TẠI (HOÀN TẤT 100%)

| Giai đoạn | Nội dung | Trạng thái | Ghi chú thực tế / File tạo ra |
|:---|:---|:---:|:---|
| **Giai đoạn 1** | Thiết lập môi trường & thư viện | ✅ **100%** | Cài `imbalanced-learn 0.14.2`, `tabulate`, kiểm tra `sklearn`, `xgboost`, `seaborn` |
| **Giai đoạn 2 (2.1–2.5)** | Trực quan hóa 2D (Baseline, ROS, SMOTE, Borderline, ADASYN) | ✅ **100%** | Xuất đủ 7 ảnh `fig2_2` → `fig2_12` vào `results/` |
| **Giai đoạn 2 (2.6–2.7)** | Thực nghiệm Categorical (`SMOTENC`, `SMOTEN`) & Đa lớp | ✅ **100%** | Xuất `categorical_multiclass_summary.md` & `fig_multiclass_strategies.png` |
| **Giai đoạn 3** | Benchmark 5 kỹ thuật × 3 mô hình (Synthetic & Thyroid Sick) | ✅ **100%** | Chạy 450 fits (15 folds/kịch bản), xuất `benchmark_results.csv` & 2 biểu đồ |
| **Giai đoạn 4** | Kiểm tra chống rò rỉ dữ liệu & chuẩn khoa học | ✅ **100%** | Bọc `MinMaxScaler -> Oversampler -> Model` trong `imblearn.pipeline.Pipeline` |
| **Giai đoạn 5** | Xuất bảng, biểu đồ và cập nhật báo cáo `results/` | ✅ **100%** | Đầy đủ file `CSV`, `MD`, `PNG` trong `results/` |
| **Giai đoạn 6 (Mở rộng)** | Điểm mới Seminar (Bài tập 1, Hybrid, Cost-Sensitive, Calibration, Leakage) | ✅ **100%** | Chạy `extended_experiments.py`, xuất `extended_summary.md` & 2 biểu đồ mới |

---

## MỤC TIÊU THỰC NGHIỆM
Tái lập 100% các thí nghiệm trong Chương 02 và mở rộng thêm các thí nghiệm kiểm chứng phản biện, bao gồm:
1. Thí nghiệm trực quan hóa hình học không gian của các thuật toán Oversampling (Hình 2.2 đến 2.12).
2. Thí nghiệm xử lý đặc trưng định danh (`RandomOverSampler`, `SMOTENC`, `SMOTEN`, `OrdinalEncoder`) và bài toán đa lớp (Multi-class).
3. Thí nghiệm đánh giá và so sánh định lượng hiệu năng mô hình (*Logistic Regression, Random Forest, XGBoost*) trên bộ dữ liệu tổng hợp (*Synthetic Moons*) và bộ dữ liệu thực tế (*Thyroid Sick*), đo lường bằng điểm **Average Precision (PR-AUC)** qua kiểm định chéo K-Fold lặp.
4. Thí nghiệm mở rộng (Điểm mới của nhóm): Giải Bài tập 1 (`KMeans-SMOTE`, `SVM-SMOTE`), Hybrid Sampling (`SMOTE-Tomek`, `SMOTE-ENN`), Cost-Sensitive Learning, đo **ROC-AUC vs PR-AUC**, **Brier Score**, **Thời gian huấn luyện**, và thực chứng **Data Leakage + Probability Calibration**.

---

## GIAI ĐOẠN 1: THIẾT LẬP MÔI TRƯỜNG & KIỂM TRA PHỤ THUỘC

- [x] **1.1. Cài đặt Python và các thư viện Machine Learning cốt lõi:**
  - Python 3.10
  - `numpy` (2.2.6), `scipy` (1.15.2), `pandas` (2.3.3)
  - `scikit-learn` (1.6.1)
  - `imbalanced-learn` (0.14.2 — *đã cài bổ sung*)
  - `xgboost` (3.2.0)
  - `matplotlib` (3.10.9), `seaborn` (0.13.2), `tabulate` (0.10.0 — *đã cài bổ sung*)
- [x] **1.2. Chạy lệnh kiểm tra môi trường thành công:**
  ```powershell
  python -c "import sklearn, imblearn, xgboost, pandas, numpy, matplotlib, seaborn; print('SUCCESS: All ML libraries are imported successfully!')"
  ```
- [x] **1.3. Cấu hình thư mục lưu trữ kết quả:**
  - Thư mục `results/` đã cấu hình tự động trong [replicate_chapter02.py](replicate_chapter02.py) và [extended_experiments.py](extended_experiments.py).

---

## GIAI ĐOẠN 2: TÁI LẬP TRỰC QUAN HÓA, ĐỊNH DANH & ĐA LỚP (`chapter02.ipynb`, `smote-plots.ipynb`)

### 2.1. Bộ dữ liệu cơ sở (Baseline Dataset)
- [x] Khởi tạo bộ dữ liệu nhị phân nhân tạo tỷ lệ 1:99 bằng `sklearn.datasets.make_classification(..., weights=[0.99], random_state=81)`.
- [x] Kiểm tra số lượng mẫu thực tế: `Counter({0: 9900, 1: 100})`.
- [x] Vẽ và lưu scatter plot phân bố mất cân bằng 1:99: [results/fig2_2_synthetic_1_99.png](../results/fig2_2_synthetic_1_99.png).

### 2.2. Random Oversampling (ROS) & Tham số Shrinkage
- [x] Chạy `RandomOverSampler(sampling_strategy=1.0, random_state=42)` thuần túy:
  - [x] Kiểm tra kích thước sau khi lấy mẫu: `Counter({0: 9900, 1: 9900})`.
  - [x] Xuất biểu đồ kiểm chứng các điểm mới bị trùng khít tọa độ lên điểm cũ: [results/fig2_3_ros_overlap.png](../results/fig2_3_ros_overlap.png).
- [x] Chạy `RandomOverSampler(sampling_strategy=1.0, random_state=42, shrinkage=0.2)`:
  - [x] Kiểm tra kích thước sau khi lấy mẫu: `Counter({0: 9900, 1: 9900})`.
  - [x] Xuất biểu đồ hiệu ứng tạo nhiễu nhẹ (jittering): [results/fig2_4_ros_shrinkage.png](../results/fig2_4_ros_shrinkage.png).

### 2.3. SMOTE & Hiện tượng Chồng lấn lớp (Class Overlap)
- [x] Chạy `SMOTE(random_state=0)` trên dữ liệu 1:99:
  - [x] Kiểm tra số lượng mẫu: `Counter({0: 9900, 1: 9900})`.
  - [x] Xuất biểu đồ nội suy tạo mạng lưới dày đặc: [results/fig2_6_smote.png](../results/fig2_6_smote.png).
- [x] Tái lập thí nghiệm chồng lấn từ `smote-plots.ipynb`:
  - [x] Tạo dữ liệu 5,000 mẫu với `class_sep=0.9`, `weights=[0.022, 0.967]`.
  - [x] Áp dụng SMOTE và xuất biểu đồ so sánh Trước/Sau cho thấy hiện tượng chồng lấn nghiêm trọng: [results/fig2_7_smote_overlap.png](../results/fig2_7_smote_overlap.png).

### 2.4. Borderline-SMOTE
- [x] Chạy `BorderlineSMOTE(random_state=0)` trên tập dữ liệu 1:99:
  - [x] Kiểm tra số lượng mẫu: `Counter({0: 9900, 1: 9900})`.
  - [x] Xuất biểu đồ kết quả cho thấy mẫu mới chỉ sinh dọc theo rìa tiếp giáp với lớp đa số (*Danger points*): [results/fig2_10_borderline_smote.png](../results/fig2_10_borderline_smote.png).

### 2.5. ADASYN (Adaptive Synthetic Sampling)
- [x] Chạy `ADASYN(random_state=0)` trên tập dữ liệu 1:99:
  - [x] Kiểm tra số lượng mẫu: `Counter({0: 9900, 1: 9900})`.
  - [x] Xuất biểu đồ kết quả cho thấy mật độ điểm tổng hợp tập trung dày hơn tại vùng khó học: [results/fig2_12_adasyn.png](../results/fig2_12_adasyn.png).

### 2.6. Xử lý Đặc trưng Định danh (Categorical Data)
- [x] Chạy `RandomOverSampler` trên dữ liệu hỗn hợp `[['abc', 1], ['def', 2], ['ghi', 3]]`: Sinh bản sao `['ghi', 3]`, giữ nguyên kiểu chuỗi.
- [x] Chạy `SMOTENC(categorical_features=[0], k_neighbors=1, random_state=1)`:
  - [x] Dữ liệu gồm cột nominal (`'small'`, `'medium'`, `'large'`) và cột số `[1..5]`.
  - [x] Kiểm tra kết quả trùng khớp sách trang 52: Sinh mẫu mới `['large', 3.005630378122263]` (cột số nội suy liên tục, cột chuỗi gán theo majority vote).
- [x] Chạy `SMOTEN(k_neighbors=1, random_state=0)` trên dữ liệu 100% định danh:
  - [x] Kiểm tra kết quả trùng khớp sách trang 53: Sinh mẫu `'abc'` cho lớp 0 dựa trên khoảng cách VDM (`['abc', 'def', 'ghi', 'ghi', 'ghi', 'abc']`).
- [x] Kiểm chứng `OrdinalEncoder` cho biến thứ bậc `['low', 'medium', 'high']` -> `[0.0, 1.0, 2.0]`.

### 2.7. Tái cân bằng trong bài toán Đa lớp (Multi-class)
- [x] Tạo dữ liệu 3 lớp với `make_classification(n_classes=3, weights=[0.1, 0.4, 0.5], n_samples=100, random_state=10)` -> `{0: 10, 1: 40, 2: 50}`.
- [x] Chạy chiến lược `sampling_strategy='minority'`: Chỉ nâng lớp 0 lên 50 -> `{0: 50, 1: 40, 2: 50}`.
- [x] Chạy chiến lược `sampling_strategy='not minority'`: Nâng lớp 1 lên 50 -> `{0: 10, 1: 50, 2: 50}`.
- [x] Chạy chiến lược `sampling_strategy='auto'` (`not majority`): Cân bằng tất cả -> `{0: 50, 1: 50, 2: 50}`.
- [x] Chạy chiến lược từ điển tùy chỉnh `sampling_strategy={0: 40, 1: 40, 2: 50}` -> `{0: 40, 1: 40, 2: 50}`.
- [x] Xuất bảng kết quả [results/categorical_multiclass_summary.md](../results/categorical_multiclass_summary.md) và biểu đồ [results/fig_multiclass_strategies.png](../results/fig_multiclass_strategies.png).

---

## GIAI ĐOẠN 3: TÁI LẬP ĐÁNH GIÁ MÔ HÌNH (BENCHMARK TỪ `model-perf-comparison.ipynb`)

### 3.1. Chuẩn bị 2 Bộ dữ liệu Kiểm thử
- [x] **Bộ dữ liệu Tổng hợp (Synthetic Moons):**
  - Sinh 1,100 mẫu bằng `make_moons(noise=0.1, random_state=14)`, giảm mẫu lớp 1 tỷ lệ 10:1 (`Counter({0: 550, 1: 55})`, tổng 605 mẫu).
- [x] **Bộ dữ liệu Thực tế (Thyroid Sick):**
  - *Khác biệt kỹ thuật so với code gốc 2023:* `imblearn.datasets.fetch_datasets()['thyroid_sick']` gặp lỗi HTTP 403 từ Zenodo nên nhóm tải bản gốc `sick` (version 1) từ OpenML.
  - Xử lý bỏ cột `TBG` (rỗng 100%), điền giá trị thiếu (`median`/`most_frequent`), One-Hot Encoding (`drop_first=True`) thành **3,772 mẫu, 30 đặc trưng** (`Counter({0: 3541, 1: 231})`, tỷ lệ ~15.3:1), cache offline tại [chapter02/thyroid_sick.npz](thyroid_sick.npz).

### 3.2. Thiết lập Pipeline & Kiểm định Chéo (Chống Rò rỉ Dữ liệu)
- [x] Sử dụng `imblearn.pipeline.Pipeline([('scaler', MinMaxScaler()), ('over', oversampler), ('model', classifier)])`.
- [x] Thiết lập kiểm định chéo phân tầng lặp: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=4)` (15 folds/kịch bản).
- [x] Thước đo đánh giá chính: **Average Precision (`scoring='average_precision'`)** tức PR-AUC.

### 3.3. Các Kịch bản Đánh giá (5 Oversampling Techniques × 3 Classifiers)
- [x] **5 Phương pháp lấy mẫu:** `Nosampling`, `RandomOverSampler`, `SMOTE`, `ADASYN`, `Borderline-SMOTE` *(bỏ tham số `n_jobs=4` đã deprecated trong `imblearn 0.14.2`)*.
- [x] **3 Mô hình phân loại:** `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier` *(bổ sung công khai kết quả XGBoost vốn bị tác giả lược bỏ khỏi sách PDF)*.

### 3.4. Thu thập & So sánh Kết quả
- [x] Lập bảng so sánh Average Precision (PR-AUC):
  - [x] **Synthetic Data:** LR (`0.715–0.771`), RF (`0.888–0.965`), XGBoost (`0.971–0.982`).
  - [x] **Thyroid Sick:** LR (`0.552–0.638`), RF (`0.807–0.837`), XGBoost (`0.964–0.975`).
- [x] Vẽ 2 đồ thị thanh ngang so sánh:
  - [results/benchmark_perf_synthetic_data.png](../results/benchmark_perf_synthetic_data.png)
  - [results/benchmark_perf_thyroid_sick.png](../results/benchmark_perf_thyroid_sick.png)
- [x] Đối chiếu phát hiện: Khớp hoàn toàn với Hình 2.15 trên Synthetic Data (RF + Borderline-SMOTE đạt **0.965** vs **0.888** baseline), đồng thời phát hiện hiệu ứng thú vị trên bản `thyroid_sick` 30 đặc trưng sạch (RF baseline đạt 0.837 cao hơn bản 52 cột cũ của tác giả).

---

## GIAI ĐOẠN 4: CHECKLIST KIỂM TRA BẪY LỖI KHOA HỌC (PITFALL CHECKLIST)

- [x] **Không xảy ra Data Leakage:** `fit_resample` chỉ chạy tự động bên trong `imblearn.pipeline.Pipeline` trên từng train fold.
- [x] **Giữ nguyên tập kiểm thử gốc:** Validation fold không bị resample.
- [x] **Không dựa vào Accuracy thuần túy:** Đánh giá bằng Average Precision (PR-AUC).
- [x] **Tính tái lập (Reproducibility):** Cố định toàn bộ `random_state`.

---

## GIAI ĐOẠN 5: XUẤT BÁO CÁO & LƯU TRỮ VÀO `results/`

- [x] Lưu bảng kết quả: [results/benchmark_results.csv](../results/benchmark_results.csv), [results/benchmark_summary.md](../results/benchmark_summary.md), [results/categorical_multiclass_summary.md](../results/categorical_multiclass_summary.md).
- [x] Lưu toàn bộ 10 hình ảnh từ phần tái lập sách vào `results/`.
- [x] Cập nhật tổng quan tại [results/README.md](../results/README.md).

---

## GIAI ĐOẠN 6: THỰC NGHIỆM MỞ RỘNG & ĐIỂM MỚI SEMINAR (`extended_experiments.py`)

- [x] **6.1. Giải Bài tập 1 & Mở rộng Hybrid / Algorithm-level:**
  - Chạy benchmark đối chứng 7 phương pháp: `Nosampling`, `Borderline-SMOTE`, `KMeans-SMOTE`, `SVM-SMOTE`, `SMOTE-Tomek`, `SMOTE-ENN`, và `Cost-Sensitive` (`class_weight='balanced'` / `scale_pos_weight`).
  - Đánh giá đa chiều trên **4 thước đo**: `PR-AUC`, `ROC-AUC`, `Brier Score`, và `Fit Time (ms)`.
  - Phát hiện nổi bật: `SVM-SMOTE` + XGBoost đạt **0.996 PR-AUC** trên Synthetic Moons; `Cost-Sensitive` đạt **0.974 PR-AUC** trên `thyroid_sick` nhanh gấp 7 lần `SMOTE-Tomek`; `KMeans-SMOTE` sụt giảm mạnh trên dữ liệu bảng nhiều biến nhị phân (`0.552 -> 0.333` LR).
- [x] **6.2. Thực chứng Data Leakage (Bài tập 3) & Hiệu chuẩn Xác suất (Trang 56):**
  - Chứng minh `SMOTE -> Split` thổi phồng giả tạo **+0.097 PR-AUC** (`0.994` vs `0.897`).
  - Chứng minh `SMOTE` làm sai lệch xác suất dự báo trung bình của LR từ **6.12% thực tế lên 22.54% (gấp 3.7x)**, và khôi phục về **6.34%** (`Brier Score` giảm từ `0.0879` xuống **`0.0297`**) nhờ **Isotonic Calibration**.
- [x] **6.3. Xuất dữ liệu & Biểu đồ mở rộng:**
  - [results/extended_benchmark_results.csv](../results/extended_benchmark_results.csv)
  - [results/extended_summary.md](../results/extended_summary.md)
  - [results/extended_benchmark_comparison.png](../results/extended_benchmark_comparison.png)
  - [results/extended_pitfalls_leakage_calibration.png](../results/extended_pitfalls_leakage_calibration.png)
