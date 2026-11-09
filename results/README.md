# KẾT QUẢ THỰC NGHIỆM & TÁI LẬP CHAPTER 02 (OVERSAMPLING METHODS)

Thư mục này chứa toàn bộ các biểu đồ trực quan hóa, bảng số liệu định lượng và báo cáo tổng hợp thu được từ quá trình **tái lập 100% Chapter 02** (*Machine Learning for Imbalanced Data*, 2023) và **thực nghiệm mở rộng (điểm mới của nhóm)**.

---

## 1. LIÊN KẾT TÀI LIỆU VÀ MÃ NGUỒN CHÍNH
- 📖 **Tài liệu Chuyên đề Đọc sách:** [chapter02/reading_notes.md](../chapter02/reading_notes.md)
- ✅ **Checklist Thực nghiệm (100% hoàn tất):** [chapter02/checklist_thuc_nghiem.md](../chapter02/checklist_thuc_nghiem.md)
- 🔬 **Nhận xét Phản biện & Hướng Mở rộng:** [chapter02/nhan_xet_va_huong_mo_rong.md](../chapter02/nhan_xet_va_huong_mo_rong.md)
- ⚡ **Script Tái lập Chapter 02:** [chapter02/replicate_chapter02.py](../chapter02/replicate_chapter02.py)
- 🧪 **Script Thực nghiệm Mở rộng (Điểm mới):** [chapter02/extended_experiments.py](../chapter02/extended_experiments.py)

---

## 2. KẾT QUẢ TÁI LẬP BENCHMARK SÁCH (`replicate_chapter02.py`)
Phương pháp đánh giá: **Repeated Stratified 5-Fold Cross Validation (3 lần lặp = 15 folds/kịch bản)**  
Độ đo hiệu năng: **Average Precision Score (PR-AUC)**

### 2.1. Dữ liệu Tổng hợp: `Synthetic Moons` (Tỷ lệ mất cân bằng 10:1)
| Kỹ thuật Lấy mẫu | Logistic Regression | Random Forest | XGBoost | Ghi chú |
|:---|:---:|:---:|:---:|:---|
| **Nosampling (Gốc)** | 0.771 | 0.888 | **0.982** | Baseline |
| **RandomOverSampler** | 0.771 | 0.933 | 0.977 | RF tăng +0.045 |
| **SMOTE** | 0.770 | 0.912 | 0.971 | RF tăng +0.024 |
| **ADASYN** | 0.742 | 0.950 | 0.974 | RF tăng +0.062 |
| **Borderline-SMOTE** | 0.715 | **0.965** | 0.973 | **RF tăng mạnh nhất (+0.077)** |

### 2.2. Dữ liệu Thực tế: `Thyroid Sick` (3,772 mẫu, 30 đặc trưng sạch từ OpenML, ~15.3:1)
| Kỹ thuật Lấy mẫu | Logistic Regression | Random Forest | XGBoost | Ghi chú |
|:---|:---:|:---:|:---:|:---|
| **Nosampling (Gốc)** | 0.552 | **0.837** | 0.968 | Baseline RF bản 30 cột cao hơn bản 52 cột cũ (0.776) |
| **RandomOverSampler** | 0.629 | 0.816 | **0.975** | LR tăng +0.077 |
| **SMOTE** | 0.633 | 0.807 | 0.968 | LR tăng +0.081 |
| **ADASYN** | **0.638** | 0.818 | 0.964 | LR đạt đỉnh (+0.086) |
| **Borderline-SMOTE** | 0.602 | 0.819 | 0.967 | LR tăng +0.050 |

- **Bảng số liệu chi tiết:** [`benchmark_results.csv`](benchmark_results.csv) | [`benchmark_summary.md`](benchmark_summary.md)
- **Kết quả Mục 2.6 & 2.7 (Định danh & Đa lớp):** [`categorical_multiclass_summary.md`](categorical_multiclass_summary.md)

---

## 3. KẾT QUẢ THỰC NGHIỆM MỞ RỘNG — ĐIỂM MỚI (`extended_experiments.py`)
- **Báo cáo số liệu đầy đủ:** [`extended_summary.md`](extended_summary.md) | [`extended_benchmark_results.csv`](extended_benchmark_results.csv)
- **Tóm tắt phát hiện nổi bật:**
  1. **`SVM-SMOTE` (Bài tập 1)** đạt **0.996 PR-AUC** trên Synthetic Moons (XGBoost), vượt mọi phương pháp trong sách.
  2. **`KMeans-SMOTE`** sụt giảm trên `thyroid_sick` (`0.333` LR) do đặc thù nhiều biến nhị phân sau One-Hot gây cụm phân mảnh.
  3. **`Cost-Sensitive Learning`** đạt **0.974 PR-AUC** (XGBoost) trên `thyroid_sick` và **nhanh gấp 7 lần** `SMOTE-Tomek` (203 ms vs 1436 ms) mà không cần sinh dữ liệu giả.
  4. **Data Leakage (Bài tập 3):** Resample trước khi chia làm PR-AUC tăng ảo **+0.097** (`0.994` vs `0.897`).
  5. **Probability Calibration (Trang 56):** SMOTE đẩy xác suất dự đoán trung bình của LR từ **6.12% thực tế lên 22.54% (gấp 3.7x)**; áp dụng **Isotonic Calibration** đưa xác suất về **6.34%** và giảm Brier Score từ `0.0879` xuống **`0.0297`**.

---

## 4. DANH MỤC HÌNH ẢNH TRỰC QUAN HÓA (12 BIỂU ĐỒ `.PNG`)

### 4.1. Trực quan hóa không gian thuật toán (Mục 2.1 – 2.7)
1. [`fig2_2_synthetic_1_99.png`](fig2_2_synthetic_1_99.png) — Dữ liệu mất cân bằng 1:99 ban đầu.
2. [`fig2_3_ros_overlap.png`](fig2_3_ros_overlap.png) — Random Oversampling thuần túy (trùng khít điểm cũ).
3. [`fig2_4_ros_shrinkage.png`](fig2_4_ros_shrinkage.png) — Random Oversampling với `shrinkage=0.2`.
4. [`fig2_6_smote.png`](fig2_6_smote.png) — SMOTE nội suy tuyến tính.
5. [`fig2_7_smote_overlap.png`](fig2_7_smote_overlap.png) — Minh họa hiện tượng chồng lấn lớp (Class Overlap) của SMOTE.
6. [`fig2_10_borderline_smote.png`](fig2_10_borderline_smote.png) — Borderline-SMOTE tập trung mẫu biên (*Danger points*).
7. [`fig2_12_adasyn.png`](fig2_12_adasyn.png) — ADASYN thích nghi mật độ theo độ khó.
8. [`fig_multiclass_strategies.png`](fig_multiclass_strategies.png) — So sánh chiến lược `sampling_strategy` trên bài toán 3 lớp.

### 4.2. Biểu đồ Benchmark & Thực nghiệm Mở rộng
9. [`benchmark_perf_synthetic_data.png`](benchmark_perf_synthetic_data.png) — Benchmark 3 mô hình trên Synthetic Moons (Đối chiếu Hình 2.15).
10. [`benchmark_perf_thyroid_sick.png`](benchmark_perf_thyroid_sick.png) — Benchmark 3 mô hình trên Thyroid Sick (Đối chiếu Hình 2.16).
11. [`extended_benchmark_comparison.png`](extended_benchmark_comparison.png) — So sánh mở rộng (KMeans-SMOTE, SVM-SMOTE, SMOTE-Tomek, SMOTE-ENN, Cost-Sensitive).
12. [`extended_pitfalls_leakage_calibration.png`](extended_pitfalls_leakage_calibration.png) — Thực chứng Data Leakage, ảo giác ROC-AUC & Hiệu chuẩn xác suất Isotonic.

---

## 5. HƯỚNG DẪN TÁI LẬP
```powershell
python chapter02/replicate_chapter02.py
python chapter02/extended_experiments.py
```
