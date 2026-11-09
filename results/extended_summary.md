# BÁO CÁO KẾT QUẢ THỰC NGHIỆM MỞ RỘNG (ĐIỂM MỚI CỦA NHÓM)

## 1. Benchmark Mở rộng: KMeans-SMOTE, SVM-SMOTE (Bài tập 1), Hybrid & Cost-Sensitive

### Dataset: `synthetic data` — Average Precision (PR-AUC)
| Method           |   Logistic Regression |   Random Forest |   XGBoost |
|:-----------------|----------------------:|----------------:|----------:|
| Nosampling       |                 0.771 |           0.888 |     0.982 |
| Borderline-SMOTE |                 0.715 |           0.965 |     0.973 |
| KMeans-SMOTE     |                 0.755 |           0.896 |     0.981 |
| SVM-SMOTE        |                 0.761 |           0.95  |     0.996 |
| SMOTE-Tomek      |                 0.77  |           0.912 |     0.971 |
| SMOTE-ENN        |                 0.77  |           0.913 |     0.971 |
| Cost-Sensitive   |                 0.772 |           0.93  |     0.974 |

### Dataset: `synthetic data` — ROC-AUC (Đối chiếu ảo giác lạc quan)
| Method           |   Logistic Regression |   Random Forest |   XGBoost |
|:-----------------|----------------------:|----------------:|----------:|
| Nosampling       |                 0.947 |           0.965 |     0.994 |
| Borderline-SMOTE |                 0.923 |           0.989 |     0.993 |
| KMeans-SMOTE     |                 0.937 |           0.971 |     0.994 |
| SVM-SMOTE        |                 0.943 |           0.989 |     1     |
| SMOTE-Tomek      |                 0.947 |           0.979 |     0.99  |
| SMOTE-ENN        |                 0.947 |           0.98  |     0.989 |
| Cost-Sensitive   |                 0.947 |           0.978 |     0.991 |

### Dataset: `synthetic data` — Brier Score (Sai số xác suất, càng thấp càng tốt)
| Method           |   Logistic Regression |   Random Forest |   XGBoost |
|:-----------------|----------------------:|----------------:|----------:|
| Nosampling       |                0.0463 |          0.0275 |    0.0086 |
| Borderline-SMOTE |                0.1913 |          0.0435 |    0.0078 |
| KMeans-SMOTE     |                0.1517 |          0.0497 |    0.0101 |
| SVM-SMOTE        |                0.1632 |          0.0645 |    0.0055 |
| SMOTE-Tomek      |                0.0914 |          0.0588 |    0.0089 |
| SMOTE-ENN        |                0.0917 |          0.0586 |    0.0104 |
| Cost-Sensitive   |                0.0958 |          0.0499 |    0.01   |

### Dataset: `synthetic data` — Fit Time trung bình (ms/fold)
| Method           |   Logistic Regression |   Random Forest |   XGBoost |
|:-----------------|----------------------:|----------------:|----------:|
| Nosampling       |                  10.6 |           421.1 |    1943.8 |
| Borderline-SMOTE |                  22.7 |           579.3 |      87.4 |
| KMeans-SMOTE     |                 637.7 |           509   |      87.7 |
| SVM-SMOTE        |                  31   |           529.9 |      72.9 |
| SMOTE-Tomek      |                  24.5 |           623   |      68.7 |
| SMOTE-ENN        |                  26.6 |           539.9 |      73.1 |
| Cost-Sensitive   |                   9.8 |           467.9 |      51.4 |

### Dataset: `thyroid_sick` — Average Precision (PR-AUC)
| Method           |   Logistic Regression |   Random Forest |   XGBoost |
|:-----------------|----------------------:|----------------:|----------:|
| Nosampling       |                 0.552 |           0.837 |     0.968 |
| Borderline-SMOTE |                 0.602 |           0.819 |     0.967 |
| KMeans-SMOTE     |                 0.333 |           0.733 |     0.967 |
| SVM-SMOTE        |                 0.609 |           0.803 |     0.97  |
| SMOTE-Tomek      |                 0.634 |           0.808 |     0.97  |
| SMOTE-ENN        |                 0.612 |           0.782 |     0.904 |
| Cost-Sensitive   |                 0.602 |           0.827 |     0.974 |

### Dataset: `thyroid_sick` — ROC-AUC (Đối chiếu ảo giác lạc quan)
| Method           |   Logistic Regression |   Random Forest |   XGBoost |
|:-----------------|----------------------:|----------------:|----------:|
| Nosampling       |                 0.927 |           0.972 |     0.997 |
| Borderline-SMOTE |                 0.941 |           0.973 |     0.997 |
| KMeans-SMOTE     |                 0.867 |           0.943 |     0.997 |
| SVM-SMOTE        |                 0.942 |           0.97  |     0.998 |
| SMOTE-Tomek      |                 0.946 |           0.971 |     0.998 |
| SMOTE-ENN        |                 0.944 |           0.97  |     0.994 |
| Cost-Sensitive   |                 0.941 |           0.974 |     0.997 |

### Dataset: `thyroid_sick` — Brier Score (Sai số xác suất, càng thấp càng tốt)
| Method           |   Logistic Regression |   Random Forest |   XGBoost |
|:-----------------|----------------------:|----------------:|----------:|
| Nosampling       |                0.0408 |          0.0313 |    0.0088 |
| Borderline-SMOTE |                0.0892 |          0.0851 |    0.011  |
| KMeans-SMOTE     |                0.0721 |          0.0618 |    0.0093 |
| SVM-SMOTE        |                0.0807 |          0.0752 |    0.0105 |
| SMOTE-Tomek      |                0.0847 |          0.08   |    0.0103 |
| SMOTE-ENN        |                0.0975 |          0.084  |    0.0164 |
| Cost-Sensitive   |                0.0967 |          0.0769 |    0.0092 |

### Dataset: `thyroid_sick` — Fit Time trung bình (ms/fold)
| Method           |   Logistic Regression |   Random Forest |   XGBoost |
|:-----------------|----------------------:|----------------:|----------:|
| Nosampling       |                  37.1 |           657.7 |     165.5 |
| Borderline-SMOTE |                 108.1 |           918.1 |     464.5 |
| KMeans-SMOTE     |                 199.4 |          1052.9 |     477.8 |
| SVM-SMOTE        |                 347.3 |          1272.7 |     560.4 |
| SMOTE-Tomek      |                 902.9 |          1748.1 |    1436.5 |
| SMOTE-ENN        |                 799.9 |          1909.7 |    1228.9 |
| Cost-Sensitive   |                  55.8 |           711.1 |     203.2 |

## 2. Thực chứng Cạm bẫy Khoa học (Data Leakage & Probability Calibration)

- **Data Leakage (Bài tập 3 - Random Forest trên `thyroid_sick`):**
  - Quy trình ĐÚNG (Split -> SMOTE chỉ trên Train): PR-AUC = **0.897**, ROC-AUC = **0.991**
  - Quy trình SAI (SMOTE toàn bộ -> Split): PR-AUC = **0.994**, ROC-AUC = **0.995** *(Thổi phồng giả tạo +0.097 PR-AUC!)*

- **Hiệu chuẩn Xác suất (Trang 56 - Logistic Regression trên `thyroid_sick`, tỷ lệ dương tính thực tế = `0.0612`):**
  - `Nosampling`: Mean Predicted Prob = **0.0621** | Brier Score = **0.0398**
  - `SMOTE (Chưa hiệu chuẩn)`: Mean Predicted Prob = **0.2254** *(Bị thổi phồng gấp 3.7x!)* | Brier Score = **0.0879**
  - `SMOTE + Isotonic Calibration`: Mean Predicted Prob = **0.0634** | Brier Score = **0.0297** *(Khôi phục chuẩn xác suất thực!)*
