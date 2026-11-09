# BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK CHAPTER 02

### Dữ liệu: `synthetic data`
| Oversampler       |   Logistic Regression |   Random Forest |   XGBoost |
|:------------------|----------------------:|----------------:|----------:|
| Nosampling        |                 0.771 |           0.888 |     0.982 |
| RandomOverSampler |                 0.771 |           0.933 |     0.977 |
| SMOTE             |                 0.77  |           0.912 |     0.971 |
| ADASYN            |                 0.742 |           0.95  |     0.974 |
| Borderline-SMOTE  |                 0.715 |           0.965 |     0.973 |

### Dữ liệu: `thyroid_sick`
| Oversampler       |   Logistic Regression |   Random Forest |   XGBoost |
|:------------------|----------------------:|----------------:|----------:|
| Nosampling        |                 0.552 |           0.837 |     0.968 |
| RandomOverSampler |                 0.629 |           0.816 |     0.975 |
| SMOTE             |                 0.633 |           0.807 |     0.968 |
| ADASYN            |                 0.638 |           0.818 |     0.964 |
| Borderline-SMOTE  |                 0.602 |           0.819 |     0.967 |

