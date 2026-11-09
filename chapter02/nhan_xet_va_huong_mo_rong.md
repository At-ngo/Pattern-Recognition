# NHẬN XÉT CHUYÊN SÂU & HƯỚNG MỞ RỘNG NGHIÊN CỨU
## CHAPTER 02: OVERSAMPLING METHODS
**Môn học:** Phân tích và Nhận dạng mẫu (Pattern Recognition) - Đại học Sài Gòn (SGU)  
**Nhóm thực hiện:** Ngô Đông Ất, Nguyễn Đăng Khoa, Lê Thanh Phát  
**Giảng viên hướng dẫn:** ThS. Nguyễn Thanh Phước  

---

## PHẦN I: NHẬN XÉT & ĐÁNH GIÁ PHẢN BIỆN CHUYÊN SÂU (CRITICAL ANALYSIS)

### 1. Bản chất Hình học & Nghịch lý Số chiều (Curse of Dimensionality trong SMOTE)
- **Bản chất toán học:** Thuật toán SMOTE cổ điển dựa trên giả định **nội suy lồi (Convex Linear Interpolation)** trong không gian khoảng cách Euclid:
  $$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \in [0, 1]$$
- **Hạn chế trong không gian nhiều chiều ($d \gg 10$):**
  - Trong không gian 2 chiều (như các ví dụ minh họa trong sách), đoạn thẳng nối giữa 2 điểm láng giềng thường nằm trọn trong cụm phân bố của lớp thiểu số.
  - Tuy nhiên, trong không gian số chiều cao (High-dimensional feature space), hiện tượng **tập trung khoảng cách (Distance Concentration Phenomenon)** xảy ra: Tỷ lệ giữa khoảng cách tới điểm xa nhất và điểm gần nhất tiến dần về 1.
  - Hậu quả: Khái niệm "láng giềng gần nhất" trong KNN mất dần tính phân biệt cục bộ. Đoạn thẳng nối giữa 2 điểm có thể cắt ngang qua vùng không gian trống (Empty Space) hoặc xuyên thủng sang vùng lãnh thổ của lớp đa số, tạo ra các "mẫu giả" (Ghost Samples) không tồn tại trong phân phối thực tế.

---

### 2. Khoảng cách giữa Dữ liệu Nhân tạo (Toy Datasets) và Dữ liệu Thực tế (Real-world Tabular Data)
Dựa trên kết quả thực nghiệm trong Chapter 02:
- **Trên tập Synthetic Moons (Hình 2.15):**
  - Borderline-SMOTE đem lại bước nhảy vọt ngoạn mục trên mô hình Random Forest (Average Precision tăng từ **0.89** lên **0.96**).
  - *Nguyên nhân:* Ranh giới hai vầng trăng là đường cong liên tục, trơn và không gian 2D rất sạch. Borderline-SMOTE tập trung lấp đầy các mẫu ngay tại rãnh ngăn cách giữa 2 vầng trăng, giúp các cây quyết định chia lát cắt biên chính xác tuyệt đối.
- **Trên tập Dữ liệu Thực tế Thyroid Sick (Hình 2.16):**
  - Hiệu quả của các phương pháp Oversampling (ROS, SMOTE, ADASYN, Borderline-SMOTE) gần như **bằng phẳng và tương đương nhau** (~0.62 trên Logistic Regression và ~0.78-0.79 trên Random Forest).
  - *Nguyên nhân:* Dữ liệu y tế thực tế chứa nhiều đặc trưng dạng rời rạc/nhị phân (các triệu chứng lâm sàng), có sự tương tác phi tuyến phức tạp và có sự chồng lấn thực sự giữa nhóm bệnh nhân và nhóm khỏe mạnh. Lúc này, các phương pháp sinh mẫu cục bộ không thể bù đắp được sự thiếu hụt thông tin bản chất của bài toán.

---

### 3. Tương tác giữa Phương pháp Lấy mẫu và Loại Mô hình Phân loại (Model - Resampling Interaction)

```
+-----------------------------------------------------------------------------------------+
|                               TƯƠNG TÁC MÔ HÌNH VÀ OVERSAMPLING                          |
+-----------------------------------------------------------------------------------------+
| 1. Mô hình Tuyến tính (Logistic Regression):                                            |
|    - Ranh giới quyết định là siêu phẳng: w^T x + b = 0.                                 |
|    - Việc nhân bản hay nội suy thêm mẫu thiểu số chủ yếu làm thay đổi hệ số chặn b      |
|      (tương đương với dịch chuyển ngưỡng quyết định).                                   |
|    - Ít có khả năng uốn lượn theo các cụm mẫu mới sinh.                                 |
+-----------------------------------------------------------------------------------------+
| 2. Mô hình Phi tuyến Ensemble (Random Forest, XGBoost):                                 |
|    - Ranh giới quyết định là tập hợp các siêu hộp trực giao (Axis-aligned hyper-boxes). |
|    - Khi có nhiều điểm tổng hợp sinh ra ở vùng biên (Borderline-SMOTE), cây quyết định  |
|      có thể tạo thêm các phân nhánh (splits) sâu hơn, ôm sát vùng thiểu số.             |
|    - Tận dụng tối đa được dữ liệu nội suy để học cấu trúc phi tuyến phức tạp.          |
+-----------------------------------------------------------------------------------------+
```

---

### 4. Tại sao Average Precision (PR-AUC) là Thước đo Vàng cho Dữ liệu Mất cân bằng?
Nghiên cứu của chương 02 đã sử dụng đúng đắn chỉ số **Average Precision (AP / PR-AUC)** thay vì ROC-AUC hay Accuracy:
- **Nghịch lý Accuracy:** Với tỷ lệ mất cân bằng 1:99, một mô hình tầm thường (dự đoán tất cả là lớp đa số) vẫn đạt **99% Accuracy**, nhưng giá trị thực tiễn bằng 0.
- **Cạm bẫy của ROC-AUC:**
  $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
  Khi lớp đa số rất lớn (TN cực lớn, ví dụ 9,900), mẫu số rất lớn khiến FPR luôn giữ ở mức cực nhỏ (gần 0) ngay cả khi mô hình dự đoán nhầm hàng chục ca bệnh (FP lớn). Do đó, đường cong ROC bị đẩy lên cao, cho điểm ROC-AUC ảo tưởng (thường > 0.95).
- **Sức mạnh của Precision-Recall Curve (PR-AUC):**
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  Cả Precision và Recall đều chỉ tập trung vào lớp thiểu số (lớp dương tính - Positive). Bất kỳ một sự gia tăng nào của số ca báo động nhầm (FP) đều lập tức kéo tụt điểm Precision, giúp phản ánh trung thực năng lực phát hiện lớp thiểu số của hệ thống.

---

### 5. Cảnh báo về Sai lệch Xác suất Hậu nghiệm (Probability Distortion)
- Khi áp dụng Oversampling (ví dụ nâng tỷ lệ từ 1:99 lên 1:1), ta đã nhân tạo hóa xác suất tiên nghiệm $P(Y=1) = 0.5$.
- Sau khi huấn luyện, hàm sigmoid/softmax của mô hình sẽ dự báo ra xác suất hậu nghiệm $P(Y=1|\mathbf{x})$ bị **thổi phồng gấp nhiều lần** so với xác suất thực tế ngoài đời sống.
- **Hậu quả:** Nếu mô hình dùng để định giá rủi ro bảo hiểm hoặc tính xác suất sập hệ thống đám mây, các giá trị xác suất này sẽ hoàn toàn sai lệch nếu không được **Hiệu chuẩn xác suất (Probability Calibration - Platt Scaling / Isotonic Regression)** trước khi đưa vào khai thác.

---

## PHẦN II: HƯỚNG MỞ RỘNG CHO SEMINAR & ĐỀ TÀI NGHIÊN CỨU (FUTURE EXTENSIONS)

Nhóm đề xuất 5 hướng mở rộng khoa học và thực nghiệm có giá trị cao để phát triển đề tài Seminar môn Nhận dạng mẫu:

```
                            CÁC HƯỚNG MỞ RỘNG ĐỀ TÀI
                                       |
    +-------------------+--------------+--------------+-------------------+
    |                   |                             |                   |
[Hướng 1]           [Hướng 2]                     [Hướng 3]           [Hướng 4]
Kỹ thuật Lấy mẫu    Biến thể Cụm &                Cost-Sensitive      Mô hình Sinh
Lai (Hybrid:        Biên Nâng cao                 & Dịch Ngưỡng       Hiện đại
SMOTE-Tomek,        (KMeans-SMOTE,                (Focal Loss,        (CTGAN, TVAE,
SMOTE-ENN)          SVM-SMOTE)                    Class-Weights)      TabDDPM)
```

### Hướng 1: Kỹ thuật Lấy mẫu Lai (Hybrid Resampling: Kết hợp Oversampling + Data Cleaning)
- **Vấn đề cần giải quyết:** SMOTE và ADASYN tạo ra hiện tượng chồng lấn lớp (Class Overlap) và sinh mẫu vào vùng nhiễu.
- **Giải pháp:**
  - **SMOTE + Tomek Links:** Sau khi tăng mẫu bằng SMOTE, tìm các cặp điểm lân cận khác lớp gần nhau nhất (Tomek Links) và xóa bỏ chúng, tạo ra khoảng cách phân tách rõ ràng (clear corridor) giữa hai lớp.
  - **SMOTE + Edited Nearest Neighbors (SMOTE-ENN):** Sau khi áp dụng SMOTE, kiểm tra từng mẫu bằng quy tắc KNN. Nếu mẫu nào có nhãn khác với phần lớn láng giềng xung quanh, nó sẽ bị loại bỏ như một điểm nhiễu.
- **Đóng góp thực nghiệm:** Đánh giá xem SMOTE-ENN có giúp cải thiện điểm PR-AUC trên bộ dữ liệu `thyroid_sick` so với SMOTE thuần túy hay không.

---

### Hướng 2: Nghiên cứu Biến thể Cấu trúc Cụm & Siêu phẳng: KMeans-SMOTE & SVM-SMOTE
- **KMeans-SMOTE:**
  - *Nguyên lý:* Phân cụm dữ liệu trước bằng K-Means. Chỉ sinh mẫu tổng hợp bên trong các cụm an toàn có tỷ lệ lớp thiểu số cao.
  - *Ý nghĩa:* Ngăn chặn triệt để hiện tượng tạo "cầu nối nhân tạo" (artificial bridges) băng qua các vùng không gian rỗng giữa các cụm.
- **SVM-SMOTE:**
  - *Nguyên lý:* Huấn luyện mô hình SVM tuyến tính hoặc RBF để xác định tập hợp Support Vectors. Chỉ tiến hành nội suy mẫu dọc theo các Support Vectors của lớp thiểu số.
  - *Ý nghĩa:* Tối ưu hóa việc làm dày đường biên cực đại hóa lề (Max-margin boundary), cực kỳ phù hợp cho các bài toán phân loại nhị phân phức tạp.

---

### Hướng 3: Học Nhạy cảm Chi phí (Cost-Sensitive Learning) & Dịch chuyển Ngưỡng Quyết định (Threshold Moving)
- **Ý tưởng:** Thay vì can thiệp làm sai lệch dữ liệu ở tầng tiền xử lý (data-level), ta can thiệp trực tiếp vào thuật toán học máy (algorithm-level):
  1. **Hiệu chỉnh hàm mất mát (Cost-Sensitive Loss):** Gán trọng số phạt $w_1 \gg w_0$ cho việc phân loại sai lớp thiểu số (ví dụ sử dụng `scale_pos_weight` trong XGBoost, `class_weight='balanced'` trong Sklearn).
  2. **Focal Loss:**
     $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
     Tự động hạ trọng số của các mẫu dễ học (lớp đa số) và tập trung gradient vào các mẫu khó học (lớp thiểu số).
  3. **Tối ưu hóa Ngưỡng quyết định (Threshold Optimization):** Giữ nguyên dữ liệu, quét các giá trị ngưỡng phân loại $T \in (0, 1)$ trên tập xác thực để cực đại hóa $F_1\text{-score}$ hoặc cực tiểu hóa tổn thất kỳ vọng theo Ma trận Chi phí (Cost Matrix).

---

### Hướng 4: Ứng dụng Mô hình Sinh Sâu Hiện Đại cho Dữ liệu Bảng (Deep Generative Models)
- **Hạn chế của SMOTE:** Chỉ có thể nội suy tuyến tính giữa 2 điểm láng giềng, hoàn toàn bất lực trong việc học phân phối xác suất đa chiều, phân phối đa đỉnh (multimodal distributions) hoặc quan hệ phi tuyến phức tạp giữa hàng chục thuộc tính.
- **Giải pháp thế hệ mới:**
  - **CTGAN (Conditional Tabular Generative Adversarial Network):** Mô hình mạng đối nghịch GAN chuyên dụng cho dữ liệu bảng, hỗ trợ trực tiếp biến số thực không chuẩn hóa và biến danh mục rời rạc.
  - **TVAE (Tabular Variational Autoencoder):** Sử dụng VAE mã hóa phân phối không gian tiềm ẩn (Latent space) để sinh mẫu thiểu số có độ đa dạng cao và thực tế.
  - **TabDDPM (Tabular Denoising Diffusion Probabilistic Models):** Ứng dụng mô hình khuếch tán xác suất (Diffusion Models) để tạo dữ liệu tổng hợp chất lượng vượt bậc cho các bài toán tài chính và y tế.

---

### Hướng 5: Xây dựng Bộ Tiêu chí Đánh giá Thực nghiệm Toàn diện Đa Chiều
Để bài báo cáo Seminar đạt chuẩn nghiên cứu khoa học chuyên sâu, nhóm cần mở rộng kiểm thử trên ít nhất 4 bộ dữ liệu chuẩn:
1. `thyroid_sick` (Y tế, tỷ lệ 15:1)
2. `mammography` (Chẩn đoán u vú, tỷ lệ 42:1)
3. `creditcard_fraud` (Gian lận thẻ tín dụng, tỷ lệ ~578:1)
4. `synthetic_moons` (Dữ liệu phi tuyến nhân tạo, tỷ lệ 10:1)

Hệ thống đánh giá trên 6 thước đo định lượng:
- **Average Precision (PR-AUC):** Độ đo chính.
- **F1-Macro / G-Mean:** Đánh giá sự cân bằng giữa Sensitivity và Specificity.
- **Brier Score:** Đo độ tin cậy của xác suất dự đoán (Calibration quality).
- **Độ lệch phân bố (Wasserstein Distance / KS-Test):** Đo khoảng cách phân phối giữa dữ liệu sinh ra và dữ liệu gốc.
- **Thời gian tính toán (Execution / Latency Time):** Đánh giá khả năng mở rộng (Scalability) trong môi trường sản xuất thực tế.
