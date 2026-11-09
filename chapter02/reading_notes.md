# CHUYÊN ĐỀ NGHIÊN CỨU & ĐỌC SÁCH
# CHAPTER 02: OVERSAMPLING METHODS (CÁC PHƯƠNG PHÁP TĂNG MẪU)
**Giáo trình:** *Machine Learning for Imbalanced Data* (Packt Publishing)  
**Môn học:** Phân tích và Nhận dạng mẫu (Pattern Recognition) - Đại học Sài Gòn (SGU)  
**Nhóm thực hiện:** Ngô Đông Ất, Nguyễn Đăng Khoa, Lê Thanh Phát  
**Giảng viên hướng dẫn:** ThS. Nguyễn Thanh Phước  

---

## MỤC LỤC TỔNG QUAN
1. [Đặt vấn đề: Mất cân bằng dữ liệu & Nhu cầu Oversampling](#1-đặt-vấn-đề-mất-cân-bằng-dữ-liệu--nhu-cầu-oversampling)
2. [Lấy mẫu ngẫu nhiên có hoàn lại (Random Oversampling - ROS)](#2-lấy-mẫu-ngẫu-nhiên-có-hoàn-lại-random-oversampling---ros)
3. [Thuật toán SMOTE (Synthetic Minority Over-sampling Technique)](#3-thuật-toán-smote-synthetic-minority-over-sampling-technique)
4. [Các biến thể cải tiến của SMOTE (SMOTE Variants)](#4-các-biến-thể-cải-tiến-của-smote-smote-variants)
   - 4.1. Borderline-SMOTE (Tập trung mẫu ranh giới nguy hiểm)
   - 4.2. ADASYN (Adaptive Synthetic Sampling - Tự thích nghi theo mật độ)
   - 4.3. Xử lý đặc trưng định danh: SMOTE-NC & SMOTEN
5. [Oversampling trong bài toán Đa lớp (Multi-class Classification)](#5-oversampling-trong-bài-toán-đa-lớp-multi-class-classification)
6. [Các ca ứng dụng thực tế trong công nghiệp (Industry Case Studies)](#6-các-ca-ứng-dụng-thực-tế-trong-công-nghiệp-industry-case-studies)
7. [Thực nghiệm So sánh Hiệu năng Mô hình (Benchmark & Analysis)](#7-thực-nghiệm-so-sánh-hiệu-năng-mô-hình-benchmark--analysis)
8. [Nguyên tắc Hướng dẫn Thực hành & Khi nào NÊN / TRÁNH Oversampling](#8-nguyên-tắc-hướng-dẫn-thực-hành--khi-nào-nên--tránh-oversampling)
9. [Bảng đối chiếu tổng hợp các phương pháp Oversampling](#9-bảng-đối-chiếu-tổng-hợp-các-phương-pháp-oversampling)
10. [Lời giải & Thảo luận chi tiết 5 Bài tập cuối chương (Exercises)](#10-lời-giải--thảo-luận-chi-tiết-5-bài-tập-cuối-chương-exercises)

---

## 1. ĐẶT VẤN ĐỀ: MẤT CÂN BẰNG DỮ LIỆU & NHU CẦU OVERSAMPLING

### 1.1. Hiện tượng Mất cân bằng dữ liệu (Class Imbalance)
Trong các bài toán học máy thực tế (như phát hiện ung thư, phát hiện giao dịch gian lận tài chính, dự báo lỗi hệ thống sự cố máy chủ), số lượng quan sát của lớp mục tiêu cần quan tâm (lớp thiểu số - *minority class*, thường gán nhãn 1) chiếm tỷ lệ rất nhỏ so với lớp thông thường (lớp đa số - *majority class*, gán nhãn 0), ví dụ tỷ lệ 1:99 hoặc 1:1000.

### 1.2. Hậu quả khi huấn luyện mô hình trên dữ liệu mất cân bằng
- **Thiên lệch ranh giới quyết định (Decision Boundary Bias):** Các hàm mất mát (loss function) chuẩn như Cross-Entropy hoặc Gini Impurity được tối ưu trên toàn bộ mẫu. Do đó, mô hình sẽ ưu tiên cực đại hóa độ chính xác tổng thể (Accuracy) bằng cách dự đoán hầu hết là lớp đa số.
- **Rủi ro bỏ sót lớp thiểu số (High False Negatives):** Mô hình có thể đạt 99% Accuracy nhưng tỷ lệ phát hiện lớp thiểu số (Recall) bằng 0. Trong y tế hay an ninh mạng, việc bỏ sót 1 ca ung thư hoặc 1 giao dịch lừa đảo nguy hiểm gấp hàng trăm lần so với báo động nhầm.

### 1.3. Khái niệm Oversampling
- **Định nghĩa:** Oversampling (Tăng mẫu) là kỹ thuật tái cân bằng dữ liệu ở mức độ tiền xử lý dữ liệu (data-level balancing), bằng cách tạo thêm các mẫu thuộc lớp thiểu số nhằm đưa tỷ lệ phân bố giữa các lớp về mức cân bằng mong muốn (thường là 1:1 hoặc tỷ lệ chỉ định).
- **So sánh với việc thu thập thêm dữ liệu:** Việc chờ đợi thu thập thêm dữ liệu thực tế thường tốn kém chi phí, thời gian và đôi khi là bất khả thi (ví dụ không thể chờ có thêm sự cố sập máy chủ để ghi log). Do đó, kỹ thuật Oversampling là giải pháp trực tiếp và khả thi nhất.

---

## 2. LẤY MẪU NGẪU NHIÊN CÓ HOÀN LẠI (RANDOM OVERSAMPLING - ROS)

### 2.1. Cơ chế hoạt động
- **Thuật toán:** Chọn ngẫu nhiên các mẫu hiện có từ lớp thiểu số và nhân bản lặp lại (duplicate with replacement) cho đến khi đạt được số lượng mong muốn.
- **API imbalanced-learn:** `RandomOverSampler(sampling_strategy=1.0, random_state=42)`
  - `sampling_strategy`: Chỉ định tỷ lệ mẫu sau khi tái cân bằng (ví dụ `1.0` để đạt tỷ lệ 1:1).

### 2.2. Nhược điểm chí mạng: Nguy cơ Quá khớp (Overfitting)
- Khi nhân bản chính xác các điểm dữ liệu cũ, mô hình học máy nhìn thấy cùng một tọa độ điểm nhiều lần, dẫn đến việc mô hình "học vẹt" (memorization) thay vì khái quát hóa (generalization) ranh giới quyết định.
- Trên đồ thị phân tán 2D, các điểm mới sinh bị đè khít lên các điểm cũ (overlap tuyệt đối), tạo cảm giác phân bố không có gì thay đổi dù số lượng mẫu đã tăng gấp hàng trăm lần.

### 2.3. Cải tiến với tham số `shrinkage`
- **Cơ chế:** Để phá vỡ tính lặp lại tuyệt đối, tham số `shrinkage` trong `RandomOverSampler` cho phép áp dụng kỹ thuật tạo nhiễu nhẹ (perturbation/jittering), làm dịch chuyển nhẹ tọa độ của các điểm nhân bản theo một phân phối Gauss nhỏ:
  $$\tilde{x} = x + \delta, \quad \delta \sim \mathcal{N}(0, \sigma^2)$$
- **Giá trị:** `shrinkage >= 0` (ví dụ `shrinkage=0.2`).
- **Hạn chế của shrinkage:** Khó chọn giá trị tối ưu; độ lệch ngẫu nhiên có thể vô tình đẩy mẫu thiểu số lấn sâu vào vùng không gian của lớp đa số.

---

## 3. THUẬT TOÁN SMOTE (SYNTHETIC MINORITY OVER-SAMPLING TECHNIQUE)

Được đề xuất bởi N. V. Chawla và cộng sự (2002), SMOTE là bước đột phá giải quyết triệt để vấn đề nhân bản trùng lặp của Random Oversampling.

### 3.1. Cơ chế Toán học: Nội suy Tuyến tính (Convex Linear Interpolation)
Thay vì sao chép mẫu, SMOTE tạo ra các quan sát tổng hợp mới nằm dọc trên đoạn thẳng nối giữa một điểm thiểu số và các điểm láng giềng gần nhất của nó:

$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}(0, 1)$$

Trong đó:
- $\mathbf{x}_i$: Mẫu lớp thiểu số đang xét.
- $\mathbf{x}_{zi}$: Một trong $k$ điểm láng giềng gần nhất (thường $k=5$) thuộc cùng lớp thiểu số.
- $\lambda$: Giá trị ngẫu nhiên trong khoảng $[0, 1]$.

```
          x_zi (Minority Neighbor)
           *
          /
         /
        o <--- x_new (Synthetic sample sinh ra ngẫu nhiên trên đoạn thẳng)
       /
      *
    x_i (Minority sample)
```

### 3.2. Thuật toán từng bước của SMOTE
1. Lọc chỉ lấy các mẫu thuộc lớp thiểu số trong tập huấn luyện.
2. Xây dựng mô hình k-Nearest Neighbors (KNN) trên tập mẫu thiểu số (mặc định $k=5$).
3. Với mỗi mẫu $\mathbf{x}_i$, xác định $k$ láng giềng gần nhất.
4. Chọn ngẫu nhiên một láng giềng $\mathbf{x}_{zi}$, tạo điểm mới nằm ngẫu nhiên trên đoạn thẳng nối giữa $\mathbf{x}_i$ và $\mathbf{x}_{zi}$.
5. Lặp lại quá trình cho đến khi đạt đủ số lượng mẫu cần tăng cường.

### 3.3. Các hạn chế & Bẫy thực tế của SMOTE (Problems with SMOTE)
1. **Chồng lấn lớp (Class Overlap):** SMOTE chỉ xét quan hệ lân cận giữa các điểm thiểu số với nhau mà hoàn toàn phớt lờ vị trí của các mẫu lớp đa số. Khi lớp thiểu số có phân bố xen kẽ hoặc nằm sát lớp đa số, việc nội suy có thể tạo ra các điểm mẫu giả nằm lọt thỏm vào vùng của lớp đa số.
2. **Khuếch đại nhiễu (Noise Amplification):** Nếu một mẫu thiểu số là điểm ngoại lai (outlier/noise), việc nội suy giữa nó và láng giềng sẽ sinh ra thêm một chuỗi các điểm ngoại lai mới, làm hỏng ranh giới phân loại.
3. **Chi phí tính toán cao:** Việc tính toán khoảng cách KNN trong không gian dữ liệu lớn hoặc nhiều chiều làm tăng đáng kể thời gian chạy pipeline huấn luyện.

---

## 4. CÁC BIẾN THỂ CẢI TIẾN CỦA SMOTE (SMOTE VARIANTS)

### 4.1. Borderline-SMOTE (Han et al., 2005)
#### Động lực cốt lõi
Các mẫu nằm sâu trong cụm thiểu số (Safe points) đã rất dễ phân loại đúng, trong khi các mẫu nằm sâu trong vùng đa số (Noise points) thường là nhiễu. Điểm then chốt quyết định chất lượng phân loại nằm ở **vùng ranh giới quyết định (Decision Boundary)**.

#### Thuật toán phân loại 3 trạng thái điểm
Borderline-SMOTE xét toàn bộ tập dữ liệu (cả lớp thiểu số và đa số) và tìm $m$ láng giềng gần nhất cho mỗi điểm thiểu số:
- **Noise points (Điểm nhiễu):** Có $m/m$ láng giềng là lớp đa số (bị bao vây hoàn toàn bởi đa số). Bỏ qua không sinh mẫu vì đây là ngoại lai.
- **Safe points (Điểm an toàn):** Số láng giềng thiểu số nhiều hơn đa số ($m_{\text{majority}} < m/2$). Bỏ qua vì đã dễ phân loại.
- **Danger points (Điểm nguy hiểm):** Số láng giềng đa số chiếm ưu thế ($m/2 \le m_{\text{majority}} < m$). Đây chính là các điểm nằm ngay trên ranh giới phân loại.

**Hành động:** Chỉ áp dụng thuật toán SMOTE để sinh dữ liệu mới từ các điểm thuộc tập **Danger points**.

### 4.2. ADASYN (Adaptive Synthetic Sampling - He et al., 2008)
#### Nguyên lý Tự thích nghi
Trong khi SMOTE sinh mẫu đồng đều trên toàn bộ lớp thiểu số, ADASYN tự động điều chỉnh số lượng mẫu cần sinh dựa trên **độ khó học (Hardness)** của từng quan sát: Mẫu nào càng khó phân loại (nằm ở vùng mật độ thấp hoặc bị lấn át bởi lớp đa số) thì càng được sinh ra nhiều mẫu tổng hợp.

#### Công thức tính toán
1. Với mỗi mẫu thiểu số $x_i$, tìm $K$ láng giềng gần nhất trên toàn bộ tập dữ liệu.
2. Tính hệ số độ khó (hardness factor):
   $$r_i = \frac{M_i}{K} \in [0, 1]$$
   *(với $M_i$ là số lượng láng giềng thuộc lớp đa số)*
3. Chuẩn hóa độ khó thành hàm mật độ xác suất:
   $$\hat{r}_i = \frac{r_i}{\sum_{j} r_j}$$
4. Số lượng mẫu tổng hợp cần sinh cho điểm $x_i$:
   $$g_i = \hat{r}_i \times G$$
   *(với $G$ là tổng số mẫu thiểu số cần sinh thêm để đạt cân bằng)*

#### Ưu & Nhược điểm của ADASYN
- **Ưu điểm:** Tự động tập trung giải quyết các vùng không gian có độ phân tách kém.
- **Nhược điểm:** Do trọng số tỷ lệ thuận với số láng giềng đa số, ADASYN rất dễ bị "đánh lừa" bởi các điểm ngoại lai (outliers), dẫn đến việc sinh ra quá nhiều mẫu giả xung quanh điểm nhiễu.

---

### 4.3. Xử lý Đặc trưng Định danh: SMOTE-NC & SMOTEN

Khoảng cách Euclid chuẩn trong SMOTE chỉ có ý nghĩa trên biến liên tục. Với biến định danh (categorical), nếu gán `['đỏ': 1, 'xanh': 0]`, nội suy có thể tạo ra giá trị `0.4` - hoàn toàn vô nghĩa.

```
       Dữ liệu Định danh (Categorical Data)
                 |
      +----------+----------+
      |                     |
Nominal (Định danh thuần)  Ordinal (Có thứ bậc: Thấp, Trung bình, Cao)
(Màu tóc, Giới tính)        --> Xử lý bằng sklearn `OrdinalEncoder`
      |
      +---> Tất cả cột là Nominal: Dùng `SMOTEN` (Value Distance Metric)
      +---> Hỗn hợp Continuous + Nominal: Dùng `SMOTE-NC`
```

#### SMOTE-NC (Nominal Continuous)
- **Điều kiện sử dụng:** Tập dữ liệu phải có **cả biến liên tục và biến định danh**.
- **Cơ chế:**
  - Đối với các đặc trưng liên tục: Dùng phép nội suy tuyến tính chuẩn của SMOTE.
  - Đối với các đặc trưng định danh: Chọn giá trị xuất hiện nhiều nhất (Majority Vote / Mode) trong số các láng giềng gần nhất.

#### SMOTEN (Synthetic Minority Over-sampling Technique for Nominal)
- **Điều kiện sử dụng:** Tập dữ liệu **chỉ toàn các biến định danh**.
- **Khoảng cách VDM (Value Distance Metric):** Thay vì khoảng cách hình học, VDM đo mức độ tương đồng giữa hai giá trị danh mục dựa trên sự tương đồng về phân phối nhãn lớp tương ứng với từng giá trị.
- **Sinh mẫu mới:** Giá trị của mọi đặc trưng trong mẫu mới được xác định thông qua biểu quyết đa số từ các láng giềng gần nhất theo khoảng cách VDM.

---

## 5. OVERSAMPLING TRONG BÀI TOÁN ĐA LỚP (MULTI-CLASS CLASSIFICATION)

Trong bài toán có nhiều hơn 2 lớp (ví dụ lớp 0: 10 mẫu, lớp 1: 40 mẫu, lớp 2: 50 mẫu), việc áp dụng Oversampling có thể điều khiển linh hoạt qua tham số `sampling_strategy`:

| Chiến lược (`str`) | Ý nghĩa & Hành vi | Ví dụ (Gốc: {0: 10, 1: 40, 2: 50}) |
|--------------------|-------------------|--------------------------------------|
| `'minority'` | Chỉ tăng mẫu cho **lớp có số lượng ít nhất** bằng số mẫu lớp lớn nhất | Tăng lớp 0 lên 50 -> Kết quả: {0: 50, 1: 40, 2: 50} |
| `'not minority'` | Tăng mẫu cho tất cả các lớp **ngoại trừ** lớp có số lượng ít nhất | Tăng lớp 1 lên 50 -> Kết quả: {0: 10, 1: 50, 2: 50} |
| `'not majority'` | Tăng mẫu cho tất cả các lớp **ngoại trừ** lớp có số lượng lớn nhất (tương đương `'auto'`) | Tăng lớp 0 và 1 lên 50 -> Kết quả: {0: 50, 1: 50, 2: 50} |
| `'all'` | Tăng mẫu cho tất cả các lớp | Tăng tất cả lên bằng lớp lớn nhất |
| `dict` tùy chỉnh | Chỉ định số lượng mẫu mục tiêu cho từng nhãn cụ thể | `{0: 40, 1: 40, 2: 50}` -> Tự do tùy biến |

> [!WARNING]
> Khi sử dụng dictionary chỉ định `sampling_strategy`, số lượng mẫu mong muốn của mỗi lớp phải **lớn hơn hoặc bằng** số mẫu ban đầu. Nếu nhỏ hơn, API sẽ ném ngoại lệ `ValueError`.

---

## 6. CÁC CA ỨNG DỤNG THỰC TẾ TRONG CÔNG NGHIỆP (INDUSTRY CASE STUDIES)

### 6.1. Grab (2021) - Phát hiện & Làm mờ Dữ liệu Nhạy cảm (PII)
- **Bối cảnh:** Grab thu thập hàng triệu ảnh đường phố để làm bản đồ giao thông. Cần nhận diện và làm mờ khuôn mặt và biển số xe để bảo vệ quyền riêng tư.
- **Thách thức:** Số lượng ảnh chứa PII (khuôn mặt/biển số) chiếm tỷ lệ rất nhỏ so với ảnh không có PII. Gán nhãn thủ công quá tốn kém.
- **Giải pháp:** Sử dụng **Random Oversampling (ROS)** để nhân bản mẫu dương tính (chứa PII). Phương pháp đơn giản này đã đem lại bước nhảy vọt hiệu năng mà không cần thuật toán sinh mẫu phức tạp.

### 6.2. Microsoft (2023) - Dự báo Leo thang Sự cố Máy chủ Trực tuyến (LSI Forecast)
- **Bối cảnh:** Dự báo các sự cố kỹ thuật thông thường có nguy cơ leo thang thành sự cố nghiêm trọng (Live Site Incidents - LSI).
- **Thách thức:** Trong 40,000 sự cố lịch sử, **chưa tới 2%** sự cố leo thang mức nghiêm trọng cao.
- **Giải pháp:** Xây dựng Pipeline cân bằng 2 giai đoạn: **SMOTE (Oversampling) kết hợp RandomUnderSampler (Undersampling)**. Mô hình tự động tối ưu hóa tỷ lệ lấy mẫu ở cả 2 bước, giúp cải thiện rõ rệt khả năng dự báo sớm.

### 6.3. Amazon (2022) - Tối ưu Đóng gói & Giảm Rác thải Bao bì
- **Bối cảnh:** Phân loại sự tương thích giữa hàng triệu sản phẩm và quy cách bao bì đóng gói.
- **Thách thức:** Chỉ **1%** dữ liệu thuộc nhóm "đóng gói không phù hợp" (minority class).
- **Kết quả so sánh các kỹ thuật:**
  - **Borderline-SMOTE:** Tăng PR-AUC từ **4% - 7%**, nhưng làm tăng thời gian huấn luyện từ 25% - 35%.
  - **Hybrid (Random Oversampling + Random Undersampling):** Tăng PR-AUC từ **6% - 10%**, tăng thời gian huấn luyện 25%.
  - **Học 2 giai đoạn (Two-phase learning with Undersampling):** Tăng PR-AUC từ **18% - 24%** mà không tăng thời gian huấn luyện.

---

## 7. THỰC NGHIỆM SO SÁNH HIỆU NĂNG MÔ HÌNH (BENCHMARK & ANALYSIS)

Trong chương này, tác giả đã tiến hành thử nghiệm chéo chuẩn hóa:
- **Bộ dữ liệu:**
  1. *Synthetic Dataset:* Dữ liệu tổng hợp 2D `make_moons` bị ép mất cân bằng (tỷ lệ 10:1).
  2. *Real-world Dataset:* Dữ liệu y tế chẩn đoán suy giáp `thyroid_sick` (3,772 mẫu, 52 đặc trưng, tỷ lệ mất cân bằng ~15:1).
- **Mô hình thử nghiệm:** Logistic Regression (mô hình tuyến tính) và Random Forest (mô hình phi tuyến dạng cây tổng hợp).
- **Quy trình đánh giá:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` đo điểm trung bình **Average Precision Score (PR-AUC)**.

### Kết quả rút ra từ thực nghiệm:
1. **Hiệu quả chung:** Hầu hết các phương pháp Oversampling đều mang lại cải thiện PR-AUC so với việc không can thiệp lấy mẫu (`Nosampling`).
2. **Độ nhạy thuật toán (Algorithm Sensitivity):** Mô hình Random Forest hưởng lợi từ các phương pháp Oversampling nhiều hơn đáng kể so với Logistic Regression trên dữ liệu tổng hợp.
3. **Hiện tượng đặc biệt của Borderline-SMOTE:** Đạt điểm Average Precision cao nhất trên dữ liệu phi tuyến tổng hợp (tăng vọt từ 0.89 lên 0.96 trên Random Forest), do tập trung tạo mẫu chính xác quanh vùng ranh giới hình vầng trăng.
4. **Độ nhạy dữ liệu thực tế (Data Sensitivity):** Trên tập dữ liệu thực tế `thyroid_sick`, tất cả các kỹ thuật lấy mẫu (ROS, SMOTE, ADASYN, Borderline-SMOTE) đều cho kết quả gần tương đương nhau (~0.62 trên Logistic Regression và ~0.78-0.79 trên Random Forest).
5. **Không có kỹ thuật "thống trị tuyệt đối":** Không có một thuật toán lấy mẫu nào luôn vượt trội trong mọi bài toán. Lựa chọn tối ưu phụ thuộc vào bản chất dữ liệu và thuật toán mô hình sử dụng.

---

## 8. NGUYÊN TẮC HƯỚNG DẪN THỰC HÀNH & KHI NÀO NÊN / TRÁNH OVERSAMPLING

### 8.1. Quy trình 5 bước thực hành chuẩn
1. **Bước 1 (Baseline):** Huấn luyện mô hình cơ sở trên dữ liệu gốc không áp dụng bất kỳ kỹ thuật lấy mẫu nào (`Nosampling`).
2. **Bước 2 (Đơn giản trước):** Thử nghiệm với `RandomOverSampler` kèm tinh chỉnh nhẹ tham số `shrinkage`.
3. **Bước 3 (Xử lý kiểu dữ liệu):**
   - Nếu có biến định danh: Mã hóa sang dạng số hoặc sử dụng `SMOTE-NC` (dữ liệu hỗn hợp) / `SMOTEN` (toàn bộ định danh).
4. **Bước 4 (Đánh giá so sánh):** Thử nghiệm lần lượt SMOTE, Borderline-SMOTE, ADASYN và so sánh qua các độ đo thích hợp: **PR-AUC (Average Precision), F1-Score, Recall**.
5. **Bước 5 (Hậu xử lý - Post-processing):**
   - **Hiệu chuẩn xác suất (Model Calibration):** Lấy mẫu làm thay đổi phân phối lớp so với thực tế, khiến xác suất dự báo bị thổi phồng. Cần hiệu chuẩn lại xác suất bằng Isotonic Regression hoặc Platt Scaling trước khi triển khai thực tế.
   - **Dịch chuyển ngưỡng quyết định (Threshold Adjustment):** Thay vì giữ ngưỡng mặc định 0.5, tối ưu hóa ngưỡng dựa trên ma trận chi phí thiệt hại thực tế.

### 8.2. Khi nào NÊN TRÁNH Oversampling?
- **Chi phí tính toán hạn chế:** Khi tập dữ liệu có hàng triệu dòng hoặc hàng ngàn chiều, Oversampling làm phình to tập dữ liệu và kéo dài thời gian huấn luyện.
- **Dữ liệu có nhiều nhiễu và ngoại lai:** Oversampling (đặc biệt là SMOTE và ADASYN) sẽ nhân rộng điểm nhiễu, làm mờ ranh giới quyết định.
- **Đã sử dụng Bộ phân loại mạnh (Strong Classifiers):** Các mô hình mạnh như Gradient Boosting (XGBoost, LightGBM, CatBoost) thường có cơ chế cân bằng trọng số nhãn (`scale_pos_weight`) hoặc điều chỉnh ngưỡng (Threshold moving) mang lại hiệu quả tương đương hoặc tốt hơn việc sinh dữ liệu giả.

---

## 9. BẢNG ĐỐI CHIẾU TỔNG HỢP CÁC PHƯƠNG PHÁP OVERSAMPLING

| Phương pháp | Cơ chế cốt lõi | Ưu điểm chính | Nhược điểm chính | Dữ liệu hỗ trợ | Khi nào nên ưu tiên chọn |
|-------------|----------------|---------------|------------------|----------------|--------------------------|
| **RandomOverSampler (ROS)** | Nhân bản ngẫu nhiên có hoàn lại các mẫu thiểu số | Cực nhanh, đơn giản, giữ nguyên phân bố thực, hỗ trợ mọi kiểu dữ liệu | Dễ gây Overfitting; khó chọn shrinkage | Mọi kiểu dữ liệu | Tập dữ liệu lớn, cần baseline nhanh, bài toán đơn giản |
| **SMOTE** | Nội suy tuyến tính KNN giữa các mẫu lớp thiểu số | Giảm False Negatives, tạo ra mẫu mới chưa từng thấy | Gây chồng lấn lớp (overlap); khuếch đại nhiễu | Số thực liên tục | Dữ liệu sạch, liên tục, mức độ mất cân bằng vừa phải |
| **Borderline-SMOTE** | Chỉ sinh mẫu nội suy từ các điểm nguy hiểm (*Danger points*) ở ranh giới | Tăng cường độ phân tách ranh giới; loại trừ điểm nhiễu (*Noise*) | Bỏ qua cấu trúc cụm bên trong; tăng thời gian huấn luyện | Số thực liên tục | Ranh giới giữa 2 lớp phức tạp, phi tuyến |
| **ADASYN** | Sinh mẫu tỷ lệ thuận theo hệ số độ khó học (mật độ lớp đa số lân cận) | Tập trung vào các vùng khó học và vùng mật độ thưa | Cực kỳ nhạy cảm với ngoại lai; có thể sinh mẫu sai lệch | Số thực liên tục | Dữ liệu có nhiều vùng mật độ không đồng đều |
| **SMOTE-NC** | Nội suy SMOTE cho biến số + Biểu quyết đa số cho biến danh mục | Xử lý trực tiếp dữ liệu dạng bảng thực tế (hỗn hợp) | Đòi hỏi phải có cả 2 kiểu biến số và danh mục | Hỗn hợp (Continuous + Nominal) | Bảng dữ liệu nghiệp vụ có cả tuổi, lương và nghề nghiệp |
| **SMOTEN** | Khoảng cách VDM (Value Distance Metric) + Biểu quyết đa số | Hoạt động trên dữ liệu hoàn toàn là nhãn danh mục | Không áp dụng cho biến số liên tục | Toàn bộ là Nominal Categorical | Dữ liệu khảo sát, khảo sát y tế toàn câu hỏi trắc nghiệm |

---

## 10. LỜI GIẢI & THẢO LUẬN CHI TIẾT 5 BÀI TẬP CUỐI CHƯƠNG (EXERCISES)

### Bài tập 1: Khám phá KMeans-SMOTE và SVM-SMOTE
- **KMeans-SMOTE:**
  - *Cơ chế:* Trước khi sinh mẫu, thuật toán áp dụng K-Means clustering trên không gian đặc trưng để nhóm dữ liệu thành các cụm. Sau đó, nó lọc ra các cụm có tỷ lệ lớp thiểu số cao và chỉ thực hiện SMOTE bên trong các cụm này.
  - *Ý nghĩa:* Khắc phục hoàn toàn hiện tượng tạo "cầu nối giả" (artificial bridges) giữa các cụm thiểu số tách biệt, ngăn chặn tuyệt đối việc sinh mẫu đè lên vùng trống giữa 2 cụm.
- **SVM-SMOTE:**
  - *Cơ chế:* Sử dụng mô hình Support Vector Machine (SVM) để xác định các Support Vectors ranh giới. Thuật toán chỉ sinh mẫu dọc theo các Support Vectors của lớp thiểu số.
  - *Ý nghĩa:* Tập trung chính xác vào ranh giới cực đại hóa lề (maximum margin boundary).

### Bài tập 2: Tỷ lệ mất cân bằng 1:20 và Quy tắc Resampling
- **Cách cân bằng:** Có thể sử dụng SMOTE hoặc Borderline-SMOTE để nâng tỷ lệ lên 1:1 hoặc 1:2 tùy bài toán.
- **Có áp dụng Resampling tại thời điểm kiểm thử / đánh giá (Test/Evaluation time) không?**
  - **TUYỆT ĐỐI KHÔNG!**
  - *Lý do:* Tập kiểm thử (Test set) phải phản ánh chính xác phân bố của thế giới thực. Nếu ta resample tập test, ta đang đánh giá mô hình trên một thế giới giả tạo đã bị bóp méo phân bố xác suất tiên nghiệm $P(Y)$, dẫn đến kết quả đánh giá không còn giá trị thực tế. Mọi thao tác lấy mẫu **CHỈ ĐƯỢC PHÉP ÁP DỤNG TRÊN TẬP HUẤN LUYỆN (TRAINING SET)**.

### Bài tập 3: Cạm bẫy khi nhân bản dữ liệu rồi mới chia Train / Test / Validation
- **Tình huống:** Có 5,000 hồ sơ, 500 hồ sơ duyệt vay (tỷ lệ 1:9). Người phát triển nhân bản 500 hồ sơ này cho cân bằng, rồi mới dùng `train_test_split` để chia train/test/val.
- **Vấn đề chí mạng: RÒ RỈ DỮ LIỆU (DATA LEAKAGE)!**
  - Khi nhân bản mẫu trước khi phân chia, các bản sao y hệt của cùng một quan sát sẽ đồng thời xuất hiện ở cả tập Train và tập Test/Val.
  - Mô hình chỉ cần "học vẹt" bản ghi ở tập Train là có thể dự đoán chính xác 100% bản ghi trùng lặp ở tập Test.
  - Dẫn đến hiện tượng: Điểm kiểm thử trên máy cực cao (ảo tưởng thành công), nhưng khi triển khai trên khách hàng thực tế thì mô hình thất bại hoàn toàn.
  - **Quy tắc vàng:** Luôn luôn chia `Train/Test` trước, sau đó chỉ áp dụng Oversampling bên trong tập Train!

### Bài tập 4: Chuẩn hóa dữ liệu (Data Normalization) có giải quyết được mất cân bằng dữ liệu không?
- **Khẳng định:** **SAI (False).** Chuẩn hóa dữ liệu không giải quyết được vấn đề mất cân bằng dữ liệu.
- **Giải thích:**
  - Chuẩn hóa dữ liệu (như Min-Max Scaling hoặc Standard Scaling $Z = \frac{X - \mu}{\sigma}$) chỉ thay đổi thang đo (scale) của các trục đặc trưng về cùng khoảng giá trị (ví dụ $[0, 1]$ hoặc mean=0, std=1), giúp các thuật toán tính khoảng cách (như KNN, SVM) hoặc Gradient Descent hội tụ tốt hơn.
  - Chuẩn hóa hoàn toàn **không làm thay đổi số lượng mẫu hay phân bố xác suất tiên nghiệm giữa các lớp** ($P(Y=1)$ và $P(Y=0)$ vẫn giữ nguyên tỷ lệ 1:99). Do đó, sự thiên lệch của hàm mất mát về phía lớp đa số vẫn tồn tại nguyên vẹn.

### Bài tập 5: Các tham số cốt lõi của API `imbalanced-learn`
- `sampling_strategy`: Điều khiển tỷ lệ số lượng mẫu mong muốn sau khi resample (dạng float, string hoặc dictionary).
- `random_state`: Khởi tạo seed ngẫu nhiên nhằm đảm bảo tính tái lập (reproducibility) của các thí nghiệm khoa học.
- `k_neighbors`: Số lượng láng giềng gần nhất dùng để nội suy trong SMOTE (mặc định là 5).
- `m_neighbors`: Số lượng láng giềng dùng để phân loại Safe/Danger/Noise trong Borderline-SMOTE.
- `shrinkage`: Hệ số co giãn độ nhiễu Gauss trong RandomOverSampler.
- `n_jobs`: Số luồng CPU chạy song song để tăng tốc tìm kiếm KNN.
