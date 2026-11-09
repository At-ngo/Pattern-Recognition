"""
Replication Script for Chapter 02: Oversampling Methods
Book: Machine Learning for Imbalanced Data (Packt Publishing)
Course: Pattern Recognition - Saigon University (SGU)
Authors: Ngo Dong At, Nguyen Dang Khoa, Le Thanh Phat
Instructor: Nguyen Thanh Phuoc
"""

import os
import sys
import time

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    ADASYN,
    SMOTENC,
    SMOTEN,
)

# Set global styles
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_and_save_scatter(X, y, title, filename, xlim=(-3, 3), ylim=(-5, 4)):
    """Helper to plot 2D scatter and save to results/"""
    plt.figure(figsize=(7, 6))
    X_df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    y_ser = pd.Series(y)

    markers = {0: "v", 1: "X"}
    palette = {0: "#1f77b4", 1: "#ff7f0e"}
    
    sns.scatterplot(
        data=X_df,
        x="feature_1",
        y="feature_2",
        hue=y_ser,
        style=y_ser,
        markers=markers,
        palette=palette,
        alpha=0.8,
        s=40,
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("feature_1", fontsize=11)
    plt.ylabel("feature_2", fontsize=11)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend(title="Class", labels=["0 (Majority)", "1 (Minority)"], loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f" [Saved Plot] {filename} -> {out_path}")


def run_visual_experiments():
    """Tái lập toàn bộ các trực quan hóa hình học trong Chapter 02"""
    print("\n" + "=" * 70)
    print("PHẦN 1: TÁI LẬP TRỰC QUAN HÓA CÁC THUẬT TOÁN OVERSAMPLING (2D PLOTS)")
    print("=" * 70)

    # 1. Tạo tập dữ liệu gốc 1:99
    X, y = make_classification(
        n_samples=10000,
        n_features=2,
        n_redundant=0,
        n_classes=2,
        flip_y=0,
        n_clusters_per_class=2,
        class_sep=0.79,
        weights=[0.99],
        random_state=81,
    )
    print(f"1. Dữ liệu gốc 1:99: {Counter(y)}")
    plot_and_save_scatter(
        X, y,
        "Figure 2.2: Dataset with imbalance ratio: 1:99",
        "fig2_2_synthetic_1_99.png"
    )

    # 2. RandomOverSampler thuần túy
    ros_vanilla = RandomOverSampler(sampling_strategy=1.0, random_state=42)
    X_ros, y_ros = ros_vanilla.fit_resample(X, y)
    print(f"2. RandomOverSampler (thuần túy): {Counter(y_ros)}")
    plot_and_save_scatter(
        X_ros, y_ros,
        "Figure 2.3: RandomOverSampler (Exact duplicate overlap)",
        "fig2_3_ros_overlap.png"
    )

    # 3. RandomOverSampler với shrinkage=0.2
    ros_shrink = RandomOverSampler(sampling_strategy=1.0, random_state=42, shrinkage=0.2)
    X_shrink, y_shrink = ros_shrink.fit_resample(X, y)
    print(f"3. RandomOverSampler (shrinkage=0.2): {Counter(y_shrink)}")
    plot_and_save_scatter(
        X_shrink, y_shrink,
        "Figure 2.4: RandomOverSampler with shrinkage=0.2",
        "fig2_4_ros_shrinkage.png"
    )

    # 4. SMOTE
    smote = SMOTE(random_state=0)
    X_smote, y_smote = smote.fit_resample(X, y)
    print(f"4. SMOTE: {Counter(y_smote)}")
    plot_and_save_scatter(
        X_smote, y_smote,
        "Figure 2.6: Dataset oversampled using SMOTE",
        "fig2_6_smote.png"
    )

    # 5. Borderline-SMOTE
    bsmote = BorderlineSMOTE(random_state=0)
    X_bsmote, y_bsmote = bsmote.fit_resample(X, y)
    print(f"5. Borderline-SMOTE: {Counter(y_bsmote)}")
    plot_and_save_scatter(
        X_bsmote, y_bsmote,
        "Figure 2.10: Dataset oversampled using Borderline-SMOTE",
        "fig2_10_borderline_smote.png"
    )

    # 6. ADASYN
    adasyn = ADASYN(random_state=0)
    X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
    print(f"6. ADASYN: {Counter(y_adasyn)}")
    plot_and_save_scatter(
        X_adasyn, y_adasyn,
        "Figure 2.12: Dataset oversampled using ADASYN",
        "fig2_12_adasyn.png"
    )

    # 7. Hiện tượng SMOTE Overlap (từ smote-plots.ipynb)
    X_ov, y_ov = make_classification(
        n_samples=5000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=[0.022, 0.967],
        class_sep=0.9,
        random_state=0,
    )
    X_ov_res, y_ov_res = SMOTE(random_state=0).fit_resample(X_ov, y_ov)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_ov[y_ov == 0, 0], X_ov[y_ov == 0, 1], c="#ff7f0e", label="Minority", s=25, alpha=0.7)
    plt.scatter(X_ov[y_ov == 1, 0], X_ov[y_ov == 1, 1], c="#1f77b4", label="Majority", s=15, alpha=0.3)
    plt.title("Initial Data (Separated)", fontsize=11, fontweight="bold")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_ov_res[y_ov_res == 0, 0], X_ov_res[y_ov_res == 0, 1], c="#ff7f0e", label="Minority (Resampled)", s=25, alpha=0.7)
    plt.scatter(X_ov_res[y_ov_res == 1, 0], X_ov_res[y_ov_res == 1, 1], c="#1f77b4", label="Majority", s=15, alpha=0.3)
    plt.title("After SMOTE (Notice Class Overlap)", fontsize=11, fontweight="bold")
    plt.legend()
    plt.suptitle("Figure 2.7: Class Overlap caused by SMOTE interpolation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_ov = os.path.join(RESULTS_DIR, "fig2_7_smote_overlap.png")
    plt.savefig(out_ov, dpi=300)
    plt.close()
    print(f" [Saved Plot] fig2_7_smote_overlap.png -> {out_ov}")


def get_datasets():
    """Chuẩn bị 2 bộ dữ liệu: Synthetic Moons và Thyroid Sick"""
    datasets = []

    # 1. Synthetic Moons (10:1 imbalance)
    X_moons, y_moons = make_moons(n_samples=1100, noise=0.1, random_state=14)
    target_stats = Counter(y_moons)
    sampling_dict = {1: int(0.1 * target_stats[1])}
    X_moons_imb, y_moons_imb = make_imbalance(
        X_moons,
        y_moons,
        sampling_strategy=sampling_dict,
        random_state=0
    )
    datasets.append({
        "name": "synthetic_moons",
        "descr": "synthetic data",
        "data": np.array(X_moons_imb, dtype=float),
        "target": np.array(y_moons_imb, dtype=int)
    })

    # 2. Thyroid Sick (Load from local cache or OpenML)
    cache_path = os.path.join(os.path.dirname(__file__), "thyroid_sick.npz")
    if os.path.exists(cache_path):
        data_npz = np.load(cache_path)
        X_thyroid = data_npz["data"]
        y_thyroid = data_npz["target"]
    else:
        from sklearn.datasets import fetch_openml
        from sklearn.impute import SimpleImputer
        print(" Tải dữ liệu 'sick' từ OpenML...")
        d = fetch_openml("sick", version=1, as_frame=True)
        X_df = d.data.drop(columns=["TBG"], errors="ignore")
        num_cols = X_df.select_dtypes(include="number").columns
        cat_cols = X_df.select_dtypes(exclude="number").columns
        X_df[num_cols] = SimpleImputer(strategy="median").fit_transform(X_df[num_cols])
        X_df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X_df[cat_cols])
        X_thyroid = pd.get_dummies(X_df, drop_first=True).astype(float).values
        y_thyroid = (d.target == "sick").astype(int).values
        np.savez(cache_path, data=X_thyroid, target=y_thyroid)

    datasets.append({
        "name": "thyroid_sick",
        "descr": "thyroid_sick",
        "data": X_thyroid,
        "target": y_thyroid
    })

    return datasets


def run_benchmark():
    """Tái lập thực nghiệm so sánh mô hình (Repeated Stratified 5-Fold Cross Validation x 3 repeats)"""
    print("\n" + "=" * 70)
    print("PHẦN 2: THỰC NGHIỆM ĐÁNH GIÁ VÀ SO SÁNH HIỆU NĂNG MÔ HÌNH (BENCHMARK)")
    print("=" * 70)

    datasets = get_datasets()
    for d in datasets:
        print(f" Dataset: {d['descr']} | Shape: {d['data'].shape} | Target: {Counter(d['target'])}")

    oversamplers = {
        "Nosampling": None,
        "RandomOverSampler": RandomOverSampler(sampling_strategy="auto", random_state=0),
        "SMOTE": SMOTE(random_state=0),
        "ADASYN": ADASYN(sampling_strategy="auto", random_state=0, n_neighbors=5),
        "Borderline-SMOTE": BorderlineSMOTE(
            sampling_strategy="auto", random_state=0, k_neighbors=5, m_neighbors=10, kind="borderline-1"
        ),
    }

    results = []

    for d in datasets:
        d_name = d["descr"]
        X, y = d["data"], d["target"]
        print(f"\n>>> Đang chạy thử nghiệm trên: {d_name} ...")

        for os_name, oversampler in oversamplers.items():
            t0 = time.time()
            models = {
                "Logistic Regression": LogisticRegression(random_state=0, max_iter=2000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=39, max_depth=3, n_jobs=4),
                "XGBoost": XGBClassifier(random_state=0, eval_metric="logloss")
            }

            for model_name, clf in models.items():
                pipe_steps = [("scaler", MinMaxScaler())]
                if oversampler is not None:
                    pipe_steps.append(("over", oversampler))
                pipe_steps.append(("model", clf))
                pipeline = Pipeline(steps=pipe_steps)

                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=4)
                cv_scores = cross_validate(
                    pipeline, X, y, scoring="average_precision", cv=cv, n_jobs=-1
                )
                ap_mean = cv_scores["test_score"].mean()
                ap_std = cv_scores["test_score"].std()

                results.append({
                    "Dataset": d_name,
                    "Oversampler": os_name,
                    "Classifier": model_name,
                    "Average_Precision_Mean": round(float(ap_mean), 3),
                    "Average_Precision_Std": round(float(ap_std), 3)
                })

            dt = time.time() - t0
            print(f"  + Hoàn thành {os_name:18s} trên {d_name} trong {dt:.1f}s")

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\n [Saved CSV] Kết quả lưu tại: {csv_path}")

    # Tạo bảng pivot cho từng Dataset
    summary_md_lines = ["# BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK CHAPTER 02\n\n"]
    for d_name in ["synthetic data", "thyroid_sick"]:
        sub_df = res_df[res_df["Dataset"] == d_name]
        pivot_df = sub_df.pivot(index="Oversampler", columns="Classifier", values="Average_Precision_Mean")
        # Sắp xếp đúng thứ tự
        pivot_df = pivot_df.reindex(["Nosampling", "RandomOverSampler", "SMOTE", "ADASYN", "Borderline-SMOTE"])
        print(f"\n--- BẢNG HIỆU NĂNG TRUNG BÌNH (AVERAGE PRECISION) TRÊN {d_name.upper()} ---")
        print(pivot_df.to_string())

        summary_md_lines.append(f"### Dữ liệu: `{d_name}`\n")
        summary_md_lines.append(pivot_df.to_markdown())
        summary_md_lines.append("\n\n")

    summary_md_path = os.path.join(RESULTS_DIR, "benchmark_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.writelines(summary_md_lines)
    print(f" [Saved Markdown] Bảng tổng kết lưu tại: {summary_md_path}")

    # Vẽ biểu đồ so sánh cột nằm ngang (Horizontal Bar Plots) đối chiếu với Hình 2.15 & 2.16
    plot_benchmark_bars(res_df)


def plot_benchmark_bars(df):
    """Vẽ biểu đồ thanh ngang so sánh giống đồ thị sách giáo trình (Hình 2.15 và 2.16)"""
    palette_samplers = ["Nosampling", "RandomOverSampler", "SMOTE", "ADASYN", "Borderline-SMOTE"]

    for d_name in ["synthetic data", "thyroid_sick"]:
        sub_df = df[df["Dataset"] == d_name]
        classifiers = ["Logistic Regression", "Random Forest", "XGBoost"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        fig.suptitle(f"Performance Comparison of Oversampling Methods on '{d_name}'\n(Metric: Average Precision - PR-AUC)", fontsize=14, fontweight="bold")

        for idx, clf_name in enumerate(classifiers):
            ax = axes[idx]
            clf_data = sub_df[sub_df["Classifier"] == clf_name].set_index("Oversampler").reindex(palette_samplers)
            baseline_val = clf_data.loc["Nosampling", "Average_Precision_Mean"]

            colors = ["#17becf" if val <= baseline_val else "#d62728" for val in clf_data["Average_Precision_Mean"]]
            # Điểm Nosampling màu xám / cyan
            colors[0] = "#7f7f7f"

            y_pos = np.arange(len(palette_samplers))
            bars = ax.barh(y_pos, clf_data["Average_Precision_Mean"], color=colors, height=0.6)
            ax.axvline(baseline_val, color="#d62728", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_val:.2f})")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(palette_samplers, fontsize=11)
            ax.set_xlabel("Average Precision (PR-AUC)", fontsize=11)
            ax.set_title(clf_name, fontsize=12, fontweight="bold")
            ax.set_xlim(0, 1.05)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10, fontweight="bold")
            ax.legend(loc="lower right")

        plt.tight_layout()
        safe_name = d_name.replace(" ", "_")
        plot_path = os.path.join(RESULTS_DIR, f"benchmark_perf_{safe_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f" [Saved Plot] benchmark_perf_{safe_name}.png -> {plot_path}")


def run_categorical_and_multiclass_experiments():
    """Tái lập Mục 2.6 (Categorical: ROS, SMOTE-NC, SMOTEN) và Mục 2.7 (Multi-class sampling strategies)"""
    print("\n" + "=" * 70)
    print("PHẦN 1B: THỰC NGHIỆM DỮ LIỆU ĐỊNH DANH (2.6) & ĐA LỚP (2.7)")
    print("=" * 70)

    md_lines = [
        "# KẾT QUẢ THỰC NGHIỆM MỤC 2.6 (ĐỊNH DANH) & MỤC 2.7 (ĐA LỚP)\n\n",
        "## 1. Mục 2.6: Xử lý Đặc trưng Định danh (Categorical Data)\n\n",
    ]

    # 2.6.1 RandomOverSampler trên dữ liệu hỗn hợp
    X_cat_mix_ros = np.array([["abc", 1], ["def", 2], ["ghi", 3]], dtype=object)
    y_cat_mix_ros = np.array([0, 0, 1])
    X_ros_cat, y_ros_cat = RandomOverSampler(random_state=0).fit_resample(X_cat_mix_ros, y_cat_mix_ros)
    print(f" [2.6.1] ROS Mixed Data -> Trước: {Counter(y_cat_mix_ros)} | Sau: {Counter(y_ros_cat)}")
    print(f"         Mẫu sinh ra:\n{X_ros_cat}")
    md_lines.append("### 1.1. RandomOverSampler trên dữ liệu hỗn hợp\n")
    md_lines.append(f"- **Trước:** `{X_cat_mix_ros.tolist()}`, nhãn `{y_cat_mix_ros.tolist()}`\n")
    md_lines.append(f"- **Sau:** `{X_ros_cat.tolist()}`, nhãn `{y_ros_cat.tolist()}`\n")
    md_lines.append("- *Nhận xét:* ROS sao chép nguyên bản ghi `['ghi', 3]` nên hoạt động tự nhiên với biến chuỗi.\n\n")

    # 2.6.2 SMOTE-NC (Nominal + Continuous)
    X_cat_mix = np.array(
        [["small", 1], ["medium", 2], ["large", 3], ["large", 4], ["large", 5]],
        dtype=object,
    )
    y_cat_mix = np.array([0, 0, 1, 0, 1])
    X_smotenc, y_smotenc = SMOTENC(
        categorical_features=[0], k_neighbors=1, random_state=1
    ).fit_resample(X_cat_mix, y_cat_mix)
    print(f" [2.6.2] SMOTE-NC -> Trước: {Counter(y_cat_mix)} | Sau: {Counter(y_smotenc)}")
    print(f"         Mẫu sinh thêm: {X_smotenc[-1]}")
    md_lines.append("### 1.2. SMOTE-NC (Nominal + Continuous)\n")
    md_lines.append(f"- **Trước:** `{X_cat_mix.tolist()}`, nhãn `{y_cat_mix.tolist()}`\n")
    md_lines.append(f"- **Sau:** `{X_smotenc.tolist()}`, nhãn `{y_smotenc.tolist()}`\n")
    md_lines.append(f"- *Mẫu tổng hợp mới:* `{X_smotenc[-1].tolist()}` -> Cột số nội suy liên tục (`~3.006`), cột chuỗi gán theo majority vote (`'large'`).\n\n")

    # 2.6.3 SMOTEN (Pure Nominal with VDM)
    X_original = np.array([["abc"], ["def"], ["ghi"], ["ghi"], ["ghi"]], dtype=object)
    y_original = np.array([0, 0, 1, 1, 1])
    X_smoten, y_smoten = SMOTEN(k_neighbors=1, random_state=0).fit_resample(X_original, y_original)
    print(f" [2.6.3] SMOTEN -> Trước: {Counter(y_original)} | Sau: {Counter(y_smoten)}")
    print(f"         Mẫu sau resample: {X_smoten.ravel().tolist()}")
    md_lines.append("### 1.3. SMOTEN (100% Nominal với Value Distance Metric - VDM)\n")
    md_lines.append(f"- **Trước:** `{X_original.ravel().tolist()}`, nhãn `{y_original.tolist()}`\n")
    md_lines.append(f"- **Sau:** `{X_smoten.ravel().tolist()}`, nhãn `{y_smoten.tolist()}`\n\n")

    # 2.6.4 OrdinalEncoder demo
    ord_data = np.array([["low"], ["medium"], ["high"], ["high"]])
    ord_enc = OrdinalEncoder(categories=[["low", "medium", "high"]])
    ord_encoded = ord_enc.fit_transform(ord_data)
    md_lines.append("### 1.4. OrdinalEncoder cho dữ liệu thứ bậc\n")
    md_lines.append(f"- Đầu vào `['low', 'medium', 'high', 'high']` -> Mã hóa `{ord_encoded.ravel().tolist()}`\n\n")

    # 2.7 Multi-class Oversampling strategies
    X_mc, y_mc = make_classification(
        n_classes=3,
        class_sep=2,
        weights=[0.1, 0.4, 0.5],
        n_clusters_per_class=1,
        n_samples=100,
        random_state=10,
    )
    orig_counts = dict(sorted(Counter(y_mc).items()))
    print(f" [2.7] Multi-class Original: {orig_counts}")

    strategies = {
        "Original": orig_counts,
        "minority": dict(sorted(Counter(SMOTE(sampling_strategy="minority", random_state=0).fit_resample(X_mc, y_mc)[1]).items())),
        "not minority": dict(sorted(Counter(SMOTE(sampling_strategy="not minority", random_state=0).fit_resample(X_mc, y_mc)[1]).items())),
        "auto / not majority": dict(sorted(Counter(SMOTE(sampling_strategy="auto", random_state=0).fit_resample(X_mc, y_mc)[1]).items())),
        "Custom dict {0:40,1:40,2:50}": dict(sorted(Counter(SMOTE(sampling_strategy={0: 40, 1: 40, 2: 50}, random_state=0).fit_resample(X_mc, y_mc)[1]).items())),
    }

    md_lines.append("## 2. Mục 2.7: Oversampling trong Bài toán Đa lớp (Multi-class)\n\n")
    md_lines.append("| Chiến lược (`sampling_strategy`) | Lớp 0 (Thiểu số) | Lớp 1 (Trung gian) | Lớp 2 (Đa số) | Tổng mẫu |\n")
    md_lines.append("|:---|:---:|:---:|:---:|:---:|\n")

    plot_rows = []
    for strat_name, counts in strategies.items():
        c0, c1, c2 = counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)
        print(f"       Strategy '{strat_name:28s}' -> {counts}")
        md_lines.append(f"| `{strat_name}` | {c0} | {c1} | {c2} | {c0+c1+c2} |\n")
        plot_rows.append({"Strategy": strat_name, "Class 0": c0, "Class 1": c1, "Class 2": c2})

    # Vẽ biểu đồ Multi-class sampling strategies
    df_mc = pd.DataFrame(plot_rows).set_index("Strategy")
    ax = df_mc.plot(kind="bar", figsize=(10, 5), color=["#ff7f0e", "#2ca02c", "#1f77b4"], width=0.72)
    plt.title("Multi-class SMOTE Resampling Strategies (Original Ratio 10:40:50)", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Samples", fontsize=11)
    plt.xlabel("Sampling Strategy", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 62)
    for container in ax.containers:
        ax.bar_label(container, fontsize=9, fontweight="bold", padding=2)
    plt.legend(title="Class")
    plt.tight_layout()
    mc_plot_path = os.path.join(RESULTS_DIR, "fig_multiclass_strategies.png")
    plt.savefig(mc_plot_path, dpi=300)
    plt.close()
    print(f" [Saved Plot] fig_multiclass_strategies.png -> {mc_plot_path}")

    summary_cat_path = os.path.join(RESULTS_DIR, "categorical_multiclass_summary.md")
    with open(summary_cat_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f" [Saved Markdown] {summary_cat_path}")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print(" KHỞI ĐỘNG CHƯƠNG TRÌNH TÁI LẬP THỰC NGHIỆM CHAPTER 02")
    print("#" * 70)
    t_start = time.time()
    
    # 1. Chạy trực quan hóa (Mục 2.1 - 2.5)
    run_visual_experiments()

    # 1B. Chạy thực nghiệm Định danh & Đa lớp (Mục 2.6 & 2.7)
    run_categorical_and_multiclass_experiments()
    
    # 2. Chạy benchmark mô hình (Mục 3)
    run_benchmark()
    
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f" HOÀN TẤT TOÀN BỘ THỰC NGHIỆM CHAPTER 02 TRONG {total_time:.1f} GIÂY!")
    print(f" Toàn bộ biểu đồ và bảng kết quả đã được lưu tại: {RESULTS_DIR}")
    print("=" * 70)
