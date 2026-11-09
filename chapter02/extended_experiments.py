"""
Extended & Novel Experiments for Chapter 02 Seminar (Điểm mới & Mở rộng của nhóm)
Covers:
1. Exercise 1: KMeans-SMOTE & SVM-SMOTE
2. Hybrid Sampling: SMOTE-Tomek & SMOTE-ENN
3. Algorithm-level Alternative: Cost-Sensitive Learning (class_weight / scale_pos_weight)
4. Multi-metric Evaluation: PR-AUC vs ROC-AUC, Brier Score (Probability Distortion), Training Time
5. Empirical Proof of Pitfalls: Data Leakage (Exercise 3) & Probability Calibration (Page 56)
"""

import os
import sys
import time
from collections import Counter

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_datasets():
    """Load the exact same Synthetic Moons & Thyroid Sick datasets"""
    X_moons, y_moons = make_moons(n_samples=1100, noise=0.1, random_state=14)
    target_stats = Counter(y_moons)
    X_moons_imb, y_moons_imb = make_imbalance(
        X_moons, y_moons, sampling_strategy={1: int(0.1 * target_stats[1])}, random_state=0
    )

    cache_path = os.path.join(os.path.dirname(__file__), "thyroid_sick.npz")
    data_npz = np.load(cache_path)
    X_thyroid, y_thyroid = data_npz["data"], data_npz["target"]

    return [
        {"descr": "synthetic data", "data": np.array(X_moons_imb, dtype=float), "target": np.array(y_moons_imb, dtype=int)},
        {"descr": "thyroid_sick", "data": X_thyroid, "target": y_thyroid},
    ]


def run_extended_benchmark():
    """So sánh thuật toán sách vs KMeans-SMOTE, SVM-SMOTE, Hybrid (SMOTE-Tomek, SMOTE-ENN) & Cost-Sensitive"""
    print("\n" + "=" * 75)
    print("MỞ RỘNG 1: SO SÁNH KMEANS-SMOTE, SVM-SMOTE, HYBRID & COST-SENSITIVE")
    print("=" * 75)

    datasets = load_datasets()
    rows = []

    for d in datasets:
        d_name = d["descr"]
        X, y = d["data"], d["target"]
        pos_ratio = float(np.sum(y == 0)) / float(np.sum(y == 1))
        print(f"\n>>> Dataset: {d_name} | Ratio neg/pos = {pos_ratio:.1f}:1")

        # Dùng KMeans ổn định để tránh fold lỗi cụm trống
        n_clust = 5 if d_name == "synthetic data" else 10
        methods = {
            "Nosampling": (None, False),
            "Borderline-SMOTE": (BorderlineSMOTE(random_state=0, kind="borderline-1"), False),
            "KMeans-SMOTE": (
                KMeansSMOTE(
                    kmeans_estimator=KMeans(n_clusters=n_clust, random_state=0, n_init="auto"),
                    cluster_balance_threshold=0.01,
                    random_state=0,
                ),
                False,
            ),
            "SVM-SMOTE": (SVMSMOTE(random_state=0), False),
            "SMOTE-Tomek": (SMOTETomek(random_state=0), False),
            "SMOTE-ENN": (SMOTEENN(random_state=0), False),
            "Cost-Sensitive": (None, True),
        }

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=4)

        for method_name, (sampler, use_cost) in methods.items():
            t0 = time.time()
            cw = "balanced" if use_cost else None
            spw = pos_ratio if use_cost else 1.0

            models = {
                "Logistic Regression": LogisticRegression(random_state=0, max_iter=2000, class_weight=cw),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=39, max_depth=3, n_jobs=4, class_weight=cw
                ),
                "XGBoost": XGBClassifier(random_state=0, eval_metric="logloss", scale_pos_weight=spw),
            }

            for clf_name, clf in models.items():
                steps = [("scaler", MinMaxScaler())]
                if sampler is not None:
                    steps.append(("sampler", sampler))
                steps.append(("model", clf))
                pipe = Pipeline(steps=steps)

                cv_res = cross_validate(
                    pipe,
                    X,
                    y,
                    scoring={
                        "pr_auc": "average_precision",
                        "roc_auc": "roc_auc",
                        "brier": "neg_brier_score",
                    },
                    cv=cv,
                    n_jobs=-1,
                )

                rows.append({
                    "Dataset": d_name,
                    "Method": method_name,
                    "Classifier": clf_name,
                    "PR_AUC": round(float(cv_res["test_pr_auc"].mean()), 3),
                    "ROC_AUC": round(float(cv_res["test_roc_auc"].mean()), 3),
                    "Brier_Score": round(float(-cv_res["test_brier"].mean()), 4),
                    "Fit_Time_ms": round(float(cv_res["fit_time"].mean() * 1000), 1),
                })

            print(f"  + Hoàn thành {method_name:18s} trong {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "extended_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n [Saved CSV] {csv_path}")

    # Plot Extended Comparison (PR-AUC across 7 methods)
    order = [
        "Nosampling",
        "Borderline-SMOTE",
        "KMeans-SMOTE",
        "SVM-SMOTE",
        "SMOTE-Tomek",
        "SMOTE-ENN",
        "Cost-Sensitive",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.suptitle(
        "Extended Benchmark: Chapter 02 Baseline vs Exercise 1, Hybrid & Cost-Sensitive (PR-AUC)",
        fontsize=13,
        fontweight="bold",
    )

    for idx, d_name in enumerate(["synthetic data", "thyroid_sick"]):
        ax = axes[idx]
        sub = df[df["Dataset"] == d_name]
        piv = sub.pivot(index="Method", columns="Classifier", values="PR_AUC").reindex(order)
        piv[["Logistic Regression", "Random Forest", "XGBoost"]].plot(
            kind="barh", ax=ax, width=0.75, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )
        ax.set_title(f"Dataset: {d_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Average Precision (PR-AUC)", fontsize=11)
        ax.set_xlim(0.4, 1.05)
        ax.legend(loc="lower right")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "extended_benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" [Saved Plot] {plot_path}")
    return df


def run_pitfalls_experiment():
    """
    Thực chứng 2 cạm bẫy lớn trong Chapter 02:
    A) Data Leakage (Bài tập 3): SMOTE trước split vs SMOTE sau split
    B) ROC-AUC illusion vs PR-AUC & Probability Distortion + Calibration (Trang 56)
    """
    print("\n" + "=" * 75)
    print("MỞ RỘNG 2: THỰC CHỨNG DATA LEAKAGE (BÀI TẬP 3) & HIỆU CHUẨN XÁC SUẤT (TR.56)")
    print("=" * 75)

    datasets = load_datasets()
    X, y = datasets[1]["data"], datasets[1]["target"]  # thyroid_sick (3772, 30), pos rate ~ 0.0612
    true_prevalence = y.mean()

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

    # A. Data Leakage vs Clean Pipeline (Random Forest)
    leaky_pr, clean_pr = [], []
    leaky_roc, clean_roc = [], []

    # Resample toàn bộ TRƯỚC KHI CHIA (SAI QUY TRÌNH - Data Leakage)
    X_scaled_all = MinMaxScaler().fit_transform(X)
    X_sm_all, y_sm_all = SMOTE(random_state=0).fit_resample(X_scaled_all, y)
    sss_leaky = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    for tr_idx, te_idx in sss_leaky.split(X_sm_all, y_sm_all):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=39, n_jobs=4)
        rf.fit(X_sm_all[tr_idx], y_sm_all[tr_idx])
        proba = rf.predict_proba(X_sm_all[te_idx])[:, 1]
        leaky_pr.append(average_precision_score(y_sm_all[te_idx], proba))
        leaky_roc.append(roc_auc_score(y_sm_all[te_idx], proba))

    # Chia TRƯỚC, SMOTE CHỈ TRÊN TRAIN (ĐÚNG QUY TRÌNH)
    # B. Calibration Experiment on clean split (Logistic Regression & Random Forest)
    brier_records = []
    for fold_i, (tr_idx, te_idx) in enumerate(sss.split(X, y)):
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]

        X_tr_sm, y_tr_sm = SMOTE(random_state=0).fit_resample(X_tr, y_tr)

        rf_clean = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=39, n_jobs=4)
        rf_clean.fit(X_tr_sm, y_tr_sm)
        proba_clean = rf_clean.predict_proba(X_te)[:, 1]
        clean_pr.append(average_precision_score(y_te, proba_clean))
        clean_roc.append(roc_auc_score(y_te, proba_clean))

        # Calibration comparison with LogisticRegression (rất rõ hiệu ứng thổi phồng xác suất)
        lr_base = LogisticRegression(max_iter=2000, random_state=0).fit(X_tr, y_tr)
        p_base = lr_base.predict_proba(X_te)[:, 1]

        lr_smote = LogisticRegression(max_iter=2000, random_state=0).fit(X_tr_sm, y_tr_sm)
        p_smote = lr_smote.predict_proba(X_te)[:, 1]

        # Recalibrate SMOTE model on original imbalanced validation distribution (using CV inside train)
        lr_calib = CalibratedClassifierCV(
            Pipeline([("sm", SMOTE(random_state=0)), ("lr", LogisticRegression(max_iter=2000, random_state=0))]),
            method="isotonic",
            cv=5,
        ).fit(X_tr, y_tr)
        p_calib = lr_calib.predict_proba(X_te)[:, 1]

        brier_records.append({
            "Fold": fold_i,
            "Base_Brier": brier_score_loss(y_te, p_base),
            "SMOTE_Brier": brier_score_loss(y_te, p_smote),
            "SMOTE_Calib_Brier": brier_score_loss(y_te, p_calib),
            "Base_MeanProb": p_base.mean(),
            "SMOTE_MeanProb": p_smote.mean(),
            "SMOTE_Calib_MeanProb": p_calib.mean(),
        })

    bdf = pd.DataFrame(brier_records)

    print(f" [A. Data Leakage] Leaky PR-AUC = {np.mean(leaky_pr):.3f} vs Clean PR-AUC = {np.mean(clean_pr):.3f} "
          f"(Ảo tưởng +{np.mean(leaky_pr)-np.mean(clean_pr):.3f}!)")
    print(f" [B. Calibration]  True Pos Rate = {true_prevalence:.4f}")
    print(f"   - Nosampling LR     : Mean Prob = {bdf['Base_MeanProb'].mean():.4f} | Brier = {bdf['Base_Brier'].mean():.4f}")
    print(f"   - SMOTE (Uncalib)   : Mean Prob = {bdf['SMOTE_MeanProb'].mean():.4f} | Brier = {bdf['SMOTE_Brier'].mean():.4f}  <-- THỔI PHỒNG XÁC SUẤT!")
    print(f"   - SMOTE + Isotonic  : Mean Prob = {bdf['SMOTE_Calib_MeanProb'].mean():.4f} | Brier = {bdf['SMOTE_Calib_Brier'].mean():.4f}  <-- ĐÃ HIỆU CHUẨN!")

    # Vẽ biểu đồ minh họa 2 cạm bẫy
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Data Leakage (ROC-AUC vs PR-AUC)
    labels = ["Clean Pipeline\n(Split -> SMOTE)", "Leaky Pipeline\n(SMOTE -> Split)"]
    pr_vals = [np.mean(clean_pr), np.mean(leaky_pr)]
    roc_vals = [np.mean(clean_roc), np.mean(leaky_roc)]
    x_pos = np.arange(len(labels))
    w = 0.35
    b1 = axes[0].bar(x_pos - w / 2, pr_vals, width=w, label="PR-AUC (Average Precision)", color="#d62728")
    b2 = axes[0].bar(x_pos + w / 2, roc_vals, width=w, label="ROC-AUC", color="#1f77b4")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].set_title("Pitfall 1: Data Leakage Inflation & ROC-AUC Illusion\n(thyroid_sick - Random Forest)", fontsize=11, fontweight="bold")
    axes[0].bar_label(b1, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")
    axes[0].bar_label(b2, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")
    axes[0].legend(loc="lower right")

    # Panel 2: Probability Calibration
    cal_labels = ["Nosampling\n(Baseline)", "SMOTE\n(Uncalibrated)", "SMOTE +\nIsotonic Calib"]
    mean_probs = [bdf["Base_MeanProb"].mean(), bdf["SMOTE_MeanProb"].mean(), bdf["SMOTE_Calib_MeanProb"].mean()]
    briers = [bdf["Base_Brier"].mean(), bdf["SMOTE_Brier"].mean(), bdf["SMOTE_Calib_Brier"].mean()]
    x2 = np.arange(len(cal_labels))
    bars_p = axes[1].bar(x2 - w / 2, mean_probs, width=w, label="Mean Predicted P(y=1)", color="#ff7f0e")
    bars_b = axes[1].bar(x2 + w / 2, briers, width=w, label="Brier Score (Lower=Better)", color="#2ca02c")
    axes[1].axhline(true_prevalence, color="black", linestyle="--", linewidth=1.3, label=f"True Positive Rate ({true_prevalence:.3f})")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cal_labels, fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, max(mean_probs) * 1.25)
    axes[1].set_ylabel("Probability / Brier Loss", fontsize=11)
    axes[1].set_title("Pitfall 2: Probability Inflation after SMOTE & Calibration Fix\n(thyroid_sick - Logistic Regression)", fontsize=11, fontweight="bold")
    axes[1].bar_label(bars_p, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")
    axes[1].bar_label(bars_b, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    pitfall_plot = os.path.join(RESULTS_DIR, "extended_pitfalls_leakage_calibration.png")
    plt.savefig(pitfall_plot, dpi=300)
    plt.close()
    print(f" [Saved Plot] {pitfall_plot}")

    return {
        "leaky_pr": float(np.mean(leaky_pr)),
        "clean_pr": float(np.mean(clean_pr)),
        "leaky_roc": float(np.mean(leaky_roc)),
        "clean_roc": float(np.mean(clean_roc)),
        "true_prev": float(true_prevalence),
        "brier_df": bdf,
    }


def save_extended_markdown(df_ext, pitfall_res):
    """Lưu báo cáo chi tiết kết quả mở rộng vào results/extended_summary.md"""
    lines = [
        "# BÁO CÁO KẾT QUẢ THỰC NGHIỆM MỞ RỘNG (ĐIỂM MỚI CỦA NHÓM)\n\n",
        "## 1. Benchmark Mở rộng: KMeans-SMOTE, SVM-SMOTE (Bài tập 1), Hybrid & Cost-Sensitive\n\n",
    ]
    for d_name in ["synthetic data", "thyroid_sick"]:
        sub = df_ext[df_ext["Dataset"] == d_name]
        piv_pr = sub.pivot(index="Method", columns="Classifier", values="PR_AUC")
        piv_roc = sub.pivot(index="Method", columns="Classifier", values="ROC_AUC")
        piv_brier = sub.pivot(index="Method", columns="Classifier", values="Brier_Score")
        piv_time = sub.pivot(index="Method", columns="Classifier", values="Fit_Time_ms")
        order = ["Nosampling", "Borderline-SMOTE", "KMeans-SMOTE", "SVM-SMOTE", "SMOTE-Tomek", "SMOTE-ENN", "Cost-Sensitive"]
        lines.append(f"### Dataset: `{d_name}` — Average Precision (PR-AUC)\n")
        lines.append(piv_pr.reindex(order).to_markdown() + "\n\n")
        lines.append(f"### Dataset: `{d_name}` — ROC-AUC (Đối chiếu ảo giác lạc quan)\n")
        lines.append(piv_roc.reindex(order).to_markdown() + "\n\n")
        lines.append(f"### Dataset: `{d_name}` — Brier Score (Sai số xác suất, càng thấp càng tốt)\n")
        lines.append(piv_brier.reindex(order).to_markdown() + "\n\n")
        lines.append(f"### Dataset: `{d_name}` — Fit Time trung bình (ms/fold)\n")
        lines.append(piv_time.reindex(order).to_markdown() + "\n\n")

    bdf = pitfall_res["brier_df"]
    lines.extend([
        "## 2. Thực chứng Cạm bẫy Khoa học (Data Leakage & Probability Calibration)\n\n",
        f"- **Data Leakage (Bài tập 3 - Random Forest trên `thyroid_sick`):**\n",
        f"  - Quy trình ĐÚNG (Split -> SMOTE chỉ trên Train): PR-AUC = **{pitfall_res['clean_pr']:.3f}**, ROC-AUC = **{pitfall_res['clean_roc']:.3f}**\n",
        f"  - Quy trình SAI (SMOTE toàn bộ -> Split): PR-AUC = **{pitfall_res['leaky_pr']:.3f}**, ROC-AUC = **{pitfall_res['leaky_roc']:.3f}** *(Thổi phồng giả tạo +{pitfall_res['leaky_pr'] - pitfall_res['clean_pr']:.3f} PR-AUC!)*\n\n",
        f"- **Hiệu chuẩn Xác suất (Trang 56 - Logistic Regression trên `thyroid_sick`, tỷ lệ dương tính thực tế = `{pitfall_res['true_prev']:.4f}`):**\n",
        f"  - `Nosampling`: Mean Predicted Prob = **{bdf['Base_MeanProb'].mean():.4f}** | Brier Score = **{bdf['Base_Brier'].mean():.4f}**\n",
        f"  - `SMOTE (Chưa hiệu chuẩn)`: Mean Predicted Prob = **{bdf['SMOTE_MeanProb'].mean():.4f}** *(Bị thổi phồng gấp {bdf['SMOTE_MeanProb'].mean()/pitfall_res['true_prev']:.1f}x!)* | Brier Score = **{bdf['SMOTE_Brier'].mean():.4f}**\n",
        f"  - `SMOTE + Isotonic Calibration`: Mean Predicted Prob = **{bdf['SMOTE_Calib_MeanProb'].mean():.4f}** | Brier Score = **{bdf['SMOTE_Calib_Brier'].mean():.4f}** *(Khôi phục chuẩn xác suất thực!)*\n",
    ])

    out_md = os.path.join(RESULTS_DIR, "extended_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f" [Saved Markdown] {out_md}")


if __name__ == "__main__":
    t0 = time.time()
    df_ext = run_extended_benchmark()
    pitfall_res = run_pitfalls_experiment()
    save_extended_markdown(df_ext, pitfall_res)
    print(f"\n HOÀN TẤT THỰC NGHIỆM MỞ RỘNG TRONG {time.time() - t0:.1f} GIÂY!")

