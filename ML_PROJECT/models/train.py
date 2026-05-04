"""
=============================================================================
AGENT 3: MODEL TRAINING & EVALUATION AGENT
=============================================================================
Purpose : Train, evaluate, and compare ML models for breast cancer detection.
Input   : Structured dict from Agent 1 (X_train, X_test, y_train, y_test, feature_names)
Output  : Structured dict with results, best model, comparison table, and plots.

Author  : ML Engineer (Agent 3)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# ---------------------------------------------------------------------------
# Dark theme (matches Agent 2 style)
# ---------------------------------------------------------------------------
BG_COLOR   = "#0E1117"
CARD_BG    = "#1A1D23"
TEXT_COLOR  = "#E8E8E8"
GRID_COLOR  = "#2A2D35"
ACCENT      = "#FFD700"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   GRID_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "grid.color":       GRID_COLOR,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
    "font.size":        11,
})

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"   -> Saved: plots/{name}")


def train_and_evaluate(agent1_output: dict) -> dict:
    """
    Train 3 models, evaluate, compare, and select the best one.

    Parameters
    ----------
    agent1_output : dict from Agent 1
        Required keys: X_train, X_test, y_train, y_test, feature_names

    Returns
    -------
    dict
        results          : {model_name: {accuracy, precision, recall, f1, confusion_matrix}}
        best_model_name  : str
        best_model       : trained sklearn estimator
        comparison_table : pd.DataFrame
    """
    X_train = agent1_output["X_train"]
    X_test  = agent1_output["X_test"]
    y_train = agent1_output["y_train"]
    y_test  = agent1_output["y_test"]
    feature_names = agent1_output["feature_names"]

    print("\n" + "=" * 65)
    print("  AGENT 3 - MODEL TRAINING & EVALUATION")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Define models to train
    # ------------------------------------------------------------------
    # - Logistic Regression: interpretable, provides feature importance
    # - SVM (RBF kernel): strong with scaled data, good for non-linear
    # - KNN: instance-based, good baseline for comparison
    # ------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training: {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained_models[name] = model

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="Malignant")
        rec  = recall_score(y_test, y_pred, pos_label="Malignant")
        f1   = f1_score(y_test, y_pred, pos_label="Malignant")
        cm   = confusion_matrix(y_test, y_pred, labels=["Benign", "Malignant"])

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm.tolist(),
        }

        print(f"   Accuracy  : {acc:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}  <- critical for medical diagnosis")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"   Confusion Matrix:\n{cm}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
        })
    comparison_df = pd.DataFrame(rows).set_index("Model")

    print("\n" + "-" * 65)
    print("  MODEL COMPARISON TABLE")
    print("-" * 65)
    print(comparison_df.to_string())

    # ------------------------------------------------------------------
    # Select best model -- PRIORITISE RECALL
    # ------------------------------------------------------------------
    best_name = comparison_df["Recall"].idxmax()
    max_recall = comparison_df["Recall"].max()
    tied = comparison_df[comparison_df["Recall"] == max_recall]
    if len(tied) > 1:
        best_name = tied["F1-Score"].idxmax()

    print(f"\n  Best Model: {best_name}")
    print(f"   -> Selected because it has the highest Recall ({results[best_name]['recall']}),")
    print("     minimising the risk of missing a Malignant diagnosis.")

    # ------------------------------------------------------------------
    # WHY RECALL IS CRITICAL
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("  WHY RECALL IS CRITICAL IN THIS MEDICAL USE CASE")
    print("-" * 65)
    print("   - A False Negative means telling a cancer patient they are healthy.")
    print("   - This can delay treatment and endanger the patient's life.")
    print("   - Recall measures the % of actual Malignant cases correctly caught.")
    print("   - We prioritise Recall over Accuracy or Precision for patient safety.")

    # ------------------------------------------------------------------
    # Feature importance (Logistic Regression coefficients)
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("  FEATURE IMPORTANCE (Logistic Regression Coefficients)")
    print("-" * 65)
    lr_model = trained_models["Logistic Regression"]
    coefs = pd.Series(lr_model.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)
    print("   Top 10 features by |coefficient|:")
    for i, (feat, coef) in enumerate(coefs.head(10).items(), 1):
        direction = "-> Malignant" if coef < 0 else "-> Benign"
        print(f"   {i:2d}. {feat:30s}  coef = {coef:+.4f}  {direction}")

    # ==================================================================
    # VISUALISATIONS
    # ==================================================================

    # --- 8 | Model Comparison Grouped Bar Chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = comparison_df.index.tolist()
    x = np.arange(len(metrics_list))
    width = 0.22
    model_colors = ["#667eea", "#FF6B6B", "#00C9A7"]

    for i, (model_name, color) in enumerate(zip(model_names, model_colors)):
        vals = comparison_df.loc[model_name, metrics_list].values
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=color, edgecolor=BG_COLOR, linewidth=1, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_list, fontsize=12)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.3)
    ax.set_title("8 | Model Performance Comparison",
                 fontsize=17, fontweight="bold", pad=15, color=ACCENT)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, "08_model_comparison.png")

    # --- 9 | Confusion Matrices Side-by-Side ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    labels = ["Benign", "Malignant"]

    for ax, (model_name, color) in zip(axes, zip(model_names, model_colors)):
        cm_data = np.array(results[model_name]["confusion_matrix"])
        sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    linewidths=1, linecolor=GRID_COLOR, cbar=False,
                    annot_kws={"fontsize": 18, "fontweight": "bold"})
        ax.set_title(model_name, fontsize=13, fontweight="bold", pad=10, color=color)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle("9 | Confusion Matrices", fontsize=17, fontweight="bold",
                 y=1.03, color=ACCENT)
    fig.tight_layout()
    _save(fig, "09_confusion_matrices.png")

    # --- 10 | Feature Importance (LR coefficients) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    top_12 = coefs.head(12)
    colours = ["#FF6B6B" if v < 0 else "#00C9A7" for v in top_12.values]
    bars = ax.barh(top_12.index[::-1], top_12.values[::-1],
                   color=colours[::-1], edgecolor=BG_COLOR, height=0.6, zorder=3)
    ax.axvline(0, color=GRID_COLOR, linewidth=0.8)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + (0.02 if w >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                f"{w:+.3f}", va="center", ha="left" if w >= 0 else "right",
                fontsize=9, color=TEXT_COLOR)
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title("10 | Feature Importance (Logistic Regression)",
                 fontsize=17, fontweight="bold", pad=15, color=ACCENT)
    ax.grid(axis="x", alpha=0.2)
    _save(fig, "10_feature_importance_lr.png")

    print("=" * 65)

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": trained_models[best_name],
        "all_models": trained_models,
        "comparison_table": comparison_df,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.data_loader import load_and_preprocess
    result = train_and_evaluate(load_and_preprocess())
    print("\n  Agent 3 completed.")
