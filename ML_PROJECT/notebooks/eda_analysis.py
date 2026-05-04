"""
=============================================================================
AGENT 2: EDA & FEATURE ANALYSIS AGENT
=============================================================================
Purpose : Perform Exploratory Data Analysis on the breast cancer dataset.
Input   : Structured dict from Agent 1 (keys: raw_df, feature_names)
Output  : Visualisations saved to notebooks/plots/ + structured dict

Author  : Data Analyst (Agent 2)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Premium colour palette & visual style
# ---------------------------------------------------------------------------
BENIGN_COLOR    = "#00C9A7"   # teal-green
MALIGNANT_COLOR = "#FF6B6B"   # coral-red
BG_COLOR        = "#0E1117"   # dark background
CARD_BG         = "#1A1D23"   # card/panel background
TEXT_COLOR      = "#E8E8E8"   # light text
GRID_COLOR      = "#2A2D35"   # subtle grid lines

COLORS = {"Benign": BENIGN_COLOR, "Malignant": MALIGNANT_COLOR}
COLOR_LIST = [BENIGN_COLOR, MALIGNANT_COLOR]

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
sns.set_theme(style="darkgrid", rc=plt.rcParams)


def _save(fig, name):
    """Save figure and close."""
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"   -> Saved: plots/{name}")
    return path


def run_eda(agent1_output: dict) -> dict:
    """
    Full EDA pipeline.

    Parameters
    ----------
    agent1_output : dict from Agent 1's load_and_preprocess()
        Required keys: "raw_df", "feature_names".

    Returns
    -------
    dict with keys: top_features, class_distribution, insights
    """
    df = agent1_output["raw_df"].copy()
    feature_names = agent1_output["feature_names"]
    insights = []
    order = ["Benign", "Malignant"]

    print("\n" + "=" * 65)
    print("  AGENT 2 - EDA & FEATURE ANALYSIS")
    print("=" * 65)

    # === 1. CLASS DISTRIBUTION (donut chart) ================================
    class_dist = df["target"].value_counts().to_dict()
    total = sum(class_dist.values())
    print("\n  Class Distribution:")
    for cls, cnt in class_dist.items():
        print(f"   {cls:12s}: {cnt}  ({cnt/total*100:.1f}%)")

    b_pct = class_dist.get("Benign", 0) / total * 100
    m_pct = class_dist.get("Malignant", 0) / total * 100
    insights.append(
        f"Dataset is {'reasonably balanced' if abs(b_pct-m_pct)<15 else 'moderately imbalanced'} "
        f"({b_pct:.0f}% Benign vs {m_pct:.0f}% Malignant)."
    )

    # -- Donut chart + bar chart side-by-side --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Donut
    counts = [class_dist.get(c, 0) for c in order]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=order, colors=COLOR_LIST, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor=BG_COLOR, linewidth=2),
        textprops=dict(color=TEXT_COLOR, fontsize=13, fontweight="bold"),
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_color(TEXT_COLOR)
    ax1.set_title("Class Distribution", fontsize=15, fontweight="bold", pad=15)
    centre = plt.Circle((0, 0), 0.55, fc=CARD_BG)
    ax1.add_artist(centre)
    ax1.text(0, 0, f"n={total}", ha="center", va="center",
             fontsize=16, fontweight="bold", color=TEXT_COLOR)

    # Bar
    bars = ax2.bar(order, counts, color=COLOR_LIST, width=0.5,
                   edgecolor=BG_COLOR, linewidth=1.5, zorder=3)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                 str(cnt), ha="center", va="bottom", fontweight="bold",
                 fontsize=14, color=TEXT_COLOR)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_ylim(0, max(counts) * 1.2)
    ax2.set_title("Sample Counts", fontsize=15, fontweight="bold", pad=15)
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle("1 | Target Variable Analysis", fontsize=17, fontweight="bold",
                 y=1.02, color="#FFD700")
    _save(fig, "01_class_distribution.png")

    # === 2. CORRELATION ANALYSIS ============================================
    df["target_num"] = df["target"].map({"Malignant": 0, "Benign": 1})
    corr_target = (df[feature_names + ["target_num"]].corr()["target_num"]
                   .drop("target_num").sort_values(key=abs, ascending=False))
    top_10 = corr_target.head(10).index.tolist()

    print("\n  Top 10 features by |correlation| with target:")
    for i, f in enumerate(top_10, 1):
        print(f"   {i:2d}. {f:30s}  r = {corr_target[f]:+.3f}")

    insights.append(f"Top-3 correlated features: {', '.join(top_10[:3])}.")

    # -- Horizontal bar chart of correlations --
    fig, ax = plt.subplots(figsize=(10, 7))
    top_15 = corr_target.head(15)
    colours = [MALIGNANT_COLOR if v < 0 else BENIGN_COLOR for v in top_15.values]
    bars = ax.barh(top_15.index[::-1], top_15.values[::-1],
                   color=colours[::-1], edgecolor=BG_COLOR, height=0.6, zorder=3)
    ax.axvline(0, color=GRID_COLOR, linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Target", fontsize=12)
    ax.set_title("2 | Feature Correlation with Diagnosis", fontsize=17,
                 fontweight="bold", pad=15, color="#FFD700")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + (0.01 if w >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f"{w:+.3f}", va="center", ha="left" if w >= 0 else "right",
                fontsize=9, color=TEXT_COLOR)
    ax.grid(axis="x", alpha=0.2)
    _save(fig, "02_feature_correlations.png")

    # -- Heatmap (top 10) --
    fig, ax = plt.subplots(figsize=(11, 9))
    cm = df[top_10 + ["target_num"]].corr()
    mask = np.triu(np.ones_like(cm, dtype=bool))
    sns.heatmap(cm, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.8, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                annot_kws={"fontsize": 9})
    ax.set_title("3 | Correlation Heatmap - Top 10 Features + Target",
                 fontsize=15, fontweight="bold", pad=15, color="#FFD700")
    _save(fig, "03_correlation_heatmap.png")

    # === 3. VIOLIN + BOX PLOTS (top 6) =====================================
    top_6 = top_10[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, feat in zip(axes.ravel(), top_6):
        parts = ax.violinplot(
            [df[df["target"] == c][feat].values for c in order],
            positions=[0, 1], showmeans=True, showmedians=True, showextrema=False
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(COLOR_LIST[i])
            pc.set_alpha(0.5)
        parts["cmeans"].set_color("#FFD700")
        parts["cmedians"].set_color(TEXT_COLOR)

        # Overlay strip / jitter
        for i, cls in enumerate(order):
            vals = df[df["target"] == cls][feat].values
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       alpha=0.15, s=8, color=COLOR_LIST[i], zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(order, fontsize=11)
        ax.set_title(feat.title(), fontsize=12, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0.15)

    fig.suptitle("4 | Feature Distributions by Class (Violin + Strip)",
                 fontsize=17, fontweight="bold", y=1.01, color="#FFD700")
    fig.tight_layout()
    _save(fig, "04_violin_plots_top6.png")
    insights.append("Violin plots show clear Benign/Malignant separation in 'worst' measurements.")

    # === 4. HISTOGRAMS + KDE (top 6) ========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, feat in zip(axes.ravel(), top_6):
        for cls, col in zip(order, COLOR_LIST):
            subset = df[df["target"] == cls][feat]
            ax.hist(subset, bins=30, alpha=0.35, color=col,
                    edgecolor="none", density=True, label=f"{cls} (hist)")
            subset.plot.kde(ax=ax, color=col, linewidth=2, label=f"{cls} (KDE)")
        ax.set_title(feat.title(), fontsize=12, fontweight="bold", pad=8)
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(axis="y", alpha=0.15)

    fig.suptitle("5 | Feature Distributions with KDE Overlay",
                 fontsize=17, fontweight="bold", y=1.01, color="#FFD700")
    fig.tight_layout()
    _save(fig, "05_histograms_kde_top6.png")
    insights.append("Malignant tumours have larger radius, perimeter, area & concavity values.")

    # === 5. PAIR PLOT (top 4 features) ======================================
    top_4 = top_10[:4]
    pair_df = df[top_4 + ["target"]].copy()
    g = sns.pairplot(pair_df, hue="target", palette=COLORS,
                     diag_kind="kde", plot_kws={"alpha": 0.4, "s": 20},
                     diag_kws={"fill": True, "alpha": 0.4, "linewidth": 2})
    g.figure.set_facecolor(BG_COLOR)
    g.figure.suptitle("6 | Pair Plot - Top 4 Discriminative Features",
                      fontsize=17, fontweight="bold", y=1.02, color="#FFD700")
    for ax in g.axes.ravel():
        ax.set_facecolor(CARD_BG)
    _save(g.figure, "06_pair_plot_top4.png")
    insights.append("Pair plot confirms multi-dimensional separability between classes.")

    # === 6. FEATURE IMPORTANCE BAR CHART ====================================
    fig, ax = plt.subplots(figsize=(10, 7))
    top_10_abs = corr_target.head(10).abs()
    gradient_colors = plt.cm.viridis(np.linspace(0.3, 0.95, len(top_10_abs)))
    bars = ax.barh(top_10_abs.index[::-1], top_10_abs.values[::-1],
                   color=gradient_colors[::-1], edgecolor=BG_COLOR, height=0.6, zorder=3)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=10, color=TEXT_COLOR)
    ax.set_xlabel("|Correlation|", fontsize=12)
    ax.set_title("7 | Top 10 Most Important Features",
                 fontsize=17, fontweight="bold", pad=15, color="#FFD700")
    ax.set_xlim(0, top_10_abs.max() * 1.15)
    ax.grid(axis="x", alpha=0.2)
    _save(fig, "07_feature_importance.png")

    # === SUMMARY =============================================================
    print(f"\n  Key Insights:")
    for i, ins in enumerate(insights, 1):
        print(f"   {i}. {ins}")
    print("=" * 65)

    df.drop(columns=["target_num"], inplace=True)

    return {"top_features": top_10, "class_distribution": class_dist, "insights": insights}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.data_loader import load_and_preprocess
    result = run_eda(load_and_preprocess())
    print("\n  Agent 2 completed.")
