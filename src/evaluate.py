"""
evaluate.py
-----------
Model evaluation: accuracy score, confusion matrix, feature importance,
and RandomForest visualisation. All plots are saved to outputs/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

try:
    from src.data_loader import COMPOUND_LABELS, get_feature_names
except ModuleNotFoundError:
    from data_loader import COMPOUND_LABELS, get_feature_names

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colour palette (matches F1 tyre colours) ────────────────────────────────
PALETTE = {
    "Soft":   "#E8002D",   # F1 red
    "Medium": "#FFF200",   # F1 yellow
    "Hard":   "#AAAAAA",   # F1 grey/white → use grey for visibility
    "bg":     "#1A1A2E",   # dark navy background
    "text":   "#EAEAEA",
    "accent": "#FF1801",   # F1 red accent
}


def evaluate(clf,
             X_test,
             y_test,
             show_report: bool = True) -> dict:
    """
    Compute and print all evaluation metrics.
    Returns a dict with accuracy and the confusion matrix.
    """
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    class_ids = sorted(COMPOUND_LABELS.keys())
    cm = confusion_matrix(y_test, y_pred, labels=class_ids)

    print("=" * 50)
    print(f"  TEST ACCURACY : {acc * 100:.2f}%")
    print("=" * 50)

    if show_report:
        labels = [COMPOUND_LABELS[i] for i in class_ids]
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    labels=class_ids,
                                    target_names=labels,
                                    zero_division=0))

    return {"accuracy": acc, "confusion_matrix": cm, "y_pred": y_pred}


def plot_confusion_matrix(y_test, y_pred, save: bool = True) -> str:
    """Plot a styled confusion matrix and save as PNG."""
    class_ids = sorted(COMPOUND_LABELS.keys())
    labels    = [COMPOUND_LABELS[i] for i in class_ids]
    cm        = confusion_matrix(y_test, y_pred, labels=class_ids)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Custom colour map: dark → red
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    heatmap = sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#333355",
        annot_kws={"size": 16, "weight": "bold", "color": "white"},
        ax=ax,
    )

    ax.set_title("Confusion Matrix – Starting Compound Prediction",
                 color=PALETTE["text"], fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Compound", color=PALETTE["text"], fontsize=12)
    ax.set_ylabel("Actual Compound",    color=PALETTE["text"], fontsize=12)
    ax.tick_params(colors=PALETTE["text"])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if heatmap.collections and heatmap.collections[0].colorbar:
        heatmap.collections[0].colorbar.ax.tick_params(colors=PALETTE["text"])
    plt.tight_layout()


    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        print(f"[Evaluate] Confusion matrix saved → '{path}'")
    plt.close()
    return path


def plot_feature_importance(clf, save: bool = True) -> str:
    """Bar chart of feature importances."""
    features    = get_feature_names()
    importances = clf.feature_importances_
    idx         = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bar_colors = [PALETTE["accent"] if i == idx[0] else "#4A6FA5" for i in idx]
    bars = ax.barh(
        [features[i] for i in idx],
        [importances[i] for i in idx],
        color=bar_colors, edgecolor="none", height=0.6,
    )

    for bar, imp in zip(bars, [importances[i] for i in idx]):
        ax.text(imp + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{imp:.3f}", va="center", color=PALETTE["text"], fontsize=11)

    ax.set_title("Feature Importance",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")
    ax.set_xlabel("Gini Importance", color=PALETTE["text"], fontsize=11)
    ax.tick_params(colors=PALETTE["text"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#444466")
    ax.spines["left"].set_color("#444466")
    ax.set_xlim(0, max(importances) * 1.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        print(f"[Evaluate] Feature importance saved → '{path}'")
    plt.close()
    return path


def plot_decision_tree(clf, max_depth: int = 3, save: bool = True) -> str:
    """Visualise a Decision Tree or sample tree from Random Forest."""
    features = get_feature_names()
    labels   = [COMPOUND_LABELS[i] for i in sorted(COMPOUND_LABELS)]

    # Support both RandomForestClassifier and DecisionTreeClassifier
    if hasattr(clf, "estimators_"):
        single_tree = clf.estimators_[0]
        title = f"Sample Tree from Random Forest (top {max_depth} levels) – F1 Starting Compound"
    else:
        single_tree = clf
        title = f"Decision Tree (top {max_depth} levels) – F1 Starting Compound"

    fig, ax = plt.subplots(figsize=(20, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    plot_tree(
        single_tree,
        max_depth=max_depth,
        feature_names=features,
        class_names=labels,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
        impurity=False,
        proportion=False,
    )

    ax.set_title(
        title,
        color=PALETTE["text"], fontsize=15, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "randomForest_tree.png")
    dt_path = os.path.join(OUTPUT_DIR, "decision_tree.png")
    if save:
        plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.savefig(dt_path, dpi=120, bbox_inches="tight", facecolor=PALETTE["bg"])
        print(f"[Evaluate] Sample tree saved → '{path}' and '{dt_path}'")
    plt.close()
    return path


def plot_class_distribution(df: pd.DataFrame, save: bool = True) -> str:
    """Pie chart of class balance in the dataset."""
    counts = df["StartingCompound"].value_counts().sort_index()
    labels = [COMPOUND_LABELS[i] for i in counts.index]
    # Map colors strictly by compound label
    colors = [PALETTE[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    pie_result = ax.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"color": PALETTE["text"], "fontsize": 12},
        pctdistance=0.75,
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2},
    )
    autotexts = pie_result[2]

    for at, label in zip(autotexts, labels):
        at.set_fontweight("bold")
        # Ensure high-contrast text: dark text on bright Yellow/Hard slices, white on Red
        if label in ("Medium", "Hard"):
            at.set_color("#1A1A2E")
        else:
            at.set_color("#FFFFFF")

    ax.set_title("Starting Compound Distribution",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        print(f"[Evaluate] Class distribution saved → '{path}'")
    plt.close()
    return path

