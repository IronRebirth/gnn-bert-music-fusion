"""
Visualization utilities for training curves, model comparison, embeddings, and case studies.
"""
from __future__ import annotations


import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_training_curves(history: dict, save_dir: str, prefix: str = ""):
    """
    Plot training/validation loss and F1 curves.

    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'val_macro_f1', etc.
                 Each value is a list (one entry per epoch).
        save_dir: directory to save plots
        prefix: filename prefix (e.g. 'task1_')
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # --- Loss ---
    if "train_loss" in history and "val_loss" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{prefix}loss.png"), dpi=150)
        plt.close(fig)

    # --- Macro-F1 ---
    if "val_macro_f1" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["val_macro_f1"], label="Val Macro-F1", linewidth=2, color="green")
        if "val_micro_f1" in history:
            ax.plot(epochs, history["val_micro_f1"], label="Val Micro-F1", linewidth=2, color="blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_title("Validation F1 Scores")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{prefix}f1.png"), dpi=150)
        plt.close(fig)


def plot_model_comparison(results: dict, save_dir: str):
    """
    Create bar charts comparing models.

    Args:
        results: {model_name: {metric: value}} e.g.
                 {"BERT": {"macro_f1": 0.6, "micro_f1": 0.7, "auc_pr_macro": 0.5}}
        save_dir: output directory
    """
    os.makedirs(save_dir, exist_ok=True)

    models = list(results.keys())
    if not models:
        return

    # Macro-F1 comparison
    metric_keys = ["macro_f1", "micro_f1", "auc_pr_macro"]
    metric_names = ["Macro-F1", "Micro-F1", "AUC-PR"]

    for mkey, mname in zip(metric_keys, metric_names):
        values = [results[m].get(mkey, 0) for m in models]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(models, values, color=sns.color_palette("viridis", len(models)))
        ax.set_ylabel(mname)
        ax.set_title(f"Model Comparison — {mname}")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"comparison_{mkey}.png"), dpi=150)
        plt.close(fig)


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    save_path: str,
    title: str = "t-SNE of Fusion Embeddings",
):
    """
    Generate t-SNE plot of embeddings coloured by label.

    Args:
        embeddings: [N, D] embedding vectors
        labels: [N] integer labels (e.g. primary genre index)
        label_names: names for each label index
        save_path: file path for output image
        title: plot title
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels.tolist()))
    palette = sns.color_palette("husl", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = label_names[lbl] if lbl < len(label_names) else f"Class {lbl}"
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[i]], label=name, alpha=0.6, s=20)

    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    """Plot per-class confusion for multi-label (as heatmap of TP/FP/FN/TN rates)."""
    from sklearn.metrics import multilabel_confusion_matrix

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    # mcm shape: [n_classes, 2, 2] — each is [[TN, FP], [FN, TP]]
    n_classes = len(label_names)

    tp = np.array([mcm[i, 1, 1] for i in range(n_classes)])
    fp = np.array([mcm[i, 0, 1] for i in range(n_classes)])
    fn = np.array([mcm[i, 1, 0] for i in range(n_classes)])
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_classes)
    width = 0.35
    ax.bar(x - width / 2, precision, width, label="Precision", color="steelblue")
    ax.bar(x + width / 2, recall, width, label="Recall", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision & Recall")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_comparison_table(results: dict) -> str:
    """
    Generate a Markdown comparison table from results dict.

    Args:
        results: {model_name: {macro_f1, micro_f1, auc_pr_macro}}

    Returns:
        Markdown string
    """
    header = "| Model | Macro-F1 | Micro-F1 | AUC-PR |\n"
    header += "| --- | ---: | ---: | ---: |\n"

    rows = []
    for name, m in results.items():
        macro = m.get("macro_f1", "—")
        micro = m.get("micro_f1", "—")
        auc = m.get("auc_pr_macro", "—")
        if isinstance(macro, float):
            macro = f"{macro:.4f}"
        if isinstance(micro, float):
            micro = f"{micro:.4f}"
        if isinstance(auc, float):
            auc = f"{auc:.4f}"
        rows.append(f"| {name} | {macro} | {micro} | {auc} |")

    return header + "\n".join(rows)
