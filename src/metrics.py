"""
Centralized evaluation metrics for all tasks.

Classification: Macro-F1, Micro-F1, Precision, Recall, AUC-PR, Per-Class Breakdown
Emotion:        MAE, R²
Retrieval:      R@1, R@5, R@10

Includes global and per-class threshold tuning on validation set only.
"""
from __future__ import annotations


import os
import json
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    mean_absolute_error,
    r2_score,
)


def apply_thresholds(y_prob: np.ndarray, thresholds: float | np.ndarray) -> np.ndarray:
    """
    Apply global or per-class thresholds to predicted probabilities.
    
    Args:
        y_prob: [N, C] predicted probabilities
        thresholds: float or [C] array of thresholds
        
    Returns:
        y_pred: [N, C] binary multi-hot array
    """
    if isinstance(thresholds, (int, float)):
        return (y_prob >= float(thresholds)).astype(int)
    thresholds = np.asarray(thresholds)
    if thresholds.ndim == 1:
        return (y_prob >= thresholds[np.newaxis, :]).astype(int)
    return (y_prob >= thresholds).astype(int)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Compute multi-label classification metrics.

    Args:
        y_true: [N, C] ground truth multi-hot
        y_pred: [N, C] predicted multi-hot (binary, after thresholding)
        y_prob: [N, C] predicted probabilities (for AUC-PR)
        threshold: not used here (thresholding done before call)

    Returns:
        dict of metrics
    """
    metrics = {}

    # Safety: ensure predictions are binary
    y_pred_bin = (y_pred > 0.5).astype(int) if y_pred.dtype != int else y_pred

    metrics["macro_f1"] = float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0))
    metrics["micro_f1"] = float(f1_score(y_true, y_pred_bin, average="micro", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred_bin, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred_bin, average="macro", zero_division=0))
    metrics["precision_micro"] = float(precision_score(y_true, y_pred_bin, average="micro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(y_true, y_pred_bin, average="micro", zero_division=0))

    if y_prob is not None:
        try:
            metrics["auc_pr_macro"] = float(
                average_precision_score(y_true, y_prob, average="macro")
            )
            metrics["auc_pr_micro"] = float(
                average_precision_score(y_true, y_prob, average="micro")
            )
        except ValueError:
            metrics["auc_pr_macro"] = 0.0
            metrics["auc_pr_micro"] = 0.0

    return metrics


def compute_detailed_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    genre_labels: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute Precision, Recall, F1, PR-AUC, and Support for each class individually.
    
    Returns:
        {class_name: {"precision": float, "recall": float, "f1": float, "pr_auc": float, "support": int}}
    """
    num_classes = y_true.shape[1]
    if genre_labels is None:
        genre_labels = [f"Class_{i}" for i in range(num_classes)]

    per_class = {}
    for c in range(num_classes):
        c_true = y_true[:, c]
        c_pred = (y_pred[:, c] > 0.5).astype(int) if y_pred.dtype != int else y_pred[:, c]
        
        prec = float(precision_score(c_true, c_pred, zero_division=0))
        rec = float(recall_score(c_true, c_pred, zero_division=0))
        f1 = float(f1_score(c_true, c_pred, zero_division=0))
        supp = int(np.sum(c_true))
        
        pr_auc = 0.0
        if y_prob is not None:
            try:
                pr_auc = float(average_precision_score(c_true, y_prob[:, c]))
            except ValueError:
                pr_auc = 0.0

        label_name = genre_labels[c] if c < len(genre_labels) else f"Class_{c}"
        per_class[label_name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "pr_auc": pr_auc,
            "support": supp,
        }

    return per_class


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "macro_f1",
    steps: int = 50,
) -> tuple[float, float]:
    """
    Find single optimal classification threshold on validation set.

    Args:
        y_true: [N, C] ground truth
        y_prob: [N, C] predicted probabilities
        metric: which metric to maximise
        steps: number of thresholds to try

    Returns:
        (best_threshold, best_metric_value)
    """
    best_thresh = 0.5
    best_val = 0.0

    for t in np.linspace(0.1, 0.9, steps):
        y_pred = (y_prob >= t).astype(int)
        m = compute_classification_metrics(y_true, y_pred, y_prob)
        if m.get(metric, 0) > best_val:
            best_val = m[metric]
            best_thresh = float(t)

    return best_thresh, best_val


def tune_per_class_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    steps: int = 50,
) -> tuple[np.ndarray, float]:
    """
    Find optimal independent threshold for each class on validation set.
    
    Args:
        y_true: [N, C] ground truth multi-hot on validation set
        y_prob: [N, C] predicted probabilities on validation set
        steps: number of search steps per class
        
    Returns:
        (best_thresholds [C], val_macro_f1)
    """
    num_classes = y_true.shape[1]
    best_thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    for c in range(num_classes):
        c_true = y_true[:, c]
        c_prob = y_prob[:, c]
        best_c_thresh = 0.5
        best_c_f1 = -1.0

        for t in np.linspace(0.05, 0.95, steps):
            c_pred = (c_prob >= t).astype(int)
            c_f1 = f1_score(c_true, c_pred, zero_division=0)
            if c_f1 > best_c_f1:
                best_c_f1 = c_f1
                best_c_thresh = float(t)

        best_thresholds[c] = best_c_thresh

    # Compute validation macro-f1 with these per-class thresholds
    y_pred = apply_thresholds(y_prob, best_thresholds)
    val_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return best_thresholds, val_macro_f1


def compute_emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics for emotion (valence/arousal).
    Only used when DEAM labels are available.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _sanitize_for_json(obj, seen=None):
    """Recursively convert numpy types to native Python types and remove circular refs."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if isinstance(obj, (dict, list)):
        if obj_id in seen:
            return "<circular reference>"
        seen.add(obj_id)

    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v, seen.copy()) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item, seen.copy()) for item in obj]
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)


def save_metrics(metrics: dict, path: str):
    """Save metrics dict to JSON file safely with numpy sanitization and circular reference protection."""
    clean_metrics = _sanitize_for_json(metrics)
    os_dir = os.path.dirname(os.path.abspath(path))
    if os_dir:
        os.makedirs(os_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(clean_metrics, f, indent=2, default=str)


def load_metrics(path: str) -> dict:
    """Load metrics from JSON file."""
    with open(path, "r") as f:
        return json.load(f)

