"""Task metrics implemented from scratch on numpy (no sklearn dependency):
accuracy, top-k, confusion matrix, per-class precision/recall/F1, macro-F1."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    idx = y_true * n_classes + y_pred
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def prf_from_cm(cm: np.ndarray) -> dict[str, np.ndarray | float]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_f1": float(f1.mean()),
    }


def topk_accuracy(y_true: np.ndarray, logits: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    hits = (topk == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def summarize(
    logits: np.ndarray,
    y_true: np.ndarray,
    classes: list[str],
    loss: float | None = None,
) -> dict:
    """Full metric summary from raw logits."""
    y_pred = logits.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, len(classes))
    prf = prf_from_cm(cm)
    per_class = {
        classes[i]: {
            "precision": round(float(prf["precision"][i]), 4),
            "recall": round(float(prf["recall"][i]), 4),
            "f1": round(float(prf["f1"][i]), 4),
            "support": int(prf["support"][i]),
        }
        for i in range(len(classes))
    }
    out = {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": prf["macro_f1"],
        "top5_accuracy": topk_accuracy(y_true, logits, 5),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_examples": int(len(y_true)),
    }
    if loss is not None:
        out["loss"] = float(loss)
    return out
