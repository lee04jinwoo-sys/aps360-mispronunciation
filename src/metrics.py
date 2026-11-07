"""Precision / recall / F1 on the *mispronounced* class, shared by all models."""
import numpy as np

from src.data_stats import L1


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def prf_by_l1(y_true: np.ndarray, y_pred: np.ndarray, speakers: np.ndarray) -> dict[str, dict[str, float]]:
    languages = np.array([L1[s] for s in speakers])
    return {
        lang: prf(y_true[languages == lang], y_pred[languages == lang])
        for lang in sorted(set(languages))
    }


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """The score threshold (predict 1 when score >= t) that maximises F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(scores):
        f1 = prf(y_true, (scores >= t).astype(int))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t
