"""Baseline: per-phone MFCC statistics + one-hot expected phone -> logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from src import dataset

N_CONTINUOUS = 78


def feature_matrix(a: dict[str, np.ndarray]) -> np.ndarray:
    """(N, 118): 78 MFCC statistics followed by a 40-way one-hot of the expected phone."""
    onehot = np.zeros((len(a["phone"]), len(dataset.PHONES)), dtype=np.float32)
    onehot[np.arange(len(a["phone"])), a["phone"]] = 1.0
    return np.concatenate([a["mfcc"], onehot], axis=1)


class Scaler:
    """Standardises the continuous columns with statistics fitted on training data."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean, self.std = mean, std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = x.copy()
        z[:, :N_CONTINUOUS] = (x[:, :N_CONTINUOUS] - self.mean) / self.std
        return z


def fit_scaler(x: np.ndarray) -> Scaler:
    mean = x[:, :N_CONTINUOUS].mean(axis=0)
    std = x[:, :N_CONTINUOUS].std(axis=0)
    std[std < 1e-8] = 1.0
    return Scaler(mean, std)


def train(a: dict[str, np.ndarray]) -> tuple[LogisticRegression, Scaler]:
    x = feature_matrix(a)
    scaler = fit_scaler(x)
    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    model.fit(scaler(x), a["label"])
    return model, scaler


def predict_scores(model: LogisticRegression, scaler: Scaler, a: dict[str, np.ndarray]) -> np.ndarray:
    """P(mispronounced) per example."""
    return model.predict_proba(scaler(feature_matrix(a)))[:, 1]
