"""CNN that classifies each expected phone as correctly / incorrectly pronounced.

Input: a 300 ms log-mel window plus a phone-mask channel (2 x 64 x 30) and the
index of the phone that *should* have been produced.
"""
import copy
from typing import Optional

import numpy as np
import torch
from torch import nn

from src import metrics


class PhoneCNN(nn.Module):
    """Four conv blocks -> global average pooling -> concat phone embedding -> MLP head."""

    def __init__(self, n_phones: int = 40, embed_dim: int = 16,
                 channels: tuple[int, ...] = (32, 64, 128, 128), dropout: float = 0.3):
        super().__init__()
        blocks, in_c = [], 2
        for c in channels:
            blocks += [nn.Conv2d(in_c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            in_c = c
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Embedding(n_phones, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(channels[-1] + embed_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(x)).flatten(1)

    def forward(self, x: torch.Tensor, phone: torch.Tensor) -> torch.Tensor:
        """Logit of "mispronounced", shape (B,)."""
        return self.head(torch.cat([self.encode(x), self.embed(phone)], dim=1)).squeeze(1)


class Normalizer:
    """Per-mel-bin standardisation of the spectrogram channel; the mask is left as is."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean, self.std = mean, std  # each (64,)

    def __call__(self, windows: np.ndarray) -> torch.Tensor:
        x = windows.astype(np.float32)
        x[:, 0] = (x[:, 0] - self.mean[None, :, None]) / self.std[None, :, None]
        return torch.from_numpy(x)


def fit_normalizer(windows: np.ndarray) -> Normalizer:
    spec = windows[:, 0].astype(np.float32)  # (N, 64, T)
    mean = spec.mean(axis=(0, 2))
    std = spec.std(axis=(0, 2))
    std[std < 1e-6] = 1.0
    return Normalizer(mean, std)


def pos_weight(labels: np.ndarray) -> float:
    """Loss weight for the minority (mispronounced) class: negatives / positives."""
    positives = int(labels.sum())
    return (len(labels) - positives) / positives


def _pick_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict_scores(model: PhoneCNN, norm: Normalizer, a: dict, device: Optional[str] = None,
                   batch_size: int = 1024) -> np.ndarray:
    """P(mispronounced) per example."""
    dev = _pick_device(device)
    model = model.to(dev).eval()
    out = []
    for i in range(0, len(a["label"]), batch_size):
        x = norm(a["windows"][i : i + batch_size]).to(dev)
        phone = torch.from_numpy(a["phone"][i : i + batch_size]).to(dev)
        out.append(torch.sigmoid(model(x, phone)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_model(train: dict, val: dict, epochs: int = 15, batch_size: int = 256, lr: float = 1e-3,
                patience: int = 3, seed: int = 0, device: Optional[str] = None):
    """Train from scratch; early-stop on validation F1 and restore the best epoch.

    Returns (model, normalizer, history). history["threshold"] is the decision
    threshold that maximised validation F1 at the best epoch.
    """
    dev = _pick_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    norm = fit_normalizer(train["windows"])
    x_train = norm(train["windows"])
    phone_train = torch.from_numpy(train["phone"])
    y_train = torch.from_numpy(train["label"]).float()

    model = PhoneCNN().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight(train["label"]), device=dev))

    hist = {"train_loss": [], "val_f1": [], "val_threshold": [], "best_epoch": 0, "threshold": 0.5}
    best_state, stale = None, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(y_train))
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[i : i + batch_size])
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx].to(dev), phone_train[idx].to(dev)), y_train[idx].to(dev))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        hist["train_loss"].append(total / len(order))

        scores = predict_scores(model, norm, val, device=str(dev))
        threshold = metrics.best_threshold(val["label"], scores)
        f1 = metrics.prf(val["label"], (scores >= threshold).astype(int))["f1"]
        hist["val_f1"].append(f1)
        hist["val_threshold"].append(threshold)

        if f1 > max(hist["val_f1"][:-1], default=-1.0):
            best_state, stale = copy.deepcopy(model.state_dict()), 0
            hist["best_epoch"], hist["threshold"] = epoch, threshold
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model.to(dev), norm, hist
