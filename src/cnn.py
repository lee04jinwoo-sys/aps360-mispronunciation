"""CNN that classifies each expected phone as correctly / incorrectly pronounced.

Input: a 300 ms log-mel window plus a phone-mask channel (2 x 64 x 30) and the
index of the phone that *should* have been produced.
"""
import copy
from typing import Optional

import numpy as np
import torch
from torch import nn

from src import metrics, pretrain

CHANNELS = (32, 64, 128, 128)


def conv_encoder(channels: tuple[int, ...] = CHANNELS) -> nn.Sequential:
    blocks, in_c = [], 2
    for c in channels:
        blocks += [nn.Conv2d(in_c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
        in_c = c
    return nn.Sequential(*blocks)


class PhoneClassifier(nn.Module):
    """Same encoder as PhoneCNN, trained to name the phone (pre-training task)."""

    def __init__(self, n_phones: int = 40, channels: tuple[int, ...] = CHANNELS):
        super().__init__()
        self.encoder = conv_encoder(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Linear(channels[-1], n_phones)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.pool(self.encoder(x)).flatten(1))


def load_encoder(model: nn.Module, state: dict) -> None:
    """Copy pre-trained encoder weights into a PhoneCNN (or any model with .encoder)."""
    model.encoder.load_state_dict(state)


class PhoneCNN(nn.Module):
    """Four conv blocks -> global average pooling -> concat phone embedding -> MLP head."""

    def __init__(self, n_phones: int = 40, embed_dim: int = 16,
                 channels: tuple[int, ...] = CHANNELS, dropout: float = 0.3):
        super().__init__()
        self.encoder = conv_encoder(channels)
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
                patience: int = 3, seed: int = 0, device: Optional[str] = None,
                init_encoder: Optional[dict] = None, norm: Optional[Normalizer] = None):
    """Train the detector; early-stop on validation F1 and restore the best epoch.

    By default it starts from scratch with a normalizer fitted on `train`. For
    the pre-training ablation, pass the pre-trained encoder weights and the
    normalizer used during pre-training; everything else stays the same (the
    head is initialized from the same seed either way).

    Returns (model, normalizer, history). history["threshold"] is the decision
    threshold that maximised validation F1 at the best epoch.
    """
    dev = _pick_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if norm is None:
        norm = fit_normalizer(train["windows"])
    x_train = norm(train["windows"])
    phone_train = torch.from_numpy(train["phone"])
    y_train = torch.from_numpy(train["label"]).float()

    model = PhoneCNN()
    if init_encoder is not None:
        load_encoder(model, init_encoder)
    model = model.to(dev)
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


def fit_spec_normalizer(spec: np.ndarray, max_rows: int = 100_000, seed: int = 0) -> Normalizer:
    """Per-mel-bin statistics from (N, 64, T) spectrograms (a random subset, to bound memory)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(spec), size=min(max_rows, len(spec)), replace=False)
    sub = spec[np.sort(idx)].astype(np.float32)
    mean, std = sub.mean(axis=(0, 2)), sub.std(axis=(0, 2))
    std[std < 1e-6] = 1.0
    return Normalizer(mean, std)


def _phone_batch(a: dict, idx: np.ndarray, norm: Normalizer) -> torch.Tensor:
    return norm(pretrain.expand_windows(a["spec"][idx], a["lo"][idx], a["hi"][idx]))


@torch.no_grad()
def phone_accuracy(model: PhoneClassifier, norm: Normalizer, a: dict, idx: np.ndarray,
                   dev: torch.device, batch_size: int = 2048) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        pred = model(_phone_batch(a, b, norm).to(dev)).argmax(1).cpu().numpy()
        correct += int((pred == a["phone"][b]).sum())
    return correct / max(len(idx), 1)


def train_phone_classifier(a: dict, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3,
                           patience: int = 2, seed: int = 0, device: Optional[str] = None):
    """Pre-train the encoder as a phone classifier; early-stop on held-out phone accuracy.

    `a` holds compact windows (spec, lo, hi), phone labels and a boolean `heldout`.
    Returns (model, normalizer, history).
    """
    dev = _pick_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_idx = np.flatnonzero(~a["heldout"])
    val_idx = np.flatnonzero(a["heldout"])
    norm = fit_spec_normalizer(a["spec"][train_idx], seed=seed)

    model = PhoneClassifier().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "val_acc": [], "best_epoch": 0}
    best_state, stale = None, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = np.sort(order[i : i + batch_size])
            x = _phone_batch(a, idx, norm).to(dev)
            y = torch.from_numpy(a["phone"][idx]).to(dev)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        hist["train_loss"].append(total / len(order))
        hist["val_acc"].append(phone_accuracy(model, norm, a, val_idx, dev))
        print(f"epoch {epoch + 1}: loss {hist['train_loss'][-1]:.3f}, held-out acc {hist['val_acc'][-1]:.3f}",
              flush=True)

        if hist["val_acc"][-1] > max(hist["val_acc"][:-1], default=-1.0):
            best_state, stale = copy.deepcopy(model.state_dict()), 0
            hist["best_epoch"] = epoch
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model.to(dev), norm, hist
