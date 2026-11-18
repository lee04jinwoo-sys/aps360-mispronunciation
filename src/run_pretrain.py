"""Pre-train the CNN encoder as a 40-way phone classifier (see results/pretrain_protocol.md).

Writes data/cache/pretrain_encoder.pt, data/cache/pretrain_norm.npz and results/pretrain.json.
Usage: python -m src.run_pretrain [data/cache]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src import cnn, dataset


def main(cache: Path, seed: int = 0) -> dict:
    with np.load(cache / "pretrain.npz", allow_pickle=False) as f:
        a = {k: f[k] for k in ("spec", "lo", "hi", "phone", "heldout", "speaker")}
    assert not set(a["speaker"]) & set(dataset.TEST_SPEAKERS), "test speaker in pre-training data"

    start = time.time()
    model, norm, hist = cnn.train_phone_classifier(a, seed=seed)
    minutes = (time.time() - start) / 60

    torch.save(model.encoder.state_dict(), cache / "pretrain_encoder.pt")
    np.savez(cache / "pretrain_norm.npz", mean=norm.mean, std=norm.std)

    counts = np.bincount(a["phone"][~a["heldout"]], minlength=len(dataset.PHONES))
    result = {
        "task": "40-way phone classification on forced-alignment segments, unannotated utterances, 18 non-test speakers",
        "seed": seed,
        "n_train": int((~a["heldout"]).sum()),
        "n_heldout": int(a["heldout"].sum()),
        "majority_class_acc": float(np.max(counts) / counts.sum()),
        "heldout_acc_best": hist["val_acc"][hist["best_epoch"]],
        "best_epoch": hist["best_epoch"],
        "epochs_run": len(hist["val_acc"]),
        "train_minutes": round(minutes, 1),
        "history": hist,
    }
    Path("results/pretrain.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/cache"))
    print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2))
