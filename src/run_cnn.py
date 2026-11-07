"""Train the CNN from scratch and write results/cnn.json and results/cnn_history.json.

Early stopping and the decision threshold use the validation speakers only; the
test speakers are scored once at the end.   Usage: python -m src.run_cnn [data/cache] [seed]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src import build_cache, cnn, metrics


def main(cache: Path, seed: int = 0) -> dict:
    train = build_cache.load_arrays(cache / "train.npz")
    val = build_cache.load_arrays(cache / "val.npz")
    test = build_cache.load_arrays(cache / "test.npz")

    start = time.time()
    model, norm, hist = cnn.train_model(train, val, seed=seed)
    minutes = (time.time() - start) / 60
    torch.save(model.state_dict(), cache / f"cnn_seed{seed}.pt")

    scores = cnn.predict_scores(model, norm, test)
    pred = (scores >= hist["threshold"]).astype(int)
    y = test["label"]
    korean = test["speaker"] == "YKWK"
    result = {
        "model": "PhoneCNN (4 conv blocks, GAP, 16-d phone embedding, MLP head), trained from scratch",
        "seed": seed,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "train_minutes": round(minutes, 1),
        "parameters": sum(p.numel() for p in cnn.PhoneCNN().parameters()),
        "epochs_run": len(hist["val_f1"]),
        "best_epoch": hist["best_epoch"],
        "threshold_from_val": hist["threshold"],
        "val_f1_best": max(hist["val_f1"]),
        "test_overall": metrics.prf(y, pred),
        "test_korean_YKWK": metrics.prf(y[korean], pred[korean]),
        "test_by_l1": metrics.prf_by_l1(y, pred, test["speaker"]),
        "n_test": int(len(y)),
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/cnn_seed{seed}.json").write_text(json.dumps(result, indent=2))
    Path(f"results/cnn_history_seed{seed}.json").write_text(json.dumps(hist, indent=2))
    return result


if __name__ == "__main__":
    r = main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/cache"),
             int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print(json.dumps(r, indent=2))
