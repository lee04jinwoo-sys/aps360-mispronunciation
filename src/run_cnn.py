"""Train the CNN and write results/<tag>_seed<seed>.json and results/<tag>_history_seed<seed>.json.

Early stopping and the decision threshold use the validation speakers only; the
test speakers are scored once at the end.
Usage: python -m src.run_cnn [data/cache] [seed] [--pretrained]
  default: from scratch (tag "cnn"); --pretrained: encoder and input statistics
  from src.run_pretrain (tag "cnn_pt"), everything else identical.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src import build_cache, cnn, metrics


def main(cache: Path, seed: int = 0, pretrained: bool = False) -> dict:
    train = build_cache.load_arrays(cache / "train.npz")
    val = build_cache.load_arrays(cache / "val.npz")
    test = build_cache.load_arrays(cache / "test.npz")

    tag, init, norm = "cnn", None, None
    if pretrained:
        tag = "cnn_pt"
        init = torch.load(cache / "pretrain_encoder.pt", map_location="cpu")
        with np.load(cache / "pretrain_norm.npz") as f:
            norm = cnn.Normalizer(f["mean"], f["std"])

    start = time.time()
    model, norm, hist = cnn.train_model(train, val, seed=seed, init_encoder=init, norm=norm)
    minutes = (time.time() - start) / 60
    torch.save(model.state_dict(), cache / f"{tag}_seed{seed}.pt")

    scores = cnn.predict_scores(model, norm, test)
    pred = (scores >= hist["threshold"]).astype(int)
    y = test["label"]
    korean = test["speaker"] == "YKWK"
    result = {
        "model": "PhoneCNN (4 conv blocks, GAP, 16-d phone embedding, MLP head), "
                 + ("encoder pre-trained as phone classifier" if pretrained else "trained from scratch"),
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
    Path(f"results/{tag}_seed{seed}.json").write_text(json.dumps(result, indent=2))
    Path(f"results/{tag}_history_seed{seed}.json").write_text(json.dumps(hist, indent=2))
    return result


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--pretrained"]
    r = main(Path(args[0] if args else "data/cache"), int(args[1]) if len(args) > 1 else 0,
             pretrained="--pretrained" in sys.argv)
    print(json.dumps(r, indent=2))
