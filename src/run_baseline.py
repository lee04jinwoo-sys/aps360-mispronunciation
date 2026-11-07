"""Train the MFCC + logistic-regression baseline and write results/baseline.json.

The decision threshold is picked on the validation set only; the test set is
scored once.   Usage: python -m src.run_baseline [data/cache]
"""
import json
import sys
from pathlib import Path

import numpy as np

from src import baseline, build_cache, metrics


def main(cache: Path, out: Path) -> dict:
    train = build_cache.load_arrays(cache / "train.npz")
    val = build_cache.load_arrays(cache / "val.npz")
    test = build_cache.load_arrays(cache / "test.npz")

    model, scaler = baseline.train(train)
    threshold = metrics.best_threshold(val["label"], baseline.predict_scores(model, scaler, val))

    scores = baseline.predict_scores(model, scaler, test)
    pred = (scores >= threshold).astype(int)
    y = test["label"]
    korean = test["speaker"] == "YKWK"

    result = {
        "model": "MFCC statistics (78) + one-hot expected phone (40) -> logistic regression, class_weight=balanced",
        "threshold_from_val": threshold,
        "test_overall": metrics.prf(y, pred),
        "test_korean_YKWK": metrics.prf(y[korean], pred[korean]),
        "test_by_l1": metrics.prf_by_l1(y, pred, test["speaker"]),
        "reference_always_mispronounced": metrics.prf(y, np.ones_like(y)),
        "n_test": int(len(y)),
        "test_error_rate": float(y.mean()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/cache"), Path("results/baseline.json"))
    print(json.dumps(r, indent=2))
