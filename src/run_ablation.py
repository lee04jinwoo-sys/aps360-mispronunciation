"""Pre-training ablation: scratch CNN vs pre-trained CNN, seeds 0-4 (results/pretrain_protocol.md).

Rebuilds test predictions of both model families from their checkpoints, checks
them against the saved test F1, and writes results/pretrain_ablation.json.
Usage: python -m src.run_ablation [data/l2arctic] [data/cache]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src import build_cache, cnn, dataset, error_analysis, metrics

SEEDS = [0, 1, 2, 3, 4]
FOCUS_PHONES = ["DH", "TH", "Z"]


def compare(scratch: list[float], pretrained: list[float]) -> dict:
    """Paired comparison by seed; the verdict is the protocol's primary rule (higher mean F1)."""
    if len(scratch) != len(pretrained):
        raise ValueError("seed lists differ in length")
    s, p = np.array(scratch), np.array(pretrained)
    return {
        "scratch_mean": float(s.mean()), "scratch_sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "pretrained_mean": float(p.mean()), "pretrained_sd": float(p.std(ddof=1)) if len(p) > 1 else 0.0,
        "mean_diff": float(p.mean() - s.mean()),
        "paired_diffs": (p - s).tolist(),
        "paired_wins": int((p > s).sum()),
        "pretraining_helps": bool(p.mean() > s.mean()),
    }


def _predictions(test, cache, tag, norm_for):
    preds = []
    for seed in SEEDS:
        saved = json.loads(Path(f"results/{tag}_seed{seed}.json").read_text())
        net = cnn.PhoneCNN()
        net.load_state_dict(torch.load(cache / f"{tag}_seed{seed}.pt", map_location="cpu"))
        pred = (cnn.predict_scores(net, norm_for, test) >= saved["threshold_from_val"]).astype(int)
        gap = abs(metrics.prf(test["label"], pred)["f1"] - saved["test_overall"]["f1"])
        assert gap < 2e-3, f"{tag} seed {seed} not reproduced (F1 differs by {gap:.4f})"
        preds.append(pred)
    return preds


def main(root: Path, cache: Path) -> dict:
    train = build_cache.load_arrays(cache / "train.npz")
    test = build_cache.load_arrays(cache / "test.npz")
    y = test["label"]
    korean = test["speaker"] == "YKWK"

    scratch_norm = cnn.fit_normalizer(train["windows"])
    with np.load(cache / "pretrain_norm.npz") as f:
        pt_norm = cnn.Normalizer(f["mean"], f["std"])

    families = {"scratch": _predictions(test, cache, "cnn", scratch_norm),
                "pretrained": _predictions(test, cache, "cnn_pt", pt_norm)}

    f1 = {k: [metrics.prf(y, p)["f1"] for p in v] for k, v in families.items()}
    f1_kr = {k: [metrics.prf(y[korean], p[korean])["f1"] for p in v] for k, v in families.items()}

    details = error_analysis.annotation_details(root, dataset.speaker_split()["test"])
    assert np.array_equal(details["label"], y)
    phones = {}
    for ph in FOCUS_PHONES:
        m = test["phone"] == dataset.PHONE_INDEX[ph]
        rate = float(y[m].mean())
        phones[ph] = {
            "n": int(m.sum()), "error_rate": rate,
            "always_positive_f1": 2 * rate / (1 + rate),
            **{f"{k}_f1_mean": float(np.mean([metrics.prf(y[m], p[m])["f1"] for p in v])) for k, v in families.items()},
            **{f"{k}_recall_mean": float(np.mean([metrics.prf(y[m], p[m])["recall"] for p in v])) for k, v in families.items()},
            **{f"{k}_precision_mean": float(np.mean([metrics.prf(y[m], p[m])["precision"] for p in v])) for k, v in families.items()},
        }

    result = {
        "primary_test_f1": compare(f1["scratch"], f1["pretrained"]),
        "per_seed_test_f1": f1,
        "secondary_korean_f1": compare(f1_kr["scratch"], f1_kr["pretrained"]),
        "secondary_focus_phones": phones,
        "pretrained_precision_recall_mean": {
            k: float(np.mean([metrics.prf(y, p)[k] for p in families["pretrained"]])) for k in ("precision", "recall")},
        "reproduction_check": "both families rebuilt from checkpoints, within 2e-3 of saved test F1",
    }
    Path("results/pretrain_ablation.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/l2arctic"),
             Path(sys.argv[2] if len(sys.argv) > 2 else "data/cache"))
    print(json.dumps(r, indent=2))
