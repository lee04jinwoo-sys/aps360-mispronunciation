"""Breakdowns of detection performance by phone, error type and confusion pair.

The cached arrays only hold the binary label.  `annotation_details` re-reads the
TextGrids (no audio) in exactly the order `build_cache` used, so the extra
columns line up row for row with the cache.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from src import dataset, metrics
from src.data_stats import L1
from src.textgrid import read_phone_tier


def annotation_details(root: str | Path, speakers: list[str]) -> dict[str, np.ndarray]:
    """Canonical phone, perceived phone, error type and label for each labelled phone."""
    canonical, perceived, error, label = [], [], [], []
    for spk in speakers:
        ann_dir = Path(root) / spk / "annotation"
        if not ann_dir.is_dir():
            raise FileNotFoundError(ann_dir)
        for path in sorted(ann_dir.glob("*.TextGrid")):
            for seg in read_phone_tier(path):
                y = dataset.segment_label(seg)
                if y is None:
                    continue
                canonical.append(seg.canonical)
                perceived.append(seg.perceived)
                error.append(seg.error)
                label.append(y)
    return {
        "canonical": np.array(canonical, dtype="<U8"),
        "perceived": np.array(perceived, dtype="<U8"),
        "error": np.array(error, dtype="<U8"),
        "label": np.array(label, dtype=np.int64),
    }


def per_phone_table(phone: np.ndarray, y: np.ndarray, pred: np.ndarray) -> list[dict]:
    """One row per phone that occurs, most error-prone first."""
    rows = []
    for idx, name in enumerate(dataset.PHONES):
        mask = phone == idx
        if not mask.any():
            continue
        rows.append(
            {"phone": name, "n": int(mask.sum()), "error_rate": float(y[mask].mean()),
             **metrics.prf(y[mask], pred[mask])}
        )
    return sorted(rows, key=lambda r: (-r["error_rate"], r["phone"]))


def recall_by_error_type(error: np.ndarray, pred: np.ndarray) -> dict[str, dict]:
    """Fraction of substitutions ('s') and deletions ('d') the model flags."""
    out = {}
    for kind in ("s", "d"):
        mask = error == kind
        n = int(mask.sum())
        out[kind] = {"n": n, "recall": float(pred[mask].mean()) if n else 0.0}
    return out


def top_confusions(canonical: np.ndarray, perceived: np.ndarray, error: np.ndarray,
                   pred: np.ndarray, k: int = 10) -> list[dict]:
    """The k most frequent substitutions (expected -> heard) and how often each is flagged."""
    subs = np.flatnonzero(error == "s")
    counts = Counter(zip(canonical[subs].tolist(), perceived[subs].tolist()))
    rows = []
    for (exp, heard), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]:
        mask = (error == "s") & (canonical == exp) & (perceived == heard)
        rows.append({"expected": exp, "heard": heard, "n": n, "detected": float(pred[mask].mean())})
    return rows


def always_positive_f1_by_l1(y: np.ndarray, speakers: np.ndarray) -> dict[str, float]:
    """F1 of labelling every phone mispronounced, per L1: a reference for the base rate."""
    languages = np.array([L1[s] for s in speakers])
    out = {}
    for lang in sorted(set(languages)):
        p = float(y[languages == lang].mean())
        out[lang] = 2 * p / (1 + p)
    return out
