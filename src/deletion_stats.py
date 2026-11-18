"""How long are the annotated segments of deleted phones?

A deleted phone has no sound of its own, so its annotated interval can be very
short or empty. phone_window() still marks at least one 10 ms frame in the mask
(see features._frame_span), so every deletion gets a usable input.

Usage: python -m src.deletion_stats [data/l2arctic]
"""
import json
import sys
from pathlib import Path

import numpy as np

from src import dataset
from src.textgrid import read_phone_tier

FRAME_MS = 10.0


def summarize_durations(segments) -> dict:
    """Duration summary (ms) of the deletion segments among `segments`."""
    ms = np.array([(s.end - s.start) * 1000.0 for s in segments if s.error == "d"])
    if len(ms) == 0:
        return {"n": 0, "zero_length": 0, "under_one_frame": 0, "median_ms": None}
    return {
        "n": int(len(ms)),
        "zero_length": int((ms <= 1e-6).sum()),
        "under_one_frame": int((ms < FRAME_MS).sum()),
        "median_ms": float(np.median(ms)),
        "p05_ms": float(np.percentile(ms, 5)),
        "p95_ms": float(np.percentile(ms, 95)),
    }


def main(root: Path) -> dict:
    out = {}
    for name, speakers in dataset.speaker_split().items():
        segs = [s for spk in speakers for p in sorted((root / spk / "annotation").glob("*.TextGrid"))
                for s in read_phone_tier(p)]
        out[name] = summarize_durations(segs)
    out["all"] = summarize_durations(
        [s for spk in dataset.L1 for p in sorted((root / spk / "annotation").glob("*.TextGrid"))
         for s in read_phone_tier(p)]
    )
    return out


if __name__ == "__main__":
    result = main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/l2arctic"))
    print(json.dumps(result, indent=2))
    Path("results/deletion_stats.json").write_text(json.dumps(result, indent=2) + "\n")
