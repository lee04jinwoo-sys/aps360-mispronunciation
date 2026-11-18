"""Write PNG images of two cleaned training samples for the progress report.

Picks, from a Korean training speaker, one DH that was pronounced correctly and one
that the annotators heard as D, and saves each model input (log-mel channel with
the phone's frames outlined by the mask) as progress/example_<n>.png.
Only the standard library and numpy are used (no plotting dependency).

Usage: python -m src.plot_example [data/l2arctic] [data/cache]
"""
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

from src import build_cache, dataset, error_analysis

SPEAKER = "HKK"  # Korean, training split
SCALE = 6  # pixels per spectrogram cell
# Anchor colours of a viridis-like map (dark purple -> yellow).
ANCHORS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)


def colormap(v: np.ndarray) -> np.ndarray:
    """Values in [0, 1] -> uint8 RGB."""
    x = np.clip(v, 0, 1) * (len(ANCHORS) - 1)
    i = np.minimum(x.astype(int), len(ANCHORS) - 2)
    t = (x - i)[..., None]
    return ((1 - t) * ANCHORS[i] + t * ANCHORS[i + 1]).astype(np.uint8)


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def render(window: np.ndarray, lo_hi: tuple[float, float]) -> np.ndarray:
    """(2, 64, T) window -> RGB image, low frequencies at the bottom, mask frames outlined in white."""
    spec = window[0].astype(np.float32)
    v = (spec - lo_hi[0]) / (lo_hi[1] - lo_hi[0])
    img = colormap(v[::-1])  # flip so frequency increases upwards
    img = img.repeat(SCALE, 0).repeat(SCALE, 1)
    cols = np.flatnonzero(window[1, 0] > 0)
    a, b = cols[0] * SCALE, (cols[-1] + 1) * SCALE - 1
    img[:, [a, a + 1, b - 1, b]] = 255
    img[[0, 1, -2, -1], a : b + 1] = 255
    return img


def main(root: Path, cache: Path, out: Path) -> dict:
    train = build_cache.load_arrays(cache / "train.npz")
    details = error_analysis.annotation_details(root, dataset.speaker_split()["train"])
    assert np.array_equal(details["label"], train["label"]), "annotation rows do not line up with the cache"

    spk = train["speaker"] == SPEAKER
    dh = details["canonical"] == "DH"
    correct = np.flatnonzero(spk & dh & (details["error"] == "correct"))[0]
    wrong = np.flatnonzero(spk & dh & (details["error"] == "s") & (details["perceived"] == "D"))[0]

    pair = train["windows"][[correct, wrong]].astype(np.float32)
    lo_hi = (float(np.percentile(pair[:, 0], 2)), float(np.percentile(pair[:, 0], 98)))
    info = {}
    for n, i in enumerate([correct, wrong], start=1):
        write_png(out / f"example_{n}.png", render(train["windows"][i], lo_hi))
        info[f"example_{n}"] = {"row": int(i), "speaker": SPEAKER, "utterance": str(train["utterance"][i]),
                                "expected": "DH", "perceived": str(details["perceived"][i]),
                                "label": int(train["label"][i])}
    (out / "examples.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/l2arctic"),
                          Path(sys.argv[2] if len(sys.argv) > 2 else "data/cache"),
                          Path("progress")), indent=2))
