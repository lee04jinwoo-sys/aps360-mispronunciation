"""Phone-recognition pre-training data from unannotated L2-ARCTIC utterances.

Only the 18 non-test speakers are used, and only utterances that have no manual
annotation, so neither the test speakers nor any detection label is seen.
Labels are the forced-alignment phones. To keep the cache small, each window is
stored without its mask channel; the mask is rebuilt from (lo, hi) frame bounds.

Usage: python -m src.pretrain [data/l2arctic] [data/cache]
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

from src import dataset, features
from src.data_stats import L1
from src.textgrid import read_phone_tier

HELDOUT_EVERY = 20  # about 5% of utterances early-stop pre-training


def pretrain_speakers() -> list[str]:
    return sorted(s for s in L1 if s not in dataset.TEST_SPEAKERS)


def unannotated_textgrids(root: str | Path, speaker: str) -> list[Path]:
    """Forced-alignment TextGrids of utterances that have no manual annotation."""
    root = Path(root)
    annotated = {p.stem for p in (root / speaker / "annotation").glob("*.TextGrid")}
    return [p for p in sorted((root / speaker / "textgrid").glob("*.TextGrid")) if p.stem not in annotated]


def phone_segments(path: Path) -> list[tuple[int, float, float]]:
    """(phone index, start, end) for every non-silence phone in the 40-phone inventory."""
    return [(dataset.PHONE_INDEX[s.canonical], s.start, s.end)
            for s in read_phone_tier(path) if s.canonical in dataset.PHONE_INDEX]


def is_heldout(speaker: str, utterance: str) -> bool:
    digest = hashlib.md5(f"{speaker}/{utterance}".encode()).hexdigest()
    return int(digest, 16) % HELDOUT_EVERY == 0


def compact_window(mel: np.ndarray, start: float, end: float) -> tuple[np.ndarray, int, int]:
    """The log-mel channel of phone_window() plus the [lo, hi) frames its mask marks."""
    win = features.phone_window(mel, start, end)
    cols = np.flatnonzero(win[1, 0] > 0)
    lo, hi = (int(cols[0]), int(cols[-1]) + 1) if len(cols) else (0, 0)
    return win[0], lo, hi


def expand_windows(spec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(N, 64, T) spectrograms + mask bounds -> (N, 2, 64, T) float32, as phone_window() builds them."""
    n, n_mel, t = spec.shape
    out = np.zeros((n, 2, n_mel, t), dtype=np.float32)
    out[:, 0] = spec
    frames = np.arange(t)[None, :]
    mask = (frames >= np.asarray(lo)[:, None]) & (frames < np.asarray(hi)[:, None])
    out[:, 1] = mask[:, None, :]
    return out


def build_arrays(root: str | Path, speakers: list[str]) -> dict[str, np.ndarray]:
    root = Path(root)
    parts = []
    for spk in speakers:
        spec, lo, hi, phone, utt, held = [], [], [], [], [], []
        for path in unannotated_textgrids(root, spk):
            segs = phone_segments(path)
            if not segs:
                continue
            mel = features.log_mel(features.load_audio(root / spk / "wav" / f"{path.stem}.wav"))
            h = is_heldout(spk, path.stem)
            for idx, start, end in segs:
                s, a, b = compact_window(mel, start, end)
                spec.append(s.astype(np.float16)); lo.append(a); hi.append(b)
                phone.append(idx); utt.append(path.stem); held.append(h)
        parts.append({
            "spec": np.stack(spec), "lo": np.array(lo, np.int8), "hi": np.array(hi, np.int8),
            "phone": np.array(phone, np.int64), "speaker": np.array([spk] * len(phone), "<U8"),
            "utterance": np.array(utt, "<U16"), "heldout": np.array(held, bool),
        })
        print(f"{spk}: {len(phone)} phones", flush=True)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main(root: Path, out_dir: Path) -> None:
    start = time.time()
    arrays = build_arrays(root, pretrain_speakers())
    assert not set(arrays["speaker"]) & set(dataset.TEST_SPEAKERS)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "pretrain.npz", **arrays)
    n, held = len(arrays["phone"]), int(arrays["heldout"].sum())
    print(f"{n} phones from {len(set(zip(arrays['speaker'], arrays['utterance'])))} utterances, "
          f"{held} held out, {(time.time() - start) / 60:.1f} min -> {out_dir / 'pretrain.npz'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/l2arctic"),
         Path(sys.argv[2] if len(sys.argv) > 2 else "data/cache"))
