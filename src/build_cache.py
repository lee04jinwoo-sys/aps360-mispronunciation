"""Build model-ready arrays (log-mel windows and MFCC statistics) per split.

Usage: python -m src.build_cache [data/l2arctic] [data/cache]
"""
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

from src import dataset, features


def build_arrays(root: str | Path, speakers: list[str]) -> dict[str, np.ndarray]:
    """Features and labels for every labelled phone of the given speakers."""
    root = Path(root)
    windows, mfcc, phone, label, speaker, utt = [], [], [], [], [], []
    for spk in speakers:
        examples = dataset.examples_for_speaker(root, spk)
        for utterance, group in groupby(examples, key=lambda e: e.utterance):
            wave = features.load_audio(root / spk / "wav" / f"{utterance}.wav")
            mel = features.log_mel(wave)
            frames = features.mfcc_frames(wave)
            for e in group:
                windows.append(features.phone_window(mel, e.start, e.end).astype(np.float16))
                mfcc.append(features.mfcc_stats_from_frames(frames, e.start, e.end))
                phone.append(e.phone_idx)
                label.append(e.label)
                speaker.append(e.speaker)
                utt.append(e.utterance)
    n_mel, n_frames = features.N_MELS, features.WINDOW_FRAMES
    return {
        "windows": np.array(windows, dtype=np.float16).reshape(-1, 2, n_mel, n_frames),
        "mfcc": np.array(mfcc, dtype=np.float32).reshape(-1, 78),
        "phone": np.array(phone, dtype=np.int64),
        "label": np.array(label, dtype=np.int64),
        "speaker": np.array(speaker, dtype="<U8"),
        "utterance": np.array(utt, dtype="<U16"),
    }


def save_arrays(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    np.savez(path, **arrays)


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, speakers in dataset.speaker_split().items():
        arrays = build_arrays(root, speakers)
        save_arrays(arrays, out_dir / f"{name}.npz")
        n, err = len(arrays["label"]), int(arrays["label"].sum())
        print(f"{name}: {n} phones, {err} mispronounced ({100 * err / n:.1f}%) -> {out_dir / f'{name}.npz'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/l2arctic"),
         Path(sys.argv[2] if len(sys.argv) > 2 else "data/cache"))
