"""Acoustic features: 16 kHz audio, log-mel spectrograms, phone windows, MFCC statistics.

All frame arithmetic assumes a 10 ms hop (100 frames per second).
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 16_000
N_FFT = 512
WIN_LENGTH = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms
N_MELS = 64
WINDOW_FRAMES = 30  # 300 ms around the phone
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 100
N_MFCC = 13

_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH, n_mels=N_MELS, center=True,
)
_MFCC = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
    melkwargs=dict(n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS),
)


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read a wav file as mono float32 at `sr` Hz."""
    data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(data.mean(axis=1))
    if orig_sr != sr:
        wave = torchaudio.functional.resample(wave, orig_sr, sr)
    return wave.numpy().astype(np.float32)


def log_mel(wave: np.ndarray) -> np.ndarray:
    """(N_MELS, T) natural-log mel power spectrogram; T = 1 + len(wave) // HOP_LENGTH."""
    with torch.no_grad():
        mel = _MEL(torch.from_numpy(np.ascontiguousarray(wave)))
    return torch.log(mel + 1e-6).numpy().astype(np.float32)


def _frame_span(start: float, end: float) -> tuple[int, int]:
    start_f = int(round(start * FRAMES_PER_SECOND))
    end_f = max(start_f + 1, int(round(end * FRAMES_PER_SECOND)))
    return start_f, end_f


def phone_window(mel: np.ndarray, start: float, end: float, frames: int = WINDOW_FRAMES) -> np.ndarray:
    """Crop `frames` mel frames centred on the phone and add a mask channel.

    Returns (2, N_MELS, frames): channel 0 is the log-mel crop, channel 1 is 1.0
    on the frames the phone itself covers and 0.0 on its context. Frames outside
    the utterance are filled with the utterance's minimum value (silence-like).
    """
    start_f, end_f = _frame_span(start, end)
    left = (start_f + end_f) // 2 - frames // 2
    window = np.full((mel.shape[0], frames), mel.min(), dtype=np.float32)
    src_lo, src_hi = max(left, 0), min(left + frames, mel.shape[1])
    if src_hi > src_lo:
        window[:, src_lo - left : src_hi - left] = mel[:, src_lo:src_hi]
    mask = np.zeros((mel.shape[0], frames), dtype=np.float32)
    mask_lo, mask_hi = max(start_f, left), min(end_f, left + frames)
    if mask_hi > mask_lo:
        mask[:, mask_lo - left : mask_hi - left] = 1.0
    return np.stack([window, mask])


def mfcc_frames(wave: np.ndarray) -> np.ndarray:
    """(39, T): 13 MFCC + delta + delta-delta per 10 ms frame."""
    with torch.no_grad():
        mfcc = _MFCC(torch.from_numpy(np.ascontiguousarray(wave)))
        d1 = torchaudio.functional.compute_deltas(mfcc)
        d2 = torchaudio.functional.compute_deltas(d1)
    return torch.cat([mfcc, d1, d2], dim=0).numpy().astype(np.float32)


def mfcc_stats_from_frames(frames: np.ndarray, start: float, end: float) -> np.ndarray:
    """Mean and std over the phone's frames of each of the 39 features -> (78,)."""
    start_f, end_f = _frame_span(start, end)
    last = frames.shape[1] - 1
    start_f = min(max(start_f, 0), last)
    end_f = min(max(end_f, start_f + 1), frames.shape[1])
    seg = frames[:, start_f:end_f]
    return np.concatenate([seg.mean(axis=1), seg.std(axis=1)]).astype(np.float32)


def mfcc_stats(wave: np.ndarray, start: float, end: float) -> np.ndarray:
    """Convenience wrapper: MFCC statistics of one phone."""
    return mfcc_stats_from_frames(mfcc_frames(wave), start, end)
