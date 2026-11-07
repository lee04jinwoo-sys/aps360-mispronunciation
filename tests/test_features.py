"""Acoustic feature extraction: audio loading, log-mel, phone windows, MFCC stats."""
import numpy as np
import pytest
import soundfile as sf

from src import features


def tone(seconds: float, sr: int, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- load_audio -------------------------------------------------------------


def test_load_audio_resamples_44k_to_16k(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, tone(1.0, 44100), 44100)

    wave = features.load_audio(path)

    assert wave.dtype == np.float32
    assert wave.ndim == 1
    assert abs(len(wave) - 16000) <= 2


def test_load_audio_mixes_stereo_to_mono(tmp_path):
    path = tmp_path / "s.wav"
    stereo = np.stack([tone(0.5, 16000), tone(0.5, 16000)], axis=1)
    sf.write(path, stereo, 16000)

    assert features.load_audio(path).ndim == 1


# --- log_mel ----------------------------------------------------------------


def test_log_mel_shape_is_64_bins_by_10ms_frames():
    mel = features.log_mel(tone(1.0, 16000))

    assert mel.shape == (64, 1 + 16000 // 160)


def test_log_mel_is_finite_for_silence():
    mel = features.log_mel(np.zeros(16000, dtype=np.float32))

    assert np.isfinite(mel).all()


# --- phone_window -----------------------------------------------------------


def ramp_mel(frames: int = 100) -> np.ndarray:
    """mel[:, t] == t, so a crop reveals exactly which frames were taken."""
    return np.tile(np.arange(frames, dtype=np.float32), (64, 1))


def test_phone_window_shape():
    w = features.phone_window(ramp_mel(), start=0.50, end=0.60)

    assert w.shape == (2, 64, 30)


def test_phone_window_is_centred_on_the_phone():
    # phone = frames 50..59, centre frame 55 -> window frames 40..69
    w = features.phone_window(ramp_mel(), start=0.50, end=0.60)

    assert w[0, 0, 0] == 40
    assert w[0, 0, -1] == 69


def test_phone_window_mask_marks_only_phone_frames():
    w = features.phone_window(ramp_mel(), start=0.50, end=0.60)

    mask = w[1, 0]
    assert mask.sum() == 10
    assert mask[10:20].all() and not mask[:10].any() and not mask[20:].any()


def test_phone_window_pads_before_utterance_start_with_floor_value():
    mel = ramp_mel() + 5.0  # min is 5.0
    w = features.phone_window(mel, start=0.0, end=0.05)

    assert (w[0, :, :10] == 5.0).all()  # padded region = utterance minimum
    assert w[1, 0].sum() == 5  # mask still covers the real phone


def test_phone_window_pads_after_utterance_end():
    mel = ramp_mel(100)
    w = features.phone_window(mel, start=0.95, end=1.00)

    assert w.shape == (2, 64, 30)
    assert (w[0, :, -10:] == mel.min()).all()


def test_phone_window_clips_mask_for_phone_longer_than_window():
    w = features.phone_window(ramp_mel(200), start=0.20, end=1.00)  # 80 frames

    assert w[1, 0].sum() == 30


def test_phone_window_never_mutates_input():
    mel = ramp_mel()
    before = mel.copy()

    features.phone_window(mel, start=0.0, end=0.05)

    assert np.array_equal(mel, before)


# --- mfcc_stats -------------------------------------------------------------


def test_mfcc_stats_has_78_features():
    # 13 MFCC + 13 delta + 13 delta-delta = 39 per frame; mean and std -> 78
    stats = features.mfcc_stats(tone(1.0, 16000), start=0.40, end=0.60)

    assert stats.shape == (78,)
    assert np.isfinite(stats).all()


def test_mfcc_stats_differ_for_different_sounds():
    sr = 16000
    low = features.mfcc_stats(tone(1.0, sr, 200), 0.4, 0.6)
    high = features.mfcc_stats(tone(1.0, sr, 3000), 0.4, 0.6)

    assert not np.allclose(low, high)


def test_mfcc_stats_tolerates_phone_shorter_than_one_frame():
    stats = features.mfcc_stats(tone(1.0, 16000), start=0.500, end=0.504)

    assert stats.shape == (78,) and np.isfinite(stats).all()
