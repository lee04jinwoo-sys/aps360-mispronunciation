"""Duration summary of deleted-phone segments and their window/mask."""
import numpy as np

from src import features
from src.deletion_stats import summarize_durations
from src.textgrid import PhoneSegment


def seg(start, end, error="d"):
    return PhoneSegment(start, end, "T", "", error)


def test_summary_counts_only_deletions():
    segs = [seg(0.0, 0.0), seg(0.0, 0.005), seg(0.1, 0.2), seg(0.0, 0.0, error="correct")]

    s = summarize_durations(segs)

    assert s["n"] == 3
    assert s["zero_length"] == 1
    assert s["under_one_frame"] == 2  # 0 ms and 5 ms; one frame is 10 ms
    assert abs(s["median_ms"] - 5.0) < 1e-9


def test_summary_of_no_deletions_is_empty():
    s = summarize_durations([seg(0.0, 0.1, error="correct")])

    assert s["n"] == 0
    assert s["median_ms"] is None


def test_zero_length_segment_still_gets_one_mask_frame():
    mel = np.zeros((features.N_MELS, 100), dtype=np.float32)

    win = features.phone_window(mel, 0.50, 0.50)

    assert win[1].sum(axis=1).max() == 1.0  # exactly one 10 ms frame marked
