"""Paired seed comparison for the pre-training ablation."""
import pytest

from src.run_ablation import compare


def test_compare_reports_means_paired_wins_and_verdict():
    r = compare([0.40, 0.42, 0.44], [0.43, 0.41, 0.47])

    assert r["scratch_mean"] == pytest.approx(0.42)
    assert r["pretrained_mean"] == pytest.approx(0.43667, abs=1e-4)
    assert r["paired_wins"] == 2
    assert r["mean_diff"] == pytest.approx(0.01667, abs=1e-4)
    assert r["pretraining_helps"] is True


def test_equal_means_do_not_count_as_help():
    r = compare([0.4, 0.5], [0.5, 0.4])

    assert r["pretraining_helps"] is False


def test_seed_lists_must_pair_up():
    with pytest.raises(ValueError):
        compare([0.4, 0.5], [0.4])
