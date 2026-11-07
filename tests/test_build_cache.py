"""Turning a speaker list into model-ready arrays."""
import numpy as np
import pytest
import soundfile as sf

from src import build_cache, dataset
from tests.test_dataset import TEXTGRID


@pytest.fixture
def corpus(tmp_path):
    spk = tmp_path / "YKWK"
    (spk / "annotation").mkdir(parents=True)
    (spk / "wav").mkdir()
    (spk / "annotation" / "arctic_a0001.TextGrid").write_text(TEXTGRID)
    t = np.arange(44100) / 44100
    sf.write(spk / "wav" / "arctic_a0001.wav", (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), 44100)
    return tmp_path


def test_arrays_have_expected_shapes_and_dtypes(corpus):
    a = build_cache.build_arrays(corpus, ["YKWK"])

    assert a["windows"].shape == (3, 2, 64, 30) and a["windows"].dtype == np.float16
    assert a["mfcc"].shape == (3, 78) and a["mfcc"].dtype == np.float32
    assert a["phone"].tolist() == [dataset.PHONE_INDEX[p] for p in ("V", "EH", "R")]
    assert a["label"].tolist() == [1, 0, 1]
    assert a["speaker"].tolist() == ["YKWK"] * 3


def test_window_mask_channel_covers_the_phone(corpus):
    a = build_cache.build_arrays(corpus, ["YKWK"])

    # first phone V is 0.1-0.3 s = 20 frames
    assert a["windows"][0, 1, 0].sum() == 20


def test_all_values_are_finite(corpus):
    a = build_cache.build_arrays(corpus, ["YKWK"])

    assert np.isfinite(a["windows"].astype(np.float32)).all()
    assert np.isfinite(a["mfcc"]).all()


def test_arrays_for_empty_speaker_list_are_empty():
    a = build_cache.build_arrays("unused", [])

    assert a["windows"].shape == (0, 2, 64, 30) and a["label"].shape == (0,)


def test_save_and_load_round_trip(corpus, tmp_path):
    a = build_cache.build_arrays(corpus, ["YKWK"])
    out = tmp_path / "split.npz"

    build_cache.save_arrays(a, out)
    b = build_cache.load_arrays(out)

    assert all(np.array_equal(a[k], b[k]) for k in a)
