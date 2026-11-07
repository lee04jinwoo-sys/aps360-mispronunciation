"""Labels, phone inventory, speaker split and example listing."""
from pathlib import Path

import pytest

from src import dataset
from src.data_stats import L1
from src.textgrid import PhoneSegment

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 5
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "sil"
        intervals [2]:
            xmin = 0.1
            xmax = 0.3
            text = "V,B,s"
        intervals [3]:
            xmin = 0.3
            xmax = 0.5
            text = "EH1"
        intervals [4]:
            xmin = 0.5
            xmax = 0.6
            text = "sil,AH,a"
        intervals [5]:
            xmin = 0.6
            xmax = 0.8
            text = "R,sil,d"
'''


def seg(error="correct", canonical="AH"):
    return PhoneSegment(0.0, 0.1, canonical, canonical, error)


# --- labels -----------------------------------------------------------------


@pytest.mark.parametrize("error, expected", [("correct", 0), ("s", 1), ("d", 1)])
def test_segment_label(error, expected):
    assert dataset.segment_label(seg(error)) == expected


def test_additions_have_no_expected_phone_so_no_label():
    assert dataset.segment_label(seg("a")) is None


# --- phone inventory --------------------------------------------------------


def test_phone_inventory_is_40_clean_arpabet_symbols():
    assert len(dataset.PHONES) == 40
    assert len(set(dataset.PHONES)) == 40
    assert all(p.isalpha() and p.isupper() for p in dataset.PHONES)


def test_phone_index_round_trips():
    for i, p in enumerate(dataset.PHONES):
        assert dataset.PHONE_INDEX[p] == i


# --- speaker split ----------------------------------------------------------


def test_test_set_is_the_standard_six_speakers():
    split = dataset.speaker_split()

    assert set(split["test"]) == {"NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"}


def test_splits_are_disjoint_and_cover_all_24_speakers():
    split = dataset.speaker_split()
    all_speakers = split["train"] + split["val"] + split["test"]

    assert len(all_speakers) == 24
    assert set(all_speakers) == set(L1)


def test_sizes_are_12_6_6():
    split = dataset.speaker_split()

    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (12, 6, 6)


@pytest.mark.parametrize("name", ["val", "test"])
def test_val_and_test_have_one_speaker_per_l1(name):
    l1s = [L1[s] for s in dataset.speaker_split()[name]]

    assert sorted(l1s) == sorted(set(l1s)) and len(set(l1s)) == 6


# --- example listing --------------------------------------------------------


@pytest.fixture
def corpus(tmp_path):
    ann = tmp_path / "YKWK" / "annotation"
    ann.mkdir(parents=True)
    (ann / "arctic_a0001.TextGrid").write_text(TEXTGRID)
    return tmp_path


def test_examples_skip_silence_and_additions(corpus):
    examples = dataset.examples_for_speaker(corpus, "YKWK")

    assert [(e.phone, e.label) for e in examples] == [("V", 1), ("EH", 0), ("R", 1)]


def test_example_carries_timing_and_ids(corpus):
    first = dataset.examples_for_speaker(corpus, "YKWK")[0]

    assert first.speaker == "YKWK"
    assert first.utterance == "arctic_a0001"
    assert (first.start, first.end) == (0.1, 0.3)


def test_missing_speaker_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.examples_for_speaker(tmp_path, "NOPE")
