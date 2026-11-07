"""Checks against the real L2-ARCTIC corpus (skipped when data/ is absent).

Expected numbers are the ones stated in the project proposal, written before
the dataset code existed.
"""
from collections import Counter
from pathlib import Path

import pytest

from src import dataset

ROOT = Path("data/l2arctic")
pytestmark = pytest.mark.skipif(not ROOT.is_dir(), reason="L2-ARCTIC not downloaded")


@pytest.fixture(scope="module")
def by_split():
    return {
        name: [e for s in speakers for e in dataset.examples_for_speaker(ROOT, s)]
        for name, speakers in dataset.speaker_split().items()
    }


def test_every_real_phone_is_in_the_inventory(by_split):
    # examples_for_speaker raises KeyError on an unknown phone, so reaching here is the check
    assert sum(len(v) for v in by_split.values()) > 100_000


def test_test_set_matches_proposal(by_split):
    test = by_split["test"]
    errors = sum(e.label for e in test)

    assert len(test) == 29_786
    assert round(100 * errors / len(test), 1) == 13.9


def test_korean_test_speaker_is_in_test_set(by_split):
    assert any(e.speaker == "YKWK" for e in by_split["test"])


def test_no_speaker_appears_in_two_splits(by_split):
    owners = {}
    for name, examples in by_split.items():
        for s in {e.speaker for e in examples}:
            assert owners.setdefault(s, name) == name


def test_total_labelled_phones_is_all_minus_additions(by_split):
    assert sum(len(v) for v in by_split.values()) == 120_243 - 1_092
