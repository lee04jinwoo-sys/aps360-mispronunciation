"""Tests for parsing L2-ARCTIC annotation TextGrids.

Phone-tier labels in the annotated subset are either a plain canonical phone
("AH0") or "CPL,PPL,type" where type is s (substitution), d (deletion) or
a (addition). Format assumed from the corpus README; checked against real
files in test_real_annotation_if_present.
"""
from pathlib import Path

import pytest

from src.textgrid import parse_phone_label, read_phone_tier

FIXTURE = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "very"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 5
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
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
            xmax = 0.7
            text = "R,sil,d"
        intervals [5]:
            xmin = 0.7
            xmax = 1.0
            text = "IY0 "
'''


def test_plain_phone_is_correct():
    assert parse_phone_label("EH1") == ("EH", "EH", "correct")


def test_substitution_strips_stress_and_case():
    assert parse_phone_label("v,b,s") == ("V", "B", "s")


def test_deletion():
    assert parse_phone_label("R,sil,d") == ("R", "SIL", "d")


def test_addition_has_no_canonical_phone():
    assert parse_phone_label("sil,AH,a") == ("SIL", "AH", "a")


def test_deviation_and_err_tags_are_kept():
    assert parse_phone_label("AH,AO*,s") == ("AH", "AO*", "s")
    assert parse_phone_label("AH,err,s") == ("AH", "ERR", "s")


def test_malformed_label_raises():
    with pytest.raises(ValueError):
        parse_phone_label("R,L")


def test_read_phone_tier_skips_silence_and_keeps_times(tmp_path):
    p = tmp_path / "x.TextGrid"
    p.write_text(FIXTURE)
    phones = read_phone_tier(p)
    assert [(s.canonical, s.perceived, s.error) for s in phones] == [
        ("V", "B", "s"),
        ("EH", "EH", "correct"),
        ("R", "SIL", "d"),
        ("IY", "IY", "correct"),
    ]
    assert phones[0].start == pytest.approx(0.1)
    assert phones[0].end == pytest.approx(0.3)


DATA = Path(__file__).resolve().parents[1] / "data"


def test_real_annotation_if_present():
    files = sorted(DATA.glob("**/annotation/*.TextGrid"))
    if not files:
        pytest.skip("L2-ARCTIC not extracted")
    for f in files[:50]:
        phones = read_phone_tier(f)
        assert phones, f
        assert all(s.end > s.start for s in phones), f
