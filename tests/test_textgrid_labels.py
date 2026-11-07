"""Annotation typos found in the real corpus must not create extra phone classes."""
import pytest

from src.textgrid import parse_phone_label


@pytest.mark.parametrize(
    "label, canonical",
    [
        ("D_,T,s", "D"),
        ("ER),sil,d", "ER"),
        ("V``,B,s", "V"),
        ("W`,sil,d", "W"),
        ("Y_,sil,d", "Y"),
        ("Z_,S,s", "Z"),
    ],
)
def test_stray_characters_are_stripped_from_the_canonical_phone(label, canonical):
    assert parse_phone_label(label)[0] == canonical


def test_stress_digit_is_still_removed():
    assert parse_phone_label("AH0")[0] == "AH"
