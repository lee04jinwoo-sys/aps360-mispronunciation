"""Error analysis: per-phone, per-error-type and per-confusion breakdowns."""
import numpy as np
import pytest

from src import dataset, error_analysis
from src.dataset import PHONE_INDEX
from tests.test_dataset import TEXTGRID  # noqa: F401  (reused fixture text)


# --- annotation details aligned with the cache ------------------------------


@pytest.fixture
def corpus(tmp_path):
    ann = tmp_path / "YKWK" / "annotation"
    ann.mkdir(parents=True)
    (ann / "arctic_a0001.TextGrid").write_text(TEXTGRID)
    return tmp_path


def test_annotation_details_follow_the_same_order_as_the_cache(corpus):
    examples = dataset.examples_for_speaker(corpus, "YKWK")

    details = error_analysis.annotation_details(corpus, ["YKWK"])

    assert len(details["error"]) == len(examples)
    assert details["canonical"].tolist() == [e.phone for e in examples]
    assert details["label"].tolist() == [e.label for e in examples]


def test_annotation_details_report_perceived_phone_and_error_type(corpus):
    details = error_analysis.annotation_details(corpus, ["YKWK"])

    # V->B substitution, correct EH, R deleted ("sil" perceived)
    assert details["error"].tolist() == ["s", "correct", "d"]
    assert details["perceived"].tolist()[0] == "B"


# --- per-phone table --------------------------------------------------------


def test_per_phone_table_counts_error_rate_and_scores():
    phone = np.array([PHONE_INDEX["DH"]] * 4 + [PHONE_INDEX["AA"]] * 4)
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])

    rows = {r["phone"]: r for r in error_analysis.per_phone_table(phone, y, pred)}

    assert rows["DH"]["n"] == 4
    assert rows["DH"]["error_rate"] == pytest.approx(0.75)
    assert rows["DH"]["recall"] == pytest.approx(2 / 3)
    assert rows["DH"]["precision"] == 1.0
    assert rows["AA"]["recall"] == 0.0


def test_per_phone_table_is_sorted_by_error_rate_descending():
    phone = np.array([PHONE_INDEX["DH"]] * 2 + [PHONE_INDEX["AA"]] * 2)
    y = np.array([1, 0, 1, 1])

    rows = error_analysis.per_phone_table(phone, y, y)

    assert [r["phone"] for r in rows] == ["AA", "DH"]


def test_per_phone_table_omits_phones_that_never_occur():
    rows = error_analysis.per_phone_table(
        np.array([PHONE_INDEX["B"]]), np.array([0]), np.array([0])
    )

    assert [r["phone"] for r in rows] == ["B"]


# --- error type -------------------------------------------------------------


def test_recall_by_error_type_splits_substitutions_and_deletions():
    error = np.array(["s", "s", "d", "d", "correct", "correct"])
    pred = np.array([1, 0, 1, 1, 0, 1])

    r = error_analysis.recall_by_error_type(error, pred)

    assert r["s"] == {"n": 2, "recall": 0.5}
    assert r["d"] == {"n": 2, "recall": 1.0}


def test_correct_phones_are_not_an_error_type():
    r = error_analysis.recall_by_error_type(np.array(["correct"]), np.array([1]))

    assert set(r) == {"s", "d"} and r["s"]["n"] == 0


# --- confusions -------------------------------------------------------------


def test_top_confusions_count_substitutions_and_report_detection_rate():
    canonical = np.array(["DH", "DH", "DH", "Z", "AA"])
    perceived = np.array(["D", "D", "Z", "S", "AA"])
    error = np.array(["s", "s", "s", "s", "correct"])
    pred = np.array([1, 0, 1, 1, 0])

    rows = error_analysis.top_confusions(canonical, perceived, error, pred, k=5)

    assert rows[0] == {"expected": "DH", "heard": "D", "n": 2, "detected": 0.5}
    assert {(r["expected"], r["heard"]) for r in rows} == {("DH", "D"), ("DH", "Z"), ("Z", "S")}


def test_top_confusions_ignore_deletions_and_correct_phones():
    canonical = np.array(["R", "AA"])
    perceived = np.array(["sil", "AA"])
    error = np.array(["d", "correct"])

    assert error_analysis.top_confusions(canonical, perceived, error, np.array([1, 0]), k=5) == []


def test_top_confusions_respects_k():
    canonical = np.array(["DH"] * 3 + ["Z"] * 2 + ["S"])
    perceived = np.array(["D"] * 3 + ["S"] * 2 + ["T"])
    error = np.array(["s"] * 6)

    rows = error_analysis.top_confusions(canonical, perceived, error, np.ones(6, dtype=int), k=2)

    assert [r["n"] for r in rows] == [3, 2]


# --- base-rate reference ----------------------------------------------------


def test_always_positive_f1_by_l1_uses_each_languages_error_rate():
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    speakers = np.array(["YKWK"] * 4 + ["TLV"] * 4)  # Korean 25 %, Vietnamese 75 %

    r = error_analysis.always_positive_f1_by_l1(y, speakers)

    assert r["Korean"] == pytest.approx(2 * 0.25 / 1.25)
    assert r["Vietnamese"] == pytest.approx(2 * 0.75 / 1.75)
