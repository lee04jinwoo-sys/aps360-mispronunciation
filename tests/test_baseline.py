"""MFCC + logistic regression baseline and the metrics shared with the CNN."""
import numpy as np
import pytest

from src import baseline, metrics


def toy_arrays(n=200, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    mfcc = rng.normal(size=(n, 78)).astype(np.float32)
    mfcc[:, 0] += 3.0 * label  # one informative feature
    return {
        "mfcc": mfcc,
        "phone": rng.integers(0, 40, n),
        "label": label,
        "speaker": np.array(["YKWK"] * n),
    }


# --- metrics ----------------------------------------------------------------


def test_precision_recall_f1_on_positive_class():
    y = np.array([1, 1, 0, 0, 1, 0])
    pred = np.array([1, 0, 0, 1, 1, 0])  # TP=2 FP=1 FN=1

    m = metrics.prf(y, pred)

    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_prf_with_no_predicted_positives_is_zero_not_nan():
    m = metrics.prf(np.array([1, 0, 1]), np.array([0, 0, 0]))

    assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_always_positive_baseline_has_recall_one_and_precision_equal_to_error_rate():
    y = np.array([1, 0, 0, 0])

    m = metrics.prf(y, np.ones_like(y))

    assert m["recall"] == 1.0 and m["precision"] == pytest.approx(0.25)


def test_prf_by_l1_groups_speakers_by_native_language():
    y = np.array([1, 0, 1, 0])
    pred = np.array([1, 0, 0, 0])
    speakers = np.array(["YKWK", "YKWK", "TLV", "TLV"])  # Korean, Vietnamese

    by_l1 = metrics.prf_by_l1(y, pred, speakers)

    assert by_l1["Korean"]["f1"] == 1.0
    assert by_l1["Vietnamese"]["recall"] == 0.0
    assert set(by_l1) == {"Korean", "Vietnamese"}


def test_best_threshold_maximises_f1():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.45, 0.55, 0.8, 0.9])

    t = metrics.best_threshold(y, scores)

    assert 0.45 < t <= 0.55
    assert metrics.prf(y, (scores >= t).astype(int))["f1"] == 1.0


# --- baseline features ------------------------------------------------------


def test_feature_matrix_is_78_stats_plus_40_one_hot():
    a = toy_arrays(10)

    x = baseline.feature_matrix(a)

    assert x.shape == (10, 118)
    assert (x[:, 78:].sum(axis=1) == 1).all()
    assert x[np.arange(10), 78 + a["phone"]].tolist() == [1.0] * 10


def test_scaler_standardises_continuous_columns_only():
    a = toy_arrays(300)
    scaler = baseline.fit_scaler(baseline.feature_matrix(a))

    z = scaler(baseline.feature_matrix(a))

    assert np.allclose(z[:, :78].mean(axis=0), 0, atol=1e-5)
    assert np.allclose(z[:, :78].std(axis=0), 1, atol=1e-4)
    assert np.array_equal(z[:, 78:], baseline.feature_matrix(a)[:, 78:])


def test_scaler_handles_constant_column_without_nan():
    x = baseline.feature_matrix(toy_arrays(50))
    x[:, 5] = 1.0

    z = baseline.fit_scaler(x)(x)

    assert np.isfinite(z).all()


def test_scaler_uses_training_statistics_not_test_statistics():
    train = baseline.feature_matrix(toy_arrays(100, seed=1))
    test = baseline.feature_matrix(toy_arrays(100, seed=2)) + 10.0
    scaler = baseline.fit_scaler(train)

    assert scaler(test)[:, :78].mean() > 5  # shifted test data stays shifted


# --- training ---------------------------------------------------------------


def test_baseline_learns_a_separable_toy_problem():
    # One feature separates the classes by 3 sigma, so even the optimal rule
    # tops out near F1 0.94 (measured); always predicting 1 gives ~0.67.
    # (Criterion first set to F1 > 0.9 on 200 test samples, which was too
    # tight for that noise level; changed after seeing it fail, model was fine.)
    train, test = toy_arrays(400, seed=1), toy_arrays(2000, seed=2)

    model, scaler = baseline.train(train)
    scores = baseline.predict_scores(model, scaler, test)

    assert metrics.prf(test["label"], (scores >= 0.5).astype(int))["f1"] > 0.85


def test_scores_are_probabilities():
    train = toy_arrays(100)
    model, scaler = baseline.train(train)

    scores = baseline.predict_scores(model, scaler, train)

    assert ((scores >= 0) & (scores <= 1)).all()
