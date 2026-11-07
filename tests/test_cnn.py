"""CNN phone-level mispronunciation classifier: shapes, normalisation, training loop."""
import numpy as np
import pytest
import torch

from src import cnn, metrics


def synthetic(n=400, seed=0):
    """Label-1 windows are brighter inside the phone mask, so the task is learnable."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    windows = rng.normal(size=(n, 2, 64, 30)).astype(np.float32)
    windows[:, 1] = 0.0
    windows[:, 1, :, 10:20] = 1.0  # mask: the phone's own frames
    windows[:, 0, :, 10:20] += 1.5 * label[:, None, None]
    return {
        "windows": windows.astype(np.float16),
        "phone": rng.integers(0, 40, n),
        "label": label,
        "speaker": np.array(["YKWK"] * n),
    }


# --- model ------------------------------------------------------------------


def test_forward_returns_one_logit_per_example():
    model = cnn.PhoneCNN().eval()

    out = model(torch.zeros(5, 2, 64, 30), torch.zeros(5, dtype=torch.long))

    assert out.shape == (5,)


def test_encoder_embedding_is_128_dimensional():
    model = cnn.PhoneCNN().eval()

    assert model.encode(torch.zeros(3, 2, 64, 30)).shape == (3, 128)


def test_expected_phone_changes_the_output():
    torch.manual_seed(0)
    model = cnn.PhoneCNN().eval()
    x = torch.randn(1, 2, 64, 30)

    a = model(x, torch.tensor([3]))
    b = model(x, torch.tensor([30]))

    assert not torch.allclose(a, b)


def test_model_is_small_enough_to_train_from_scratch():
    n = sum(p.numel() for p in cnn.PhoneCNN().parameters())

    assert n < 1_000_000


# --- normaliser -------------------------------------------------------------


def test_normaliser_standardises_each_mel_bin_of_the_spectrogram_channel():
    a = synthetic(300)
    norm = cnn.fit_normalizer(a["windows"])

    z = norm(a["windows"]).numpy()

    assert np.allclose(z[:, 0].mean(axis=(0, 2)), 0, atol=1e-2)
    assert np.allclose(z[:, 0].std(axis=(0, 2)), 1, atol=2e-2)


def test_normaliser_leaves_the_mask_channel_untouched():
    a = synthetic(50)
    norm = cnn.fit_normalizer(a["windows"])

    z = norm(a["windows"]).numpy()

    assert np.array_equal(z[:, 1], a["windows"][:, 1].astype(np.float32))


def test_normaliser_uses_given_statistics_on_new_data():
    train = synthetic(100, seed=1)["windows"]
    shifted = (synthetic(100, seed=2)["windows"].astype(np.float32) + 10).astype(np.float16)
    norm = cnn.fit_normalizer(train)

    assert norm(shifted).numpy()[:, 0].mean() > 5


# --- loss weighting ---------------------------------------------------------


def test_pos_weight_is_negatives_over_positives():
    assert cnn.pos_weight(np.array([0, 0, 0, 0, 0, 1])) == pytest.approx(5.0)


# --- training ---------------------------------------------------------------


def test_training_learns_a_learnable_task():
    train, val, test = synthetic(600, 1), synthetic(300, 2), synthetic(300, 3)

    model, norm, hist = cnn.train_model(train, val, epochs=6, batch_size=64, seed=0, device="cpu")
    scores = cnn.predict_scores(model, norm, test, device="cpu")
    pred = (scores >= hist["threshold"]).astype(int)

    assert metrics.prf(test["label"], pred)["f1"] > 0.85


def test_best_epoch_weights_are_restored():
    train, val = synthetic(300, 1), synthetic(200, 2)

    model, norm, hist = cnn.train_model(train, val, epochs=4, batch_size=64, seed=0, device="cpu")
    scores = cnn.predict_scores(model, norm, val, device="cpu")
    f1 = metrics.prf(val["label"], (scores >= hist["threshold"]).astype(int))["f1"]

    assert f1 == pytest.approx(max(hist["val_f1"]), abs=1e-6)
    assert hist["best_epoch"] == int(np.argmax(hist["val_f1"]))


def test_training_is_deterministic_for_a_seed():
    train, val = synthetic(200, 1), synthetic(100, 2)

    _, _, h1 = cnn.train_model(train, val, epochs=2, batch_size=64, seed=7, device="cpu")
    _, _, h2 = cnn.train_model(train, val, epochs=2, batch_size=64, seed=7, device="cpu")

    assert h1["train_loss"] == h2["train_loss"]


def test_early_stopping_stops_before_max_epochs_when_val_does_not_improve():
    train = synthetic(200, 1)
    val = synthetic(100, 2)
    val["label"] = 1 - val["label"]  # inverted labels: val F1 cannot keep improving

    _, _, hist = cnn.train_model(train, val, epochs=30, batch_size=64, patience=2, seed=0, device="cpu")

    assert len(hist["val_f1"]) < 30


def test_scores_are_probabilities():
    a = synthetic(50)
    model, norm, _ = cnn.train_model(a, a, epochs=1, batch_size=32, seed=0, device="cpu")

    s = cnn.predict_scores(model, norm, a, device="cpu")

    assert s.shape == (50,) and ((s >= 0) & (s <= 1)).all()
