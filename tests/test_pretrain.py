"""Pre-training data (unannotated, non-test speakers) and encoder transfer."""
import numpy as np
import pytest
import torch

from src import cnn, dataset, features, pretrain

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0.0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0.0
        xmax = 1.0
        intervals: size = 4
            intervals [1]:
                xmin = 0.0
                xmax = 0.1
                text = "sil"
            intervals [2]:
                xmin = 0.1
                xmax = 0.3
                text = "DH"
            intervals [3]:
                xmin = 0.3
                xmax = 0.5
                text = "AH0"
            intervals [4]:
                xmin = 0.5
                xmax = 1.0
                text = "sp"
'''


def make_speaker(root, spk, annotated, unannotated):
    for sub in ("annotation", "textgrid"):
        (root / spk / sub).mkdir(parents=True)
    for u in annotated + unannotated:
        (root / spk / "textgrid" / f"{u}.TextGrid").write_text(TEXTGRID)
    for u in annotated:
        (root / spk / "annotation" / f"{u}.TextGrid").write_text(TEXTGRID)


# --- which data is used -----------------------------------------------------


def test_pretrain_speakers_exclude_every_test_speaker():
    spk = pretrain.pretrain_speakers()

    assert len(spk) == 18
    assert not set(spk) & set(dataset.TEST_SPEAKERS)


def test_only_unannotated_utterances_are_used(tmp_path):
    make_speaker(tmp_path, "ABA", annotated=["a1"], unannotated=["u1", "u2"])

    paths = pretrain.unannotated_textgrids(tmp_path, "ABA")

    assert [p.stem for p in paths] == ["u1", "u2"]


def test_phone_segments_drop_silence_and_stress(tmp_path):
    p = tmp_path / "x.TextGrid"
    p.write_text(TEXTGRID)

    segs = pretrain.phone_segments(p)

    assert [(dataset.PHONES[i], s, e) for i, s, e in segs] == [("DH", 0.1, 0.3), ("AH", 0.3, 0.5)]


def test_held_out_split_is_deterministic_and_small():
    utts = [f"arctic_a{i:04d}" for i in range(2000)]

    first = [pretrain.is_heldout("ABA", u) for u in utts]
    again = [pretrain.is_heldout("ABA", u) for u in utts]

    assert first == again
    assert 0.02 < np.mean(first) < 0.09


# --- compact windows --------------------------------------------------------


@pytest.mark.parametrize("start,end", [(0.5, 0.6), (0.0, 0.05), (0.95, 1.0), (0.5, 0.5), (0.2, 0.9)])
def test_compact_window_expands_back_to_phone_window(start, end):
    rng = np.random.default_rng(0)
    mel = rng.normal(size=(features.N_MELS, 100)).astype(np.float32)

    spec, lo, hi = pretrain.compact_window(mel, start, end)
    full = pretrain.expand_windows(spec[None].astype(np.float16), np.array([lo]), np.array([hi]))[0]

    ref = features.phone_window(mel, start, end)
    np.testing.assert_allclose(full[0], ref[0].astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(full[1], ref[1])


# --- model transfer ---------------------------------------------------------


def test_phone_classifier_outputs_40_logits():
    model = cnn.PhoneClassifier().eval()

    assert model(torch.zeros(3, 2, 64, 30)).shape == (3, 40)


def test_pretrained_encoder_is_loaded_into_detector():
    torch.manual_seed(0)
    clf = cnn.PhoneClassifier()
    torch.manual_seed(1)
    det = cnn.PhoneCNN()

    cnn.load_encoder(det, clf.encoder.state_dict())

    for a, b in zip(det.encoder.state_dict().values(), clf.encoder.state_dict().values()):
        assert torch.equal(a, b)


def test_train_model_starts_from_given_encoder():
    """With zero epochs of change (lr=0), the trained encoder equals the initial one."""
    rng = np.random.default_rng(0)
    n = 200
    data = {
        "windows": rng.normal(size=(n, 2, 64, 30)).astype(np.float16),
        "phone": rng.integers(0, 40, n),
        "label": rng.integers(0, 2, n),
    }
    clf = cnn.PhoneClassifier()
    norm = cnn.Normalizer(np.zeros(64, np.float32), np.ones(64, np.float32))

    model, used_norm, _ = cnn.train_model(data, data, epochs=1, lr=0.0, device="cpu",
                                          init_encoder=clf.encoder.state_dict(), norm=norm)

    assert used_norm is norm
    for k, v in clf.encoder.state_dict().items():
        if "running" in k or "num_batches" in k:
            continue  # batch-norm statistics still update in train mode
        assert torch.equal(model.encoder.state_dict()[k].cpu(), v)


def test_phone_classifier_learns_a_separable_toy_task():
    rng = np.random.default_rng(0)
    n = 600
    phone = rng.integers(0, 3, n)
    spec = rng.normal(size=(n, 64, 30)).astype(np.float32)
    for k in range(3):
        spec[phone == k, k * 20 : k * 20 + 8, :] += 3.0  # each class lights up a different mel band
    lo, hi = np.full(n, 10), np.full(n, 20)
    arrays = {"spec": spec.astype(np.float16), "lo": lo, "hi": hi, "phone": phone,
              "heldout": np.arange(n) % 5 == 0}

    model, norm, hist = cnn.train_phone_classifier(arrays, epochs=5, batch_size=32, patience=5,
                                                   device="cpu", seed=0)

    assert hist["val_acc"][hist["best_epoch"]] > 0.9
