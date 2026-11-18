# Pre-training ablation: protocol (written 2026-11-17, before any pre-training result)

## Question

Does initializing the CNN encoder from a 40-way phone classifier, trained on the
unannotated utterances of non-test speakers, improve mispronunciation detection
compared with the same CNN trained from scratch?

## Pre-training data

- Speakers: all 18 non-test speakers (train + validation speakers). The six test
  speakers (NJS, TLV, TNI, TXHC, YKWK, ZHAA) are excluded entirely.
- Utterances: those that have a forced-alignment TextGrid but **no** manual
  annotation (17,379). Annotated utterances are never used for pre-training.
- Labels: the forced-alignment phone of each segment (40-class inventory;
  silences dropped). Inputs are built exactly like the detection inputs
  (300 ms log-mel window + phone mask).
- About 5% of these utterances (chosen by a fixed hash of the utterance id) are
  held out to early-stop pre-training on phone accuracy.

## Fine-tuning

Identical to the scratch CNN (`src/cnn.py: train_model`): same architecture,
data, optimizer, learning rate, batch size, epochs, early stopping on validation
F1 and threshold tuned on validation speakers. The only differences:
1. encoder weights start from the pre-trained encoder;
2. spectrograms are standardized with the pre-training statistics, so the encoder
   sees inputs on the scale it was trained on.

Seeds 0-4, as for the scratch CNN.

## Decision rule (fixed now)

- **Primary:** mean test F1 (mispronounced class, all six test speakers) over
  seeds 0-4. Pre-training "helps" if this mean is higher than the scratch CNN's
  (0.4496). Reported together with the per-seed paired comparison (seed i vs
  seed i) and the difference in means relative to the seed standard deviations.
- **Secondary (not part of the verdict):** Korean speaker YKWK F1; per-phone F1
  on DH and TH compared with the always-"mispronounced" F1 (0.730, 0.613), to see
  whether either model uses the audio for these phones.
- The test set is scored once per trained model. No setting is changed after
  looking at test results; any change is recorded here as "protocol changed"
  with the reason.
