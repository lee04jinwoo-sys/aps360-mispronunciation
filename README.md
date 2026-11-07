# Phone-Level Mispronunciation Detection for Korean Learners of English

APS360 (Applied Fundamentals of Deep Learning, U of T, Fall 2026) individual project.

Given a recording of a read English sentence and its canonical phone sequence, classify each phone as
correctly pronounced or mispronounced, using a CNN over log-mel spectrogram segments.

## Data

[L2-ARCTIC v5.0](https://psi.engr.tamu.edu/l2-arctic-corpus/) (CC BY-NC 4.0). The corpus is **not**
redistributed here; request your own copy from the official site and unzip it into `data/`.

## Layout

- `proposal/` – LaTeX source of the project proposal (course template)
- `src/` – data processing and model code

## Status (progress report, 2026-11)

- [x] Data pipeline: L2-ARCTIC phone segments -> 300 ms log-mel windows + phone mask, speaker-disjoint split (12/6/6)
- [x] Baseline: 78 MFCC statistics + one-hot expected phone -> logistic regression
- [x] CNN trained from scratch (one seed so far)
- [ ] Weakly supervised pre-training on forced-alignment labels (planned)

### Baseline results (`results/baseline.json`)

Test = the six standard speakers (29,786 phones, 13.9% mispronounced). Threshold chosen on the validation speakers only.

| | Precision | Recall | F1 (mispronounced) |
|---|---|---|---|
| All test speakers | 0.335 | 0.604 | 0.431 |
| Korean (YKWK) | 0.362 | 0.635 | 0.461 |
| Always predict "mispronounced" | 0.139 | 1.000 | 0.244 |

### CNN results (`results/cnn_seed0.json`, seed 0)

Same test speakers and protocol; early stopping and the threshold use validation speakers only.

| | Precision | Recall | F1 (mispronounced) |
|---|---|---|---|
| All test speakers | 0.383 | 0.532 | 0.445 |
| Korean (YKWK) | 0.460 | 0.575 | 0.511 |

One seed only, so the +0.014 overall gain over the baseline is not yet distinguishable from run-to-run noise.
Validation F1 plateaus after epoch 3 while training loss keeps falling.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m src.build_cache data/l2arctic data/cache   # ~15 s
python -m src.run_baseline
python -m src.run_cnn data/cache 0   # ~3 min on Apple MPS
python -m pytest -q
```

## AI assistance

An AI coding assistant (Claude Code) wrote much of the code and tests under my direction: I set the
project direction, the data split and the evaluation protocol, and reviewed the results. This is stated
here for transparency.
