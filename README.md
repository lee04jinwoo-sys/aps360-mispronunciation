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
- [x] CNN trained from scratch (5 seeds)
- [ ] Weakly supervised pre-training on forced-alignment labels (planned)

### Baseline results (`results/baseline.json`)

Test = the six standard speakers (29,786 phones, 13.9% mispronounced). Threshold chosen on the validation speakers only.

| | Precision | Recall | F1 (mispronounced) |
|---|---|---|---|
| All test speakers | 0.335 | 0.604 | 0.431 |
| Korean (YKWK) | 0.362 | 0.635 | 0.461 |
| Always predict "mispronounced" | 0.139 | 1.000 | 0.244 |

### CNN results (5 seeds, `results/cnn_seeds_summary.json`)

Same test speakers and protocol; early stopping and the threshold use validation speakers only. Mean ± sd over training seeds 0-4.

| | CNN | Baseline |
|---|---|---|
| F1, all test speakers | 0.450 ± 0.013 | 0.431 |
| F1, Korean (YKWK) | 0.480 ± 0.019 | 0.461 |
| Precision / recall, all | 0.383 ± 0.031 / 0.550 ± 0.042 | 0.335 / 0.604 |

All 5 seeds beat the baseline overall; 4 of 5 on the Korean speaker. Per L1 the gain is in Arabic (+0.03),
Chinese (+0.02) and Spanish (+0.04); Hindi and Vietnamese show no difference. The spread covers training seeds only,
not the choice of test speakers (one per L1). Validation F1 plateaus after about epoch 3 while training loss keeps falling.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m src.build_cache data/l2arctic data/cache   # ~15 s
python -m src.run_baseline
python -m src.run_cnn data/cache 0   # one seed, ~3 min on Apple MPS
python -m pytest -q
```
