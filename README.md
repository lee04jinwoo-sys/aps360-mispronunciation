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
- [x] Error analysis by phone, error type, substitution pair and L1
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

### Error analysis (`results/error_analysis.json`)

Test speakers; CNN = mean over the 5 seeds. Predictions are rebuilt from the saved checkpoints and checked against the
saved test F1 before anything is reported. Of 4,140 errors, 3,199 are substitutions and 941 are deletions.

**The two most common substitutions are DH->D (451) and Z->S (412), 27% of all substitutions.** Phones with the
highest error rates: DH 57.5%, Z 52.4%, TH 44.1%, JH 40.9%.

**On those phones the models are not hearing anything.** For DH and TH the baseline's precision equals the phone's error
rate (0.575 and 0.441) at recall 1.00, i.e. it flags every instance, and its F1 (0.730, 0.613) equals the
always-"mispronounced" F1. The CNN lands on the same values (0.724, 0.603). The expected-phone input alone explains these.

**Where the CNN differs from the baseline** (different thresholds, so these are different operating points, not like for like):

| | Baseline | CNN (mean) |
|---|---|---|
| D->T flagged | 0.26 | 0.48 |
| P->B flagged | 0.26 | 0.40 |
| OW->AO flagged | 0.79 | 0.47 |
| V->F flagged | 0.77 | 0.50 |
| Recall on substitutions | 0.575 | 0.528 ± 0.043 |
| Recall on deletions | 0.702 | 0.624 ± 0.090 |

The overall gain comes from precision (0.335 -> 0.383) at lower recall (0.604 -> 0.550). Neither model detects K->G (n=49;
0.00 and 0.04), a voicing contrast.

**L1.** Vietnamese has twice the error rate of the others (24.3% vs 10-13%), which inflates its F1. Margin of the CNN
over the always-"mispronounced" F1 of the same language: Korean +0.265 (largest), Spanish +0.229, Arabic +0.195,
Chinese +0.187, Vietnamese +0.181, Hindi +0.154 (smallest).

Caveats: one test speaker per L1; rare phones are noisy (ZH n=33, AX n=1); the substitution table only covers
substitutions with a recorded perceived phone.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m src.build_cache data/l2arctic data/cache   # ~15 s
python -m src.run_baseline
python -m src.run_cnn data/cache 0   # one seed, ~3 min on Apple MPS (seeds 0-4 for the error analysis)
python -m src.run_error_analysis data/l2arctic data/cache   # ~15 s
python -m pytest -q
```
