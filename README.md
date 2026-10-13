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
