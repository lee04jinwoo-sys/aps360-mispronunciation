"""Error analysis of the baseline and the five CNN seeds on the test speakers.

Predictions are rebuilt from the saved checkpoints and the thresholds stored in
results/*.json, then checked against the saved test F1 before anything is
reported.   Usage: python -m src.run_error_analysis [data/l2arctic] [data/cache]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src import baseline, build_cache, cnn, dataset, error_analysis, metrics

SEEDS = [0, 1, 2, 3, 4]
TOP_PHONES = 12
TOP_CONFUSIONS = 12


def _mean_tables(tables: list[list[dict]]) -> list[dict]:
    """Average precision/recall/F1 per phone across seeds (n and error rate are fixed)."""
    by_seed = [{r["phone"]: r for r in t} for t in tables]
    rows = []
    for phone, first in by_seed[0].items():
        row = {"phone": phone, "n": first["n"], "error_rate": first["error_rate"]}
        for key in ("precision", "recall", "f1"):
            row[key] = float(np.mean([t[phone][key] for t in by_seed]))
        rows.append(row)
    return rows


def main(root: Path, cache: Path) -> dict:
    train = build_cache.load_arrays(cache / "train.npz")
    val = build_cache.load_arrays(cache / "val.npz")
    test = build_cache.load_arrays(cache / "test.npz")
    y = test["label"]

    details = error_analysis.annotation_details(root, dataset.speaker_split()["test"])
    assert np.array_equal(details["label"], y), "annotation rows do not line up with the cache"
    assert [dataset.PHONES[i] for i in test["phone"]] == details["canonical"].tolist()

    # Baseline
    model, scaler = baseline.train(train)
    b_threshold = metrics.best_threshold(val["label"], baseline.predict_scores(model, scaler, val))
    b_pred = (baseline.predict_scores(model, scaler, test) >= b_threshold).astype(int)
    saved = json.loads(Path("results/baseline.json").read_text())
    assert abs(metrics.prf(y, b_pred)["f1"] - saved["test_overall"]["f1"]) < 1e-6, "baseline not reproduced"

    # CNN seeds, from checkpoints
    norm = cnn.fit_normalizer(train["windows"])
    cnn_preds = []
    for seed in SEEDS:
        saved = json.loads(Path(f"results/cnn_seed{seed}.json").read_text())
        net = cnn.PhoneCNN()
        net.load_state_dict(torch.load(cache / f"cnn_seed{seed}.pt", map_location="cpu"))
        scores = cnn.predict_scores(net, norm, test)
        pred = (scores >= saved["threshold_from_val"]).astype(int)
        gap = abs(metrics.prf(y, pred)["f1"] - saved["test_overall"]["f1"])
        assert gap < 2e-3, f"seed {seed} not reproduced (F1 differs by {gap:.4f})"
        cnn_preds.append(pred)

    phone, err = test["phone"], details["error"]
    b_table = error_analysis.per_phone_table(phone, y, b_pred)
    c_table = _mean_tables([error_analysis.per_phone_table(phone, y, p) for p in cnn_preds])
    c_by_phone = {r["phone"]: r for r in c_table}
    b_by_phone = {r["phone"]: r for r in b_table}

    per_phone = [
        {"phone": r["phone"], "n": r["n"], "error_rate": r["error_rate"],
         "baseline": {k: b_by_phone[r["phone"]][k] for k in ("precision", "recall", "f1")},
         "cnn_mean": {k: c_by_phone[r["phone"]][k] for k in ("precision", "recall", "f1")}}
        for r in b_table
    ]

    kinds = [error_analysis.recall_by_error_type(err, p) for p in cnn_preds]
    error_type = {
        k: {"n": kinds[0][k]["n"],
            "baseline_recall": error_analysis.recall_by_error_type(err, b_pred)[k]["recall"],
            "cnn_recall_mean": float(np.mean([m[k]["recall"] for m in kinds])),
            "cnn_recall_std": float(np.std([m[k]["recall"] for m in kinds], ddof=1))}
        for k in ("s", "d")
    }

    conf_base = error_analysis.top_confusions(details["canonical"], details["perceived"], err, b_pred, TOP_CONFUSIONS)
    conf_cnn = [error_analysis.top_confusions(details["canonical"], details["perceived"], err, p, TOP_CONFUSIONS)
                for p in cnn_preds]
    confusions = [
        {"expected": c["expected"], "heard": c["heard"], "n": c["n"],
         "baseline_detected": c["detected"],
         "cnn_detected_mean": float(np.mean([run[i]["detected"] for run in conf_cnn]))}
        for i, c in enumerate(conf_base)
    ]

    result = {
        "n_test": int(len(y)),
        "n_substitutions": int((err == "s").sum()),
        "n_deletions": int((err == "d").sum()),
        "per_phone": per_phone,
        "error_type_recall": error_type,
        "top_substitutions": confusions,
        "always_positive_f1_by_l1": error_analysis.always_positive_f1_by_l1(y, test["speaker"]),
        "error_rate_by_l1": {
            lang: float(np.mean(y[np.array([dataset_l1 == lang for dataset_l1 in _l1(test["speaker"])])]))
            for lang in sorted(set(_l1(test["speaker"])))
        },
        "reproduction_check": "baseline exact; CNN seeds within 2e-3 of saved test F1",
    }
    Path("results/error_analysis.json").write_text(json.dumps(result, indent=2))
    return result


def _l1(speakers: np.ndarray) -> list[str]:
    from src.data_stats import L1
    return [L1[s] for s in speakers]


if __name__ == "__main__":
    r = main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/l2arctic"),
             Path(sys.argv[2] if len(sys.argv) > 2 else "data/cache"))
    print(json.dumps({k: r[k] for k in ("n_substitutions", "n_deletions", "error_type_recall",
                                        "always_positive_f1_by_l1", "error_rate_by_l1")}, indent=2))
    print("\nphone  n     err%   base_rec cnn_rec  base_f1 cnn_f1")
    for row in r["per_phone"][:TOP_PHONES]:
        print(f"{row['phone']:<5} {row['n']:>5} {100*row['error_rate']:>6.1f} "
              f"{row['baseline']['recall']:>8.2f} {row['cnn_mean']['recall']:>7.2f} "
              f"{row['baseline']['f1']:>8.2f} {row['cnn_mean']['f1']:>6.2f}")
    print("\nexpected->heard  n   base_det cnn_det")
    for c in r["top_substitutions"]:
        print(f"{c['expected']:>3}->{c['heard']:<4} {c['n']:>5} {c['baseline_detected']:>8.2f} {c['cnn_detected_mean']:>7.2f}")
