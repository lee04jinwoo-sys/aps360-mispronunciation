"""Count phone segments and errors in the L2-ARCTIC annotated subset.

Usage: python -m src.data_stats [data/l2arctic]
"""
import sys
from collections import Counter
from pathlib import Path

from src.textgrid import read_phone_tier

L1 = {
    "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
    "BWC": "Chinese", "LXC": "Chinese", "NCC": "Chinese", "TXHC": "Chinese",
    "ASI": "Hindi", "RRBI": "Hindi", "SVBI": "Hindi", "TNI": "Hindi",
    "HJK": "Korean", "HKK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
}


def main(root: Path) -> None:
    by_l1: dict[str, Counter] = {}
    korean_subs: Counter = Counter()
    durations = []
    n_files = 0
    failures = []
    for f in sorted(root.glob("*/annotation/*.TextGrid")):
        speaker = f.parts[-3]
        try:
            segs = read_phone_tier(f)
        except ValueError as e:
            failures.append(f"{f}: {e}")
            continue
        n_files += 1
        c = by_l1.setdefault(L1[speaker], Counter())
        for s in segs:
            c[s.error] += 1
            if s.error != "a":
                durations.append(s.end - s.start)
            if L1[speaker] == "Korean" and s.error == "s":
                korean_subs[f"{s.canonical}->{s.perceived}"] += 1

    print(f"files parsed: {n_files}, failed: {len(failures)}")
    for line in failures[:10]:
        print("  FAIL", line)
    total = Counter()
    print(f"{'L1':<11}{'phones':>8}{'correct':>9}{'sub':>7}{'del':>6}{'add':>6}{'err%':>7}")
    for l1, c in sorted(by_l1.items()):
        total += c
        n = sum(c.values())
        err = c["s"] + c["d"] + c["a"]
        print(f"{l1:<11}{n:>8}{c['correct']:>9}{c['s']:>7}{c['d']:>6}{c['a']:>6}{100 * err / n:>6.1f}%")
    n = sum(total.values())
    err = total["s"] + total["d"] + total["a"]
    print(f"{'TOTAL':<11}{n:>8}{total['correct']:>9}{total['s']:>7}{total['d']:>6}{total['a']:>6}{100 * err / n:>6.1f}%")
    durations.sort()
    print(f"phone duration (s): median {durations[len(durations) // 2]:.3f}, "
          f"p95 {durations[int(len(durations) * 0.95)]:.3f}, max {durations[-1]:.3f}")
    print("top Korean-L1 substitutions:", korean_subs.most_common(12))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/l2arctic"))
