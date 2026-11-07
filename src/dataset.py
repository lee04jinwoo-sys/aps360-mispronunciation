"""Labels, phone inventory, speaker split and example listing for L2-ARCTIC."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.data_stats import L1
from src.textgrid import PhoneSegment, read_phone_tier

# 39 ARPAbet phones in the forced-alignment files plus AX, which only the
# manual annotations use.
PHONES = [
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R",
    "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
PHONE_INDEX = {p: i for i, p in enumerate(PHONES)}

# Standard test speakers (one per L1) used by prior work on L2-ARCTIC.
TEST_SPEAKERS = ["NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"]
# Validation: the alphabetically first remaining speaker of each L1.
VAL_SPEAKERS = ["ABA", "BWC", "ASI", "HJK", "EBVS", "HQTV"]


def speaker_split() -> dict[str, list[str]]:
    """Speaker-disjoint train/val/test split (12/6/6)."""
    held_out = set(TEST_SPEAKERS) | set(VAL_SPEAKERS)
    return {
        "train": sorted(s for s in L1 if s not in held_out),
        "val": sorted(VAL_SPEAKERS),
        "test": sorted(TEST_SPEAKERS),
    }


def segment_label(seg: PhoneSegment) -> Optional[int]:
    """1 = mispronounced (substitution or deletion), 0 = correct, None = addition.

    An addition has no expected phone, so there is nothing to classify.
    """
    if seg.error == "a":
        return None
    return 0 if seg.error == "correct" else 1


@dataclass(frozen=True)
class Example:
    speaker: str
    utterance: str
    phone: str
    phone_idx: int
    label: int
    start: float
    end: float


def examples_for_speaker(root: str | Path, speaker: str) -> list[Example]:
    """All labelled phone segments in a speaker's manually annotated utterances."""
    ann_dir = Path(root) / speaker / "annotation"
    if not ann_dir.is_dir():
        raise FileNotFoundError(ann_dir)
    examples = []
    for path in sorted(ann_dir.glob("*.TextGrid")):
        for seg in read_phone_tier(path):
            label = segment_label(seg)
            if label is None:
                continue
            examples.append(
                Example(speaker, path.stem, seg.canonical, PHONE_INDEX[seg.canonical],
                        label, seg.start, seg.end)
            )
    return examples
