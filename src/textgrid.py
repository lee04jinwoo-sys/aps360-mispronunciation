"""Minimal reader for L2-ARCTIC annotation TextGrids (phone tier only)."""
import re
from dataclasses import dataclass
from pathlib import Path

SILENCE = {"", "SIL", "SP", "SPN"}
ERROR_TYPES = {"s", "d", "a"}


@dataclass
class PhoneSegment:
    start: float
    end: float
    canonical: str
    perceived: str
    error: str  # "correct", "s", "d" or "a"


def _norm(phone: str) -> str:
    """Upper-case and drop the ARPAbet stress digit (AH0 -> AH)."""
    return re.sub(r"\d$", "", phone.strip().upper())


def parse_phone_label(label: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in label.split(",")]
    if len(parts) == 1:
        phone = _norm(parts[0])
        return phone, phone, "correct"
    if len(parts) != 3 or parts[2].lower() not in ERROR_TYPES:
        raise ValueError(f"unexpected phone label: {label!r}")
    return _norm(parts[0]), _norm(parts[1]), parts[2].lower()


_INTERVAL = re.compile(
    r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "((?:[^"]|"")*)"', re.S
)


def read_phone_tier(path: Path) -> list[PhoneSegment]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r'name = "phones".*?(?=item \[\d+\]:|\Z)', text, re.S)
    if not m:
        raise ValueError(f"no phones tier in {path}")
    segments = []
    for start, end, label in _INTERVAL.findall(m.group(0)):
        label = label.replace('""', '"').strip()
        canonical, perceived, error = parse_phone_label(label) if label else ("", "", "correct")
        if error == "correct" and canonical in SILENCE:
            continue
        segments.append(PhoneSegment(float(start), float(end), canonical, perceived, error))
    return segments
