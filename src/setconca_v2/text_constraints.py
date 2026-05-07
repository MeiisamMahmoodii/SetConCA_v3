from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set


DEFAULT_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "after",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "with",
    "without",
    "during",
    "new",
}


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


@dataclass(frozen=True)
class LengthBand:
    label: str
    min_words: int
    max_words: int


DEFAULT_LENGTH_BANDS: List[LengthBand] = [
    LengthBand("5-7", 5, 7),
    LengthBand("10-12", 10, 12),
    LengthBand("15-17", 15, 17),
    LengthBand("20-22", 20, 22),
]


def normalize_token(token: str) -> str:
    return token.lower().strip("-'_ ")


def tokenize_words(text: str) -> List[str]:
    return [normalize_token(m.group(0)) for m in WORD_RE.finditer(text) if normalize_token(m.group(0))]


def count_words(text: str) -> int:
    return len(tokenize_words(text))


def extract_banned_words(
    text: str,
    *,
    max_words: int = 8,
    min_len: int = 4,
    extra_stopwords: Iterable[str] | None = None,
) -> List[str]:
    stop = set(DEFAULT_STOPWORDS)
    if extra_stopwords:
        stop.update(normalize_token(x) for x in extra_stopwords)

    tokens = [
        t
        for t in tokenize_words(text)
        if len(t) >= min_len and t not in stop and not t.isdigit()
    ]
    counts = Counter(tokens)
    ranked = sorted(counts, key=lambda t: (-counts[t], -len(t), t))
    return ranked[:max_words]


def contains_banned_word(text: str, banned_words: Sequence[str]) -> List[str]:
    tokens = set(tokenize_words(text))
    banned = [normalize_token(w) for w in banned_words]
    return sorted({w for w in banned if w in tokens})


def clean_model_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = text.strip("\"'“”‘’ ")

    # Common instruction-tuned formats.
    for marker in ("Rewrite:", "Sentence:", "Output:", "Paraphrase:"):
        if marker.lower() in text.lower():
            parts = re.split(re.escape(marker), text, flags=re.IGNORECASE)
            text = parts[-1].strip()

    lines = [ln.strip(" -\t") for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[0]
    return text.strip("\"'“”‘’ ")


def validate_rewrite(text: str, banned_words: Sequence[str], band: LengthBand) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    n_words = count_words(text)
    if n_words < band.min_words or n_words > band.max_words:
        reasons.append(f"word_count={n_words}, expected={band.min_words}-{band.max_words}")
    hits = contains_banned_word(text, banned_words)
    if hits:
        reasons.append("banned_words=" + ",".join(hits))
    if not text or len(text) < 5:
        reasons.append("empty_or_too_short")
    return not reasons, reasons

