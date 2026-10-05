from __future__ import annotations

import re
import unicodedata

WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)

SCAFFOLD = frozenset(
    {
        "ạ", "bác", "bác sĩ", "cho", "cho biết", "có", "của", "em", "giúp",
        "hỏi", "là", "mình", "muốn", "nhờ", "ơi", "tôi", "vậy", "xin",
        "được", "thế nào", "như thế nào", "có phải", "xin hỏi",
    }
)


def normalize_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def concise_vietnamese(text: str, *, maximum_tokens: int = 24) -> str:
    """Remove deterministic question scaffolding while preserving content words."""
    normalized = normalize_query(text)
    lowered = normalized.casefold()
    for phrase in sorted(SCAFFOLD, key=len, reverse=True):
        lowered = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", lowered)
    tokens = WORD.findall(lowered)
    cleaned = [
        token
        for token in tokens
        if token not in SCAFFOLD and (len(token) > 1 or token.isdigit())
    ]
    if not cleaned:
        return normalized
    # Preserve both the medical subject and question intent; only cap very long
    # narratives deterministically.
    return " ".join(cleaned[:maximum_tokens])


def build_variants(text: str) -> dict[str, str]:
    return {"Q0": normalize_query(text), "Q1": concise_vietnamese(text)}
