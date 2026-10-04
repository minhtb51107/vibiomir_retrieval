"""Source-verbatim chunk context expansion for Phase 10A calibration.

A baseline chunk is a contiguous slice ``normalized_text[start:end]`` of its
processed source document. Expansion widens that slice into a larger contiguous
slice of the *same* document, targeting a token budget measured with a supplied
offset tokenizer (BGE-M3 in production). No text is generated, translated,
normalized, or rewritten: the emitted string is always exactly
``document_text[span_start:span_end]`` and always contains the original chunk.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Mapping

from src.processing.chunking import Tokenizer

# Snapping outward to a whitespace boundary avoids cutting a word in half, but
# is bounded so unspaced scripts (e.g. Chinese) never expand by more than this.
MAX_WORD_SNAP_CHARACTERS = 32


@dataclass(frozen=True, slots=True)
class ExpandedSpan:
    doc_id: int
    start: int
    end: int
    text: str
    original_start: int
    original_end: int
    original_tokens: int
    window_tokens: int
    covers_whole_document: bool

    @property
    def changed(self) -> bool:
        return (self.start, self.end) != (self.original_start, self.original_end)


@dataclass(frozen=True, slots=True)
class _TokenIndex:
    starts: list[int]
    ends: list[int]


def _snap_left(text: str, position: int) -> int:
    limit = max(0, position - MAX_WORD_SNAP_CHARACTERS)
    cursor = position
    while cursor > limit and cursor > 0 and not text[cursor - 1].isspace():
        cursor -= 1
    if cursor == 0 or (cursor > 0 and text[cursor - 1].isspace()):
        return cursor
    return position


def _snap_right(text: str, position: int) -> int:
    limit = min(len(text), position + MAX_WORD_SNAP_CHARACTERS)
    cursor = position
    while cursor < limit and cursor < len(text) and not text[cursor].isspace():
        cursor += 1
    if cursor == len(text) or (cursor < len(text) and text[cursor].isspace()):
        return cursor
    return position


def compute_window(
    *,
    token_count: int,
    first_token: int,
    last_token_exclusive: int,
    target_tokens: int,
) -> tuple[int, int]:
    """Return a token window [a, b) around the chunk with ``target_tokens`` tokens.

    The window is centred on the original chunk, clamped to the document, and any
    budget that cannot be spent on one side is spent on the other.
    """
    chunk_tokens = last_token_exclusive - first_token
    budget = max(0, target_tokens - chunk_tokens)
    before = budget // 2
    after = budget - before
    a = first_token - before
    b = last_token_exclusive + after
    if a < 0:
        b += -a
        a = 0
    if b > token_count:
        a -= b - token_count
        b = token_count
        a = max(a, 0)
    return a, b


class SourceWindowExpander:
    """Expand baseline chunk offsets into contiguous source windows."""

    def __init__(
        self,
        documents: Mapping[int, str],
        tokenizer: Tokenizer,
        target_tokens: int,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        self.documents = documents
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
        self._index: dict[int, _TokenIndex] = {}

    def _tokens(self, doc_id: int) -> _TokenIndex:
        index = self._index.get(doc_id)
        if index is None:
            spans = self.tokenizer.spans(self.documents[doc_id])
            index = _TokenIndex(
                starts=[int(span.start) for span in spans],
                ends=[int(span.end) for span in spans],
            )
            self._index[doc_id] = index
        return index

    def expand(
        self, *, doc_id: int, start: int, end: int, original_text: str
    ) -> ExpandedSpan:
        document = self.documents.get(doc_id)
        if document is None:
            raise KeyError(f"missing processed source document for doc_id {doc_id}")
        if not 0 <= start < end <= len(document):
            raise ValueError(f"invalid chunk offsets for doc_id {doc_id}")
        if document[start:end] != original_text:
            raise ValueError(f"chunk offsets do not reproduce chunk text for doc {doc_id}")
        index = self._tokens(doc_id)
        count = len(index.starts)
        first = bisect.bisect_left(index.starts, start)
        last_exclusive = bisect.bisect_right(index.ends, end)
        if count == 0 or last_exclusive <= first:
            return ExpandedSpan(
                doc_id, start, end, original_text, start, end, 0, 0, False
            )
        a, b = compute_window(
            token_count=count,
            first_token=first,
            last_token_exclusive=last_exclusive,
            target_tokens=self.target_tokens,
        )
        if a == 0:
            span_start = min(index.starts[0], start)
        else:
            span_start = min(_snap_left(document, index.starts[a]), start)
        if b == count:
            span_end = max(index.ends[count - 1], end)
        else:
            span_end = max(_snap_right(document, index.ends[b - 1]), end)
        span_start = max(0, min(span_start, start))
        span_end = min(len(document), max(span_end, end))
        text = document[span_start:span_end]
        if document[start:end] not in text:  # defensive; guaranteed by construction
            raise ValueError(f"expanded span lost original chunk for doc {doc_id}")
        return ExpandedSpan(
            doc_id=doc_id,
            start=span_start,
            end=span_end,
            text=text,
            original_start=start,
            original_end=end,
            original_tokens=last_exclusive - first,
            window_tokens=b - a,
            covers_whole_document=(a == 0 and b == count),
        )
