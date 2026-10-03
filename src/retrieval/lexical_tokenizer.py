from __future__ import annotations

import unicodedata

from src.indexing.embedder import normalize_text


TOKENIZER_NAME = "unicode_words_han_unigrams_bigrams_v1"


def _is_han(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
    )


def tokenize_lexical(text: str) -> list[str]:
    """Tokenize without language models or dictionary-dependent segmentation.

    Latin and other non-Han letters/numbers form Unicode-aware word tokens.
    Each contiguous Han run emits namespaced character unigrams and bigrams.
    Namespaces prevent accidental collisions between token families.
    """

    normalized = normalize_text(text).casefold()
    tokens: list[str] = []
    word: list[str] = []
    han_run: list[str] = []

    def flush_word() -> None:
        if word:
            tokens.append("w:" + "".join(word))
            word.clear()

    def flush_han() -> None:
        if not han_run:
            return
        tokens.extend("h1:" + character for character in han_run)
        tokens.extend(
            "h2:" + han_run[index] + han_run[index + 1]
            for index in range(len(han_run) - 1)
        )
        han_run.clear()

    for character in normalized:
        if _is_han(character):
            flush_word()
            han_run.append(character)
            continue
        flush_han()
        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or (category.startswith("M") and word):
            word.append(character)
        else:
            flush_word()
    flush_han()
    flush_word()
    return tokens
