from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from urllib.parse import unquote


SEPARATORS = re.compile(r"[\s/\\._\-?&=+:#;,|()\[\]{}]+")
VIETNAMESE_CHARACTERS = frozenset(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
VIETNAMESE_URL_HINTS = frozenset(
    {
        "benh",
        "benhvien",
        "bacsi",
        "suckhoe",
        "thuoc",
        "duoc",
        "viem",
        "dieutri",
        "trieuchung",
        "phongkham",
    }
)


def is_han(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
    )


def accent_fold(value: str) -> str:
    replaced = value.replace("đ", "d").replace("Đ", "D")
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", replaced)
        if not unicodedata.category(character).startswith("M")
    )


def lexical_tokens(
    text: str,
    *,
    stopwords: frozenset[str] = frozenset(),
    accent_folded: bool = True,
) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    tokens: list[str] = []
    word: list[str] = []
    han_run: list[str] = []

    def flush_word() -> None:
        if not word:
            return
        value = "".join(word)
        folded = accent_fold(value).casefold()
        if value not in stopwords and folded not in stopwords:
            tokens.append("w:" + value)
            if accent_folded and folded != value:
                tokens.append("w:" + folded)
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
        if is_han(character):
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


@dataclass(frozen=True, slots=True)
class UrlTextFeatures:
    original_url: str
    domain: str
    scheme: str
    normalized_text: str
    tokens: tuple[str, ...]
    token_counts: tuple[tuple[str, int], ...]
    path_token_count: int
    numeric_ratio: float
    has_han: bool
    vietnamese_like: bool
    opaque: bool
    text_hash_u64: int


def _split_url(url: str) -> tuple[str, str, str, str]:
    without_fragment = url.split("#", 1)[0]
    if "://" in without_fragment:
        scheme, remainder = without_fragment.split("://", 1)
    else:
        scheme, remainder = "", without_fragment
    authority, separator, path_query = remainder.partition("/")
    if not separator and ("?" in authority):
        authority, path_query = authority.split("?", 1)
        path_query = "?" + path_query
    authority = authority.rsplit("@", 1)[-1]
    hostname = authority.split(":", 1)[0].strip("[]").casefold()
    path, question, query = path_query.partition("?")
    return scheme.casefold(), hostname, path, query if question else ""


def extract_url_features(
    url: str,
    *,
    stopwords: frozenset[str] = frozenset(),
    maximum_characters: int = 4096,
) -> UrlTextFeatures:
    original = str(url)
    scheme, hostname, path, query = _split_url(original[:maximum_characters])
    # Keep the host in the normalized representation for provenance and exact-text
    # diagnostics, but do not mix it into lexical evidence. Accent folding a
    # Vietnamese query term such as "cơm" to ``com`` must not match the .com TLD.
    # Domain evidence belongs to the explicit domain-prior/fallback layer.
    encoded_text = " ".join(part for part in (hostname, path, query) if part)
    decoded = unquote(encoded_text, errors="replace") if "%" in encoded_text else encoded_text
    normalized = unicodedata.normalize("NFC", SEPARATORS.sub(" ", decoded)).strip().casefold()
    lexical_encoded_text = " ".join(part for part in (path, query) if part)
    lexical_decoded = (
        unquote(lexical_encoded_text, errors="replace")
        if "%" in lexical_encoded_text
        else lexical_encoded_text
    )
    lexical_text = unicodedata.normalize(
        "NFC", SEPARATORS.sub(" ", lexical_decoded)
    ).strip().casefold()
    tokens = lexical_tokens(lexical_text, stopwords=stopwords, accent_folded=True)
    path_decoded = unquote(path, errors="replace") if "%" in path else path
    path_text = SEPARATORS.sub(" ", path_decoded)
    path_tokens = lexical_tokens(path_text, stopwords=stopwords, accent_folded=True)
    counts = Counter(tokens)
    alpha_numeric = [character for character in normalized if character.isalnum()]
    numeric_ratio = (
        sum(character.isdigit() for character in alpha_numeric) / len(alpha_numeric)
        if alpha_numeric
        else 1.0
    )
    has_han = any(is_han(character) for character in normalized)
    words = {token[2:] for token in tokens if token.startswith("w:")}
    vietnamese_like = any(character in VIETNAMESE_CHARACTERS for character in normalized) or bool(
        words.intersection(VIETNAMESE_URL_HINTS)
    )
    meaningful_words = [
        token for token in tokens if token.startswith("w:") and not token[2:].isdigit()
    ]
    opaque = numeric_ratio >= 0.5 or (not meaningful_words and not has_han) or len(tokens) <= 1
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
    return UrlTextFeatures(
        original_url=original,
        domain=hostname,
        scheme=scheme,
        normalized_text=normalized,
        tokens=tuple(tokens),
        token_counts=tuple(sorted(counts.items())),
        path_token_count=len(path_tokens),
        numeric_ratio=round(numeric_ratio, 6),
        has_han=has_han,
        vietnamese_like=vietnamese_like,
        opaque=opaque,
        text_hash_u64=int.from_bytes(digest, "little", signed=False),
    )
