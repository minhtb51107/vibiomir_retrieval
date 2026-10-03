from __future__ import annotations

import codecs
import re
import unicodedata
from dataclasses import dataclass

from charset_normalizer import from_bytes


CHARSET_RE = re.compile(rb"charset\s*=\s*[\"']?([^\s;\"'>/]+)", re.IGNORECASE)
BOMS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


@dataclass(frozen=True, slots=True)
class DecodeResult:
    text: str
    selected_encoding: str
    http_encoding: str | None
    meta_encoding: str | None
    bom_encoding: str | None
    detected_encoding: str | None
    replacement_character_count: int
    nfc_changed: bool
    severe_decoding_problem: bool


def _encoding(name: str | None) -> str | None:
    if not name:
        return None
    try:
        return codecs.lookup(name.strip()).name
    except LookupError:
        return None


def _http_encoding(content_type: str | None) -> str | None:
    if not content_type:
        return None
    match = re.search(r"charset\s*=\s*[\"']?([^\s;\"'>]+)", content_type, re.I)
    return _encoding(match.group(1)) if match else None


def _meta_encoding(body: bytes) -> str | None:
    match = CHARSET_RE.search(body[:16_384])
    if not match:
        return None
    return _encoding(match.group(1).decode("ascii", errors="ignore"))


def decode_body(
    body: bytes,
    content_type_header: str | None,
    *,
    severe_replacement_ratio: float = 0.02,
) -> DecodeResult:
    http = _http_encoding(content_type_header)
    meta = _meta_encoding(body)
    bom = next((name for marker, name in BOMS if body.startswith(marker)), None)
    match = from_bytes(body).best() if body else None
    detected = _encoding(match.encoding if match is not None else None)
    candidates = [http, bom, meta, detected, "utf-8"]
    selected = "utf-8"
    decoded: str | None = None
    for candidate in dict.fromkeys(item for item in candidates if item):
        try:
            decoded = body.decode(candidate, errors="strict")
            selected = candidate
            break
        except (LookupError, UnicodeDecodeError):
            continue
    if decoded is None:
        decoded = body.decode("utf-8", errors="replace")
    normalized = unicodedata.normalize("NFC", decoded)
    replacements = normalized.count("\ufffd")
    severe = replacements / max(len(normalized), 1) > severe_replacement_ratio
    return DecodeResult(
        text=normalized,
        selected_encoding=selected,
        http_encoding=http,
        meta_encoding=meta,
        bom_encoding=bom,
        detected_encoding=detected,
        replacement_character_count=replacements,
        nfc_changed=normalized != decoded,
        severe_decoding_problem=severe,
    )
