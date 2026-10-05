from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import pyarrow.parquet as pq

UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
PERCENT = re.compile(r"%([0-9A-Fa-f]{2})")


def _decode_unreserved(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        character = chr(int(match.group(1), 16))
        return character if character in UNRESERVED else "%" + match.group(1).upper()
    return PERCENT.sub(replace, value)


def normalize_url(url: str, *, strip_tracking: frozenset[str] = frozenset()) -> str:
    text = unicodedata.normalize("NFC", str(url).strip())
    split = urlsplit(text)
    scheme = split.scheme.casefold()
    host = (split.hostname or "").casefold().rstrip(".")
    port = split.port
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = _decode_unreserved(split.path or "/")
    if path != "/":
        path = path.rstrip("/") or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=True)
        if key.casefold() not in strip_tracking
    ]
    query = urlencode(query_pairs, doseq=True, safe="/:@-._~", quote_via=quote)
    return urlunsplit((scheme, netloc, path, query, ""))


def canonical_key(url: str, *, strip_tracking: frozenset[str] = frozenset()) -> str:
    normalized = normalize_url(url, strip_tracking=strip_tracking)
    split = urlsplit(normalized)
    return urlunsplit(("", split.netloc, split.path, split.query, ""))


def hostname(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").casefold().rstrip(".")
    except ValueError:
        return ""


def map_result_urls(
    result_urls: list[str],
    *,
    corpus_path: str | Path,
    tracking_parameters: list[str],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Scan the official corpus once and resolve only the supplied result URLs."""
    tracking = frozenset(value.casefold() for value in tracking_parameters)
    exact_targets = set(result_urls)
    normalized_targets: dict[str, set[str]] = defaultdict(set)
    canonical_targets: dict[str, set[str]] = defaultdict(set)
    target_hosts = set()
    for url in result_urls:
        normalized_targets[normalize_url(url, strip_tracking=tracking)].add(url)
        canonical_targets[canonical_key(url, strip_tracking=tracking)].add(url)
        target_hosts.add(hostname(url))
    matches: dict[str, dict[str, set[tuple[int, str, str]]]] = {
        url: {"exact": set(), "normalized": set(), "canonical": set()}
        for url in result_urls
    }
    domain_counts: Counter[str] = Counter()
    parquet = pq.ParquetFile(corpus_path)
    for batch in parquet.iter_batches(batch_size=65536, columns=["id", "url"]):
        ids = batch.column(0).to_pylist()
        urls = batch.column(1).to_pylist()
        for doc_id, official in zip(ids, urls, strict=True):
            official = str(official)
            official_host = hostname(official)
            domain_counts[official_host] += 1
            if official in exact_targets:
                matches[official]["exact"].add((int(doc_id), official, official_host))
            if official_host not in target_hosts:
                continue
            normalized = normalize_url(official, strip_tracking=tracking)
            for result_url in normalized_targets.get(normalized, ()):
                matches[result_url]["normalized"].add((int(doc_id), official, official_host))
            canonical = canonical_key(official, strip_tracking=tracking)
            for result_url in canonical_targets.get(canonical, ()):
                matches[result_url]["canonical"].add((int(doc_id), official, official_host))
    output = []
    for result_url in result_urls:
        levels = matches[result_url]
        if len(levels["exact"]) == 1:
            candidates = levels["exact"]
            level = "exact"
        elif len(levels["exact"]) > 1:
            candidates = levels["exact"]
            level = None
        elif len(levels["normalized"]) == 1:
            candidates = levels["normalized"]
            level = "normalized"
        elif len(levels["normalized"]) > 1:
            candidates = levels["normalized"]
            level = None
        else:
            candidates = levels["canonical"]
            level = "canonical" if len(candidates) == 1 else None
        if len(candidates) == 1:
            candidate = next(iter(candidates))
            status = "exact_official_match" if level == "exact" else "normalized_official_match"
            output.append(
                {
                    "result_url": result_url,
                    "status": status,
                    "match_level": level,
                    "doc_id": candidate[0],
                    "official_url": candidate[1],
                    "official_domain": candidate[2],
                    "candidate_count": 1,
                }
            )
        elif len(candidates) > 1:
            output.append(
                {
                    "result_url": result_url,
                    "status": "ambiguous_match",
                    "match_level": None,
                    "doc_id": None,
                    "official_url": None,
                    "official_domain": None,
                    "candidate_count": len(candidates),
                }
            )
        else:
            output.append(
                {
                    "result_url": result_url,
                    "status": "no_official_match",
                    "match_level": None,
                    "doc_id": None,
                    "official_url": None,
                    "official_domain": None,
                    "candidate_count": 0,
                }
            )
    return output, dict(domain_counts)
