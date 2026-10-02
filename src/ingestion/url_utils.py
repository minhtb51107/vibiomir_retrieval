from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def prepare_fetch_url(original_url: str) -> str:
    """Strip only the fragment; preserve scheme, authority, path, and query."""
    parts = urlsplit(original_url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"unsupported or malformed URL: {original_url!r}")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def domain_from_url(url: str) -> str:
    hostname = urlsplit(url).hostname
    if not hostname:
        raise ValueError(f"URL has no hostname: {url!r}")
    domain = hostname.rstrip(".").lower()
    return domain[4:] if domain.startswith("www.") else domain
