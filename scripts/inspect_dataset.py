#!/usr/bin/env python3
"""
scripts/inspect_dataset.py
==========================
Phase 1 — Dataset Understanding (metadata inspection only).

Inspects query.parquet and links_corpus.parquet and prints a structured
statistical report.  Optionally writes machine-readable stats to a JSON
file.  No data is mutated; no network requests are made.

Usage
-----
    python scripts/inspect_dataset.py \\
        --query  data/raw/query.parquet \\
        --corpus data/raw/links_corpus.parquet \\
        [--json-out artifacts/dataset_stats.json]

Requirements: Python 3.11+, pandas, pyarrow, numpy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ── Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError) ────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

WIDTH = 72


def banner(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def sub(title: str) -> None:
    print(f"\n-- {title}")


def kv(label: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{label:<42} {value}")


def require_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: {path} is missing required columns: {missing}\n"
            f"  Found: {list(df.columns)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Query inspection
# ─────────────────────────────────────────────────────────────────────────────

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def inspect_queries(path: Path) -> dict[str, Any]:
    """Inspect query.parquet and return a stats dict."""
    banner(f"QUERY FILE  —  {path.name}")

    # ── Schema (without loading data) ───────────────────────────────────────
    schema = pq.read_schema(path)
    sub("Schema / dtypes")
    for i, field in enumerate(schema):
        kv(f"  [{i}] {field.name}", str(field.type))

    # ── Load ─────────────────────────────────────────────────────────────────
    df = pd.read_parquet(path, engine="pyarrow")
    require_columns(df, ["id", "query"], path)

    # ── Row / ID counts ───────────────────────────────────────────────────────
    sub("Row counts")
    n_rows = len(df)
    n_unique_ids = int(df["id"].nunique())
    n_dup_ids = int(df["id"].duplicated().sum())
    kv("Total rows", n_rows)
    kv("Unique id count", n_unique_ids)
    kv("Duplicate id rows", n_dup_ids)

    # ── Missing ───────────────────────────────────────────────────────────────
    sub("Missing values")
    n_missing_id = int(df["id"].isna().sum())
    n_missing_query = int(df["query"].isna().sum())
    kv("Missing id", n_missing_id)
    kv("Missing query", n_missing_query)

    # ── Empty / whitespace ────────────────────────────────────────────────────
    sub("Empty / whitespace queries")
    null_mask = df["query"].isna()
    empty_mask = (~null_mask) & (df["query"].str.strip() == "")
    n_null_query = int(null_mask.sum())
    n_empty_query = int(empty_mask.sum())
    kv("NULL query rows", n_null_query)
    kv("Whitespace-only query rows", n_empty_query)

    # ── Length statistics ─────────────────────────────────────────────────────
    sub("Query character-length statistics")
    lengths = df["query"].dropna().str.len()
    len_stats = {
        "count": int(lengths.count()),
        "min": int(lengths.min()),
        "p25": float(lengths.quantile(0.25)),
        "median": float(lengths.median()),
        "mean": float(lengths.mean()),
        "p75": float(lengths.quantile(0.75)),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    }
    for k, v in len_stats.items():
        kv(k, f"{v:.1f}" if isinstance(v, float) else v)

    # ── Unicode anomalies ─────────────────────────────────────────────────────
    sub("Unicode / encoding anomalies")
    qs = df["query"].dropna()

    # Control chars: use vectorised regex (1 pass, compiled)
    n_ctrl = int(qs.str.contains(CONTROL_RE.pattern, regex=True).sum())
    # U+FFFD: plain substring search
    n_fffd = int(qs.str.contains("\ufffd", regex=False).sum())
    # Non-ASCII: plain regex
    n_non_ascii = int(qs.str.contains(r"[^\x00-\x7F]", regex=True).sum())
    # NFC: must iterate (no vectorised unicodedata)
    n_not_nfc = int(sum(unicodedata.normalize("NFC", s) != s for s in qs))

    kv("Rows with control characters", n_ctrl)
    kv("Rows containing U+FFFD (replacement char)", n_fffd)
    kv("Rows with non-ASCII characters", n_non_ascii)
    kv("Rows not in NFC normalization", n_not_nfc)

    # ── Examples ──────────────────────────────────────────────────────────────
    sub("Representative examples (first 5 rows)")
    for _, row in df.head(5).iterrows():
        print(f"   id={row['id']}  |  query={str(row['query'])[:120]!r}")

    df_len = df.assign(_len=df["query"].str.len())
    sub("Shortest 3 queries")
    for _, row in df_len.nsmallest(3, "_len").iterrows():
        print(f"   id={row['id']}  len={int(row['_len'])}  |  query={str(row['query'])[:120]!r}")
    sub("Longest 3 queries")
    for _, row in df_len.nlargest(3, "_len").iterrows():
        print(f"   id={row['id']}  len={int(row['_len'])}  |  query={str(row['query'])[:120]!r}")

    return {
        "path": str(path),
        "total_rows": n_rows,
        "unique_ids": n_unique_ids,
        "duplicate_id_rows": n_dup_ids,
        "missing_id": n_missing_id,
        "missing_query": n_missing_query,
        "null_query_rows": n_null_query,
        "whitespace_only_query_rows": n_empty_query,
        "length_stats": len_stats,
        "unicode": {
            "rows_with_control_chars": n_ctrl,
            "rows_with_ufffd": n_fffd,
            "rows_with_non_ascii": n_non_ascii,
            "rows_not_nfc": n_not_nfc,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Corpus inspection — optimised for 4.4M rows
# ─────────────────────────────────────────────────────────────────────────────

# Known file extensions for binary download detection
FILE_EXTS: frozenset[str] = frozenset({
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".xls", ".xlsx", ".epub", ".zip", ".tar", ".gz", ".rar", ".7z",
})

# Simple UTM/tracking prefix strings (plain substring, no regex needed)
TRACKING_SUBSTRINGS: tuple[str, ...] = (
    "utm_source=", "utm_medium=", "utm_campaign=",
    "utm_term=", "utm_content=",
    "fbclid=", "gclid=", "msclkid=",
)


def _fast_domain_counts(urls: pd.Series) -> tuple[Counter[str], int]:
    """
    Extract netloc using pandas str.split on fixed delimiters.
    str.split() does not use Python's regex engine — it uses the C-level
    str.split() which is much faster than str.extract for 4.4M rows.

    Algorithm:
      1. Split at '://' (max 1 split) -> take part after.
      2. Split at '/' (max 1 split) -> take part before (= netloc).
      3. Lowercase and strip 'www.' prefix.
    """
    urls = urls.fillna("")

    # Step 1: extract everything after '://'
    # str.split(n=1) returns a list; .str[1] takes the second element (or NaN)
    after_scheme = urls.str.split("://", n=1).str[1].fillna("")

    # Step 2: take everything before the first '/'
    # (this gives us "host:port" or just "host")
    netloc_with_port = after_scheme.str.split("/", n=1).str[0].fillna("")

    # Step 3: strip port numbers (:80, :8080, etc.)
    netloc = netloc_with_port.str.split(":", n=1).str[0].fillna("")

    # Step 4: lowercase, strip leading 'www.'
    netloc = netloc.str.lower()
    has_www = netloc.str.startswith("www.")
    netloc = netloc.where(~has_www, netloc.str[4:])

    # Empty strings -> __NO_DOMAIN__
    netloc = netloc.where(netloc != "", "__NO_DOMAIN__")

    c = Counter(netloc)
    return c, len(c)


def _fast_ext_counts(
    urls: pd.Series,
) -> Counter[str]:
    """
    Extract URL path file extensions using pandas str operations.

    Algorithm:
      1. Strip scheme (before '://').
      2. Strip query string (at '?') and fragment (at '#').
      3. After the host, take the path; find last '.' in path.
      4. Validate extension (1-6 alnum chars).
    """
    _NO = "__no_ext__"
    urls = urls.fillna("")

    # Step 1: strip scheme -> get 'host/path?query#frag'
    after_scheme = urls.str.split("://", n=1).str[1].fillna("")

    # Step 2: strip query string — take part before '?'
    no_query = after_scheme.str.split("?", n=1).str[0].fillna("")

    # Step 3: strip fragment — take part before '#'
    no_frag = no_query.str.split("#", n=1).str[0].fillna("")

    # Step 4: strip host (up to first '/') to get the path
    # host_and_path[1] is the path (or NaN if no '/' after host)
    parts = no_frag.str.split("/", n=1)
    path_part = parts.str[1].fillna("")  # empty string if no path

    # Step 5: get last component of path and its extension
    # Take everything after last '/' then after last '.'
    last_component = path_part.str.rsplit("/", n=1).str[-1].fillna("")

    # Step 6: take suffix after last '.'
    has_dot = last_component.str.contains(".", regex=False)
    suffix_raw = last_component.str.rsplit(".", n=1).str[-1].fillna("")

    # Build extension: '.' + suffix if suffix is valid alnum 1-6 chars
    # Validate with a simple regex (only on the final suffix string, not URLs)
    # This regex runs over a short 1-6 char string per row — negligible cost
    valid_suffix = suffix_raw.str.match(r"^[a-zA-Z0-9]{1,6}$")
    ext = suffix_raw.str.lower()

    # Combine conditions: must have a dot, must be valid, path must be non-empty
    ext_result = ext.where(has_dot & valid_suffix & (path_part != ""), _NO)
    ext_result = ext_result.fillna(_NO)
    # Add the dot prefix for non-placeholder values
    is_real = ext_result != _NO
    ext_result = ext_result.where(~is_real, "." + ext_result)

    return Counter(ext_result)


def _fast_id_gap_analysis(ids: pd.Series) -> dict[str, Any]:
    """
    Compute gap statistics using numpy diff on sorted IDs — O(n log n).
    """
    sorted_ids: np.ndarray = np.sort(ids.to_numpy())
    diffs = np.diff(sorted_ids)
    gap_mask = diffs > 1
    # A difference of N between adjacent IDs means N - 1 IDs are absent.
    missing_per_gap = diffs[gap_mask] - 1
    total_missing = int(sorted_ids[-1]) - int(sorted_ids[0]) + 1 - len(sorted_ids)
    return {
        "id_min": int(sorted_ids[0]),
        "id_max": int(sorted_ids[-1]),
        "gap_count": int(gap_mask.sum()),
        "total_missing_ids": total_missing,
        "max_gap_size": int(missing_per_gap.max()) if gap_mask.any() else 0,
        "all_gap_sizes": [int(g) for g in missing_per_gap.tolist()],
    }


def inspect_corpus(path: Path) -> dict[str, Any]:
    """Inspect links_corpus.parquet and return a stats dict."""
    banner(f"CORPUS FILE  —  {path.name}")

    # ── Schema ────────────────────────────────────────────────────────────────
    schema = pq.read_schema(path)
    sub("Schema / dtypes")
    for i, field in enumerate(schema):
        kv(f"  [{i}] {field.name}", str(field.type))

    # ── Load only needed columns ──────────────────────────────────────────────
    print("\n  [loading id + url columns ...]")
    t0 = time.perf_counter()
    df = pd.read_parquet(path, engine="pyarrow", columns=["id", "url"])
    require_columns(df, ["id", "url"], path)
    print(f"  [loaded {len(df):,} rows in {time.perf_counter() - t0:.1f}s]")

    total_rows = len(df)

    # ── Row / ID counts ───────────────────────────────────────────────────────
    sub("Row counts")
    unique_ids = int(df["id"].nunique())
    dup_id_rows = int(df["id"].duplicated().sum())
    kv("Total rows", total_rows)
    kv("Unique id count", unique_ids)
    kv("Duplicate id rows (non-first)", dup_id_rows)

    # ── ID gap analysis ───────────────────────────────────────────────────────
    sub("ID sequence analysis")
    gap_stats = _fast_id_gap_analysis(df["id"].dropna())
    id_min = gap_stats["id_min"]
    id_max = gap_stats["id_max"]
    expected_range = id_max - id_min + 1
    kv("ID range min", id_min)
    kv("ID range max", id_max)
    kv("Expected sequential IDs (max-min+1)", expected_range)
    kv("IDs match sequential range?", unique_ids == expected_range)
    kv("Gap positions in sequence", gap_stats["gap_count"])
    kv("Total missing IDs in range", gap_stats["total_missing_ids"])
    kv("Max single gap size", gap_stats["max_gap_size"])
    kv("All gap sizes", gap_stats["all_gap_sizes"])

    # ── Missing values ────────────────────────────────────────────────────────
    sub("Missing values")
    null_ids = int(df["id"].isna().sum())
    null_urls = int(df["url"].isna().sum())
    kv("Missing id", null_ids)
    kv("Missing url", null_urls)

    # ── URL uniqueness ────────────────────────────────────────────────────────
    sub("URL uniqueness")
    unique_urls = int(df["url"].nunique())
    dup_url_rows = int(df["url"].duplicated().sum())
    kv("Unique URL count", unique_urls)
    kv("Duplicate URL rows (non-first)", dup_url_rows)

    # ── Scheme detection — pyarrow compute for speed ──────────────────────────
    sub("HTTP vs HTTPS")
    # Use PyArrow string_is_ascii / starts_with for C-level speed
    url_arr: pa.ChunkedArray = pa.chunked_array(
        [pa.array(df["url"].to_numpy(dtype=object, na_value=None), type=pa.string())]
    )
    is_null_arr = pc.is_null(url_arr)
    is_https_arr = pc.starts_with(url_arr, pattern="https://")
    is_http_arr = pc.and_(
        pc.starts_with(url_arr, pattern="http://"),
        pc.invert(is_https_arr),
    )
    # Other scheme: not null, not https, not http, not empty string
    is_empty_arr = pc.equal(url_arr, "")
    # is_other = (not null) AND (not https) AND (not http) AND (not empty)
    is_other_arr = pc.and_(
        pc.and_(
            pc.and_(pc.invert(pc.is_null(url_arr)), pc.invert(is_https_arr)),
            pc.invert(is_http_arr),
        ),
        pc.invert(is_empty_arr),
    )

    n_https = int(pc.sum(is_https_arr).as_py())
    n_http = int(pc.sum(is_http_arr).as_py())
    n_other = int(pc.sum(is_other_arr).as_py())

    kv("HTTPS URLs", n_https)
    kv("HTTP URLs", n_http)
    kv("Other scheme / malformed", n_other)
    kv("Missing (null)", null_urls)

    # For sample rows we still need pandas boolean masks
    is_https_pd = is_https_arr.to_pandas()
    is_http_pd = pc.and_(pc.starts_with(url_arr, pattern="http://"), pc.invert(is_https_arr)).to_pandas()

    # ── Domain distribution ───────────────────────────────────────────────────
    sub("Domain distribution (top 30)")
    t1 = time.perf_counter()
    domain_counts, n_distinct_domains = _fast_domain_counts(df["url"])
    top_domains = domain_counts.most_common(30)
    print(f"  [domain extraction done in {time.perf_counter() - t1:.1f}s]")
    kv("Total distinct domains", n_distinct_domains)
    print()
    print(f"  {'Domain':<55} {'Count':>10}  {'%':>6}")
    print(f"  {'-'*55}  {'-'*10}  {'-'*6}")
    for domain, count in top_domains:
        pct = count / total_rows * 100
        print(f"  {domain:<55} {count:>10,}  {pct:>5.1f}%")

    # ── Extension distribution ────────────────────────────────────────────────
    sub("URL path extension distribution (top 20)")
    t2 = time.perf_counter()
    ext_counts = _fast_ext_counts(df["url"])
    top_exts = ext_counts.most_common(20)
    print(f"  [extension extraction done in {time.perf_counter() - t2:.1f}s]")
    for ext, count in top_exts:
        pct = count / total_rows * 100
        print(f"  {ext:<32} {count:>10,}  {pct:>5.1f}%")

    n_pdf_by_ext = ext_counts.get(".pdf", 0)
    n_file_by_ext = sum(ext_counts.get(e, 0) for e in FILE_EXTS)
    kv("URLs with .pdf extension", n_pdf_by_ext)
    kv("URLs with any known file extension", n_file_by_ext)

    # ── Query strings, tracking params, fragments (all PyArrow C-level) ───────
    sub("Query strings & tracking parameters")

    # pc.count_substring returns int per element; pc.sum aggregates — C-level
    n_query_str = int(pc.sum(
        pc.greater(pc.count_substring(url_arr, pattern="?"), 0)
    ).as_py())
    kv("URLs with query string (?)", n_query_str)

    # Check each tracking prefix with C-level count_substring; OR across them
    # Initialize to all False: cast a constant-zero array to bool
    false_arr: pa.ChunkedArray = pc.equal(url_arr, "__NOMATCH__9x__")
    has_tracking_arr = false_arr
    for ts in TRACKING_SUBSTRINGS:
        has_tracking_arr = pc.or_(
            has_tracking_arr,
            pc.greater(pc.count_substring(url_arr, pattern=ts, ignore_case=True), 0),
        )
    n_tracking = int(pc.sum(has_tracking_arr).as_py())
    kv("URLs with likely tracking parameters", n_tracking)

    n_fragment = int(pc.sum(
        pc.greater(pc.count_substring(url_arr, pattern="#"), 0)
    ).as_py())
    kv("URLs with fragments (#)", n_fragment)

    # A case-insensitive '.pdf' anywhere in the URL is a deliberately broad
    # metadata-only signal. It does not establish the response Content-Type.
    n_likely_pdf = int(pc.sum(
        pc.greater(pc.count_substring(url_arr, pattern=".pdf", ignore_case=True), 0)
    ).as_py())
    kv("URLs with likely PDF pattern (.pdf)", n_likely_pdf)

    # ── Representative examples ───────────────────────────────────────────────
    sub("Representative examples (first 5 rows)")
    for _, row in df.head(5).iterrows():
        print(f"   id={row['id']}  |  url={str(row['url'])[:100]!r}")

    sub("Sample HTTPS URLs")
    for _, row in df[is_https_pd].head(3).iterrows():
        print(f"   id={row['id']}  |  url={str(row['url'])[:100]!r}")

    sub("Sample HTTP URLs")
    for _, row in df[is_http_pd].head(3).iterrows():
        print(f"   id={row['id']}  |  url={str(row['url'])[:100]!r}")

    if n_other > 0:
        is_other_pd = is_other_arr.to_pandas()
        sub("Sample other-scheme / malformed URLs (up to 5)")
        for _, row in df[is_other_pd].head(5).iterrows():
            print(f"   id={row['id']}  |  url={str(row['url'])[:100]!r}")

    if n_pdf_by_ext > 0:
        sub("Sample .pdf extension URLs (up to 5)")
        pdf_mask = df["url"].str.contains(".pdf", case=False, regex=False, na=False)
        for _, row in df[pdf_mask].head(5).iterrows():
            print(f"   id={row['id']}  |  url={str(row['url'])[:100]!r}")

    # ── ID integrity summary ──────────────────────────────────────────────────
    banner("ID INTEGRITY CHECK")
    kv("Total rows", total_rows)
    kv("Unique IDs", unique_ids)
    kv("Duplicate ID rows", dup_id_rows)
    kv("ID range", f"{id_min} - {id_max}")
    kv("IDs are unique (safe as doc_id)?", dup_id_rows == 0)
    if dup_id_rows > 0:
        dup_ex = df[df["id"].duplicated(keep=False)].sort_values("id").head(10)
        sub("Duplicate ID examples")
        for _, row in dup_ex.iterrows():
            print(f"   id={row['id']}  |  url={str(row['url'])[:80]!r}")

    return {
        "path": str(path),
        "total_rows": total_rows,
        "unique_ids": unique_ids,
        "duplicate_id_rows": dup_id_rows,
        "missing_id": null_ids,
        "missing_url": null_urls,
        "unique_urls": unique_urls,
        "duplicate_url_rows": dup_url_rows,
        "id_sequence": gap_stats,
        "scheme": {
            "https": n_https,
            "http": n_http,
            "other_or_malformed": n_other,
            "missing_null": null_urls,
        },
        "domains": {
            "distinct_count": n_distinct_domains,
            "top_30": [{"domain": d, "count": c} for d, c in top_domains],
        },
        "extensions": {
            "top_20": [{"ext": e, "count": c} for e, c in top_exts],
            "pdf_extension_count": n_pdf_by_ext,
            "any_file_extension_count": n_file_by_ext,
        },
        "url_patterns": {
            "has_query_string": n_query_str,
            "has_likely_tracking_parameter": n_tracking,
            "has_fragment": n_fragment,
            "has_likely_pdf_pattern": n_likely_pdf,
        },
        "notes": {
            "pdf_pattern_note": (
                "PDF-like URL metadata does not confirm HTTP Content-Type. "
                "Actual content type is unknown until Phase 2 crawling."
            ),
            "language_note": (
                "Domain TLD and hostname are used as language proxies (.vn=Vietnamese, "
                ".cn/.net with Chinese names=Chinese). Actual page language requires "
                "content inspection and has not been measured."
            ),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 — Dataset inspection for ViBioMIR retrieval project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=Path,
        default=Path("data/raw/query.parquet"),
        help="Path to query.parquet",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/raw/links_corpus.parquet"),
        help="Path to links_corpus.parquet",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Skip corpus inspection (faster, for query-only runs)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/dataset_stats.json"),
        help="Write machine-readable stats to this JSON file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths_to_check = [args.query] + ([] if args.skip_corpus else [args.corpus])
    for p in paths_to_check:
        if not p.exists():
            sys.exit(f"ERROR: File not found: {p}")

    t_start = time.perf_counter()

    all_stats: dict[str, Any] = {}

    all_stats["query"] = inspect_queries(args.query)

    if not args.skip_corpus:
        all_stats["corpus"] = inspect_corpus(args.corpus)

    elapsed = time.perf_counter() - t_start

    # ── Performance summary ───────────────────────────────────────────────────
    banner("PERFORMANCE SUMMARY")
    kv("Total runtime (s)", f"{elapsed:.1f}")

    all_stats["run_meta"] = {
        "runtime_seconds": round(elapsed, 2),
    }

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print(f"\n  [stats written to {args.json_out}]")

    banner("DONE — No data was mutated.  No network requests were made.")


if __name__ == "__main__":
    main()
