from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from .models import CorpusRecord
from .url_utils import domain_from_url, prepare_fetch_url


def iter_corpus_records(
    path: str | Path,
    *,
    limit: int | None,
    domains: Sequence[str] = (),
    doc_ids: set[int] | None = None,
    start_after_id: int | None = None,
    batch_size: int = 65_536,
) -> Iterator[CorpusRecord]:
    """Stream selected records without materializing the 4.4M-row corpus."""
    parquet_path = Path(path)
    schema_names = set(pq.read_schema(parquet_path).names)
    missing = {"id", "url"} - schema_names
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {sorted(missing)}")

    normalized_domains = tuple(d.lower() for d in domains)
    per_domain_limit = None
    if limit is not None and normalized_domains:
        per_domain_limit = max(
            1, (limit + len(normalized_domains) - 1) // len(normalized_domains)
        )
    domain_counts = {domain: 0 for domain in normalized_domains}
    seen_ids: set[int] = set()
    yielded = 0

    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(columns=["id", "url"], batch_size=batch_size):
        ids = batch.column("id").to_pylist()
        urls = batch.column("url").to_pylist()
        for raw_id, raw_url in zip(ids, urls, strict=True):
            if raw_id is None or raw_url is None:
                continue
            doc_id = int(raw_id)
            if start_after_id is not None and doc_id <= start_after_id:
                continue
            if doc_ids is not None and doc_id not in doc_ids:
                continue
            original_url = str(raw_url)
            try:
                fetch_url = prepare_fetch_url(original_url)
                domain = domain_from_url(fetch_url)
            except ValueError:
                fetch_url = original_url.split("#", 1)[0]
                domain = ""
            if normalized_domains:
                if domain not in domain_counts:
                    continue
                if per_domain_limit is not None and domain_counts[domain] >= per_domain_limit:
                    continue
            if doc_id in seen_ids:
                raise ValueError(f"duplicate doc_id selected from corpus: {doc_id}")
            seen_ids.add(doc_id)
            if normalized_domains:
                domain_counts[domain] += 1
            yield CorpusRecord(doc_id, original_url, fetch_url)
            yielded += 1
            if limit is not None and yielded >= limit:
                return
            if doc_ids is not None and seen_ids == doc_ids:
                return
