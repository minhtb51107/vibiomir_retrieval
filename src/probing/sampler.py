from __future__ import annotations

import csv
import hashlib
import heapq
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq

from src.ingestion.url_utils import domain_from_url, prepare_fetch_url

from .models import ProbeConfig, SampleRecord, SamplingRule


def _stable_rank(seed: str, rule_name: str, doc_id: int, url: str) -> int:
    value = f"{seed}\0{rule_name}\0{doc_id}\0{url}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def _matches(rule: SamplingRule, url: str, domain: str) -> bool:
    parts = urlsplit(url)
    path = parts.path.lower()
    if rule.kind == "domain":
        assert rule.value is not None
        target = rule.value.lower()
        return domain == target or domain.endswith(f".{target}")
    if rule.kind == "query_string":
        return bool(parts.query)
    if rule.kind == "scheme":
        return parts.scheme.lower() == (rule.value or "").lower()
    if rule.kind == "suffix":
        return path.endswith((rule.value or "").lower())
    if rule.kind == "no_extension":
        final_component = path.rstrip("/").rsplit("/", 1)[-1]
        return not final_component or "." not in final_component
    raise ValueError(f"unknown sampling rule kind: {rule.kind}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_samples(config: ProbeConfig) -> tuple[list[SampleRecord], dict[str, int]]:
    """Select the lowest seeded hashes per rule in one full metadata scan."""
    input_path = Path(config.input_path)
    required = {"id", "url"}
    missing = required - set(pq.read_schema(input_path).names)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    # Entries are (-rank, -doc_id, doc_id, url, fetch_url, domain). Python's
    # min-heap then keeps the worst (largest rank) candidate at index zero.
    heaps: dict[str, list[tuple[int, int, int, str, str, str]]] = {
        rule.name: [] for rule in config.sampling_rules
    }
    rules_by_name = {rule.name: rule for rule in config.sampling_rules}

    parquet = pq.ParquetFile(input_path)
    for batch in parquet.iter_batches(
        columns=["id", "url"], batch_size=config.batch_size
    ):
        for raw_id, raw_url in zip(
            batch.column("id").to_pylist(),
            batch.column("url").to_pylist(),
            strict=True,
        ):
            if raw_id is None or raw_url is None:
                continue
            doc_id = int(raw_id)
            original_url = str(raw_url)
            try:
                fetch_url = prepare_fetch_url(original_url)
                domain = domain_from_url(fetch_url)
            except ValueError:
                continue
            for rule in config.sampling_rules:
                if not _matches(rule, fetch_url, domain):
                    continue
                rank = _stable_rank(config.seed, rule.name, doc_id, original_url)
                entry = (-rank, -doc_id, doc_id, original_url, fetch_url, domain)
                heap = heaps[rule.name]
                if len(heap) < rule.quota:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)

    selected: dict[int, SampleRecord] = {}
    rule_counts: dict[str, int] = {}
    for rule_name, heap in heaps.items():
        rule = rules_by_name[rule_name]
        rule_counts[rule_name] = len(heap)
        for _, _, doc_id, original_url, fetch_url, domain in heap:
            record = selected.setdefault(
                doc_id,
                SampleRecord(
                    doc_id=doc_id,
                    original_url=original_url,
                    fetch_url=fetch_url,
                    domain=domain,
                ),
            )
            record.sampling_categories.append(rule_name)
            if (
                rule.inferred_language
                and rule.inferred_language != "unknown"
                and record.inferred_language is None
            ):
                record.inferred_language = rule.inferred_language

    records = sorted(selected.values(), key=lambda item: (item.domain, item.doc_id))
    for record in records:
        record.sampling_categories.sort()
    if len(records) > config.maximum_live_urls:
        raise ValueError(
            f"selected {len(records)} URLs, above configured maximum {config.maximum_live_urls}"
        )
    return records, rule_counts


def write_manifest(
    config: ProbeConfig,
    records: list[SampleRecord],
    rule_counts: dict[str, int],
) -> tuple[Path, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sample_manifest.json"
    csv_path = output_dir / "sample_manifest.csv"
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_path": config.input_path,
            "input_sha256": _file_sha256(Path(config.input_path)),
            "seed": config.seed,
            "maximum_live_urls": config.maximum_live_urls,
            "selected_count": len(records),
            "rule_selected_counts": rule_counts,
            "rules": [
                {
                    "name": rule.name,
                    "kind": rule.kind,
                    "value": rule.value,
                    "quota": rule.quota,
                    "inferred_language": rule.inferred_language,
                }
                for rule in config.sampling_rules
            ],
        },
        "records": [record.as_dict() for record in records],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "doc_id",
                "domain",
                "inferred_language",
                "sampling_categories",
                "original_url",
                "fetch_url",
            ],
        )
        writer.writeheader()
        for record in records:
            row = record.as_dict()
            row["sampling_categories"] = ";".join(record.sampling_categories)
            writer.writerow(row)
    return json_path, csv_path


def load_manifest(path: str | Path) -> list[SampleRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SampleRecord(**item) for item in payload["records"]]
