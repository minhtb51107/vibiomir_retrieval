#!/usr/bin/env python3
"""Audit and migrate ViBioMIR to the 2026-10-12 official corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hostname(url: str) -> str:
    return (urlsplit(str(url)).hostname or "").lower().removeprefix("www.")


def language_map() -> dict[str, str]:
    path = ROOT / "artifacts/source_census/technical_triage.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {source: row.get("likely_language", "unknown") for source, row in data["sources"].items()}


def corpus_arrays(path: Path) -> tuple[np.ndarray, int, bool]:
    table = pq.read_table(path, columns=["id"])
    ids = table.column("id").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    unique = np.unique(ids)
    return ids, len(unique), bool(len(ids) < 2 or np.all(ids[1:] >= ids[:-1]))


def corpus_diff(old_path: Path, new_path: Path) -> dict:
    old_ids, old_unique, old_sorted = corpus_arrays(old_path)
    new_ids, new_unique, new_sorted = corpus_arrays(new_path)
    retained = np.intersect1d(old_ids, new_ids, assume_unique=old_unique == len(old_ids) and new_unique == len(new_ids))
    removed = np.setdiff1d(old_ids, new_ids, assume_unique=old_unique == len(old_ids) and new_unique == len(new_ids))
    introduced = np.setdiff1d(new_ids, old_ids, assume_unique=old_unique == len(old_ids) and new_unique == len(new_ids))

    new_table = pq.read_table(new_path, columns=["id", "url"])
    raw_new_ids = new_table.column("id").combine_chunks().to_numpy(zero_copy_only=False)
    new_order = np.argsort(raw_new_ids, kind="stable")
    new_table_ids = raw_new_ids[new_order]
    new_urls = pc.take(new_table.column("url").combine_chunks(), pa.array(new_order, type=pa.int64()))
    language_by_domain = language_map()
    new_language = Counter()
    new_domains = Counter()
    for batch in pq.ParquetFile(new_path).iter_batches(columns=["url"], batch_size=65536):
        for url in batch.column(0).to_pylist():
            domain = hostname(url)
            new_domains[domain] += 1
            language = "en" if domain == "pubmed.ncbi.nlm.nih.gov" else language_by_domain.get(domain, "unknown")
            new_language[language] += 1

    retained_count = changed_urls = 0
    removed_language = Counter()
    unexpected_removed_domains = Counter()
    for batch in pq.ParquetFile(old_path).iter_batches(columns=["id", "url"], batch_size=65536):
        batch_ids = batch.column(0).to_numpy(zero_copy_only=False)
        batch_urls = batch.column(1).to_pylist()
        positions = np.searchsorted(new_table_ids, batch_ids)
        present = positions < len(new_table_ids)
        present[present] &= new_table_ids[positions[present]] == batch_ids[present]
        retained_count += int(present.sum())
        present_indexes = np.flatnonzero(present)
        if len(present_indexes):
            selected_new_urls = pc.take(new_urls, pa.array(positions[present_indexes], type=pa.int64())).to_pylist()
            changed_urls += sum(batch_urls[int(old_index)] != new_url for old_index, new_url in zip(present_indexes, selected_new_urls, strict=True))
        for index in np.flatnonzero(~present):
            domain = hostname(batch_urls[int(index)])
            language = language_by_domain.get(domain, "unknown")
            removed_language[language] += 1
            if language != "zh":
                unexpected_removed_domains[domain] += 1

    return {
        "old": {"path": str(old_path), "rows": len(old_ids), "unique_ids": old_unique, "duplicate_ids": len(old_ids) - old_unique, "ids_sorted": old_sorted, "sha256": sha256(old_path)},
        "new": {"path": str(new_path), "rows": len(new_ids), "unique_ids": new_unique, "duplicate_ids": len(new_ids) - new_unique, "ids_sorted": new_sorted, "sha256": sha256(new_path), "url_unique": int(pc.count_distinct(new_urls).as_py()), "url_duplicates": len(new_ids) - int(pc.count_distinct(new_urls).as_py())},
        "retained_ids": len(retained), "removed_ids": len(removed), "newly_introduced_ids": len(introduced),
        "retained_url_mapping_changes": changed_urls, "stream_retained_crosscheck": retained_count,
        "new_language_by_source_association": dict(sorted(new_language.items())),
        "new_top_domains": dict(new_domains.most_common(30)),
        "removed_language_by_source_association": dict(sorted(removed_language.items())),
        "unexpected_non_chinese_removed_top_domains": dict(unexpected_removed_domains.most_common(30)),
        "new_pubmed_ids": int(new_domains["pubmed.ncbi.nlm.nih.gov"]),
        "organizer_declared_language_composition": {"vi": 714469, "zh": 240000, "en": 240000},
        "organizer_change_accounting": {
            "retained_vietnamese_ids": 714469,
            "retained_chinese_ids": 240000,
            "new_english_ids": 240000,
            "removed_chinese_ids": len(removed) - int(removed_language.get("vi", 0)),
            "unexpected_vietnamese_removals": int(removed_language.get("vi", 0)),
            "unexpected_vietnamese_removed_domains": {domain: count for domain, count in unexpected_removed_domains.items() if domain.endswith(".vn") or domain == "vinmec.com"},
            "basis": "organizer-declared new totals plus exact ID-set/domain diff; the local old file contains 26,129 vinmec.com IDs absent from the new file",
        },
    }


def retained_local(documents: Path, chunks: Path, new_ids: np.ndarray, *, allowed_domains: set[str] | None = None) -> dict:
    retained_doc_ids: set[int] = set()
    all_doc_ids: set[int] = set()
    retained_domains = Counter()
    languages = Counter()
    mapping = language_map()
    for batch in pq.ParquetFile(documents).iter_batches(columns=["doc_id", "original_url", "extraction_status", "normalized_text"], batch_size=8192):
        for doc_id, url, status, text in zip(*(column.to_pylist() for column in batch.columns), strict=True):
            domain = hostname(url)
            if allowed_domains is not None and domain not in allowed_domains:
                continue
            doc_id = int(doc_id); all_doc_ids.add(doc_id)
            pos = int(np.searchsorted(new_ids, doc_id))
            if pos < len(new_ids) and int(new_ids[pos]) == doc_id and status == "SUCCESS" and str(text or "").strip():
                retained_doc_ids.add(doc_id); retained_domains[domain] += 1
                languages["en" if domain == "pubmed.ncbi.nlm.nih.gov" else mapping.get(domain, "unknown")] += 1
    retained_chunks = 0
    for batch in pq.ParquetFile(chunks).iter_batches(columns=["doc_id"], batch_size=65536):
        retained_chunks += sum(int(doc_id) in retained_doc_ids for doc_id in batch.column(0).to_pylist())
    return {"old_local_official_docs": len(all_doc_ids), "retained_official_docs": len(retained_doc_ids), "removed_official_docs": len(all_doc_ids-retained_doc_ids), "retained_usable_docs": len(retained_doc_ids), "retained_chunks": retained_chunks, "source_domain_count": len(retained_domains), "language_counts": dict(sorted(languages.items())), "source_counts": dict(retained_domains.most_common())}


def reconcile(config: dict, corpus: dict) -> dict:
    new_ids, _, _ = corpus_arrays(ROOT / config["inputs"]["official_corpus"])
    new_ids = np.sort(new_ids)
    broad_docs = ROOT / config["inputs"]["broad_documents"]
    broad_chunks = ROOT / config["inputs"]["broad_chunks"]
    broad = retained_local(broad_docs, broad_chunks, new_ids)
    census_roots = [ROOT / path for path in config["inputs"]["focused_roots"]]
    focused_allow = [
        {"medlatec.vn"},
        {"v.familydoctor.com.cn", "vinmec.com"},
        {"tiemchunglongchau.com.vn", "cancer.39.net", "suckhoedoisong.vn"},
        {"benhviennhitrunguong.gov.vn", "zydcd.com", "hellobacsi.com"},
    ]
    census_results = {}
    focused_parts=[]
    for path,allowed in zip(census_roots,focused_allow,strict=True):
        census_results[path.name] = retained_local(path / "documents.parquet", path / "chunks.parquet", new_ids)
        focused_parts.append(retained_local(path / "documents.parquet", path / "chunks.parquet", new_ids,allowed_domains=allowed))
    focused={
        "old_local_official_docs":sum(row["old_local_official_docs"] for row in focused_parts),
        "retained_official_docs":sum(row["retained_official_docs"] for row in focused_parts),
        "removed_official_docs":sum(row["removed_official_docs"] for row in focused_parts),
        "retained_usable_docs":sum(row["retained_usable_docs"] for row in focused_parts),
        "retained_chunks":sum(row["retained_chunks"] for row in focused_parts),
        "source_domain_count":len({source for row in focused_parts for source in row["source_counts"]}),
        "language_counts":dict(sum((Counter(row["language_counts"]) for row in focused_parts),Counter())),
        "source_counts":dict(sum((Counter(row["source_counts"]) for row in focused_parts),Counter())),
    }
    return {
        "focused_current_best": focused,
        "phase11a_m16_corpus": {**focused, "note": "M16 uses the same nine-source document corpus; candidate reconciliation remains pending"},
        "broad_existing": broad,
        "acquired_source_census_roots": census_results,
        "new_pubmed_official_ids": corpus["new_pubmed_ids"],
        "new_pubmed_with_local_content": broad["source_counts"].get("pubmed.ncbi.nlm.nih.gov", 0),
        "new_pubmed_without_local_content": corpus["new_pubmed_ids"] - broad["source_counts"].get("pubmed.ncbi.nlm.nih.gov", 0),
    }


def audit(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["outputs"]["artifacts"]
    diff = corpus_diff(ROOT / config["inputs"]["old_official_corpus"], ROOT / config["inputs"]["official_corpus"])
    if diff["new"]["rows"] != 1_194_469 or diff["new"]["duplicate_ids"]:
        raise ValueError(f"new official corpus invariant failed: {diff['new']}")
    local = reconcile(config, diff)
    result = {"status": "MIGRATION_AUDIT_COMPLETE", "official_corpus_diff": diff, "local_retention": local, "no_crawling": True}
    atomic_json(output / "migration_audit.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit"])
    parser.add_argument("--config", default="configs/phase11b_new_corpus.yaml")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT / args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
