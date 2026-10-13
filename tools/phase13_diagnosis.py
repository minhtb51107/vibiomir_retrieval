#!/usr/bin/env python3
"""Read-only Phase 12 failure diagnosis and Phase 13 planning artifacts."""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.union_experiment import _iter_json_array


OUT = ROOT / "artifacts/phase13_diagnosis"
P11_JSON = ROOT / "submissions/00_READY_TO_UPLOAD/phase11c_BROAD_EXISTING_NEW_DOCS20_CHUNKS40.json"
P12_JSON = ROOT / "submissions/00_READY_TO_UPLOAD/phase12_FULL_PUBMED_DENSE_DOCS20_CHUNKS40.json"
P11_ROOT = ROOT / "data/source_census/phase11c_broad_existing_new"
P12_ROOT = ROOT / "data/pubmed_phase12/retrieval"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def hostname(url: str) -> str:
    return (urlsplit(str(url)).hostname or "unknown").lower().removeprefix("www.")


def distribution(values: list[float | int]) -> dict:
    if not values:
        return {"count": 0, "min": None, "p5": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "min": float(array.min()), "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def relative_inversions(old: list, new: list) -> int:
    retained = [value for value in old if value in set(new)]
    positions = {value: index for index, value in enumerate(new)}
    order = [positions[value] for value in retained]
    return sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))


def topic_bucket(text: str) -> str:
    lowered = text.lower()
    buckets = {
        "tiêu hóa": ("dạ dày", "ruột", "tiêu hóa", "gan", "trĩ"),
        "hô hấp": ("phổi", "ho ", "hô hấp", "hen", "mũi", "họng"),
        "tim mạch": ("tim", "huyết áp", "mạch", "đột quỵ"),
        "sản-phụ khoa": ("thai", "kinh nguyệt", "âm đạo", "tử cung", "buồng trứng"),
        "nhi khoa": ("trẻ", "bé ", "sơ sinh"),
        "ung thư": ("ung thư", "khối u", "u ác"),
        "da liễu": ("da ", "mụn", "ngứa", "tóc"),
        "cơ-xương-khớp": ("xương", "khớp", "cột sống", "đau lưng"),
        "nội tiết-chuyển hóa": ("tiểu đường", "đái tháo đường", "tuyến giáp", "cholesterol"),
        "thuốc-điều trị": ("thuốc", "liều", "điều trị", "tác dụng phụ"),
    }
    for name, words in buckets.items():
        if any(word in lowered for word in words):
            return name
    return "khác"


def diverse_selection(ordered_rows: list[dict], queries: dict[int, str], count: int, excluded: set[int] | None = None) -> list[int]:
    """Round-robin a ranked diagnostic band across observed medical topics."""
    excluded = excluded or set(); buckets: dict[str, list[int]] = defaultdict(list)
    for row in ordered_rows:
        query_id = int(row["query_id"])
        if query_id not in excluded:
            buckets[topic_bucket(queries[query_id])].append(query_id)
    selected=[]; names=sorted(buckets)
    while len(selected) < count and names:
        remaining=[]
        for name in names:
            if buckets[name] and len(selected) < count:
                selected.append(buckets[name].pop(0))
            if buckets[name]:
                remaining.append(name)
        names=remaining
    if len(selected) != count:
        raise ValueError(f"unable to select {count} diverse review queries")
    return selected


def load_queries() -> dict[int, str]:
    table = pq.read_table(ROOT / "data/raw/query.parquet", columns=["id", "query"])
    return {int(row["id"]): str(row["query"]) for row in table.to_pylist()}


def compare_outputs(pubmed_doc_ids: set[int]) -> tuple[dict, dict[int, dict]]:
    per_query: dict[int, dict] = {}
    totals = Counter()
    for old, new in zip(_iter_json_array(P11_JSON), _iter_json_array(P12_JSON), strict=True):
        query_id = int(old["id"])
        if query_id != int(new["id"]):
            raise ValueError(f"query order mismatch at {query_id}")
        old_docs = list(map(int, old["relevant_docs"])); new_docs = list(map(int, new["relevant_docs"]))
        old_chunks = [(int(row["doc_id"]), str(row["chunk_text"])) for row in old["relevant_chunks"]]
        new_chunks = [(int(row["doc_id"]), str(row["chunk_text"])) for row in new["relevant_chunks"]]
        retained_docs = [doc for doc in old_docs if doc in set(new_docs)]
        displaced_docs = [doc for doc in old_docs if doc not in set(new_docs)]
        inserted_docs = [doc for doc in new_docs if doc not in set(old_docs)]
        retained_chunks = [row for row in old_chunks if row in set(new_chunks)]
        displaced_chunks = [row for row in old_chunks if row not in set(new_chunks)]
        inserted_chunks = [row for row in new_chunks if row not in set(old_chunks)]
        pubmed_at = {k: sum(doc in pubmed_doc_ids for doc in new_docs[:k]) for k in (5, 10, 20)}
        pubmed_chunks_at = {k: sum(doc in pubmed_doc_ids for doc, _ in new_chunks[:k]) for k in (10, 20, 40)}
        row = {
            "query_id": query_id,
            "retained_phase11c_documents": retained_docs,
            "displaced_phase11c_documents": displaced_docs,
            "inserted_documents": inserted_docs,
            "inserted_pubmed_documents": [doc for doc in inserted_docs if doc in pubmed_doc_ids],
            "document_rank_changes": sum(old_docs.index(doc) != new_docs.index(doc) for doc in retained_docs),
            "document_relative_order_inversions": relative_inversions(old_docs, new_docs),
            "retained_chunk_count": len(retained_chunks), "displaced_chunk_count": len(displaced_chunks),
            "inserted_chunk_count": len(inserted_chunks),
            "inserted_pubmed_chunk_count": sum(doc in pubmed_doc_ids for doc, _ in inserted_chunks),
            "chunk_relative_order_inversions": relative_inversions(old_chunks, new_chunks),
            "pubmed_documents_at": pubmed_at, "pubmed_chunks_at": pubmed_chunks_at,
            "phase11c_chunk_owners_absent_from_top20_docs": sum(doc not in set(old_docs) for doc, _ in old_chunks),
            "phase12_chunk_owners_absent_from_top20_docs": sum(doc not in set(new_docs) for doc, _ in new_chunks),
        }
        row["document_chunk_disagreement"] = abs(len(displaced_docs) / 20 - len(displaced_chunks) / 40)
        per_query[query_id] = row
        totals.update({
            "retained_document_slots": len(retained_docs), "displaced_document_slots": len(displaced_docs),
            "inserted_document_slots": len(inserted_docs), "inserted_pubmed_document_slots": len(row["inserted_pubmed_documents"]),
            "retained_chunk_slots": len(retained_chunks), "displaced_chunk_slots": len(displaced_chunks),
            "inserted_chunk_slots": len(inserted_chunks), "inserted_pubmed_chunk_slots": row["inserted_pubmed_chunk_count"],
            "queries_docs_changed": bool(displaced_docs), "queries_chunks_changed": bool(displaced_chunks),
            "document_relative_order_inversions": row["document_relative_order_inversions"],
            "chunk_relative_order_inversions": row["chunk_relative_order_inversions"],
        })
    rows = list(per_query.values())
    largest = sorted(rows, key=lambda r: (len(r["inserted_pubmed_documents"]), r["inserted_pubmed_chunk_count"], -r["query_id"]), reverse=True)[:20]
    largest_ids = {row["query_id"] for row in largest}
    least = sorted((r for r in rows if r["query_id"] not in largest_ids), key=lambda r: (len(r["inserted_pubmed_documents"]), r["inserted_pubmed_chunk_count"], r["query_id"]))[:20]
    excluded = largest_ids | {row["query_id"] for row in least}
    disagreement = sorted((r for r in rows if r["query_id"] not in excluded), key=lambda r: (r["document_chunk_disagreement"], len(r["displaced_phase11c_documents"]), r["query_id"]), reverse=True)[:20]
    summary = {
        "queries": len(rows), **dict(totals),
        "retained_docs_per_query": distribution([len(r["retained_phase11c_documents"]) for r in rows]),
        "displaced_docs_per_query": distribution([len(r["displaced_phase11c_documents"]) for r in rows]),
        "retained_chunks_per_query": distribution([r["retained_chunk_count"] for r in rows]),
        "displaced_chunks_per_query": distribution([r["displaced_chunk_count"] for r in rows]),
        "pubmed_docs_at_5": distribution([r["pubmed_documents_at"][5] for r in rows]),
        "pubmed_docs_at_10": distribution([r["pubmed_documents_at"][10] for r in rows]),
        "pubmed_docs_at_20": distribution([r["pubmed_documents_at"][20] for r in rows]),
        "pubmed_chunks_at_10": distribution([r["pubmed_chunks_at"][10] for r in rows]),
        "pubmed_chunks_at_20": distribution([r["pubmed_chunks_at"][20] for r in rows]),
        "pubmed_chunks_at_40": distribution([r["pubmed_chunks_at"][40] for r in rows]),
        "highlight_query_ids": {
            "largest_pubmed_displacement": [r["query_id"] for r in largest],
            "least_pubmed_displacement": [r["query_id"] for r in least],
            "major_document_chunk_disagreement": [r["query_id"] for r in disagreement],
        },
    }
    return summary, per_query


def candidate_sources(parts_root: Path) -> tuple[dict[str, str], dict[int, str], dict[str, dict]]:
    chunk_source: dict[str, str] = {}
    document_arm: dict[int, str] = {}
    stats: dict[str, dict] = {}
    for marker_path in sorted(parts_root.glob("*.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8")); source = str(marker["source"])
        docs: set[int] = set(); rows = 0
        for batch in pq.ParquetFile(marker_path.with_suffix(".parquet")).iter_batches(columns=["chunk_id", "doc_id"], batch_size=32768):
            for chunk_id, doc_id in zip(*(column.to_pylist() for column in batch.columns), strict=True):
                chunk_id = str(chunk_id)
                previous = chunk_source.setdefault(chunk_id, source)
                if previous != source:
                    raise ValueError(f"chunk source conflict: {chunk_id}")
                doc_id = int(doc_id); docs.add(doc_id); rows += 1
                previous_arm = document_arm.setdefault(doc_id, source)
                if previous_arm != source:
                    raise ValueError(f"document candidate-arm conflict: {doc_id}: {previous_arm} vs {source}")
        stats[source] = {"candidate_rows": rows, "unique_candidate_documents": len(docs), **marker}
    return chunk_source, document_arm, stats


def score_analysis(chunk_source: dict[str, str]) -> tuple[dict, dict[tuple[int, str], float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    per_query_source: dict[tuple[int, str], list[float]] = defaultdict(list)
    score_lookup: dict[tuple[int, str], float] = {}
    connection = sqlite3.connect(P12_ROOT / "round1/source_scores.sqlite")
    try:
        for query_id, chunk_id, score in connection.execute("SELECT query_id,chunk_id,rerank_score FROM pairs"):
            source = chunk_source[str(chunk_id)]; value = float(score)
            scores[source].append(value); per_query_source[(int(query_id), source)].append(value)
            score_lookup[(int(query_id), str(chunk_id))] = value
    finally:
        connection.close()
    pilot_path = ROOT / "data/source_census/phase11b_new_corpus/round1/pilot_rerank_pool.new_official.parquet"
    pilot_keys=set()
    for batch in pq.ParquetFile(pilot_path).iter_batches(columns=["query_id","chunk_id"],batch_size=32768):
        pilot_keys.update((int(query_id),str(chunk_id)) for query_id,chunk_id in zip(*(column.to_pylist() for column in batch.columns),strict=True))
    pilot_db = sqlite3.connect(ROOT / "data/retrieval/phase7_reranking/rerank_scores.sqlite")
    try:
        for query_id, chunk_id, score in pilot_db.execute("SELECT query_id,chunk_id,rerank_score FROM scores"):
            if (int(query_id),str(chunk_id)) not in pilot_keys:
                continue
            value = float(score); scores["ORIGINAL_PILOT"].append(value)
            per_query_source[(int(query_id), "ORIGINAL_PILOT")].append(value)
            score_lookup[(int(query_id), str(chunk_id))] = value
    finally:
        pilot_db.close()
    result = {}
    for source, values in sorted(scores.items()):
        margins = []
        query_maxima = []
        for (query_id, current), rows in per_query_source.items():
            if current != source:
                continue
            ordered = sorted(rows, reverse=True); query_maxima.append(ordered[0])
            if len(ordered) > 1:
                margins.append(ordered[0] - ordered[1])
        result[source] = {
            "candidate_score_distribution": distribution(values),
            "per_query_max_score_distribution": distribution(query_maxima),
            "per_query_top1_minus_top2_margin": distribution(margins),
        }
    return result, score_lookup


def document_metadata(paths: list[Path]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for path in paths:
        columns = [name for name in ("doc_id", "original_url", "title", "language_signal") if name in pq.ParquetFile(path).schema_arrow.names]
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=16384):
            for row in batch.to_pylist():
                doc_id = int(row["doc_id"]); url = str(row.get("original_url") or "")
                result.setdefault(doc_id, {
                    "source": hostname(url), "url": url, "title": str(row.get("title") or ""),
                    "language": "en" if hostname(url) == "pubmed.ncbi.nlm.nih.gov" else str(row.get("language_signal") or "unknown"),
                })
    return result


def ranking_top_details(root: Path, group: str, chunk_source: dict[str, str], score_lookup: dict[tuple[int, str], float]) -> tuple[dict[int, list[dict]], dict[int, dict[int, float]]]:
    chunks: dict[int, list[dict]] = defaultdict(list); doc_scores: dict[int, dict[int, float]] = defaultdict(dict)
    path = root / f"rankings/{group}/reranked_chunks.parquet"
    for batch in pq.ParquetFile(path).iter_batches(columns=["query_id", "rerank_rank", "chunk_id", "doc_id", "chunk_text"], batch_size=8192):
        for row in batch.to_pylist():
            query_id = int(row["query_id"]); rank = int(row["rerank_rank"])
            chunk_id = str(row["chunk_id"]); doc_id = int(row["doc_id"])
            score = score_lookup[(query_id, chunk_id)]
            source = chunk_source.get(chunk_id, "ORIGINAL_PILOT")
            if rank <= 40:
                chunks[query_id].append({"rank": rank, "chunk_id": chunk_id, "doc_id": doc_id, "arm": source, "reranker_score": score, "text": str(row["chunk_text"])})
            if doc_id not in doc_scores[query_id]:
                doc_scores[query_id][doc_id] = score
    return chunks, doc_scores


def enrich_highlights(summary: dict, per_query: dict[int, dict], queries: dict[int, str], metadata: dict[int, dict], p11_chunks: dict, p12_chunks: dict, p11_doc_scores: dict, p12_doc_scores: dict) -> None:
    selected = {qid for values in summary["highlight_query_ids"].values() for qid in values}
    details = {}
    p11_records = {int(row["id"]): row for row in _iter_json_array(P11_JSON) if int(row["id"]) in selected}
    p12_records = {int(row["id"]): row for row in _iter_json_array(P12_JSON) if int(row["id"]) in selected}
    for query_id in sorted(selected):
        old = p11_records[query_id]; new = p12_records[query_id]; base = per_query[query_id]
        def docs(record, scores):
            return [{"rank": rank, "doc_id": int(doc), **metadata.get(int(doc), {"source": "unknown", "title": "", "language": "unknown", "url": ""}), "reranker_score": scores.get(query_id, {}).get(int(doc))} for rank, doc in enumerate(record["relevant_docs"], 1)]
        details[str(query_id)] = {
            "query_vi": queries[query_id], "topic_bucket": topic_bucket(queries[query_id]),
            "metrics": base, "phase11c_documents": docs(old, p11_doc_scores), "phase12_documents": docs(new, p12_doc_scores),
            "phase11c_top_chunks": [{**row, "source": metadata.get(row["doc_id"], {}).get("source", "unknown"), "text": row["text"][:1200]} for row in p11_chunks[query_id][:10]],
            "phase12_top_chunks": [{**row, "source": metadata.get(row["doc_id"], {}).get("source", "unknown"), "text": row["text"][:1200]} for row in p12_chunks[query_id][:10]],
        }
    summary["highlight_details"] = details


def final_source_composition(p12_records: dict[int, dict], metadata: dict[int, dict], document_arm: dict[int, str]) -> dict:
    doc_by_rank = defaultdict(Counter); chunk_by_rank = defaultdict(Counter); doc_slots = Counter(); chunk_slots = Counter(); arm_docs = Counter(); arm_chunks = Counter(); unique_docs = defaultdict(set)
    for query_id, record in p12_records.items():
        for rank, doc in enumerate(record["relevant_docs"], 1):
            doc = int(doc); source = metadata.get(doc, {}).get("source", "unknown"); arm = document_arm.get(doc, "UNMAPPED")
            doc_by_rank[rank][source] += 1; doc_slots[source] += 1; arm_docs[arm] += 1; unique_docs[source].add(doc)
        for rank, row in enumerate(record["relevant_chunks"], 1):
            doc = int(row["doc_id"]); source = metadata.get(doc, {}).get("source", "unknown"); arm = document_arm.get(doc, "UNMAPPED")
            chunk_by_rank[rank][source] += 1; chunk_slots[source] += 1; arm_chunks[arm] += 1
    return {
        "document_slots_by_domain": dict(doc_slots.most_common()), "unique_final_documents_by_domain": {key: len(value) for key, value in sorted(unique_docs.items())},
        "chunk_slots_by_domain": dict(chunk_slots.most_common()), "document_slots_by_candidate_arm": dict(arm_docs.most_common()), "chunk_slots_by_candidate_arm": dict(arm_chunks.most_common()),
        "document_source_by_rank": {str(rank): dict(counts.most_common()) for rank, counts in sorted(doc_by_rank.items())},
        "chunk_arm_by_rank": {str(rank): dict(counts.most_common()) for rank, counts in sorted(chunk_by_rank.items())},
    }


def coverage_and_funnel(candidate_stats: dict, composition: dict) -> dict:
    migration = json.loads((ROOT / "artifacts/phase11b_new_corpus/migration_audit.json").read_text(encoding="utf-8"))
    broad = migration["local_retention"]["broad_existing"]
    processing = json.loads((ROOT / "artifacts/phase12_pubmed/processing_quality_audit.json").read_text(encoding="utf-8"))
    official_domains = Counter()
    for batch in pq.ParquetFile(ROOT / "data/raw/links_corpus_vi_en_zh.parquet").iter_batches(columns=["url"], batch_size=65536):
        official_domains.update(hostname(url) for url in batch.column(0).to_pylist())
    local_domains = Counter(broad["source_counts"]); local_domains["pubmed.ncbi.nlm.nih.gov"] = processing["valid_documents"]
    language_map = json.loads((ROOT / "artifacts/source_census/technical_triage.json").read_text(encoding="utf-8"))["sources"]
    missing_domains = []
    for domain, count in official_domains.most_common():
        local = int(local_domains.get(domain, 0)); language = "en" if domain == "pubmed.ncbi.nlm.nih.gov" else language_map.get(domain, {}).get("likely_language", "unknown")
        missing_domains.append({"domain": domain, "language_association": language, "official_documents": count, "local_usable_documents": local, "missing_local_content": count - local, "coverage_fraction": local / count if count else 0.0})
    pilot_path = ROOT / "data/source_census/phase11b_new_corpus/round1/pilot_rerank_pool.new_official.parquet"
    pilot_pool = pq.ParquetFile(pilot_path)
    candidate_stats["ORIGINAL_PILOT"] = {"candidate_rows": pilot_pool.metadata.num_rows, "unique_candidate_documents": len(set(pq.read_table(pilot_path, columns=["doc_id"]).column(0).to_pylist())), "retrieval": "frozen historical pilot candidate pool"}
    focused_counts = migration["local_retention"]["focused_current_best"]["source_counts"]
    final_chunk_slots = composition["chunk_slots_by_candidate_arm"]
    final_doc_slots = composition["document_slots_by_candidate_arm"]
    retrieval = {
        "PUBMED": "BGE-M3 dense top-100 documents/query, then every evidence chunk from those documents",
        "BROAD_EXTRA": "global BM25 top-100 chunks/query",
        "ORIGINAL_PILOT": "frozen historical pilot hybrid/reranker pool",
    }
    funnel = {}
    for source, stats in sorted(candidate_stats.items()):
        indexed_docs = processing["valid_documents"] if source == "PUBMED" else (9051 if source == "BROAD_EXTRA" else (stats["unique_candidate_documents"] if source == "ORIGINAL_PILOT" else focused_counts.get(source)))
        rows = int(stats["candidate_rows"]); final_chunks = int(final_chunk_slots.get(source, 0))
        final_docs = int(final_doc_slots.get(source, 0))
        funnel[source] = {
            "indexed_valid_documents": indexed_docs, "indexed_chunks": stats.get("source_chunk_count", 79071 if source == "BROAD_EXTRA" else (processing["chunks"] if source == "PUBMED" else None)),
            "retrieval_method": retrieval.get(source, "per-source dense + BM25 + RRF, fixed M16"),
            "candidate_pairs": rows, "candidate_pairs_per_query": rows / 1200, "unique_candidate_documents": int(stats["unique_candidate_documents"]),
            "candidates_reaching_reranker": rows, "final_document_slots": final_docs, "final_chunk_slots": final_chunks,
            "candidate_to_final_chunk_discard_fraction": 1 - final_chunks / rows if rows else None,
            "query_language": "vi", "source_language": "en" if source == "PUBMED" else ("mixed vi/zh/unknown" if source in {"BROAD_EXTRA", "ORIGINAL_PILOT"} else language_map.get(source, {}).get("likely_language", "unknown")),
            "chunks_per_indexed_document": (stats.get("source_chunk_count", processing["chunks"] if source == "PUBMED" else (79071 if source == "BROAD_EXTRA" else None)) / indexed_docs) if indexed_docs and (stats.get("source_chunk_count") is not None or source in {"PUBMED", "BROAD_EXTRA"}) else None,
            "text_availability": ({"abstract_documents": 239770, "title_only_documents": 126, "unavailable_official_records": 104} if source == "PUBMED" else "All counted documents are retained usable extractions with source-grounded chunks."),
        }
    declared = migration["official_corpus_diff"]["organizer_declared_language_composition"]
    local_language = {**broad["language_counts"], "en": processing["valid_documents"]}
    return {
        "official_corpus": {"documents": sum(declared.values()), "declared_language_counts": declared},
        "local_usable_union": {"documents": broad["retained_usable_docs"] + processing["valid_documents"], "chunks": broad["retained_chunks"] + processing["chunks"], "identified_language_counts": local_language, "note": "5,702 broad-existing documents have unknown source-associated language; they are not silently assigned to VI or ZH."},
        "language_coverage": {
            "en": {"official": declared["en"], "local_identified": processing["valid_documents"], "missing": declared["en"] - processing["valid_documents"]},
            "vi": {"official": declared["vi"], "local_identified": broad["language_counts"].get("vi", 0), "unclassified_local_may_include_vi": broad["language_counts"].get("unknown", 0)},
            "zh": {"official": declared["zh"], "local_identified": broad["language_counts"].get("zh", 0), "unclassified_local_may_include_zh": broad["language_counts"].get("unknown", 0)},
        },
        "logical_candidate_funnel": funnel,
        "highest_missing_domains": sorted(missing_domains, key=lambda row: (row["missing_local_content"], row["official_documents"]), reverse=True)[:30],
        "highest_missing_vietnamese_domains": [row for row in sorted(missing_domains, key=lambda row: row["missing_local_content"], reverse=True) if row["language_association"] == "vi"][:20],
        "coverage_caveat": "Official language totals are organizer-declared. Local source-associated labels are deterministic but incomplete, so unknown documents remain a separate bucket.",
    }


def write_review(path: Path, selected: dict[str, list[int]], queries: dict[int, str], metadata: dict[int, dict], p11_records: dict, p12_records: dict, p11_chunks: dict, p12_chunks: dict, p11_scores: dict, p12_scores: dict, per_query: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stratum", "query_id", "topic_bucket", "query_vi", "phase11c_top20_docs_json", "phase12_top20_docs_json", "phase11c_top10_chunks_json", "phase12_top10_chunks_json", "displaced_replacement_pairs_json", "pubmed_docs_top20", "pubmed_chunks_top40", "human_judgment", "reviewer_notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for stratum, ids in selected.items():
            for query_id in ids:
                def docs(record, scores):
                    return [{"rank": rank, "doc_id": int(doc), "source": metadata.get(int(doc), {}).get("source", "unknown"), "title": metadata.get(int(doc), {}).get("title", ""), "reranker_score": scores.get(query_id, {}).get(int(doc))} for rank, doc in enumerate(record["relevant_docs"], 1)]
                def chunks(rows):
                    return [{"rank": row["rank"], "chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "source": metadata.get(row["doc_id"], {}).get("source", "unknown"), "reranker_score": row["reranker_score"], "original_text_snippet": row["text"][:800], "snippet_truncated": len(row["text"]) > 800} for row in rows[:10]]
                old_docs = docs(p11_records[query_id], p11_scores); new_docs = docs(p12_records[query_id], p12_scores)
                displaced = [row for row in old_docs if row["doc_id"] in per_query[query_id]["displaced_phase11c_documents"]]
                replacements = [row for row in new_docs if row["doc_id"] in per_query[query_id]["inserted_documents"]]
                pairs = [{"displaced": displaced[i] if i < len(displaced) else None, "replacement": replacements[i] if i < len(replacements) else None} for i in range(max(len(displaced), len(replacements)))]
                writer.writerow({
                    "stratum": stratum, "query_id": query_id, "topic_bucket": topic_bucket(queries[query_id]), "query_vi": queries[query_id],
                    "phase11c_top20_docs_json": json.dumps(old_docs, ensure_ascii=False), "phase12_top20_docs_json": json.dumps(new_docs, ensure_ascii=False),
                    "phase11c_top10_chunks_json": json.dumps(chunks(p11_chunks[query_id]), ensure_ascii=False), "phase12_top10_chunks_json": json.dumps(chunks(p12_chunks[query_id]), ensure_ascii=False),
                    "displaced_replacement_pairs_json": json.dumps(pairs, ensure_ascii=False),
                    "pubmed_docs_top20": per_query[query_id]["pubmed_documents_at"][20], "pubmed_chunks_top40": per_query[query_id]["pubmed_chunks_at"][40],
                    "human_judgment": "", "reviewer_notes": "",
                })


def markdown(summary: dict, coverage: dict, scores: dict) -> tuple[str, str]:
    disp = summary["aggregate"]
    pubmed = scores["by_candidate_arm"]["PUBMED"]["candidate_score_distribution"]
    broad = scores["by_candidate_arm"]["BROAD_EXTRA"]["candidate_score_distribution"]
    diagnosis = f"""# Phase 12 failure diagnosis

## Kết luận ngắn gọn

**Nút thắt mạnh nhất quan sát được là cạnh tranh/xếp hạng ứng viên (candidate competition and source fusion), không phải thiếu dữ liệu đơn thuần.** Phase 12 thêm 239.896 tài liệu PubMed nhưng FINAL giảm từ `0.0492` xuống `0.0451`. PubMed thay thế {disp['displaced_document_slots']:,} vị trí tài liệu và {disp['displaced_chunk_slots']:,} vị trí đoạn văn; thay đổi xảy ra ở {disp['queries_docs_changed']:,}/1.200 truy vấn tài liệu và {disp['queries_chunks_changed']:,}/1.200 truy vấn đoạn văn.

“Candidate competition” nghĩa là nhiều kết quả cùng tranh một số vị trí đầu ra hữu hạn. PubMed đưa {coverage['logical_candidate_funnel']['PUBMED']['candidate_pairs']:,} cặp query–chunk vào reranker, trong khi mỗi nguồn tập trung chỉ có 19.200. Hệ thống xếp toàn bộ chunk theo một điểm chung rồi lấy lần xuất hiện đầu tiên của mỗi tài liệu. Vì vậy nguồn có rất nhiều chunk có nhiều cơ hội tạo một điểm cực đại và đẩy tài liệu Việt Nam đã được kiểm chứng ra ngoài.

## Bằng chứng về điểm số

- PubMed median reranker score: `{pubmed['median']:.4f}`; p95: `{pubmed['p95']:.4f}`.
- BROAD_EXTRA median: `{broad['median']:.4f}`; p95: `{broad['p95']:.4f}`.
- Đây **không** chứng minh thang điểm bị hỏng. Điểm PubMed không cao hơn mọi nguồn mạnh; vấn đề quan sát được là khối lượng ứng viên lớn cộng với phép lấy max-chunk ở cấp tài liệu.
- Kết quả leaderboard cho biết các thay thế này có hại về tổng thể, nhưng không cho biết từng PubMed result nào sai. File `human_review.csv` được tạo để con người gắn nhãn mà không giả mạo gold của organizer.

## Phạm vi bao phủ

Corpus chính thức có {coverage['official_corpus']['documents']:,} ID. Local union hiện có {coverage['local_usable_union']['documents']:,} tài liệu dùng được: 35.690 broad-existing và 239.896 PubMed. English PubMed gần đầy đủ (thiếu 104 record không khả dụng), nhưng độ phủ Việt Nam/Trung Quốc vẫn rất nhỏ và 5.702 tài liệu local chưa có nhãn ngôn ngữ chắc chắn.

## Bất định lớn nhất

Không có hidden gold theo truy vấn. Ta chưa biết thất bại chủ yếu do (1) dense VI→EN lấy sai bài, (2) reranker VI→EN đánh giá sai, hay (3) cơ chế max-chunk/fusion ưu tiên PubMed quá mức. Review 60 truy vấn và các replay offline có kiểm soát là mốc tiếp theo để tách ba nguyên nhân này.

## Tệp bằng chứng

- `ranking_displacement.json`: thay thế theo từng query và 60 trường hợp nổi bật.
- `source_score_analysis.json`: phân phối điểm, margin và thành phần theo rank.
- `coverage_funnel.json`: số lượng qua từng tầng corpus → candidate → reranker → output.
- `human_review.csv`: phiếu đánh giá RELEVANT / PARTIALLY_RELEVANT / IRRELEVANT / UNCLEAR.
"""
    plan = """# Phase 13 prioritized plan

## Mốc triển khai kế tiếp

**Phase 13A: xây dựng replay fusion có kiểm soát trên score hiện có, trước khi chạy model mới.** Giữ nguyên candidate và reranker score; thay cách chuyển chunk-score thành document competition bằng quota/cap được khai báo, document-level aggregation ổn định, và consistency giữa docs/chunks. Đánh giá bằng 60 nhãn người thật, tỷ lệ giữ lại control, source displacement, và replay determinism. Không dùng leaderboard để debug.

## A. Candidate recall: hybrid có kiểm soát

1. Tạo union lexical BM25 + multilingual dense ở cấp **document**, rồi giới hạn số chunk đại diện/tài liệu trước rerank. Failure addressed: 349k PubMed chunks làm candidate volume mất cân bằng. Input: embedding/BM25/index hiện có. Cost: CPU replay thấp; model inference bằng 0 cho phase đầu. Diagnostic: candidate diversity, số tài liệu/query, overlap dense–lexical. Acceptance: không nguồn nào có lợi chỉ vì nhiều chunk; control replay giữ nguyên khi tắt arm. Fallback: cap theo document với top-N chunk bằng dense/lexical score.
2. Thêm medical synonym expansion dạng từ điển cục bộ, versioned (VI–EN thuật ngữ/viết tắt), không dùng API đóng. Failure addressed: lexical cross-language gap. Cost: thấp cho query expansion/BM25; dense không đổi. Diagnostic: candidate changes trên review set. Acceptance: tăng relevant/partial candidates theo human labels mà không tăng irrelevant displacement. Fallback: chỉ dùng synonym cho lexical arm, không sửa query gốc.

## B. Ranking và source fusion

1. So sánh offline các aggregation đã khai báo: max chunk hiện tại, top-k mean, và document score + best evidence score. Failure addressed: extreme-value bias của max trên nhiều chunk. Input: 603.587 score hiện có. Cost: phút CPU. Diagnostic: displacement, margin, review precision@20. Acceptance: cải thiện human-judged precision và giữ/khôi phục phần lớn Phase 11C relevant results. Fallback: gated additive PubMed arm chỉ thay thế khi margin vượt ngưỡng đo từ review set.
2. Không dùng source boost tùy tiện. Nếu cần calibration, fit/choose trên review labels và báo riêng theo language/source, với holdout. Acceptance: calibration ổn định trên bootstrap/holdout; fallback: source-neutral document aggregation.

## C. Chunk selection và consistency

1. Chọn docs trước, sau đó chọn evidence chunks thuộc docs đã chọn; cho phép một quota nhỏ chunk-only nếu contract cần. Failure addressed: nhiều chunk owner không nằm trong top20 docs và output không nhất quán. Cost: CPU only. Diagnostic: owner-consistency, unique docs represented, redundancy. Acceptance: 100% ownership/provenance, giảm trùng nội dung, không giảm human evidence relevance. Fallback: tối đa 2 chunk/tài liệu rồi fill theo score.
2. Deduplicate near-identical chunks trong cùng article trước competition. Input: text/hash hiện có. Acceptance: không mất exact provenance; fallback: exact-normalized duplicates only.

## D. Mở rộng nội dung thật

1. Ưu tiên domain Việt Nam có official gap lớn trong `coverage_funnel.json`; crawl lịch sự theo robots/rate limit, resumable và disk gate 20 GiB. Failure addressed: local VI coverage remains a small fraction of 714.469 official IDs. Cost: network + extraction; embedding chỉ sau khi retrieval design passes. Diagnostic: success/extraction rate and lexical candidate yield on review queries. Acceptance: source quality gate plus measurable new candidate coverage. Fallback: smaller per-domain exploration allocation.
2. Chinese remains a later multilingual track; a negative Vietnamese-query lexical result must not be interpreted as no Chinese value.

## Stage gates

1. Annotate 60-query review set with two passes or adjudication.
2. Implement CPU-only document aggregation/fusion replay using frozen scores.
3. Accept only variants that improve human-judged precision/coverage and pass frozen control/provenance/determinism.
4. Only then authorize missing-pair inference or new acquisition.
5. Use at most one leaderboard submission for the selected Phase 13 architecture; keep remaining quota reserved.
"""
    return diagnosis, plan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = load_queries()
    pubmed_doc_ids = set(map(int, pq.read_table(ROOT / "data/pubmed_phase12/processed/documents.parquet", columns=["doc_id"]).column(0).to_pylist()))
    aggregate, per_query = compare_outputs(pubmed_doc_ids)
    rows=list(per_query.values())
    largest_order=sorted(rows,key=lambda r:(len(r["inserted_pubmed_documents"]),r["inserted_pubmed_chunk_count"],-r["query_id"]),reverse=True)
    review_largest=diverse_selection(largest_order[:240],queries,20)
    least_order=sorted(rows,key=lambda r:(len(r["inserted_pubmed_documents"]),r["inserted_pubmed_chunk_count"],r["query_id"]))
    review_least=diverse_selection(least_order[:240],queries,20,set(review_largest))
    disagreement_order=sorted(rows,key=lambda r:(r["document_chunk_disagreement"],len(r["displaced_phase11c_documents"]),r["query_id"]),reverse=True)
    review_disagreement=diverse_selection(disagreement_order[:400],queries,20,set(review_largest)|set(review_least))
    review_selection={"large_pubmed_displacement":review_largest,"low_pubmed_displacement":review_least,"strong_document_chunk_disagreement":review_disagreement}
    aggregate["human_review_query_ids"]=review_selection
    chunk_source, document_arm, candidate_stats = candidate_sources(P12_ROOT / "round1/source_candidate_parts")
    pilot_path = ROOT / "data/source_census/phase11b_new_corpus/round1/pilot_rerank_pool.new_official.parquet"
    for doc_id in pq.read_table(pilot_path, columns=["doc_id"]).column(0).to_pylist():
        document_arm.setdefault(int(doc_id), "ORIGINAL_PILOT")
    by_source, score_lookup = score_analysis(chunk_source)
    metadata = document_metadata([P12_ROOT / "round1/combined_documents.parquet"])
    p12_chunks, p12_doc_scores = ranking_top_details(P12_ROOT, "FULL", chunk_source, score_lookup)
    p11_chunks, p11_doc_scores = ranking_top_details(P11_ROOT, "BROAD", chunk_source, score_lookup)
    enrich_highlights(aggregate, per_query, queries, metadata, p11_chunks, p12_chunks, p11_doc_scores, p12_doc_scores)
    p11_selected = set(sum(aggregate["highlight_query_ids"].values(), [])) | set(sum(review_selection.values(), []))
    p11_records = {int(row["id"]): row for row in _iter_json_array(P11_JSON) if int(row["id"]) in p11_selected}
    p12_records_all = {int(row["id"]): row for row in _iter_json_array(P12_JSON)}
    p12_records = {qid: p12_records_all[qid] for qid in p11_selected}
    composition = final_source_composition(p12_records_all, metadata, document_arm)
    coverage = coverage_and_funnel(candidate_stats, composition)
    score_report = {
        "observed_only_no_gold_claims": True, "by_candidate_arm": by_source,
        "final_source_composition": composition,
        "ranking_semantics": "All chunks are globally sorted by reranker score; a document rank is its first/highest-scoring chunk occurrence.",
        "observed_interpretation": "PubMed has a lower median than strong Vietnamese sources but 349,187 candidate pairs. Its much larger candidate multiplicity plus max-chunk document aggregation creates an extreme-value opportunity advantage. This is evidence of volume-sensitive fusion, not proof that score calibration itself is invalid.",
    }
    phase11c_metrics={"final":0.0492,"docs_f2":0.0491,"chunks_f2":0.0492,"docs_precision":0.1352,"docs_recall":0.0457,"chunks_precision":0.1236,"chunks_recall":0.0467}
    phase12_metrics={"final":0.0451,"docs_f2":0.0446,"chunks_f2":0.0457,"docs_precision":0.1235,"docs_recall":0.0415,"chunks_precision":0.1122,"chunks_recall":0.0437}
    displacement = {"organizer_results":{"phase11c":phase11c_metrics,"phase12":phase12_metrics,"phase12_minus_phase11c":{key:phase12_metrics[key]-phase11c_metrics[key] for key in phase11c_metrics}}, "aggregate": aggregate, "per_query": {str(key): value for key, value in sorted(per_query.items())}}
    atomic_json(OUT / "ranking_displacement.json", displacement)
    atomic_json(OUT / "source_score_analysis.json", score_report)
    atomic_json(OUT / "coverage_funnel.json", coverage)
    write_review(OUT / "human_review.csv", review_selection, queries, metadata, p11_records, p12_records, p11_chunks, p12_chunks, p11_doc_scores, p12_doc_scores, per_query)
    diagnosis, plan = markdown(displacement, coverage, score_report)
    (OUT / "diagnosis_summary.md").write_text(diagnosis, encoding="utf-8")
    (OUT / "phase13_prioritized_plan.md").write_text(plan, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "queries": 1200, "review_rows": 60, "no_model_inference": True, "no_crawling": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
