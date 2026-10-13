#!/usr/bin/env python3
"""M1.5 CPU-only benchmark discovery, document registry and candidate ledger."""
from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.silver import normalize_query
from src.evaluation.system_registry import (
    aggregate_candidate_rows, atomic_parquet_writer, canonical_domain,
    commit_parquet, content_digest, query_groups,
)
from src.submission.common import atomic_write_json, file_sha256
from src.runtime.state_file import update_json

OUT = ROOT / "artifacts/system_upgrade_m15"
DATA = ROOT / "data/system_upgrade_m15"
OFFICIAL = ROOT / "data/raw/links_corpus_vi_en_zh.parquet"
QUERIES = ROOT / "data/raw/query.parquet"
BROAD = ROOT / "data/source_census/phase11a_bottleneck/broad/documents.parquet"
PUBMED = ROOT / "data/pubmed_phase12/processed/documents.parquet"
BROAD_EXTRA = ROOT / "data/source_census/phase11c_broad_existing_new/broad_extra_documents.parquet"
P11 = ROOT / "data/source_census/phase11c_broad_existing_new"
P12 = ROOT / "data/pubmed_phase12/retrieval"
PILOT = ROOT / "data/source_census/phase11b_new_corpus/round1/pilot_rerank_pool.new_official.parquet"
P12_CANONICAL_DOCUMENTS = P12 / "round1/combined_documents.parquet"


def language(value: str, *, pubmed: bool = False) -> str:
    if pubmed:
        return "en"
    value = str(value or "").casefold()
    return "vi" if value.startswith("vi") else "zh" if value.startswith("zh") else "unknown"


def official_subset(local_ids: set[int]) -> dict[int, str]:
    result = {}
    for batch in pq.ParquetFile(OFFICIAL).iter_batches(columns=["id", "url"], batch_size=131072):
        for doc_id, url in zip(*(col.to_pylist() for col in batch.columns), strict=True):
            if int(doc_id) in local_ids:
                result[int(doc_id)] = str(url)
    return result


def official_id_set() -> set[int]:
    values: set[int] = set()
    for batch in pq.ParquetFile(OFFICIAL).iter_batches(columns=["id"], batch_size=131072):
        values.update(map(int, batch.column(0).to_pylist()))
    return values


def benchmark_discovery(official_ids: set[int]) -> dict:
    """Count evidence classes without promoting title/abstract queries to gold."""
    official_queries = {normalize_query(str(row["query"])) for row in pq.read_table(QUERIES, columns=["query"]).to_pylist()}
    title_candidates = Counter(); title_answer_marker = Counter(); exact_official_overlap = Counter()
    structured = []
    for batch in pq.ParquetFile(BROAD).iter_batches(
        columns=["doc_id", "original_url", "title", "language_signal", "normalized_text", "sections"], batch_size=512,
    ):
        for row in batch.to_pylist():
            if int(row["doc_id"]) not in official_ids:
                continue
            lang = language(row["language_signal"]); title = str(row.get("title") or "").strip()
            text = str(row.get("normalized_text") or "")
            sections = row.get("sections") or []
            if ("?" in title or "？" in title) and len(text) >= 80:
                title_candidates[lang] += 1
                if any(item.get("section_type") == "answer" for item in sections):
                    title_answer_marker[lang] += 1
                if normalize_query(title) in official_queries:
                    exact_official_overlap[lang] += 1
            for index, section in enumerate(sections):
                if section.get("section_type") != "question":
                    continue
                question = " ".join(str(section.get("text") or "").split())
                answer_parts = []
                for following in sections[index + 1:]:
                    if following.get("section_type") == "question":
                        break
                    candidate = " ".join(str(following.get("text") or "").split())
                    if candidate:
                        answer_parts.append(candidate)
                answer = " ".join(answer_parts)
                if ("?" in question or "？" in question) and len(answer) >= 80:
                    structured.append({
                        "doc_id": int(row["doc_id"]), "source_url": str(row["original_url"]),
                        "language": lang, "question": question, "answer_excerpt": answer[:600],
                    })
    pubmed_title_questions = 0
    for batch in pq.ParquetFile(PUBMED).iter_batches(columns=["title", "abstract_text"], batch_size=4096):
        for title, abstract in zip(*(col.to_pylist() for col in batch.columns), strict=True):
            if ("?" in str(title) or "？" in str(title)) and len(str(abstract or "")) >= 80:
                pubmed_title_questions += 1
    # The four NEW-official structural hits were inspected and are non-medical
    # interview questions from one infrastructure article.
    primary_eligible = []
    report = {
        "status": "INSUFFICIENT_FOR_ARCHITECTURE_SELECTION",
        "m1_pair_count": 21,
        "m1_failure_mode": "M1 scanned only BROAD title questions and required an answer section whose own text was >=80 characters; many Q&A pages store a short answer heading followed by content sections.",
        "classes": {
            "A_genuine_source_linked": {
                "structural_candidates": len(structured), "eligible_biomedical_pairs": len(primary_eligible),
                "inspection_result": "No useful non-title biomedical pair: all four questions concern a highway inspection.",
                "examples": structured,
            },
            "B_near_verbatim_title_derived": {
                "broad_question_title_with_usable_text": dict(sorted(title_candidates.items())),
                "broad_question_title_with_answer_marker": dict(sorted(title_answer_marker.items())),
                "pubmed_english_question_title_with_abstract": pubmed_title_questions,
                "eligible_for_primary_heldout": False,
                "official_query_exact_normalized_overlap": dict(sorted(exact_official_overlap.items())),
            },
            "C_derived_or_synthetic": {"pairs": 0, "eligible_for_primary_heldout": False},
            "D_external_verified_ids": {"pairs": 0, "eligible_for_primary_heldout": True, "finding": "No local public benchmark manifest maps verified questions to NEW-official doc IDs."},
        },
        "primary_track_counts": {"vi_to_vi": 0, "vi_to_en": 0, "en_to_en": 0, "zh_to_zh": 0},
        "vi_to_en_status": "INSUFFICIENT",
        "leakage_policy": "Title-derived candidates remain audit-only; primary A/D split would be clustered by normalized near-duplicate query and source document before assignment.",
        "pubmed_policy": "PMID mappings prove document identity only. Abstracts/titles were not converted into relevance questions.",
    }
    atomic_write_json(OUT / "benchmark_discovery_report.json", report)
    return report


REGISTRY_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("official_url", pa.string()), ("canonical_domain", pa.string()),
    ("language_signal", pa.string()), ("content_available", pa.bool_()),
    ("document_text_reference", pa.string()), ("dense_index_available", pa.bool_()),
    ("sparse_index_available", pa.bool_()), ("retrieval_arms", pa.list_(pa.string())),
    ("content_hash", pa.string()), ("provenance_status", pa.string()),
])


def build_registry(official_ids: set[int]) -> tuple[Path, dict]:
    broad_ids = set()
    for batch in pq.ParquetFile(BROAD).iter_batches(columns=["doc_id"], batch_size=65536):
        broad_ids.update(map(int, batch.column(0).to_pylist()))
    pubmed_ids = set()
    for batch in pq.ParquetFile(PUBMED).iter_batches(columns=["doc_id"], batch_size=65536):
        pubmed_ids.update(map(int, batch.column(0).to_pylist()))
    if broad_ids & pubmed_ids:
        raise ValueError(f"broad/PubMed doc ID overlap: {len(broad_ids & pubmed_ids)}")
    removed_broad_ids = broad_ids - official_ids
    broad_ids &= official_ids
    phase12_ids = set(map(int, pq.read_table(P12_CANONICAL_DOCUMENTS, columns=["doc_id"]).column(0).to_pylist()))
    pilot_extra_ids = phase12_ids - broad_ids - pubmed_ids
    if not pilot_extra_ids <= official_ids:
        raise ValueError(f"pilot canonical IDs absent from official corpus: {len(pilot_extra_ids - official_ids)}")
    local_ids = broad_ids | pubmed_ids | pilot_extra_ids
    official = official_subset(local_ids)
    if len(official) != len(local_ids):
        raise ValueError(f"local IDs absent from official corpus: {len(local_ids - set(official))}")
    broad_extra = set(map(int, pq.read_table(BROAD_EXTRA, columns=["doc_id"]).column(0).to_pylist()))
    destination = DATA / "canonical_document_registry.parquet"
    writer, temporary = atomic_parquet_writer(destination, REGISTRY_SCHEMA)
    content_hashes = Counter(); url_mismatches = 0; language_counts = Counter(); rows_written = 0
    references = {str(BROAD.relative_to(ROOT)), str(PUBMED.relative_to(ROOT)), str(P12_CANONICAL_DOCUMENTS.relative_to(ROOT))}
    try:
        for label, path, is_pubmed in (("broad", BROAD, False), ("pubmed", PUBMED, True), ("pilot_extra", P12_CANONICAL_DOCUMENTS, False)):
            for batch in pq.ParquetFile(path).iter_batches(
                columns=["doc_id", "original_url", "language_signal", "normalized_text"], batch_size=1024,
            ):
                output = []
                for row in batch.to_pylist():
                    doc_id = int(row["doc_id"])
                    if doc_id not in local_ids:
                        continue
                    if label == "pilot_extra" and doc_id not in pilot_extra_ids:
                        continue
                    url = official[doc_id]
                    mismatch = str(row["original_url"]) != url; url_mismatches += mismatch
                    lang = language(row["language_signal"], pubmed=is_pubmed); language_counts[lang] += 1
                    digest = content_digest(str(row.get("normalized_text") or "")); content_hashes[digest] += 1
                    if is_pubmed:
                        dense, sparse, arms, ref = True, False, ["PUBMED"], str(PUBMED.relative_to(ROOT))
                    elif label == "pilot_extra":
                        dense, sparse, arms, ref = False, False, ["ORIGINAL_PILOT"], str(P12_CANONICAL_DOCUMENTS.relative_to(ROOT))
                    elif doc_id in broad_extra:
                        dense, sparse, arms, ref = False, True, ["BROAD_EXTRA"], str(BROAD.relative_to(ROOT))
                    else:
                        arm = canonical_domain(url)
                        dense, sparse, arms, ref = True, True, [arm], str(BROAD.relative_to(ROOT))
                    output.append({
                        "doc_id": doc_id, "official_url": url, "canonical_domain": canonical_domain(url),
                        "language_signal": lang, "content_available": True,
                        "document_text_reference": f"{ref}#doc_id={doc_id}",
                        "dense_index_available": dense, "sparse_index_available": sparse,
                        "retrieval_arms": arms, "content_hash": digest,
                        "provenance_status": "VERIFIED_OFFICIAL_ID_AND_URL" if not mismatch else "OFFICIAL_ID_URL_MISMATCH_REVIEW",
                    })
                writer.write_table(pa.Table.from_pylist(output, schema=REGISTRY_SCHEMA)); rows_written += len(output)
        commit_parquet(writer, temporary, destination)
    except Exception:
        writer.close(); temporary.unlink(missing_ok=True); raise
    audit = {
        "passed": rows_written == 275877 and url_mismatches == 0,
        "registry_path": str(destination.relative_to(ROOT)), "rows": rows_written,
        "unique_doc_ids": len(local_ids), "official_membership": len(official),
        "language_counts": dict(sorted(language_counts.items())),
        "dense_available": len(pubmed_ids) + len(broad_ids - broad_extra),
        "sparse_available": len(broad_ids), "broad_extra_sparse_only": len(broad_extra),
        "m1_union_count": 275586,
        "official_pilot_documents_omitted_from_m1_union": len(pilot_extra_ids),
        "corrected_retrieval_participating_union": len(local_ids),
        "document_reference_files": sorted(references),
        "missing_reference_files": sum(not (ROOT / value).exists() for value in references),
        "official_url_mismatches": url_mismatches,
        "old_corpus_broad_rows_excluded": len(removed_broad_ids),
        "duplicate_content_groups": sum(count > 1 for count in content_hashes.values()),
        "duplicate_content_instances": sum(count for count in content_hashes.values() if count > 1),
        "size_bytes": destination.stat().st_size, "sha256": file_sha256(destination),
    }
    atomic_write_json(OUT / "registry_audit.json", audit)
    if not audit["passed"]:
        raise ValueError(f"registry audit failed: {audit}")
    return destination, audit


LEDGER_SCHEMA = pa.schema([
    ("query_id", pa.int64()), ("doc_id", pa.int64()), ("canonical_domain", pa.string()),
    ("candidate_arms", pa.list_(pa.string())), ("original_rank_by_arm_json", pa.string()),
    ("candidate_chunk_count", pa.int64()), ("candidate_row_count", pa.int64()),
    ("duplicate_chunk_competition", pa.int64()), ("best_reranker_score", pa.float64()),
    ("mean_top2_score", pa.float64()), ("mean_top3_score", pa.float64()),
    ("reached_phase11c_docs20", pa.bool_()), ("reached_phase12_docs20", pa.bool_()),
])


def _top20(path: Path) -> dict[int, set[int]]:
    result = defaultdict(set)
    for batch in pq.ParquetFile(path).iter_batches(columns=["query_id", "rank", "doc_id"], batch_size=32768):
        for query_id, rank, doc_id in zip(*(col.to_pylist() for col in batch.columns), strict=True):
            if int(rank) <= 20:
                result[int(query_id)].add(int(doc_id))
    return result


def build_ledger(registry: Path, state_path: Path | None = None) -> tuple[Path, dict]:
    canonical_domains = {
        int(doc_id): str(domain) for doc_id, domain in zip(
            *(col.to_pylist() for col in pq.read_table(registry, columns=["doc_id", "canonical_domain"]).columns), strict=True,
        )
    }
    paths = []
    for marker in sorted((P12 / "round1/source_candidate_parts").glob("*.json")):
        paths.append((str(json.loads(marker.read_text(encoding="utf-8"))["source"]), marker.with_suffix(".parquet")))
    paths.append(("ORIGINAL_PILOT", PILOT))
    generators = [(arm, iter(query_groups(path))) for arm, path in paths]
    finals11 = _top20(P11 / "rankings/BROAD/reranked_documents.parquet")
    finals12 = _top20(P12 / "rankings/FULL/reranked_documents.parquet")
    score_db = sqlite3.connect(f"file:{(P12 / 'round1/source_scores.sqlite').resolve()}?mode=ro", uri=True)
    pilot_db = sqlite3.connect(f"file:{(ROOT / 'data/retrieval/phase7_reranking/rerank_scores.sqlite').resolve()}?mode=ro", uri=True)
    query_ids = list(map(int, pq.read_table(QUERIES, columns=["id"]).column(0).to_pylist()))
    destination = DATA / "document_candidate_ledger.parquet"
    writer, temporary = atomic_parquet_writer(destination, LEDGER_SCHEMA)
    total_rows = 0; multi_arm = 0; duplicate_competition = 0; lost = 0
    arm_pairs = Counter(); arm_rows = Counter(); source_top20 = Counter(); per_query_counts = []
    phase11_only = phase12_only = shared_final = 0
    try:
        for query_number, query_id in enumerate(query_ids, 1):
            groups = []
            for arm, generator in generators:
                rows = next(generator)
                if not rows or int(rows[0]["query_id"]) != query_id:
                    raise ValueError(f"candidate arm query mismatch: arm={arm} expected={query_id}")
                groups.append((arm, rows)); arm_rows[arm] += len(rows)
            scores = {str(chunk): float(score) for chunk, score in score_db.execute(
                "SELECT chunk_id,rerank_score FROM pairs WHERE query_id=?", (query_id,),
            )}
            for chunk, score in pilot_db.execute(
                "SELECT chunk_id,rerank_score FROM scores WHERE query_id=?", (query_id,),
            ):
                previous = scores.setdefault(str(chunk), float(score))
                if previous != float(score):
                    raise ValueError(f"score-cache conflict query={query_id} chunk={chunk}")
            rows = aggregate_candidate_rows(query_id, groups, scores, finals11[query_id], finals12[query_id], canonical_domains)
            writer.write_table(pa.Table.from_pylist(rows, schema=LEDGER_SCHEMA))
            total_rows += len(rows); per_query_counts.append(len(rows))
            multi_arm += sum(len(row["candidate_arms"]) > 1 for row in rows)
            duplicate_competition += sum(row["duplicate_chunk_competition"] for row in rows)
            lost += sum(not row["reached_phase12_docs20"] for row in rows)
            for row in rows:
                arm_pairs.update(row["candidate_arms"])
                if row["reached_phase12_docs20"]:
                    source_top20[row["canonical_domain"]] += 1
            shared_final += len(finals11[query_id] & finals12[query_id])
            phase11_only += len(finals11[query_id] - finals12[query_id])
            phase12_only += len(finals12[query_id] - finals11[query_id])
            if state_path is not None and (query_number == 1 or query_number % 25 == 0):
                update_json(state_path, {"status": "RUNNING", "stage": "CANDIDATE_LEDGER", "done": query_number,
                                        "total": len(query_ids), "checkpoint": f"query:{query_id}"})
        commit_parquet(writer, temporary, destination)
    except Exception:
        writer.close(); temporary.unlink(missing_ok=True); raise
    audit = {
        "passed": total_rows > 0 and len(per_query_counts) == 1200,
        "ledger_path": str(destination.relative_to(ROOT)), "rows": total_rows, "queries": len(per_query_counts),
        "unique_candidate_docs_per_query": {"min": min(per_query_counts), "median": sorted(per_query_counts)[600], "max": max(per_query_counts)},
        "candidate_rows_by_arm": dict(sorted(arm_rows.items())),
        "unique_query_document_pairs_by_arm": dict(sorted(arm_pairs.items())),
        "multi_arm_query_document_pairs": multi_arm,
        "duplicate_chunk_competition_rows": duplicate_competition,
        "phase12_candidates_not_in_docs20": lost,
        "phase12_candidate_to_docs20_discard_fraction": lost / total_rows,
        "phase11c_vs_phase12_docs20": {"shared": shared_final, "phase11c_only_displaced": phase11_only, "phase12_only_inserted": phase12_only},
        "phase12_top20_source_slots": dict(source_top20.most_common()),
        "structural_largest_loss_stage": "candidate/reranker document pool -> final Docs@20",
        "relevance_warning": "Discard counts are structural, not gold relevance loss.",
        "size_bytes": destination.stat().st_size, "sha256": file_sha256(destination),
    }
    score_db.close(); pilot_db.close()
    atomic_write_json(OUT / "candidate_ledger_audit.json", audit)
    return destination, audit


def write_markdown(benchmark: dict, registry: dict, ledger: dict, runtime: dict) -> None:
    (OUT / "benchmark_quality_report.md").write_text(f"""# Chất lượng benchmark discovery

- M1 chỉ thấy 21 pair vì lọc title có dấu hỏi, `section_type=answer`, và yêu cầu chính answer block đó dài ≥80 ký tự.
- Scan rộng tìm {benchmark['classes']['A_genuine_source_linked']['structural_candidates']} structural non-title Q&A candidates nhưng **0** pair y-sinh đủ tin cậy: bốn câu thuộc một bài phỏng vấn về cao tốc, một câu là tên/chức danh bị extractor gắn nhầm.
- Title-derived audit pool lớn nhưng không được nâng thành gold: {sum(benchmark['classes']['B_near_verbatim_title_derived']['broad_question_title_with_usable_text'].values()):,} broad titles và {benchmark['classes']['B_near_verbatim_title_derived']['pubmed_english_question_title_with_abstract']:,} PubMed titles.
- VI→EN: **INSUFFICIENT**. PMID chỉ xác nhận document identity, không cung cấp câu hỏi tiếng Việt.

Kết luận: benchmark chính vẫn không đủ để chọn kiến trúc. Không tạo relevance labels giả và không thay thế silver benchmark M1.
""", encoding="utf-8")
    (OUT / "bottleneck_decision.md").write_text(f"""# Quyết định bottleneck M1.5

## Hạ tầng đã xác nhận

- Registry: {registry['rows']:,} canonical documents, URL mismatch={registry['official_url_mismatches']}, missing paths={registry['missing_reference_files']}.
- Candidate ledger: {ledger['rows']:,} query-document pairs cho 1.200 queries.
- Phase 12 loại khỏi Docs@20 {ledger['phase12_candidates_not_in_docs20']:,} candidate pairs ({ledger['phase12_candidate_to_docs20_discard_fraction']:.2%}). Đây là mất mát cấu trúc, relevance vẫn UNKNOWN.
- Multi-arm query/document pairs: {ledger['multi_arm_query_document_pairs']:,}; duplicate arm/chunk rows: {ledger['duplicate_chunk_competition_rows']:,}.

## Quyết định

Offline architecture comparison vẫn **NO-GO** vì primary A/D benchmark có 0 pair. Thử nghiệm kiến trúc có giá trị cao nhất tiếp theo, sau khi có judgment set, là **document-first unified candidate fusion với bounded representative chunks/document**, dùng ledger hiện tại và exact scores; không cần model inference cho replay đầu tiên.

Ước lượng replay CPU: 5–15 phút, RAM <1 GiB, disk <250 MiB; phải chạy supervisor nếu được phê duyệt. Không được diễn giải candidate discard như official recall.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", type=Path)
    args = parser.parse_args()
    started = time.time(); process = psutil.Process(); OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    if args.state_path:
        update_json(args.state_path, {"status": "RUNNING", "stage": "BENCHMARK_DISCOVERY", "done": 0, "total": 3})
    official_ids = official_id_set()
    benchmark_path = OUT / "benchmark_discovery_report.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    else:
        benchmark = benchmark_discovery(official_ids)
    if args.state_path:
        update_json(args.state_path, {"stage": "REGISTRY", "done": 1, "total": 3, "checkpoint": str(benchmark_path)})
    registry_path = DATA / "canonical_document_registry.parquet"; registry_audit_path = OUT / "registry_audit.json"
    if registry_path.exists() and registry_audit_path.exists():
        registry = json.loads(registry_audit_path.read_text(encoding="utf-8"))
        if not registry.get("passed") or registry.get("sha256") != file_sha256(registry_path):
            raise ValueError("existing registry checkpoint failed integrity verification")
    else:
        registry_path, registry = build_registry(official_ids)
    if args.state_path:
        update_json(args.state_path, {"stage": "CANDIDATE_LEDGER", "done": 0, "total": 1200, "checkpoint": str(registry_path)})
    ledger_path, ledger = build_ledger(registry_path, args.state_path)
    total_size = registry["size_bytes"] + ledger["size_bytes"]
    if total_size > 1024 ** 3:
        raise ValueError(f"M1.5 outputs exceed 1 GiB safety gate: {total_size}")
    runtime = {"seconds": time.time() - started, "peak_rss_bytes": process.memory_info().peak_wset if hasattr(process.memory_info(), "peak_wset") else process.memory_info().rss,
               "additional_disk_bytes": total_size, "cpu_only": True, "no_model_inference": True, "no_crawling": True, "no_index_rebuild": True}
    write_markdown(benchmark, registry, ledger, runtime)
    if args.state_path:
        update_json(args.state_path, {"status": "READY_FOR_HUMAN_REVIEW", "stage": "MILESTONE_1_5_COMPLETE",
                                     "done": 1200, "total": 1200, "checkpoint": str(ledger_path), "integrity": "passed"})
    print(json.dumps({"benchmark": benchmark["status"], "registry": registry["rows"], "ledger": ledger["rows"], "runtime": runtime}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
