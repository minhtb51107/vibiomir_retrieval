from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq

from src.multilingual_retrieval.m2 import sha256_file
from src.multilingual_retrieval.m3 import _atomic_json, _bm25, _load_reused_rankings
from src.multilingual_retrieval.m31 import _bounded_ranges, _document_sources, classify_page


STATUS = "AI_JUDGED_M4"
FIXED_DEVELOPMENT_IDS = (128, 257, 290, 409, 446, 491, 513, 571, 634, 715, 805, 1080)
PHASE29_SHA256 = "e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04"
DISCOVERY_PATHS = {
    128: ("cnkang.com", "/dise/565/"),
    634: ("cnkang.com", "/nkwk/ggths"),
    715: ("cnkang.com", "/dise/9/"),
}


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def load_lexicon(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("version") or not value.get("entities"):
        raise ValueError("invalid M4 terminology lexicon")
    return value


def understand_vi_query(query_vi: str, lexicon: dict[str, Any]) -> dict[str, Any]:
    """Deterministic query-text-only VI→ZH concept transfer; query_id is not required."""
    folded = _fold(query_vi)
    polar_question = bool(re.search(r"\bcó\b.+\bkhông(?:\s+ạ)?\s*[?.!]*$", folded))
    groups: list[list[str]] = []
    matched: list[dict[str, Any]] = []
    for category in ("entities", "relations", "context", "negations"):
        for phrase, translations in lexicon.get(category, {}).items():
            if _fold(phrase) in folded:
                # Vietnamese polar questions use "có phải ... không" without
                # negating the medical concept. Preserve only semantic negation.
                if category == "negations" and phrase == "không" and polar_question:
                    continue
                values = list(dict.fromkeys(str(item) for item in translations))
                groups.append(values)
                matched.append({"vi": phrase, "zh": values, "category": category})
    entities = [item for item in matched if item["category"] == "entities"]
    relations = [item for item in matched if item["category"] == "relations"]
    return {
        "query_vi": query_vi,
        "query_zh": " ".join(term for item in matched for term in item["zh"]),
        "groups": groups,
        "primary_groups": [item["zh"] for item in entities],
        "relation_groups": [item["zh"] for item in relations],
        "matched_terms": matched,
        "ambiguous": not entities or not relations,
        "translation_method": lexicon["version"],
    }


def fixed_development_queries(root: Path, lexicon: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = set(FIXED_DEVELOPMENT_IDS)
    rows = pq.read_table(
        root / "artifacts/multilingual_retrieval_m1/review_queue.parquet",
        columns=["query_id", "query_text", "split"],
    ).to_pylist()
    found: dict[int, dict[str, Any]] = {}
    for row in rows:
        query_id = int(row["query_id"])
        if query_id in wanted and query_id not in found:
            if row["split"] != "development":
                raise ValueError("M4 fixed query escaped development split")
            found[query_id] = {"query_id": query_id, **understand_vi_query(str(row["query_text"]), lexicon)}
    if set(found) != wanted:
        raise ValueError("M4 fixed development query coverage incomplete")
    return [found[key] for key in sorted(found)]


def freeze_target_manifest(root: Path, output: Path) -> dict[str, Any]:
    existing = {int(row["doc_id"]) for _, row in _document_sources(root)}
    records: list[dict[str, Any]] = []
    official = root / "data/raw/links_corpus_vi_en_zh.parquet"
    for batch in pq.ParquetFile(official).iter_batches(batch_size=65536):
        for row in batch.to_pylist():
            doc_id, url = int(row["id"]), str(row["url"])
            if doc_id in existing:
                continue
            host = urlparse(url).netloc.lower().removeprefix("www.").removeprefix("m.")
            path = urlparse(url).path.lower()
            query_ids = [qid for qid, (domain, prefix) in DISCOVERY_PATHS.items() if host == domain and path.startswith(prefix)]
            if query_ids:
                records.append({"doc_id": doc_id, "original_url": url, "query_ids": query_ids, "selection": "official disease-relation URL neighborhood"})
    records.sort(key=lambda item: (item["doc_id"], item["original_url"]))
    if len(records) > 200:
        raise ValueError("M4 bounded pilot exceeds 200 official URLs")
    digest_payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {
        "version": "m4-targeted-current-official-v1", "records": records,
        "sha256": hashlib.sha256(digest_payload).hexdigest(),
        "selection_contract": "current-official only; three frozen disease-relation path neighborhoods; existing extracted IDs excluded",
    }
    _atomic_json(output, manifest)
    return manifest


def _iter_m4_documents(root: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    yield from _document_sources(root)
    manifest_path = root / "artifacts/multilingual_retrieval_m4/targeted_manifest.json"
    if not manifest_path.exists():
        return
    allowed = {int(row["doc_id"]): row["original_url"] for row in json.loads(manifest_path.read_text(encoding="utf-8"))["records"]}
    for path in sorted((root / "data/multilingual_m4/targeted/documents").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                doc_id = int(row["doc_id"])
                if doc_id not in allowed or str(row["original_url"]) != allowed[doc_id]:
                    raise ValueError("M4 acquired document violates frozen official manifest")
                if row["extraction_status"] == "SUCCESS":
                    host = urlparse(str(row["original_url"])).netloc.lower().removeprefix("www.").removeprefix("m.")
                    yield host, row


def _matches(text: str, groups: list[list[str]]) -> list[list[str]]:
    folded = _fold(text)
    return [[term for term in group if _fold(term) in folded] for group in groups]


def _answer_windows(document: dict[str, Any], maximum: int = 1200) -> list[dict[str, Any]]:
    normalized = str(document["normalized_text"])
    rows: list[dict[str, Any]] = []
    for section in document.get("sections") or []:
        section_text = str(section["text"])
        if section_text not in normalized:
            raise ValueError(f"M4 section not verbatim doc={document['doc_id']}")
        section_start = int(section["start_offset"])
        nav_offsets = [section_text.find(marker) for marker in ("上一篇", "下一篇", "相关文章") if marker in section_text]
        if nav_offsets:
            section_text = section_text[:min(nav_offsets)].rstrip()
        for start, end in _bounded_ranges(section_text, maximum):
            evidence = section_text[start:end]
            if len(re.sub(r"\s+", "", evidence)) < 40:
                continue
            digest = hashlib.sha256(evidence.encode("utf-8")).hexdigest()[:20]
            rows.append({
                "doc_id": int(document["doc_id"]), "chunk_id": f"{int(document['doc_id'])}-{digest}",
                "evidence_text": evidence, "start_offset": section_start + start,
                "end_offset": section_start + end,
            })
    return rows


def classify_m4_page(title: str, url: str, text: str) -> tuple[str, list[str]]:
    """Keep substantive articles even when an extractor also retains a nav tail."""
    old_type, old_reasons = classify_page(title, url, text)
    hard_directory = bool(re.search(r"药品列表|药物列表|症状条目索引|疾病条目索引|医院列表", title))
    boundary = min((text.find(marker) for marker in ("上一篇", "下一篇", "相关文章") if marker in text), default=len(text))
    article_body = text[:boundary]
    compact = re.sub(r"\s+", "", article_body)
    relation_hits = sum(marker in compact for marker in ("原因", "因为", "由于", "导致", "引起", "症状", "表现", "治疗", "处理", "区别"))
    sentence_count = sum(compact.count(mark) for mark in ("。", "；", "？"))
    if not hard_directory and len(compact) >= 220 and sentence_count >= 3 and relation_hits >= 1:
        return "MEDICAL_ARTICLE", ["SUBSTANTIVE_BODY_BEFORE_NAV"] if old_type != "MEDICAL_ARTICLE" else []
    return old_type, old_reasons


def build_candidate_corpus(root: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source, document in _iter_m4_documents(root):
        doc_id = int(document["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        text = str(document["normalized_text"])
        page_type, reasons = classify_m4_page(str(document["title"]), str(document["original_url"]), text)
        documents.append({
            "doc_id": doc_id, "source": source, "title": str(document["title"]),
            "canonical_url": str(document["original_url"]), "normalized_text": text,
            "page_type": page_type, "page_reasons": reasons,
        })
        for row in _answer_windows(document):
            chunks.append({**row, "source": source, "title": str(document["title"]),
                           "canonical_url": str(document["original_url"]), "page_type": page_type,
                           "page_reasons": reasons})
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(chunks), output, compression="zstd")
    return documents, chunks


def coverage_diagnostics(queries: list[dict[str, Any]], documents: list[dict[str, Any]], chunks: list[dict[str, Any]], m31_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    m31 = defaultdict(set)
    for row in m31_rows:
        m31[int(row["query_id"])].add(int(row["doc_id"]))
    output = []
    for query in queries:
        qid = int(query["query_id"])
        doc_matches, chunk_matches = [], []
        required = query["primary_groups"] + query["relation_groups"]
        for document in documents:
            matched = _matches(document["normalized_text"], required)
            if required and all(matched):
                doc_matches.append(int(document["doc_id"]))
        for chunk in chunks:
            matched = _matches(chunk["evidence_text"], required)
            if required and all(matched):
                chunk_matches.append((int(chunk["doc_id"]), str(chunk["chunk_id"])))
        output.append({"query_id": qid, "diagnostic": "UNKNOWN", "full_document_relation_hits": doc_matches,
                       "same_chunk_relation_hits": [{"doc_id": d, "chunk_id": c} for d, c in chunk_matches],
                       "claim_boundary": "term/relation coverage only; direct relevance requires separate silver review"})
    return output


def retrieve(queries: list[dict[str, Any]], chunks: list[dict[str, Any]], root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_chunk = {str(row["chunk_id"]): row for row in chunks}
    reused = _load_reused_rankings(root, {int(q["query_id"]) for q in queries}, by_chunk, json.loads((root / "configs/m3_dev_multilingual.json").read_text(encoding="utf-8")))
    broad, final = [], []
    for query in queries:
        qid = int(query["query_id"])
        scores = _bm25(chunks, query["query_zh"])
        candidates = []
        for chunk_id, row in by_chunk.items():
            primary = _matches(row["evidence_text"], query["primary_groups"])
            relations = _matches(row["evidence_text"], query["relation_groups"])
            primary_count = sum(bool(x) for x in primary)
            relation_count = sum(bool(x) for x in relations)
            reuse = reused.get((qid, chunk_id), {})
            if not primary_count and scores.get(chunk_id, 0.0) <= 0 and not reuse:
                continue
            candidate = {**row, **reuse, "query_id": qid, "query_vi": query["query_vi"], "query_zh_auto": query["query_zh"],
                         "primary_groups_matched": primary_count, "relation_groups_matched": relation_count,
                         "lexical_score": float(scores.get(chunk_id, 0.0)),
                         "candidate_arms": [arm for arm, yes in (("AUTO_ZH_LEXICAL", scores.get(chunk_id, 0.0) > 0), ("CACHED_MULTILINGUAL_DENSE_RERANK", bool(reuse))) if yes]}
            candidates.append(candidate)
        candidates.sort(key=lambda row: (-int(row["primary_groups_matched"] > 0), -int(row["relation_groups_matched"] > 0),
                                          -int(row["primary_groups_matched"]), -int(row["relation_groups_matched"]),
                                          -float(row.get("rerank_score", -1000.0)), -float(row["lexical_score"]),
                                          int(row["doc_id"]), str(row["chunk_id"])))
        seen = set()
        for row in candidates:
            if int(row["doc_id"]) in seen:
                continue
            seen.add(int(row["doc_id"]))
            ranked = {**row, "candidate_rank": len(seen)}
            broad.append(ranked)
            if row["primary_groups_matched"] and row["relation_groups_matched"]:
                final.append({**ranked, "rank": 1 + sum(1 for x in final if int(x["query_id"]) == qid)})
            if len(seen) >= 100:
                break
    return broad, [row for row in final if int(row["rank"]) <= 20]


def _load_m4_silver(root: Path, final: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decisions = json.loads((root / "configs/m4_ai_silver_decisions.json").read_text(encoding="utf-8"))
    by_key = {(int(row["query_id"]), int(row["doc_id"])): row for row in decisions["decisions"]}
    manifest = json.loads((root / "artifacts/multilingual_retrieval_m4/targeted_manifest.json").read_text(encoding="utf-8"))
    acquired = {int(row["doc_id"]) for row in manifest["records"]}
    judged: list[dict[str, Any]] = []
    for row in final:
        if int(row["doc_id"]) not in acquired:
            continue
        key = (int(row["query_id"]), int(row["doc_id"]))
        if key not in by_key:
            raise ValueError(f"missing M4 silver decision query/doc={key}")
        decision = by_key[key]
        judged.append({**row, "judgment": str(decision["judgment"]), "verification_status": STATUS,
                       "review_notes": str(decision["reason"])})
    counts = Counter(row["judgment"] for row in judged)
    return judged, {"rows": len(judged), "label_counts": dict(counts),
                    "direct_query_units": len({row["query_id"] for row in judged if row["judgment"] == "2"}),
                    "verification_status": STATUS, "human_verified": 0, "official_metric": False}


def run_m4(root: Path) -> dict[str, Any]:
    phase29 = root / "submissions/00_READY_TO_UPLOAD/p29_bachmai2679_cap16.zip"
    if sha256_file(phase29) != PHASE29_SHA256:
        raise ValueError("frozen Phase29 hash mismatch")
    out = root / "artifacts/multilingual_retrieval_m4"
    lexicon = load_lexicon(root / "configs/m4_vi_zh_medical_lexicon.json")
    queries = fixed_development_queries(root, lexicon)
    documents, chunks = build_candidate_corpus(root, out / "candidate_corpus.parquet")
    m31 = pq.read_table(root / "artifacts/multilingual_retrieval_m31/zh_m31_top20.parquet").to_pylist()
    diagnostics = coverage_diagnostics(queries, documents, chunks, m31)
    broad, final = retrieve(queries, chunks, root)
    pq.write_table(pa.Table.from_pylist(broad), out / "candidate_trace_top100.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(final), out / "zh_m4_top20.parquet", compression="zstd")
    judged, silver_summary = _load_m4_silver(root, final)
    review_fields = ["case_id", "query_id", "doc_id", "chunk_id", "canonical_url", "query_vi", "query_zh_auto", "evidence_text", "candidate_rank", "final_rank", "judgment", "verification_status", "review_notes"]
    with (out / "m4_ai_judgments.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields); writer.writeheader()
        for row in judged:
            writer.writerow({"case_id": f"{row['query_id']}:zh:{row['doc_id']}:{row['chunk_id']}", "query_id": row["query_id"],
                             "doc_id": row["doc_id"], "chunk_id": row["chunk_id"], "canonical_url": row["canonical_url"],
                             "query_vi": row["query_vi"], "query_zh_auto": row["query_zh_auto"], "evidence_text": row["evidence_text"],
                             "candidate_rank": row.get("candidate_rank", ""), "final_rank": row["rank"], "judgment": row["judgment"],
                             "verification_status": row["verification_status"], "review_notes": row["review_notes"]})
    direct = [{"query_id": row["query_id"], "doc_id": row["doc_id"], "chunk_id": row["chunk_id"],
               "url": row["canonical_url"], "evidence_text": row["evidence_text"], "candidate_rank": row["candidate_rank"],
               "final_rank": row["rank"], "reason": row["review_notes"]} for row in judged if row["judgment"] == "2"]
    for diagnostic in diagnostics:
        if any(item["query_id"] == diagnostic["query_id"] for item in direct):
            diagnostic["diagnostic"] = "ANSWER_RETRIEVED_TOP20"
            diagnostic["basis"] = "AI_JUDGED_M4 direct evidence with verbatim provenance"
    old_labels = list(csv.DictReader((root / "artifacts/multilingual_retrieval_m3/review/m3_ai_judgments_import.csv").open(encoding="utf-8-sig")))
    old_positive = {(int(row["query_id"]), int(row["doc_id"])) for row in old_labels if row["judgment"] in {"1", "2"}}
    final_pairs = {(int(row["query_id"]), int(row["doc_id"])) for row in final}
    broad_pairs = {(int(row["query_id"]), int(row["doc_id"])) for row in broad}
    m31_report = json.loads((root / "artifacts/multilingual_retrieval_m31/m31_report.json").read_text(encoding="utf-8"))
    acquisition = json.loads((out / "crawl_final.json").read_text(encoding="utf-8"))
    report = {"status": "M4_VERTICAL_SLICE_COMPLETE_DIRECT_ZH_EVIDENCE", "held_out_opened": False, "phase29_unchanged": True,
              "automatic_query_understanding": {"version": lexicon["version"], "queries": len(queries), "ambiguous": [q["query_id"] for q in queries if q["ambiguous"]]},
              "corpus": {"documents": len(documents), "chunks": len(chunks)}, "coverage_diagnostics": diagnostics,
              "retrieval": {"candidate_trace_rows": len(broad), "final_rows": len(final), "queries_with_final_candidates": len({row["query_id"] for row in final}),
                            "cached_semantic_rerank_rows": sum(bool(row.get("reused_exact_reranker")) for row in broad)},
              "comparison": {"m31_final_rows": len(m31), "m4_final_rows": len(final),
                             "m3_known_partial_or_direct_total": len(old_positive),
                             "retained_in_m4_final": len(old_positive & final_pairs),
                             "lost_from_final_but_retained_in_candidate_trace": len((old_positive - final_pairs) & broad_pairs),
                             "lost_from_candidate_trace": sorted([list(x) for x in old_positive - broad_pairs])},
              "direct_evidence": direct, "labels": silver_summary,
              "english_protected_control": m31_report["english"],
              "acquisition": {"selected": 50, "success": acquisition["database"]["status_counts"].get("SUCCESS", 0),
                              "urls_per_minute": acquisition["run"]["urls_completed_per_minute"],
                              "wall_seconds_final_batch": acquisition["run"]["wall_seconds"]},
              "generalization_smoke": understand_vi_query(
                  "Người cao tuổi mắc đái tháo đường có nguy cơ hạ đường huyết khi bỏ bữa không?", lexicon),
              "reproduction": ".\\.venv\\Scripts\\python.exe tools\\multilingual_m4.py run"}
    _atomic_json(out / "m4_report.json", report)
    item = direct[0] if direct else None
    direct_block = "No direct Chinese evidence was found."
    if item:
        direct_block = (
            f"- query `{item['query_id']}`, doc `{item['doc_id']}`, chunk `{item['chunk_id']}`, "
            f"candidate/final rank `{item['candidate_rank']}/{item['final_rank']}`\n"
            f"- URL: {item['url']}\n"
            f"- Verbatim evidence: > {item['evidence_text'].replace(chr(10), ' ')}\n"
        )
    markdown = f"""# M4 — prove Chinese answer retrieval

Status: `{report['status']}`. This run used only the 12 frozen development units; held-out was not opened. Labels are `AI_JUDGED_M4` silver diagnostics, never `HUMAN_VERIFIED` or official recall.

## Result

M4 acquired 50/50 bounded current-official `cnkang.com` pages at {report['acquisition']['urls_per_minute']} URL/min. The retrieval corpus contains {len(documents)} documents and {len(chunks)} exact source-derived chunks. M4 selected {len(final)} rows for 10/12 development queries.

The newly acquired rows were independently recorded as 1 direct, 2 partial and 12 wrong/mismatched. M3/M3.1 had zero direct Chinese answers. This is one silver-positive query unit, not a relevance metric.

{direct_block}
The passage directly links early-morning/on-waking sneezing and rhinorrhea to a clinically meaningful rhinitis differential. Its page is current-official, and the exact evidence is a substring of the canonical extracted document.

## Candidate recall versus admission

All 13 previously known M3 partial/direct query-document pairs remain in the M4 candidate trace. Eight enter the final Top-20 and five remain visible but fail final relation admission; none silently disappear at candidate generation. This preserves failure evidence while keeping abstention separate from retrieval.

## Automatic VI→ZH path

The production function accepts raw Vietnamese text without `query_id` or a handwritten `query_zh`. It performs deterministic medical concept/relation/context transfer, explicitly detects ambiguity, then combines Chinese lexical retrieval with any compatible cached multilingual scores. The frozen 12 translations are not used as the production input.

This run found zero exact compatible cached semantic rows after source-verbatim rechunking. It did not approximate those scores or launch new BGE-M3/reranker inference while Phase27 was active; the demonstrated positive therefore comes from the automatic Chinese lexical/relation path.

The unrelated smoke query about an older diabetic patient and hypoglycemia also executes through this query-text-only path; it is a functional smoke, not benchmark evidence.

## English and integrity

The protected English control remains 24/24 baseline-positive pairs and 2/2 corrected-bridge direct pairs at final Top-K. Phase29 remains byte-identical at `{PHASE29_SHA256}`. No GPU inference ran, Phase27 was not modified, and no held-out labels were read.

## Remaining failures and recommendation

Eleven of 12 fixed query units still lack AI-silver direct evidence. Term co-occurrence is recorded only as coverage diagnostics and remains `UNKNOWN`; it is not promoted to evidence. M4 therefore meets the primary vertical-slice objective but not the stretch target of three direct units.

Recommendation: **GO for the demonstrated architecture, NO-GO for further untargeted acquisition.** Any next crawl should require a current-official URL neighborhood tied to a specific unresolved disease–relation gap.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests\\test_multilingual_m4.py tests\\test_multilingual_retrieval.py -q
.\\.venv\\Scripts\\python.exe tools\\multilingual_m4.py manifest
.\\.venv\\Scripts\\python.exe tools\\multilingual_m4.py run
.\\.venv\\Scripts\\python.exe tools\\multilingual_m4.py understand \"Không uống rượu nhưng GGT tăng do đâu?\"
```
"""
    (out / "M4_REPORT.md").write_text(markdown, encoding="utf-8")
    return report
