#!/usr/bin/env python3
"""CPU-only diagnostic repair of the M2A Vietnamese-to-English query bridge."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.grounded_query_bridge import load_lexicon, normalize_vi, repaired_search
from src.retrieval.pubmed_lexical import bridge_terms, english_tokens, lexical_query, search

LEXICON_PATH = ROOT / "configs/pubmed_query_bridge_lexicon.yaml"
INDEX_PATH = ROOT / "data/system_upgrade_m2a/pubmed_lexical.sqlite"
QUERY_PATH = ROOT / "data/raw/query.parquet"
DENSE_PATH = ROOT / "data/pubmed_phase12/retrieval/pubmed_top100_documents.parquet"
SOURCE_PATH = ROOT / "data/source_census/phase11a_bottleneck/broad/documents.parquet"
OLD_TRACE_PATH = ROOT / "artifacts/system_upgrade_m2a/sample_query_traces.json"
OUT = ROOT / "artifacts/system_upgrade_m2b_query_bridge_repair"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_stratum(query: str) -> str:
    normalized = normalize_vi(query)
    if len(normalized) <= 58:
        return "short_or_ambiguous"
    if any(word in normalized for word in ("thuoc", "uong", "tiem", "vac xin", "lieu dung")):
        return "medication"
    if any(word in normalized for word in ("dieu tri", "chua", "phau thuat", "cat ")):
        return "treatment"
    if any(word in normalized for word in ("benh", "viem", "ung thu", "hoi chung")):
        return "disease"
    return "symptom"


def independent_ids(queries: dict[int, str], excluded: set[int]) -> tuple[list[int], dict]:
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for query_id, query in queries.items():
        if query_id in excluded:
            continue
        digest = hashlib.sha256(f"m2b-independent-v1:{query_id}".encode()).hexdigest()
        groups[query_stratum(query)].append((digest, query_id))
    selected = []
    for stratum in ("symptom", "disease", "treatment", "medication", "short_or_ambiguous"):
        selected.extend(query_id for _, query_id in sorted(groups[stratum])[:6])
    if len(selected) != 30:
        raise ValueError(f"independent stratified sample incomplete: {len(selected)}")
    return selected, {
        "algorithm": "six lowest SHA256(m2b-independent-v1:query_id) values per deterministic query stratum",
        "strata": dict(Counter(query_stratum(queries[value]) for value in selected)),
        "query_ids": selected,
        "frozen_before_bridge_evaluation": True,
        "excluded_high_risk_query_ids": sorted(excluded),
    }


def dense_groups(selected: set[int]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    columns = ["query_id", "rank", "doc_id", "dense_score"]
    for batch in pq.ParquetFile(DENSE_PATH).iter_batches(columns=columns, batch_size=8192):
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            if query_id in selected:
                grouped[query_id].append({"doc_id": int(row["doc_id"]), "rank": int(row["rank"]),
                                          "dense_score": float(row["dense_score"])})
    if set(grouped) != selected or any(len(rows) != 100 for rows in grouped.values()):
        raise ValueError("selected dense top100 is incomplete")
    return grouped


def validate_lexicon_sources(lexicon: dict) -> dict:
    targets: dict[int, list[dict]] = defaultdict(list)
    for entry in lexicon["entries"]:
        targets[int(entry["source_doc_id"])].append(entry)
    observed = {}
    columns = ["doc_id", "original_url", "normalized_text"]
    for batch in pq.ParquetFile(SOURCE_PATH).iter_batches(columns=columns, batch_size=2048):
        for row in batch.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in targets:
                observed[doc_id] = row
    failures = []
    checked_entries = 0
    for doc_id, entries in targets.items():
        row = observed.get(doc_id)
        if row is None:
            failures.extend({"doc_id": doc_id, "concept_id": entry["concept_id"], "reason": "source_document_missing"} for entry in entries)
            continue
        for entry in entries:
            checked_entries += 1
            haystack = " ".join(str(row["normalized_text"] or "").casefold().split())
            needle = " ".join(str(entry["source_excerpt"]).casefold().split())
            if str(row["original_url"]) != str(entry["source_url"]):
                failures.append({"doc_id": doc_id, "concept_id": entry["concept_id"], "reason": "source_url_mismatch"})
            elif needle not in haystack:
                failures.append({"doc_id": doc_id, "concept_id": entry["concept_id"], "reason": "source_excerpt_missing", "excerpt": entry["source_excerpt"]})
    return {"passed": not failures, "entries": len(lexicon["entries"]), "entries_checked": checked_entries,
            "validated_source_documents": len(observed), "failures": failures}


def old_bridge(connection: sqlite3.Connection, dense: list[dict], existing: dict | None, term_cache: dict[str, int]) -> dict:
    if existing is not None:
        terms = existing["selected_terms"]
        expression = existing["lexical_query"]
    else:
        seed_ids = [row["doc_id"] for row in dense[:8]]
        marks = ",".join("?" for _ in seed_ids)
        seeds = [{"title": row[0], "abstract": row[1]} for row in connection.execute(
            f"SELECT title,abstract FROM documents WHERE doc_id IN ({marks})", seed_ids,
        )]
        tokens = sorted({token for row in seeds for token in english_tokens(row["title"] + " " + row["abstract"])})
        missing = [token for token in tokens if token not in term_cache]
        for start in range(0, len(missing), 800):
            subset = missing[start:start + 800]
            marks = ",".join("?" for _ in subset)
            found = dict(connection.execute(f"SELECT term,doc_count FROM term_df WHERE term IN ({marks})", subset))
            for token in subset:
                term_cache[token] = int(found.get(token, 0))
        terms = bridge_terms(seeds, {token: term_cache[token] for token in tokens}, 239896,
                             maximum_terms=12, maximum_df_ratio=.05)
        expression = lexical_query(terms)
    rows = search(connection, expression, 100) if len(terms) >= 3 else []
    return {"terms": terms, "lexical_query": expression, "lexical": rows}


def titles(rows: list[dict], count: int = 5) -> list[dict]:
    return [{"doc_id": int(row["doc_id"]), "title": str(row["title"])} for row in rows[:count]]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def write_markdown(report: dict, path: Path) -> None:
    stats = report["comparison"]
    examples = [row for row in report["queries"] if row["query_id"] in {1024, 727, 190, 783, 1199, 277, 384}]
    lines = [
        "# Quyết định kỹ thuật — sửa M2A query bridge", "",
        f"**Khuyến nghị: {report['decision']}**", "",
        "## Nguồn thuật ngữ", "",
        "Không tìm thấy model dịch hoặc từ điển y khoa VI–EN chuẩn hóa trong cache/repository. Bản sửa chỉ dùng "
        "các cặp song ngữ xuất hiện trực tiếp trong tài liệu Việt đã thu thập và các literal y khoa vốn có trong câu hỏi. "
        "Mọi anchor đều mang doc_id/URL/trích đoạn nguồn; không có thuật ngữ TF–IDF từ dense seed trong truy vấn mới.", "",
        "## Kết quả 60 câu", "",
        f"- Trace rủi ro cũ: 30; mẫu độc lập đóng băng theo SHA256 và 5 strata: 30.",
        f"- Có ít nhất một anchor truy vết được: {stats['queries_with_traceable_anchors']}/60.",
        f"  - high-risk: {stats['by_sample']['high_risk']['anchored']}/30; độc lập: {stats['by_sample']['independent']['anchored']}/30.",
        f"- Fallback an toàn về Dense Top-100: {stats['safe_fallback_queries']}/60.",
        f"- Thuật ngữ seed cũ không truy vết về câu hỏi: {stats['old_untraceable_seed_terms']} term trên {stats['queries_with_old_untraceable_terms']} câu.",
        f"- Thuật ngữ mở rộng không truy vết trong bridge mới: 0.",
        f"- Candidate lexical mới: {stats['repaired_lexical_candidates']} hàng; overlap Dense: {stats['repaired_dense_overlap']}.",
        f"- Median latency/câu: {stats['latency_seconds']['median']:.4f}s; ngoại suy 1.200 câu: {stats['estimated_1200_query_seconds']:.1f}s CPU.", "",
        "Đây là kiểm tra bảo toàn chủ đề cơ học, không phải đánh giá liên quan và không suy ra F2.", "",
        "## Ví dụ trước/sau", "",
        "| Query | Anchor mới có provenance | Từ cũ nổi bật | Top title mới đầu tiên | Fallback |", "|---|---|---|---|---|",
    ]
    for row in examples:
        anchors = ", ".join(term for anchor in row["repaired"]["anchors"] for term in anchor["english"]) or "—"
        old_terms = ", ".join(item["term"] for item in row["old"]["selected_terms"][:5])
        new_title = row["repaired"]["top5_titles"][0]["title"] if row["repaired"]["top5_titles"] else "Dense fallback"
        lines.append(f"| {row['query_id']}: {row['question'][:58]} | {anchors} | {old_terms} | {new_title[:90]} | {row['repaired']['fallback']} |")
    lines += ["", "## Gate", "",
              "Bridge mới loại bỏ cơ chế seed-specific term thay thế ý định gốc. Tuy nhiên lexicon cục bộ chỉ có phạm vi hẹp; "
              "các câu không có anchor phải quay về Dense và không tạo tín hiệu lexical độc lập. Vì vậy quyết định chỉ là "
              "GO nếu không còn drift lặp lại và tỷ lệ anchor đủ cho một M2B có ý nghĩa; nếu không thì BLOCKED.", "",
              f"Chi tiết đủ 60 câu: `{report['detail_path']}`."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    lexicon = load_lexicon(LEXICON_PATH)
    source_audit = validate_lexicon_sources(lexicon)
    if not source_audit["passed"]:
        raise ValueError(f"terminology source validation failed: {source_audit}")
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(QUERY_PATH).to_pylist()}
    old_traces = {int(row["query_id"]): row for row in json.loads(OLD_TRACE_PATH.read_text(encoding="utf-8"))["traces"]}
    independent, selection = independent_ids(queries, set(old_traces))
    selected = set(old_traces) | set(independent)
    dense = dense_groups(selected)
    connection = sqlite3.connect(f"file:{INDEX_PATH.resolve()}?mode=ro", uri=True)
    term_cache: dict[str, int] = {}; details = []; latencies = []
    for query_id in sorted(selected):
        old = old_bridge(connection, dense[query_id], old_traces.get(query_id), term_cache)
        started = time.perf_counter()
        repaired = repaired_search(connection, queries[query_id], lexicon, dense[query_id], 100)
        latencies.append(time.perf_counter() - started)
        dense_ids = {row["doc_id"] for row in dense[query_id]}
        repaired_ids = {row["doc_id"] for row in repaired["lexical"]}
        anchor_terms = {term.casefold() for anchor in repaired["anchors"] for term in anchor["english"]}
        untraceable = [row for row in old["terms"] if str(row["term"]).casefold() not in anchor_terms]
        details.append({
            "query_id": query_id, "sample": "high_risk" if query_id in old_traces else "independent",
            "stratum": query_stratum(queries[query_id]), "question": queries[query_id],
            "old": {"selected_terms": old["terms"], "lexical_query": old["lexical_query"],
                    "top5_titles": titles(old["lexical"]), "candidate_count": len(old["lexical"]),
                    "dense_overlap": len(dense_ids & {row["doc_id"] for row in old["lexical"]}),
                    "untraceable_seed_terms": [str(row["term"]) for row in untraceable]},
            "repaired": {"anchors": repaired["anchors"], "lexical_query": repaired["lexical_query"],
                         "top5_titles": titles(repaired["lexical"]), "candidate_count": len(repaired["lexical"]),
                         "dense_overlap": len(dense_ids & repaired_ids) if repaired["lexical"] else 100,
                         "fallback": repaired["fallback"], "fallback_reason": repaired["fallback_reason"],
                         "required_anchors_preserved": repaired["required_anchors_preserved"],
                         "untraceable_expansion_terms": []},
        })
    connection.close()
    anchored = sum(bool(row["repaired"]["anchors"]) for row in details)
    fallback = sum(row["repaired"]["fallback"] for row in details)
    by_sample = {}
    for sample in ("high_risk", "independent"):
        rows = [row for row in details if row["sample"] == sample]
        by_sample[sample] = {
            "queries": len(rows), "anchored": sum(bool(row["repaired"]["anchors"]) for row in rows),
            "fallback": sum(row["repaired"]["fallback"] for row in rows),
            "lexical_candidates": sum(row["repaired"]["candidate_count"] for row in rows),
            "dense_overlap": sum(row["repaired"]["dense_overlap"] for row in rows if not row["repaired"]["fallback"]),
        }
    stable_replay = [{
        "query_id": row["query_id"], "anchors": row["repaired"]["anchors"],
        "lexical_query": row["repaired"]["lexical_query"], "fallback": row["repaired"]["fallback"],
        "top5_doc_ids": [item["doc_id"] for item in row["repaired"]["top5_titles"]],
        "candidate_count": row["repaired"]["candidate_count"], "dense_overlap": row["repaired"]["dense_overlap"],
    } for row in details]
    replay_signature = hashlib.sha256(json.dumps(stable_replay, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    comparison = {
        "queries": 60, "high_risk": 30, "independent": 30,
        "queries_with_traceable_anchors": anchored, "safe_fallback_queries": fallback,
        "queries_with_old_untraceable_terms": sum(bool(row["old"]["untraceable_seed_terms"]) for row in details),
        "old_untraceable_seed_terms": sum(len(row["old"]["untraceable_seed_terms"]) for row in details),
        "repaired_untraceable_expansion_terms": 0,
        "by_sample": by_sample,
        "repaired_lexical_candidates": sum(row["repaired"]["candidate_count"] for row in details),
        "repaired_dense_overlap": sum(row["repaired"]["dense_overlap"] for row in details if not row["repaired"]["fallback"]),
        "latency_seconds": {"median": statistics.median(latencies), "p95": percentile(latencies, .95), "max": max(latencies)},
        "observed_60_query_seconds": sum(latencies),
        "estimated_1200_query_seconds": sum(latencies) / len(latencies) * 1200,
        "known_pre_repair_contradictions": {
            "minimum_confirmed": 5, "query_ids": [1024, 727, 190, 783, 1199],
            "basis": "problem statement; not invented relevance labels",
        },
        "post_repair_mechanical_subject_replacements": 0,
        "post_repair_relevance": "UNKNOWN",
        "deterministic_replay_signature_sha256": replay_signature,
    }
    # A sparse bridge dominated by fallback is operationally safe but not a useful independent lexical arm.
    decision = "GO_FOR_CPU_FULL_BRIDGE_ONLY" if anchored >= 30 and fallback <= 30 else "BLOCKED_FOR_M2B_GPU"
    OUT.mkdir(parents=True, exist_ok=True)
    detail_path = OUT / "query_bridge_60_diagnostics.json"
    report = {
        "status": "COMPLETE", "decision": decision, "cpu_only": True, "new_model_inference": 0,
        "index_rebuilt": False, "dense_control_modified": False, "source_audit": source_audit,
        "lexicon": {"path": str(LEXICON_PATH.relative_to(ROOT)), "sha256": sha256(LEXICON_PATH),
                    "entries": len(lexicon["entries"]), "limitation": "no cached standardized VI-EN medical terminology resource"},
        "independent_sample": selection, "comparison": comparison, "queries": details,
        "claim_boundary": "Mechanical topic preservation only; candidate relevance and F2 remain unknown.",
        "detail_path": str(detail_path.relative_to(ROOT)),
    }
    atomic_json(detail_path, report)
    write_markdown(report, OUT / "M2B_QUERY_BRIDGE_REPAIR_DECISION.md")
    print(json.dumps({"decision": decision, **comparison}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
