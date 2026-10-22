from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from src.multilingual_retrieval.m2 import chinese_evidence_quality, sha256_file, _zh_bigrams


SILVER_STATUS = "AI_JUDGED_M3"
SOURCE_PATHS = {
    "cnkang.com": "data/source_acquisition/cnkang.com/final1000",
    "familydoctor.com.cn": "data/source_acquisition/familydoctor.com.cn/final1000",
    "a-hospital.com": "data/source_acquisition/a-hospital.com/final1000",
}
NAVIGATION_PARAGRAPHS = (
    "上一篇", "下一篇", "相关文章", "精彩推荐", "热门推荐", "更多医学百科条目",
    "专家答疑", "最新文章", "找医生", "我要咨询", "疾病常识", "诊断治疗",
)


def _host(url: str) -> str:
    return (urllib.parse.urlsplit(str(url)).hostname or "").casefold().removeprefix("www.")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def current_official_records(path: Path, domains: set[str]) -> dict[str, dict[int, str]]:
    output = {domain: {} for domain in domains}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["id", "url"], batch_size=65536):
        for row in batch.to_pylist():
            domain = _host(row["url"])
            if domain in output:
                output[domain][int(row["id"])] = str(row["url"])
    return output


def _clean_evidence(text: str) -> str:
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n+", str(text))]
    kept: list[str] = []
    for paragraph in paragraphs:
        if not paragraph or paragraph in NAVIGATION_PARAGRAPHS:
            continue
        if any(paragraph.startswith(value) for value in NAVIGATION_PARAGRAPHS) and len(paragraph) < 120:
            continue
        if len(re.findall(r"2025-\d\d-\d\d", paragraph)) >= 2:
            continue
        kept.append(paragraph)
    return "\n".join(kept)


def filter_current_chinese_cache(root: Path, output_path: Path) -> dict[str, Any]:
    official = current_official_records(
        root / "data/raw/links_corpus_vi_en_zh.parquet", set(SOURCE_PATHS)
    )
    rows: list[dict[str, Any]] = []
    census: dict[str, Any] = {}
    for domain, relative in SOURCE_PATHS.items():
        base = root / relative
        current_ids = set(official[domain])
        cached_ids: set[int] = set()
        for batch in pq.ParquetFile(base / "documents.parquet").iter_batches(
            columns=["doc_id"], batch_size=512
        ):
            cached_ids.update(int(value) for value in batch.column(0).to_pylist())
        current_cached = cached_ids & current_ids
        rejected = 0
        for batch in pq.ParquetFile(base / "chunks.parquet").iter_batches(batch_size=256):
            for row in batch.to_pylist():
                doc_id = int(row["doc_id"])
                if doc_id not in current_cached:
                    continue
                evidence = _clean_evidence(str(row["raw_text"]))
                passed, reasons = chinese_evidence_quality(evidence)
                if not passed:
                    rejected += 1
                    continue
                rows.append({
                    "doc_id": doc_id,
                    "chunk_id": str(row["chunk_id"]),
                    "evidence_text": evidence,
                    "canonical_url": official[domain][doc_id],
                    "source": domain,
                    "chunk_index": int(row["chunk_index"]),
                    "quality_status": "PASS",
                    "quality_reasons": reasons,
                    "acquisition_arm": "historical_cache_current_official",
                })
        census[domain] = {
            "current_official_ids": len(current_ids),
            "cached_documents": len(cached_ids),
            "current_official_cached_documents": len(current_cached),
            "eligible_chunks": sum(row["source"] == domain for row in rows),
            "rejected_chunks": rejected,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="zstd")
    return {"sources": census, "eligible_documents": len({row["doc_id"] for row in rows}), "eligible_chunks": len(rows)}


def targeted_document_chunks(documents_directory: Path) -> list[dict[str, Any]]:
    """Create small exact-source sections without loading a tokenizer/model."""
    rows: list[dict[str, Any]] = []
    if not documents_directory.exists():
        return rows
    for path in sorted(documents_directory.glob("*.parquet")):
        columns = ["doc_id", "original_url", "normalized_text", "sections", "extraction_status"]
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=64):
            for document in batch.to_pylist():
                if str(document["extraction_status"]) != "SUCCESS":
                    continue
                normalized = str(document["normalized_text"])
                for section_index, section in enumerate(document["sections"] or []):
                    raw = str(section["text"])
                    evidence = raw.strip()
                    passed, reasons = chinese_evidence_quality(evidence)
                    if not passed:
                        continue
                    if evidence not in normalized:
                        raise ValueError("targeted section is not verbatim in normalized document")
                    digest = hashlib.sha256(evidence.encode("utf-8")).hexdigest()[:20]
                    rows.append({
                        "doc_id": int(document["doc_id"]),
                        "chunk_id": f"{int(document['doc_id'])}-{digest}",
                        "evidence_text": evidence,
                        "canonical_url": str(document["original_url"]),
                        "source": "a-hospital.com",
                        "chunk_index": section_index,
                        "quality_status": "PASS",
                        "quality_reasons": reasons,
                        "acquisition_arm": "m3_targeted_current_official",
                    })
    return rows


def select_targeted_official_manifest(root: Path, config: dict[str, Any], path: Path) -> dict[str, Any]:
    official = current_official_records(
        root / "data/raw/links_corpus_vi_en_zh.parquet", {"a-hospital.com"}
    )["a-hospital.com"]
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for query in config["zh_queries"]:
        terms = list(query["core"]) + list(query["support"])
        for doc_id, url in official.items():
            decoded = urllib.parse.unquote(url).casefold()
            matched = [term for term in terms if term.casefold() in decoded]
            if not matched or doc_id in seen:
                continue
            seen.add(doc_id)
            records.append({
                "doc_id": doc_id,
                "original_url": url,
                "query_ids": [int(query["query_id"])],
                "url_matched_terms": matched,
            })
    records.sort(key=lambda row: (row["doc_id"], row["original_url"]))
    payload = {
        "version": "m3-targeted-current-official-v1",
        "selection": "current-official A-Hospital URLs whose decoded canonical URL contains a frozen development medical concept",
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["sha256"] = sha256_file(path)
    return payload


def _bm25(rows: list[dict[str, Any]], query: str) -> dict[str, float]:
    query_terms = _zh_bigrams(query)
    documents = {str(row["chunk_id"]): _zh_bigrams(str(row["evidence_text"])) for row in rows}
    lengths = {key: len(value) for key, value in documents.items()}
    average = sum(lengths.values()) / max(1, len(lengths))
    frequencies = Counter(term for terms in documents.values() for term in set(terms))
    scores: dict[str, float] = defaultdict(float)
    n = len(documents)
    for key, terms in documents.items():
        counts = Counter(terms)
        for term in query_terms:
            if term not in counts:
                continue
            inverse = math.log(1.0 + (n - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            frequency = counts[term]
            denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * lengths[key] / max(average, 1.0))
            scores[key] += inverse * frequency * 2.2 / denominator
    return scores


def _concept_counts(row: dict[str, Any], query: dict[str, Any]) -> tuple[int, int]:
    text = str(row["evidence_text"]).casefold()
    return (
        sum(str(term).casefold() in text for term in query["core"]),
        sum(str(term).casefold() in text for term in query["support"]),
    )


def _load_reused_rankings(root: Path, qids: set[int], eligible: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    reused: dict[tuple[int, str], dict[str, Any]] = {}
    for domain, source_id in config["chinese_sources"].items():
        path = root / "data/source_relevance_probe" / source_id / "reranked_chunks.parquet"
        columns = ["query_id", "doc_id", "chunk_id", "dense_rank", "dense_score", "sparse_rank", "hybrid_rank", "rerank_rank", "rerank_score"]
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=2048):
            for row in batch.to_pylist():
                key = (int(row["query_id"]), str(row["chunk_id"]))
                if key[0] not in qids or str(row["chunk_id"]) not in eligible:
                    continue
                reused[key] = {
                    "dense_rank": row["dense_rank"], "dense_score": row["dense_score"],
                    "sparse_rank_old": row["sparse_rank"], "hybrid_rank_old": row["hybrid_rank"],
                    "rerank_rank_old": row["rerank_rank"], "rerank_score": row["rerank_score"],
                    "reused_exact_reranker": True,
                }
    return reused


def retrieve_chinese(root: Path, config: dict[str, Any], chunks_path: Path, output: Path) -> list[dict[str, Any]]:
    corpus = pq.read_table(chunks_path).to_pylist()
    by_chunk = {str(row["chunk_id"]): row for row in corpus}
    queries = {int(row["query_id"]): row for row in config["zh_queries"]}
    reused = _load_reused_rankings(root, set(queries), by_chunk, config)
    output_rows: list[dict[str, Any]] = []
    for query_id, query in sorted(queries.items()):
        scores = _bm25(corpus, str(query["query_zh"]))
        candidates: list[dict[str, Any]] = []
        for chunk_id, row in by_chunk.items():
            core, support = _concept_counts(row, query)
            reuse = reused.get((query_id, chunk_id), {})
            if scores.get(chunk_id, 0.0) <= 0 and not reuse:
                continue
            candidates.append({
                **row, **reuse, "query_id": query_id, "query_zh": query["query_zh"],
                "lexical_score": float(scores.get(chunk_id, 0.0)),
                "core_concept_matches": core, "support_concept_matches": support,
            })
        candidates.sort(key=lambda row: (
            -int(row["core_concept_matches"] > 0), -int(row["core_concept_matches"]),
            -int(row["support_concept_matches"]), -float(row["lexical_score"]),
            -float(row.get("rerank_score", -1000.0)), int(row["doc_id"]), str(row["chunk_id"]),
        ))
        seen: set[int] = set()
        for candidate in candidates:
            if int(candidate["doc_id"]) in seen:
                continue
            seen.add(int(candidate["doc_id"]))
            candidate["rank"] = len(seen)
            output_rows.append(candidate)
            if len(seen) == 20:
                break
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), output, compression="zstd")
    return output_rows


def build_english_protected_arm(root: Path, config: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    baseline = pq.read_table(root / "artifacts/multilingual_retrieval_m1/cached_retrieval_top20.parquet").to_pylist()
    baseline = [row for row in baseline if row["target_language"] == "en" and row["arm"] == config["english_policy"]["control_arm"]]
    bridge = pq.read_table(root / "artifacts/multilingual_retrieval_m2/en_medical_bridge_fts5_top20.parquet").to_pylist()
    bridge_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in bridge:
        bridge_by_query[int(row["query_id"])].append(row)
    by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline:
        by_query[int(row["query_id"])].append(row)
    prefix = int(config["english_policy"]["protected_control_prefix"])
    maximum = int(config["english_policy"]["maximum_bridge_additions"])
    result: list[dict[str, Any]] = []
    for query_id, control_rows in sorted(by_query.items()):
        ordered = sorted(control_rows, key=lambda row: (int(row["rank"]), int(row["doc_id"])))
        selected = [{**row, "m3_origin": "frozen_control"} for row in ordered[:prefix]]
        seen = {int(row["doc_id"]) for row in selected}
        additions = 0
        for row in sorted(bridge_by_query.get(query_id, []), key=lambda value: (int(value["rank"]), int(value["doc_id"]))):
            if int(row["doc_id"]) in seen or additions >= maximum:
                continue
            selected.append({**row, "arm": "m3_protected_bridge", "m3_origin": "corrected_medical_bridge"})
            seen.add(int(row["doc_id"])); additions += 1
        for row in ordered[prefix:]:
            if len(selected) == 20:
                break
            if int(row["doc_id"]) not in seen:
                selected.append({**row, "m3_origin": "frozen_control_tail"}); seen.add(int(row["doc_id"]))
        for rank, row in enumerate(selected[:20], 1):
            result.append({**row, "rank": rank, "arm": "m3_protected_bridge"})
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(result), output, compression="zstd")
    return result


def _load_silver(path: Path, candidates: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lookup = {(int(row["query_id"]), int(row["doc_id"])): str(row["evidence_text"]) for row in candidates}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row.get("verification_status") != SILVER_STATUS or row.get("judgment") not in {"0", "1", "2", "UNSURE"}:
            raise ValueError("invalid M3 silver contract")
        text = lookup.get((int(row["query_id"]), int(row["doc_id"])))
        if text is None:
            raise ValueError("M3 silver candidate is absent")
        if row.get("evidence_span") and row["evidence_span"] not in text:
            raise ValueError("M3 silver evidence is not verbatim")
    return rows


def run_m3(root: Path) -> dict[str, Any]:
    config_path = root / "configs/m3_dev_multilingual.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("scope") != "development_only" or config.get("held_out_opened") is not False:
        raise ValueError("M3 must remain development-only")
    phase29 = root / "submissions/00_READY_TO_UPLOAD/p29_bachmai2679_cap16.zip"
    actual = sha256_file(phase29)
    if actual != config["phase29_sha256"]:
        raise ValueError("frozen Phase29 SHA-256 mismatch")
    out = root / "artifacts/multilingual_retrieval_m3"
    filtered_path = out / "current_official_chinese_chunks.parquet"
    census = filter_current_chinese_cache(root, filtered_path)
    manifest = select_targeted_official_manifest(root, config, out / "targeted_manifest.json")
    filtered = pq.read_table(filtered_path).to_pylist()
    targeted = targeted_document_chunks(root / "data/multilingual_m3/targeted/documents")
    if targeted:
        official_ids = {
            int(row["doc_id"])
            for row in manifest["records"]
        }
        if any(int(row["doc_id"]) not in official_ids for row in targeted):
            raise ValueError("targeted extraction contains a document outside the frozen official manifest")
        identities = {(int(row["doc_id"]), str(row["chunk_id"])) for row in filtered}
        filtered.extend(row for row in targeted if (int(row["doc_id"]), str(row["chunk_id"])) not in identities)
        pq.write_table(pa.Table.from_pylist(filtered), filtered_path, compression="zstd")
    zh = retrieve_chinese(root, config, filtered_path, out / "zh_m3_top20.parquet")
    en = build_english_protected_arm(root, config, out / "en_m3_protected_top20.parquet")
    silver = _load_silver(out / "ai_judgments_m3.csv", zh)

    m1_silver_path = root / "artifacts/multilingual_retrieval_m1/ai_judgments_import.csv"
    with m1_silver_path.open("r", encoding="utf-8-sig", newline="") as stream:
        m1_silver = list(csv.DictReader(stream))
    relevant_en = {(int(row["query_id"]), int(row["doc_id"])) for row in m1_silver if row["target_language"] == "en" and row["judgment"] in {"1", "2"}}
    m3_en = {(int(row["query_id"]), int(row["doc_id"])) for row in en}
    m2_silver_path = root / "artifacts/multilingual_retrieval_m2/ai_judgments_m2.csv"
    with m2_silver_path.open("r", encoding="utf-8-sig", newline="") as stream:
        m2_silver = list(csv.DictReader(stream))
    new_direct = {(int(row["query_id"]), int(row["doc_id"])) for row in m2_silver if row["target_language"] == "en" and row["judgment"] == "2"}
    zh_counts = Counter(row["judgment"] for row in silver)
    report = {
        "status": "M3_DEVELOPMENT_VERTICAL_SLICE",
        "held_out_opened": False,
        "phase29": {"path": str(phase29), "sha256": actual, "unchanged": True},
        "current_official_chinese_cache": census,
        "targeted_acquisition": {
            "manifest": str(out / "targeted_manifest.json"), "rows": len(manifest["records"]),
            "sha256": manifest["sha256"], "executed": bool(targeted),
            "eligible_targeted_chunks": len(targeted),
        },
        "zh": {
            "queries": len(config["zh_queries"]), "ranking_rows": len(zh),
            "queries_with_candidates": len({row["query_id"] for row in zh}),
            "silver_rows": len(silver), "silver_counts": dict(zh_counts),
            "direct_evidence_query_units": len({int(row["query_id"]) for row in silver if row["judgment"] == "2"}),
            "official_metric": False,
        },
        "en": {
            "policy": "frozen hybrid top-12 + up to 8 corrected medical-bridge documents, then frozen tail fill",
            "known_baseline_relevant_retained": sum(pair in m3_en for pair in relevant_en),
            "known_baseline_relevant_total": len(relevant_en),
            "corrected_bridge_direct_retained": sum(pair in m3_en for pair in new_direct),
            "corrected_bridge_direct_total": len(new_direct),
            "official_metric": False,
        },
        "artifacts": {
            "zh_rankings": str(out / "zh_m3_top20.parquet"),
            "en_rankings": str(out / "en_m3_protected_top20.parquet"),
        },
    }
    _atomic_json(out / "m3_report.json", report)
    return report


def demo(root: Path, query_id: int) -> dict[str, Any]:
    out = root / "artifacts/multilingual_retrieval_m3"
    zh = [row for row in pq.read_table(out / "zh_m3_top20.parquet").to_pylist() if int(row["query_id"]) == query_id][:3]
    en = [row for row in pq.read_table(out / "en_m3_protected_top20.parquet").to_pylist() if int(row["query_id"]) == query_id][:3]
    query_rows = pq.read_table(root / "artifacts/multilingual_retrieval_m1/review_queue.parquet", columns=["query_id", "query_text"]).to_pylist()
    query = next((str(row["query_text"]) for row in query_rows if int(row["query_id"]) == query_id), "")
    return {"query_id": query_id, "query_vi": query, "en": en, "zh": zh, "labels_are_silver_only": True}
