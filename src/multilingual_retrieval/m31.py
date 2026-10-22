from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from src.multilingual_retrieval.m2 import chinese_evidence_quality, sha256_file
from src.multilingual_retrieval.m3 import (
    SOURCE_PATHS,
    _atomic_json,
    _bm25,
    build_english_protected_arm,
    current_official_records,
)


M31_STATUS = "AI_JUDGED_M3_1"
DIRECTORY_TITLE_PATTERNS = (
    r"药品列表", r"药物列表", r"症状条目索引", r"疾病条目索引", r"医院列表",
    r"药品分类", r"分类查看", r"条目索引-[A-Z]", r"症状查询专题",
)
DIRECTORY_BODY_PATTERNS = (
    "本药品列表被可按中成药和西药分类查看", "中成药物列表", "西药列表",
    "按症状名称开头字母查询症状", "点击部位名称可查询更多相关疾病",
)
NAVIGATION_MARKERS = (
    "上一篇", "下一篇", "相关文章", "精彩推荐", "热门推荐", "更多医学百科条目",
    "专家答疑", "最新文章", "找医生", "我要咨询", "疾病常识", "诊断治疗",
)
ANSWER_SIGNALS = {
    "cause": ("原因", "因为", "由于", "导致", "引起", "与", "可能"),
    "treatment": ("治疗", "处理", "建议", "应", "需要", "用药", "方案"),
    "safety": ("可以", "不宜", "避免", "饮食", "加重", "建议"),
    "interaction": ("影响", "相互作用", "同时服用", "吸收", "药物"),
    "risk": ("危险因素", "风险因素", "年龄", "增加", "易患"),
    "differential": ("鉴别", "可能", "排除", "检查", "并非"),
    "symptom": ("症状", "表现", "可出现", "常见"),
}


def classify_page(title: str, url: str, normalized_text: str) -> tuple[str, list[str]]:
    target = f"{title}\n{url}"
    reasons: list[str] = []
    for pattern in DIRECTORY_TITLE_PATTERNS:
        if re.search(pattern, target, flags=re.I):
            reasons.append(f"TITLE_OR_URL:{pattern}")
    for marker in DIRECTORY_BODY_PATTERNS:
        if marker in normalized_text:
            reasons.append(f"BODY_DIRECTORY:{marker}")
    paragraphs = [part.strip() for part in re.split(r"\n+", normalized_text) if part.strip()]
    if paragraphs:
        short_ratio = sum(len(part) <= 24 for part in paragraphs) / len(paragraphs)
        navigation_hits = sum(any(marker in part for marker in NAVIGATION_MARKERS) for part in paragraphs)
        if len(paragraphs) >= 25 and short_ratio >= 0.72 and navigation_hits >= 3:
            reasons.append("LINK_LIST_STRUCTURE")
    return ("DIRECTORY_OR_NAVIGATION", reasons) if reasons else ("MEDICAL_ARTICLE", [])


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S(?:.*?\S)?(?=\n{2,}|$)", text, re.S)]


def _bounded_ranges(text: str, maximum: int) -> Iterable[tuple[int, int]]:
    ranges = _paragraph_ranges(text)
    start: int | None = None
    end: int | None = None
    for paragraph_start, paragraph_end in ranges:
        if paragraph_end - paragraph_start > maximum:
            if start is not None and end is not None:
                yield start, end
                start = end = None
            offset = paragraph_start
            while offset < paragraph_end:
                boundary = min(offset + maximum, paragraph_end)
                if boundary < paragraph_end:
                    sentence = max(text.rfind("。", offset + maximum // 2, boundary), text.rfind("；", offset + maximum // 2, boundary))
                    if sentence >= 0:
                        boundary = sentence + 1
                yield offset, boundary
                offset = boundary
            continue
        if start is None:
            start, end = paragraph_start, paragraph_end
        elif paragraph_end - start <= maximum:
            end = paragraph_end
        else:
            assert end is not None
            yield start, end
            start, end = paragraph_start, paragraph_end
    if start is not None and end is not None:
        yield start, end


def classify_chunk(text: str) -> tuple[bool, list[str]]:
    passed, reasons = chinese_evidence_quality(text)
    reasons = list(reasons)
    normalized = re.sub(r"\s+", "", text)
    if any(marker in normalized for marker in DIRECTORY_BODY_PATTERNS):
        reasons.append("DIRECTORY_FRAGMENT")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 12 and sum(len(line) <= 24 for line in lines) / len(lines) >= 0.75:
        reasons.append("LINK_LIST_FRAGMENT")
    repeated = Counter(lines)
    if lines and sum(count - 1 for count in repeated.values()) / len(lines) >= 0.35:
        reasons.append("REPETITIVE_FRAGMENT")
    return passed and not reasons, sorted(set(reasons))


def _document_sources(root: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    domains = set(SOURCE_PATHS)
    official = current_official_records(root / "data/raw/links_corpus_vi_en_zh.parquet", domains)
    seen: set[int] = set()
    for domain, relative in SOURCE_PATHS.items():
        allowed = set(official[domain])
        path = root / relative / "documents.parquet"
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                doc_id = int(row["doc_id"])
                if doc_id in allowed and doc_id not in seen and row["extraction_status"] == "SUCCESS":
                    seen.add(doc_id)
                    yield domain, row
    manifest_path = root / "artifacts/multilingual_retrieval_m3/targeted_manifest.json"
    targeted_allowed = {int(row["doc_id"]) for row in json.loads(manifest_path.read_text(encoding="utf-8"))["records"]}
    for path in sorted((root / "data/multilingual_m3/targeted/documents").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                doc_id = int(row["doc_id"])
                if doc_id not in targeted_allowed:
                    raise ValueError("targeted document is outside frozen current-official manifest")
                if doc_id not in seen and row["extraction_status"] == "SUCCESS":
                    seen.add(doc_id)
                    yield "a-hospital.com", row


def build_quality_corpus(root: Path, config: dict[str, Any], output: Path, audit_path: Path) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    maximum = int(config["maximum_chunk_characters"])
    page_counts: Counter[str] = Counter()
    for domain, document in _document_sources(root):
        doc_id = int(document["doc_id"])
        title = str(document["title"])
        url = str(document["original_url"])
        normalized = str(document["normalized_text"])
        page_type, page_reasons = classify_page(title, url, normalized)
        page_counts[page_type] += 1
        if page_type != "MEDICAL_ARTICLE":
            audit.append({"doc_id": doc_id, "chunk_id": "", "source": domain, "status": "REJECT_PAGE", "reasons": page_reasons, "title": title, "canonical_url": url})
            continue
        chunk_index = 0
        for section in document["sections"] or []:
            section_text = str(section["text"])
            section_start = int(section["start_offset"])
            if section_text not in normalized:
                raise ValueError(f"section is not verbatim doc={doc_id}")
            for local_start, local_end in _bounded_ranges(section_text, maximum):
                evidence = section_text[local_start:local_end]
                passed, chunk_reasons = classify_chunk(evidence)
                digest = hashlib.sha256(evidence.encode("utf-8")).hexdigest()[:20]
                chunk_id = f"{doc_id}-{digest}"
                record = {
                    "doc_id": doc_id, "chunk_id": chunk_id, "source": domain,
                    "canonical_url": url, "title": title, "page_type": page_type,
                    "page_reasons": page_reasons, "chunk_reasons": chunk_reasons,
                    "start_offset": section_start + local_start,
                    "end_offset": section_start + local_end,
                }
                if not passed:
                    audit.append({**record, "status": "REJECT_CHUNK"})
                    continue
                if evidence not in normalized:
                    raise ValueError(f"chunk is not verbatim doc={doc_id}")
                accepted.append({
                    **record, "evidence_text": evidence, "chunk_index": chunk_index,
                    "quality_status": "PASS_M3_1",
                })
                chunk_index += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(accepted), output, compression="zstd")
    pq.write_table(pa.Table.from_pylist(audit), audit_path, compression="zstd")
    return {
        "documents": sum(page_counts.values()), "page_types": dict(page_counts),
        "accepted_chunks": len(accepted), "accepted_documents": len({row["doc_id"] for row in accepted}),
        "rejected_records": len(audit), "maximum_chunk_characters": maximum,
    }


def _group_matches(text: str, query: dict[str, Any]) -> tuple[list[bool], list[list[str]]]:
    folded = text.casefold()
    matched_terms = [[term for term in group if term.casefold() in folded] for group in query["concept_groups"]]
    return [bool(terms) for terms in matched_terms], matched_terms


def _answer_signal_count(text: str, intent: str) -> int:
    keys = set(intent.split("_"))
    signals = []
    for key in keys:
        signals.extend(ANSWER_SIGNALS.get(key, ()))
    return sum(signal in text for signal in set(signals))


def retrieve_m31(config: dict[str, Any], corpus_path: Path, output: Path) -> list[dict[str, Any]]:
    corpus = pq.read_table(corpus_path).to_pylist()
    result: list[dict[str, Any]] = []
    for query in sorted(config["zh_queries"], key=lambda row: int(row["query_id"])):
        query_id = int(query["query_id"])
        specific_query = " ".join(term for group in query["concept_groups"] for term in group)
        specific_query += " " + " ".join(query.get("context", []))
        scores = _bm25(corpus, specific_query)
        candidates: list[dict[str, Any]] = []
        for row in corpus:
            flags, terms = _group_matches(str(row["evidence_text"]), query)
            if not any(flags):
                continue
            answer_signals = _answer_signal_count(str(row["evidence_text"]), str(query["intent"]))
            context_matches = sum(term.casefold() in str(row["evidence_text"]).casefold() for term in query.get("context", []))
            candidates.append({
                **row, "query_id": query_id, "query_zh": query["query_zh"],
                "concept_groups_matched": sum(flags), "all_concept_groups_matched": all(flags),
                # Group zero is the disease/entity anchor.  Context-only hits
                # (for example 凌晨 without 喷嚏) must never fill the result set.
                "primary_concept_matched": bool(flags and flags[0]),
                "matched_terms": terms, "answer_signal_count": answer_signals,
                "context_matches": context_matches, "lexical_score": float(scores.get(str(row["chunk_id"]), 0.0)),
            })
        for row in candidates:
            row["partial_gate_pass"] = bool(
                row["primary_concept_matched"] and row["answer_signal_count"] > 0
            )
            row["direct_gate_pass"] = bool(
                row["all_concept_groups_matched"] and row["answer_signal_count"] > 0
            )
        candidates = [row for row in candidates if row["partial_gate_pass"]]
        candidates.sort(key=lambda row: (
            -int(row["direct_gate_pass"]), -int(row["all_concept_groups_matched"]),
            -int(row["concept_groups_matched"]),
            -int(row["answer_signal_count"] > 0), -int(row["answer_signal_count"]),
            -int(row["context_matches"]), -float(row["lexical_score"]),
            int(row["doc_id"]), str(row["chunk_id"]),
        ))
        seen: set[int] = set()
        for row in candidates:
            if int(row["doc_id"]) in seen:
                continue
            seen.add(int(row["doc_id"]))
            result.append({**row, "rank": len(seen)})
            if len(seen) == 20:
                break
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(result), output, compression="zstd")
    return result


def load_m3_silver(root: Path) -> list[dict[str, str]]:
    path = root / "artifacts/multilingual_retrieval_m3/review/m3_ai_judgments_import.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 240:
        raise ValueError("expected frozen 240-row M3 silver review")
    for row in rows:
        if row.get("verification_status") != "AI_JUDGED_M3" or row.get("judgment") not in {"0", "1", "2", "UNSURE"}:
            raise ValueError("invalid M3 silver contract")
    return rows


def failure_funnel(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = load_m3_silver(root)
    full_path = root / "artifacts/multilingual_retrieval_m3/review/m3_ai_review_full.csv"
    with full_path.open("r", encoding="utf-8-sig", newline="") as stream:
        full = {row["case_id"]: row for row in csv.DictReader(stream)}
    categories: Counter[str] = Counter()
    by_query: dict[int, Counter[str]] = defaultdict(Counter)
    for label in rows:
        row = full[label["case_id"]]
        if label["judgment"] == "1":
            category = "PARTIAL_ANSWER"
        elif any(value in label["review_notes"] for value in ("Danh mục", "điều hướng", "chỉ mục/danh sách")):
            category = "DIRECTORY_OR_NAVIGATION"
        elif int(row["evidence_chars"]) > 3000:
            category = "EXTRACTION_GRANULARITY"
        else:
            query = next(item for item in config["zh_queries"] if int(item["query_id"]) == int(label["query_id"]))
            flags, _ = _group_matches(row["evidence_text"], query)
            category = "WRONG_TOPIC" if not any(flags) else "TOPIC_ONLY_NOT_ANSWER_BEARING"
        categories[category] += 1
        by_query[int(label["query_id"])][category] += 1
    return {"total": len(rows), "categories": dict(categories), "by_query": {str(key): dict(value) for key, value in sorted(by_query.items())}}


def _load_m31_silver(path: Path, rankings: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lookup = {(int(row["query_id"]), int(row["doc_id"]), str(row["chunk_id"])): str(row["evidence_text"]) for row in rankings}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row.get("verification_status") != M31_STATUS or row.get("judgment") not in {"0", "1", "2", "UNSURE"}:
            raise ValueError("invalid M3.1 silver contract")
        text = lookup.get((int(row["query_id"]), int(row["doc_id"]), str(row["chunk_id"])))
        if text is None:
            raise ValueError("M3.1 silver candidate absent from ranking")
        if row.get("evidence_span") and row["evidence_span"] not in text:
            raise ValueError("M3.1 silver evidence is not verbatim")
    return rows


def run_m31(root: Path) -> dict[str, Any]:
    config_path = root / "configs/m31_dev_multilingual.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("scope") != "development_only" or config.get("held_out_opened") is not False:
        raise ValueError("M3.1 must remain development-only")
    phase29 = root / "submissions/00_READY_TO_UPLOAD/p29_bachmai2679_cap16.zip"
    actual = sha256_file(phase29)
    if actual != config["phase29_sha256"]:
        raise ValueError("frozen Phase29 SHA-256 mismatch")
    out = root / "artifacts/multilingual_retrieval_m31"
    corpus_path = out / "quality_corpus.parquet"
    quality = build_quality_corpus(root, config, corpus_path, out / "quality_rejections.parquet")
    zh = retrieve_m31(config, corpus_path, out / "zh_m31_top20.parquet")
    en = build_english_protected_arm(root, config, out / "en_m31_protected_top20.parquet")
    m3 = pq.read_table(root / "artifacts/multilingual_retrieval_m3/zh_m3_top20.parquet").to_pylist()
    m3_pairs = {(int(row["query_id"]), int(row["doc_id"])) for row in m3}
    m31_pairs = {(int(row["query_id"]), int(row["doc_id"])) for row in zh}
    old_silver = load_m3_silver(root)
    old_positive = {(int(row["query_id"]), int(row["doc_id"])) for row in old_silver if row["judgment"] in {"1", "2"}}
    new_silver = _load_m31_silver(out / "ai_judgments_m31.csv", zh)
    query_ids = {int(row["query_id"]) for row in config["zh_queries"]}
    retrieved_query_ids = {int(row["query_id"]) for row in zh}
    direct_gate_rows = [row for row in zh if row["direct_gate_pass"]]

    m1_path = root / "artifacts/multilingual_retrieval_m1/ai_judgments_import.csv"
    with m1_path.open("r", encoding="utf-8-sig", newline="") as stream:
        m1 = list(csv.DictReader(stream))
    known_en = {(int(row["query_id"]), int(row["doc_id"])) for row in m1 if row["target_language"] == "en" and row["judgment"] in {"1", "2"}}
    en_pairs = {(int(row["query_id"]), int(row["doc_id"])) for row in en}
    m2_path = root / "artifacts/multilingual_retrieval_m2/ai_judgments_m2.csv"
    with m2_path.open("r", encoding="utf-8-sig", newline="") as stream:
        m2 = list(csv.DictReader(stream))
    bridge_direct = {(int(row["query_id"]), int(row["doc_id"])) for row in m2 if row["target_language"] == "en" and row["judgment"] == "2"}

    report = {
        "status": "M3_1_DEVELOPMENT_VERTICAL_SLICE",
        "held_out_opened": False,
        "phase29": {"sha256": actual, "unchanged": True},
        "failure_funnel_m3": failure_funnel(root, config),
        "quality_corpus": quality,
        "comparison": {
            "m3_rows": len(m3), "m31_rows": len(zh),
            "same_query_doc_pairs": len(m3_pairs & m31_pairs),
            "new_query_doc_pairs": len(m31_pairs - m3_pairs),
            "removed_query_doc_pairs": len(m3_pairs - m31_pairs),
            "m3_known_partial_or_direct_total": len(old_positive),
            "m3_known_partial_or_direct_retained": len(old_positive & m31_pairs),
            "top_k_cap": 20,
            "queries_with_admitted_candidates": len(retrieved_query_ids),
            "queries_abstained": len(query_ids - retrieved_query_ids),
            "abstained_query_ids": sorted(query_ids - retrieved_query_ids),
            "direct_gate_rows": len(direct_gate_rows),
        },
        "m31_silver": {
            "rows": len(new_silver), "label_counts": dict(Counter(row["judgment"] for row in new_silver)),
            "direct_query_units": len({int(row["query_id"]) for row in new_silver if row["judgment"] == "2"}),
            "verification_status": M31_STATUS, "official_metric": False,
        },
        "english": {
            "known_baseline_positive_retained": len(known_en & en_pairs),
            "known_baseline_positive_total": len(known_en),
            "corrected_bridge_direct_retained": len(bridge_direct & en_pairs),
            "corrected_bridge_direct_total": len(bridge_direct),
            "final_top_k_checked": True,
        },
        "validation": {
            "current_official_membership": True,
            "canonical_chunk_verbatim": True,
            "quality_decision_traced": True,
            "development_only": True,
            "held_out_opened": False,
        },
        "artifacts": {
            "zh_rankings": str(out / "zh_m31_top20.parquet"),
            "quality_rejections": str(out / "quality_rejections.parquet"),
            "en_rankings": str(out / "en_m31_protected_top20.parquet"),
        },
    }
    _atomic_json(out / "m31_report.json", report)
    return report
