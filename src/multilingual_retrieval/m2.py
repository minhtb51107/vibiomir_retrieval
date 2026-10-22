from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.retrieval.pubmed_lexical import search


AI_STATUS = "AI_JUDGED"
LABELS = {"0", "1", "2", "UNSURE"}
ZH_PLACEHOLDERS = {
    "我要咨询", "我要提问", "找医生", "投诉", "查看更多", "在线咨询", "免费咨询",
    "图文咨询", "电话咨询", "视频咨询", "立即咨询",
}
ZH_BOILERPLATE = (
    "为您推荐", "相关问答", "相关疾病", "相关医生", "热门推荐", "更多精彩内容",
    "扫码下载", "版权所有", "免责声明", "未经授权", "关注微信公众号",
)
ZH_STOP = set("的了是在和与及或有无会可吗呢什么怎么为何一个这种那些进行相关治疗原因症状")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chinese_evidence_quality(text: str) -> tuple[bool, list[str]]:
    normalized = re.sub(r"\s+", "", str(text))
    han = re.findall(r"[\u3400-\u9fff]", normalized)
    reasons: list[str] = []
    if not normalized:
        reasons.append("EMPTY")
    if normalized in ZH_PLACEHOLDERS or any(normalized == value for value in ZH_PLACEHOLDERS):
        reasons.append("PLACEHOLDER_ONLY")
    if len(han) < 40:
        reasons.append("TOO_FEW_HAN_CHARACTERS")
    boilerplate_hits = sum(value in normalized for value in ZH_BOILERPLATE)
    if boilerplate_hits >= 2 and len(han) < 160:
        reasons.append("NAVIGATION_BOILERPLATE_DOMINANT")
    if normalized.count("我要咨询") >= 2 or normalized.count("相关问答") >= 3:
        reasons.append("REPEATED_TEMPLATE")
    if "擅长" in normalized and "找医生" in normalized and len(han) < 220:
        reasons.append("DOCTOR_DIRECTORY")
    if normalized.count("专家答疑") >= 2 or len(re.findall(r"2025-\d{2}-\d{2}", normalized)) >= 3:
        reasons.append("FAQ_LINK_LIST")
    if normalized.count("医院地址") >= 2 or normalized.count("联系电话") >= 2:
        reasons.append("HOSPITAL_DIRECTORY")
    return not reasons, reasons


def load_silver(path: Path, review_path: Path) -> list[dict[str, str]]:
    review = pq.read_table(review_path, columns=["case_id", "split"]).to_pylist()
    development = {str(row["case_id"]) for row in review if row["split"] == "development"}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("AI silver label file is empty")
    seen: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id in seen:
            raise ValueError(f"duplicate AI silver case_id: {case_id}")
        seen.add(case_id)
        if case_id not in development:
            raise ValueError(f"AI silver row is not in development split: {case_id}")
        if row.get("verification_status") != AI_STATUS:
            raise ValueError(f"AI silver status must remain {AI_STATUS}: {case_id}")
        if row.get("judgment") not in LABELS:
            raise ValueError(f"invalid AI silver label: {case_id}")
    return rows


def _bridge_rows(path: Path, query_ids: set[int]) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "query-only" not in str(payload.get("version", "")):
        raise ValueError("English bridge is not marked query-only")
    return {int(row["query_id"]): row for row in payload["rows"] if int(row["query_id"]) in query_ids}


def _compile_fts5(value: str) -> str:
    # Same punctuation-preserving compilation used by the frozen Phase18 run.
    parts = re.split(r'("[^"]*")', value)
    for index in range(0, len(parts), 2):
        part = parts[index]
        part = re.sub(r"\bt\(\d+;\d+\)", lambda m: f'"{m.group(0)}"', part, flags=re.I)
        part = re.sub(r"\b\d+:\d+\b", lambda m: f'"{m.group(0)}"', part)
        part = re.sub(r"\b\d+\.\d+\b", lambda m: f'"{m.group(0)}"', part)
        part = re.sub(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", lambda m: f'"{m.group(0)}"', part)
        part = re.sub(r"\b[A-Za-z0-9]+/[A-Za-z0-9]+\b", lambda m: f'"{m.group(0)}"', part)
        parts[index] = part
    return "".join(parts)


def _load_pubmed_documents(connection: sqlite3.Connection, doc_ids: set[int]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    ids = sorted(doc_ids)
    for offset in range(0, len(ids), 500):
        window = ids[offset:offset + 500]
        placeholders = ",".join("?" for _ in window)
        for row in connection.execute(
            f"SELECT doc_id,official_url,title,abstract FROM documents WHERE doc_id IN ({placeholders})", window
        ):
            output[int(row[0])] = {
                "doc_id": int(row[0]), "canonical_url": str(row[1]), "title": str(row[2]), "abstract": str(row[3])
            }
    return output


def _zh_bigrams(text: str) -> list[str]:
    runs = re.findall(r"[\u3400-\u9fff]+|[A-Za-z0-9]+", str(text).casefold())
    tokens: list[str] = []
    for run in runs:
        if re.fullmatch(r"[a-z0-9]+", run):
            if len(run) >= 2:
                tokens.append(run)
            continue
        clean = "".join(char for char in run if char not in ZH_STOP)
        tokens.extend(clean[i:i + 2] for i in range(max(0, len(clean) - 1)))
    return sorted(set(token for token in tokens if token.strip()))


def _build_zh_fts(index_path: Path, chunks_path: Path) -> dict[str, int]:
    if index_path.exists():
        index_path.unlink()
    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA cache_size=-8192")
    connection.executescript("""
    CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_id INTEGER, raw_text TEXT, source_url TEXT);
    CREATE VIRTUAL TABLE chunks_fts USING fts5(tokens, content='', tokenize='unicode61');
    """)
    inserted = rejected = 0
    columns = ["chunk_id", "doc_id", "raw_text", "source_url"]
    for batch in pq.ParquetFile(chunks_path).iter_batches(columns=columns, batch_size=256):
        with connection:
            for row in batch.to_pylist():
                passed, _ = chinese_evidence_quality(str(row["raw_text"]))
                tokens = _zh_bigrams(str(row["raw_text"]))
                if not passed or not tokens:
                    rejected += 1; continue
                cursor = connection.execute(
                    "INSERT INTO chunks(chunk_id,doc_id,raw_text,source_url) VALUES(?,?,?,?)",
                    (str(row["chunk_id"]), int(row["doc_id"]), str(row["raw_text"]), str(row["source_url"])),
                )
                rowid = int(cursor.lastrowid)
                connection.execute("INSERT INTO chunks_fts(rowid,tokens) VALUES(?,?)", (rowid, " ".join(tokens)))
                inserted += 1
    connection.close()
    return {"inserted": inserted, "rejected": rejected}


def _search_zh(index_path: Path, bridge_path: Path, query_ids: set[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bridge_payload = json.loads(bridge_path.read_text(encoding="utf-8"))
    bridge = {int(row["query_id"]): row for row in bridge_payload["rows"]}
    if set(bridge) != query_ids or "query-only" not in bridge_payload["version"]:
        raise ValueError("Chinese development bridge does not match frozen development queries")
    connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON"); connection.execute("PRAGMA cache_size=-8192")
    output: list[dict[str, Any]] = []
    try:
        for query_id in sorted(query_ids):
            tokens = _zh_bigrams(bridge[query_id]["query_zh"])
            expression = " OR ".join(f'"{token}"' for token in tokens)
            rows = connection.execute(
                "SELECT c.chunk_id,c.doc_id,c.raw_text,c.source_url,bm25(chunks_fts) score "
                "FROM chunks_fts JOIN chunks c ON c.rowid=chunks_fts.rowid WHERE chunks_fts MATCH ? "
                "ORDER BY score,c.doc_id,c.chunk_id LIMIT 100", (expression,),
            ).fetchall()
            seen: set[int] = set()
            for chunk_id, doc_id, raw_text, source_url, score in rows:
                if int(doc_id) in seen:
                    continue
                seen.add(int(doc_id))
                output.append({
                    "query_id": query_id, "target_language": "zh", "doc_id": int(doc_id),
                    "chunk_id": str(chunk_id), "evidence_text": str(raw_text), "canonical_url": str(source_url),
                    "rank": len(seen), "score": float(score), "query_zh_m2": bridge[query_id]["query_zh"],
                })
                if len(seen) == 20:
                    break
    finally:
        connection.close()
    return output, bridge_payload


def run_vertical_slice(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    m1 = root / "artifacts/multilingual_retrieval_m1"
    out = root / "artifacts/multilingual_retrieval_m2"
    out.mkdir(parents=True, exist_ok=True)
    review_path = m1 / "review_queue.parquet"
    silver_path = m1 / "ai_judgments_import.csv"
    silver = load_silver(silver_path, review_path)
    phase29_zip = root / "submissions/00_READY_TO_UPLOAD/p29_bachmai2679_cap16.zip"
    phase29_expected = "e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04"
    phase29_actual = sha256_file(phase29_zip)
    if phase29_actual != phase29_expected:
        raise ValueError("frozen Phase29 BEST SHA-256 mismatch")
    silver_by_case = {row["case_id"]: row for row in silver}
    review = [row for row in pq.read_table(review_path).to_pylist() if row["split"] == "development"]
    review_by_case = {str(row["case_id"]): row for row in review}
    if set(silver_by_case) - set(review_by_case):
        raise ValueError("AI silver contains unknown development cases")

    # ZH: hold the query and cached ranking fixed, change extraction/index eligibility only.
    zh_query_ids = {int(row["query_id"]) for row in review if row["target_language"] == "zh"}
    phase7 = root / "data/retrieval/phase7_reranking/reranked_chunks_depth100.parquet"
    zh_all: dict[int, list[dict[str, Any]]] = defaultdict(list)
    cols = ["query_id", "doc_id", "chunk_id", "chunk_text", "source_url", "rerank_rank", "rerank_score"]
    for batch in pq.ParquetFile(phase7).iter_batches(columns=cols, batch_size=2048):
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            if query_id not in zh_query_ids:
                continue
            passed, reasons = chinese_evidence_quality(str(row["chunk_text"]))
            zh_all[query_id].append({
                "query_id": query_id, "target_language": "zh", "doc_id": int(row["doc_id"]),
                "chunk_id": str(row["chunk_id"]), "evidence_text": str(row["chunk_text"]),
                "canonical_url": str(row["source_url"]), "original_rank": int(row["rerank_rank"]),
                "score": float(row["rerank_score"]), "quality_pass": passed,
                "quality_reasons": reasons,
            })
    zh_rows: list[dict[str, Any]] = []
    for query_id in sorted(zh_query_ids):
        seen_docs: set[int] = set(); rank = 0
        for row in sorted(zh_all[query_id], key=lambda x: (x["original_rank"], x["doc_id"], x["chunk_id"])):
            if not row["quality_pass"] or row["doc_id"] in seen_docs:
                continue
            seen_docs.add(row["doc_id"]); rank += 1
            zh_rows.append({**row, "m2_rank": rank, "promoted_from_below_20": row["original_rank"] > 20})
            if rank == 20:
                break
    pq.write_table(pa.Table.from_pylist(zh_rows), out / "zh_quality_filtered_top20.parquet")

    zh_labeled = []
    for case_id, label in silver_by_case.items():
        item = review_by_case[case_id]
        if item["target_language"] != "zh":
            continue
        passed, reasons = chinese_evidence_quality(str(item["evidence_text"]))
        zh_labeled.append({"case_id": case_id, "judgment": label["judgment"], "quality_pass": passed, "reasons": reasons})

    # EN: replace only the Helsinki query representation with the frozen query-only medical bridge.
    en_query_ids = {int(row["query_id"]) for row in review if row["target_language"] == "en"}
    bridge_path = root / "configs/phase18_pubmed_english_bridge_60.json"
    bridge = _bridge_rows(bridge_path, en_query_ids)
    missing_bridge = sorted(en_query_ids - bridge.keys())
    db_path = root / "data/system_upgrade_m2a/pubmed_lexical.sqlite"
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON"); connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-8192"); connection.execute("PRAGMA mmap_size=0")
    en_hits: list[dict[str, Any]] = []
    try:
        for query_id in sorted(bridge):
            item = bridge[query_id]
            if item["status"] == "AMBIGUOUS":
                continue
            compiled = _compile_fts5(str(item["english_query"]))
            for hit in search(connection, compiled, limit=20):
                en_hits.append({
                    "query_id": query_id, "target_language": "en", "doc_id": int(hit["doc_id"]),
                    "rank": int(hit["rank"]), "score": float(hit["bm25_score"]),
                    "query_en_m2": str(item["english_query"]), "fts5_query": compiled,
                    "canonical_url": str(hit["official_url"]), "title": str(hit["title"]),
                })
        documents = _load_pubmed_documents(connection, {row["doc_id"] for row in en_hits})
    finally:
        connection.close()
    for row in en_hits:
        doc = documents[row["doc_id"]]
        row["evidence_text"] = (doc["title"] + "\n\n" + doc["abstract"]).strip()
    pq.write_table(pa.Table.from_pylist(en_hits), out / "en_medical_bridge_fts5_top20.parquet")

    baseline = pq.read_table(m1 / "cached_retrieval_top20.parquet").to_pylist()
    baseline_en_rows = [
        r for r in baseline
        if r["target_language"] == "en" and r["arm"] == "lexical"
        and int(r["query_id"]) in en_query_ids and int(r["rank"]) <= 20
    ]
    baseline_en = {(int(r["query_id"]), int(r["doc_id"])) for r in baseline_en_rows}
    repaired_en = {(int(r["query_id"]), int(r["doc_id"])) for r in en_hits}
    # RRF preserves the original Phase19 lexical arm while allowing the medical bridge to add evidence.
    en_union_rows: list[dict[str, Any]] = []
    for query_id in sorted(en_query_ids):
        scores: dict[int, float] = defaultdict(float); metadata: dict[int, dict[str, Any]] = {}
        for row in baseline_en_rows:
            if int(row["query_id"]) != query_id: continue
            doc_id = int(row["doc_id"]); scores[doc_id] += 1.0 / (60 + int(row["rank"])); metadata[doc_id] = row
        for row in en_hits:
            if int(row["query_id"]) != query_id: continue
            doc_id = int(row["doc_id"]); scores[doc_id] += 1.0 / (60 + int(row["rank"])); metadata[doc_id] = row
        for rank, doc_id in enumerate(sorted(scores, key=lambda value: (-scores[value], value))[:20], 1):
            en_union_rows.append({"query_id": query_id, "doc_id": doc_id, "rank": rank, "rrf_score": scores[doc_id]})
    pq.write_table(pa.Table.from_pylist(en_union_rows), out / "en_baseline_plus_medical_bridge_top20.parquet")
    union_en = {(int(r["query_id"]), int(r["doc_id"])) for r in en_union_rows}
    en_known = []
    for case_id, label in silver_by_case.items():
        item = review_by_case[case_id]
        if item["target_language"] != "en" or label["judgment"] == "UNSURE":
            continue
        pair = (int(item["query_id"]), int(item["doc_id"]))
        en_known.append({
            "case_id": case_id, "query_id": pair[0], "doc_id": pair[1], "judgment": label["judgment"],
            "baseline_retrieved": pair in baseline_en, "m2_retrieved": pair in repaired_en,
            "m2_union_retrieved": pair in union_en,
        })

    # A real Chinese lexical vertical slice: quality-filtered char-bigram FTS over the existing canonical chunks.
    zh_bridge_path = root / "configs/m2_dev_chinese_bridge.json"
    zh_index_path = out / "zh_quality_char_bigram.sqlite"
    zh_index = _build_zh_fts(zh_index_path, root / "data/chunks/phase4_bge_m3/tokens_512_overlap_64.parquet")
    zh_lexical, zh_bridge_payload = _search_zh(zh_index_path, zh_bridge_path, zh_query_ids)
    pq.write_table(pa.Table.from_pylist(zh_lexical), out / "zh_medical_bridge_fts_top20.parquet")

    # A small, separately versioned AI-silver review of the new top-1 results.
    # It is never merged into human_labels.csv and is used only as a development diagnostic.
    m2_silver_path = out / "ai_judgments_m2.csv"
    m2_silver: list[dict[str, str]] = []
    if m2_silver_path.exists():
        with m2_silver_path.open("r", encoding="utf-8-sig", newline="") as stream:
            m2_silver = list(csv.DictReader(stream))
        candidate_text = {
            (str(r["target_language"]), int(r["query_id"]), int(r["doc_id"])): str(r["evidence_text"])
            for r in en_hits + zh_lexical
        }
        for row in m2_silver:
            if row.get("verification_status") != "AI_JUDGED_M2" or row.get("judgment") not in LABELS:
                raise ValueError("invalid M2 AI-silver contract")
            key = (row["target_language"], int(row["query_id"]), int(row["doc_id"]))
            if key not in candidate_text:
                raise ValueError(f"M2 AI-silver candidate missing: {key}")
            if row.get("evidence_span") and row["evidence_span"] not in candidate_text[key]:
                raise ValueError(f"M2 AI-silver evidence is not verbatim: {key}")

    promoted_zh = [row for row in zh_rows if row["promoted_from_below_20"]]
    baseline_en_by_query = Counter(q for q, _ in baseline_en)
    repaired_en_by_query = Counter(q for q, _ in repaired_en)
    def retained(rows: list[dict[str, Any]], labels: set[str], field: str) -> int:
        return sum(row["judgment"] in labels and bool(row[field]) for row in rows)

    report = {
        "status": "M2_DEVELOPMENT_VERTICAL_SLICE_COMPLETE",
        "scope": "development_only",
        "held_out_opened": False,
        "frozen_phase29": {"path": str(phase29_zip), "sha256": phase29_actual, "unchanged": True},
        "silver_contract": {
            "path": str(silver_path), "sha256": sha256_file(silver_path), "rows": len(silver),
            "verification_status": AI_STATUS, "human_labels_modified": False,
            "limitations": "AI silver is review aid, not HUMAN_VERIFIED or official relevance ground truth.",
        },
        "m2_ai_silver": {
            "path": str(m2_silver_path), "rows": len(m2_silver),
            "label_counts": dict(Counter(row["judgment"] for row in m2_silver)),
            "verification_status": "AI_JUDGED_M2", "official_metric": False,
        },
        "english_translation_repair": {
            "bridge_path": str(bridge_path), "bridge_sha256": sha256_file(bridge_path),
            "development_queries": len(en_query_ids), "bridge_queries": len(bridge), "missing_bridge_queries": missing_bridge,
            "baseline_query_doc_pairs_top20": len(baseline_en), "m2_query_doc_pairs_top20": len(repaired_en),
            "overlap_pairs": len(baseline_en & repaired_en), "new_pairs": len(repaired_en - baseline_en),
            "known_direct_label2_total": sum(r["judgment"] == "2" for r in en_known),
            "known_direct_label2_baseline_retrieved": retained(en_known, {"2"}, "baseline_retrieved"),
            "known_direct_label2_m2_retrieved": retained(en_known, {"2"}, "m2_retrieved"),
            "known_direct_label2_union_retrieved": retained(en_known, {"2"}, "m2_union_retrieved"),
            "known_positive_label1_or2_total": sum(r["judgment"] in {"1", "2"} for r in en_known),
            "known_positive_baseline_retrieved": retained(en_known, {"1", "2"}, "baseline_retrieved"),
            "known_positive_m2_retrieved": retained(en_known, {"1", "2"}, "m2_retrieved"),
            "known_positive_union_retrieved": retained(en_known, {"1", "2"}, "m2_union_retrieved"),
            "note": "Known-label comparison is pooled/incomplete and is not recall over all relevant PubMed documents.",
        },
        "chinese_extraction_repair": {
            "development_queries": len(zh_query_ids), "cached_rows_examined": sum(len(v) for v in zh_all.values()),
            "top20_rows_after_filter": len(zh_rows), "promoted_rows_from_below_old_top20": len(promoted_zh),
            "labeled_rows": len(zh_labeled), "labeled_rows_rejected": sum(not r["quality_pass"] for r in zh_labeled),
            "rejected_by_silver_label": dict(Counter(r["judgment"] for r in zh_labeled if not r["quality_pass"])),
            "rejection_reasons": dict(Counter(reason for r in zh_labeled for reason in r["reasons"])),
            "note": "This measures extraction/index eligibility. Promoted rows remain unjudged until separately reviewed.",
            "char_bigram_index": zh_index,
            "query_bridge_path": str(zh_bridge_path),
            "query_bridge_sha256": sha256_file(zh_bridge_path),
            "lexical_queries": len({row["query_id"] for row in zh_lexical}),
            "lexical_query_doc_pairs_top20": len(zh_lexical),
        },
        "artifacts": {
            "english_top20": str(out / "en_medical_bridge_fts5_top20.parquet"),
            "chinese_top20": str(out / "zh_quality_filtered_top20.parquet"),
            "chinese_lexical_top20": str(out / "zh_medical_bridge_fts_top20.parquet"),
            "english_union_top20": str(out / "en_baseline_plus_medical_bridge_top20.parquet"),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    (out / "m2_vertical_slice_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
