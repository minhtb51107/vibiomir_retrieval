from __future__ import annotations

import math
import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.storage.body_archive import BodyArchive

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]")
VI_RE = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯ]|[àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)
MEDICAL_TERMS = (
    "病", "医生", "治疗", "症状", "患者", "药", "健康", "检查", "诊断", "医学",
    "bệnh", "bác sĩ", "điều trị", "triệu chứng", "thuốc", "sức khỏe", "xét nghiệm",
    "chẩn đoán", "bệnh nhân", "y tế", "disease", "doctor", "treatment", "symptom", "health",
)


def content_signals(text: str) -> dict[str, int]:
    folded = text.casefold()
    return {
        "han": len(HAN_RE.findall(text)),
        "latin": len(LATIN_RE.findall(text)),
        "vietnamese_specific": len(VI_RE.findall(text)),
        "medical_terms": sum(term in folded for term in MEDICAL_TERMS),
    }


def language_label(han: int, latin: int, vietnamese_specific: int) -> str:
    denominator = max(han + latin, 1)
    if han / denominator >= 0.55:
        return "mostly Chinese"
    if latin / denominator >= 0.55 and vietnamese_specific > 0:
        return "mostly Vietnamese"
    return "mixed/unknown"


def early_stop_decision(*, access_rate: float, usable_rate: float, unsafe_antibot: bool) -> tuple[bool, str]:
    if access_rate > 0.80:
        return True, "robots/access restriction above 80%"
    if usable_rate < 0.20:
        return True, "usable extraction below 20%"
    if unsafe_antibot:
        return True, "repeated anti-bot behavior unsafe"
    return False, "checkpoint passed"


def tier_for(*, usable_rate: float, access_rate: float, projected_50k_fits: bool, throughput_acceptable: bool) -> str:
    if access_rate >= 0.80:
        return "BLOCKED"
    if usable_rate >= 0.80 and projected_50k_fits and throughput_acceptable:
        return "A"
    if usable_rate >= 0.50:
        return "B"
    return "C"


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def analyze_source(
    *, domain: str, corpus_rows: int, corpus_share: float, root: str | Path,
    stage: str, concurrency: int, delay_seconds: float,
) -> dict[str, Any]:
    root = Path(root)
    connection = sqlite3.connect(root / "crawl.sqlite")
    connection.row_factory = sqlite3.Row
    crawl_rows = list(connection.execute("SELECT * FROM crawl_results ORDER BY doc_id"))
    connection.close()
    status_counts = Counter(str(row["status"]) for row in crawl_rows)
    attempted = len(crawl_rows)
    successes = status_counts["SUCCESS"]
    access = status_counts["ROBOTS_BLOCKED"] + status_counts["ACCESS_RESTRICTED"]
    timeouts = status_counts["TIMEOUT"] + status_counts["NETWORK_ERROR"] + status_counts["RETRY_EXHAUSTED"]
    elapsed_seconds = sum(int(row["elapsed_ms"]) for row in crawl_rows) / 1000
    average_latency = elapsed_seconds / max(sum(int(row["attempt_count"]) for row in crawl_rows), 1)
    paced_seconds = delay_seconds * max(successes - 1, 0) + average_latency
    modeled_crawl_seconds = max(elapsed_seconds / max(concurrency, 1), paced_seconds)
    observed_new_rows = 0
    observed_new_wall_seconds = 0.0
    for summary_path in sorted(root.glob("crawl_*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        observed_new_rows += int(summary["run"]["completed"])
        observed_new_wall_seconds += float(summary["run"]["wall_seconds"])
    reused_rows = max(attempted - observed_new_rows, 0)
    if observed_new_rows:
        modeled_crawl_seconds = (
            observed_new_wall_seconds
            + modeled_crawl_seconds / max(attempted, 1) * reused_rows
        )

    archive_original = archive_compressed = archive_count = 0
    with BodyArchive(root / "bodies") as archive:
        for row in archive.iter_metadata():
            archive_count += 1
            archive_original += int(row["original_length"])
            archive_compressed += int(row["compressed_length"])

    stage_root = root / stage
    documents_path = stage_root / "documents.parquet"
    chunks_path = stage_root / "chunks.parquet"
    documents = pq.read_table(documents_path).to_pylist()
    chunks = pq.read_table(chunks_path, columns=["doc_id", "token_count"]).to_pylist()
    extraction_summary = json.loads((stage_root / "extraction_summary.json").read_text(encoding="utf-8"))
    chunk_summary = json.loads((stage_root / "chunk_summary.json").read_text(encoding="utf-8"))
    doc_chunk_counts: Counter[int] = Counter(int(row["doc_id"]) for row in chunks)
    usable_ids = set()
    language_totals = Counter()
    exact_text = Counter()
    empty_or_near = 0
    navigation_only = 0
    title_available = 0
    qa_structured = 0
    clean_lengths = []
    for row in documents:
        text = str(row.get("normalized_text") or "")
        signals = content_signals(text)
        language_totals.update({key: signals[key] for key in ("han", "latin", "vietnamese_specific")})
        success = row["extraction_status"] == "SUCCESS"
        if len(text.strip()) < 100:
            empty_or_near += 1
        if success and (int(row.get("paragraph_count") or 0) < 2 or int(row.get("clean_char_count") or 0) < 500):
            navigation_only += 1
        if row.get("title"):
            title_available += 1
        if "qa_structured" in str(row.get("extraction_method") or ""):
            qa_structured += 1
        if text:
            exact_text[text] += 1
        if success:
            clean_lengths.append(int(row.get("clean_char_count") or 0))
        if (
            success and int(row.get("clean_char_count") or 0) >= 500
            and int(row.get("paragraph_count") or 0) >= 2 and signals["medical_terms"] >= 1
        ):
            usable_ids.add(int(row["doc_id"]))
    usable = len(usable_ids)
    usable_rate = usable / max(attempted, 1)
    access_rate = access / max(attempted, 1)
    stop, stop_reason = early_stop_decision(access_rate=access_rate, usable_rate=usable_rate, unsafe_antibot=False)
    usable_chunk_counts = [doc_chunk_counts[doc_id] for doc_id in usable_ids]
    retained_bytes = (
        (root / "crawl.sqlite").stat().st_size + sum(path.stat().st_size for path in (root / "bodies").glob("*"))
        + documents_path.stat().st_size + chunks_path.stat().st_size
    )
    bytes_per_attempt = retained_bytes / max(attempted, 1)
    seconds_per_attempt = (
        modeled_crawl_seconds + float(extraction_summary["runtime_seconds"]) + float(chunk_summary["runtime_seconds"])
    ) / max(attempted, 1)
    def projection(count: int) -> dict[str, float]:
        return {
            "retained_bytes": round(bytes_per_attempt * count),
            "raw_transfer_bytes": round(archive_original / max(attempted, 1) * count),
            "pipeline_seconds": round(seconds_per_attempt * count, 2),
        }
    return {
        "domain": domain, "corpus_rows": corpus_rows, "corpus_share": corpus_share,
        "stage": stage, "attempted": attempted, "status_counts": dict(sorted(status_counts.items())),
        "fetch_successes": successes, "fetch_success_rate": round(successes / max(attempted, 1), 6),
        "robots_access_count": access, "robots_access_rate": round(access_rate, 6),
        "http_failure_count": status_counts["HTTP_ERROR"],
        "timeout_network_count": timeouts,
        "extraction_successes": sum(row["extraction_status"] == "SUCCESS" for row in documents),
        "usable_documents": usable, "usable_doc_rate": round(usable_rate, 6),
        "empty_or_near_empty": empty_or_near, "navigation_or_short_count": navigation_only,
        "duplicate_text_rows_beyond_first": sum(max(count - 1, 0) for count in exact_text.values()),
        "title_available": title_available, "qa_structured": qa_structured,
        "language_signal": language_label(language_totals["han"], language_totals["latin"], language_totals["vietnamese_specific"]),
        "language_character_counts": dict(language_totals),
        "archive": {
            "bodies": archive_count, "raw_bytes": archive_original, "compressed_bytes": archive_compressed,
            "average_raw_bytes_per_body": round(archive_original / max(archive_count, 1), 2),
            "average_compressed_bytes_per_body": round(archive_compressed / max(archive_count, 1), 2),
        },
        "average_clean_characters": round(statistics.mean(clean_lengths), 2) if clean_lengths else 0,
        "chunks": {
            "total": len(chunks),
            "usable_documents_with_chunks": sum(value > 0 for value in usable_chunk_counts),
            "per_usable_mean": round(statistics.mean(usable_chunk_counts), 4) if usable_chunk_counts else 0,
            "per_usable_median": round(statistics.median(usable_chunk_counts), 4) if usable_chunk_counts else 0,
            "per_usable_p95": round(_percentile([float(value) for value in usable_chunk_counts], 0.95), 4),
            "projected_per_10000_usable_docs": round(statistics.mean(usable_chunk_counts) * 10000) if usable_chunk_counts else 0,
        },
        "performance": {
            "crawl_time_basis": "observed wall time for new source-specific runs plus modeled Phase 8 reuse time from measured elapsed_ms and configured concurrency/delay",
            "observed_new_rows": observed_new_rows,
            "observed_new_wall_seconds": round(observed_new_wall_seconds, 3),
            "reused_phase8_rows": reused_rows,
            "modeled_crawl_seconds": round(modeled_crawl_seconds, 3),
            "urls_per_minute": round(attempted / max(modeled_crawl_seconds, 0.001) * 60, 3),
            "extraction_seconds": extraction_summary["runtime_seconds"],
            "extraction_docs_per_second": round(attempted / max(float(extraction_summary["runtime_seconds"]), 0.001), 3),
            "chunking_seconds": chunk_summary["runtime_seconds"],
            "usable_docs_per_minute": round(usable / max(modeled_crawl_seconds, 0.001) * 60, 3),
        },
        "storage": {
            "retained_measured_bytes": retained_bytes,
            "mib_per_1000_usable_docs": round(retained_bytes / max(usable, 1) * 1000 / 1048576, 3),
        },
        "projections": {
            "10k": projection(10_000), "50k": projection(50_000), "full_source": projection(corpus_rows),
        },
        "early_stop": stop, "early_stop_reason": stop_reason,
    }
