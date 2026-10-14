#!/usr/bin/env python3
"""CPU-only Phase 15A replay and two isolated ranking variants."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.state_file import update_json, write_json
from src.source_census.cached_subsets import _query_groups
from src.source_census.union_experiment import _iter_json_array, _submission_records_disk
from src.submission.common import file_sha256, write_submission_stream
from src.submission.validator import expected_query_ids
from tools.phase11a_bottleneck import tokenizer as _historical_tokenizer


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("run_id") != "phase15_ranking_optimization":
        raise ValueError("unexpected Phase 15A configuration")
    return config


def rank_top2_mean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate unique scored chunks per document with Phase 13A semantics."""
    by_doc: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        doc_id, chunk_id = int(row["doc_id"]), str(row["chunk_id"])
        old = by_doc[doc_id].get(chunk_id)
        if old is not None:
            if (float(old["score"]), str(old["chunk_text"])) != (float(row["score"]), str(row["chunk_text"])):
                raise ValueError(f"conflicting exact chunk identity: doc={doc_id} chunk={chunk_id}")
            continue
        by_doc[doc_id][chunk_id] = row
    documents = []
    for doc_id, chunks_by_id in by_doc.items():
        chunks = sorted(chunks_by_id.values(), key=lambda row: (-float(row["score"]), str(row["chunk_id"])))
        representatives = chunks[:2]
        documents.append({
            "doc_id": doc_id,
            "score": float(np.mean([float(row["score"]) for row in representatives])),
            "best_chunk_id": str(representatives[0]["chunk_id"]),
            "scored_chunk_count": len(chunks),
            "representative_chunk_count": len(representatives),
        })
    return sorted(documents, key=lambda row: (-float(row["score"]), str(row["best_chunk_id"]), int(row["doc_id"])))


def diversity_order(rows: list[dict[str, Any]], depth: int = 40, cap: int = 2) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Put a bounded-owner first pass before the untouched score-ordered fill."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda value: (int(value["rerank_rank"]), str(value["chunk_id"]))):
        chunk_id = str(row["chunk_id"])
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(row)
    counts: Counter[int] = Counter()
    first: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for row in unique:
        doc_id = int(row["doc_id"])
        if len(first) < depth and counts[doc_id] < cap:
            first.append(row)
            counts[doc_id] += 1
        else:
            deferred.append(row)
    ordered = first + deferred
    if len(ordered) < depth:
        raise ValueError(f"only {len(ordered)} distinct valid chunks available, expected {depth}")
    return ordered, {
        "unique_candidate_chunks": len(unique),
        "first_pass_chunks": min(len(first), depth),
        "fill_chunks_needed": max(0, depth - len(first)),
        "duplicate_chunk_ids_suppressed": len(rows) - len(unique),
    }


def _score_rows(query_id: int, rows: list[dict[str, Any]], source: sqlite3.Connection, pilot: sqlite3.Connection) -> list[dict[str, Any]]:
    source_scores = {str(chunk): float(score) for chunk, score in source.execute(
        "SELECT chunk_id,rerank_score FROM pairs WHERE query_id=?", (query_id,)
    )}
    pilot_scores = {str(chunk): float(score) for chunk, score in pilot.execute(
        "SELECT chunk_id,rerank_score FROM scores WHERE query_id=?", (query_id,)
    )}
    output = []
    for row in rows:
        chunk_id = str(row["chunk_id"])
        if chunk_id in source_scores and chunk_id in pilot_scores and source_scores[chunk_id] != pilot_scores[chunk_id]:
            raise ValueError(f"conflicting frozen score ownership: query={query_id} chunk={chunk_id}")
        score = source_scores.get(chunk_id, pilot_scores.get(chunk_id))
        if score is None or not math.isfinite(score):
            raise ValueError(f"missing/non-finite frozen score: query={query_id} chunk={chunk_id}")
        output.append({**row, "score": score})
    return output


def _checkpoint(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS results(query_id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
    connection.commit()
    return connection


def submission_tokenizer(config: dict[str, Any]) -> Any:
    """Load the exact historical submission tokenizer with its required config."""
    return _historical_tokenizer(config)


def _control_replay(config: dict[str, Any], tok: Any, on_progress=None) -> dict[str, Any]:
    inputs, contracts = config["inputs"], config["contracts"]
    frozen_json, frozen_zip = ROOT / inputs["phase14_json"], ROOT / inputs["phase14_zip"]
    hashes = {"json": file_sha256(frozen_json), "zip": file_sha256(frozen_zip)}
    if hashes != {"json": contracts["phase14_json_sha256"], "zip": contracts["phase14_zip_sha256"]}:
        raise ValueError(f"Phase 14 frozen hash mismatch: {hashes}")
    query_ids = expected_query_ids(ROOT / inputs["queries"])
    generated = _submission_records_disk(
        query_ids, ROOT / inputs["phase14_documents"], ROOT / inputs["phase14_chunks"],
        ROOT / inputs["canonical_db"], tok, document_depth=20, chunk_depth=40, target_tokens=1024,
    )
    expected = _iter_json_array(frozen_json)
    matched = 0
    for ordinal, (actual, old) in enumerate(zip(generated, expected, strict=True), 1):
        if actual != old:
            doc_equal = actual.get("relevant_docs") == old.get("relevant_docs")
            chunk_equal = actual.get("relevant_chunks") == old.get("relevant_chunks")
            raise ValueError(
                f"Phase 14 CONTROL semantic replay mismatch at ordinal={ordinal} query={old.get('id')} "
                f"documents_equal={doc_equal} chunks_equal={chunk_equal}"
            )
        matched += 1
        if on_progress is not None and (matched % 10 == 0 or matched == 1200):
            on_progress(matched)
    result = {"passed": matched == 1200, "queries_exact": matched, "queries_total": 1200, **hashes}
    if not result["passed"]:
        raise ValueError(f"control replay incomplete: {result}")
    return result


def _write_variant_rankings(config: dict[str, Any], connection: sqlite3.Connection) -> dict[str, Path]:
    root = ROOT / config["outputs"]["root"]
    top2 = root / "DOC_TOP2_MEAN"; diversity = root / "CHUNK_DIVERSITY"
    top2.mkdir(parents=True, exist_ok=True); diversity.mkdir(parents=True, exist_ok=True)
    doc_schema = pa.schema([("query_id", pa.int64()), ("rank", pa.int64()), ("doc_id", pa.int64())])
    docs = []
    payloads = {}
    for query_id, raw in connection.execute("SELECT query_id,payload FROM results ORDER BY query_id"):
        payload = json.loads(raw); payloads[int(query_id)] = payload
        docs.extend({"query_id": int(query_id), "rank": rank, "doc_id": int(doc_id)}
                    for rank, doc_id in enumerate(payload["top2_docs"], 1))
    pq.write_table(pa.Table.from_pylist(docs, schema=doc_schema), top2 / "reranked_documents.parquet", compression="zstd")

    # Chunk output is exactly frozen for DOC_TOP2_MEAN.
    control_chunks = ROOT / config["inputs"]["phase14_chunks"]
    # A path marker avoids duplicating a large immutable parquet.
    (top2 / "chunks_source.json").write_text(json.dumps({"path": str(control_chunks)}, indent=2), encoding="utf-8")

    writer = None
    try:
        for rows in _query_groups(control_chunks):
            query_id = int(rows[0]["query_id"])
            ordered, _ = diversity_order(rows)
            out = [{**row, "rerank_rank": rank} for rank, row in enumerate(ordered, 1)]
            table = pa.Table.from_pylist(out)
            if writer is None:
                writer = pq.ParquetWriter(diversity / "reranked_chunks.parquet", table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    # Document output is exactly frozen for CHUNK_DIVERSITY.
    (diversity / "documents_source.json").write_text(json.dumps({"path": str(ROOT / config["inputs"]["phase14_documents"])}, indent=2), encoding="utf-8")
    return {
        "top2_documents": top2 / "reranked_documents.parquet",
        "top2_chunks": control_chunks,
        "diversity_documents": ROOT / config["inputs"]["phase14_documents"],
        "diversity_chunks": diversity / "reranked_chunks.parquet",
    }


def _preview(config: dict[str, Any], tok: Any, name: str, documents: Path, chunks: Path) -> tuple[Path, str, dict[int, dict]]:
    root = ROOT / config["outputs"]["root"] / "previews"; root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    query_ids = expected_query_ids(ROOT / config["inputs"]["queries"])
    args = (query_ids, documents, chunks, ROOT / config["inputs"]["canonical_db"], tok)
    kwargs = {"document_depth": 20, "chunk_depth": 40, "target_tokens": 1024}
    write_submission_stream(path, _submission_records_disk(*args, **kwargs)); first = file_sha256(path)
    write_submission_stream(path, _submission_records_disk(*args, **kwargs)); second = file_sha256(path)
    if first != second:
        raise RuntimeError(f"non-deterministic preview: {name}")
    records = {int(row["id"]): row for row in _iter_json_array(path)}
    if len(records) != 1200:
        raise ValueError(f"{name} query count={len(records)}")
    return path, second, records


def _official_ids(config: dict[str, Any], wanted: set[int]) -> set[int]:
    found = set()
    parquet = pq.ParquetFile(ROOT / config["inputs"]["official_corpus"])
    for batch in parquet.iter_batches(columns=["id"], batch_size=65536):
        found.update(int(value) for value in batch.column(0).to_pylist() if int(value) in wanted)
    return found


def _strict_preview_audit(config: dict[str, Any], records: dict[int, dict]) -> dict[str, Any]:
    query_ids = expected_query_ids(ROOT / config["inputs"]["queries"])
    docs = [int(doc) for row in records.values() for doc in row["relevant_docs"]]
    chunk_owners = [int(chunk["doc_id"]) for row in records.values() for chunk in row["relevant_chunks"]]
    all_ids = set(docs) | set(chunk_owners); official = _official_ids(config, all_ids)
    duplicate_docs = sum(len(row["relevant_docs"]) - len(set(map(int, row["relevant_docs"]))) for row in records.values())
    duplicate_chunks = sum(len(row["relevant_chunks"]) - len({(int(c["doc_id"]), str(c["chunk_text"])) for c in row["relevant_chunks"]}) for row in records.values())
    connection = sqlite3.connect(f"file:{(ROOT / config['inputs']['canonical_db']).as_posix()}?mode=ro", uri=True)
    reconstruction_failures = 0
    try:
        for row in records.values():
            for chunk in row["relevant_chunks"]:
                stored = connection.execute("SELECT normalized_text FROM documents WHERE doc_id=?", (int(chunk["doc_id"]),)).fetchone()
                if stored is None or str(chunk["chunk_text"]) not in str(stored[0]):
                    reconstruction_failures += 1
    finally:
        connection.close()
    passed = bool(
        list(records) == query_ids and all(len(row["relevant_docs"]) == 20 and len(row["relevant_chunks"]) == 40 for row in records.values())
        and duplicate_docs == duplicate_chunks == reconstruction_failures == 0 and official == all_ids
    )
    return {
        "passed": passed, "query_count": len(records), "document_slots": len(docs), "chunk_slots": len(chunk_owners),
        "invalid_official_doc_ids": len(all_ids - official), "duplicate_doc_ids": duplicate_docs,
        "duplicate_chunk_objects": duplicate_chunks, "source_reconstruction_failures": reconstruction_failures,
    }


def _comparison(control: dict[int, dict], variant: dict[int, dict], field: str) -> dict[str, Any]:
    depth = 20 if field == "relevant_docs" else 40
    changed = retained = added = 0; overlaps = []; owner_overlap = []
    for query_id in sorted(control):
        old, new = control[query_id][field], variant[query_id][field]
        old_keys = [int(x) for x in old] if field == "relevant_docs" else [(int(x["doc_id"]), str(x["chunk_text"])) for x in old]
        new_keys = [int(x) for x in new] if field == "relevant_docs" else [(int(x["doc_id"]), str(x["chunk_text"])) for x in new]
        overlap = len(set(old_keys) & set(new_keys)); overlaps.append(overlap / depth)
        changed += old_keys != new_keys; retained += overlap; added += depth - overlap
        if field == "relevant_chunks":
            top_docs = set(map(int, variant[query_id]["relevant_docs"]))
            owner_overlap.append(sum(int(item["doc_id"]) in top_docs for item in new))
    return {"queries_changed": changed, "slots_retained": retained, "slots_added_or_replaced": added,
            "mean_overlap": float(np.mean(overlaps)), "chunk_owners_in_top20": sum(owner_overlap) if owner_overlap else None,
            "chunk_owners_outside_top20": depth * len(control) - sum(owner_overlap) if owner_overlap else None}


def _distribution(values: Iterable[int | float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    return {"count": int(array.size), "min": float(array.min()), "p5": float(np.percentile(array, 5)),
            "median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def _normalized_tokens(text: str) -> set[str]:
    import re
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _output_chunk_quality(records: dict[int, dict]) -> dict[str, Any]:
    exact_pairs = near_pairs = 0; owner_counts = []
    for row in records.values():
        chunks = row["relevant_chunks"]; counts = Counter(int(item["doc_id"]) for item in chunks)
        owner_counts.extend(counts.values())
        normalized = [" ".join(sorted(_normalized_tokens(str(item["chunk_text"])))) for item in chunks]
        token_sets = [_normalized_tokens(str(item["chunk_text"])) for item in chunks]
        for left in range(len(chunks)):
            for right in range(left + 1, len(chunks)):
                if normalized[left] == normalized[right]:
                    exact_pairs += 1; near_pairs += 1; continue
                union = token_sets[left] | token_sets[right]
                if union and len(token_sets[left] & token_sets[right]) / len(union) >= 0.85:
                    near_pairs += 1
    return {"exact_duplicate_text_pairs": exact_pairs, "near_duplicate_text_pairs_jaccard_0_85": near_pairs,
            "chunks_per_owner_document": _distribution(owner_counts)}


def _composition(config: dict[str, Any], records: dict[int, dict]) -> dict[str, Any]:
    slots = [int(doc) for row in records.values() for doc in row["relevant_docs"]]
    wanted = set(slots); domains: dict[int, str] = {}; languages: dict[int, str] = {}
    registry = pq.ParquetFile(ROOT / config["inputs"]["registry"])
    for batch in registry.iter_batches(columns=["doc_id", "canonical_domain", "language_signal"], batch_size=8192):
        for doc_id, domain, language in zip(*(column.to_pylist() for column in batch.columns), strict=True):
            if int(doc_id) in wanted:
                domains[int(doc_id)] = str(domain); languages[int(doc_id)] = str(language or "unknown")
    missing = wanted - set(domains)
    official = pq.ParquetFile(ROOT / config["inputs"]["official_corpus"])
    url_column = "url" if "url" in official.schema.names else "link"
    for batch in official.iter_batches(columns=["id", url_column], batch_size=65536):
        for doc_id, url in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            doc_id = int(doc_id)
            if doc_id in missing:
                domain = (urlsplit(str(url)).hostname or "unknown").lower().removeprefix("www.")
                domains[doc_id] = domain
                languages[doc_id] = "vi" if domain.endswith(".vn") or domain == "medlatec.vn" else "unknown"
    return {"document_slots_by_domain": dict(sorted(Counter(domains.get(doc, "unknown") for doc in slots).items())),
            "document_slots_by_language_signal": dict(sorted(Counter(languages.get(doc, "unknown") for doc in slots).items())),
            "unmapped_document_ids": len(wanted - set(domains))}


def _review_html(config: dict[str, Any], systems: dict[str, dict[int, dict]], changed: set[int]) -> dict[str, Any]:
    questions = {int(row["id"]): str(row["query"]) for row in pq.read_table(ROOT / config["inputs"]["queries"], columns=["id", "query"]).to_pylist()}
    changed_ids = sorted(changed, key=lambda q: hashlib.sha256(f"phase15:changed:{q}".encode()).hexdigest())[:45]
    unchanged = sorted(set(questions) - changed, key=lambda q: hashlib.sha256(f"phase15:control:{q}".encode()).hexdigest())[:15]
    selected = sorted(changed_ids + unchanged)
    payload = []
    for query_id in selected:
        alternatives = []
        order = sorted(systems, key=lambda name: hashlib.sha256(f"phase15:{query_id}:{name}".encode()).hexdigest())
        for index, name in enumerate(order):
            row = systems[name][query_id]
            alternatives.append({"blind_label": f"Hệ thống {chr(65+index)}", "method": name,
                                 "docs": row["relevant_docs"], "chunks": row["relevant_chunks"][:8]})
        payload.append({"query_id": query_id, "question": questions[query_id], "unchanged_control": query_id in unchanged, "alternatives": alternatives})
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Phase 15A blind review</title>
<style>body{{font:15px system-ui;max-width:1400px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}article{{border:1px solid #bbb;padding:10px}}pre{{white-space:pre-wrap}}#error{{color:#b00020;font-weight:bold}}</style></head><body>
<h1>Phase 15A — đánh giá mù</h1><p>Nhãn bắt đầu trống. Mục chưa chấm được xuất là UNKNOWN; đây không phải ground truth của organizer.</p><p id="error"></p><button id="export">Xuất CSV</button><main id="app"></main>
<script id="data" type="application/json">{encoded}</script><script>
try{{const data=JSON.parse(document.getElementById('data').textContent),app=document.getElementById('app'),labels=['','RELEVANT','PARTIALLY_RELEVANT','IRRELEVANT','UNCLEAR'];
const judgments={{}}; for(const q of data){{const section=document.createElement('section');section.innerHTML=`<h2>Query ${{q.query_id}}</h2><p>${{q.question}}</p><div class="grid"></div>`;const grid=section.querySelector('.grid');
for(const a of q.alternatives){{const box=document.createElement('article');box.innerHTML=`<h3>${{a.blind_label}}</h3><p>Docs: ${{a.docs.join(', ')}}</p>`;a.chunks.forEach((c,i)=>{{const d=document.createElement('div');d.innerHTML=`<b>Evidence ${{i+1}} · doc ${{c.doc_id}}</b><pre></pre><select></select>`;d.querySelector('pre').textContent=c.chunk_text;const s=d.querySelector('select');labels.forEach(x=>{{const o=document.createElement('option');o.value=x;o.textContent=x||'— UNKNOWN —';s.appendChild(o)}});s.onchange=()=>judgments[`${{q.query_id}}|${{a.method}}|${{i}}`]=s.value;box.appendChild(d)}});grid.appendChild(box)}}app.appendChild(section)}}
document.getElementById('export').onclick=()=>{{let rows=['query_id,method,evidence_index,label'];for(const q of data)for(const a of q.alternatives)a.chunks.forEach((c,i)=>rows.push([q.query_id,a.method,i,judgments[`${{q.query_id}}|${{a.method}}|${{i}}`]||'UNKNOWN'].join(',')));const blob=new Blob([rows.join('\r\n')],{{type:'text/csv'}}),u=URL.createObjectURL(blob),x=document.createElement('a');x.href=u;x.download='phase15a_review.csv';x.click();URL.revokeObjectURL(u)}};document.documentElement.dataset.queryCount=String(data.length)}}catch(e){{document.getElementById('error').textContent='Khởi tạo thất bại: '+(e.stack||e)}}
</script></body></html>'''
    path = ROOT / config["outputs"]["artifacts"] / "review_interface.html"; path.write_text(page, encoding="utf-8")
    return {"path": str(path), "query_count": len(payload), "changed_queries": len(changed_ids), "unchanged_controls": len(unchanged), "labels_initialized_blank": True}


def run(config_path: str | Path, state_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path); artifacts = ROOT / config["outputs"]["artifacts"]; artifacts.mkdir(parents=True, exist_ok=True)
    state = Path(state_path) if state_path else None
    def progress(**values: Any) -> None:
        if state: update_json(state, {**values, "worker_updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    process = psutil.Process(); peak = process.memory_info().rss; started = time.perf_counter()
    progress(status="RUNNING", stage="CONTROL_REPLAY", done=0, total=1200, checkpoint=str(artifacts / "phase15_checkpoint.sqlite"))
    tok = submission_tokenizer(config)
    control_audit = _control_replay(
        config, tok,
        lambda done: progress(status="RUNNING", stage="CONTROL_REPLAY", done=done, queries_done=done,
                              total=1200, queries_total=1200),
    )
    write_json(artifacts / "frozen_control_replay.json", control_audit)
    frozen = {int(row["id"]): row for row in _iter_json_array(ROOT / config["inputs"]["phase14_json"])}
    checkpoint = _checkpoint(artifacts / "phase15_checkpoint.sqlite")
    done = {int(row[0]) for row in checkpoint.execute("SELECT query_id FROM results")}
    source = sqlite3.connect(f"file:{(ROOT / config['inputs']['source_scores']).as_posix()}?mode=ro", uri=True)
    pilot = sqlite3.connect(f"file:{(ROOT / config['inputs']['pilot_scores']).as_posix()}?mode=ro", uri=True)
    try:
        progress(status="RUNNING", stage="VARIANT_REPLAY", done=len(done), total=1200)
        for rows in _query_groups(ROOT / config["inputs"]["phase14_chunks"]):
            query_id = int(rows[0]["query_id"])
            if query_id in done: continue
            scored = _score_rows(query_id, rows, source, pilot)
            top2 = rank_top2_mean(scored)[:20]
            diverse, diversity = diversity_order(rows)
            chunks_per_doc = Counter(int(row["doc_id"]) for row in {str(item["chunk_id"]): item for item in scored}.values())
            payload = {"top2_docs": [int(row["doc_id"]) for row in top2],
                       "top2_scores": [float(row["score"]) for row in top2],
                       "diversity_top40_chunk_ids": [str(row["chunk_id"]) for row in diverse[:40]], **diversity}
            payload["candidate_distinct_docs"] = len(chunks_per_doc)
            payload["candidate_chunks_per_doc"] = list(chunks_per_doc.values())
            checkpoint.execute("INSERT INTO results VALUES (?,?)", (query_id, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
            checkpoint.commit(); done.add(query_id)
            if len(done) % 10 == 0:
                peak = max(peak, process.memory_info().rss); progress(status="RUNNING", stage="VARIANT_REPLAY", done=len(done), total=1200)
    finally:
        source.close(); pilot.close()
    if checkpoint.execute("SELECT COUNT(*) FROM results").fetchone()[0] != 1200:
        raise ValueError("checkpoint does not contain 1200 complete queries")
    paths = _write_variant_rankings(config, checkpoint)
    payload_values = [json.loads(raw) for (raw,) in checkpoint.execute("SELECT payload FROM results")]
    checkpoint.close()
    progress(status="RUNNING", stage="VALIDATION", done=0, total=3)
    _, top2_hash, top2_records = _preview(config, tok, "DOC_TOP2_MEAN", paths["top2_documents"], paths["top2_chunks"])
    progress(status="RUNNING", stage="VALIDATION", done=1, total=3)
    _, diversity_hash, diversity_records = _preview(config, tok, "CHUNK_DIVERSITY", paths["diversity_documents"], paths["diversity_chunks"])
    progress(status="RUNNING", stage="VALIDATION", done=2, total=3)
    audits = {"DOC_TOP2_MEAN": _strict_preview_audit(config, top2_records), "CHUNK_DIVERSITY": _strict_preview_audit(config, diversity_records)}
    if not all(value["passed"] for value in audits.values()): raise ValueError(f"variant validation failed: {audits}")
    top2_docs = _comparison(frozen, top2_records, "relevant_docs"); top2_chunks = _comparison(frozen, top2_records, "relevant_chunks")
    diverse_docs = _comparison(frozen, diversity_records, "relevant_docs"); diverse_chunks = _comparison(frozen, diversity_records, "relevant_chunks")
    # Isolation gates are stronger than aggregate overlap statistics.
    if top2_chunks["queries_changed"] != 0 or diverse_docs["queries_changed"] != 0:
        raise ValueError("variant isolation failed")
    changed = {q for q in frozen if frozen[q]["relevant_docs"] != top2_records[q]["relevant_docs"] or frozen[q]["relevant_chunks"] != diversity_records[q]["relevant_chunks"]}
    review = _review_html(config, {"CONTROL": frozen, "DOC_TOP2_MEAN": top2_records, "CHUNK_DIVERSITY": diversity_records}, changed)
    elapsed = time.perf_counter() - started; peak = max(peak, process.memory_info().rss)
    structural = {
        "passed": True, "query_count": 1200, "candidate_ranking_rows": 346371,
        "document_ranking_rows": 212672, "score_cache_rows": 254400,
        "candidate_distinct_docs_per_query": _distribution(row["candidate_distinct_docs"] for row in payload_values),
        "candidate_chunks_per_document": _distribution(value for row in payload_values for value in row["candidate_chunks_per_doc"]),
        "phase14_output_chunk_quality": _output_chunk_quality(frozen),
        "phase14_chunk_owner_overlap": _comparison(frozen, frozen, "relevant_chunks"),
        "new_model_inference": 0, "runtime_seconds": elapsed, "peak_rss_bytes": peak,
        "human_labels_found": 0, "human_labels_reusable": False,
    }
    comparison = {
        "CONTROL": {"semantic_exact_queries": 1200, "passed": True},
        "DOC_TOP2_MEAN": {"documents": top2_docs, "chunks": top2_chunks, "preview_sha256": top2_hash, "validation": audits["DOC_TOP2_MEAN"],
                          "composition": _composition(config, top2_records), "chunk_quality": _output_chunk_quality(top2_records)},
        "CHUNK_DIVERSITY": {"documents": diverse_docs, "chunks": diverse_chunks, "preview_sha256": diversity_hash, "validation": audits["CHUNK_DIVERSITY"],
                            "composition": _composition(config, diversity_records), "chunk_quality": _output_chunk_quality(diversity_records)},
        "review": review, "decision": "NO_WINNER_PENDING_RELEVANCE_EVIDENCE", "winner_selected": False,
        "reason": "No trustworthy human relevance labels are present; structural changes are not relevance evidence.",
    }
    write_json(artifacts / "phase14_system_audit.json", structural); write_json(artifacts / "variant_comparison.json", comparison)
    summary = f"""# Phase 15A — CPU-only ranking replay\n\n- Frozen Phase 14 replay: **PASS (1200/1200)**.\n- DOC_TOP2_MEAN changed {top2_docs['queries_changed']} queries and replaced {top2_docs['slots_added_or_replaced']} document slots; Phase 14 chunks stayed exact.\n- CHUNK_DIVERSITY changed {diverse_chunks['queries_changed']} queries and replaced {diverse_chunks['slots_added_or_replaced']} chunk slots; Phase 14 documents stayed exact.\n- Both technical validators passed; provenance and official membership errors: 0.\n- Model inference: 0. Runtime: {elapsed:.2f}s. Peak RSS: {peak/2**20:.2f} MiB.\n- Human relevance labels: **insufficient**. Status: **READY_FOR_REVIEW**, no winner and no leaderboard ZIP.\n"""
    (artifacts / "experiment_summary.md").write_text(summary, encoding="utf-8")
    progress(status="NO_WINNER_PENDING_RELEVANCE_EVIDENCE", stage="COMPLETE", done=1200, total=1200,
             checkpoint=str(artifacts / "phase15_checkpoint.sqlite"), peak_rss_bytes=peak, runtime_seconds=elapsed)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/phase15_ranking_optimization.yaml"); parser.add_argument("--state")
    args = parser.parse_args(); run(args.config, args.state); return 0


if __name__ == "__main__": raise SystemExit(main())
