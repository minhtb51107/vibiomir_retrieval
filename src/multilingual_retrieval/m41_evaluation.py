from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psutil
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.multilingual_retrieval.m2 import sha256_file
from src.multilingual_retrieval.m4 import PHASE29_SHA256, fixed_development_queries, load_lexicon, _iter_m4_documents
from src.multilingual_retrieval.m41 import ChineseRetriever, group_hits, retrieve_by_text, semantic_preflight


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pairs(rows: list[dict[str, Any]]) -> set[tuple[int, int]]:
    return {(int(r["query_id"]), int(r["doc_id"])) for r in rows}


# Source-based provisional assessment, not labels supplied to retrieval. All
# unresolved answer-absence claims remain UNKNOWN rather than corpus negatives.
FAILURES = {
    128: ("ANSWER_ALREADY_RETRIEVED_DIFFERENTIAL_ONLY", "1595280", "The source matches morning/on-waking sneezing but does not establish why the patient wakes or the actual diagnosis."),
    257: ("MISSING_ANSWER_CONTENT_UNCONFIRMED", "3078924", "Drug catalog co-occurrence does not answer beef intake during intestinal spasm; absence elsewhere is not established."),
    290: ("REQUIRED_ENTITY_MISSING_IN_CANDIDATES", "1595304", "Rhinitis passages discuss obstruction but do not establish nasopharyngeal-cyst causation and treatment."),
    409: ("PARTIAL_EVIDENCE_AND_QUERY_UNDERSTANDING_LIMIT", "3016157,3078850", "Hypertrichosis explanation and folliculitis-drug indications answer different subparts; no complete axillary assessment found."),
    446: ("POPULATION_AND_BILIRUBIN_TYPE_MISMATCH", "3062674", "The strongest old match concerns neonatal hemolysis and predominantly unconjugated bilirubin; do not promote it for elevated direct bilirubin."),
    491: ("QUERY_UNDERSTANDING_ERROR", "3103691", "The long question includes left-sided pain, AST and a round lung shadow; the finite lexicon fails to represent the complete radiographic relationship."),
    513: ("MULTI_ENTITY_ANSWER_CONTENT_UNCONFIRMED", "3078711", "Vitamin C and hypothyroidism may co-occur in long lists without an interaction claim. No reviewed direct answer is established."),
    571: ("PARTIAL_MECHANISM_NOT_MANAGEMENT", "3003971,3056474", "Malaria explains splenomegaly; this is not guidance for painful post-malaria splenomegaly. Full-document hits alone do not establish a cross-chunk answer."),
    634: ("WRONG_DISCOVERY_NEIGHBORHOOD", "1872374,1872415", "M4 /nkwk/ggths pages concern femoral-head necrosis (股骨头坏死), not gamma-glutamyl transferase. GGT remains unsupported by reviewed evidence."),
    715: ("PARTIAL_TRANSMISSION_NOT_ERADICATION_ANSWER", "1662742", "The article explains HP transmission and advises formal treatment, but provides neither an infection symptom set nor an eradication regimen."),
    805: ("PARTIAL_RISK_NOT_AGE_RELATION", "3094304,3820076", "Male-infertility and urethritis passages give partial mechanisms, not the requested age-specific susceptibility."),
    1080: ("POPULATION_CONTEXT_AND_PAIN_LOCATION_GAP", "3055183,3483799", "Pregnancy/labor and supine-position passages do not establish whether right chest pain at 38 weeks is a labor sign."),
}


def audit_old_labels(root: Path, m4rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final = {(int(r["query_id"]), int(r["doc_id"]), str(r["chunk_id"])): r for r in m4rows}
    audited = []
    for original in read_csv(root / "artifacts/multilingual_retrieval_m4/m4_ai_judgments.csv"):
        key = (int(original["query_id"]), int(original["doc_id"]), original["chunk_id"])
        row = final[key]
        revised = "1" if key[:2] == (128, 1595280) else original["judgment"]
        explanation = ("Useful disease differential and explicit morning/on-waking relation; awakening mechanism and patient-specific cause are not answered. Downgrade complete/direct-answer claim."
                       if key[:2] == (128,1595280) else "Source reread supports the earlier partial/mismatch assessment; no upgraded direct claim.")
        audited.append({"query_id": key[0], "doc_id": key[1], "chunk_id": key[2],
                        "original_judgment": original["judgment"], "revised_judgment": revised,
                        "verification_status": "AI_JUDGED_M41_AUDIT_UNBLINDED",
                        "evidence_relevance": "USEFUL_DIFFERENTIAL" if key[:2] == (128,1595280) else ("PARTIAL" if revised=="1" else "MISMATCH"),
                        "answer_completeness": "INCOMPLETE" if revised=="1" else "DOES_NOT_ANSWER",
                        "provenance": "CHECKED_SEPARATELY", "reason": explanation,
                        "canonical_url": row["canonical_url"]})
    return audited


def audit_exact_cache(root: Path, queries: dict[int,str], rows: list[dict[str, Any]], trace: list[dict[str, Any]], final: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    identities = {(int(r["doc_id"]),r["evidence_text"]):r["chunk_id"] for r in rows}
    by_id = {r["chunk_id"] for r in rows}
    measured = Counter()
    recovered = []
    contracts = []
    # Predicate pushdown, no dataset-wide fetchall, no source SQLite writes.
    for source in ("S1", "S2", "S3"):
        path = root / f"data/source_relevance_probe/{source}/reranked_chunks.parquet"
        connection = sqlite3.connect(f"file:{(path.parent/'rerank_scores.sqlite').as_posix()}?mode=ro",uri=True)
        connection.execute("PRAGMA cache_size=-1024")
        contract = json.loads(connection.execute("SELECT value FROM manifest WHERE key='signature'").fetchone()[0])
        required = {"model_name":"BAAI/bge-reranker-v2-m3", "resolved_revision":"953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                    "max_sequence_length":512,"inference_precision":"float16","batch_size":2}
        if any(contract.get(key)!=value for key,value in required.items()):
            connection.close()
            raise ValueError(f"cached reranker contract mismatch {source}")
        contracts.append({"source":source, **required})
        scanner = ds.dataset(path, format="parquet").scanner(
            columns=["query_id","query_text","doc_id","chunk_id","chunk_text","rerank_score","rerank_rank"],
            filter=ds.field("query_id").isin(sorted(queries)), batch_size=64,
            use_threads=False, batch_readahead=1, fragment_readahead=1)
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                measured["rows_examined"] += 1
                if row["chunk_id"] in by_id:
                    measured["same_chunk_id"] += 1
                if (int(row["doc_id"]),row["chunk_text"]) in identities:
                    measured["same_document_and_exact_text"] += 1
                    if row["query_text"] == queries[int(row["query_id"])]:
                        measured["same_query_document_exact_text"] += 1
                        score=connection.execute("SELECT rerank_score FROM scores WHERE query_id=? AND chunk_id=?",(int(row["query_id"]),row["chunk_id"])).fetchone()
                        if score is None or not math.isfinite(float(score[0])) or float(score[0])!=float(row["rerank_score"]):
                            raise ValueError("cached parquet/SQLite exact score mismatch")
                        chunk=identities[(int(row["doc_id"]),row["chunk_text"])]
                        cand=next((r for r in trace if r["query_id"]==int(row["query_id"]) and r["chunk_id"]==chunk),None)
                        chosen=next((r for r in final if r["query_id"]==int(row["query_id"]) and r["chunk_id"]==chunk),None)
                        recovered.append({"query_id":row["query_id"],"doc_id":row["doc_id"],"m4_chunk_id":chunk,
                                          "historical_chunk_id":row["chunk_id"],"exact_rerank_score":score[0],
                                          "historical_rerank_rank":row["rerank_rank"],"m41_candidate_rank":cand["candidate_rank"] if cand else "",
                                          "m41_final_rank":chosen["rank"] if chosen else "", "reference_source":source,
                                          "query_text_sha256":hashlib.sha256(row["query_text"].encode()).hexdigest(),
                                          "chunk_text_sha256":hashlib.sha256(row["chunk_text"].encode()).hexdigest()})
        connection.close()
    write_csv(output,recovered,["query_id","doc_id","m4_chunk_id","historical_chunk_id","exact_rerank_score","historical_rerank_rank","m41_candidate_rank","m41_final_rank","reference_source","query_text_sha256","chunk_text_sha256"])
    return {**dict(measured), "same_chunk_id": measured["same_chunk_id"],
            "same_query_document_exact_text": measured["same_query_document_exact_text"],
            "validated_exact_references":len(recovered), "queries_with_exact_reference":len({r["query_id"] for r in recovered}),
            "reference_rows_in_lexical_final":sum(r["m41_final_rank"]!="" for r in recovered),
            "contracts":contracts,"scores_imported": 0,
            "contract": "Exact query/doc/text and SQLite score checked. This sparse scored subset is a diagnostic ablation, not a complete semantic ranking."}


def evaluate(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    process = psutil.Process()
    rss_before = process.memory_info().rss/2**20
    available_before = psutil.virtual_memory().available/2**20
    out = root / "artifacts/multilingual_retrieval_m41"
    out.mkdir(parents=True, exist_ok=True)
    frozen_paths = [root / "artifacts/multilingual_retrieval_m4" / name for name in
                    ("candidate_corpus.parquet","zh_m4_top20.parquet","candidate_trace_top100.parquet","m4_report.json","M4_REPORT.md")]
    frozen_paths += [root / "configs/m4_ai_silver_decisions.json", root / "configs/m4_vi_zh_medical_lexicon.json"]
    frozen = {str(path.relative_to(root)):sha256_file(path) for path in frozen_paths}
    best = root / "submissions/00_READY_TO_UPLOAD/p29_bachmai2679_cap16.zip"
    if sha256_file(best) != PHASE29_SHA256:
        raise ValueError("Phase29 immutable hash mismatch")
    semantic = semantic_preflight(root)
    retriever = ChineseRetriever.from_root(root)
    index_rss_mib = process.memory_info().rss/2**20
    queries = fixed_development_queries(root,load_lexicon(root/"configs/m4_vi_zh_medical_lexicon.json"))
    m4 = pq.read_table(root/"artifacts/multilingual_retrieval_m4/zh_m4_top20.parquet").to_pylist()
    documents = {int(r["doc_id"]):r for _,r in _iter_m4_documents(root)}
    allowed = {}
    for batch in pq.ParquetFile(root/"data/raw/links_corpus_vi_en_zh.parquet").iter_batches(batch_size=4096,columns=["id","url"]):
        for row in batch.to_pylist():
            if int(row["id"]) in documents:
                allowed[int(row["id"])] = row["url"]
    if set(allowed) != set(documents):
        raise ValueError("non-official document present")
    for row in retriever.rows:
        doc = documents[int(row["doc_id"])]
        if row["canonical_url"] != allowed[int(row["doc_id"])]:
            raise ValueError("canonical URL mismatch")
        text = doc["normalized_text"]
        if row["evidence_text"] not in text or text[row["start_offset"]:row["end_offset"]] != row["evidence_text"]:
            raise ValueError("verbatim/offset mismatch")
    final, trace, taxonomy, review, ablations = [], [], [], [], []
    m3 = read_csv(root/"artifacts/multilingual_retrieval_m3/review/m3_ai_judgments_import.csv")
    positives = {(int(r["query_id"]),int(r["doc_id"])) for r in m3 if r["judgment"] in {"1","2"}}
    m4pairs = pairs(m4)
    benchmark_start = time.perf_counter()
    for query in queries:
        result = retrieve_by_text(query["query_vi"],root=root,retriever=retriever)
        replay = retrieve_by_text(query["query_vi"],root=root,retriever=retriever)
        if result["results"] != replay["results"] or result["candidates"] != replay["candidates"]:
            raise ValueError("nondeterministic ranking")
        qid = int(query["query_id"])
        no_qualifiers = retriever.retrieve(query["query_vi"],root=root,rank_qualifiers=False)
        before_ids=[(r["doc_id"],r["chunk_id"]) for r in no_qualifiers["results"]]
        after_ids=[(r["doc_id"],r["chunk_id"]) for r in result["results"]]
        ablations.append({"query_id":qid,"same_corpus_and_admission":True,"only_rank_qualifiers_changed":True,
                          "selection_changed":before_ids!=after_ids,
                          "without_qualifier_results":[{"doc_id":r["doc_id"],"chunk_id":r["chunk_id"],"rank":r["rank"]} for r in no_qualifiers["results"]],
                          "with_qualifier_results":[{"doc_id":r["doc_id"],"chunk_id":r["chunk_id"],"rank":r["rank"]} for r in result["results"]]})
        current = [{**r,"query_id":qid} for r in result["results"]]
        candidates = [{**r,"query_id":qid} for r in result["candidates"]]
        final.extend(current); trace.extend(candidates)
        parsed = result["understanding"]
        full = [docid for docid,doc in documents.items() if parsed["entity_groups"] and group_hits(doc["normalized_text"],parsed["entity_groups"]) == len(parsed["entity_groups"])]
        same_chunk = {int(r["doc_id"]) for r in retriever.rows if parsed["entity_groups"] and group_hits(r["evidence_text"],parsed["entity_groups"]) == len(parsed["entity_groups"])}
        related = []
        for docid in sorted({int(d) for d in FAILURES[qid][1].split(",")}):
            raw = documents.get(docid)
            if raw:
                relevant_rows = [r for r in candidates if int(r["doc_id"])==docid]
                related.append({"doc_id":docid,"title":raw["title"],"canonical_url":raw["original_url"],
                                "candidate":relevant_rows[:1],"original_excerpt":raw["normalized_text"][:600]})
        hypothesis, ids, reason = FAILURES[qid]
        taxonomy.append({"query_id":qid,"query_vi":query["query_vi"],"primary_diagnostic":hypothesis,
                         "m4_query_representation":query["query_zh"],
                         "answer_absence":"UNKNOWN", "reason":reason, "understanding":parsed,
                         "full_doc_entity_hits":full,"potential_cross_chunk_entity_hits":sorted(set(full)-same_chunk),
                         "cross_chunk_relation_answer":"UNVERIFIED", "examples":related,
                         "final_count":len(current),"candidate_count":len(candidates),
                         "runtime_seconds":result["elapsed_seconds"]})
        # At most one high-value new/uncertain row per query; no labels supplied.
        chosen = next((r for r in current if (qid,int(r["doc_id"])) not in m4pairs),None)
        if qid == 128:
            chosen = next((r for r in current if int(r["doc_id"])==1595280),chosen)
        if chosen:
            review.append({**chosen,"query_vi":query["query_vi"],"judgment":"",
                           "verification_status":"PENDING_INDEPENDENT_REVIEW",
                           "review_reason":"New selection or uncertain answer completeness; no retrieval score shown"})
        elif qid in {446,513,571,715,805}:
            chosen = next((r for r in candidates if r["entity_hits"]),None)
            if chosen:
                review.append({**chosen,"query_vi":query["query_vi"],"judgment":"",
                               "verification_status":"PENDING_INDEPENDENT_REVIEW",
                               "review_reason":"Uncertain completeness, population, or multi-entity relation; admission status hidden"})
    development_seconds = time.perf_counter()-benchmark_start
    audited = audit_old_labels(root,m4)
    cache = audit_exact_cache(root,{int(q["query_id"]):q["query_vi"] for q in queries},retriever.rows,trace,final,out/"cache_score_recovery.csv")
    retained = []
    for qid,docid in sorted(positives):
        old = next(r for r in m3 if int(r["query_id"])==qid and int(r["doc_id"])==docid)
        cand = next((r for r in trace if r["query_id"]==qid and int(r["doc_id"])==docid),None)
        selected = next((r for r in final if r["query_id"]==qid and int(r["doc_id"])==docid),None)
        retained.append({"query_id":qid,"doc_id":docid,"m3_original_chunk_id":old["chunk_id"],
                         "original_judgment":old["judgment"],"candidate_present":cand is not None,
                         "candidate_rank":cand["candidate_rank"] if cand else None,
                         "final_rank":selected["rank"] if selected else None,
                         "admission_reasons":cand["admission_reasons"] if cand else ["NO_LEXICAL_ENTITY_CANDIDATE"],
                         "text_equivalence":"RECHUNKED_NOT_AUTOMATIC_LABEL_TRANSFER"})
    smokes = []
    for text in ("Tăng huyết áp có gây đau đầu vào buổi sáng không?",
                 "Trẻ em hen suyễn ho buổi sáng điều trị thế nào?",
                 "Người bệnh thận mạn có thể uống ibuprofen không?",
                 "Bệnh Wilson có thể dùng trientine không?"):
        value = retrieve_by_text(text,root=root,retriever=retriever)
        smokes.append({"query_vi":text,"understanding":value["understanding"],"results":len(value["results"]),
                       "candidates":len(value["candidates"]),"runtime_seconds":value["elapsed_seconds"],"relevance_labels_assigned":False})
    en_path=root/"artifacts/multilingual_retrieval_m31/en_m31_protected_top20.parquet"
    en=pairs(pq.read_table(en_path,columns=["query_id","doc_id"]).to_pylist())
    en_known=pairs([r for r in read_csv(root/"artifacts/multilingual_retrieval_m1/ai_judgments_import.csv") if r["target_language"]=="en" and r["judgment"] in {"1","2"}])
    bridge=pairs([r for r in read_csv(root/"artifacts/multilingual_retrieval_m2/ai_judgments_m2.csv") if r["target_language"]=="en" and r["judgment"]=="2"])
    if len(en & en_known)!=24 or len(en & bridge)!=2:
        raise ValueError("English positive preservation failed")
    if any(sha256_file(root/path)!=digest for path,digest in frozen.items()) or sha256_file(best)!=PHASE29_SHA256:
        raise ValueError("frozen inputs changed")
    report={"status":"COMPLETE_LIGHTWEIGHT_M41_SEMANTIC_DEFERRED", "held_out_opened":False,
            "frozen_m4_inputs":frozen,"phase29_sha256":PHASE29_SHA256,
            "baseline":{"m4_final_rows":len(m4),"m4_config_decisions":len(json.loads((root/"configs/m4_ai_silver_decisions.json").read_text())["decisions"]),
                        "m4_runtime_silver_counts":dict(Counter(r["original_judgment"] for r in audited))},
            "m41":{"final_rows":len(final),"queries_with_results":len({r["query_id"] for r in final}),
                   "candidate_rows":len(trace),"known_partial_pairs":len(positives),
                   "known_partial_candidate_retained":sum(r["candidate_present"] for r in retained),
                   "known_partial_top100_retained":sum(r["candidate_rank"] is not None and r["candidate_rank"]<=100 for r in retained),
                   "known_partial_final_retained":len(positives & pairs(final)),
                   "revised_audit_counts":dict(Counter(r["revised_judgment"] for r in audited)),
                   "independently_blind_verified_direct_queries":0,"human_verified":0},
            "resource":{"rss_before_mib":rss_before,"rss_after_mib":process.memory_info().rss/2**20,
                        "rss_after_lexical_index_mib":index_rss_mib,
                        "process_peak_working_set_mib":getattr(process.memory_info(),"peak_wset",process.memory_info().rss)/2**20,
                        "available_before_mib":available_before,"available_after_mib":psutil.virtual_memory().available/2**20,
                        "development_12_plus_replay_seconds":development_seconds,"total_seconds":time.perf_counter()-started},
            "cache_compatibility":cache,"semantic_pilot":semantic,
            "english":{"baseline_positive_retained":len(en&en_known),"bridge_direct_retained":len(en&bridge),
                       "control_sha256":sha256_file(en_path)},"smokes":smokes,
            "validation":{"official_membership":True,"canonical_url":True,"verbatim_and_offsets":True,"determinism":True},
            "decision":"GO_QUERY_TEXT_FUNCTIONALITY_NO_GO_DIRECT_ANSWER_GENERALIZATION_CLAIM"}
    write_csv(out/"m4_evidence_reassessment.csv",audited,list(audited[0]))
    write_csv(out/"independent_review_queue.csv",review,["query_id","query_vi","doc_id","chunk_id","canonical_url","evidence_text","judgment","verification_status","review_reason"])
    write_csv(out/"partial_retention.csv",retained,list(retained[0]))
    for name,value in (("development_comparison.json",report),("candidate_failure_taxonomy.json",taxonomy),("qualifier_ablation.json",ablations)):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    pq.write_table(pa.Table.from_pylist(final),out/"zh_m41_top20.parquet")
    return {"status":report["status"],"m41":report["m41"],"resource":report["resource"],"semantic_pilot":semantic,"cache":cache}
