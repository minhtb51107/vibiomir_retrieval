from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from src.multilingual_retrieval.m2 import sha256_file
from src.multilingual_retrieval.m3 import _atomic_json
from src.multilingual_retrieval.m4 import PHASE29_SHA256, fixed_development_queries, load_lexicon
from src.multilingual_retrieval.m41 import ROOT, semantic_preflight, understand
from src.multilingual_retrieval.m41_evaluation import audit_exact_cache, read_csv

REVIEW_STATUS = "AI_JUDGED_M41_ASSISTANT_UNBLINDED"
EMBEDDING = {"name":"BAAI/bge-m3", "revision":"5617a9f61b028005a4858fdac845db406aefb181",
             "device":"cuda", "batch_size":4, "query_batch_size":4, "max_length":512,
             "normalize_embeddings":True, "use_half_on_cuda":True, "dimension":1024, "seed":2026}
RERANKER = {"model_name":"BAAI/bge-reranker-v2-m3", "resolved_revision":"953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "max_sequence_length":512,"inference_precision":"float16","batch_size":2}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def freeze_json(path: Path, value: dict[str,Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"frozen M4.2 input changed: {path}")
        return
    _atomic_json(path,value)


def require_idle(root: Path = ROOT) -> dict[str,Any]:
    state = semantic_preflight(root)
    if state["phase27_pids"]:
        raise RuntimeError("DEFERRED_PHASE27_ACTIVE: no model load permitted")
    return state  # Idle alone is NOT a resource/equivalence PASS.


def validate_review(rows: list[dict[str,str]], queue: list[dict[str,str]]) -> list[dict[str,Any]]:
    key = lambda r:(int(r["query_id"]),int(r["doc_id"]),str(r["chunk_id"]))
    expected = {key(r):r for r in queue}
    if len(expected)!=6 or len(rows)!=6:
        raise ValueError("expected exactly six frozen review records")
    joined,seen=[],set()
    for row in rows:
        identity=key(row)
        if identity in seen or identity not in expected:
            raise ValueError("duplicate or unrecognized query/doc/chunk review identity")
        if row.get("verification_status")!=REVIEW_STATUS:
            raise ValueError("review must remain AI_JUDGED_M41_ASSISTANT_UNBLINDED")
        label=str(row.get("judgment",row.get("label","")))
        if label not in {"0","1","2","UNSURE"}:
            raise ValueError("invalid silver label")
        source=expected[identity]
        for field in ("evidence_text","canonical_url","query_vi"):
            if row.get(field) and row[field]!=source[field]:
                raise ValueError(f"review source mismatch: {field}")
        seen.add(identity)
        joined.append({**source,"judgment":label,"verification_status":REVIEW_STATUS,
                       "review_notes":row.get("review_notes",row.get("reason","")),
                       "evidence_sha256":hashlib.sha256(source["evidence_text"].encode()).hexdigest(),
                       "human_verified":False,"blind_review":False})
    if seen!=set(expected):
        raise ValueError("review set incomplete")
    if Counter(r["judgment"] for r in joined)!=Counter({"1":3,"0":3}):
        raise ValueError("review counts differ from frozen user-provided 3 partial / 3 wrong summary")
    return joined


def import_review(root: Path, out: Path) -> dict[str,Any]:
    path=root/"artifacts/multilingual_retrieval_m41/review/M41_assistant_silver_review_6.csv"
    if not path.exists():
        return {"status":"BLOCKED_REVIEW_FILE_MISSING","path":str(path),"imported":0,
                "user_reported_summary":{"partial":3,"wrong":3,"direct":0},
                "summary_is_not_row_level_labels":True}
    queue=read_csv(root/"artifacts/multilingual_retrieval_m41/independent_review_queue.csv")
    joined=validate_review(read_csv(path),queue)
    payload={"input_sha256":sha256_file(path),"rows":joined,"status":REVIEW_STATUS}
    freeze_json(out/"assistant_review_diagnostics.json",payload)
    return {"status":"IMPORTED_SEPARATELY","imported":len(joined),"input_sha256":payload["input_sha256"],
            "labels":dict(Counter(r["judgment"] for r in joined)),"human_verified":0,"blind":False}


def make_manifest(root: Path, chunks: list[dict[str,Any]]) -> dict[str,Any]:
    baseline=json.loads((root/"artifacts/multilingual_retrieval_m41/development_comparison.json").read_text(encoding="utf-8"))
    for relative,expected in baseline["frozen_m4_inputs"].items():
        if sha256_file(root/relative)!=expected:
            raise ValueError(f"M4 baseline differs from frozen M4.1: {relative}")
    queries=fixed_development_queries(root,load_lexicon(root/"configs/m4_vi_zh_medical_lexicon.json"))
    representations=[{"query_id":q["query_id"],"query_vi":q["query_vi"],
                      "query_zh":understand(q["query_vi"],root)["query_zh"],
                      "understanding":understand(q["query_vi"],root)} for q in queries]
    chosen=sorted(representations,key=lambda q:(hashlib.sha256(q["query_vi"].encode()).hexdigest(),q["query_id"]))[:5]
    chosen_chunks=sorted(chunks,key=lambda r:(hashlib.sha256(r["evidence_text"].encode()).hexdigest(),int(r["doc_id"]),r["chunk_id"]))[:10]
    texts=[{"kind":"QUERY_VI","query_id":q["query_id"],"text":q["query_vi"]} for q in chosen]
    texts += [{"kind":"QUERY_ZH_AUTO","query_id":q["query_id"],"text":q["query_zh"]} for q in chosen]
    texts += [{"kind":"CHUNK","doc_id":r["doc_id"],"chunk_id":r["chunk_id"],"text":r["evidence_text"]} for r in chosen_chunks]
    if len(texts)!=20 or len(chunks)!=2188 or len({r["doc_id"] for r in chunks})>368:
        raise ValueError("frozen M4 corpus / trial cardinality mismatch")
    config=yaml.safe_load((root/"configs/source_relevance_probe.yaml").read_text(encoding="utf-8"))
    if config["models"]["embedder"]!=EMBEDDING:
        raise ValueError("pinned embedding contract changed")
    model=config["models"]["reranker"]
    actual={"model_name":model["name"],"resolved_revision":model["revision"],"max_sequence_length":model["max_sequence_length"],
            "inference_precision":model["precision"],"batch_size":model["batch_size"]}
    if actual!=RERANKER:
        raise ValueError("pinned reranker contract changed")
    hashes={str(p.relative_to(root)):sha256_file(p) for p in (
        root/"artifacts/multilingual_retrieval_m4/candidate_corpus.parquet",
        root/"artifacts/multilingual_retrieval_m4/zh_m4_top20.parquet",
        root/"artifacts/multilingual_retrieval_m41/zh_m41_top20.parquet",
        root/"configs/m4_vi_zh_medical_lexicon.json",root/"configs/m41_query_terms.json",
        root/"configs/source_relevance_probe.yaml",root/"src/reranking/reranker.py")}
    return {"version":"m42-bounded-semantic-v1","inputs_sha256":hashes,"embedding":EMBEDDING,
            "reranker":RERANKER,"tokenizer_revision":RERANKER["resolved_revision"],
            "tokenization":{"truncation":"longest_first","max_length":512,"score":"raw_logit"},
            "queries":representations,"trial20":texts,"trial20_sha256":digest(texts),
            "corpus_rows":[{"doc_id":r["doc_id"],"chunk_id":r["chunk_id"],
                            "text_sha256":hashlib.sha256(r["evidence_text"].encode()).hexdigest()} for r in chunks],
            "evaluation":{"document_top_k":[20,100],"final_top_k":20,"union_rrf_constant":60,
                          "max_rerank_documents_per_query":8,"max_new_rerank_pairs":96,
                          "unjudged_is_negative":False,"held_out_used":False}}


def commit_references(path: Path, refs: list[dict[str,str]], contract: dict[str,Any]) -> dict[str,Any]:
    connection=sqlite3.connect(path)
    try:
        connection.execute("PRAGMA cache_size=-1024")
        connection.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, query_id INTEGER, doc_id INTEGER, chunk_id TEXT, score REAL, reference_source TEXT, historical_chunk_id TEXT, contract TEXT)")
        before=connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        with connection:
            for row in refs:
                score=float(row["exact_rerank_score"])
                if not math.isfinite(score):
                    raise ValueError("nonfinite cached score")
                identity={"query_sha256":row["query_text_sha256"],"doc_id":int(row["doc_id"]),
                          "chunk_text_sha256":row["chunk_text_sha256"],"contract":contract}
                key=digest(identity)
                old=connection.execute("SELECT score FROM scores WHERE key=?",(key,)).fetchone()
                if old is not None and float(old[0])!=score:
                    raise ValueError("conflicting exact cache reference")
                connection.execute("INSERT OR IGNORE INTO scores VALUES (?,?,?,?,?,?,?,?)",(
                    key,int(row["query_id"]),int(row["doc_id"]),row["m4_chunk_id"],score,
                    row["reference_source"],row["historical_chunk_id"],json.dumps(contract,sort_keys=True)))
        check=connection.execute("PRAGMA quick_check").fetchone()[0]
        if check!="ok":
            raise ValueError("owned score checkpoint integrity failed")
        after=connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        return {"rows":after,"new_rows":after-before,"integrity":check,"no_inference":True}
    finally:
        connection.close()


def verify_tokenizer_contract(root: Path) -> dict[str,Any]:
    snapshot=root/".cache/huggingface/models--BAAI--bge-reranker-v2-m3/snapshots"/RERANKER["resolved_revision"]
    names=("tokenizer_config.json","tokenizer.json","special_tokens_map.json","sentencepiece.bpe.model")
    hashes={name:sha256_file(snapshot/name) for name in names}
    config=json.loads((snapshot/"tokenizer_config.json").read_text(encoding="utf-8"))
    if config.get("tokenizer_class")!="XLMRobertaTokenizer" or config.get("model_max_length",0)<512:
        raise ValueError("cached tokenizer incompatible with pinned pair contract")
    return {"revision":RERANKER["resolved_revision"],"files_sha256":hashes,
            "tokenizer_class":config["tokenizer_class"],"native_max_length":config["model_max_length"],
            "applied_pair_max_length":512,"truncation":"longest_first",
            "loader_sha256":sha256_file(root/"src/reranking/reranker.py"),
            "evidence":"Pinned local revision plus loader uses the same revision for tokenizer and model; historical SQLite does not store tokenizer file hashes",
            "empirical_equivalence":"NOT_RUN"}


def rank_dense(query_vectors: np.ndarray, chunk_vectors: np.ndarray, rows: list[dict[str,Any]], k: int) -> list[list[dict[str,Any]]]:
    """Pure comparison utility: never loads a model or assigns relevance labels."""
    if k not in {20,100} or chunk_vectors.shape!=(len(rows),1024) or query_vectors.ndim!=2 or query_vectors.shape[1]!=1024:
        raise ValueError("dense shape / budget mismatch")
    for vectors in (query_vectors,chunk_vectors):
        if vectors.dtype!=np.float32 or not np.isfinite(vectors).all():
            raise ValueError("expected finite float32 vectors")
        if np.any(np.linalg.norm(vectors,axis=1)==0):
            raise ValueError("zero vectors cannot satisfy normalized embedding contract")
    ranked=[]
    for query in query_vectors:
        values=chunk_vectors@query
        order=sorted(range(len(rows)),key=lambda i:(-float(values[i]),int(rows[i]["doc_id"]),str(rows[i]["chunk_id"])))
        seen,selected=set(),[]
        for index in order:
            doc=int(rows[index]["doc_id"])
            if doc in seen:
                continue
            seen.add(doc); selected.append({"doc_id":doc,"chunk_id":rows[index]["chunk_id"],"rank":len(selected)+1,"dense_score":float(values[index])})
            if len(selected)==k:
                break
        ranked.append(selected)
    return ranked


def union_rrf(arms: list[list[dict[str,Any]]], k: int = 100) -> list[dict[str,Any]]:
    scores: dict[int,float]={}
    representative={}
    for arm in arms:
        seen=set()
        for row in arm[:k]:
            doc=int(row["doc_id"])
            if doc in seen:
                raise ValueError("arm must be document-deduplicated")
            seen.add(doc)
            scores[doc]=scores.get(doc,0.0)+1/(60+int(row["rank"]))
            if doc not in representative:
                representative[doc]=row
    return [{"doc_id":doc,"chunk_id":representative[doc]["chunk_id"],"rank":rank,"rrf_score":scores[doc]}
            for rank,doc in enumerate(sorted(scores,key=lambda d:(-scores[d],d))[:k],1)]


def retention(arms: dict[str,dict[int,list[dict[str,Any]]]], partial_pairs: set[tuple[int,int]], query_ids: set[int]) -> dict[str,Any]:
    result={}
    for name,by_query in arms.items():
        if set(by_query)!=query_ids:
            raise ValueError("every arm must cover the same frozen query set, including empty results")
        result[name]={}
        for k in (20,100):
            selected={(qid,int(r["doc_id"])) for qid,rows in by_query.items() for r in rows[:k]}
            result[name][str(k)]={"known_partial_pairs_found":len(selected&partial_pairs),
                                  "known_partial_pairs_total":len(partial_pairs),
                                  "not_official_recall":True,"unjudged_is_negative":False}
    return result


def new_evidence_review(queries: dict[int,str], arms: dict[str,dict[int,list[dict[str,Any]]]],
                        baseline_pairs: set[tuple[int,int]], chunks: list[dict[str,Any]], maximum: int=12) -> list[dict[str,Any]]:
    by_chunk={r["chunk_id"]:r for r in chunks}
    selected={}
    for by_query in arms.values():
        for qid,rows in by_query.items():
            for row in rows[:20]:
                if (qid,int(row["doc_id"])) in baseline_pairs:
                    continue
                source=by_chunk[row["chunk_id"]]
                if int(source["doc_id"])!=int(row["doc_id"]):
                    raise ValueError("review chunk owner mismatch")
                identity=(qid,int(row["doc_id"]),row["chunk_id"])
                selected[identity]={"case_id":digest(identity)[:16],"query_vi":queries[qid],
                                    "query_id":qid,"doc_id":row["doc_id"],"chunk_id":row["chunk_id"],
                                    "canonical_url":source["canonical_url"],"evidence_text":source["evidence_text"],
                                    "judgment":"","verification_status":"PENDING_INDEPENDENT_REVIEW"}
    return sorted(selected.values(),key=lambda r:r["case_id"])[:maximum]


def prepare(root: Path = ROOT) -> dict[str,Any]:
    out=root/"artifacts/multilingual_retrieval_m42"
    out.mkdir(parents=True,exist_ok=True)
    before=semantic_preflight(root)
    if sha256_file(root/"submissions/00_READY_TO_UPLOAD/p29_bachmai2679_cap16.zip")!=PHASE29_SHA256:
        raise ValueError("Phase29 frozen hash mismatch")
    chunks=pq.read_table(root/"artifacts/multilingual_retrieval_m4/candidate_corpus.parquet").to_pylist()
    manifest=make_manifest(root,chunks)
    freeze_json(out/"frozen_manifest.json",manifest)
    baseline_hashes={str(p.relative_to(root)):sha256_file(p) for p in (
        root/"artifacts/multilingual_retrieval_m41/development_comparison.json",
        root/"artifacts/multilingual_retrieval_m41/independent_review_queue.csv",
        root/"artifacts/multilingual_retrieval_m31/en_m31_protected_top20.parquet")}
    freeze_json(out/"baseline_integrity.json",baseline_hashes)
    tokenizer=verify_tokenizer_contract(root)
    freeze_json(out/"tokenizer_contract.json",tokenizer)
    review=import_review(root,out)
    queries={int(q["query_id"]):q["query_vi"] for q in manifest["queries"]}
    cache=audit_exact_cache(root,queries,chunks,[],[],out/"exact_reference_verification.csv")
    refs=read_csv(out/"exact_reference_verification.csv")
    if len(refs)!=19:
        raise ValueError("expected 19 reverified exact references")
    checkpoint=commit_references(out/"exact_scores.sqlite",refs,{**RERANKER,"tokenizer_revision":RERANKER["resolved_revision"],"truncation":"longest_first","normalization":"raw_logit"})
    after=semantic_preflight(root)
    report={"status":"PREPARATION_COMPLETE_INFERENCE_DEFERRED", "phase27_preflight":before,
            "phase27_after":after,"review":review,"cache":cache,"score_checkpoint":checkpoint,
            "tokenizer_contract":tokenizer,
            "trial20_manifest_sha256":manifest["trial20_sha256"],"model_loaded":False,
            "resource_benchmark":"NOT_RUN","equivalence_benchmark":"NOT_RUN",
            "dense_arms":"NOT_RUN","reranker_arm":"NOT_RUN","new_evidence_review_queue":"NOT_GENERATED_NO_NEW_INFERENCE",
            "next_condition":"Phase27 process tree must exit; then existing resource and equivalence gates plus measured trial are mandatory",
            "decision":"DEFER_SEMANTIC_PILOT_PENDING_PHASE27_AND_REVIEW_FILE",
            "phase29_sha256":PHASE29_SHA256,"human_verified":0}
    for relative,expected in manifest["inputs_sha256"].items():
        if sha256_file(root/relative)!=expected:
            raise ValueError("frozen input changed during preparation")
    for relative,expected in baseline_hashes.items():
        if sha256_file(root/relative)!=expected:
            raise ValueError("baseline changed during preparation")
    _atomic_json(out/"preparation_report.json",report)
    return report
