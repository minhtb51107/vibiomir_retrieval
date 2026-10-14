#!/usr/bin/env python3
"""Phase 15B: bounded unified Vietnamese BM25 candidate-discovery pilot."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.retrieval.vietnamese_lexical import add_documents, content_hash, open_index, search, tokens
from src.runtime.progress_contract import StageProgress, progress_snapshot
from src.runtime.state_file import update_json
from src.submission.common import atomic_write_json, file_sha256

STAGES = ("MANIFEST", "INDEX_BUILD", "PILOT_RETRIEVAL", "AUDIT", "DECISION", "PACKAGING")
SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("official_url", pa.string()), ("canonical_domain", pa.string()),
    ("content_hash", pa.string()), ("source_reference", pa.string()),
])


def load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("run_id") != "phase15b_candidate_retrieval_pilot":
        raise ValueError("unexpected Phase 15B configuration")
    return value


def sha_order(seed: str, value: object) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def topic(text: str) -> str:
    value = text.casefold()
    word_set = set(tokens(value))
    groups = (
        ("medication", ({"thuốc", "uống", "tiêm", "vaccine"}, ("vắc xin",))),
        ("treatment", ({"chữa"}, ("điều trị", "phẫu thuật", "cắt bỏ"))),
        ("symptom", ({"đau", "sốt", "ho", "ngứa", "sưng", "nôn", "mệt"}, ())),
        ("disease", ({"viêm", "bệnh", "nhiễm", "u"}, ("ung thư",))),
    )
    return next((name for name, (words, phrases) in groups
                 if word_set.intersection(words) or any(phrase in value for phrase in phrases)), "other")


def stratified_queries(path: Path, seed: str, count: int = 120) -> list[dict]:
    rows = [{"query_id": int(r["id"]), "query_text": str(r["query"])} for r in pq.read_table(path).to_pylist()]
    if len(rows) != 1200 or len({r["query_id"] for r in rows}) != 1200:
        raise ValueError("official query contract mismatch")
    lengths = sorted(len(tokens(r["query_text"])) for r in rows)
    cuts = [lengths[int(len(lengths) * q)] for q in (.25, .5, .75)]
    def length_bucket(n: int) -> int: return sum(n > cut for cut in cuts)
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        row["length_bucket"] = length_bucket(len(tokens(row["query_text"])))
        row["topic_proxy"] = topic(row["query_text"])
        groups[(row["length_bucket"], row["topic_proxy"])].append(row)
    quotas = {key: count * len(value) / len(rows) for key, value in groups.items()}
    allocated = {key: math.floor(value) for key, value in quotas.items()}
    for key in sorted(groups, key=lambda k: (-(quotas[k] - allocated[k]), k))[:count-sum(allocated.values())]:
        allocated[key] += 1
    chosen = []
    for key, values in groups.items():
        chosen.extend(sorted(values, key=lambda r: sha_order(seed, r["query_id"]))[:allocated[key]])
    return sorted(chosen, key=lambda r: r["query_id"])


def state_writer(state_path: Path, pipeline_started: dt.datetime):
    stage_starts: dict[str, dt.datetime] = {}
    def write(stage: str, operation: str, done: int | None, total: int | None, **extra) -> None:
        now = dt.datetime.now(dt.timezone.utc); stage_starts.setdefault(stage, now)
        snapshot = progress_snapshot(StageProgress(stage, operation, done, total, stage_starts[stage], now, True), ordered_stages=STAGES)
        snapshot["pipeline_elapsed_seconds"] = round((now-pipeline_started).total_seconds(), 1)
        update_json(state_path, {"status":"RUNNING", **snapshot, **extra})
    return write


def _source_rows(path: Path):
    columns = ["doc_id", "original_url", "title", "normalized_text", "language_signal", "extraction_status"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=columns):
        yield from batch.to_pylist()


def build_manifest(config: dict, write, artifacts: Path) -> dict:
    inputs, outputs = config["inputs"], config["outputs"]
    registry_path = ROOT/inputs["registry"]
    selected: dict[int, dict] = {}
    table = pq.read_table(registry_path, columns=["doc_id","official_url","canonical_domain","language_signal","content_available","document_text_reference","content_hash","provenance_status"])
    for row in table.to_pylist():
        if str(row["language_signal"]).startswith("vi") and row["content_available"] and str(row["provenance_status"]).startswith("VERIFIED"):
            selected[int(row["doc_id"])] = {**row, "source_reference":str(row["document_text_reference"]).split("#")[0]}
    phase14_source = str(inputs["phase14_documents"])
    for row in _source_rows(ROOT/phase14_source):
        if str(row["language_signal"]).startswith("vi") and row["extraction_status"] == "SUCCESS":
            selected.setdefault(int(row["doc_id"]), {"doc_id":int(row["doc_id"]), "official_url":row["original_url"],
                "canonical_domain":urlsplit(row["original_url"]).netloc.lower().removeprefix("www."), "source_reference":phase14_source,
                "content_hash":None, "provenance_status":"VERIFIED_PHASE14_NEW_OFFICIAL"})
    official = {}
    wanted = set(selected)
    pf = pq.ParquetFile(ROOT/inputs["official_corpus"])
    for batch in pf.iter_batches(batch_size=65536, columns=["id","url"]):
        for doc_id, url in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if int(doc_id) in wanted: official[int(doc_id)] = str(url)
    missing_official = sorted(wanted-set(official))
    url_mismatch = [doc for doc in wanted if str(selected[doc]["official_url"]) != official.get(doc)]
    if missing_official or url_mismatch:
        raise ValueError(f"manifest official closure failed missing={len(missing_official)} url_mismatch={len(url_mismatch)}")
    source_paths = sorted({str(v["source_reference"]).replace("\\","/") for v in selected.values()})
    found: dict[int, tuple[str,str,str]] = {}
    text_bytes = 0
    for source in source_paths:
        targets = {doc for doc,row in selected.items() if str(row["source_reference"]).replace("\\","/") == source}
        for row in _source_rows(ROOT/source):
            doc_id=int(row["doc_id"])
            if doc_id not in targets: continue
            title, body = str(row["title"] or ""), str(row["normalized_text"] or "")
            digest=content_hash(title,body); found[doc_id]=(digest,title,body); text_bytes += len(title.encode())+len(body.encode())
        write("MANIFEST", f"Xác minh nội dung {source}", len(found), len(selected))
    missing_content=sorted(wanted-set(found))
    if missing_content: raise ValueError(f"canonical content incomplete: missing={len(missing_content)} first={missing_content[:5]}")
    rows=[]
    for doc_id in sorted(selected):
        digest=found[doc_id][0]
        rows.append({"doc_id":doc_id,"official_url":official[doc_id],"canonical_domain":selected[doc_id]["canonical_domain"],
                     "content_hash":digest,"source_reference":str(selected[doc_id]["source_reference"]).replace("\\","/")})
    manifest=ROOT/outputs["manifest"]; manifest.parent.mkdir(parents=True,exist_ok=True)
    tmp=manifest.with_suffix(".parquet.tmp"); pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),tmp,compression="zstd"); os.replace(tmp,manifest)
    projected=min(text_bytes*2.5 + len(rows)*4096, 2**63-1)
    report={"passed":True,"documents":len(rows),"domains":len({r['canonical_domain'] for r in rows}),"source_files":source_paths,
            "official_missing":0,"url_mismatches":0,"content_missing":0,"duplicate_doc_ids":0,"source_text_bytes":text_bytes,
            "projected_index_bytes_conservative":int(projected),"manifest_sha256":file_sha256(manifest)}
    atomic_write_json(artifacts/"manifest_audit.json",report); write("MANIFEST","Manifest NEW-official tiếng Việt hoàn tất",len(rows),len(rows))
    return report


def preflight(config: dict, artifacts: Path, write) -> dict:
    contracts, resources, outputs = config["contracts"],config["resources"],config["outputs"]
    hashes={"json":file_sha256(ROOT/config["inputs"]["phase14_json"]),"zip":file_sha256(ROOT/config["inputs"]["phase14_zip"])}
    frozen=hashes=={"json":contracts["phase14_json_sha256"],"zip":contracts["phase14_zip_sha256"]}
    manifest=build_manifest(config,write,artifacts)
    free=shutil.disk_usage(ROOT).free; rss=psutil.Process().memory_info().rss
    projected, sample = sample_index_projection(config)
    passed=frozen and projected <= resources["maximum_additional_disk_gib"]*2**30 and free-projected >= resources["minimum_free_disk_gib"]*2**30
    report={"passed":passed,"frozen_phase14":frozen,"phase14_hashes":hashes,"manifest":manifest,
            "free_disk_gib":free/2**30,"projected_additional_disk_gib":projected/2**30,"preflight_rss_mib":rss/2**20,
            "projection_basis":sample,
            "limits":resources,"model_inference":0}
    atomic_write_json(artifacts/"resource_preflight.json",report)
    if not passed: raise ValueError(f"Phase15B resource/frozen preflight failed: {report}")
    return report


def _manifest_maps(config: dict) -> tuple[dict[int,dict],dict[str,set[int]]]:
    manifest={int(r["doc_id"]):r for r in pq.read_table(ROOT/config["outputs"]["manifest"]).to_pylist()}
    by_source: dict[str,set[int]]=defaultdict(set)
    for doc,row in manifest.items(): by_source[str(row["source_reference"])].add(doc)
    return manifest,by_source


def sample_index_projection(config: dict, sample_count: int = 768) -> tuple[int,dict]:
    """Measure a real contentless FTS index and conservatively project its size."""
    manifest,by_source=_manifest_maps(config)
    selected=set(sorted(manifest,key=lambda doc:sha_order(config["seed"]+":index-sample",doc))[:sample_count])
    sample_path=ROOT/config["outputs"]["data"]/"preflight_sample_fts5.sqlite"
    for path in sample_path.parent.glob(sample_path.name+"*"): path.unlink(missing_ok=True)
    connection=open_index(sample_path); rows=[]; text_bytes=0
    for source,targets in by_source.items():
        needed=targets & selected
        if not needed: continue
        for row in _source_rows(ROOT/source):
            doc=int(row["doc_id"])
            if doc not in needed: continue
            title,body=str(row["title"] or ""),str(row["normalized_text"] or ""); text_bytes+=len(title.encode())+len(body.encode())
            meta=manifest[doc]; rows.append({"doc_id":doc,"official_url":meta["official_url"],"canonical_domain":meta["canonical_domain"],
                "title":title,"body":body,"content_hash":meta["content_hash"],"source_reference":source})
    add_documents(connection,rows); connection.execute("PRAGMA wal_checkpoint(TRUNCATE)"); connection.close()
    measured=sum(path.stat().st_size for path in sample_path.parent.glob(sample_path.name+"*"))
    total_text=int(json.loads((ROOT/config["outputs"]["artifacts"]/"manifest_audit.json").read_text(encoding="utf-8"))["source_text_bytes"])
    # A 25% safety factor covers distribution and SQLite page effects without
    # replacing the measured index ratio with an arbitrary corpus multiplier.
    projected=math.ceil(measured/max(text_bytes,1)*total_text*1.25)
    return projected,{"sample_documents":len(rows),"sample_text_bytes":text_bytes,"sample_index_bytes":measured,"safety_factor":1.25,
                      "projected_index_bytes":projected,"engine":"SQLite FTS5 contentless"}


def build_index(config: dict, write) -> dict:
    output=ROOT/config["outputs"]["index"]; manifest,by_source=_manifest_maps(config); connection=open_index(output)
    existing=int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]); total=len(manifest)
    batch_size=int(config["resources"]["index_batch_rows"]); batch=[]
    start=time.perf_counter()
    for source,targets in sorted(by_source.items()):
        for row in _source_rows(ROOT/source):
            doc_id=int(row["doc_id"])
            if doc_id not in targets: continue
            meta=manifest[doc_id]; batch.append({"doc_id":doc_id,"official_url":meta["official_url"],"canonical_domain":meta["canonical_domain"],
                "title":str(row["title"] or ""),"body":str(row["normalized_text"] or ""),"content_hash":meta["content_hash"],"source_reference":source})
            if len(batch)>=batch_size:
                add_documents(connection,batch); batch=[]
                done=int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
                write("INDEX_BUILD","Ghi SQLite FTS5 contentless theo batch",done,total,index_path=str(output))
    if batch: add_documents(connection,batch)
    done=int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]); write("INDEX_BUILD","Đóng index FTS5",done,total,index_path=str(output))
    integrity=connection.execute("PRAGMA integrity_check").fetchone()[0]; count=int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]); connection.close()
    if count!=total or integrity!="ok": raise ValueError(f"index closure failed count={count}/{total} integrity={integrity}")
    return {"passed":True,"documents":count,"resumed_documents":existing,"integrity":integrity,"seconds":time.perf_counter()-start,
            "index_bytes":sum(p.stat().st_size for p in output.parent.glob(output.name+'*')),"index_sha256":file_sha256(output)}


def init_candidates(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True,exist_ok=True); c=sqlite3.connect(path); c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=FULL")
    c.execute("CREATE TABLE IF NOT EXISTS queries(query_id INTEGER PRIMARY KEY, query_text TEXT NOT NULL, length_bucket INTEGER NOT NULL, topic_proxy TEXT NOT NULL)")
    c.execute("CREATE TABLE IF NOT EXISTS candidates(query_id INTEGER,rank INTEGER,doc_id INTEGER,bm25_score REAL,canonical_domain TEXT,content_hash TEXT,PRIMARY KEY(query_id,doc_id),UNIQUE(query_id,rank))")
    c.commit(); return c


def retrieve(config: dict, write) -> dict:
    queries=stratified_queries(ROOT/config["inputs"]["queries"],config["seed"],config["contracts"]["pilot_query_count"])
    idx=sqlite3.connect(f"file:{ROOT/config['outputs']['index']}?mode=ro",uri=True); out=init_candidates(ROOT/config["outputs"]["candidates"])
    completed={int(r[0]) for r in out.execute("SELECT DISTINCT query_id FROM candidates GROUP BY query_id HAVING COUNT(*)=?",(config["contracts"]["lexical_depth"],))}
    latencies=[]
    for ordinal,q in enumerate(queries,1):
        if q["query_id"] not in completed:
            tick=time.perf_counter(); rows=search(idx,q["query_text"],config["contracts"]["lexical_depth"]); latencies.append(time.perf_counter()-tick)
            if len(rows)!=config["contracts"]["lexical_depth"]: raise ValueError(f"query {q['query_id']} returned {len(rows)}")
            with out:
                out.execute("INSERT OR REPLACE INTO queries VALUES(?,?,?,?)",(q["query_id"],q["query_text"],q["length_bucket"],q["topic_proxy"]))
                out.execute("DELETE FROM candidates WHERE query_id=?",(q["query_id"],))
                out.executemany("INSERT INTO candidates VALUES(?,?,?,?,?,?)",[(q["query_id"],r["rank"],r["doc_id"],r["bm25_score"],r["canonical_domain"],r["content_hash"]) for r in rows])
        done=int(out.execute("SELECT COUNT(*) FROM queries").fetchone()[0]); write("PILOT_RETRIEVAL","Truy vấn BM25 Top-100",done,len(queries))
    idx.close(); out.close()
    return {"passed":True,"queries":len(queries),"candidate_rows":len(queries)*config["contracts"]["lexical_depth"],
            "latency_seconds":{"median":statistics.median(latencies) if latencies else 0,"max":max(latencies) if latencies else 0}}


def _score_keys(path: Path, table: str, query_ids: set[int]) -> set[tuple[int,str]]:
    c=sqlite3.connect(f"file:{path}?mode=ro",uri=True); keys=set()
    for q in query_ids:
        for chunk, in c.execute(f"SELECT chunk_id FROM {table} WHERE query_id=?",(q,)): keys.add((q,str(chunk)))
    c.close(); return keys


def audit(config: dict, write, artifacts: Path, index_report: dict, retrieval_report: dict) -> dict:
    c=sqlite3.connect(ROOT/config["outputs"]["candidates"]); query_rows=c.execute("SELECT query_id,query_text FROM queries ORDER BY query_id").fetchall()
    candidates=defaultdict(list)
    for q,rank,doc,score,domain,digest in c.execute("SELECT query_id,rank,doc_id,bm25_score,canonical_domain,content_hash FROM candidates ORDER BY query_id,rank"):
        candidates[int(q)].append({"rank":rank,"doc_id":doc,"score":score,"domain":domain,"hash":digest})
    c.close(); qids=set(candidates)
    baseline=defaultdict(set)
    pf=pq.ParquetFile(ROOT/config["inputs"]["phase14_ranked_chunks"])
    for batch in pf.iter_batches(batch_size=65536,columns=["query_id","doc_id"]):
        for q,d in zip(batch.column(0).to_pylist(),batch.column(1).to_pylist()):
            if int(q) in qids: baseline[int(q)].add(int(d))
    overlap=[]; lexical_only=0; domain_counts=[]; duplicate_slots=0
    candidate_docs=set()
    for q,rows in candidates.items():
        docs={int(r["doc_id"]) for r in rows}; candidate_docs |= docs; shared=len(docs & baseline[q]); overlap.append(shared)
        lexical_only += len(docs-baseline[q]); domain_counts.append(len({r["domain"] for r in rows}))
        duplicate_slots += len(rows)-len({r["hash"] for r in rows})
    # Select up to two original chunks/doc by transparent lexical token overlap.
    wanted=defaultdict(list)
    query_text=dict(query_rows)
    for q,rows in candidates.items():
        for r in rows: wanted[int(r["doc_id"])].append(q)
    best: dict[tuple[int,int],list[tuple[tuple,int,str]]] = defaultdict(list); owner={}
    seen_chunks=set(); sources=[config["inputs"]["broad_chunks"],config["inputs"]["closure_chunks"],config["inputs"]["phase14_chunks"]]
    audit_total=len(sources)+1+120
    for source_no,source in enumerate(sources,1):
        path=ROOT/source
        if not path.exists(): continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=["chunk_id","doc_id","normalized_text"]):
            for row in batch.to_pylist():
                doc=int(row["doc_id"]); chunk=str(row["chunk_id"])
                if doc not in wanted or chunk in seen_chunks: continue
                seen_chunks.add(chunk); owner[chunk]=doc; chunk_terms=Counter(tokens(row["normalized_text"] or ""))
                for q in wanted[doc]:
                    qterms=set(tokens(query_text[q])); hit=sum(chunk_terms[t] for t in qterms); key=(q,doc)
                    best[key].append(((-hit,int(row.get("chunk_index",0) or 0),chunk),doc,chunk))
        write("AUDIT",f"Chọn evidence nguyên bản: {Path(source).name}",source_no,audit_total)
    evidence=[]
    for (q,doc),values in best.items():
        for _,_,chunk in sorted(values)[:config["contracts"]["evidence_chunks_per_document"]]: evidence.append((q,chunk,doc))
    expected_docs=sum(len(v) for v in candidates.values()); docs_without=sum(1 for q,rows in candidates.items() for r in rows if (q,int(r['doc_id'])) not in best)
    source_keys=_score_keys(ROOT/config["inputs"]["phase14_scores"],"pairs",qids); pilot_keys=_score_keys(ROOT/config["inputs"]["pilot_scores"],"scores",qids)
    evidence_keys={(q,ch) for q,ch,_ in evidence}; reused=len(evidence_keys & (source_keys|pilot_keys)); missing=len(evidence_keys-(source_keys|pilot_keys))
    write("AUDIT","Đối soát exact score keys",len(sources)+1,audit_total)
    # Determinism: repeat every query against the immutable index and compare exact IDs/scores.
    idx=sqlite3.connect(f"file:{ROOT/config['outputs']['index']}?mode=ro",uri=True); mismatches=0
    for ordinal,(q,text) in enumerate(query_rows,1):
        repeat=search(idx,text,config["contracts"]["lexical_depth"])
        expected=[(r["doc_id"],r["score"]) for r in candidates[q]]; actual=[(r["doc_id"],r["bm25_score"]) for r in repeat]
        mismatches += expected != actual
        if ordinal%20==0: write("AUDIT","Lặp retrieval để kiểm tra determinism",len(sources)+1+ordinal,audit_total)
    idx.close(); write("AUDIT","Kiểm tra provenance và frozen control",audit_total,audit_total)
    frozen={"json":file_sha256(ROOT/config["inputs"]["phase14_json"]),"zip":file_sha256(ROOT/config["inputs"]["phase14_zip"])}
    contracts=config["contracts"]; frozen_ok=frozen=={"json":contracts["phase14_json_sha256"],"zip":contracts["phase14_zip_sha256"]}
    passed=(len(qids)==120 and mismatches==0 and docs_without==0 and frozen_ok and all(owner[ch]==doc for _,ch,doc in evidence))
    report={"passed":passed,"claim_boundary":"Candidate structure only; lexical-only is UNKNOWN relevance and is not Recall/F2 evidence.",
      "queries":len(qids),"lexical_candidate_rows":sum(map(len,candidates.values())),"unique_candidate_documents":len(candidate_docs),
      "phase14_candidate_overlap":{"mean_at_100":statistics.mean(overlap),"min":min(overlap),"max":max(overlap),"total_shared_slots":sum(overlap)},
      "lexical_only_query_document_pairs":lexical_only,"duplicate_content_exposure_slots":duplicate_slots,
      "source_diversity":{"mean_domains_per_query":statistics.mean(domain_counts),"min":min(domain_counts),"max":max(domain_counts)},
      "evidence_pairs":len(evidence_keys),"candidate_documents_without_original_chunks":docs_without,"exact_scores_reused":reused,
      "genuinely_missing_reranker_pairs":missing,"new_model_inference":0,"determinism_mismatches":mismatches,
      "provenance_owner_mismatches":sum(owner.get(ch)!=doc for _,ch,doc in evidence),"frozen_phase14":frozen_ok,"phase14_hashes":frozen,
      "index":index_report,"retrieval":retrieval_report}
    atomic_write_json(artifacts/"pilot_candidate_audit.json",report)
    if not passed: raise ValueError(f"Phase15B audit failed: {report}")
    return report


def decision(report: dict, artifacts: Path) -> str:
    # Engineering GO only: candidates are not labeled relevance evidence.
    independent = report["lexical_only_query_document_pairs"] > 0 and report["phase14_candidate_overlap"]["mean_at_100"] < 100
    verdict="GO_FOR_BOUNDED_RELEVANCE_EVALUATION" if independent else "NO_GO_NO_INDEPENDENT_SIGNAL"
    text=f"""# Phase 15B pilot decision\n\n- Verdict: **{verdict}**\n- 120-query structural audit: **PASS**.\n- Lexical-only query-document pairs: **{report['lexical_only_query_document_pairs']:,}**.\n- Mean overlap with Phase 14 candidate universe: **{report['phase14_candidate_overlap']['mean_at_100']:.2f}/100**.\n- Exact missing reranker pairs: **{report['genuinely_missing_reranker_pairs']:,}**; inference executed: **0**.\n\nĐây chỉ là bằng chứng arm BM25 tạo tín hiệu candidate độc lập, không phải bằng chứng relevance hay F2. GO chỉ cho phép thiết kế một bước đánh giá/rerank nhỏ sau phê duyệt; không cho phép mở rộng 1.200 query hoặc tạo submission.\n"""
    (artifacts/"PHASE15B_DECISION.md").write_text(text,encoding="utf-8"); return verdict


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/phase15b_candidate_retrieval.yaml"); parser.add_argument("--state-path"); parser.add_argument("--preflight-only",action="store_true"); args=parser.parse_args()
    config=load_config(ROOT/args.config); artifacts=ROOT/config["outputs"]["artifacts"]; artifacts.mkdir(parents=True,exist_ok=True)
    state_path=Path(args.state_path) if args.state_path else ROOT/config["outputs"]["run"]/"state.json"; started=dt.datetime.now(dt.timezone.utc); write=state_writer(state_path,started)
    baseline_rss=psutil.Process().memory_info().rss; write("MANIFEST","Khởi tạo manifest NEW-official",0,1)
    resource=preflight(config,artifacts,write)
    if args.preflight_only:
        update_json(state_path,{"status":"READY_FOR_NEXT_STAGE","stage":"MANIFEST","preflight":resource,"worker_alive":False}); return 0
    index_report=build_index(config,write); atomic_write_json(artifacts/"index_build_report.json",index_report)
    retrieval_report=retrieve(config,write); audit_report=audit(config,write,artifacts,index_report,retrieval_report)
    write("DECISION","Áp dụng gate quyết định kỹ thuật",0,1); verdict=decision(audit_report,artifacts); write("DECISION","Quyết định pilot hoàn tất",1,1)
    peak=psutil.Process().memory_info().rss
    resource.update(actual_index_bytes=index_report["index_bytes"],incremental_rss_mib=max(0,peak-baseline_rss)/2**20,free_disk_gib_after=shutil.disk_usage(ROOT).free/2**30)
    resource["passed_runtime_limits"] = resource["incremental_rss_mib"] <= config["resources"]["maximum_incremental_rss_mib"] and index_report["index_bytes"] <= config["resources"]["maximum_additional_disk_gib"]*2**30
    atomic_write_json(artifacts/"resource_report.json",resource)
    if not resource["passed_runtime_limits"]: raise ValueError(f"runtime resource gate failed: {resource}")
    now=dt.datetime.now(dt.timezone.utc); terminal=progress_snapshot(StageProgress("PACKAGING","NOT_APPLICABLE — pilot không tạo ZIP",1,1,now,now,False),ordered_stages=STAGES,terminal=True)
    update_json(state_path,{"status":"READY_FOR_NEXT_STAGE",**terminal,"decision":verdict,"packaging_status":"NOT_APPLICABLE","checkpoint":str(ROOT/config['outputs']['candidates']),"new_model_inference":0,"worker_alive":False})
    print(json.dumps({"status":"READY_FOR_NEXT_STAGE","decision":verdict,"audit":audit_report,"resource":resource},ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
