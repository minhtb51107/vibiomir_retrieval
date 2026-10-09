from __future__ import annotations
import gc,hashlib,json,shutil,sqlite3,subprocess,sys,time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from src.retrieval.dense_retriever import normalize_text
from src.source_census.depth1000 import seed_exact_scores
from src.source_census.pipeline import atomic_json,deterministic_domain_samples

G5A=["benhviennhitrunguong.gov.vn","zydcd.com","hellobacsi.com"]
GROUPS={"G5A":G5A}
def host(url): return (urlsplit(str(url)).hostname or "").lower().removeprefix("www.")

def phase_contract(config):
    """Return the configured focused-scaling contract without changing legacy G5A semantics."""
    key=str(config.get("focused_scaling",{}).get("phase_key","phase10e"))
    phase=config[key]
    sources=list(phase["targets"])
    group=str(config.get("focused_scaling",{}).get("group","G5A"))
    experiment=str(config.get("focused_scaling",{}).get("experiment","phase10e_g5a_11200"))
    return key,phase,sources,group,experiment

def _crawl_sets(paths,sources=G5A):
    result={s:{"attempted":set(),"success":set()} for s in sources}
    for path in paths:
        connection=sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro",uri=True)
        for doc_id,url,status in connection.execute("SELECT doc_id,original_url,status FROM crawl_results"):
            source=host(url)
            if source in result:
                result[source]["attempted"].add(int(doc_id))
                if status=="SUCCESS": result[source]["success"].add(int(doc_id))
        connection.close()
    return result

def _path_bytes_without_temporary_files(paths):
    total=0
    for value in paths:
        path=Path(value)
        if path.is_file():
            if not path.name.endswith(".tmp"): total+=path.stat().st_size
            continue
        if path.is_dir():
            total+=sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.name.endswith(".tmp"))
    return total

def _sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def _cached_model_revision(model,revision,cache_dir=None):
    slug="models--"+str(model).replace("/","--")
    roots=[]
    if cache_dir: roots.append(Path(cache_dir))
    roots.extend([Path(".cache/huggingface"),Path.home()/".cache/huggingface/hub"])
    return any((root/slug/"snapshots"/str(revision)).is_dir() for root in roots)

def _planning_projection(config,source_rows,free_bytes,crawl_minutes):
    """Project a reserve run from durable source and completed Phase10E evidence."""
    _,phase,sources,_,_=phase_contract(config); planning=phase.get("planning")
    if not planning: return None
    baselines=planning["source_baselines"]; projected_sources={}; final_chunks=0; new_chunks=0; final_clean=0; new_clean=0
    for source in sources:
        current=source_rows[source]; baseline=baselines[source]
        usable=int(baseline["usable_documents"]); successful=max(int(current["already_successful"]),1)
        projected_new_usable=round(int(current["new_ids_required"])*(usable/successful))
        projected_final_usable=usable+projected_new_usable
        chunks_per_usable=float(baseline["chunks"])/max(usable,1)
        bytes_per_usable=float(baseline["clean_utf8_bytes"])/max(usable,1)
        source_final_chunks=round(projected_final_usable*chunks_per_usable)
        source_final_clean=round(projected_final_usable*bytes_per_usable)
        row={
            "observed_usable_yield":usable/successful,
            "projected_new_usable_documents":projected_new_usable,
            "projected_final_usable_documents":projected_final_usable,
            "observed_chunks_per_usable_document":chunks_per_usable,
            "projected_final_chunks":source_final_chunks,
            "projected_new_embedding_rows":max(0,source_final_chunks-int(baseline["chunks"])),
            "projected_final_clean_utf8_bytes":source_final_clean,
            "projected_incremental_clean_utf8_bytes":max(0,source_final_clean-int(baseline["clean_utf8_bytes"])),
        }
        projected_sources[source]=row; final_chunks+=source_final_chunks; new_chunks+=row["projected_new_embedding_rows"]
        final_clean+=source_final_clean; new_clean+=row["projected_incremental_clean_utf8_bytes"]
    reference_report=json.loads(Path(planning["evidence"]["streaming_reference_report"]).read_text(encoding="utf-8"))
    reference_retained=_path_bytes_without_temporary_files(planning["reference_retained_paths"])
    scale=sum(int(source_rows[s]["new_ids_required"]) for s in sources)/float(planning["reference_incremental_ids"])
    contingency=1.0+float(planning.get("retained_contingency_fraction",0.15))
    retained=round(reference_retained*scale*contingency)
    embedding_bytes=final_chunks*int(config["models"]["embedder"]["dimension"])*4
    transient_peak=retained+embedding_bytes
    reference_new_chunks=int(planning["reference_new_chunks"])
    extraction_seconds=float(reference_report["extraction"]["runtime_seconds"])/int(planning["reference_incremental_ids"])*sum(int(source_rows[s]["new_ids_required"]) for s in sources)
    chunk_seconds=float(reference_report["chunking"]["runtime_seconds"])/reference_new_chunks*new_chunks
    embedding_seconds=float(reference_report["embeddings"]["runtime_seconds"])/reference_new_chunks*new_chunks
    candidate_seconds=float(reference_report["candidate_cache"]["runtime_seconds"])
    rerank_seconds=28800/20.0+60.0
    package_seconds=float(planning.get("packaging_audit_minutes",25))*60.0
    modeled_seconds=crawl_minutes*60+extraction_seconds+chunk_seconds+embedding_seconds+candidate_seconds+rerank_seconds+package_seconds
    memory=reference_report["embeddings"].get("memory_gate",{})
    return {
        "evidence":planning["evidence"],"sources":projected_sources,
        "projected_final_usable_documents":sum(x["projected_final_usable_documents"] for x in projected_sources.values()),
        "projected_final_chunks":final_chunks,"projected_new_embedding_rows":new_chunks,
        "projected_final_clean_utf8_bytes":final_clean,"projected_incremental_clean_utf8_bytes":new_clean,
        "reference_retained_bytes_excluding_temporary_files":reference_retained,
        "projected_retained_bytes_with_15pct_contingency":retained,
        "projected_transient_peak_bytes":transient_peak,
        "projected_minimum_free_bytes":free_bytes-transient_peak,
        "disk_gate_passed_at_transient_peak":free_bytes-transient_peak>=float(config["acquisition"]["minimum_free_gib"])*2**30,
        "runtime_seconds":{"concurrent_acquisition":crawl_minutes*60,"extraction":extraction_seconds,"chunking":chunk_seconds,"streaming_embeddings":embedding_seconds,"source_candidates":candidate_seconds,"rerank_conservative":rerank_seconds,"ranking_packaging_audit":package_seconds,"end_to_end_modeled":modeled_seconds,"end_to_end_with_25pct_contingency":modeled_seconds*1.25},
        "streaming_embedding_memory":{"observed_reference_peak_rss_mib":memory.get("observed_peak_rss_mib"),"observed_reference_peak_private_mib":memory.get("observed_peak_private_mib"),"projected_peak_private_mib":memory.get("projected_peak_private_mib"),"scales_with_corpus_rows":False,"parquet_batch_rows":int(config["embedding_recovery"]["parquet_batch_rows"]),"num_workers":int(config["embedding_recovery"]["num_workers"]),"prefetching":bool(config["embedding_recovery"]["prefetching"])},
        "methodology":"Current source-specific usable/chunk/text density; completed G1A streaming stage rates; measured G1A retained paths excluding *.tmp, scaled by incremental IDs with 15% contingency; transient peak adds one full embedding matrix for transactional output.",
    }

def prepare_manifest(config,enforce_disk_gate=True):
    _,phase,sources,_,experiment=phase_contract(config); targets={k:int(v) for k,v in phase["targets"].items()}
    crawl=_crawl_sets(phase["previous_crawl_databases"],sources)
    usable={s:set() for s in sources}
    for path in phase["previous_documents"]:
        for row in pq.read_table(path,columns=["doc_id","original_url","extraction_status","normalized_text"]).to_pylist():
            source=host(row["original_url"])
            if source in usable and row["extraction_status"]=="SUCCESS" and str(row["normalized_text"] or "").strip(): usable[source].add(int(row["doc_id"]))
    samples,populations=deterministic_domain_samples(config["inputs"]["corpus"],domains=set(sources),sample_size=max(12000,max(targets.values())),seed=phase["sampling_seed"])
    sources={}; total=0
    for source in targets:
        population=int(populations[source]); target=min(targets[source],population)
        successful=crawl[source]["success"]; attempted=crawl[source]["attempted"]
        # Preserve every successfully acquired official ID. Failed IDs are not
        # recrawled; deterministic unattempted IDs replace them to reach target.
        need=max(0,target-len(successful))
        incremental=[row for row in samples[source] if int(row["doc_id"]) not in attempted][:need]
        if len(incremental)!=need: raise RuntimeError(f"insufficient unattempted IDs for {source}")
        target_ids=sorted(successful)+[int(row["doc_id"]) for row in incremental]
        if len(target_ids)!=target or len(target_ids)!=len(set(target_ids)): raise RuntimeError(f"invalid target population {source}")
        sources[source]={"official_population":population,"already_attempted":len(attempted),"already_successful":len(successful),"usable_documents":len(usable[source]),"current_effective_depth":len(usable[source]),"target_depth":target,"new_ids_required":need,"failed_ids_not_retried":sorted(attempted-successful),"target_ids":target_ids,"incremental_records":incremental}
        total+=need
    free=shutil.disk_usage(Path.cwd()).free
    # Measured retained Phase10D directories, excluding submissions, were about
    # 2.28 GiB for 6,300 IDs. Add 35% contingency for checkpoints and packaging.
    measured_bytes_per_id=2.28*2**30/6300
    projected=total*measured_bytes_per_id*1.35
    rates=phase.get("observed_urls_per_minute",{"benhviennhitrunguong.gov.vn":59.27,"zydcd.com":39.41,"hellobacsi.com":55.84})
    conservative_rate=float(phase.get("fallback_urls_per_minute",20.0))
    crawl_minutes=max((sources[s]["new_ids_required"]/float(rates.get(s,conservative_rate)) for s in targets),default=0.0)
    planning=_planning_projection(config,sources,free,crawl_minutes)
    if planning:
        projected=planning["projected_retained_bytes_with_15pct_contingency"]
        gate=bool(planning["disk_gate_passed_at_transient_peak"])
        end_to_end=planning["runtime_seconds"]["end_to_end_with_25pct_contingency"]/3600
    else:
        gate=free-projected>=20*2**30; end_to_end=crawl_minutes/60+1.75
    result={"format_version":1,"experiment":experiment,"selection":"all prior successful official IDs plus lowest SHA-256(seed:doc_id) ranked unattempted official IDs; prior failures are not recrawled","sources":sources,"total_incremental_ids":total,"free_disk_before_bytes":free,"free_disk_before_gib":free/2**30,"projected_retained_bytes":round(projected),"projected_retained_gib":projected/2**30,"expected_free_disk_after_gib":(free-projected)/2**30,"minimum_free_disk_gib":float(config["acquisition"].get("minimum_free_gib",20)),"disk_gate_passed":gate,"projected_concurrent_crawl_minutes":crawl_minutes,"projected_end_to_end_hours":end_to_end,"resource_projection":planning,"future_checkpoint_prepared_only":phase.get("future_targets")}
    atomic_json(Path(phase["manifest"]),result)
    if enforce_disk_gate and not result["disk_gate_passed"]: raise RuntimeError("Phase10E violates 20 GiB disk floor")
    return result

def preflight(config):
    """Fail before acquisition when a focused-scaling dependency or contract is missing."""
    _,phase,sources,group,experiment=phase_contract(config)
    required=[
        config["inputs"]["corpus"],config["inputs"]["queries"],config["inputs"]["pilot_documents"],
        config["inputs"]["pilot_chunks"],config["inputs"]["pilot_rerank_pool"],
        config["inputs"]["pilot_rerank_scores"],config["inputs"]["c1_candidate_pool"],
        config["candidate_cache"]["calibrated_cap_artifact"],phase["previous_chunks"],
        phase["previous_embeddings"],phase["previous_score_database"],
        config["embedding_recovery"]["profile_artifact"],
        config["embedding_recovery"]["equivalence_artifact"],
        config["models"]["reranker"]["equivalence"]["investigation_artifact"],
        config["focused_scaling"]["control_submission_json"],
        config["focused_scaling"].get("control_submission_zip"),"submissions/MANIFEST.csv",
        "tools/competition_supervisor.py",
        *phase["previous_crawl_databases"],*phase["previous_documents"],*phase.get("supplemental_previous_chunks",[]),
    ]
    required=[value for value in required if value]
    missing=[str(p) for p in required if not Path(p).exists()]
    calibration={}
    calibration_path=Path(config["candidate_cache"]["calibrated_cap_artifact"])
    if calibration_path.exists(): calibration=json.loads(calibration_path.read_text(encoding="utf-8"))
    chosen=int(calibration.get("chosen_m",-1))
    contract=preflight_contract_checks(config,chosen,sources)
    model_cache={
        "embedder_revision_available":_cached_model_revision(config["models"]["embedder"]["name"],config["models"]["embedder"]["revision"],config["models"]["embedder"].get("cache_dir")),
        "reranker_revision_available":_cached_model_revision(config["models"]["reranker"]["name"],config["models"]["reranker"]["revision"],config["models"]["reranker"].get("cache_dir")),
    }
    control_hashes={}
    for kind in ("json","zip"):
        path=config["focused_scaling"].get(f"control_submission_{kind}")
        expected=config["focused_scaling"].get(f"control_submission_{kind}_sha256")
        actual=_sha256_file(path) if path and Path(path).exists() else None
        control_hashes[kind]={"path":path,"expected_sha256":expected,"actual_sha256":actual,"passed":expected is None or actual==expected}
    manifest=prepare_manifest(config,enforce_disk_gate=False) if not missing and all(contract.values()) else None
    writable=[]
    for directory in [Path(config["outputs"]["root"]),Path(config["outputs"]["artifacts"]),Path(config["outputs"]["submissions"]),Path(config["focused_scaling"]["audit_path"]).parent]:
        try:
            directory.mkdir(parents=True,exist_ok=True); probe=directory/".phase10e_preflight_write"; probe.write_text("ok",encoding="utf-8"); probe.unlink(); writable.append(str(directory))
        except OSError as exc: missing.append(f"writable:{directory}:{exc}")
    failures=[]
    if missing: failures.append("missing/unwritable dependencies: "+", ".join(missing))
    if not all(contract.values()): failures.append("fixed experiment contract failed: "+", ".join(k for k,v in contract.items() if not v))
    if not all(model_cache.values()): failures.append("pinned local model revision unavailable: "+", ".join(k for k,v in model_cache.items() if not v))
    if not all(value["passed"] for value in control_hashes.values()): failures.append("control submission hash mismatch")
    if manifest and not manifest["disk_gate_passed"]: failures.append("20 GiB disk gate failed")
    result={"experiment":experiment,"group":group,"passed":not failures,"failures":failures,"chosen_m":chosen,"calibration_artifact":str(calibration_path),"contract":contract,"model_cache":model_cache,"control_hashes":control_hashes,"stale_run_recovery":"competition_supervisor reconciles RUNNING state against recorded process liveness","dependencies_checked":len(required),"writable_directories":writable,"manifest":phase["manifest"],"disk_gate_passed":bool(manifest and manifest["disk_gate_passed"])}
    atomic_json(Path(config["outputs"]["artifacts"])/"preflight.json",result)
    if failures: raise RuntimeError("Phase10E preflight failed: "+"; ".join(failures))
    return result

def preflight_contract_checks(config,chosen,sources):
    _,phase,_,_,_=phase_contract(config)
    control_artifact=Path(config["models"]["reranker"]["equivalence"].get("control_artifact",""))
    return {
        "chosen_m_is_8":chosen==8 and list(config["candidate_cache"]["candidate_caps"])==[8],
        "reranker_revision_fixed":config["models"]["reranker"]["revision"]=="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "embedder_revision_fixed":config["models"]["embedder"]["revision"]=="5617a9f61b028005a4858fdac845db406aefb181",
        "embedder_dimension_1024":int(config["models"]["embedder"]["dimension"])==1024,
        "reranker_fp16":str(config["models"]["reranker"].get("precision","fp16")).lower()=="fp16",
        "reranker_batch_2":int(config["models"]["reranker"]["batch_size"])==2,
        "reranker_max_length_512":int(config["models"]["reranker"]["max_length"])==512,
        "submission_top10_top20":int(config["submission"]["documents_per_query"])==10 and int(config["submission"]["chunks_per_query"])==20,
        "source_targets_match":set(sources)==set(phase["targets"]),
        "experiment_scoped_reranker_control":bool(str(control_artifact)) and control_artifact.parent==Path(config["outputs"]["artifacts"]),
        "streaming_embeddings_bounded":int(config.get("embedding_recovery",{}).get("parquet_batch_rows",0))>0 and int(config.get("embedding_recovery",{}).get("num_workers",-1))==0 and config.get("embedding_recovery",{}).get("prefetching") is False,
    }

def _write(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); pq.write_table(pa.Table.from_pylist(rows),tmp); tmp.replace(path)

def assemble_union(config):
    _,phase,sources,_,_=phase_contract(config); manifest=json.loads(Path(phase["manifest"]).read_text(encoding="utf-8")); target={int(x) for s in sources for x in manifest["sources"][s]["target_ids"]}
    docs={}
    for path in [*phase["previous_documents"],phase["new_documents"]]:
        for row in pq.read_table(path).to_pylist():
            if int(row["doc_id"]) in target: docs.setdefault(int(row["doc_id"]),row)
    chunks={}
    for path in [phase["previous_chunks"],*phase.get("supplemental_previous_chunks",[]),phase["new_chunks"]]:
        for row in pq.read_table(path).to_pylist():
            if int(row["doc_id"]) in target: chunks.setdefault(str(row["chunk_id"]),row)
    docrows=sorted(docs.values(),key=lambda r:int(r["doc_id"])); chunkrows=sorted(chunks.values(),key=lambda r:(int(r["doc_id"]),int(r["chunk_index"]),str(r["chunk_id"])))
    _write(config["outputs"]["documents"],docrows); _write(config["outputs"]["chunks"],chunkrows)
    per={}
    for source in sources:
        ids=set(manifest["sources"][source]["target_ids"]); sd=[r for r in docrows if int(r["doc_id"]) in ids]; sc=[r for r in chunkrows if int(r["doc_id"]) in ids]
        per[source]={"target_depth":len(ids),"documents_present":len(sd),"usable_documents":sum(r["extraction_status"]=="SUCCESS" and bool(str(r["normalized_text"] or "").strip()) for r in sd),"chunk_count":len(sc),"status_counts":dict(Counter(str(r["extraction_status"]) for r in sd))}
    triage={"healthy_sources":sources,"sources":{s:{"corpus_rows":manifest["sources"][s]["official_population"],"sampled":per[s]["target_depth"],"extracted_rows":per[s]["documents_present"],"usable_documents":per[s]["usable_documents"],"likely_language":"zh" if s.endswith(".cn") or s=="zydcd.com" else "vi","healthy":True,"technical_only":True} for s in sources}}
    atomic_json(Path(config["outputs"]["artifacts"])/"technical_triage.json",triage)
    result={"target_official_ids":len(target),"documents":len(docrows),"chunks":len(chunkrows),"sources":per}; atomic_json(Path(config["outputs"]["artifacts"])/"corpus_assembly.json",result); return result

def verify_union(config):
    """Prove candidate inputs are the old/new union, grouped by G5A source."""
    _,phase,sources,_,_=phase_contract(config)
    def usable(path):
        result={s:set() for s in sources}
        for row in pq.read_table(path,columns=["doc_id","original_url","extraction_status","normalized_text"]).to_pylist():
            source=host(row["original_url"])
            if source in result and row["extraction_status"]=="SUCCESS" and str(row["normalized_text"] or "").strip(): result[source].add(int(row["doc_id"]))
        return result
    old={s:set() for s in sources}
    for path in phase["previous_documents"]:
        values=usable(path)
        for source in sources: old[source].update(values[source])
    new=usable(phase["new_documents"]); final=usable(config["outputs"]["documents"])
    old_chunk_rows=[]
    for path in [phase["previous_chunks"],*phase.get("supplemental_previous_chunks",[])]:
        old_chunk_rows.extend(pq.read_table(path,columns=["doc_id","chunk_id"]).to_pylist())
    new_chunk_rows=pq.read_table(phase["new_chunks"],columns=["doc_id","chunk_id"]).to_pylist()
    chunk_rows=pq.read_table(config["outputs"]["chunks"],columns=["doc_id","chunk_id"]).to_pylist()
    old_chunk_ids={s:{str(row["chunk_id"]) for row in old_chunk_rows if int(row["doc_id"]) in old[s]} for s in sources}
    new_chunk_ids={s:{str(row["chunk_id"]) for row in new_chunk_rows if int(row["doc_id"]) in new[s]} for s in sources}
    final_chunk_ids={s:{str(row["chunk_id"]) for row in chunk_rows if int(row["doc_id"]) in final[s]} for s in sources}
    source_rows={}
    for source in sources:
        union=old[source]|new[source]
        if final[source] != union:
            raise ValueError(f"Phase10E searchable documents are not the old/new union for {source}")
        source_rows[source]={
            "previous_usable_doc_ids":len(old[source]),"new_usable_doc_ids":len(new[source]),
            "duplicate_overlap":len(old[source]&new[source]),"union_usable_doc_ids":len(union),
            "final_searchable_doc_count":len(final[source]),
            "previous_chunk_count":len(old_chunk_ids[source]),"new_chunk_count":len(new_chunk_ids[source]),
            "duplicate_chunk_overlap":len(old_chunk_ids[source]&new_chunk_ids[source]),
            "union_chunk_count":len(old_chunk_ids[source]|new_chunk_ids[source]),
            "final_searchable_chunk_count":len(final_chunk_ids[source]),
        }
        if final_chunk_ids[source] != old_chunk_ids[source] | new_chunk_ids[source]:
            raise ValueError(f"Phase10E searchable chunks are not the old/new union for {source}")
    result={"passed":True,"candidate_documents":config["outputs"]["documents"],"candidate_chunks":config["outputs"]["chunks"],"sources":source_rows,"new_documents_total":sum(len(x) for x in new.values())}
    atomic_json(Path(config["outputs"]["artifacts"])/"union_accounting.json",result)
    return result

def assemble_embeddings(config,state=None):
    summary=Path(config["outputs"]["artifacts"])/"embedding_summary.json"
    if summary.exists() and (Path(config["outputs"]["root"])/"round1/chunk_embeddings.f32").exists(): return json.loads(summary.read_text(encoding="utf-8"))
    command=[sys.executable,"tools/phase10e_embed_streaming.py","prepare","--config",config["_config_path"]]
    subprocess.run(command,cwd=Path.cwd(),check=True)
    command=[sys.executable,"tools/phase10e_embed_streaming.py","encode","--config",config["_config_path"]]
    if state is not None: command.extend(["--run-state",str(state)])
    subprocess.run(command,cwd=Path.cwd(),check=True)
    return json.loads(summary.read_text(encoding="utf-8"))

def _text_sha256(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

def ensure_reranker_equivalence_control(config, sample_size=8):
    """Build/verify a deterministic exact-reuse control for this experiment."""
    _,phase,_,_,experiment=phase_contract(config)
    rule=config["models"]["reranker"]["equivalence"]
    destination=Path(config["outputs"]["root"])/"round1/source_scores.sqlite"
    source=Path(phase["previous_score_database"])
    investigation=json.loads(Path(rule["investigation_artifact"]).read_text(encoding="utf-8"))
    dst=sqlite3.connect(f"file:{destination.resolve()}?mode=ro",uri=True)
    src=sqlite3.connect(f"file:{source.resolve()}?mode=ro",uri=True)
    selected=[]
    for query_id,chunk_id,query_text,chunk_text,score in dst.execute(
        "SELECT query_id,chunk_id,query_text,chunk_text,rerank_score FROM pairs "
        "WHERE rerank_score IS NOT NULL ORDER BY query_id,chunk_id"
    ):
        old=src.execute(
            "SELECT rerank_score FROM pairs WHERE query_id=? AND chunk_id=?",
            (int(query_id),str(chunk_id)),
        ).fetchone()
        if old is None or float(old[0])!=float(score):
            continue
        selected.append({
            "query_id":int(query_id),"chunk_id":str(chunk_id),
            "query_text_sha256":_text_sha256(query_text),
            "chunk_text_sha256":_text_sha256(chunk_text),
            "reference_score":float(score),
        })
        if len(selected)==sample_size:
            break
    dst.close(); src.close()
    if len(selected)!=sample_size:
        raise RuntimeError(f"insufficient exact-reuse pairs for reranker control: {len(selected)}/{sample_size}")
    result={
        "format_version":1,"experiment":experiment,
        "selection_rule":"first exact reused (query_id, chunk_id) keys in ascending deterministic order",
        "source_score_database":str(source),"destination_score_database":str(destination),
        "investigation_artifact":str(rule["investigation_artifact"]),
        "contract":investigation["contract"],"pairs":selected,
        "exact_reuse_verified":True,
    }
    path=Path(rule["control_artifact"])
    if path.exists():
        existing=json.loads(path.read_text(encoding="utf-8"))
        if existing!=result:
            raise RuntimeError(f"reranker equivalence control drift: {path}")
    else:
        atomic_json(path,result)
    return result

def seed_scores(config):
    _,phase,_,_,_=phase_contract(config); result=seed_exact_scores(Path(config["outputs"]["root"])/"round1/source_scores.sqlite",phase["previous_score_database"]); atomic_json(Path(config["outputs"]["artifacts"])/"rerank_seed_summary.json",result); ensure_reranker_equivalence_control(config); return result

def finalize_report(config, submission, assembly, embedding, candidate_summary, seed, audit):
    """Write the durable Phase 10E report from already-produced stage evidence."""
    _,phase,_,group,experiment=phase_contract(config); artifacts=Path(config["outputs"]["artifacts"])
    manifest=json.loads(Path(phase["manifest"]).read_text(encoding="utf-8"))
    acquisition=json.loads((artifacts/"parallel_acquisition.json").read_text(encoding="utf-8"))
    extraction=json.loads((artifacts/"extract.json").read_text(encoding="utf-8"))
    chunking=json.loads((artifacts/"chunk.json").read_text(encoding="utf-8"))
    report={
        "experiment":experiment,
        "scientific_question":config.get("focused_scaling",{}).get("scientific_question","Does organizer-relevant source-family signal continue beyond about 1000 official documents per source?"),
        "intended_variable":f"{group} source acquisition depth only",
        "manifest":manifest,
        "acquisition":acquisition,
        "extraction":extraction,
        "chunking":chunking,
        "corpus_assembly":assembly,
        "embeddings":embedding,
        "candidate_cache":candidate_summary,
        "rerank_seed":seed,
        "submission":submission,
        "pre_submission_audit":audit,
        "status":audit["status"],
        "future_checkpoint_prepared_only":phase.get("future_targets"),
    }
    path=artifacts.parent/f"{experiment}_report.json"
    atomic_json(path,report)
    return report
