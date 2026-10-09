from __future__ import annotations
import gc,json,shutil,sqlite3,subprocess,sys,time
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
    result={"format_version":1,"experiment":experiment,"selection":"all prior successful official IDs plus lowest SHA-256(seed:doc_id) ranked unattempted official IDs; prior failures are not recrawled","sources":sources,"total_incremental_ids":total,"free_disk_before_bytes":free,"free_disk_before_gib":free/2**30,"projected_retained_bytes":round(projected),"projected_retained_gib":projected/2**30,"expected_free_disk_after_gib":(free-projected)/2**30,"minimum_free_disk_gib":20.0,"disk_gate_passed":free-projected>=20*2**30,"projected_concurrent_crawl_minutes":crawl_minutes,"projected_end_to_end_hours":crawl_minutes/60+1.75,"future_checkpoint_prepared_only":phase.get("future_targets")}
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
        config["models"]["reranker"]["equivalence"]["investigation_artifact"],
        config["focused_scaling"]["control_submission_json"],
        *phase["previous_crawl_databases"],*phase["previous_documents"],
    ]
    missing=[str(p) for p in required if not Path(p).exists()]
    calibration={}
    calibration_path=Path(config["candidate_cache"]["calibrated_cap_artifact"])
    if calibration_path.exists(): calibration=json.loads(calibration_path.read_text(encoding="utf-8"))
    chosen=int(calibration.get("chosen_m",-1))
    contract=preflight_contract_checks(config,chosen,sources)
    manifest=prepare_manifest(config,enforce_disk_gate=False) if not missing and all(contract.values()) else None
    writable=[]
    for directory in [Path(config["outputs"]["root"]),Path(config["outputs"]["artifacts"]),Path(config["outputs"]["submissions"]),Path(config["focused_scaling"]["audit_path"]).parent]:
        try:
            directory.mkdir(parents=True,exist_ok=True); probe=directory/".phase10e_preflight_write"; probe.write_text("ok",encoding="utf-8"); probe.unlink(); writable.append(str(directory))
        except OSError as exc: missing.append(f"writable:{directory}:{exc}")
    failures=[]
    if missing: failures.append("missing/unwritable dependencies: "+", ".join(missing))
    if not all(contract.values()): failures.append("fixed experiment contract failed: "+", ".join(k for k,v in contract.items() if not v))
    if manifest and not manifest["disk_gate_passed"]: failures.append("20 GiB disk gate failed")
    result={"experiment":experiment,"group":group,"passed":not failures,"failures":failures,"chosen_m":chosen,"calibration_artifact":str(calibration_path),"contract":contract,"dependencies_checked":len(required),"writable_directories":writable,"manifest":phase["manifest"],"disk_gate_passed":bool(manifest and manifest["disk_gate_passed"])}
    atomic_json(Path(config["outputs"]["artifacts"])/"preflight.json",result)
    if failures: raise RuntimeError("Phase10E preflight failed: "+"; ".join(failures))
    return result

def preflight_contract_checks(config,chosen,sources):
    _,phase,_,_,_=phase_contract(config)
    return {
        "chosen_m_is_8":chosen==8 and list(config["candidate_cache"]["candidate_caps"])==[8],
        "reranker_revision_fixed":config["models"]["reranker"]["revision"]=="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "reranker_batch_2":int(config["models"]["reranker"]["batch_size"])==2,
        "reranker_max_length_512":int(config["models"]["reranker"]["max_length"])==512,
        "submission_top10_top20":int(config["submission"]["documents_per_query"])==10 and int(config["submission"]["chunks_per_query"])==20,
        "source_targets_match":set(sources)==set(phase["targets"]),
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
    for path in [phase["previous_chunks"],phase["new_chunks"]]:
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
    old_chunk_rows=pq.read_table(phase["previous_chunks"],columns=["doc_id","chunk_id"]).to_pylist()
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

def seed_scores(config):
    _,phase,_,_,_=phase_contract(config); result=seed_exact_scores(Path(config["outputs"]["root"])/"round1/source_scores.sqlite",phase["previous_score_database"]); atomic_json(Path(config["outputs"]["artifacts"])/"rerank_seed_summary.json",result); return result

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
