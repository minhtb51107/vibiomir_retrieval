from __future__ import annotations
import gc,json,shutil,sqlite3,time
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

def _crawl_sets(paths):
    result={s:{"attempted":set(),"success":set()} for s in G5A}
    for path in paths:
        connection=sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro",uri=True)
        for doc_id,url,status in connection.execute("SELECT doc_id,original_url,status FROM crawl_results"):
            source=host(url)
            if source in result:
                result[source]["attempted"].add(int(doc_id))
                if status=="SUCCESS": result[source]["success"].add(int(doc_id))
        connection.close()
    return result

def prepare_manifest(config):
    phase=config["phase10e"]; targets={k:int(v) for k,v in phase["targets"].items()}
    crawl=_crawl_sets(phase["previous_crawl_databases"])
    usable={s:set() for s in G5A}
    for path in phase["previous_documents"]:
        for row in pq.read_table(path,columns=["doc_id","original_url","extraction_status","normalized_text"]).to_pylist():
            source=host(row["original_url"])
            if source in usable and row["extraction_status"]=="SUCCESS" and str(row["normalized_text"] or "").strip(): usable[source].add(int(row["doc_id"]))
    samples,populations=deterministic_domain_samples(config["inputs"]["corpus"],domains=set(G5A),sample_size=12000,seed=phase["sampling_seed"])
    sources={}; total=0
    for source in G5A:
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
    rates={"benhviennhitrunguong.gov.vn":59.27,"zydcd.com":39.41,"hellobacsi.com":55.84}
    crawl_minutes=max(sources[s]["new_ids_required"]/rates[s] for s in G5A)
    result={"format_version":1,"experiment":"phase10e_g5a_11200","selection":"all prior successful official IDs plus lowest SHA-256(seed:doc_id) ranked unattempted official IDs; prior failures are not recrawled","sources":sources,"total_incremental_ids":total,"free_disk_before_bytes":free,"free_disk_before_gib":free/2**30,"projected_retained_bytes":round(projected),"projected_retained_gib":projected/2**30,"expected_free_disk_after_gib":(free-projected)/2**30,"minimum_free_disk_gib":20.0,"disk_gate_passed":free-projected>=20*2**30,"projected_concurrent_crawl_minutes":crawl_minutes,"projected_end_to_end_hours":crawl_minutes/60+1.75,"future_checkpoint_prepared_only":phase["future_targets"]}
    atomic_json(Path(phase["manifest"]),result)
    if not result["disk_gate_passed"]: raise RuntimeError("Phase10E violates 20 GiB disk floor")
    return result

def _write(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); pq.write_table(pa.Table.from_pylist(rows),tmp); tmp.replace(path)

def assemble_union(config):
    phase=config["phase10e"]; manifest=json.loads(Path(phase["manifest"]).read_text(encoding="utf-8")); target={int(x) for s in G5A for x in manifest["sources"][s]["target_ids"]}
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
    for source in G5A:
        ids=set(manifest["sources"][source]["target_ids"]); sd=[r for r in docrows if int(r["doc_id"]) in ids]; sc=[r for r in chunkrows if int(r["doc_id"]) in ids]
        per[source]={"target_depth":len(ids),"documents_present":len(sd),"usable_documents":sum(r["extraction_status"]=="SUCCESS" and bool(str(r["normalized_text"] or "").strip()) for r in sd),"chunk_count":len(sc),"status_counts":dict(Counter(str(r["extraction_status"]) for r in sd))}
    triage={"healthy_sources":G5A,"sources":{s:{"corpus_rows":manifest["sources"][s]["official_population"],"sampled":per[s]["target_depth"],"extracted_rows":per[s]["documents_present"],"usable_documents":per[s]["usable_documents"],"likely_language":"zh" if s=="zydcd.com" else "vi","healthy":True,"technical_only":True} for s in G5A}}
    atomic_json(Path(config["outputs"]["artifacts"])/"technical_triage.json",triage)
    result={"target_official_ids":len(target),"documents":len(docrows),"chunks":len(chunkrows),"sources":per}; atomic_json(Path(config["outputs"]["artifacts"])/"corpus_assembly.json",result); return result

def verify_union(config):
    """Prove candidate inputs are the old/new union, grouped by G5A source."""
    phase=config["phase10e"]
    def usable(path):
        result={s:set() for s in G5A}
        for row in pq.read_table(path,columns=["doc_id","original_url","extraction_status","normalized_text"]).to_pylist():
            source=host(row["original_url"])
            if source in result and row["extraction_status"]=="SUCCESS" and str(row["normalized_text"] or "").strip(): result[source].add(int(row["doc_id"]))
        return result
    old={s:set() for s in G5A}
    for path in phase["previous_documents"]:
        values=usable(path)
        for source in G5A: old[source].update(values[source])
    new=usable(phase["new_documents"]); final=usable(config["outputs"]["documents"])
    old_chunk_rows=pq.read_table(phase["previous_chunks"],columns=["doc_id","chunk_id"]).to_pylist()
    new_chunk_rows=pq.read_table(phase["new_chunks"],columns=["doc_id","chunk_id"]).to_pylist()
    chunk_rows=pq.read_table(config["outputs"]["chunks"],columns=["doc_id","chunk_id"]).to_pylist()
    old_chunk_ids={s:{str(row["chunk_id"]) for row in old_chunk_rows if int(row["doc_id"]) in old[s]} for s in G5A}
    new_chunk_ids={s:{str(row["chunk_id"]) for row in new_chunk_rows if int(row["doc_id"]) in new[s]} for s in G5A}
    final_chunk_ids={s:{str(row["chunk_id"]) for row in chunk_rows if int(row["doc_id"]) in final[s]} for s in G5A}
    sources={}
    for source in G5A:
        union=old[source]|new[source]
        if final[source] != union:
            raise ValueError(f"Phase10E searchable documents are not the old/new union for {source}")
        sources[source]={
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
    result={"passed":True,"candidate_documents":config["outputs"]["documents"],"candidate_chunks":config["outputs"]["chunks"],"sources":sources,"new_documents_total":sum(len(x) for x in new.values())}
    atomic_json(Path(config["outputs"]["artifacts"])/"union_accounting.json",result)
    return result

def assemble_embeddings(config,state=None):
    phase=config["phase10e"]; chunks=pq.read_table(config["outputs"]["chunks"],columns=["chunk_id","normalized_text"]).to_pylist(); dim=int(config["models"]["embedder"]["dimension"]); output=Path(config["outputs"]["root"])/"round1/chunk_embeddings.f32"; output.parent.mkdir(parents=True,exist_ok=True)
    oldchunks=pq.read_table(phase["previous_chunks"],columns=["chunk_id"]).column(0).to_pylist(); oldvec=np.memmap(phase["previous_embeddings"],dtype=np.float32,mode="r",shape=(len(oldchunks),dim)); lookup={str(c):i for i,c in enumerate(oldchunks)}
    tmp=output.with_suffix('.f32.tmp'); vectors=np.memmap(tmp,dtype=np.float32,mode='w+',shape=(len(chunks),dim)); missing=[]; reused=0
    for i,row in enumerate(chunks):
        j=lookup.get(str(row["chunk_id"]));
        if j is None: missing.append(i)
        else: vectors[i]=oldvec[j]; reused+=1
    del oldvec,lookup
    started=time.perf_counter()
    if missing:
        import torch
        from sentence_transformers import SentenceTransformer
        m=config["models"]["embedder"]; model=SentenceTransformer(m["name"],revision=m["revision"],device='cpu'); model.max_seq_length=int(m["max_length"]); model.half(); model.to('cuda')
        for off in range(0,len(missing),256):
            ids=missing[off:off+256]; encoded=model.encode([normalize_text(str(chunks[i]["normalized_text"])) for i in ids],batch_size=int(m["batch_size"]),show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=True); vectors[ids]=np.asarray(encoded,dtype=np.float32); vectors.flush()
            if state: atomic_json(state,{"stage":"EMBEDDINGS","status":"RUNNING","total":len(missing),"done":min(off+len(ids),len(missing)),"reused_embeddings":reused})
        del model; torch.cuda.empty_cache()
    vectors.flush(); del vectors; tmp.replace(output); result={"chunks":len(chunks),"reused_embeddings":reused,"new_embeddings":len(missing),"runtime_seconds":time.perf_counter()-started,"path":str(output)}; atomic_json(Path(config["outputs"]["artifacts"])/"embedding_summary.json",result); gc.collect(); return result

def seed_scores(config):
    result=seed_exact_scores(Path(config["outputs"]["root"])/"round1/source_scores.sqlite",config["phase10e"]["previous_score_database"]); atomic_json(Path(config["outputs"]["artifacts"])/"rerank_seed_summary.json",result); return result

def finalize_report(config, submission, assembly, embedding, candidate_summary, seed, audit):
    """Write the durable Phase 10E report from already-produced stage evidence."""
    phase=config["phase10e"]; artifacts=Path(config["outputs"]["artifacts"])
    manifest=json.loads(Path(phase["manifest"]).read_text(encoding="utf-8"))
    acquisition=json.loads((artifacts/"parallel_acquisition.json").read_text(encoding="utf-8"))
    extraction=json.loads((artifacts/"extract.json").read_text(encoding="utf-8"))
    chunking=json.loads((artifacts/"chunk.json").read_text(encoding="utf-8"))
    report={
        "experiment":"phase10e_g5a_11200",
        "scientific_question":"Does organizer-relevant G5A signal continue beyond about 1000 official documents per source?",
        "intended_variable":"G5A source acquisition depth only",
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
        "future_checkpoint_prepared_only":phase["future_targets"],
    }
    path=artifacts.parent/"phase10e_g5a_11200_report.json"
    atomic_json(path,report)
    return report
