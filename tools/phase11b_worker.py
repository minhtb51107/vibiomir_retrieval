#!/usr/bin/env python3
"""Resumable Phase 11B new-corpus control and document-depth curve worker."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.source_census.cached_subsets import _query_groups, build_cached_subset_rankings
from src.source_census.pipeline import atomic_json, build_source_candidates, load_config
from src.source_census.union_experiment import _audit_ranked_provenance_closure, _build_canonical_store, _filter_parquet, _iter_json_array, _official_id_audit, _stream_validate
from src.submission.common import file_sha256, write_deterministic_zip, write_submission_stream
from src.submission.validator import expected_query_ids, normalize_validation_result
from src.validation.pre_submission import audit_score_cache
from tools.phase11a_bottleneck import _copy_or_link, _init_score_db, _seed_exact, host, tokenizer
from tools.phase11b_new_corpus import audit as migration_audit, corpus_arrays

PARENTS=[
    ("med",Path("data/source_census/phase10f_medlatec_10000"),{"medlatec.vn"},"configs/phase10f_medlatec_10000.yaml"),
    ("g1",Path("data/source_census/phase10e_g1a_10010"),{"v.familydoctor.com.cn","vinmec.com"},"configs/phase10e_g1a_10010.yaml"),
    ("g6",Path("data/source_census/phase10e_g6b_15000"),{"tiemchunglongchau.com.vn","cancer.39.net","suckhoedoisong.vn"},"configs/phase10e_g6b_15000.yaml"),
    ("g5",Path("data/source_census/phase10e_g5a_11200"),{"benhviennhitrunguong.gov.vn","zydcd.com","hellobacsi.com"},"configs/phase10e_g5a_11200.yaml"),
]
UNAFFECTED={"medlatec.vn","tiemchunglongchau.com.vn","suckhoedoisong.vn","benhviennhitrunguong.gov.vn","hellobacsi.com"}
RECOMPUTE={"cancer.39.net","zydcd.com"}


def state(path:Path,**fields)->None:
    try: current=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception: current={}
    current.update(fields);atomic_json(path,current)


def source_slug(source:str)->str:
    return "".join(ch if ch.isalnum() else "_" for ch in source).strip("_")


def official_ids(path:Path)->np.ndarray:
    values,_,_=corpus_arrays(path);return np.sort(values)


def contains(sorted_ids:np.ndarray,value:int)->bool:
    position=int(np.searchsorted(sorted_ids,value));return position<len(sorted_ids) and int(sorted_ids[position])==value


def retained_doc_ids(parent:Path,sources:set[str],ids:np.ndarray)->set[int]:
    selected=set()
    for batch in pq.ParquetFile(parent/"documents.parquet").iter_batches(columns=["doc_id","original_url","extraction_status","normalized_text"],batch_size=8192):
        for doc_id,url,status,text in zip(*(column.to_pylist() for column in batch.columns),strict=True):
            if host(str(url)) in sources and status=="SUCCESS" and str(text or "").strip() and contains(ids,int(doc_id)): selected.add(int(doc_id))
    return selected


def chunk_ids_for_documents(path:Path,documents:set[int])->set[str]:
    selected=set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["chunk_id","doc_id"],batch_size=65536):
        for chunk_id,doc_id in zip(batch.column(0).to_pylist(),batch.column(1).to_pylist(),strict=True):
            if int(doc_id) in documents:selected.add(str(chunk_id))
    return selected


def filter_embeddings(parent:Path,selected_docs:set[int],destination:Path,dimension:int)->dict:
    chunks=parent/"chunks.parquet";source=parent/"round1/chunk_embeddings.f32"
    total=pq.ParquetFile(chunks).metadata.num_rows; retained=0
    for batch in pq.ParquetFile(chunks).iter_batches(columns=["doc_id"],batch_size=65536): retained+=sum(int(x) in selected_docs for x in batch.column(0).to_pylist())
    expected=retained*dimension*4
    if destination.exists() and destination.stat().st_size==expected:return {"rows":retained,"reused":True}
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix(".tmp.f32")
    src=np.memmap(source,dtype=np.float32,mode="r",shape=(total,dimension));dst=np.memmap(temporary,dtype=np.float32,mode="w+",shape=(retained,dimension))
    input_offset=output_offset=0
    for batch in pq.ParquetFile(chunks).iter_batches(columns=["doc_id"],batch_size=2048):
        mask=np.asarray([int(x) in selected_docs for x in batch.column(0).to_pylist()],dtype=bool);count=int(mask.sum())
        if count: dst[output_offset:output_offset+count]=src[input_offset:input_offset+len(mask)][mask];output_offset+=count
        input_offset+=len(mask)
    dst.flush();del dst,src
    if output_offset!=retained:raise ValueError(f"embedding filter mismatch {output_offset}!={retained}")
    os.replace(temporary,destination);return {"rows":retained,"reused":False}


def prepare_source(source:str,parent:Path,config_path:str,ids:np.ndarray,root:Path,run_state:Path)->Path:
    work=root/"recomputed"/source_slug(source);docs=work/"documents.parquet";chunks=work/"chunks.parquet";artifacts=work/"artifacts";round1=work/"round1"
    selected=retained_doc_ids(parent,{source},ids)
    if not selected:raise ValueError(f"no retained official documents for {source}")
    if not docs.exists():_filter_parquet([parent/"documents.parquet"],docs,"doc_id",selected)
    selected_chunks=chunk_ids_for_documents(parent/"chunks.parquet",selected)
    if not chunks.exists() or pq.ParquetFile(chunks).metadata.num_rows!=len(selected_chunks):
        _filter_parquet([parent/"chunks.parquet"],chunks,"chunk_id",selected_chunks)
    config=load_config(config_path);dimension=int(config["models"]["embedder"]["dimension"])
    embedding=filter_embeddings(parent,selected,round1/"chunk_embeddings.f32",dimension)
    artifacts.mkdir(parents=True,exist_ok=True);atomic_json(artifacts/"technical_triage.json",{"healthy_sources":[source]});atomic_json(artifacts/"candidate_cap_calibration.json",{"chosen_m":16})
    config["inputs"]["corpus"]="data/raw/links_corpus_vi_en_zh.parquet"
    config["candidate_cache"].update({"candidate_caps":[16],"dense_depth":16,"sparse_depth":16,"calibrated_cap_artifact":str(artifacts/"candidate_cap_calibration.json")})
    config["outputs"].update({"root":str(work),"artifacts":str(artifacts),"documents":str(docs),"chunks":str(chunks)})
    build_source_candidates(config,run_state=run_state,groups={"NEW":[source]})
    marker=next((round1/"source_candidate_parts").glob("*.json"));part=marker.with_suffix(".parquet")
    atomic_json(artifacts/"retained_source.json",{"source":source,"documents":len(selected),"chunks":embedding["rows"],"embedding_reused":embedding["reused"]})
    return part


def prepare_candidates(config:dict,run_state:Path)->tuple[Path,list[str],dict]:
    output=ROOT/config["outputs"]["root"];parts=output/"round1/source_candidate_parts";parts.mkdir(parents=True,exist_ok=True)
    ids=official_ids(ROOT/config["inputs"]["official_corpus"]);old_parts=ROOT/config["inputs"]["m16_root"]/"round1/source_candidate_parts"
    for source in UNAFFECTED:
        slug=source_slug(source);_copy_or_link(old_parts/f"{slug}.parquet",parts/f"{slug}.parquet");_copy_or_link(old_parts/f"{slug}.json",parts/f"{slug}.json")
    # Removed sources must never survive a restart from an older candidate directory.
    for source in {"vinmec.com","v.familydoctor.com.cn"}:
        for suffix in (".json",".parquet"):(parts/f"{source_slug(source)}{suffix}").unlink(missing_ok=True)
    parent_by_source={source:(parent,cfg) for _,parent,sources,cfg in PARENTS for source in sources}
    for source in sorted(RECOMPUTE):
        part=prepare_source(source,*parent_by_source[source],ids,output,run_state);marker=part.with_suffix(".json")
        shutil.copy2(part,parts/part.name);shutil.copy2(marker,parts/marker.name)
    sources=sorted(UNAFFECTED|RECOMPUTE);expected=len(sources)*1200*16;db=output/"round1/source_scores.sqlite"
    valid=False
    if db.exists():
        connection=sqlite3.connect(db);valid=int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])==expected;connection.close()
    if not valid:_init_score_db(parts,db)
    seed_path=ROOT/config["outputs"]["artifacts"]/"score_seed_summary.json"
    if valid and seed_path.exists():seed=json.loads(seed_path.read_text(encoding="utf-8"))
    else:
        caches=[ROOT/config["inputs"]["m16_root"]/"round1/source_scores.sqlite",*(ROOT/parent/"round1/source_scores.sqlite" for _,parent,_,_ in PARENTS)]
        seed=_seed_exact(db,caches);atomic_json(seed_path,seed)
    state(run_state,stage="RERANK",status="RUNNING",total=seed["total"],done=seed["reused"],remaining=seed["missing"],exact_scores_reused=seed["reused"],integrity=seed["sqlite_integrity"],checkpoint=str(seed_path))
    return output,sources,seed


def runtime_config(config:dict,output:Path)->Path:
    runtime=load_config("configs/phase10f_medlatec_10000.yaml");runtime["inputs"]["corpus"]=config["inputs"]["official_corpus"];runtime["outputs"]["root"]=str(output);runtime["outputs"]["artifacts"]=config["outputs"]["artifacts"]
    runtime["candidate_cache"].update({"candidate_caps":[16],"dense_depth":16,"sparse_depth":16})
    path=output/"runtime.yaml";path.write_text(yaml.safe_dump(runtime,sort_keys=False,allow_unicode=True),encoding="utf-8");return path


def prepare_official_pilot_pool(config:dict,output:Path,ids:np.ndarray)->tuple[Path,dict]:
    """Persist an experiment-scoped view of the frozen pilot pool for NEW official IDs."""
    source=ROOT/load_config("configs/phase10f_medlatec_10000.yaml")["inputs"]["pilot_rerank_pool"]
    destination=output/"round1/pilot_rerank_pool.new_official.parquet"
    report_path=ROOT/config["outputs"]["artifacts"]/"pilot_reconciliation.json"
    if destination.exists() and report_path.exists():
        report=json.loads(report_path.read_text(encoding="utf-8"))
        if pq.ParquetFile(destination).metadata.num_rows==int(report["retained_rows"]):
            return destination,report
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix(".parquet.tmp");temporary.unlink(missing_ok=True)
    parquet=pq.ParquetFile(source);writer=pq.ParquetWriter(temporary,parquet.schema_arrow)
    total=retained=0;all_docs=set();retained_docs=set();removed_chunks=set();removed_docs=set();queries=set()
    try:
        for batch in parquet.iter_batches(batch_size=4096):
            table=pa.Table.from_batches([batch]);doc_values=list(map(int,table["doc_id"].to_pylist()))
            mask=np.asarray([contains(ids,value) for value in doc_values],dtype=bool)
            total+=len(mask);all_docs.update(doc_values)
            if mask.any():
                selected=table.filter(pa.array(mask));writer.write_table(selected);retained+=selected.num_rows
                retained_docs.update(map(int,selected["doc_id"].to_pylist()));queries.update(map(int,selected["query_id"].to_pylist()))
            if (~mask).any():
                removed=table.filter(pa.array(~mask));removed_docs.update(map(int,removed["doc_id"].to_pylist()));removed_chunks.update(map(str,removed["chunk_id"].to_pylist()))
    finally:
        writer.close()
    if queries!=set(map(int,pq.read_table(ROOT/config["inputs"]["queries"],columns=["id"]).column(0).to_pylist())):
        temporary.unlink(missing_ok=True);raise ValueError("NEW-official pilot pool no longer covers every query")
    os.replace(temporary,destination)
    report={"source_pool":str(source),"filtered_pool":str(destination),"total_rows":total,"retained_rows":retained,"removed_rows":total-retained,"total_doc_ids":len(all_docs),"retained_doc_ids":len(retained_docs),"removed_doc_ids":len(removed_docs),"removed_unique_chunk_ids":len(removed_chunks),"queries":len(queries),"new_model_inference":0}
    atomic_json(report_path,report);return destination,report


def score(config:dict,output:Path,run_state:Path)->None:
    db=sqlite3.connect(output/"round1/source_scores.sqlite");missing=int(db.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NULL").fetchone()[0]);db.close()
    if not missing:return
    subprocess.run([sys.executable,"tools/score_source_census_persistent.py","--config",str(runtime_config(config,output)),"--window-size","2000","--run-state",str(run_state),"--metrics-out",str(ROOT/config["outputs"]["artifacts"]/"rerank_metrics.json")],cwd=ROOT,check=True)


def canonical(config:dict,output:Path,rankings:Path)->tuple[Path,Path,dict]:
    chunk_ids=set();doc_ids=set()
    for batch in pq.ParquetFile(rankings/"NEW/reranked_chunks.parquet").iter_batches(columns=["chunk_id","doc_id"],batch_size=8192):
        chunk_ids.update(map(str,batch.column(0).to_pylist()));doc_ids.update(map(int,batch.column(1).to_pylist()))
    base=load_config("configs/phase10f_medlatec_10000.yaml")
    broad_docs=ROOT/config["inputs"]["broad_documents"];broad_chunks=ROOT/config["inputs"]["broad_chunks"]
    pilot_docs=ROOT/base["inputs"]["pilot_documents"];pilot_chunks=ROOT/base["inputs"]["pilot_chunks"]
    docs=output/"round1/combined_documents.parquet";chunks=output/"round1/combined_chunks.parquet"
    doc_result=_filter_parquet([pilot_docs,broad_docs],docs,"doc_id",doc_ids)
    chunk_result=_filter_parquet([pilot_chunks,broad_chunks],chunks,"chunk_id",chunk_ids)
    per_source=Counter()
    for batch in pq.ParquetFile(docs).iter_batches(columns=["original_url"],batch_size=8192):
        per_source.update(host(str(url)) for url in batch.column(0).to_pylist())
    return docs,chunks,{"documents":len(doc_ids),"chunks":len(chunk_ids),"sources":dict(per_source),"document_filter":doc_result,"chunk_filter":chunk_result}


def _depth_records(control_json:Path,documents_path:Path,depth:int):
    document_groups=iter(_query_groups(documents_path))
    for control in _iter_json_array(control_json):
        query_id=int(control["id"]);rows=next(document_groups,None)
        if not rows or int(rows[0]["query_id"])!=query_id:raise ValueError(f"missing document ranking for query {query_id}")
        rows.sort(key=lambda row:(int(row["rank"]),int(row["doc_id"])))
        docs=[];seen=set()
        for row in rows:
            doc_id=int(row["doc_id"])
            if doc_id not in seen:docs.append(doc_id);seen.add(doc_id)
            if len(docs)==depth:break
        if len(docs)!=depth:raise ValueError(f"query {query_id} supports only {len(docs)} documents, requested {depth}")
        yield {"id":query_id,"relevant_docs":docs,"relevant_chunks":control["relevant_chunks"]}
    if next(document_groups,None) is not None:raise ValueError("document ranking contains unexpected queries")


def _package_depth_variant(runtime:dict,rankings:Path,control_json:Path,name:str,depth:int,output_dir:Path,canonical_db:Path)->dict:
    output_dir.mkdir(parents=True,exist_ok=True);json_path=output_dir/f"{name}.json";zip_path=output_dir/f"{name}.zip"
    query_ids=expected_query_ids(runtime["inputs"]["queries"]);documents=rankings/"NEW/reranked_documents.parquet"
    first=None;validation=None
    for _ in range(2):
        write_submission_stream(json_path,_depth_records(control_json,documents,depth));write_deterministic_zip(json_path,zip_path)
        current=(file_sha256(json_path),file_sha256(zip_path))
        validation=normalize_validation_result(_stream_validate(json_path,zip_path,query_ids,canonical_db,document_depth=depth,chunk_depth=20))
        if not validation["passed"]:raise ValueError(f"submission validation failed: {validation}")
        if first is None:first=current
        elif current!=first:raise RuntimeError(f"submission regeneration was not deterministic: {name}")
    return {"name":name,"json_path":str(json_path),"zip_path":str(zip_path),"json_sha256":first[0],"zip_sha256":first[1],"validation":validation,"determinism_verified":True,"model_inference_during_packaging":0,"readiness_status":"STRUCTURALLY_VALIDATED_NOT_SCIENTIFICALLY_READY"}


def package(config:dict,output:Path,sources:list[str],seed:dict,run_state:Path,*,reuse_reconciled:bool=False)->dict:
    artifacts=ROOT/config["outputs"]["artifacts"];artifacts.mkdir(parents=True,exist_ok=True);atomic_json(artifacts/"technical_triage.json",{"healthy_sources":sources})
    runtime=load_config("configs/phase10f_medlatec_10000.yaml");runtime["inputs"]["corpus"]=config["inputs"]["official_corpus"];runtime["outputs"].update({"root":str(output),"artifacts":str(artifacts)})
    ids=official_ids(ROOT/config["inputs"]["official_corpus"]);pilot_pool,pilot_reconciliation=prepare_official_pilot_pool(config,output,ids);runtime["inputs"]["pilot_rerank_pool"]=str(pilot_pool)
    rankings=output/"rankings"
    if reuse_reconciled:
        required=[rankings/"NEW/reranked_chunks.parquet",rankings/"NEW/reranked_documents.parquet",output/"round1/canonical_text.sqlite",artifacts/"provenance_closure.json"]
        missing=[str(path) for path in required if not path.exists()]
        if missing:raise ValueError(f"package-only reconciliation checkpoint is incomplete: {missing}")
        closure=json.loads((artifacts/"provenance_closure.json").read_text(encoding="utf-8"))
        if not closure.get("passed"):raise ValueError("durable provenance closure checkpoint did not pass")
    else:
        build_cached_subset_rankings(runtime,{"NEW":sources},rankings)
    counts=[]
    for batch in pq.ParquetFile(rankings/"NEW/reranked_documents.parquet").iter_batches(columns=["query_id"],batch_size=4096):counts.extend(batch.column(0).to_pylist())
    frequency=Counter(map(int,counts));distribution=np.asarray([frequency[q] for q in sorted(frequency)],dtype=np.int64)
    depth_stats={"min":int(distribution.min()),"p5":float(np.percentile(distribution,5)),"median":float(np.median(distribution)),"p95":float(np.percentile(distribution,95)),"max":int(distribution.max())}
    requested=list(map(int,config["submission"]["requested_document_depths"]));feasible=[depth for depth in requested if depth<=depth_stats["min"]]
    store=output/"round1/canonical_text.sqlite"
    if reuse_reconciled:
        docs=output/"round1/combined_documents.parquet";chunks=output/"round1/combined_chunks.parquet"
        connection=sqlite3.connect(store);integrity=str(connection.execute("PRAGMA integrity_check").fetchone()[0]);canonical_store={"chunks":int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),"documents":int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),"sqlite_integrity":integrity};connection.close()
        if integrity!="ok":raise ValueError(f"canonical checkpoint integrity failed: {integrity}")
        corpus={"documents":pq.ParquetFile(docs).metadata.num_rows,"chunks":pq.ParquetFile(chunks).metadata.num_rows,"reused_reconciled_checkpoint":True}
    else:
        docs,chunks,corpus=canonical(config,output,rankings);canonical_store=_build_canonical_store(chunks,docs,store)
        closure=_audit_ranked_provenance_closure(rankings/"NEW/reranked_chunks.parquet",store,ROOT/config["inputs"]["official_corpus"],artifacts/"provenance_closure.json")
    cache=audit_score_cache(output/"round1/source_scores.sqlite",output/"round1/source_candidate_parts");packages={}
    existing_control=ROOT/config["outputs"]["submissions"]/"phase11b_NEW_CORPUS_CONTROL.json"
    if not existing_control.exists():raise ValueError(f"validated Docs@10 control package is missing: {existing_control}")
    control_json=artifacts/"docs10_chunks20_frozen_control.json"
    if not control_json.exists():shutil.copy2(existing_control,control_json)
    if file_sha256(control_json)!=file_sha256(existing_control):raise ValueError("frozen Docs@10 control differs from validated package JSON")
    control_validation=normalize_validation_result(_stream_validate(existing_control,ROOT/config["outputs"]["submissions"]/"phase11b_NEW_CORPUS_CONTROL.zip",expected_query_ids(runtime["inputs"]["queries"]),store,document_depth=10,chunk_depth=20))
    control_official=_official_id_audit(control_json,ROOT/config["inputs"]["official_corpus"])
    if not control_validation["passed"] or not control_official["passed"]:raise ValueError("frozen Docs@10 control failed validation")
    for depth in feasible:
        runtime["submission"]["documents_per_query"]=depth;runtime["submission"]["chunks_per_query"]=20
        name="phase11b_NEW_CORPUS_CONTROL" if depth==10 else f"phase11b_DOCS{depth}_CHUNKS20"
        state(run_state,stage="PACKAGE_RECONCILIATION",status="RUNNING",total=len(feasible),done=len(packages),current_document_depth=depth,checkpoint=str(artifacts/"provenance_closure.json"),rerank_scores_preserved=134400)
        built=_package_depth_variant(runtime,rankings,control_json,name,depth,ROOT/config["outputs"]["submissions"],store);official=_official_id_audit(Path(built["json_path"]),ROOT/config["inputs"]["official_corpus"])
        comparison={"queries":0,"document_prefix_equal":0,"chunks20_equal":0}
        for reference,current in zip(_iter_json_array(control_json),_iter_json_array(Path(built["json_path"])),strict=True):
            comparison["queries"]+=1;comparison["document_prefix_equal"]+=list(current["relevant_docs"][:10])==list(reference["relevant_docs"]);comparison["chunks20_equal"]+=current["relevant_chunks"]==reference["relevant_chunks"]
        control={"passed":comparison=={"queries":1200,"document_prefix_equal":1200,"chunks20_equal":1200},"docs10_control":str(control_json),"fixed_ranking_depth_comparison":comparison}
        passed=bool(built["validation"]["passed"] and built["determinism_verified"] and official["passed"] and cache["passed"] and control["passed"])
        if passed:built["readiness_status"]="READY_FOR_LEADERBOARD"
        packages[str(depth)]={"status":"READY_FOR_LEADERBOARD" if passed else "NEEDS_AGENT","package":built,"official_id_audit":official,"control_replay":control}
    report={"status":"READY_FOR_LEADERBOARD" if packages and all(row["status"]=="READY_FOR_LEADERBOARD" for row in packages.values()) else "NEEDS_AGENT","new_official_corpus":config["inputs"]["official_corpus"],"corpus":corpus,"candidate_rows":seed["total"],"score_reuse":seed,"pilot_reconciliation":pilot_reconciliation,"canonical_store":canonical_store,"provenance_closure":closure,"unique_document_depth":depth_stats,"requested_depths":requested,"feasible_depths":feasible,"chunk_depth":20,"packages":packages,"no_crawling":True,"score_cache":cache}
    atomic_json(artifacts/"phase11b_report.json",report);atomic_json(artifacts/"phase11b_pre_submit_audit.json",report)
    state(run_state,stage="COMPLETE",status=report["status"],total=1,done=1,integrity="ok" if report["status"]=="READY_FOR_LEADERBOARD" else "failed",checkpoint=str(artifacts/"phase11b_pre_submit_audit.json"))
    return report


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--run-state",required=True);parser.add_argument("--resume-from",choices=["auto","package"],default="auto");args=parser.parse_args();config=yaml.safe_load((ROOT/args.config).read_text(encoding="utf-8"));run_state=ROOT/args.run_state
    if args.resume_from=="package":
        output=ROOT/config["outputs"]["root"];seed_path=ROOT/config["outputs"]["artifacts"]/"score_seed_summary.json"
        if not seed_path.exists():raise ValueError(f"missing durable score seed checkpoint: {seed_path}")
        seed=json.loads(seed_path.read_text(encoding="utf-8"));sources=sorted(UNAFFECTED|RECOMPUTE)
        connection=sqlite3.connect(output/"round1/source_scores.sqlite");total=int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]);done=int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0]);integrity=str(connection.execute("PRAGMA integrity_check").fetchone()[0]);connection.close()
        if (total,done,integrity)!=(134400,134400,"ok"):raise ValueError(f"package-only score checkpoint invalid: total={total} done={done} integrity={integrity}")
        state(run_state,stage="PACKAGE_RECONCILIATION",status="RUNNING",total=1,done=0,checkpoint=str(seed_path),rerank_scores_preserved=done)
        result=package(config,output,sources,seed,run_state,reuse_reconciled=True);print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result["status"]=="READY_FOR_LEADERBOARD" else 2
    migration_path=ROOT/config["outputs"]["artifacts"]/"migration_audit.json"
    if migration_path.exists():
        migration=json.loads(migration_path.read_text(encoding="utf-8"))
        if migration.get("status")!="MIGRATION_AUDIT_COMPLETE":raise ValueError(f"invalid migration checkpoint: {migration_path}")
    else:
        state(run_state,stage="MIGRATION_AUDIT",status="RUNNING",total=1,done=0)
        migration=migration_audit(ROOT/args.config)
    state(run_state,stage="CANDIDATE_RECONCILIATION",status="RUNNING",total=7,done=5,checkpoint=str(migration_path),migration_checkpoint_reused=True)
    output,sources,seed=prepare_candidates(config,run_state);score(config,output,run_state);state(run_state,stage="PACKAGE",status="RUNNING",total=1,done=0,checkpoint="score cache complete")
    result=package(config,output,sources,seed,run_state);print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result["status"]=="READY_FOR_LEADERBOARD" else 2


if __name__=="__main__":raise SystemExit(main())
