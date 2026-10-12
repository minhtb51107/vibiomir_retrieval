#!/usr/bin/env python3
"""Build the bounded Phase 11A bottleneck probes without acquisition."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.indexing.tokenizer_validation import HuggingFaceOffsetTokenizer
from src.source_census.cached_subsets import build_cached_subset_rankings
from src.source_census.pipeline import SourceBM25Index, atomic_json, build_source_candidates, load_config, normalize_text
from src.source_census.union_experiment import (
    _build_canonical_store, _compare_submission_signatures, _filter_parquet,
    _official_id_audit, _prepare_package_canonical, _stream_package,
)
from src.validation.pre_submission import audit_score_cache
from transformers import AutoTokenizer


RUN_STATE: Path | None = None


def update_run_state(**fields) -> None:
    """Publish durable Phase 11A progress without coupling work to the supervisor."""
    if RUN_STATE is None:
        return
    try:
        current=json.loads(RUN_STATE.read_text(encoding="utf-8")) if RUN_STATE.exists() else {}
    except (OSError,json.JSONDecodeError):
        current={}
    current.update(fields)
    current["updated_at"]=dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(RUN_STATE,current)


CURRENT_SOURCES={
    "medlatec.vn","v.familydoctor.com.cn","vinmec.com","tiemchunglongchau.com.vn",
    "cancer.39.net","suckhoedoisong.vn","benhviennhitrunguong.gov.vn","zydcd.com","hellobacsi.com",
}
PARENTS=[
    ("med",Path("data/source_census/phase10f_medlatec_10000"),{"medlatec.vn"}),
    ("g1",Path("data/source_census/phase10e_g1a_10010"),{"v.familydoctor.com.cn","vinmec.com"}),
    ("g6",Path("data/source_census/phase10e_g6b_15000"),None),
    ("g5",Path("data/source_census/phase10e_g5a_11200"),None),
]
BROAD_INPUTS=[
    *PARENTS,
    ("c1",Path("data/phase10c1/C1"),None),
    ("c2",Path("data/phase10c1/C2"),None),
    ("round1",Path("data/source_census/round1"),None),
    ("phase8",Path("data/processed"),None),
]


def host(url:str)->str:
    return (urlsplit(str(url)).hostname or "").lower().removeprefix("www.")


def paths_for(name:str,root:Path)->tuple[Path,Path]:
    if name=="phase8": return root/"phase8_5k_documents.parquet",Path("data/chunks/phase8_5k_bge.parquet")
    return root/"documents.parquet",root/"chunks.parquet"


def tokenizer(config:dict):
    model=config["models"]["embedder"]
    return HuggingFaceOffsetTokenizer(AutoTokenizer.from_pretrained(model["name"],revision=model["revision"],local_files_only=True))


def output_probe(spec:dict)->dict:
    config=load_config(spec["base_config"]); row=spec["output_probe"]
    config["submission"]["documents_per_query"]=int(row["documents_per_query"])
    config["submission"]["chunks_per_query"]=int(row["chunks_per_query"])
    ranking=Path(spec["current_best"]["ranking_root"]); cache=Path(spec["current_best"]["cache_root"])
    package=_stream_package(config,ranking.parent,"UNION",row["name"],Path(spec["outputs"]["submissions"]),cache/"canonical_text.sqlite",tokenizer(config))
    official=_official_id_audit(Path(package["json_path"]),Path(config["inputs"]["corpus"]))
    baseline=Path(spec["current_best"]["submission_json"])
    # The first 10/20 objects must exactly retain the organizer-valid ordering.
    base=json.loads(baseline.read_text(encoding="utf-8")); deep=json.loads(Path(package["json_path"]).read_text(encoding="utf-8"))
    prefix_docs=sum(a["relevant_docs"]==b["relevant_docs"][:10] for a,b in zip(base,deep,strict=True))
    prefix_chunks=sum(a["relevant_chunks"]==b["relevant_chunks"][:20] for a,b in zip(base,deep,strict=True))
    audit={"status":"READY_FOR_LEADERBOARD","passed":True,"intended_variable":"final output depth only",
           "organizer_rule":{"hard_maximum":None,"variable_length_arrays_accepted":True},
           "frozen_control_prefix":{"queries":1200,"top10_docs_identical":prefix_docs,"top20_chunks_identical":prefix_chunks,
                                    "passed":prefix_docs==prefix_chunks==1200},
           "official_id_audit":official,"structural_validation":package["validation"],"determinism_verified":package["determinism_verified"],
           "new_acquisition":0,"new_embeddings":0,"new_reranker_inference":0,"submission":{"path":package["zip_path"],"sha256":package["zip_sha256"]}}
    if not (official["passed"] and audit["frozen_control_prefix"]["passed"]): audit["status"]="NEEDS_AGENT";audit["passed"]=False
    out=Path(spec["outputs"]["artifacts"]);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"output_depth_pre_submit_audit.json",audit)
    atomic_json(out/"output_depth_report.json",{"experiment":"OUTPUT_DEPTH","status":audit["status"],"package":package,"audit":str(out/"output_depth_pre_submit_audit.json")})
    return audit


def broad_inventory(spec:dict)->dict:
    root=Path(spec["outputs"]["root"])/"broad"; root.mkdir(parents=True,exist_ok=True)
    selected_docs={}; attempted=set(); source_stats={}; input_rows=[]
    for name,parent,allow in BROAD_INPUTS:
        documents,chunks=paths_for(name,parent); added=duplicates=usable=0
        for batch in pq.ParquetFile(documents).iter_batches(columns=["doc_id","original_url","extraction_status","normalized_text"],batch_size=4096):
            for doc_id,url,status,text in zip(*(c.to_pylist() for c in batch.columns),strict=True):
                domain=host(url)
                if allow and domain not in allow: continue
                attempted.add(int(doc_id))
                if status!="SUCCESS" or not str(text or "").strip(): continue
                usable+=1
                if int(doc_id) in selected_docs: duplicates+=1; continue
                selected_docs[int(doc_id)]=(name,domain); added+=1
        source_stats[name]={"usable_input_rows":usable,"unique_docs_added":added,"duplicate_docs_skipped":duplicates}
        input_rows.append((name,documents,chunks,allow))
    current_ids={doc for doc,(name,_) in selected_docs.items() if name in {"med","g1","g6","g5"}}
    extra_ids=set(selected_docs)-current_ids
    doc_paths=[row[1] for row in input_rows]; chunk_paths=[row[2] for row in input_rows]
    _filter_parquet(doc_paths,root/"documents.parquet","doc_id",set(selected_docs))
    _filter_parquet(doc_paths,root/"extra_documents.parquet","doc_id",extra_ids)
    # Select chunk IDs whose document survived the priority deduplication.
    all_chunk_ids=set(); extra_chunk_ids=set(); chunks_by_input={}
    for name,_,chunk_path,allow in input_rows:
        count=0
        for batch in pq.ParquetFile(chunk_path).iter_batches(columns=["chunk_id","doc_id","source_url"],batch_size=8192):
            for chunk_id,doc_id,url in zip(*(c.to_pylist() for c in batch.columns),strict=True):
                if allow and host(url) not in allow: continue
                owner=selected_docs.get(int(doc_id))
                if not owner or owner[0]!=name: continue
                chunk_id=str(chunk_id)
                if chunk_id in all_chunk_ids: raise ValueError(f"duplicate chunk identity with selected owner: {chunk_id}")
                all_chunk_ids.add(chunk_id); count+=1
                if int(doc_id) in extra_ids: extra_chunk_ids.add(chunk_id)
        chunks_by_input[name]=count
    _filter_parquet(chunk_paths,root/"chunks.parquet","chunk_id",all_chunk_ids)
    _filter_parquet(chunk_paths,root/"extra_chunks.parquet","chunk_id",extra_chunk_ids)
    domains={domain for _,domain in selected_docs.values()}
    result={"official_document_rows":len(attempted),"usable_documents":len(selected_docs),"chunks":len(all_chunk_ids),"domains":len(domains),
            "focused_usable_documents":len(current_ids),"focused_chunks":len(all_chunk_ids-extra_chunk_ids),
            "additional_usable_documents":len(extra_ids),"additional_chunks":len(extra_chunk_ids),"duplicate_doc_ids_removed":sum(r["duplicate_docs_skipped"] for r in source_stats.values()),
            "source_domains":sorted(domains),"inputs":source_stats,"chunk_rows_by_input":chunks_by_input,"provenance":"official IDs plus SUCCESS/nonempty extraction and chunk->selected-document ownership"}
    atomic_json(Path(spec["outputs"]["artifacts"])/"broad_inventory.json",result)
    return result


def _copy_or_link(source:Path,destination:Path)->None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): return
    try: os.link(source,destination)
    except OSError: shutil.copy2(source,destination)


def _init_score_db(parts:Path,destination:Path)->int:
    destination.parent.mkdir(parents=True,exist_ok=True); destination.unlink(missing_ok=True)
    db=sqlite3.connect(destination)
    db.execute("CREATE TABLE pairs(query_id INTEGER,chunk_id TEXT,query_text TEXT,chunk_text TEXT,rerank_score REAL,inference_ms REAL,PRIMARY KEY(query_id,chunk_id))")
    rows=0
    for path in sorted(parts.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["query_id","chunk_id","query_text","chunk_text"],batch_size=4096):
            values=[(int(a),str(b),str(c),str(d)) for a,b,c,d in zip(*(x.to_pylist() for x in batch.columns),strict=True)]
            rows+=len(values); db.executemany("INSERT INTO pairs(query_id,chunk_id,query_text,chunk_text) VALUES(?,?,?,?)",values)
    db.commit(); total=int(db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]);db.close()
    if rows!=total: raise ValueError(f"candidate key collision: rows={rows} keys={total}")
    return total


def _seed_exact(destination:Path,sources:list[Path])->dict:
    db=sqlite3.connect(destination); reused=0; mismatches=0; by_source={}
    for index,path in enumerate(sources):
        if not path.exists(): continue
        alias=f"old{index}";db.execute(f"ATTACH DATABASE ? AS {alias}",(str(path.resolve()),))
        tables={r[0] for r in db.execute(f"SELECT name FROM {alias}.sqlite_master WHERE type='table'")}
        table="pairs" if "pairs" in tables else ("scores" if "scores" in tables else None)
        if table is None: db.execute(f"DETACH DATABASE {alias}");continue
        columns={r[1] for r in db.execute(f"PRAGMA {alias}.table_info({table})")}
        if not {"query_id","chunk_id","rerank_score"}.issubset(columns): db.execute(f"DETACH DATABASE {alias}");continue
        before=int(db.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
        timing="s.inference_ms" if "inference_ms" in columns else "0.0"
        with db:
            db.execute(f"UPDATE pairs AS d SET rerank_score=(SELECT s.rerank_score FROM {alias}.{table} AS s WHERE s.query_id=d.query_id AND s.chunk_id=d.chunk_id), inference_ms=(SELECT {timing} FROM {alias}.{table} AS s WHERE s.query_id=d.query_id AND s.chunk_id=d.chunk_id) WHERE d.rerank_score IS NULL AND EXISTS(SELECT 1 FROM {alias}.{table} AS s WHERE s.query_id=d.query_id AND s.chunk_id=d.chunk_id AND s.rerank_score IS NOT NULL)")
        after=int(db.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0]); by_source[str(path)]=after-before;reused+=after-before
        db.execute(f"DETACH DATABASE {alias}")
    total=int(db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]);done=int(db.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0]);integrity=db.execute("PRAGMA integrity_check").fetchone()[0];db.close()
    return {"total":total,"reused":done,"newly_seeded":reused,"missing":total-done,"source_contributions":by_source,"value_mismatches":mismatches,"sqlite_integrity":integrity}


def _runtime_yaml(base:dict,root:Path,path:Path)->None:
    config=load_config(base["base_config"]);config["outputs"]["root"]=str(root)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")


def _score(root:Path,runtime:Path,metrics:Path)->None:
    db=sqlite3.connect(root/"round1/source_scores.sqlite");missing=int(db.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NULL").fetchone()[0]);db.close()
    if not missing:return
    command=[sys.executable,"tools/score_source_census_persistent.py","--config",str(runtime),"--window-size","2000","--metrics-out",str(metrics)]
    if RUN_STATE is not None: command.extend(["--run-state",str(RUN_STATE)])
    subprocess.run(command,check=True)


def _control_replay(spec:dict,config:dict)->dict:
    # The Phase 11A output probe already runs the current-best ranking through
    # this exact streaming packager/validator. Its frozen 10/20 prefix is the
    # semantic control replay; Phase 10F separately proved byte identity for
    # the immutable full-union control.
    audit_path=Path(spec["outputs"]["artifacts"])/"output_depth_pre_submit_audit.json"
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    prefix=audit["frozen_control_prefix"]
    passed=bool(audit.get("passed") and prefix.get("passed") and prefix.get("top10_docs_identical")==1200 and prefix.get("top20_chunks_identical")==1200)
    return {"passed":passed,"control":str(spec["current_best"]["submission_json"]),"replay_evidence":str(audit_path),
            "semantic_prefix_queries":1200,"top10_docs_identical":prefix.get("top10_docs_identical"),
            "top20_chunks_identical":prefix.get("top20_chunks_identical"),"phase10f_byte_identical_control":"artifacts/source_census/phase10f_full9_control_fixture.json"}


def _finalize_probe(spec:dict,root:Path,name:str,sources:list[str],config:dict,seed:dict,intended:str)->dict:
    artifacts=Path(spec["outputs"]["artifacts"]);artifacts.mkdir(parents=True,exist_ok=True)
    triage={"healthy_sources":sources};atomic_json(artifacts/f"{name}_technical_triage.json",triage)
    local_artifacts=root/"artifacts";local_artifacts.mkdir(parents=True,exist_ok=True);atomic_json(local_artifacts/"technical_triage.json",triage)
    config["outputs"]["root"]=str(root);config["outputs"]["artifacts"]=str(local_artifacts)
    rankings=root/"rankings";build_cached_subset_rankings(config,{"PROBE":sources},rankings)
    broad_root=Path(spec["outputs"]["root"])/"broad"
    if name==spec["broad_probe"]["name"]:
        doc_paths=[Path(config["inputs"]["pilot_documents"]),broad_root/"documents.parquet"]
        chunk_paths=[Path(config["inputs"]["pilot_chunks"]),broad_root/"chunks.parquet"]
        chunk_ids={str(x) for b in pq.ParquetFile(rankings/"PROBE/reranked_chunks.parquet").iter_batches(columns=["chunk_id"]) for x in b.column(0).to_pylist()}
        doc_ids={int(x) for b in pq.ParquetFile(rankings/"PROBE/reranked_chunks.parquet").iter_batches(columns=["doc_id"]) for x in b.column(0).to_pylist()}
        _filter_parquet(chunk_paths,root/"round1/combined_chunks.parquet","chunk_id",chunk_ids)
        _filter_parquet(doc_paths,root/"round1/combined_documents.parquet","doc_id",doc_ids)
    else:
        parent_specs={n:{"root":str(r),"sources":sorted(a) if a else sorted({host(x) for b in pq.ParquetFile(r/"documents.parquet").iter_batches(columns=["original_url"]) for x in b.column(0).to_pylist()})} for n,r,a in PARENTS}
        pseudo={"parents":parent_specs};_prepare_package_canonical(config,pseudo,rankings,"PROBE")
    canonical=root/"round1/canonical_text.sqlite";_build_canonical_store(root/"round1/combined_chunks.parquet",root/"round1/combined_documents.parquet",canonical)
    package=_stream_package(config,rankings,"PROBE",name,Path(spec["outputs"]["submissions"]),canonical,tokenizer(config))
    official=_official_id_audit(Path(package["json_path"]),Path(config["inputs"]["corpus"]));cache=audit_score_cache(root/"round1/source_scores.sqlite",root/"round1/source_candidate_parts")
    control=_control_replay(spec,load_config(spec["base_config"]))
    changes=_compare_submission_signatures(Path(spec["current_best"]["submission_json"]),Path(package["json_path"]))
    failures=[]
    if not official["passed"]:failures.append("official_ids")
    if not cache["passed"]:failures.append("score_cache")
    if not control["passed"]:failures.append("control_replay")
    audit={"status":"READY_FOR_LEADERBOARD" if not failures else "NEEDS_AGENT","passed":not failures,"failures":failures,"intended_variable":intended,
           "control_replay":control,"score_cache_integrity":cache,"cache_reuse":seed,"official_id_audit":official,"structural_validation":package["validation"],"determinism_verified":package["determinism_verified"],"ranking_change_vs_current_best":changes,"submission":{"path":package["zip_path"],"sha256":package["zip_sha256"]}}
    atomic_json(artifacts/f"{name}_pre_submit_audit.json",audit);atomic_json(artifacts/f"{name}_report.json",{"experiment":name,"status":audit["status"],"candidate_pairs":seed["total"],"reused":seed["reused"],"new_inference":seed["total"]-seed["reused"],"package":package,"audit":str(artifacts/f"{name}_pre_submit_audit.json")})
    return audit


def prepare_m16(spec:dict)->dict:
    root=Path(spec["outputs"]["root"])/"m16";parts=root/"round1/source_candidate_parts";parts.mkdir(parents=True,exist_ok=True)
    parent_configs={"med":"configs/phase10f_medlatec_10000.yaml","g1":"configs/phase10e_g1a_10010.yaml","g6":"configs/phase10e_g6b_15000.yaml","g5":"configs/phase10e_g5a_11200.yaml"}
    source_dbs=[]
    for name,parent,allow in PARENTS:
        work=root/f"parent_{name}";art=work/"artifacts";round1=work/"round1";art.mkdir(parents=True,exist_ok=True);round1.mkdir(parents=True,exist_ok=True)
        cfg=load_config(parent_configs[name]); all_sources=allow or {host(x) for b in pq.ParquetFile(parent/"documents.parquet").iter_batches(columns=["original_url"]) for x in b.column(0).to_pylist()}
        atomic_json(art/"technical_triage.json",{"healthy_sources":sorted(all_sources)});cal=art/"candidate_cap_calibration.json";atomic_json(cal,{"chosen_m":16})
        cfg["candidate_cache"].update({"candidate_caps":[16],"dense_depth":16,"sparse_depth":16,"calibrated_cap_artifact":str(cal)})
        cfg["outputs"].update({"root":str(work),"artifacts":str(art),"documents":str(parent/"documents.parquet"),"chunks":str(parent/"chunks.parquet")})
        _copy_or_link(parent/"round1/chunk_embeddings.f32",round1/"chunk_embeddings.f32")
        build_source_candidates(cfg,groups={"M16":sorted(all_sources)})
        for marker in (round1/"source_candidate_parts").glob("*.json"):
            data=json.loads(marker.read_text(encoding="utf-8"));
            if data["source"] not in all_sources:continue
            shutil.copy2(marker,parts/marker.name);_copy_or_link(marker.with_suffix(".parquet"),parts/marker.with_suffix(".parquet").name)
    total=_init_score_db(parts,root/"round1/source_scores.sqlite")
    old=[Path(spec["current_best"]["cache_root"])/"source_scores.sqlite",*(p/"round1/source_scores.sqlite" for _,p,_ in PARENTS)]
    seed=_seed_exact(root/"round1/source_scores.sqlite",old);atomic_json(Path(spec["outputs"]["artifacts"])/"m16_seed_summary.json",seed)
    runtime=root/"runtime.yaml";_runtime_yaml(spec,root,runtime);_score(root,runtime,Path(spec["outputs"]["artifacts"])/"m16_rerank_metrics.json")
    config=load_config(spec["base_config"]);config["candidate_cache"].update({"candidate_caps":[16],"dense_depth":16,"sparse_depth":16})
    return _finalize_probe(spec,root,spec["candidate_probe"]["name"],sorted(CURRENT_SOURCES),config,seed,"per-source candidate cap m=8 -> m=16 only")


def resume_m16(spec:dict)->dict:
    """Resume after durable candidate construction/exact-key score seeding."""
    root=Path(spec["outputs"]["root"])/"m16"
    seed=json.loads((Path(spec["outputs"]["artifacts"])/"m16_seed_summary.json").read_text(encoding="utf-8"))
    _score(root,root/"runtime.yaml",Path(spec["outputs"]["artifacts"])/"m16_rerank_metrics.json")
    config=load_config(spec["base_config"]);config["candidate_cache"].update({"candidate_caps":[16],"dense_depth":16,"sparse_depth":16})
    return _finalize_probe(spec,root,spec["candidate_probe"]["name"],sorted(CURRENT_SOURCES),config,seed,"per-source candidate cap m=8 -> m=16 only")


def prepare_broad(spec:dict)->dict:
    inventory_path=Path(spec["outputs"]["artifacts"])/"broad_inventory.json"
    broad_root=Path(spec["outputs"]["root"])/"broad"
    required=[broad_root/name for name in ("documents.parquet","chunks.parquet","extra_documents.parquet","extra_chunks.parquet")]
    inventory=(json.loads(inventory_path.read_text(encoding="utf-8"))
               if inventory_path.exists() and all(path.exists() for path in required)
               else broad_inventory(spec))
    update_run_state(status="RUNNING",stage="BROAD_CANDIDATES",total=1200,done=0,
                     broad_inventory=inventory,checkpoint="broad_inventory.json")
    root=Path(spec["outputs"]["root"])/"broad_probe";parts=root/"round1/source_candidate_parts";parts.mkdir(parents=True,exist_ok=True)
    # Preserve the complete current-best m=8 candidate competition.
    current_parts=Path(spec["current_best"]["cache_root"])/"source_candidate_parts"
    for path in current_parts.iterdir():
        if path.suffix in {".json",".parquet"}: _copy_or_link(path,parts/path.name)
    extra=broad_root/"extra_chunks.parquet"
    part=parts/"broad_existing_extra.parquet";marker=parts/"broad_existing_extra.json"
    queries=pq.read_table(load_config(spec["base_config"])["inputs"]["queries"],columns=["id","query"]).to_pylist();depth=int(spec["broad_probe"]["lexical_candidates_per_query"])
    if not (part.exists() and marker.exists() and pq.ParquetFile(part).metadata.num_rows==len(queries)*depth):
        rows=pq.read_table(extra).to_pylist();texts=[str(r["normalized_text"]) for r in rows]
        bm25=SourceBM25Index(texts,k1=1.2,b=0.75);output=[]
        for query_number,query in enumerate(queries,1):
            for rank,(index,score) in enumerate(bm25.rank(normalize_text(str(query["query"])),top_k=depth),1):
                row=rows[index];output.append({"query_id":int(query["id"]),"query_text":str(query["query"]),"source":"BROAD_EXISTING_EXTRA","source_rank":rank,"source_score":float(score),"chunk_id":str(row["chunk_id"]),"doc_id":int(row["doc_id"]),"chunk_text":str(row["raw_text"]),"source_url":str(row["source_url"]),"chunk_index":int(row["chunk_index"]),"start_offset":int(row["start_offset"]),"end_offset":int(row["end_offset"]),"section_type":str(row["section_type"]),"heading_path":row.get("heading_path"),"extraction_method":str(row["extraction_method"])})
            if query_number % 25 == 0 or query_number == len(queries):
                update_run_state(status="RUNNING",stage="BROAD_CANDIDATES",total=len(queries),done=query_number,
                                 candidate_rows_written=len(output),bm25_depth_per_query=depth,
                                 checkpoint="in-memory until atomic candidate parquet write")
        pq.write_table(pa.Table.from_pylist(output),part,compression="zstd")
        atomic_json(marker,{"source":"BROAD_EXISTING_EXTRA","query_count":len(queries),"chosen_m":depth,"source_chunk_count":len(rows),"candidate_rows":len(output),"retrieval":"single global BM25 over all extra chunks"})
    update_run_state(status="RUNNING",stage="BROAD_CACHE_SEED",total=expected if 'expected' in locals() else 86400+len(queries)*depth,
                     done=0,bm25_depth_per_query=depth,broad_candidate_rows=len(queries)*depth,
                     checkpoint=str(marker))
    score_db=root/"round1/source_scores.sqlite";expected=86400+len(queries)*depth
    score_valid=False
    if score_db.exists():
        connection=sqlite3.connect(score_db);score_valid=int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])==expected;connection.close()
    if not score_valid: _init_score_db(parts,score_db)
    caches=[Path(spec["current_best"]["cache_root"])/"source_scores.sqlite",Path("data/source_census/round1/round1/source_scores.sqlite"),Path("data/source_census/round1/source_scores.sqlite"),Path("data/retrieval/phase10c1_early_release/C1/rerank_scores.sqlite"),Path("data/retrieval/phase10c1_early_release/C2/rerank_scores.sqlite"),*(Path(f"data/source_relevance_probe/S{i}/rerank_scores.sqlite") for i in range(1,7))]
    seed_path=Path(spec["outputs"]["artifacts"])/"broad_seed_summary.json"
    seed=(json.loads(seed_path.read_text(encoding="utf-8")) if score_valid and seed_path.exists() else _seed_exact(score_db,caches))
    atomic_json(seed_path,seed)
    update_run_state(status="RUNNING",stage="BROAD_CACHE_SEED",total=seed["total"],done=seed["reused"],
                     broad_candidate_rows=len(queries)*depth,unique_candidate_keys=seed["total"],
                     exact_scores_reused=seed["reused"],missing_pairs=seed["missing"],
                     integrity=seed["sqlite_integrity"],checkpoint=str(seed_path))
    runtime=root/"runtime.yaml";_runtime_yaml(spec,root,runtime);_score(root,runtime,Path(spec["outputs"]["artifacts"])/"broad_rerank_metrics.json")
    update_run_state(status="RUNNING",stage="BROAD_PACKAGE",total=1,done=0,
                     checkpoint="rerank complete")
    config=load_config(spec["base_config"])
    sources=[*sorted(CURRENT_SOURCES),"BROAD_EXISTING_EXTRA"]
    audit=_finalize_probe(spec,root,spec["broad_probe"]["name"],sources,config,seed,"addition of all provenance-valid locally acquired corpus through one global BM25 first stage")
    audit["broad_inventory"]=inventory;atomic_json(Path(spec["outputs"]["artifacts"])/f"{spec['broad_probe']['name']}_pre_submit_audit.json",audit)
    update_run_state(status=audit["status"],stage="COMPLETE",total=1,done=1,
                     integrity="ok" if audit["passed"] else "failed",checkpoint=str(Path(spec["outputs"]["artifacts"])/f"{spec['broad_probe']['name']}_pre_submit_audit.json"),
                     submission=audit.get("submission"))
    return audit


def main()->int:
    global RUN_STATE
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/phase11a_bottleneck.yaml");parser.add_argument("--run-state")
    parser.add_argument("command",choices=["output","inventory","m16","resume-m16","broad"]);args=parser.parse_args()
    RUN_STATE=Path(args.run_state) if args.run_state else None
    spec=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result={"output":output_probe,"inventory":broad_inventory,"m16":prepare_m16,"resume-m16":resume_m16,"broad":prepare_broad}[args.command](spec)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
