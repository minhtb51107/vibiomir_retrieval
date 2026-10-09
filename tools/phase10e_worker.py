#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.cached_subsets import build_cached_subset_rankings
from src.source_census.phase10e import (
    assemble_embeddings, assemble_union, finalize_report, phase_contract, seed_scores, verify_union,
)
from src.source_census.pipeline import (
    _merge_source_parts, atomic_json, build_source_candidates, load_config,
    merge_acquisition, score_database_status,
)
from src.validation.pre_submission import (
    audit_reuse, audit_score_cache, semantic_submission_comparison,
)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def stage(path: Path, name: str, **extra: object) -> None:
    atomic_json(path, {"stage": name, "status": "RUNNING", **extra})


def scientific_audit(config, submission, seed, replay):
    root=Path(config["outputs"]["root"]); _,phase,_,group,_=phase_contract(config)
    cache=audit_score_cache(root/"round1/source_scores.sqlite",root/"round1/source_candidate_parts")
    reuse=audit_reuse(root/"round1/source_scores.sqlite",phase["previous_score_database"],seed)
    fixed={
        "query_set":config["inputs"]["queries"]=="data/raw/query.parquet",
        "control_corpus":config["inputs"]["pilot_documents"]=="data/processed/phase3_pilot_documents.parquet" and config["inputs"]["pilot_chunks"]=="data/chunks/phase4_bge_m3/tokens_512_overlap_64.parquet",
        "candidate_cap":max(config["candidate_cache"]["candidate_caps"])==8,
        "reranker_revision":config["models"]["reranker"]["revision"]=="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "top_k":config["submission"]["documents_per_query"]==10 and config["submission"]["chunks_per_query"]==20,
        "chunk_expansion":config["submission"]["expansion_tokens"]==1024,
    }
    structural={"passed":bool(submission.get("validation",{}).get("valid")) and bool(submission.get("determinism_verified")),"validation":submission.get("validation"),"determinism_verified":submission.get("determinism_verified")}
    failures=[]
    for label,item in (("score_cache",cache),("cache_reuse",reuse),("control_replay",replay)):
        if not item["passed"]: failures.append(label+": "+"; ".join(item.get("failures",["failed"])))
    if not all(fixed.values()): failures.append("experiment contract changed outside source depth")
    if not structural["passed"]: failures.append("structural validation/determinism failed")
    return {
        "status":"READY_FOR_LEADERBOARD" if not failures else "NEEDS_AGENT",
        "passed":not failures,"failures":failures,
        "score_cache_integrity":cache,
        "score_distribution_sanity":{"passed":cache["passed"],"distribution":cache["score_distribution"],"distinct_inference_ms_count":cache["distinct_inference_ms_count"]},
        "cache_reuse_audit":reuse,"control_replay":replay,
        "experiment_contract":{"intended_variable":f"{group} source acquisition depth only","fixed_field_checks":fixed,"passed":all(fixed.values())},
        "structural_validation":structural,
    }


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--run-state",required=True); parser.add_argument("--resume-from",choices=["full","source_candidates","rerank"],default="full"); args=parser.parse_args()
    config=load_config(args.config); out=config["outputs"]; _,phase,sources,group,experiment=phase_contract(config); groups={group:sources}; state=Path(args.run_state); python=sys.executable
    Path(out["artifacts"]).mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(phase["manifest"]).read_text(encoding="utf-8"))

    if args.resume_from=="full":
        stage(state,"ACQUIRE",selected=manifest["total_incremental_ids"],completed=0,total_sources=len(sources))
        run([python,"tools/phase10e_acquire.py","--config",args.config,"--run-state",args.run_state])
        stage(state,"MERGE_ACQUISITION"); merge_acquisition(config)
        stage(state,"EXTRACT")
        run([python,"scripts/extract_production.py","--crawl-database",out["crawl_database"],"--archive-directory",out["body_archive"],"--output-directory",out["processed_parts"],"--checkpoint",str(Path(out["root"])/"extract.sqlite"),"--summary-out",str(Path(out["artifacts"])/"extract.json")])
        run([python,"scripts/merge_document_partitions.py","--input-directory",out["processed_parts"],"--output",phase["new_documents"],"--summary-out",str(Path(out["artifacts"])/"new_document_merge.json")])
        stage(state,"CHUNK")
        run([python,"scripts/chunk_production.py","--documents",phase["new_documents"],"--output-directory",out["chunks_parts"],"--checkpoint",str(Path(out["root"])/"chunk.sqlite"),"--summary-out",str(Path(out["artifacts"])/"chunk.json")])
        run([python,"scripts/merge_chunk_partitions.py","--input-directory",out["chunks_parts"],"--output",phase["new_chunks"],"--summary-out",str(Path(out["artifacts"])/"new_chunk_merge.json")])
        stage(state,"ASSEMBLE"); assembly=assemble_union(config)
        stage(state,"EMBEDDINGS"); embedding=assemble_embeddings(config,state)
    else:
        required=[Path(phase["new_documents"]),Path(phase["new_chunks"]),Path(out["documents"]),Path(out["chunks"]),Path(out["root"])/"round1/chunk_embeddings.f32",Path(out["artifacts"])/"corpus_assembly.json",Path(out["artifacts"])/"embedding_summary.json"]
        missing=[str(path) for path in required if not path.exists()]
        if missing: raise FileNotFoundError(f"cannot resume SOURCE_CANDIDATES; missing durable artifacts: {missing}")
        assembly=json.loads((Path(out["artifacts"])/"corpus_assembly.json").read_text(encoding="utf-8"))
        embedding=json.loads((Path(out["artifacts"])/"embedding_summary.json").read_text(encoding="utf-8"))
    union=verify_union(config)
    if args.resume_from!="rerank":
        stage(state,"SOURCE_CANDIDATES",completed_sources=0,total_sources=len(sources),total_query_source_pairs=1200*len(sources),completed_query_source_pairs=0)
        candidates=build_source_candidates(config,run_state=state,groups=groups)
        seed=seed_scores(config)
    else:
        required=[Path(out["root"])/"round1/source_scores.sqlite",Path(out["artifacts"])/"source_candidate_summary.json",Path(out["artifacts"])/"rerank_seed_summary.json"]
        missing=[str(path) for path in required if not path.exists()]
        if missing: raise FileNotFoundError(f"cannot resume RERANK; missing durable artifacts: {missing}")
        candidates=json.loads(required[1].read_text(encoding="utf-8")); seed=json.loads(required[2].read_text(encoding="utf-8"))
    stage(state,"RERANK",**score_database_status(config),cached_pairs_reused=seed["reused"])
    restarts=0; loads=0
    while int(score_database_status(config)["remaining"]):
        loads+=1
        completed=subprocess.run([python,"tools/score_source_census_persistent.py","--config",args.config,"--window-size","2000","--run-state",args.run_state,"--metrics-out",str(Path(out["artifacts"])/"rerank_metrics.json"),"--model-load-count",str(loads),"--scorer-restarts",str(restarts)],cwd=ROOT)
        if completed.returncode:
            restarts+=1
            if restarts>10: raise RuntimeError(f"Phase10E reranker exceeded restart limit: {completed.returncode}")
    status=score_database_status(config,verify_integrity=True)
    if status["integrity"]!="ok" or int(status["remaining"]): raise RuntimeError(f"invalid Phase10E score cache: {status}")

    stage(state,"RANKINGS")
    rankings=Path(out["root"])/"rankings"; build_cached_subset_rankings(config,groups,rankings)
    canonical=Path(out["root"])/"round1"
    _merge_source_parts([Path(config["inputs"]["pilot_chunks"]),Path(out["chunks"])],canonical/"combined_chunks.parquet")
    _merge_source_parts([Path(config["inputs"]["pilot_documents"]),Path(out["documents"])],canonical/"combined_documents.parquet")
    stage(state,"PACKAGE",completed=0,total=1)
    marker=Path(out["artifacts"])/f"submission_{group}.json"
    run([python,"tools/source_census_cached_subset.py","--config",args.config,"package-existing","--ranking-root",str(rankings),"--group",group,"--submission-name",config["focused_scaling"]["submission_name"],"--output-dir",out["submissions"],"--marker",str(marker)])
    submission=json.loads(marker.read_text(encoding="utf-8"))

    stage(state,"CONTROL_REPLAY")
    depth_config="configs/source_census_depth1000.yaml"; control_root=Path(out["root"])/f"control_depth1000_{group.lower()}"; control_name=f"phase10e_control_{group}"
    run([python,"tools/source_census_cached_subset.py","--config",depth_config,"make-subset","--sources",",".join(sources),"--name",control_name,"--work-dir",str(control_root),"--output-dir",str(control_root)])
    replay=semantic_submission_comparison(config["focused_scaling"]["control_submission_json"],control_root/f"{control_name}.json")
    audit=scientific_audit(config,submission,seed,replay)
    audit["change_accounting"]={"old_new_union":union,"reused_rerank_pairs":seed["reused"],"newly_inferred_pairs":audit["score_cache_integrity"]["non_null_scored_count"]-seed["reused"]}
    audit_path=Path(config["focused_scaling"]["audit_path"]); atomic_json(audit_path,audit)
    report=finalize_report(config,submission,assembly,embedding,candidates,seed,audit)
    atomic_json(state,{"stage":audit["status"],"status":audit["status"],"report":str(Path(out["artifacts"]).parent/f"{experiment}_report.json"),"audit":str(audit_path)})
    if not audit["passed"]: raise RuntimeError("Phase10E mandatory scientific gate failed")
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
