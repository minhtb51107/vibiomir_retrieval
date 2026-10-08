#!/usr/bin/env python3
"""Bounded, read-only repeatability diagnosis for the source-score gate."""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.reranking.reranker import TransformerCrossEncoderReranker
from src.source_census.pipeline import atomic_json,load_config


def sample(database: Path, count: int=8):
    connection=sqlite3.connect(f"file:{database.resolve()}?mode=ro",uri=True)
    rows=connection.execute("SELECT query_id,chunk_id,query_text,chunk_text,rerank_score FROM pairs WHERE rerank_score IS NOT NULL ORDER BY query_id,chunk_id LIMIT ?",(count,)).fetchall()
    connection.close()
    return rows


def scorer(config):
    model=config["models"]["reranker"]
    value=TransformerCrossEncoderReranker(model_name=model["name"],revision=model["revision"],cache_dir=model["cache_dir"],device="cuda",batch_size=2,max_length=512,use_half_on_cuda=True,seed=2026)
    value.score_pairs([("warmup","warmup")])
    return value


def score(config, rows, repeats: int):
    model=scorer(config); pairs=[(str(row[2]),str(row[3])) for row in rows]
    outputs=[model.score_pairs(pairs).astype(float).tolist() for _ in range(repeats)]
    metadata=model.metadata(); model.release()
    return {"metadata":metadata,"scores":outputs}


def score_in_original_depth1000_batch_context(config, rows):
    """Reconstruct batch-of-two neighbours from the depth1000 missing-key order."""
    depth=sqlite3.connect("data/source_census/depth1000/round1/source_scores.sqlite")
    depth.execute("ATTACH DATABASE 'data/source_census/round1/source_scores.sqlite' AS prior")
    missing=depth.execute(
        "SELECT d.query_id,d.chunk_id,d.query_text,d.chunk_text FROM pairs d "
        "LEFT JOIN prior.pairs p ON p.query_id=d.query_id AND p.chunk_id=d.chunk_id "
        "WHERE p.rerank_score IS NULL ORDER BY d.query_id,d.chunk_id"
    ).fetchall(); depth.close()
    position={(int(row[0]),str(row[1])):index for index,row in enumerate(missing)}
    model=scorer(config); result=[]
    for target in rows:
        key=(int(target[0]),str(target[1]))
        if key not in position:
            result.append({"query_id":key[0],"chunk_id":key[1],"not_newly_inferred_at_depth1000":True})
            continue
        index=position[key]; start=(index//2)*2
        context=missing[start:start+2]; values=model.score_pairs([(str(row[2]),str(row[3])) for row in context])
        offset=index-start
        result.append({"query_id":int(target[0]),"chunk_id":str(target[1]),"original_missing_index":index,"batch_chunk_ids":[str(row[1]) for row in context],"reconstructed_score":float(values[offset])})
    model.release(); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/phase10e_g5a_11200.yaml"); parser.add_argument("--child-output"); parser.add_argument("--output",default="artifacts/source_census/phase10e_g5a_11200/reranker_equivalence_investigation.json"); parser.add_argument("--fresh-processes",type=int,default=3); args=parser.parse_args()
    config=load_config(args.config); database=Path(config["outputs"]["root"])/"round1/source_scores.sqlite"; rows=sample(database)
    if args.child_output:
        atomic_json(args.child_output,score(config,rows,1)); return 0
    same=score(config,rows,5); fresh=[]
    with tempfile.TemporaryDirectory(prefix="phase10e_equivalence_") as directory:
        for index in range(args.fresh_processes):
            output=Path(directory)/f"fresh_{index}.json"
            subprocess.run([sys.executable,__file__,"--config",args.config,"--child-output",str(output)],cwd=ROOT,check=True)
            fresh.append(json.loads(output.read_text(encoding="utf-8")))
    reference=np.asarray([float(row[4]) for row in rows]); same_values=np.asarray(same["scores"]); fresh_values=np.asarray([item["scores"][0] for item in fresh])
    all_current=np.concatenate([same_values,fresh_values],axis=0)
    reference_order=np.lexsort((np.asarray([str(row[1]) for row in rows]),-reference)).tolist()
    orders=[np.lexsort((np.asarray([str(row[1]) for row in rows]),-values)).tolist() for values in all_current]
    pair_rows=[]
    for column,row in enumerate(rows):
        values=all_current[:,column]
        pair_rows.append({"query_id":int(row[0]),"chunk_id":str(row[1]),"reference_score":float(reference[column]),"same_process_scores":[float(x) for x in same_values[:,column]],"fresh_process_scores":[float(x) for x in fresh_values[:,column]],"max_absolute_difference_vs_reference":float(np.max(np.abs(values-reference[column])))})
    original_context=score_in_original_depth1000_batch_context(config,rows)
    result={
        "contract":{"model":config["models"]["reranker"]["name"],"revision":config["models"]["reranker"]["revision"],"tokenizer_revision":config["models"]["reranker"]["revision"],"batch_size":2,"max_length":512,"precision":"float16","eval_mode":True,"padding":True,"truncation":"longest_first","score_extraction":"logits.reshape(-1).float().cpu().numpy()","input_order":"query, chunk"},
        "pairs":pair_rows,
        "same_process_max_absolute_difference":float(np.max(np.ptp(same_values,axis=0))),
        "fresh_process_max_absolute_difference":float(np.max(np.ptp(fresh_values,axis=0))),
        "max_absolute_difference_vs_reference":float(np.max(np.abs(all_current-reference))),
        "reference_order":reference_order,"current_orders":orders,"all_orders_equal":all(order==reference_order for order in orders),
        "metadata_equal_across_processes":all(item["metadata"]==same["metadata"] for item in fresh),"metadata":same["metadata"],
        "original_depth1000_batch_context":original_context,
        "original_context_matches_reference":all(item.get("not_newly_inferred_at_depth1000") or item["reconstructed_score"]==float(row[4]) for item,row in zip(original_context,rows,strict=True)),
    }
    atomic_json(args.output,result); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
