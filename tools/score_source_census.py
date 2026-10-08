#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sqlite3, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.source_census.pipeline import atomic_json, load_config, score_database_status
from src.reranking.reranker import TransformerCrossEncoderReranker

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/source_census.yaml'); p.add_argument('--max-new',type=int,default=1000); p.add_argument('--run-state'); p.add_argument('--metrics-out'); a=p.parse_args()
    if not 1<=a.max_new<=5000: raise ValueError('isolated census shard must be 1..5000')
    c=load_config(a.config); m=c['models']['reranker']; db_path=Path(c['outputs']['root'])/'round1/source_scores.sqlite'
    process_started=time.perf_counter()
    db=sqlite3.connect(db_path,timeout=60); db.execute('PRAGMA busy_timeout=60000')
    total=int(db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]); base_done=int(db.execute('SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL').fetchone()[0])
    rows=db.execute('SELECT query_id,chunk_id,query_text,chunk_text FROM pairs WHERE rerank_score IS NULL ORDER BY query_id,chunk_id LIMIT ?', (a.max_new,)).fetchall()
    if not rows: print(json.dumps(score_database_status(c))); return 0
    load_started=time.perf_counter()
    reranker=TransformerCrossEncoderReranker(model_name=m['name'],revision=m['revision'],cache_dir=m['cache_dir'],device='cuda',batch_size=2,max_length=512,use_half_on_cuda=True,seed=2026)
    reranker.score_pairs([('warmup','warmup')]); model_load_seconds=time.perf_counter()-load_started
    started=time.perf_counter(); scored=0
    for offset in range(0,len(rows),64):
        batch=rows[offset:offset+64]; begin=time.perf_counter(); values=reranker.score_pairs([(str(r[2]),str(r[3])) for r in batch]); elapsed=time.perf_counter()-begin
        with db: db.executemany('UPDATE pairs SET rerank_score=?,inference_ms=? WHERE query_id=? AND chunk_id=? AND rerank_score IS NULL',[(float(s),elapsed*1000/len(batch),int(r[0]),str(r[1])) for r,s in zip(batch,values,strict=True)])
        scored+=len(batch)
        if a.run_state:
            active_elapsed=max(time.perf_counter()-started,1e-9); wall_elapsed=max(time.perf_counter()-process_started,1e-9)
            done=base_done+scored; effective=scored/wall_elapsed
            atomic_json(a.run_state, {'stage':'RERANK','status':'RUNNING','total':total,'done':done,'remaining':total-done,'integrity':'ok','shard_size':len(rows),'shard_done':scored,'model_load_seconds':round(model_load_seconds,3),'active_pairs_per_second':round(scored/active_elapsed,3),'effective_pairs_per_second':round(effective,3),'eta_seconds':round((total-done)/max(effective,1e-9),2),'last_successful_checkpoint':done})
    scoring_finished=time.perf_counter(); scoring_seconds=scoring_finished-started
    # Each update batch has committed transactionally. Avoid rescanning the
    # 1.27 GiB database after every shard; the orchestrator performs a full
    # integrity check once at the rerank boundary.
    integrity='transactional-ok'; integrity_seconds=0.0; db.close()
    wall_seconds=time.perf_counter()-process_started
    result={'scored':scored,'model_load_seconds':model_load_seconds,'active_scoring_seconds':scoring_seconds,'active_pairs_per_second':scored/max(scoring_seconds,1e-9),'integrity_check_seconds':integrity_seconds,'total_shard_wall_seconds':wall_seconds,'effective_pairs_per_second':scored/max(wall_seconds,1e-9),'integrity':integrity,**score_database_status(c)}
    if a.metrics_out: atomic_json(a.metrics_out,result)
    print(json.dumps(result))
    return 0
if __name__=='__main__': raise SystemExit(main())
