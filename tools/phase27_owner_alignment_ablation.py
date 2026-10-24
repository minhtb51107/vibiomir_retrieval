"""Cache-only replication of Phase25 corroboration against immutable Phase27.

Outputs are proposed document selections, not approved submissions. No models,
network, source crawling, old-approval transfer or leaderboard calls.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.runtime.state_file import write_json, update_json
from src.retrieval.owner_alignment import POLICY, select_documents, independent, records, verify_sources
from src.submission.common import file_sha256 as sha

NAME='phase27_owner_alignment_ablation'
ART=ROOT/'artifacts'/NAME
RUN=ROOT/'artifacts/runs'/NAME
BEST=ROOT/'submissions/00_READY_TO_UPLOAD/p27_sdso20749_cap16.zip'
BEST_SHA='b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e'
P27=ROOT/'data/source_census/phase27_suckhoedoisong_remaining_20749/submission_experiment'
HIST=ROOT/'artifacts/p25_doc_evidence_consensus'


def readonly(path):
    con=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=10)
    con.execute('PRAGMA query_only=ON');con.execute('PRAGMA cache_size=-2048');con.execute('PRAGMA mmap_size=0')
    return con


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def state(stage,done,total,**fields):
    update_json(RUN/'state.json',dict(stage=stage,status='RUNNING',done=done,total=total,
        checkpoint=str(ART/'selection.sqlite'),heartbeat_at=time.time(),**fields),preserve_terminal=False)


def guard(start=False):
    import psutil
    import shutil
    memory=psutil.virtual_memory();info=psutil.Process().memory_info()
    telemetry=dict(available_mib=memory.available/2**20,rss_mib=info.rss/2**20,
        peak_rss_mib=getattr(info,'peak_wset',info.rss)/2**20,
        private_mib=getattr(info,'private',info.vms)/2**20,
        disk_free_gib=shutil.disk_usage(ROOT).free/2**30)
    if telemetry['available_mib']<(512 if start else 384) or telemetry['disk_free_gib']<20:
        raise RuntimeError('RESOURCE_GATE '+json.dumps(telemetry))
    return telemetry


def checkpoint(signature):
    db=sqlite3.connect(ART/'selection.sqlite');db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS meta(signature TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS selections(query_id INTEGER PRIMARY KEY,docs TEXT,changes TEXT,chunks_sha TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS proposal_evidence(query_id INTEGER,inserted INTEGER,removed INTEGER,payload TEXT,PRIMARY KEY(query_id,inserted,removed))')
    stored=db.execute('SELECT signature FROM meta').fetchone()
    if stored and stored[0]!=signature:
        db.close();raise ValueError('checkpoint signature mismatch')
    if not stored:db.execute('INSERT INTO meta VALUES(?)',(signature,));db.commit()
    if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':
        db.close();raise ValueError('checkpoint corrupt')
    return db


def historical():
    audit=json.loads((HIST/'final_audit.json').read_text(encoding='utf-8'))
    review=json.loads((HIST/'comparative_review.json').read_text(encoding='utf-8'))
    if sha(HIST/'comparative_review.json')!=audit['comparative_review_sha256']:raise ValueError('historical review hash')
    if sha(Path(audit['zip_path']))!=audit['zip_sha256']:raise ValueError('historical output hash mismatch')
    con=readonly(HIST/'selection.sqlite')
    raw={int(q):json.loads(changes) for q,changes in con.execute('SELECT query_id,changes FROM selections')}
    con.close()
    approved={(int(r['query_id']),int(r['inserted']),int(r['removed'])) for r in review['approvals']}
    if len(approved)!=21 or sum(map(len,raw.values()))!=52:raise ValueError('historical census mismatch')
    proposed={(q,c['inserted'],c['removed']) for q,changes in raw.items() for c in changes}
    if not approved<=proposed:raise ValueError('historical approval outside proposals')
    # Completed historical pipeline is not rerun. Bind archival outputs and
    # recheck the 42 approved source quotes using its existing durable checkpoint.
    base=ROOT/'submissions/00_READY_TO_UPLOAD/p25_tiem8270_cap16.json'
    output=Path(audit['zip_path']).with_suffix('.json')
    if sha(output)!=audit['json_sha256']:raise ValueError('historical JSON hash')
    old_contract=json.loads((HIST/'frozen_inputs.json').read_text(encoding='utf-8'))
    if sha(base)!=old_contract['base_json_sha256'] or old_contract['policy']!=POLICY:
        raise ValueError('historical control/policy changed')
    old_source=readonly(ROOT/'data/source_census/phase25_tiemchung_remaining_8270/submission_experiment/cap16/round1/canonical_text.sqlite')
    for approval in review['approvals']:
        guard()
        for field,doc in [('candidate_quote',approval['inserted']),('baseline_quote',approval['removed'])]:
            owner=old_source.execute('SELECT normalized_text FROM documents WHERE doc_id=?',(doc,)).fetchone()
            if not owner or not approval[field] or approval[field] not in owner[0]:raise ValueError('historical approved source quotation differs')
    old_source.close()
    result=dict(policy=POLICY,raw_proposals=52,raw_queries=sum(bool(v) for v in raw.values()),
        actual_substitutions=21,actual_queries=21,chunk_changes=0,
        selected_triplets=sorted(approved),unapproved_proposals=31,
        official_outcome='UNKNOWN_NO_SEPARATE_RESULT_RECOVERED',
        verification='Existing 1200-query checkpoint/52 proposals, frozen output hashes and42 canonical quotes; no expensive historical recomputation',
        regressions='Historical rejected AI-review proposals include disease/entity, population and temporal mismatches; no organizer per-query labels',
        reuse='Pure selector reusable unchanged. Old deadline, baseline paths, review approvals and outputs are NOT reusable on Phase27.',
        novelty='Same hypothesis/policy on a later frozen control; a replication, not a new retrieval method.')
    write_json(ART/'historical_comparison.json',result)
    return result,approved,proposed


def document_groups(path):
    import pyarrow.parquet as pq
    current=None; rows=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=['query_id','rank','doc_id']):
        guard()
        for row in batch.to_pylist():
            q=int(row['query_id'])
            if current is not None and current!=q:yield current,rows;rows=[]
            current=q;rows.append((int(row['rank']),int(row['doc_id'])))
    if rows:yield current,rows


def cached_competitor(scores,canonical,q,doc,query):
    lo=f'{doc}-';hi=f'{doc}.'
    row=scores.execute('SELECT chunk_id,query_text,chunk_text,rerank_score FROM pairs '
        'WHERE query_id=? AND chunk_id>=? AND chunk_id<? ORDER BY rerank_score DESC,chunk_id LIMIT 1',
        (q,lo,hi)).fetchone()
    if not row:return dict(status='MISSING_EXACT_CACHE_REFERENCE',evidence=None,score=None)
    key,text,evidence,score=row
    if text!=query:raise ValueError('cached original-query identity mismatch')
    source=canonical.execute('SELECT normalized_text FROM documents WHERE doc_id=?',(doc,)).fetchone()
    if not source or evidence not in source[0]:raise ValueError('cached competitor text not canonical')
    import math
    if not math.isfinite(score):raise ValueError('nonfinite cached score')
    return dict(status='CURRENT_FROZEN_CACHE_REFERENCE_ONLY',chunk_id=key,evidence=evidence,score=score)


def execute():
    from urllib.parse import urlsplit
    started=time.perf_counter();guard(start=True);state('HISTORICAL_ARCHIVE_CHECK',0,1)
    history,old_selected,old_proposed=historical()
    guard(start=True)
    import pyarrow.parquet as pq
    canonical_path=P27/'cap16/round1/canonical_text.sqlite'
    score_path=P27/'round1/source_scores.sqlite'
    ranking_path=P27/'rankings/CAP16/reranked_documents.parquet'
    inputs=[BEST,BEST.with_suffix('.json'),canonical_path,score_path,ranking_path,P27/'runtime.yaml',
            ROOT/'data/raw/links_corpus_vi_en_zh.parquet',ROOT/'data/raw/query.parquet',
            ROOT/'tools/cached_doc_evidence_experiment.py',ROOT/'src/retrieval/owner_alignment.py',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    if hashes[str(BEST.relative_to(ROOT))]!=BEST_SHA:raise ValueError('BEST hash mismatch')
    with zipfile.ZipFile(BEST) as archive:
        if len(archive.namelist())!=1:raise ValueError('BEST member schema')
        member_hash=hashlib.sha256()
        with archive.open(archive.namelist()[0]) as member:
            for block in iter(lambda:member.read(1024*1024),b''):member_hash.update(block)
        if member_hash.hexdigest()!=sha(BEST.with_suffix('.json')):raise ValueError('external JSON differs from frozen ZIP')
    contract=dict(control='PHASE27_0.0923',policy=POLICY,inputs=hashes,old_approvals_used=False,
        treatment='B_PROPOSED_NOT_REVIEW_APPROVED',chunks='EXACTLY_UNCHANGED',review_seed=NAME+'-v1')
    signature=digest(contract)
    if (ART/'frozen_inputs.json').exists() and json.loads((ART/'frozen_inputs.json').read_text(encoding='utf-8'))!=contract:
        raise ValueError('frozen input contract changed')
    write_json(ART/'frozen_inputs.json',contract)
    db=checkpoint(signature);canonical=readonly(canonical_path);scores=readonly(score_path)
    queries={int(r['id']):r['query'] for b in pq.ParquetFile(ROOT/'data/raw/query.parquet').iter_batches(batch_size=64) for r in b.to_pylist()}
    metadata={}
    for b in pq.ParquetFile(ROOT/'artifacts/multilingual_retrieval_m1/review_queue.parquet').iter_batches(
            batch_size=128,columns=['query_id','split']):
        for r in b.to_pylist():metadata[int(r['query_id'])]=r['split']
    ranks=iter(document_groups(ranking_path));owners=set();changes=[];stats=collections.Counter()
    for position,record in enumerate(records(BEST.with_suffix('.json')),1):
        guard();q=int(record['id']);rq,rd=next(ranks)
        if rq!=q or [d for _,d in sorted(rd)[:20]]!=record['relevant_docs']:raise ValueError('Phase27 Docs20 replay mismatch')
        owners.update(verify_sources(record,canonical))
        docs,proposals=select_documents(record)
        if docs[:15]!=record['relevant_docs'][:15]:raise ValueError('protected head displaced')
        before_chunks=digest(record['relevant_chunks'])
        # B is represented as an immutable-control chunk reference, never edited text.
        after={**record,'relevant_docs':docs}
        if digest(after['relevant_chunks'])!=before_chunks:raise ValueError('Chunks40 changed')
        saved=db.execute('SELECT docs,changes,chunks_sha FROM selections WHERE query_id=?',(q,)).fetchone()
        values=(json.dumps(docs),json.dumps(proposals),before_chunks)
        if saved and tuple(saved)!=values:raise ValueError('deterministic checkpoint replay differs')
        if not saved:
            db.execute('INSERT INTO selections VALUES(?,?,?,?)',(q,*values));db.commit()
        stats['queries']+=1;stats['queries_changed']+=bool(proposals);stats['proposals']+=len(proposals)
        stats['no_substitution_justified_by_proxy']+=not proposals
        stats['strong_head_positions_displaced']+=sum(a!=b for a,b in zip(record['relevant_docs'][:15],docs[:15]))
        for c in proposals:
            inserted=[dict(rank=i,doc_id=int(w['doc_id']),text=w['chunk_text']) for i,w in enumerate(record['relevant_chunks'],1) if int(w['doc_id'])==c['inserted']]
            removed=[dict(rank=i,doc_id=int(w['doc_id']),text=w['chunk_text']) for i,w in enumerate(record['relevant_chunks'],1) if int(w['doc_id'])==c['removed']]
            competitor=cached_competitor(scores,canonical,q,c['removed'],queries[q])
            payload=dict(query_id=q,query_vi=queries[q],**c,inserted_windows=inserted,
                removed_selected_windows=removed,competitor=competitor,
                candidate_deep_rank=next((rank for rank,doc in rd if doc==c['inserted']),None),
                split=metadata.get(q,'OUTSIDE_SEALED_BENCHMARK'),
                same_old_approved_triplet=(q,c['inserted'],c['removed']) in old_selected,
                same_old_proposed_triplet=(q,c['inserted'],c['removed']) in old_proposed)
            encoded=json.dumps(payload,ensure_ascii=False)
            previous=db.execute('SELECT payload FROM proposal_evidence WHERE query_id=? AND inserted=? AND removed=?',
                                (q,c['inserted'],c['removed'])).fetchone()
            if previous and previous[0]!=encoded:raise ValueError('proposal evidence replay mismatch')
            db.execute('INSERT OR IGNORE INTO proposal_evidence VALUES(?,?,?,?)',(q,c['inserted'],c['removed'],encoded));db.commit()
            # Keep full text on disk; the in-memory full-corpus proposal list is metadata only.
            meta={k:v for k,v in payload.items() if k not in {'inserted_windows','removed_selected_windows','competitor'}}
            meta.update(candidate_window_count=len(inserted),distinct_candidate_texts=len({digest(w['text']) for w in inserted}),
                        baseline_selected_window_count=len(removed),competitor_status=competitor['status'],
                        competitor_cache_score=competitor['score'])
            changes.append(meta)
        if position%25==0:state('CORROBORATION_AND_PROVENANCE',position,1200,telemetry=guard())
    if stats['queries']!=1200 or next(ranks,None) is not None:raise ValueError('1200-query coverage')
    official={}
    for b in pq.ParquetFile(ROOT/'data/raw/links_corpus_vi_en_zh.parquet').iter_batches(batch_size=1024):
        guard()
        for r in b.to_pylist():
            if int(r['id']) in owners:official[int(r['id'])]=r['url']
    if owners-official.keys():raise ValueError('nonofficial owners')
    for c in changes:
        c['inserted_url']=official[c['inserted']];c['removed_url']=official[c['removed']]
        c['inserted_domain']=(urlsplit(c['inserted_url']).hostname or '').removeprefix('www.')
        c['removed_domain']=(urlsplit(c['removed_url']).hostname or '').removeprefix('www.')
    # Freeze a bounded, source/depth/history-stratified queue BEFORE quality judgment.
    eligible=[c for c in changes if c['split']!='held_out']
    key=lambda c:digest([NAME,c['query_id'],c['inserted'],c['removed']])
    strata=collections.defaultdict(list)
    for c in eligible:strata[(c['inserted_domain'],c['same_old_proposed_triplet'])].append(c)
    chosen=[];seen=set()
    for group in sorted(strata):
        for c in sorted(strata[group],key=key):
            if len(chosen)>=16:break
            if c['query_id'] not in seen:chosen.append(c);seen.add(c['query_id']);break
    for c in sorted(eligible,key=lambda c:(c['split']!='development',key(c))):
        if len(chosen)>=16:break
        if c['query_id'] not in seen:chosen.append(c);seen.add(c['query_id'])
    queue=[];mapping={}
    for c in chosen:
        guard()
        case=key(c)[:20];candidate_side='LEFT' if int(case[-1],16)%2 else 'RIGHT'
        payload=json.loads(db.execute('SELECT payload FROM proposal_evidence WHERE query_id=? AND inserted=? AND removed=?',
                          (c['query_id'],c['inserted'],c['removed'])).fetchone()[0])
        candidate=[w['text'] for w in payload['inserted_windows']]
        baseline=[w['text'] for w in payload['removed_selected_windows']]
        if not baseline and payload['competitor']['evidence']:baseline=[payload['competitor']['evidence']]
        queue.append(dict(case_id=case,query_vi=c['query_vi'],evidence_left=candidate if candidate_side=='LEFT' else baseline,
            evidence_right=baseline if candidate_side=='LEFT' else candidate,
            url_left=c['inserted_url'] if candidate_side=='LEFT' else c['removed_url'],
            url_right=c['removed_url'] if candidate_side=='LEFT' else c['inserted_url'],
            judgment='UNJUDGED',comparison_limit='selected windows versus selected-or-cached competitor; absence is not irrelevance'))
        mapping[case]=dict(query_id=c['query_id'],inserted=c['inserted'],removed=c['removed'],candidate_side=candidate_side,
            baseline_evidence_origin='SELECTED_CHUNKS' if payload['removed_selected_windows'] else payload['competitor']['status'])
    protocol=dict(seed=NAME+'-v1',queue_cases=[r['case_id'] for r in queue],maximum=16,
        sealed_held_out_excluded=True,judgments_used=False,input_signature=signature,
        selection='one per source/history stratum then development-first deterministic hash; unique query')
    for name,value in [('source_review_queue.json',{'cases':queue}),('review_mapping.json',mapping),('review_protocol.json',protocol)]:
        target=ART/name
        if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=value:raise ValueError('frozen review output changed')
        write_json(target,value)
    # Only now use old development AI labels for retention diagnostics, never selection.
    with (ROOT/'artifacts/multilingual_retrieval_m1/ai_judgments_import.csv').open(encoding='utf-8-sig',newline='') as f:
        positives=[r for r in csv.DictReader(f) if metadata.get(int(r['query_id']))=='development' and r['judgment'] in {'1','2'}]
    positive_retention=collections.Counter()
    for r in positives:
        selected=db.execute('SELECT docs,changes FROM selections WHERE query_id=?',(int(r['query_id']),)).fetchone()
        bdocs=json.loads(selected[0]);cs=json.loads(selected[1]);adocs=list(bdocs)
        for c in cs:adocs[c['position']-1]=c['removed']
        d=int(r['doc_id']);positive_retention['baseline_positive_pairs_present']+=d in adocs
        positive_retention['baseline_positive_pairs_lost']+=d in adocs and d not in bdocs
        positive_retention['new_positive_pairs_present']+=d not in adocs and d in bdocs
    if sha(BEST)!=BEST_SHA:raise ValueError('BEST modified')
    integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
    db.execute('PRAGMA wal_checkpoint(FULL)');db.close();canonical.close();scores.close()
    # Keep full source evidence private, separate from aggregate report.
    write_json(ART/'changed_pairs.json',{'proposals':changes})
    domains=collections.Counter(c['inserted_domain'] for c in changes)
    report=dict(status='COMPLETE',treatment='B_PROPOSED_NOT_REVIEW_APPROVED',statistics=dict(stats),
        source_insertions=dict(domains),source_displacements=dict(collections.Counter(c['removed_domain'] for c in changes)),
        old_approved_triplets_reproposed=sum(c['same_old_approved_triplet'] for c in changes),
        old_proposed_triplets_reproposed=sum(c['same_old_proposed_triplet'] for c in changes),
        source_review_cases=len(queue),review_status='UNJUDGED',human_verified=0,
        development_silver_retention=dict(positive_retention),chunks_changed=0,
        exact_source_spans_verified=48000,official_owners_verified=len(owners),sqlite_integrity=integrity,
        historical=history,policy_equivalent_to_phase25=True,baseline_and_evidence_differ=True,
        decision='NO_GO_AUTOMATIC_PROMOTION_SAME_POLICY_REPLICATION_NEEDS_NEW_COMPARATIVE_EVIDENCE',
        strong_document_boundary='Ranks1-15 and multi-window tail protected. This does not guarantee clinical relevance of remaining displaced tails.',
        runtime_seconds=time.perf_counter()-started,telemetry=guard(),
        frozen_best_sha256=BEST_SHA,source_review_queue_sha256=sha(ART/'source_review_queue.json'),
        limitations='Agreement is lexical independence, not medical relevance. No old approvals transferred or hidden labels used.')
    write_json(ART/'ab_report.json',report)
    (ART/'REPORT.md').write_text('# Phase27 owner alignment: executed replication\n\n'
        +json.dumps({k:v for k,v in report.items() if k!='historical'},ensure_ascii=False,indent=2)
        +'\n\nHistorical selector unchanged; new control, not a new algorithm or measured score gain. '
        'No ZIP/submission. Review queue frozen before any new quality judgment.\n',encoding='utf-8')
    state('FINISHED',1200,1200);update_json(RUN/'state.json',{'status':'COMPLETE','requires_agent':False})
    print(json.dumps({k:v for k,v in report.items() if k not in {'historical'}},ensure_ascii=False))


def worker():
    ART.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    while True:
        try:guard(start=True);break
        except RuntimeError as exc:
            state('WAIT_FOR_RESOURCES',0,1200,error=str(exc),wait_elapsed_seconds=time.monotonic()-started)
            if time.monotonic()-started>=180:
                write_json(ART/'ab_report.json',dict(status='BLOCKED',reason=str(exc),queries_evaluated=0,
                    decision='NO_GO_EXECUTION_UNTIL_RESOURCE_GATE_PASSES',not_an_ablation_result=True))
                update_json(RUN/'state.json',{'status':'BLOCKED','requires_agent':True});return 0
            time.sleep(5)
    try:execute();return 0
    except Exception as exc:
        write_json(ART/'error.json',dict(type=type(exc).__name__,message=str(exc),checkpoint=str(ART/'selection.sqlite')))
        update_json(RUN/'state.json',dict(status='NEEDS_AGENT',error=str(exc),requires_agent=True));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['supervise','worker']);args=parser.parse_args()
    for s in (sys.stdout,sys.stderr):
        if hasattr(s,'reconfigure'):s.reconfigure(encoding='utf-8',errors='backslashreplace')
    if args.mode=='worker':return worker()
    from tools import competition_supervisor as supervisor
    (RUN/'logs').mkdir(parents=True,exist_ok=True)
    sys.stdout=(RUN/'logs/supervisor.log').open('a',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    supervisor.prevent_sleep=lambda active:None
    supervisor.SUCCESS_TERMINAL_STATES.update({'COMPLETE','BLOCKED'})
    sys.argv=[str(Path(__file__)),'--run-id',NAME,'--minimum-free-gib','20','--',sys.executable,'-X','utf8',str(Path(__file__)),'worker']
    return supervisor.main()


if __name__=='__main__':raise SystemExit(main())
