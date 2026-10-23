"""Small frozen-corpus Dense execution; invoked only after a same-runtime trial PASS."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import psutil

from src.multilingual_retrieval.m42 import digest, rank_dense, retention, union_rrf, new_evidence_review
from src.multilingual_retrieval.m42_execution import validate_manifest, preflight, runtime_identity
from src.multilingual_retrieval.m3 import _atomic_json


def require_trial(report, manifest, runtime):
    if report.get('passed') is not True or report.get('trial20_sha256') != manifest['trial20_sha256']:
        raise ValueError('mandatory frozen trial20 not passed')
    if report.get('runtime_identity') != runtime:
        raise ValueError('trial runtime/GPU changed; run the frozen trial on this machine first')


def vector_digest(values):
    return hashlib.sha256(np.asarray(values,dtype=np.float32).tobytes()).hexdigest()


def checkpoint_open(directory, signature, rows):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'vectors.f32'
    con=sqlite3.connect(directory/'embedding_checkpoint.sqlite')
    con.execute('PRAGMA cache_size=-1024')
    con.execute('CREATE TABLE IF NOT EXISTS metadata(signature TEXT NOT NULL)')
    con.execute('CREATE TABLE IF NOT EXISTS windows(start INTEGER PRIMARY KEY,count INTEGER NOT NULL,sha256 TEXT NOT NULL)')
    stored=con.execute('SELECT signature FROM metadata').fetchone()
    if stored and stored[0]!=signature: con.close(); raise ValueError('embedding checkpoint signature changed')
    if stored and not path.exists(): con.close(); raise ValueError('checkpoint vectors missing')
    if not stored:
        if path.exists(): con.close(); raise ValueError('orphan vectors require explicit recovery; not overwritten')
        vectors=np.memmap(path,dtype=np.float32,mode='w+',shape=(rows,1024)); vectors.flush()
        con.execute('INSERT INTO metadata VALUES(?)',(signature,)); con.commit()
    else:
        if path.stat().st_size != rows*1024*4: con.close(); raise ValueError('vector file length mismatch')
        vectors=np.memmap(path,dtype=np.float32,mode='r+',shape=(rows,1024))
    done=set()
    for start,count,expected in con.execute('SELECT start,count,sha256 FROM windows ORDER BY start'):
        if start%4 or count!=min(4,rows-start) or start<0 or start+count>rows:
            con.close(); raise ValueError('invalid checkpoint window')
        if vector_digest(vectors[start:start+count])!=expected:
            con.close(); raise ValueError('committed vector checksum mismatch')
        done.add(start)
    return con,vectors,done


def document_rows(rows, queries):
    grouped={q:[] for q in queries}
    for row in rows:
        q=int(row['query_id'])
        if q in grouped: grouped[q].append(row)
    for q, values in grouped.items():
        field='rank' if values and 'rank' in values[0] else 'candidate_rank'
        values.sort(key=lambda r:(int(r.get(field,10**9)),int(r['doc_id']),r['chunk_id']))
        selected=[]; seen=set()
        for r in values:
            doc=int(r['doc_id'])
            if doc not in seen:
                seen.add(doc); selected.append({'doc_id':doc,'chunk_id':r['chunk_id'],'rank':len(selected)+1})
        grouped[q]=selected[:100]
    return grouped


def run(root:Path):
    out=root/'artifacts/multilingual_retrieval_m42'
    manifest=validate_manifest(root); identity=runtime_identity()
    trial=json.loads((out/'trial20_execution.json').read_text(encoding='utf-8'))
    require_trial(trial,manifest,identity)
    gate=preflight(root)
    if gate['status']!='PASS': return gate
    chunks=pq.read_table(root/'artifacts/multilingual_retrieval_m4/candidate_corpus.parquet').to_pylist()
    actual=[{'doc_id':r['doc_id'],'chunk_id':r['chunk_id'],'text_sha256':hashlib.sha256(r['evidence_text'].encode()).hexdigest()} for r in chunks]
    if actual!=manifest['corpus_rows']: raise ValueError('frozen corpus row/text identity changed')
    qs=manifest['queries']; qids={int(q['query_id']) for q in qs}
    texts=[r['evidence_text'] for r in chunks]+[q['query_vi'] for q in qs]+[q['query_zh'] for q in qs]
    signature=digest({'manifest':digest(manifest),'runtime':identity,'rows':len(texts),'contract':manifest['embedding']})
    directory=out/'semantic_execution'; con,vectors,done=checkpoint_open(directory,signature,len(texts))
    def state(value):
        value['heartbeat_at']=dt.datetime.now(dt.timezone.utc).isoformat()
        _atomic_json(directory/'state.json',value)
        if os.environ.get('M42_RUN_STATE'):
            _atomic_json(Path(os.environ['M42_RUN_STATE']),value)
    started=time.perf_counter(); initial=len(done); measurements=[]
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from src.indexing.embedder import normalize_text
        if psutil.virtual_memory().available/2**20 < gate['required_warm_ram_mib']:
            raise RuntimeError('warm resource gate failed before model loading')
        torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
        load=time.perf_counter()
        model=SentenceTransformer(manifest['embedding']['name'],revision=manifest['embedding']['revision'],
                                  device='cuda',model_kwargs={'torch_dtype':torch.float16},local_files_only=True)
        model.max_seq_length=512; model_seconds=time.perf_counter()-load
        for start in range(0,len(texts),4):
            if start in done: continue
            if psutil.virtual_memory().available/2**20 < gate['required_warm_ram_mib']:
                raise RuntimeError('memory pressure; committed windows preserved, no automatic retry')
            encoded=np.asarray(model.encode([normalize_text(t) for t in texts[start:start+4]],batch_size=4,
                show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=True),dtype=np.float32)
            if encoded.shape!=(len(texts[start:start+4]),1024) or not np.isfinite(encoded).all() or np.max(np.abs(np.linalg.norm(encoded,axis=1)-1))>.001:
                raise ValueError('finite/shape/normalization gate failed')
            vectors[start:start+len(encoded)]=encoded; vectors.flush()
            con.execute('INSERT INTO windows VALUES(?,?,?)',(start,len(encoded),vector_digest(encoded))); con.commit(); done.add(start)
            info=psutil.Process().memory_info()
            measurements.append({'rss_mib':info.rss/2**20,'private_mib':getattr(info,'private',info.vms)/2**20,
                                  'gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
            state({'status':'RUNNING','stage':'DENSE_EMBEDDINGS','done':min(start+4,len(texts)),
                'total':len(texts),'signature':signature,'new_windows':len(done)-initial,'elapsed_seconds':time.perf_counter()-started})
        del model; torch.cuda.empty_cache()
        arms={}
        for name,relative in [('M4_LEXICAL','artifacts/multilingual_retrieval_m4/candidate_trace_top100.parquet'),
                              ('M4_FINAL','artifacts/multilingual_retrieval_m4/zh_m4_top20.parquet'),
                              ('M41_FINAL','artifacts/multilingual_retrieval_m41/zh_m41_top20.parquet')]:
            arms[name]=document_rows(pq.read_table(root/relative).to_pylist(),qids)
        # M4.1 did not persist its full trace. Replay unchanged CPU lexical policy,
        # requiring its final selected identities to reproduce the frozen baseline.
        from src.multilingual_retrieval.m41 import ChineseRetriever
        lexical=ChineseRetriever(chunks); arms['M41_LEXICAL']={}
        for q in qs:
            qid=int(q['query_id']); replay=lexical.retrieve(q['query_vi'],root=root)
            identities=lambda values:[(int(r['doc_id']),r['chunk_id']) for r in values]
            if identities(replay['results'])!=identities(arms['M41_FINAL'][qid]):
                raise ValueError('M4.1 lexical replay no longer matches frozen final baseline')
            arms['M41_LEXICAL'][qid]=document_rows([{**r,'query_id':qid} for r in replay['candidates']],{qid})[qid]
        for name,offset in [('VI_DENSE',len(chunks)),('ZH_DENSE',len(chunks)+len(qs))]:
            ranked=rank_dense(np.asarray(vectors[offset:offset+len(qs)]),np.asarray(vectors[:len(chunks)]),chunks,100)
            arms[name]={int(q['query_id']):r for q,r in zip(qs,ranked,strict=True)}
        arms['RRF']={q:union_rrf([arms[n][q] for n in ('M41_LEXICAL','VI_DENSE','ZH_DENSE')],100) for q in qids}
        partial=list(csv.DictReader((root/'artifacts/multilingual_retrieval_m41/partial_retention.csv').open(encoding='utf-8-sig')))
        positives={(int(r['query_id']),int(r['doc_id'])) for r in partial}
        if len(positives)!=13: raise ValueError('known partial set changed')
        found=retention(arms,positives,qids)
        positions=[]
        for q,doc in sorted(positives):
            old=next(r for r in partial if int(r['query_id'])==q and int(r['doc_id'])==doc)
            item={'query_id':q,'doc_id':doc,'old_chunk_id':old['m3_original_chunk_id'],
                  'exact_evidence_window_status':'UNVERIFIED_RECHUNKED_NO_LABEL_TRANSFER','arms':{}}
            for name,by_q in arms.items():
                row=next((r for r in by_q[q] if int(r['doc_id'])==doc),None)
                item['arms'][name]={'rank':row['rank'],'chunk_id':row['chunk_id']} if row else None
            positions.append(item)
        baseline={(q,int(r['doc_id'])) for name in ('M4_FINAL','M41_FINAL') for q,rs in arms[name].items() for r in rs[:20]}
        queue=new_evidence_review({int(q['query_id']):q['query_vi'] for q in qs},
                                 {n:arms[n] for n in ('VI_DENSE','ZH_DENSE','RRF')},baseline,chunks,12)
        _atomic_json(directory/'review_queue.json',queue)
        _atomic_json(directory/'rankings.json',{n:{str(q):rows for q,rows in g.items()} for n,g in arms.items()})
        report={'status':'DENSE_COMPLETE_PENDING_SOURCE_REVIEW','queries':12,'chunks':2188,'checkpoint_rows':len(texts),
            'runtime_identity':identity,'retention':found,'partial_positions':positions,'review_rows':len(queue),
            'model_load_seconds':model_seconds,'elapsed_seconds':time.perf_counter()-started,
            'sampled_peaks':{k:max(r[k] for r in measurements) for k in measurements[0]} if measurements else {},
            'reranker':'NOT_RUN_SEPARATE_RESOURCE_AND_19_REFERENCE_EQUIVALENCE_REQUIRED',
            'official_relevance_metrics':None,'human_verified':0}
        taxonomy=json.loads((root/'artifacts/multilingual_retrieval_m41/candidate_failure_taxonomy.json').read_text(encoding='utf-8'))
        for row in taxonomy:
            qid=int(row['query_id'])
            row['m42_diagnostic']={'query_understanding':next(q['understanding'] for q in qs if int(q['query_id'])==qid),
                'known_partial_positions':[r for r in positions if r['query_id']==qid],
                'answer_relevance':'UNKNOWN_PENDING_SOURCE_REVIEW',
                'source_coverage':'UNKNOWN_NOT_INFERRED_FROM_DENSE_FAILURE',
                'ranking_admission':'Dense Top20 is candidate discovery, not verified answer admission'}
        _atomic_json(directory/'failure_taxonomy.json',taxonomy)
        _atomic_json(directory/'dense_report.json',report)
        state({'status':'READY_FOR_HUMAN_REVIEW','stage':'DENSE_COMPLETE','done':len(texts),'total':len(texts)})
        return report
    finally:
        con.close()
