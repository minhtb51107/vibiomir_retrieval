"""Fixed-document RRF evidence-window ablation using cached Colab vectors only."""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import sys
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from src.multilingual_retrieval.m2 import sha256_file
from src.multilingual_retrieval.m41 import navigation_fragment
from src.multilingual_retrieval.m42 import digest
from tools.analyze_m42_colab import verify_archive, verify_windows, normalized

SEED='m42-fixed-doc-window-review-v1'
BEST_SHA='b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e'


def cosine_scores(matrix,query):
    matrix=np.asarray(matrix,dtype=np.float64); query=np.asarray(query,dtype=np.float64)
    norms=np.linalg.norm(matrix,axis=1); norm=np.linalg.norm(query)
    if not np.isfinite(matrix).all() or not np.isfinite(query).all() or norm==0 or np.any(norms==0):
        raise ValueError('nonfinite/zero cached vectors')
    return np.einsum('ij,j->i',matrix,query,optimize=False)/(norms*norm)


def replace_windows(control,chunks,scores):
    owners=collections.defaultdict(list)
    for i,c in enumerate(chunks): owners[int(c['doc_id'])].append(i)
    selected=[]
    for row in control:
        indices=owners.get(int(row['doc_id']))
        if not indices: raise ValueError('RRF owner absent from frozen corpus')
        best=min(indices,key=lambda i:(-float(scores[i]),str(chunks[i]['chunk_id']),i))
        selected.append({**row,'chunk_id':chunks[best]['chunk_id']})
    if [{k:v for k,v in r.items() if k!='chunk_id'} for r in selected]!=[{k:v for k,v in r.items() if k!='chunk_id'} for r in control]:
        raise ValueError('fixed-document/rank/score invariant failed')
    return selected


def navigation_proxy(row):
    text=row['evidence_text']; lines=[s.strip() for s in text.splitlines() if s.strip()]
    reasons=[]
    if navigation_fragment(text): reasons.append('SHORT_LINK_LIST')
    if lines and (lines[0] in {'参看','目录','专家答疑','常见病自测目录'} or
                  (lines[0].startswith('健康问答网关于') and lines[0].endswith('相关提问'))):
        reasons.append('NAVIGATION_HEADING')
    return bool(reasons),reasons


def select_review(pairs,maximum=24):
    changed=[r for r in pairs if r['changed']]
    key=lambda r:hashlib.sha256(f"{SEED}:{r['query_id']}:{r['doc_id']}:{r['a_chunk_id']}:{r['b_chunk_id']}".encode()).hexdigest()
    ordered=sorted(changed,key=key); selected=[]; used=set()
    def take(row):
        identity=(row['query_id'],row['doc_id'])
        if identity not in used and len(selected)<maximum:
            used.add(identity); selected.append(row)
    # Coverage first: one changed Top20 representative per development query.
    for q in sorted({r['query_id'] for r in changed}):
        candidates=[r for r in ordered if r['query_id']==q and r['rank']<=20]
        if candidates: take(candidates[0])
    # Explicitly cover both improvement/risk proxies, including deep candidates.
    for stratum in sorted({r['stratum'] for r in changed}):
        if any(r['stratum']==stratum for r in selected): continue
        candidates=[r for r in ordered if r['stratum']==stratum and (r['query_id'],r['doc_id']) not in used]
        if candidates: take(min(candidates,key=lambda r:(sum(x['query_id']==r['query_id'] for x in selected),key(r))))
    while len(selected)<maximum:
        candidates=[r for r in ordered if (r['query_id'],r['doc_id']) not in used]
        if not candidates: break
        take(min(candidates,key=lambda r:(sum(x['query_id']==r['query_id'] for x in selected),
                                         sum(x['stratum']==r['stratum'] for x in selected),key(r))))
    return selected


def csv_payload(rows,fields):
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def frozen_text(path,text):
    if path.exists():
        if path.read_bytes()!=text.encode('utf-8'): raise ValueError(f'frozen output changed: {path}')
    else: path.write_bytes(text.encode('utf-8'))


def main():
    started=time.perf_counter(); process=psutil.Process(); before=process.memory_info(); samples=[]; stop=threading.Event()
    def sample():
        while not stop.is_set():
            info=process.memory_info()
            samples.append((info.rss,getattr(info,'private',info.vms),psutil.virtual_memory().available))
            stop.wait(.05)
    monitor=threading.Thread(target=sample,daemon=True); monitor.start()
    try: return execute(started,process,before,samples)
    finally: stop.set(); monitor.join(timeout=2)


def execute(started,process,before,samples):
    base=ROOT/'artifacts/multilingual_retrieval_m42'; out=base/'window_ablation'; out.mkdir(parents=True,exist_ok=True)
    returned=base/'colab_results'; run=returned/'extracted/artifacts/multilingual_retrieval_m42'; semantic=run/'semantic_execution'
    best=ROOT/'submissions/00_READY_TO_UPLOAD/p27_sdso20749_cap16.zip'
    if sha256_file(best)!=BEST_SHA: raise ValueError('Phase27 BEST changed')
    integrity=verify_archive(returned/'M42_Colab_Dense_Results.zip',returned/'extracted')
    manifest=json.loads((base/'frozen_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['inputs_sha256'].items():
        if sha256_file(ROOT/name)!=expected: raise ValueError(f'frozen input changed: {name}')
    trial=json.loads((run/'trial20_execution.json').read_text(encoding='utf-8'))
    if trial.get('passed') is not True or trial['trial20_sha256']!=manifest['trial20_sha256']:
        raise ValueError('frozen trial not passed')
    path=semantic/'colab_embedding_checkpoint/vectors.f32'
    if path.stat().st_size!=2212*1024*4: raise ValueError('vector file length mismatch')
    vectors=np.memmap(path,dtype=np.float32,mode='r',shape=(2212,1024))
    integrity['checkpoint']=verify_windows(path.with_name('embedding_checkpoint.sqlite'),vectors)
    if not np.isfinite(vectors).all() or np.max(np.abs(np.linalg.norm(vectors,axis=1)-1))>.001:
        raise ValueError('cached normalization/finite failure')
    chunks=pq.read_table(ROOT/'artifacts/multilingual_retrieval_m4/candidate_corpus.parquet').to_pylist()
    actual=[{'doc_id':c['doc_id'],'chunk_id':c['chunk_id'],'text_sha256':hashlib.sha256(c['evidence_text'].encode()).hexdigest()} for c in chunks]
    if actual!=manifest['corpus_rows'] or len(chunks)!=2188: raise ValueError('corpus/order/provenance changed')
    by_chunk={c['chunk_id']:c for c in chunks}; index={c['chunk_id']:i for i,c in enumerate(chunks)}
    official={}
    for batch in ds.dataset(ROOT/'data/raw/links_corpus_vi_en_zh.parquet').scanner(columns=['id','url'],
            filter=ds.field('id').isin(sorted({int(c['doc_id']) for c in chunks})),batch_size=256,use_threads=False).to_batches():
        for r in batch.to_pylist(): official[int(r['id'])]=r['url']
    if any(official.get(int(c['doc_id']))!=c['canonical_url'] for c in chunks): raise ValueError('official URL/owner mismatch')
    integrity['official_documents']=len(official); integrity_seconds=time.perf_counter()-started
    saved=json.loads((semantic/'rankings.json').read_text(encoding='utf-8')); controls=saved['RRF']
    if set(controls)!={str(q['query_id']) for q in manifest['queries']}: raise ValueError('query-set mismatch')
    pairs=[]; challenger={}; query_map={int(q['query_id']):q['query_vi'] for q in manifest['queries']}
    for i,q in enumerate(manifest['queries']):
        qid=int(q['query_id']); control=controls[str(qid)]
        if len(control)!=100 or len({r['doc_id'] for r in control})!=100: raise ValueError('control doc budget/uniqueness')
        scores=cosine_scores(vectors[:2188],vectors[2200+i])
        selected=replace_windows(control,chunks,scores)
        if selected!=replace_windows(control,chunks,scores): raise ValueError('non-deterministic selection')
        challenger[str(qid)]=selected
        for rank,(a,b) in enumerate(zip(control,selected,strict=True),1):
            old,new=by_chunk[a['chunk_id']],by_chunk[b['chunk_id']]
            if int(old['doc_id'])!=int(a['doc_id']) or int(new['doc_id'])!=int(a['doc_id']) or rank!=a['rank']:
                raise ValueError('owner/rank invariant')
            an,ar=navigation_proxy(old); bn,br=navigation_proxy(new)
            if float(scores[index[b['chunk_id']]])+1e-15<float(scores[index[a['chunk_id']]]): raise ValueError('not best cosine')
            transition=('NAV' if an else 'BODY')+'_TO_'+('NAV' if bn else 'BODY')
            pairs.append({'query_id':qid,'rank':rank,'doc_id':a['doc_id'],'canonical_url':old['canonical_url'],
                'a_chunk_id':a['chunk_id'],'b_chunk_id':b['chunk_id'],'changed':a['chunk_id']!=b['chunk_id'],
                'rrf_score_unchanged':a['rrf_score'],'a_cosine':float(scores[index[a['chunk_id']]]),
                'b_cosine':float(scores[index[b['chunk_id']]]),'a_navigation_proxy':an,'b_navigation_proxy':bn,
                'a_navigation_reasons':'|'.join(ar),'b_navigation_reasons':'|'.join(br),
                'cached_document_page_type':old['page_type'],
                'stratum':('TOP20' if rank<=20 else 'RANK21_100')+':'+transition,
                'a_text_sha256':hashlib.sha256(old['evidence_text'].encode()).hexdigest(),
                'b_text_sha256':hashlib.sha256(new['evidence_text'].encode()).hexdigest(),
                'a_evidence_text':old['evidence_text'],'b_evidence_text':new['evidence_text']})
    selected=select_review(pairs); mapping={}; queue=[]
    for row in selected:
        case=digest([SEED,row['query_id'],row['doc_id'],row['a_chunk_id'],row['b_chunk_id']])[:20]
        swap=int(hashlib.sha256((SEED+':side:'+case).encode()).hexdigest()[-1],16)%2
        left,right=('b','a') if swap else ('a','b')
        queue.append({'case_id':case,'query_vi':query_map[row['query_id']],'doc_id':row['doc_id'],
            'canonical_url':row['canonical_url'],'evidence_left':row[left+'_evidence_text'],
            'evidence_right':row[right+'_evidence_text'],'left_judgment':'','right_judgment':'','preference':'',
            'review_notes':'','verification_status':'PENDING_INDEPENDENT_REVIEW'})
        mapping[case]={k:row[k] for k in ('query_id','doc_id','rank','a_chunk_id','b_chunk_id','stratum')}
        mapping[case].update(left_arm=left.upper(),right_arm=right.upper())
    queue.sort(key=lambda r:r['case_id'])
    frozen_text(out/'blind_review_queue.csv',csv_payload(queue,list(queue[0])))
    frozen_text(out/'review_mapping.json',json.dumps(mapping,ensure_ascii=False,sort_keys=True,indent=2))
    protocol={'seed':SEED,'selection':'One hash-selected Top20 case per query; cover depth x navigation transition strata; fill24 balancing query/stratum counts',
        'case_ids':[r['case_id'] for r in queue],'maximum':24,
        'inputs':{'frozen_manifest_sha256':sha256_file(base/'frozen_manifest.json'),
                  'vector_sha256':sha256_file(path),'rankings_sha256':sha256_file(semantic/'rankings.json')},
        'queue_sha256':sha256_file(out/'blind_review_queue.csv'),'mapping_sha256':sha256_file(out/'review_mapping.json'),
        'implementation_sha256':sha256_file(Path(__file__)),
        'clinical_judgments_used_for_selection':False,
        'promotion':'No automatic promotion; new independent source review required, not cosine or silver retention alone'}
    frozen_text(out/'review_selection.json',json.dumps(protocol,ensure_ascii=False,sort_keys=True,indent=2))
    frozen_text(out/'paired_comparison.csv',csv_payload(pairs,list(pairs[0])))
    # Labels are loaded only AFTER selection and review queue have been frozen.
    with (ROOT/'artifacts/multilingual_retrieval_m41/partial_retention.csv').open(encoding='utf-8-sig') as f: partial=list(csv.DictReader(f))
    with (ROOT/'artifacts/multilingual_retrieval_m3/review/m3_ai_review_full.csv').open(encoding='utf-8-sig') as f: oldreview=list(csv.DictReader(f))
    known=[]
    for old in partial:
        q=str(old['query_id']); doc=int(old['doc_id'])
        reference=[r for r in oldreview if r['query_id']==q and int(r['doc_id'])==doc and r['chunk_id']==old['m3_original_chunk_id']]
        if len(reference)!=1: raise ValueError('old silver window unresolved')
        text=reference[0]['evidence_text']; record={'query_id':int(q),'doc_id':doc,'arms':{}}
        for arm,rows in [('A',controls[q]),('B',challenger[q])]:
            found=next((r for r in rows if int(r['doc_id'])==doc),None)
            record['arms'][arm]={'rank':found['rank'] if found else None,
                'chunk_id':found['chunk_id'] if found else None,
                'exact_old_excerpt':bool(found and text in by_chunk[found['chunk_id']]['evidence_text']),
                'normalized_old_excerpt':bool(found and normalized(text) in normalized(by_chunk[found['chunk_id']]['evidence_text']))}
        known.append(record)
    metrics={}
    for k in (20,100):
        rows=[r for r in pairs if r['rank']<=k]
        stats={'rows':len(rows),'changed':sum(r['changed'] for r in rows),
               'changed_queries':len({r['query_id'] for r in rows if r['changed']}),
               'navigation_a':sum(r['a_navigation_proxy'] for r in rows),'navigation_b':sum(r['b_navigation_proxy'] for r in rows),
               'changed_transitions':dict(collections.Counter(r['stratum'].split(':')[1] for r in rows if r['changed']))}
        for arm in ('A','B'):
            present=[r['arms'][arm] for r in known if r['arms'][arm]['rank'] and r['arms'][arm]['rank']<=k]
            stats[arm]={'partial_document_pairs':len(present),'exact_old_windows':sum(r['exact_old_excerpt'] for r in present),
                        'normalized_old_windows':sum(r['normalized_old_excerpt'] for r in present)}
        metrics[str(k)]=stats
    if sha256_file(best)!=BEST_SHA: raise ValueError('BEST mutated')
    if integrity['zip_sha256']!=sha256_file(returned/'M42_Colab_Dense_Results.zip'): raise ValueError('Colab inputs mutated')
    after=process.memory_info()
    report={'status':'COMPLETE_PENDING_INDEPENDENT_REVIEW','policy':'m42-fixed-doc-best-true-cosine-zh-v1',
        'selection':'Same-owner maximum true cosine in float64; lexicographic chunk_id then corpus index tie-break',
        'queries':12,'frozen_chunks':2188,'official_owners':len(official),'document_id_order_score_invariant':True,
        'metrics':metrics,'partial_window_positions':known,'integrity':integrity,
        'review':{'pairs':len(queue),'queries':len({v['query_id'] for v in mapping.values()}),
                  'strata':dict(collections.Counter(v['stratum'] for v in mapping.values())),
                  'selection_sha256':sha256_file(out/'review_selection.json'),'clinical_judgments_used':False},
        'output_hashes':{n:sha256_file(out/n) for n in ('paired_comparison.csv','blind_review_queue.csv','review_mapping.json')},
        'timing':{'integrity_seconds':integrity_seconds,'total_seconds':time.perf_counter()-started},
        'memory':{'rss_start_mib':before.rss/2**20,'rss_end_mib':after.rss/2**20,
                  'process_peak_working_set_mib':getattr(after,'peak_wset',after.rss)/2**20,
                  'peak_private_sampled_mib':max(s[1] for s in samples)/2**20,
                  'minimum_available_ram_sampled_mib':min(s[2] for s in samples)/2**20},
        'human_verified':0,'held_out_used':False,'gpu_inference':False,'reranker_score_reuse':False,
        'phase27_sha256':BEST_SHA,'decision':'NO_GO_AUTOMATIC_PROMOTION_PENDING_SOURCE_REVIEW'}
    (out/'ablation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('partial_window_positions','integrity')},ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
