"""Read-only result verification and cached-vector arithmetic; no model inference."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import sqlite3
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from src.multilingual_retrieval.m42 import digest, rank_dense, union_rrf
from src.multilingual_retrieval.m2 import sha256_file


def verify_archive(archive, extracted):
    checked=[]
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError('ZIP CRC failed')
        for member in z.infolist():
            if member.is_dir(): continue
            name=Path(member.filename)
            target=(extracted/name).resolve()
            if not target.is_relative_to(extracted.resolve()): raise ValueError('unsafe ZIP member')
            h=hashlib.sha256()
            with z.open(member) as f:
                for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
            if sha256_file(target)!=h.hexdigest(): raise ValueError(f'extracted member differs: {name}')
            checked.append(member.filename)
    actual={p.relative_to(extracted).as_posix() for p in extracted.rglob('*') if p.is_file()}
    return {'zip_sha256':sha256_file(archive),'crc':'PASS','members_matched':len(checked),
            'extra_extracted_files':sorted(actual-set(checked))}


def verify_windows(db, vectors):
    con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        checks={name:con.execute(f'PRAGMA {name}').fetchone()[0] for name in ('quick_check','integrity_check')}
        if any(v!='ok' for v in checks.values()): raise ValueError('SQLite integrity failure')
        cursor=0; windows=0
        for start,count,expected in con.execute('SELECT start,count,sha256 FROM windows ORDER BY start'):
            if start!=cursor or count!=min(4,len(vectors)-start): raise ValueError('checkpoint coverage gap/overlap')
            if hashlib.sha256(np.asarray(vectors[start:start+count],dtype=np.float32).tobytes()).hexdigest()!=expected:
                raise ValueError('checkpoint window checksum failed')
            cursor+=count; windows+=1
        if cursor!=len(vectors): raise ValueError('incomplete embedding checkpoint')
        signatures=con.execute('SELECT signature FROM metadata').fetchall()
        if len(signatures)!=1: raise ValueError('checkpoint signature cardinality')
        return {**checks,'windows':windows,'rows_verified':cursor,'signature':signatures[0][0]}
    finally: con.close()


def positions(rows):
    return {(int(r['doc_id']),str(r['chunk_id'])):i for i,r in enumerate(rows,1)}


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC',text).split())


def reconstruct(rows):
    end=max(int(r['end_offset']) for r in rows)
    value=['\0']*end
    for row in rows:
        a,b=int(row['start_offset']),int(row['end_offset']); text=row['evidence_text']
        if b-a!=len(text): raise ValueError('source offset length mismatch')
        for i,char in enumerate(text,a):
            if value[i]!='\0' and value[i]!=char: raise ValueError('inconsistent source overlap')
            value[i]=char
    return ''.join(value)


def main():
    base=ROOT/'artifacts/multilingual_retrieval_m42'
    downloaded=base/'colab_results'
    extracted=downloaded/'extracted'
    run=extracted/'artifacts/multilingual_retrieval_m42'
    semantic=run/'semantic_execution'
    out=base/'colab_analysis'; out.mkdir(parents=True,exist_ok=True)
    validation=verify_archive(downloaded/'M42_Colab_Dense_Results.zip',extracted)
    trial=json.loads((run/'trial20_execution.json').read_text(encoding='utf-8'))
    dense=json.loads((semantic/'dense_report.json').read_text(encoding='utf-8'))
    manifest=json.loads((base/'frozen_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['inputs_sha256'].items():
        if sha256_file(ROOT/name)!=expected: raise ValueError(f'frozen source changed: {name}')
    if trial.get('passed') is not True or trial.get('status')!='PASS': raise ValueError('trial not confirmed PASS')
    if trial['trial20_sha256']!=manifest['trial20_sha256'] or digest(manifest['trial20'])!=manifest['trial20_sha256']:
        raise ValueError('frozen trial digest mismatch')
    if dense['status']!='DENSE_COMPLETE_PENDING_SOURCE_REVIEW' or dense['checkpoint_rows']!=2212:
        raise ValueError('Dense completion not confirmed')
    if trial['runtime_identity']!=dense['runtime_identity']: raise ValueError('runtime drift inside Colab run')
    if json.loads((semantic/'colab_runtime_identity.json').read_text(encoding='utf-8'))!=dense['runtime_identity']:
        raise ValueError('runtime identity file differs from report')
    chunks=pq.read_table(ROOT/'artifacts/multilingual_retrieval_m4/candidate_corpus.parquet').to_pylist()
    if [{'doc_id':r['doc_id'],'chunk_id':r['chunk_id'],'text_sha256':hashlib.sha256(r['evidence_text'].encode()).hexdigest()} for r in chunks]!=manifest['corpus_rows']:
        raise ValueError('corpus row identity mismatch')
    chunkmap={r['chunk_id']:r for r in chunks}; queries=manifest['queries']; qids={str(q['query_id']) for q in queries}
    path=semantic/'colab_embedding_checkpoint/vectors.f32'
    if path.stat().st_size!=2212*1024*4: raise ValueError('vector file length mismatch')
    vectors=np.memmap(path,dtype=np.float32,mode='r',shape=(2212,1024))
    validation['checkpoint']=verify_windows(path.with_name('embedding_checkpoint.sqlite'),vectors)
    validation['checkpoint_signature_matches_progress']=validation['checkpoint']['signature']==json.loads((semantic/'colab_progress.json').read_text(encoding='utf-8'))['signature']
    validation['local_signature_formula_matches']=digest({'manifest':digest(manifest),'runtime':dense['runtime_identity'],'rows':2212,'contract':manifest['embedding']})==validation['checkpoint']['signature']
    norms=np.linalg.norm(vectors,axis=1)
    validation['vectors']={'shape':list(vectors.shape),'finite':bool(np.isfinite(vectors).all()),
                          'min_norm':float(norms.min()),'max_norm':float(norms.max())}
    if not validation['vectors']['finite'] or np.max(np.abs(norms-1))>.001: raise ValueError('vector finite/norm failure')
    rankings=json.loads((semantic/'rankings.json').read_text(encoding='utf-8'))
    partial=list(csv.DictReader((ROOT/'artifacts/multilingual_retrieval_m41/partial_retention.csv').open(encoding='utf-8-sig')))
    positive={(int(r['query_id']),int(r['doc_id'])) for r in partial}
    recomputed={}; page_types={}; rank_checks={}; stats={}
    for arm,by_query in rankings.items():
        if set(by_query)!=qids: raise ValueError('query-set mismatch')
        pages=collections.Counter(); armstats={}
        for q,rows in by_query.items():
            if len(rows)>100 or len({int(r['doc_id']) for r in rows})!=len(rows): raise ValueError('budget/doc duplicate')
            for i,r in enumerate(rows,1):
                c=chunkmap[r['chunk_id']]
                if int(r['doc_id'])!=int(c['doc_id']) or int(r['rank'])!=i: raise ValueError('rank/owner mismatch')
                if i<=20: pages[c['page_type']]+=1
        for k in (20,100):
            found={(int(q),int(r['doc_id'])) for q,rows in by_query.items() for r in rows[:k]}&positive
            armstats[str(k)]={'known_partial_pairs_found':len(found),'queries_with_known_partial':len({q for q,d in found}),
                             'observed_rows':sum(len(rows[:k]) for rows in by_query.values())}
            if len(found)!=dense['retention'][arm][str(k)]['known_partial_pairs_found']: raise ValueError('reported retention mismatch')
        recomputed[arm]=armstats; page_types[arm]=dict(pages)
    for arm,offset in [('VI_DENSE',2188),('ZH_DENSE',2200)]:
        fresh=rank_dense(np.asarray(vectors[offset:offset+12]),np.asarray(vectors[:2188]),chunks,100)
        identity_diffs=0; max_score_diff=0.
        for q,values in zip(queries,fresh,strict=True):
            saved=rankings[arm][str(q['query_id'])]
            identity_diffs+=sum((x['doc_id'],x['chunk_id'])!=(y['doc_id'],y['chunk_id']) for x,y in zip(values,saved,strict=True))
            max_score_diff=max(max_score_diff,max(abs(r['dense_score']-s['dense_score']) for r,s in zip(values,saved,strict=True)))
        rank_checks[arm]={'top100_identity_position_diffs':identity_diffs,'max_score_difference_cpu_replay':max_score_diff}
    rrf_diffs=0
    for q in qids:
        fresh=union_rrf([rankings[n][q] for n in ('M41_LEXICAL','VI_DENSE','ZH_DENSE')],100)
        rrf_diffs+=sum((r['doc_id'],r['chunk_id'])!=(s['doc_id'],s['chunk_id']) for r,s in zip(fresh,rankings['RRF'][q],strict=True))
    rank_checks['RRF']={'top100_identity_position_diffs':rrf_diffs}
    validation['ranking_arithmetic_replay']=rank_checks
    validation['frozen_final_baselines']={}
    for arm,relative in [('M4_FINAL','multilingual_retrieval_m4/zh_m4_top20.parquet'),('M41_FINAL','multilingual_retrieval_m41/zh_m41_top20.parquet')]:
        frozen=pq.read_table(ROOT/'artifacts'/relative).to_pylist(); diff=0
        for q in qids:
            rows=sorted((r for r in frozen if str(r['query_id'])==q),key=lambda r:r['rank'])
            diff+=int([(int(r['doc_id']),r['chunk_id']) for r in rows]!=[(int(r['doc_id']),r['chunk_id']) for r in rankings[arm][q]])
        validation['frozen_final_baselines'][arm]={'queries_differing':diff}
    def check_partial_positions(rows):
        if {(int(r['query_id']),int(r['doc_id'])) for r in rows}!=positive or len(rows)!=13:
            raise ValueError('reported partial identity set changed')
        for row in rows:
            for arm,by_query in rankings.items():
                actual=next((r for r in by_query[str(row['query_id'])] if int(r['doc_id'])==int(row['doc_id'])),None)
                expected={'rank':actual['rank'],'chunk_id':actual['chunk_id']} if actual else None
                if row['arms'][arm]!=expected: raise ValueError('reported partial position differs from ranking')
    check_partial_positions(dense['partial_positions'])
    taxonomy=json.loads((semantic/'failure_taxonomy.json').read_text(encoding='utf-8'))
    inherited=json.loads((ROOT/'artifacts/multilingual_retrieval_m41/candidate_failure_taxonomy.json').read_text(encoding='utf-8'))
    if [{k:v for k,v in row.items() if k!='m42_diagnostic'} for row in taxonomy]!=inherited:
        raise ValueError('inherited failure taxonomy changed')
    check_partial_positions([r for row in taxonomy for r in row['m42_diagnostic']['known_partial_positions']])
    validation['failure_taxonomy']={'units_verified':len(taxonomy),'inherited_m41_unchanged':True,'partial_positions_match':True}
    # Exact old evidence-window retention, separately from document identity.
    with (ROOT/'artifacts/multilingual_retrieval_m3/review/m3_ai_review_full.csv').open(encoding='utf-8-sig') as f:
        oldreview=list(csv.DictReader(f))
    pairrows=[]
    for old in partial:
        q=str(old['query_id']); doc=int(old['doc_id'])
        reference=[r for r in oldreview if r['query_id']==q and int(r['doc_id'])==doc and r['chunk_id']==old['m3_original_chunk_id']]
        if len(reference)!=1: raise ValueError('old partial evidence identity unresolved')
        text=reference[0]['evidence_text']; row={'query_id':int(q),'doc_id':doc,'old_chunk_id':old['m3_original_chunk_id'],'arms':{}}
        owner=[c for c in chunks if int(c['doc_id'])==doc]; assembled=reconstruct(owner)
        row['old_window_availability']={'exact_single':any(text in c['evidence_text'] for c in owner),
            'normalized_single':any(normalized(text) in normalized(c['evidence_text']) for c in owner),
            'exact_reconstructed':text in assembled,'normalized_reconstructed':normalized(text) in normalized(assembled)}
        for arm,by_query in rankings.items():
            candidate=next((r for r in by_query[q] if int(r['doc_id'])==doc),None)
            row['arms'][arm]={'rank':candidate['rank'] if candidate else None,
                'exact_old_excerpt_in_selected_chunk':bool(candidate and text in chunkmap[candidate['chunk_id']]['evidence_text']),
                'normalized_old_excerpt_in_selected_chunk':bool(candidate and normalized(text) in normalized(chunkmap[candidate['chunk_id']]['evidence_text'])),
                'exact_old_excerpt_in_any_owner_chunk':any(text in c['evidence_text'] for c in chunks if int(c['doc_id'])==doc)}
        pairrows.append(row)
    review=json.loads((semantic/'review_queue.json').read_text(encoding='utf-8'))
    review_identity=[]
    for row in review:
        source=chunkmap[row['chunk_id']]
        if row['evidence_text']!=source['evidence_text'] or int(row['doc_id'])!=int(source['doc_id']) or row['canonical_url']!=source['canonical_url']:
            raise ValueError('review evidence/source mismatch')
        review_identity.append({'query_id':row['query_id'],'doc_id':row['doc_id'],'chunk_id':row['chunk_id'],
            'page_type':source['page_type'], 'arms':{a:next((r['rank'] for r in byq[str(row['query_id'])] if r['chunk_id']==row['chunk_id']),None) for a,byq in rankings.items()}})
    official={}
    for batch in ds.dataset(ROOT/'data/raw/links_corpus_vi_en_zh.parquet').scanner(columns=['id','url'],
            filter=ds.field('id').isin(sorted({int(r['doc_id']) for r in chunks})),batch_size=256,use_threads=False).to_batches():
        for r in batch.to_pylist(): official[int(r['id'])]=r['url']
    if any(official.get(int(r['doc_id']))!=r['canonical_url'] for r in chunks): raise ValueError('current official URL/membership mismatch')
    validation['official_docs']=len(official); validation['review_exact_provenance_passed']=len(review)
    trialvectors=np.load(run/'trial20_vectors.npy',allow_pickle=False)
    validation['trial_vectors']={'shape':list(trialvectors.shape),'finite':bool(np.isfinite(trialvectors).all()),
                                 'max_norm_error':float(np.max(np.abs(np.linalg.norm(trialvectors,axis=1)-1)))}
    if trialvectors.shape!=(20,1024) or not validation['trial_vectors']['finite']: raise ValueError('invalid trial vectors')
    trial_indices=[]
    for t in manifest['trial20']:
        if t['kind']=='CHUNK': index=next(i for i,c in enumerate(chunks) if c['chunk_id']==t['chunk_id'])
        else:
            offset=2188 if t['kind']=='QUERY_VI' else 2200
            index=offset+next(i for i,q in enumerate(queries) if int(q['query_id'])==int(t['query_id']))
        trial_indices.append(index)
    subset=np.asarray(vectors[trial_indices])
    validation['trial_to_dense_same_run_consistency']={'maximum_absolute_difference':float(np.max(np.abs(subset-trialvectors))),
        'minimum_cosine':float(np.min(np.sum(subset*trialvectors,axis=1)/(np.linalg.norm(subset,axis=1)*np.linalg.norm(trialvectors,axis=1))))}
    perquery=[]; representatives=collections.Counter()
    for q in queries:
        key=str(q['query_id']); item={'query_id':q['query_id'],'arms':{}}
        for arm,by_query in rankings.items():
            rows=by_query[key]
            item['arms'][arm]={'known_partial_top20':sum((int(key),int(r['doc_id'])) in positive for r in rows[:20]),
                'directory_tagged_top20':sum(chunkmap[r['chunk_id']]['page_type']=='DIRECTORY_OR_NAVIGATION' for r in rows[:20]),
                'novel_docs_top20_vs_m41_candidates20':len({r['doc_id'] for r in rows[:20]}-{r['doc_id'] for r in rankings['M41_LEXICAL'][key][:20]})}
        for row in rankings['RRF'][key][:20]:
            other=next((r for r in rankings['ZH_DENSE'][key] if int(r['doc_id'])==int(row['doc_id'])),None)
            if other:
                representatives['same_document_in_zh_top100']+=1
                if other['chunk_id']!=row['chunk_id']:
                    representatives['different_representative_chunk']+=1
                    if chunkmap[row['chunk_id']]['page_type']=='DIRECTORY_OR_NAVIGATION' and chunkmap[other['chunk_id']]['page_type']=='MEDICAL_ARTICLE':
                        representatives['article_tag_to_directory_tag']+=1
        perquery.append(item)
    validation['source_target_equivalence']='NOT_VERIFIED_DIFFERENT_RUNTIME_NO_SCORE_REUSE'
    report={'validation':validation,'runtime_reported':dense['runtime_identity'],'trial_reported':trial,
            'dense_elapsed_seconds':dense['elapsed_seconds'],'recomputed_retention':recomputed,
            'page_types_top20':page_types,'known_partial_positions_and_windows':pairrows,
            'review_queue_identity':review_identity,'per_query':perquery,
            'rrf_representative_diagnostics':dict(representatives),'held_out_used':False,'new_inference':False}
    assessment=out/'assistant_review_12.csv'
    if assessment.exists():
        with assessment.open(encoding='utf-8-sig') as f: judgments=list(csv.DictReader(f))
        keys=lambda rows:{(int(r['query_id']),int(r['doc_id']),r['chunk_id']) for r in rows}
        if len(judgments)!=12 or keys(judgments)!=keys(review): raise ValueError('assistant review identity mismatch')
        if any(r['verification_status']!='AI_JUDGED_M42_ASSISTANT_UNBLINDED' for r in judgments):
            raise ValueError('assistant label provenance changed')
        report['provisional_assistant_review']={'sha256':sha256_file(assessment),
            'counts':dict(collections.Counter(r['provisional_assessment'] for r in judgments)),
            'human_verified':0,'blind':False,'official_metric':False}
    (out/'verified_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ['validation','recomputed_retention','rrf_representative_diagnostics']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
