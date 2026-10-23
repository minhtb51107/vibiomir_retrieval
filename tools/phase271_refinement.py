"""Development-only cached ablations; frozen submissions are read-only inputs."""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pyarrow.dataset as ds
from tools.phase27_final_relevance_check import records, chunk_key, ngrams
from src.multilingual_retrieval.m2 import sha256_file

P27 = 'b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e'
P29 = 'e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04'


def promotional(row):
    """Dominant boilerplate only; preserve clinical articles with small footers."""
    parts = [s.strip() for s in re.split(r'[\n.!?]+', row['chunk_text']) if s.strip()]
    markers = ('đọc ngay', 'video nhiều người', 'mời đón xem', 'địa chỉ:',
               'hotline', 'điện thoại:', 'đặt lịch', 'dịch vụ y tế', 'bảo hiểm y tế')
    flagged = [p for p in parts if any(m in p.casefold() for m in markers)]
    density = sum(map(len, flagged)) / max(1, sum(map(len, parts)))
    return len(flagged) >= 2 and density >= .5


def overlap(a, b):
    if int(a['doc_id']) != int(b['doc_id']):
        return False
    aa, bb = ngrams(a['chunk_text']), ngrams(b['chunk_text'])
    return bool(aa and bb and len(aa & bb) / min(len(aa), len(bb)) >= .8)


def refine(original, fallback, policy):
    """Keep Docs20 fixed; backfill chunks in frozen order, never invent scores."""
    selected, deferred, seen = [], [], set()
    rejected = collections.Counter()
    for row in original + fallback:
        key = chunk_key(row)
        if key in seen:
            continue
        seen.add(key)
        reason = None
        if policy == 'quality' and promotional(row):
            reason = 'DOMINANT_BOILERPLATE'
        if policy == 'redundancy' and any(overlap(row, old) for old in selected):
            reason = 'SAME_OWNER_80PCT_OVERLAP'
        if reason:
            rejected[reason] += 1
            deferred.append(row)
        else:
            selected.append(row)
        if len(selected) == 40:
            break
    # Never silently lower output depth; expose inability to obtain 40 safe chunks.
    padding = 40 - len(selected)
    selected += deferred[:padding]
    if len(selected) != 40:
        raise ValueError('insufficient distinct frozen evidence')
    return selected, dict(rejected), padding


def main():
    started = time.perf_counter()
    out = ROOT / 'artifacts/phase271_cached_refinement'
    out.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'submissions/00_READY_TO_UPLOAD' / f for f in
             ('p27_sdso20749_cap16.zip', 'p29_bachmai2679_cap16.zip')]
    for path, expected in zip(paths, (P27, P29)):
        if sha256_file(path) != expected:
            raise ValueError('frozen ZIP changed')
    review = ds.dataset(ROOT / 'artifacts/multilingual_retrieval_m1/review_queue.parquet').to_table(
        filter=ds.field('split') == 'development').to_pylist()
    dev = {int(r['query_id']) for r in review}
    labels = []
    for name in ('artifacts/multilingual_retrieval_m1/ai_judgments_import.csv',
                 'artifacts/multilingual_retrieval_m2/ai_judgments_m2.csv'):
        with (ROOT / name).open(encoding='utf-8-sig', newline='') as f:
            labels += [r for r in csv.DictReader(f) if int(r['query_id']) in dev and r['judgment'] in {'1','2'}]
    # Passage retention is assessed only where a silver case has an exact review excerpt.
    by_case = {r['case_id']: r for r in review}
    result = {name: collections.Counter() for name in ('redundancy', 'quality')}
    detail, examples, source_delta = [], [], {n: collections.Counter() for n in result}
    urls = {int(r['doc_id']): r['canonical_url'] for r in review}
    from urllib.parse import urlsplit
    def source(row):
        return urlsplit(urls.get(int(row['doc_id']), '')).hostname or 'UNKNOWN'
    with zipfile.ZipFile(paths[0]) as za, zipfile.ZipFile(paths[1]) as zb:
        with io.TextIOWrapper(za.open(za.namelist()[0]), encoding='utf-8') as a, io.TextIOWrapper(zb.open(zb.namelist()[0]), encoding='utf-8') as b:
            for new, old in zip(records(a), records(b), strict=True):
                q = int(new['id'])
                if q != int(old['id']): raise ValueError('query alignment')
                if q not in dev: continue
                orig = new['relevant_chunks']; orig_keys = {chunk_key(c) for c in orig}
                for policy, stats in result.items():
                    chosen, reasons, padding = refine(orig, old['relevant_chunks'], policy)
                    keys = {chunk_key(c) for c in chosen}
                    stats.update(queries=1, changed_queries=chosen != orig,
                                 inserted_slots=len(keys-orig_keys), displaced_slots=len(orig_keys-keys),
                                 fallback_padding_slots=padding, docs_changed=0)
                    stats['top5_original_objects_lost_from_top5'] += sum(chunk_key(c) not in {chunk_key(x) for x in chosen[:5]} for c in orig[:5])
                    stats['top5_original_objects_lost_from_final40'] += sum(chunk_key(c) not in keys for c in orig[:5])
                    stats.update(reasons)
                    removed = [c for c in orig if chunk_key(c) not in keys]
                    added = [c for c in chosen if chunk_key(c) not in orig_keys]
                    source_delta[policy].update(source(c) for c in added)
                    source_delta[policy].subtract(source(c) for c in removed)
                    for label in labels:
                        if int(label['query_id']) != q: continue
                        doc = int(label['doc_id'])
                        present = doc in new['relevant_docs']
                        stats['silver_docs_baseline_present'] += present
                        stats['silver_docs_retained'] += present  # Docs20 immutable in both arms.
                        case = by_case.get(label.get('case_id'))
                        if case and case.get('evidence_text'):
                            text = case['evidence_text']
                            supported = lambda rows: any(int(c['doc_id']) == doc and text in c['chunk_text'] for c in rows)
                            was = supported(orig)
                            stats['silver_exact_excerpt_baseline_present'] += was
                            stats['silver_exact_excerpt_retained'] += was and supported(chosen)
                    detail.append({'query_id': q, 'policy': policy, 'reasons': reasons,
                                   'inserted_slots': len(keys-orig_keys), 'padding': padding,
                                   'chunks': chosen, 'relevant_docs': new['relevant_docs']})
                    examples.append({'query_id':q, 'policy':policy,
                                     'removed': [{'doc_id':c['doc_id'], 'text_sha256':hashlib.sha256(c['chunk_text'].encode()).hexdigest()} for c in removed],
                                     'added': [{'doc_id':c['doc_id'], 'text_sha256':hashlib.sha256(c['chunk_text'].encode()).hexdigest()} for c in added]})
    # Resolve source identity from current official membership, not sparse silver metadata.
    needed = {int(c['doc_id']) for r in detail for c in r['chunks']}
    needed |= {int(c['doc_id']) for r in examples for c in r['removed']}
    for batch in ds.dataset(ROOT/'data/raw/links_corpus_vi_en_zh.parquet').scanner(
            columns=['id','url'],filter=ds.field('id').isin(sorted(needed)),batch_size=256,use_threads=False).to_batches():
        for row in batch.to_pylist(): urls[int(row['id'])]=row['url']
    if needed-set(urls): raise ValueError('non-official candidate owner')
    source_delta={n:collections.Counter() for n in result}
    for r in examples:
        source_delta[r['policy']].update(source(c) for c in r['added'])
        source_delta[r['policy']].subtract(source(c) for c in r['removed'])
    # A second complete policy pass is tested separately; hash serializes deterministic results.
    payload = json.dumps(detail, ensure_ascii=False, sort_keys=True)
    (out / 'development_candidates.json').write_text(payload, encoding='utf-8')
    report = {'version':'phase271-dev-v1', 'control_sha256':P27, 'fallback_sha256':P29,
              'development_queries':len(dev), 'held_out_used':False, 'new_inference':False,
              'policies':{k:dict(v) for k,v in result.items()},
              'candidate_sha256':hashlib.sha256(payload.encode()).hexdigest(),
              'source_delta':{k:dict(v) for k,v in source_delta.items()}, 'examples':examples,
              'score_provenance':'Frozen final rank priority reused; no numeric score assigned to expanded text',
              'relevance_status':'UNVERIFIED; no improvement metric inferred from deduplication',
              'decision':'NO_GO_SUBMISSION_PENDING_INDEPENDENT_PASSAGE_REVIEW',
              'zip_created':False,'runtime_seconds':time.perf_counter()-started}
    for path, expected in zip(paths, (P27,P29)):
        if sha256_file(path) != expected: raise ValueError('ZIP modified during experiment')
    (out / 'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'}, ensure_ascii=False,indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
