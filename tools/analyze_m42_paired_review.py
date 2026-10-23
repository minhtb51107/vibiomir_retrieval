"""Read-only validation and policy decoding of a report-exposed AI CSV review."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/multilingual_retrieval_m42/window_ablation'
STATUS = 'AI_JUDGED_M42_ASSISTANT_REPORT_EXPOSED_NOT_HUMAN_VERIFIED'
SOURCE_FIELDS = ('query_vi', 'doc_id', 'canonical_url', 'evidence_left', 'evidence_right')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    # newline='' is required: never normalize or repair the source evidence.
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError('CSV column mismatch')
        return rows


def keyed(rows):
    result = {}
    for row in rows:
        if not row['case_id'] or row['case_id'] in result:
            raise ValueError('missing or duplicate case_id')
        result[row['case_id']] = row
    return result


def analyze(queue, reviewed, mapping, chunks, official, queries):
    original, reviewed = keyed(queue), keyed(reviewed)
    if set(original) != set(reviewed) or set(original) != set(mapping):
        raise ValueError('case set mismatch')
    cases = []
    preferences = collections.Counter()
    labels = {arm: collections.Counter() for arm in ('A', 'B')}
    depth = collections.defaultdict(collections.Counter)
    transitions = collections.Counter()
    for case_id in sorted(original):
        source, row, info = original[case_id], reviewed[case_id], mapping[case_id]
        if any(source[k] != row[k] for k in SOURCE_FIELDS):
            raise ValueError(f'source field mismatch: {case_id}')
        if {info['left_arm'], info['right_arm']} != {'A', 'B'}:
            raise ValueError('side mapping is not a permutation of A/B')
        if int(row['doc_id']) != int(info['doc_id']):
            raise ValueError('document mapping mismatch')
        if queries[int(info['query_id'])] != row['query_vi']:
            raise ValueError('original query mismatch')
        if official.get(int(row['doc_id'])) != row['canonical_url']:
            raise ValueError('current official membership/URL mismatch')
        if row['verification_status'] != STATUS:
            raise ValueError('unexpected review status; do not infer human verification')
        if row['preference'] not in {'LEFT', 'RIGHT', 'TIE', 'UNSURE'}:
            raise ValueError('unknown preference')
        decoded = {}
        for side in ('left', 'right'):
            arm = info[side + '_arm']
            chunk_id = info[arm.lower() + '_chunk_id']
            chunk = chunks[chunk_id]
            if (int(chunk['doc_id']) != int(row['doc_id']) or
                    chunk['canonical_url'] != row['canonical_url'] or
                    chunk['evidence_text'] != row['evidence_' + side]):
                raise ValueError(f'canonical evidence/owner mismatch: {case_id}/{side}')
            label = row[side + '_judgment']
            if label not in {'0', '1', '2', 'UNSURE'}:
                raise ValueError('unknown answer label')
            decoded[arm] = label
            labels[arm][label] += 1
        # Deliberately do not infer an arm from LEFT/RIGHT position or score labels.
        pref = row['preference']
        winner = info[pref.lower() + '_arm'] if pref in {'LEFT', 'RIGHT'} else pref
        preferences[winner] += 1
        depth['TOP20' if info['rank'] <= 20 else 'RANK21_100'][winner] += 1
        transitions[decoded['A'] + '->' + decoded['B']] += 1
        cases.append(dict(case_id=case_id, query_id=info['query_id'], doc_id=info['doc_id'],
                          rank=info['rank'], a_chunk_id=info['a_chunk_id'],
                          b_chunk_id=info['b_chunk_id'], a_label=decoded['A'],
                          b_label=decoded['B'], relative_preference=winner,
                          notes=row['review_notes']))
    return dict(cases=cases, preferences=dict(preferences),
                answer_labels={a: dict(v) for a, v in labels.items()},
                label_transitions=dict(transitions),
                depth_preferences={k: dict(v) for k, v in depth.items()},
                cases_checked=len(cases), evidence_windows_checked=2 * len(cases),
                review_status=STATUS, independent_blind=False, human_verified=0,
                gold_labels=False, automatic_promotion=False)


def main():
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    names = ['blind_review_queue.csv', 'review_mapping.json', 'review_selection.json',
             'M42_assistant_paired_review_24.csv']
    before = {n: sha(BASE / n) for n in names}
    protocol = json.loads((BASE / 'review_selection.json').read_text(encoding='utf-8'))
    if (before['blind_review_queue.csv'] != protocol['queue_sha256'] or
            before['review_mapping.json'] != protocol['mapping_sha256']):
        raise ValueError('frozen queue/mapping checksum mismatch')
    manifest_path = ROOT / 'artifacts/multilingual_retrieval_m42/frozen_manifest.json'
    if sha(manifest_path) != protocol['inputs']['frozen_manifest_sha256']:
        raise ValueError('frozen manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    corpus_path = ROOT / 'artifacts/multilingual_retrieval_m4/candidate_corpus.parquet'
    expected = next(v for k, v in manifest['inputs_sha256'].items()
                    if Path(k).name == 'candidate_corpus.parquet')
    if sha(corpus_path) != expected:
        raise ValueError('frozen canonical corpus checksum mismatch')
    chunks = {r['chunk_id']: r for r in pq.read_table(corpus_path).to_pylist()}
    mapping = json.loads((BASE / 'review_mapping.json').read_text(encoding='utf-8'))
    queue = read_csv(BASE / 'blind_review_queue.csv')
    if len(queue) != 24 or set(keyed(queue)) != set(protocol['case_ids']):
        raise ValueError('expected frozen 24 cases')
    owners = sorted({int(r['doc_id']) for r in queue})
    official = {}
    for batch in ds.dataset(ROOT / 'data/raw/links_corpus_vi_en_zh.parquet').scanner(
            columns=['id', 'url'], filter=ds.field('id').isin(owners),
            batch_size=256, use_threads=False).to_batches():
        for row in batch.to_pylist():
            official[int(row['id'])] = row['url']
    best = ROOT / 'submissions/00_READY_TO_UPLOAD/p27_sdso20749_cap16.zip'
    expected_best = 'b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e'
    if sha(best) != expected_best:
        raise ValueError('Phase27 hash mismatch')
    report = analyze(queue, read_csv(BASE / names[-1]), mapping, chunks, official,
                     {int(q['query_id']): q['query_vi'] for q in manifest['queries']})
    report['input_sha256'] = before
    report['phase27_sha256'] = expected_best
    report['current_official_owners_checked'] = len(official)
    if before != {n: sha(BASE / n) for n in names} or sha(best) != expected_best:
        raise ValueError('input changed during audit')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
