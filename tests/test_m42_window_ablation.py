import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tools.m42_window_ablation import (
    cosine_scores, csv_payload, frozen_text, navigation_proxy,
    replace_windows, select_review,
)


def test_true_cosine_not_magnitude_dot_product():
    scores = cosine_scores([[100, 100], [1, 0]], [1, 0])
    assert scores[1] > scores[0]
    assert scores[1] == pytest.approx(1)
    for matrix, query in [([[0, 0]], [1, 0]), ([[1, 0]], [0, 0]),
                          ([[float('nan'), 0]], [1, 0])]:
        with pytest.raises(ValueError, match='nonfinite/zero'):
            cosine_scores(matrix, query)


def test_same_owner_fixed_ranking_scores_and_deterministic_ties():
    chunks = [{'doc_id': 1, 'chunk_id': 'z'}, {'doc_id': 1, 'chunk_id': 'a'},
              {'doc_id': 2, 'chunk_id': 'other'}]
    control = [{'doc_id': 1, 'chunk_id': 'z', 'rank': 1, 'rrf_score': .03},
               {'doc_id': 2, 'chunk_id': 'other', 'rank': 2, 'rrf_score': .02}]
    result = replace_windows(control, chunks, [.7, .7, 1.])
    assert result == [{**control[0], 'chunk_id': 'a'}, control[1]]
    assert control[0]['chunk_id'] == 'z'
    assert replace_windows(control, list(reversed(chunks)), [1., .7, .7]) == result
    with pytest.raises(ValueError, match='owner absent'):
        replace_windows([{'doc_id': 3}], chunks, [.7, .7, 1.])


def test_navigation_flag_is_only_diagnostic():
    flag, reasons = navigation_proxy({'evidence_text': '目录\n各种疾病链接'})
    assert flag and 'NAVIGATION_HEADING' in reasons
    # Selection does not filter directory evidence, even with a diagnostic flag.
    chunks = [{'doc_id': 1, 'chunk_id': 'directory', 'evidence_text': '目录'},
              {'doc_id': 1, 'chunk_id': 'body', 'evidence_text': 'clinical body'}]
    assert replace_windows([{'doc_id': 1, 'chunk_id': 'body'}], chunks, [.9, .8])[0]['chunk_id'] == 'directory'


def test_review_selection_reproducible_and_covers_risk_and_deep_strata():
    rows = []
    for q in range(12):
        for i, transition in enumerate(['BODY_TO_BODY', 'BODY_TO_NAV', 'NAV_TO_BODY', 'NAV_TO_NAV']):
            for deep in [False, True]:
                rows.append(dict(query_id=q, doc_id=i + 10 * deep,
                                 a_chunk_id=f'{q}-{i}-{deep}-a', b_chunk_id=f'{q}-{i}-{deep}-b',
                                 changed=True, rank=i + 1 + 20 * deep,
                                 stratum=('RANK21_100' if deep else 'TOP20') + ':' + transition))
    selected = select_review(rows)
    assert len(selected) == 24
    assert selected == select_review(list(reversed(rows)))
    assert {r['query_id'] for r in selected} == set(range(12))
    assert len({r['stratum'] for r in selected}) == 8
    assert len({(r['query_id'], r['doc_id']) for r in selected}) == 24


def test_utf8_csv_roundtrip_preserves_verbatim_newlines_and_freeze(tmp_path):
    text = '脾肿大\n原因,"引用"\n第二行'
    payload = csv_payload([{'case_id': 'blind', 'evidence_left': text}], ['case_id', 'evidence_left'])
    target = tmp_path / 'review.csv'
    frozen_text(target, payload)
    frozen_text(target, payload)
    with target.open(encoding='utf-8', newline='') as stream:
        assert next(csv.DictReader(stream))['evidence_left'] == text
    assert '\r\n' not in target.read_bytes().decode('utf-8')
    with pytest.raises(ValueError, match='frozen output changed'):
        frozen_text(target, payload + 'tampered')
    assert next(csv.DictReader(io.StringIO(payload, newline='')))['evidence_left'] == text


def test_local_artifacts_blinding_and_exact_source_roundtrip():
    root = Path(__file__).resolve().parents[1]
    output = root / 'artifacts/multilingual_retrieval_m42/window_ablation'
    if not (output / 'paired_comparison.csv').exists():
        pytest.skip('private local frozen artifacts are not shipped with source')
    import pyarrow.parquet as pq
    chunks = {r['chunk_id']: r for r in pq.read_table(
        root / 'artifacts/multilingual_retrieval_m4/candidate_corpus.parquet').to_pylist()}
    with (output / 'paired_comparison.csv').open(encoding='utf-8', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 1200
    for pair in pairs:
        for arm in ['a', 'b']:
            source = chunks[pair[arm + '_chunk_id']]
            assert int(source['doc_id']) == int(pair['doc_id'])
            assert source['evidence_text'] == pair[arm + '_evidence_text']
            assert source['canonical_url'] == pair['canonical_url']
    mapping = json.loads((output / 'review_mapping.json').read_text(encoding='utf-8'))
    with (output / 'blind_review_queue.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        assert set(reader.fieldnames) == {
            'case_id', 'query_vi', 'doc_id', 'canonical_url', 'evidence_left',
            'evidence_right', 'left_judgment', 'right_judgment', 'preference',
            'review_notes', 'verification_status'}
        queue = list(reader)
    assert len(queue) == 24
    for row in queue:
        info = mapping[row['case_id']]
        assert row['verification_status'] == 'PENDING_INDEPENDENT_REVIEW'
        assert not any(row[k] for k in ['left_judgment', 'right_judgment', 'preference'])
        for side in ['left', 'right']:
            arm = info[side + '_arm'].lower()
            assert row['evidence_' + side] == chunks[info[arm + '_chunk_id']]['evidence_text']
