import copy

import pytest

from tools.analyze_m42_paired_review import STATUS, analyze, main


def fixture_data():
    queue = [dict(case_id='opaque', query_vi='Câu hỏi gốc', doc_id='123',
                  canonical_url='https://example.org/official',
                  evidence_left='甲\n乙', evidence_right='丙\n丁')]
    review = [{**queue[0], 'left_judgment': '0', 'right_judgment': '1',
               'preference': 'RIGHT', 'review_notes': 'review is not gold',
               'verification_status': STATUS}]
    mapping = {'opaque': dict(query_id=7, doc_id=123, rank=2,
                             a_chunk_id='a', b_chunk_id='b', left_arm='B', right_arm='A')}
    chunks = {key: dict(doc_id=123, canonical_url=queue[0]['canonical_url'], evidence_text=text)
              for key, text in [('a', '丙\n丁'), ('b', '甲\n乙')]}
    return [queue, review, mapping, chunks,
            {123: queue[0]['canonical_url']}, {7: queue[0]['query_vi']}]


def test_mapping_not_side_position_decodes_preference_and_labels():
    result = analyze(*fixture_data())
    assert result['preferences'] == {'A': 1}
    assert result['answer_labels'] == {'A': {'1': 1}, 'B': {'0': 1}}
    assert result['label_transitions'] == {'1->0': 1}
    assert result['human_verified'] == 0 and not result['independent_blind']


def test_relative_preference_not_inferred_from_answer_label():
    args = fixture_data()
    args[1][0]['left_judgment'] = '1'
    result = analyze(*args)
    assert result['preferences'] == {'A': 1}
    assert result['label_transitions'] == {'1->1': 1}


@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'text', 'newlines',
                                  'owner', 'official_url', 'side_mapping', 'human_status'])
def test_fail_closed_for_identity_text_provenance_or_status_damage(damage):
    args = copy.deepcopy(fixture_data())
    if damage == 'duplicate':
        args[1].append(args[1][0].copy())
    elif damage == 'missing':
        args[1].clear()
    elif damage == 'text':
        args[1][0]['evidence_left'] += ' extra'
    elif damage == 'newlines':
        args[1][0]['evidence_left'] = args[1][0]['evidence_left'].replace('\n', '\r\n')
    elif damage == 'owner':
        args[3]['b']['doc_id'] = 456
    elif damage == 'official_url':
        args[4][123] += '/changed'
    elif damage == 'side_mapping':
        args[2]['opaque']['left_arm'] = 'A'
    elif damage == 'human_status':
        args[1][0]['verification_status'] = 'HUMAN_VERIFIED'
    with pytest.raises(ValueError):
        analyze(*args)


def test_actual_private_24_review_rows_when_available(capsys):
    from tools.analyze_m42_paired_review import BASE
    if not (BASE / 'M42_assistant_paired_review_24.csv').exists():
        pytest.skip('private reviewer evidence is not shipped')
    result = main()
    assert result['cases_checked'] == 24
    assert result['evidence_windows_checked'] == 48
    assert result['current_official_owners_checked'] == 22
    assert result['preferences'] == {'TIE': 17, 'B': 1, 'A': 6}
    assert result['answer_labels'] == {'A': {'0': 22, '1': 2}, 'B': {'0': 22, '1': 2}}
    assert result['label_transitions'] == {'0->0': 21, '0->1': 1, '1->0': 1, '1->1': 1}
    capsys.readouterr()
