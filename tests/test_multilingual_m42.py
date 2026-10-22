from __future__ import annotations

import numpy as np
import pytest

from src.multilingual_retrieval import m42


def review_fixture():
    queue=[{"query_id":str(i),"doc_id":str(100+i),"chunk_id":f"{i}-c","query_vi":"Câu hỏi?",
            "canonical_url":f"https://example.invalid/{i}","evidence_text":"原文证据"} for i in range(6)]
    labels=[{**row,"judgment":"1" if i<3 else "0","verification_status":m42.REVIEW_STATUS} for i,row in enumerate(queue)]
    return queue,labels


def test_assistant_review_join_is_order_independent_and_separate():
    queue,labels=review_fixture()
    result=m42.validate_review(list(reversed(labels)),queue)
    assert len(result)==6
    assert all(not row["human_verified"] and not row["blind_review"] for row in result)
    assert all(row["evidence_text"]=="原文证据" for row in result)


@pytest.mark.parametrize("failure",["owner","duplicate","human","text"])
def test_import_rejects_identity_status_and_provenance_failures(failure):
    queue,labels=review_fixture()
    if failure=="owner": labels[0]["doc_id"]="999"
    if failure=="duplicate": labels[0]=dict(labels[1])
    if failure=="human": labels[0]["verification_status"]="HUMAN_VERIFIED"
    if failure=="text": labels[0]["evidence_text"]="invented"
    with pytest.raises(ValueError): m42.validate_review(labels,queue)


def test_missing_review_does_not_reconstruct_labels(tmp_path):
    result=m42.import_review(tmp_path,tmp_path)
    assert result["status"]=="BLOCKED_REVIEW_FILE_MISSING"
    assert result["imported"]==0
    assert not (tmp_path/"assistant_review_diagnostics.json").exists()


def test_active_phase27_stops_before_model_load(monkeypatch):
    monkeypatch.setattr(m42,"semantic_preflight",lambda root:{"phase27_pids":[8872]})
    with pytest.raises(RuntimeError,match="DEFERRED_PHASE27_ACTIVE"): m42.require_idle()


def test_frozen_manifest_resume_and_corruption_fail_closed(tmp_path):
    path=tmp_path/"manifest.json"
    m42.freeze_json(path,{"trial":[1,2]})
    m42.freeze_json(path,{"trial":[1,2]})
    with pytest.raises(ValueError): m42.freeze_json(path,{"trial":[2,1]})
    path.write_text("null",encoding="utf-8")
    with pytest.raises(ValueError): m42.freeze_json(path,{"trial":[1,2]})


def test_exact_checkpoint_resume_never_overwrites_score(tmp_path):
    refs=[{"query_id":"1","doc_id":"2","m4_chunk_id":"2-c","exact_rerank_score":"0.5",
           "reference_source":"S1","historical_chunk_id":"2-old","query_text_sha256":"q","chunk_text_sha256":"t"}]
    path=tmp_path/"scores.sqlite"
    assert m42.commit_references(path,refs,m42.RERANKER)["new_rows"]==1
    assert m42.commit_references(path,refs,m42.RERANKER)["new_rows"]==0
    with pytest.raises(ValueError,match="conflicting"):
        m42.commit_references(path,[{**refs[0],"exact_rerank_score":"0.8"}],m42.RERANKER)


def test_dense_document_budget_and_stable_ties():
    rows=[{"doc_id":2,"chunk_id":"2-b"},{"doc_id":1,"chunk_id":"1-a"},{"doc_id":2,"chunk_id":"2-a"}]
    query=np.zeros((1,1024),np.float32); query[0,0]=1
    vectors=np.zeros((3,1024),np.float32); vectors[:,0]=1
    result=m42.rank_dense(query,vectors,rows,20)[0]
    assert [(r["doc_id"],r["chunk_id"]) for r in result]==[(1,"1-a"),(2,"2-a")]
    vectors[0,0]=np.nan
    with pytest.raises(ValueError): m42.rank_dense(query,vectors,rows,20)


def test_blind_queue_does_not_leak_scores_ranks_or_arms():
    rows=[{"doc_id":1,"chunk_id":"1-a","canonical_url":"https://example.invalid/1","evidence_text":"原文"}]
    arms={"dense":{7:[{"doc_id":1,"chunk_id":"1-a","rank":1,"dense_score":0.99}]}}
    result=m42.new_evidence_review({7:"Câu hỏi?"},arms,set(),rows)
    assert len(result)==1
    assert not ({"rank","dense_score","arm","branch"}&set(result[0]))
    assert result[0]["judgment"]==""
    assert not m42.new_evidence_review({7:"Câu hỏi?"},arms,{(7,1)},rows)


def test_equal_query_coverage_and_no_fake_relevance_metrics():
    result=m42.retention({"dense":{1:[{"doc_id":2}],2:[]}}, {(1,2)}, {1,2})
    assert result["dense"]["20"]["known_partial_pairs_found"]==1
    assert result["dense"]["20"]["not_official_recall"]
    with pytest.raises(ValueError): m42.retention({"dense":{1:[]}},set(),{1,2})


def test_rrf_deduplicates_between_arms_and_rejects_duplicate_within_arm():
    row={"doc_id":1,"chunk_id":"1-a","rank":1}
    assert len(m42.union_rrf([[row],[row]]))==1
    with pytest.raises(ValueError): m42.union_rrf([[row,row]])


def test_tokenizer_class_mismatch_stops_before_import(tmp_path):
    import json
    snapshot=tmp_path/".cache/huggingface/models--BAAI--bge-reranker-v2-m3/snapshots"/m42.RERANKER["resolved_revision"]
    snapshot.mkdir(parents=True)
    for name in ("tokenizer.json","special_tokens_map.json","sentencepiece.bpe.model"):
        (snapshot/name).write_text("test fixture",encoding="utf-8")
    (snapshot/"tokenizer_config.json").write_text(json.dumps({"tokenizer_class":"WrongClass","model_max_length":8192}),encoding="utf-8")
    with pytest.raises(ValueError,match="tokenizer incompatible"):
        m42.verify_tokenizer_contract(tmp_path)
