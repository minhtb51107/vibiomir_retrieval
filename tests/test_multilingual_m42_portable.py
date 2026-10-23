import json
from pathlib import Path

import numpy as np
import pytest

from src.multilingual_retrieval.m42_dense import checkpoint_open, require_trial, vector_digest, document_rows
from tools.m42_portable import verify, file_hash, safe_path


def test_trial_is_bound_to_manifest_and_target_runtime():
    report={'passed':True,'trial20_sha256':'abc','runtime_identity':{'gpu':'one'}}
    require_trial(report,{'trial20_sha256':'abc'},{'gpu':'one'})
    with pytest.raises(ValueError,match='runtime/GPU'):
        require_trial(report,{'trial20_sha256':'abc'},{'gpu':'two'})
    with pytest.raises(ValueError,match='not passed'):
        require_trial(report,{'trial20_sha256':'different'},{'gpu':'one'})


def test_resume_verifies_committed_vectors_and_does_not_reset(tmp_path):
    con,vectors,done=checkpoint_open(tmp_path,'signature',8)
    assert not done
    values=np.ones((4,1024),dtype=np.float32)/32
    vectors[:4]=values; vectors.flush()
    con.execute('INSERT INTO windows VALUES(?,?,?)',(0,4,vector_digest(values))); con.commit(); con.close(); del vectors
    con,vectors,done=checkpoint_open(tmp_path,'signature',8)
    assert done=={0} and np.array_equal(vectors[:4],values)
    con.close(); del vectors
    with pytest.raises(ValueError,match='signature'):
        checkpoint_open(tmp_path,'changed',8)


def test_corrupt_commit_fails_closed(tmp_path):
    con,vectors,_=checkpoint_open(tmp_path,'s',4)
    con.execute('INSERT INTO windows VALUES(?,?,?)',(0,4,'wrong')); con.commit(); con.close(); del vectors
    with pytest.raises(ValueError,match='checksum'):
        checkpoint_open(tmp_path,'s',4)


def test_document_budget_dedup_is_stable():
    rows=[{'query_id':1,'doc_id':10,'chunk_id':'a','candidate_rank':1},
          {'query_id':1,'doc_id':10,'chunk_id':'b','candidate_rank':2},
          {'query_id':1,'doc_id':20,'chunk_id':'c','candidate_rank':3}]
    assert [r['doc_id'] for r in document_rows(rows,{1})[1]]==[10,20]


def test_frozen_final_order_not_replaced_by_candidate_order():
    rows=[{'query_id':1,'doc_id':10,'chunk_id':'a','candidate_rank':1,'rank':2},
          {'query_id':1,'doc_id':20,'chunk_id':'b','candidate_rank':5,'rank':1}]
    assert [r['doc_id'] for r in document_rows(rows,{1})[1]]==[20,10]


def test_portable_hashes_and_no_traversal(tmp_path):
    (tmp_path/'file.txt').write_text('payload')
    manifest={'files':{'file.txt':file_hash(tmp_path/'file.txt')}}
    (tmp_path/'portable_manifest.json').write_text(json.dumps(manifest))
    assert verify(tmp_path,runtime=False,models=False)['status']=='VERIFIED'
    (tmp_path/'file.txt').write_text('modified')
    with pytest.raises(ValueError,match='checksum'):
        verify(tmp_path,runtime=False,models=False)
    with pytest.raises(ValueError,match='unsafe'):
        safe_path(tmp_path,'../escape')
    with pytest.raises(ValueError,match='unsafe'):
        safe_path(tmp_path,'C:/outside')
