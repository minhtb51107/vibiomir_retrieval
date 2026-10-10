import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.source_census.pipeline import (
    SourceBM25Index,
    _merge_source_parts,
    _bm25_rank,
    balanced_groups,
    balanced_round2_groups,
    calibrate_cap,
    load_config,
    score_gate,
    stable_fold,
    stable_top_k_indices,
)
from src.source_census.phase10e import assemble_union, phase_contract, preflight_contract_checks
from src.source_census.round3 import ROUND3_GROUPS
from src.source_census.depth1000 import DEPTH_GROUPS, seed_exact_scores
from src.source_census.cached_subsets import _require_complete_score_cache
from src.validation.pre_submission import audit_score_cache, score_distribution
from src.source_census.union_experiment import (
    _build_canonical_store,
    _filter_parquet,
    _iter_json_array,
    _merge_score_databases,
    _reference_json_sha256,
)


def test_stable_fold_is_deterministic():
    assert stable_fold("example.org:42", 5) == stable_fold("example.org:42", 5)


def test_balanced_groups_are_disjoint():
    triage = {
        "healthy_sources": ["a", "b", "c", "d", "e", "f"],
        "sources": {
            name: {"corpus_rows": index * 100, "usable_documents": 20 + index * 10, "likely_language": "vi" if index % 2 else "zh"}
            for index, name in enumerate(["a", "b", "c", "d", "e", "f"], 1)
        },
    }
    groups = balanced_groups({}, triage)
    flattened = [item for values in groups.values() for item in values]
    assert sorted(flattened) == sorted(triage["healthy_sources"])
    assert len(flattened) == len(set(flattened))


def test_round2_groups_are_disjoint_and_parent_balanced():
    parents = {
        "A": [f"a{index}" for index in range(21)],
        "C": [f"c{index}" for index in range(21)],
    }
    metadata = {
        source: {
            "corpus_rows": 100 + index * 31,
            "sampled": 100,
            "usable_documents": 20 + index % 81,
            "chunk_count": 5 + index * 3,
            "likely_language": ("vi", "zh", "unknown")[index % 3],
        }
        for index, source in enumerate([*parents["A"], *parents["C"]])
    }
    first = balanced_round2_groups(parents, metadata)
    second = balanced_round2_groups(parents, metadata)
    assert first == second
    flattened = []
    for rows in first["groups"].values():
        assert len(rows) == 6
        assert sum(row["parent"] == "A" for row in rows) == 3
        assert sum(row["parent"] == "C" for row in rows) == 3
        flattened.extend(row["source"] for row in rows)
    assert sorted(flattened) == sorted([*parents["A"], *parents["C"]])
    assert len(flattened) == len(set(flattened)) == 42


def test_round3_fixed_groups_are_six_disjoint_trios():
    assert list(ROUND3_GROUPS) == ["G1A", "G1B", "G5A", "G5B", "G6A", "G6B"]
    flattened = [source for sources in ROUND3_GROUPS.values() for source in sources]
    assert all(len(sources) == 3 for sources in ROUND3_GROUPS.values())
    assert len(flattened) == len(set(flattened)) == 18


def test_depth1000_groups_preserve_all_nine_sources_without_overlap():
    assert list(DEPTH_GROUPS) == ["G1A", "G5A", "G6B"]
    flattened = [source for sources in DEPTH_GROUPS.values() for source in sources]
    assert all(len(sources) == 3 for sources in DEPTH_GROUPS.values())
    assert len(flattened) == len(set(flattened)) == 9
    assert "v.familydoctor.com.cn" in flattened
    assert "suckhoedoisong.vn" in flattened


def test_cached_score_validation_accepts_complete_non_round1_size(tmp_path, monkeypatch):
    parts = tmp_path / "round1" / "source_candidate_parts"
    parts.mkdir(parents=True)
    pq.write_table(pa.table({"query_id": [1, 1], "chunk_id": ["a", "b"]}), parts / "one.parquet")
    monkeypatch.setattr(
        "src.source_census.cached_subsets.score_database_status",
        lambda config, verify_integrity=False: {
            "total": 2, "done": 2, "remaining": 0, "integrity": "ok",
            "future_schema_field": "accepted",
        },
    )
    status = _require_complete_score_cache({"outputs": {"root": str(tmp_path)}})
    assert status["done"] == 2


def test_cached_score_validation_rejects_missing_candidate_key(tmp_path, monkeypatch):
    parts = tmp_path / "round1" / "source_candidate_parts"
    parts.mkdir(parents=True)
    pq.write_table(pa.table({"query_id": [1, 1], "chunk_id": ["a", "b"]}), parts / "one.parquet")
    monkeypatch.setattr(
        "src.source_census.cached_subsets.score_database_status",
        lambda config, verify_integrity=False: {"total": 1, "done": 1, "remaining": 0, "integrity": "ok"},
    )
    with pytest.raises(ValueError, match="expected candidate keys=2"):
        _require_complete_score_cache({"outputs": {"root": str(tmp_path)}})


def _score_db(path, rows):
    import sqlite3
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE pairs(query_id INTEGER,chunk_id TEXT,query_text TEXT,chunk_text TEXT,rerank_score REAL,inference_ms REAL,PRIMARY KEY(query_id,chunk_id))")
    connection.executemany("INSERT INTO pairs VALUES (?,?,?,?,?,?)", rows)
    connection.commit(); connection.close()


def test_exact_score_seed_reuses_only_full_composite_key(tmp_path):
    old = tmp_path / "old.sqlite"; destination = tmp_path / "destination.sqlite"
    _score_db(old, [(1,"a","q1","a",1.25,10.0),(1,"b","q1","b",2.5,20.0),(2,"a","q2","a",3.75,30.0)])
    _score_db(destination, [(1,"a","q1","a",None,None),(1,"b","q1","b",None,None),(2,"b","q2","b",None,None)])
    result = seed_exact_scores(destination, old)
    import sqlite3
    connection = sqlite3.connect(destination)
    rows = connection.execute("SELECT query_id,chunk_id,rerank_score,inference_ms FROM pairs ORDER BY query_id,chunk_id").fetchall()
    connection.close()
    assert rows == [(1,"a",1.25,10.0),(1,"b",2.5,20.0),(2,"b",None,None)]
    assert result["reused"] == 2 and result["missing"] == 1
    assert result["distinct_seeded_scores"] == 2
    assert result["incorrectly_seeded_nonmatches"] == 0


def test_union_score_cache_preserves_exact_parent_values(tmp_path):
    roots=[]
    for name,rows in (
        ("a",[(1,"a","q","a",1.25,10.0),(2,"b","q2","b",2.5,20.0)]),
        ("b",[(1,"c","q","c",-3.0,30.0)]),
    ):
        root=tmp_path/name; parts=root/"round1/source_candidate_parts"; parts.mkdir(parents=True)
        _score_db(root/"round1/source_scores.sqlite",rows)
        (parts/f"{name}.json").write_text(json.dumps({"source":name}),encoding="utf-8")
        pq.write_table(pa.table({"query_id":[row[0] for row in rows],"chunk_id":[row[1] for row in rows]}),parts/f"{name}.parquet")
        roots.append(root)
    spec={"parents":{"A":{"root":str(roots[0]),"sources":["a"]},"B":{"root":str(roots[1]),"sources":["b"]}}}
    result=_merge_score_databases(spec,tmp_path/"union.sqlite")
    import sqlite3
    connection=sqlite3.connect(tmp_path/"union.sqlite")
    rows=connection.execute("SELECT query_id,chunk_id,rerank_score,inference_ms FROM pairs ORDER BY query_id,chunk_id").fetchall(); connection.close()
    assert rows==[(1,"a",1.25,10.0),(1,"c",-3.0,30.0),(2,"b",2.5,20.0)]
    assert result["total"]==3 and result["new_inference"]==0
    assert result["value_mismatches"]=={"A":0,"B":0}


def test_union_score_cache_excludes_unselected_sources_from_shared_parent(tmp_path):
    root=tmp_path/"shared"; parts=root/"round1/source_candidate_parts"; parts.mkdir(parents=True)
    _score_db(root/"round1/source_scores.sqlite",[
        (1,"selected","q","selected",1.25,10.0),
        (1,"excluded","q","excluded",9.5,20.0),
    ])
    for source,chunk in (("selected.example","selected"),("excluded.example","excluded")):
        stem=source.replace(".","_")
        (parts/f"{stem}.json").write_text(json.dumps({"source":source}),encoding="utf-8")
        pq.write_table(pa.table({"query_id":[1],"chunk_id":[chunk]}),parts/f"{stem}.parquet")
    spec={"parents":{"SHARED":{"root":str(root),"sources":["selected.example"]}}}
    result=_merge_score_databases(spec,tmp_path/"union.sqlite")
    import sqlite3
    connection=sqlite3.connect(tmp_path/"union.sqlite")
    keys=connection.execute("SELECT query_id,chunk_id FROM pairs").fetchall(); connection.close()
    assert keys==[(1,"selected")]
    assert result["total"]==1


def test_union_canonical_filter_is_complete_and_deduplicated(tmp_path):
    first=tmp_path/"first.parquet"; second=tmp_path/"second.parquet"; output=tmp_path/"out.parquet"
    pq.write_table(pa.table({"chunk_id":["a","b"],"doc_id":[1,2]}),first)
    pq.write_table(pa.table({"chunk_id":["b","c"],"doc_id":[2,3]}),second)
    result=_filter_parquet([first,second],output,"chunk_id",{"a","b","c"})
    assert result=={"selected":3,"written":3,"duplicate_input_rows":1}
    assert pq.read_table(output).column("chunk_id").to_pylist()==["a","b","c"]


def test_union_canonical_filter_unifies_null_and_string_columns(tmp_path):
    first=tmp_path/"first.parquet"; second=tmp_path/"second.parquet"; output=tmp_path/"out.parquet"
    pq.write_table(pa.table({"doc_id":[1],"error_message":pa.array([None],type=pa.null())}),first)
    pq.write_table(pa.table({"doc_id":[2],"error_message":pa.array(["failed"],type=pa.string())}),second)
    result=_filter_parquet([first,second],output,"doc_id",{1,2})
    table=pq.read_table(output)
    assert result["written"]==2
    assert table.schema.field("error_message").type==pa.string()
    assert table.column("error_message").to_pylist()==[None,"failed"]


def test_union_streaming_json_reader_handles_large_record_boundaries(tmp_path):
    path=tmp_path/"submission.json"
    expected=[{"id":1,"value":"x"*(1024*1024+17)},{"id":2,"value":"y"}]
    path.write_text(json.dumps(expected,separators=(",",":")),encoding="utf-8")
    assert list(_iter_json_array(path))==expected


def test_union_submission_comparison_streams_exact_semantics(tmp_path):
    from src.source_census.union_experiment import _compare_submission_signatures
    reference=tmp_path/"reference.json"; current=tmp_path/"current.json"
    reference.write_text(json.dumps([
        {"id":1,"relevant_docs":[1,2],"relevant_chunks":[{"doc_id":1,"chunk_text":"same"}]},
        {"id":2,"relevant_docs":[3],"relevant_chunks":[{"doc_id":3,"chunk_text":"old"}]},
    ],separators=(",",":")),encoding="utf-8")
    current.write_text(json.dumps([
        {"id":1,"relevant_docs":[1,2],"relevant_chunks":[{"doc_id":1,"chunk_text":"same"}]},
        {"id":2,"relevant_docs":[4],"relevant_chunks":[{"doc_id":3,"chunk_text":"new"}]},
    ],separators=(",",":")),encoding="utf-8")
    assert _compare_submission_signatures(reference,current)=={
        "queries_compared":2,"queries_top10_docs_changed":1,"queries_top20_chunks_changed":1,
    }


def test_union_canonical_store_is_exact_and_integrity_checked(tmp_path):
    chunks=tmp_path/"chunks.parquet"; documents=tmp_path/"documents.parquet"; database=tmp_path/"canonical.sqlite"
    pq.write_table(pa.table({"chunk_id":["a","b"],"doc_id":[1,2],"raw_text":["one","two"],"start_offset":[0,1],"end_offset":[3,4]}),chunks)
    pq.write_table(pa.table({"doc_id":[1,2],"normalized_text":["one document","a two document"]}),documents)
    result=_build_canonical_store(chunks,documents,database)
    import sqlite3
    connection=sqlite3.connect(database)
    stored=connection.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall(); connection.close()
    assert result=={"chunks":2,"documents":2,"sqlite_integrity":"ok"}
    assert stored==[("a",1,"one",0,3),("b",2,"two",1,4)]


def test_union_control_reference_accepts_one_root_json_zip(tmp_path):
    import hashlib
    import zipfile
    payload=b'[{"id":1}]\n'; path=tmp_path/"control.zip"
    with zipfile.ZipFile(path,"w") as archive: archive.writestr("control.json",payload)
    assert _reference_json_sha256({"submission_zip":str(path)})==hashlib.sha256(payload).hexdigest()


def test_cache_coverage_rejects_equal_row_count_with_wrong_key(tmp_path):
    root = tmp_path / "round1"; parts = root / "source_candidate_parts"; parts.mkdir(parents=True)
    pq.write_table(pa.table({"query_id":[1,1],"chunk_id":["a","b"]}),parts/"one.parquet")
    _score_db(root/"source_scores.sqlite",[(1,"a","q","a",1.0,1.0),(1,"wrong","q","x",2.0,1.0)])
    with pytest.raises(ValueError, match="missing score keys=1.*unexpected score keys=1"):
        _require_complete_score_cache({"outputs":{"root":str(tmp_path)}})


def test_scientific_gate_rejects_depth1000_constant_score_corruption(tmp_path):
    root=tmp_path/"round1"; parts=root/"source_candidate_parts"; parts.mkdir(parents=True)
    count=1001
    pq.write_table(pa.table({"query_id":list(range(count)),"chunk_id":[f"c{i}" for i in range(count)]}),parts/"source.parquet")
    _score_db(root/"source_scores.sqlite",[(i,f"c{i}","q","text",-8.8203125,30.37255468746025) for i in range(count)])
    result=audit_score_cache(root/"source_scores.sqlite",parts)
    assert not result["passed"]
    assert result["score_distribution"]["distinct"] == 1
    assert result["distinct_inference_ms_count"] == 1
    assert "degenerate rerank scores" in result["failures"]


def test_score_distribution_reports_required_quantiles():
    result=score_distribution([1.0,2.0,3.0,4.0,5.0])
    assert result == {"count":5,"distinct":5,"min":1.0,"median":3.0,"p95":5.0,"max":5.0}


def test_score_gate_preserves_ambiguous_groups():
    assert score_gate({"A": .0010, "B": .00095, "C": .0002})["status"] == "AMBIGUOUS"
    result = score_gate({"A": .0011, "B": .0009, "C": .0002})
    assert result["status"] == "ADVANCE"
    assert result["winner"] == "A"


def test_cached_bm25_is_exactly_equivalent():
    texts = ["đau đầu chóng mặt", "đau bụng", "头痛 治疗", "unrelated"]
    index = SourceBM25Index(texts, k1=1.2, b=.75)
    for query in ("đau đầu", "头痛", "missing", "đau"):
        expected = _bm25_rank(texts, query, k1=1.2, b=.75, top_k=3)
        actual = index.rank(query, top_k=3)
        assert [row for row, _ in actual] == [row for row, _ in expected]
        assert [score for _, score in actual] == [score for _, score in expected]


def test_partial_top_k_matches_stable_full_sort_with_boundary_ties():
    scores = np.asarray([.4, .9, .9, .1, .8, .8], dtype=np.float32)
    expected = np.argsort(-scores, kind="stable")[:4]
    assert stable_top_k_indices(scores, 4).tolist() == expected.tolist()


def test_persisted_candidate_cap_contract_does_not_require_recalibration(tmp_path):
    artifact = tmp_path / "prior_calibration.json"
    artifact.write_text('{"method":"prior validated calibration","chosen_m":8}', encoding="utf-8")
    config = {
        "candidate_cache": {"candidate_caps": [8], "calibrated_cap_artifact": str(artifact)},
        "inputs": {},
        "outputs": {"artifacts": str(tmp_path / "current")},
    }
    result = calibrate_cap(config)
    assert result["chosen_m"] == 8
    assert result["reused_persisted_calibration"] is True
    assert result["source_artifact"] == str(artifact)


def test_persisted_candidate_cap_must_match_allowed_contract(tmp_path):
    artifact = tmp_path / "prior_calibration.json"
    artifact.write_text('{"chosen_m":4}', encoding="utf-8")
    config = {
        "candidate_cache": {"candidate_caps": [8], "calibrated_cap_artifact": str(artifact)},
        "inputs": {},
        "outputs": {"artifacts": str(tmp_path / "current")},
    }
    with pytest.raises(ValueError, match="chosen_m=4"):
        calibrate_cap(config)


def test_source_part_merge_normalizes_all_null_heading_lists(tmp_path):
    with_headings = pa.table({
        "query_id": [1],
        "heading_path": pa.array([["Question"]], type=pa.list_(pa.string())),
    })
    without_headings = pa.table({
        "query_id": [2],
        "heading_path": pa.array([[None]], type=pa.list_(pa.null())),
    })
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    output = tmp_path / "merged.parquet"
    pq.write_table(with_headings, first)
    pq.write_table(without_headings, second)

    assert _merge_source_parts([first, second], output) == 2
    merged = pq.read_table(output)
    assert merged.schema.field("heading_path").type == pa.list_(pa.string())
    assert merged.num_rows == 2
def test_g1a_focused_scaling_preflight_contract_is_fixed():
    config=load_config("configs/phase10e_g1a_10010.yaml")
    key,phase,sources,group,experiment=phase_contract(config)
    assert key=="phase10e_g1a"
    assert group=="G1A"
    assert experiment=="phase10e_g1a_10010"
    assert sources==["medlatec.vn","v.familydoctor.com.cn","vinmec.com"]
    assert phase["targets"]=={"medlatec.vn":5000,"v.familydoctor.com.cn":10,"vinmec.com":5000}
    assert all(preflight_contract_checks(config,8,sources).values())
    assert config["candidate_cache"]["calibrated_cap_artifact"]=="artifacts/source_census/depth1000/candidate_cap_calibration.json"
    assert config["inputs"]["c1_candidate_pool"]=="data/retrieval/phase10c1_early_release/C1/candidate_pool.parquet"


def test_g1a_preflight_rejects_changed_candidate_cap():
    config=load_config("configs/phase10e_g1a_10010.yaml")
    _,_,sources,_,_=phase_contract(config)
    config["candidate_cache"]["candidate_caps"]=[4]
    checks=preflight_contract_checks(config,8,sources)
    assert checks["chosen_m_is_8"] is False


def test_g6b_reserve_contract_is_fixed_and_not_launched():
    config=load_config("configs/phase10e_g6b_15000.yaml")
    key,phase,sources,group,experiment=phase_contract(config)
    assert (key,group,experiment)==("phase10e_g6b","G6B","phase10e_g6b_15000")
    assert sources==["tiemchunglongchau.com.vn","cancer.39.net","suckhoedoisong.vn"]
    assert set(phase["targets"].values())=={5000}
    assert all(preflight_contract_checks(config,8,sources).values())
    assert phase["supplemental_previous_chunks"]==[
        "data/phase10c1/C3/new_chunks.parquet",
        "data/source_census/round1/chunks.parquet",
    ]
    assert config["models"]["reranker"]["equivalence"]["control_artifact"].startswith(
        "artifacts/source_census/phase10e_g6b_15000/"
    )


def test_medlatec_depth_checkpoints_are_nested_and_deterministic():
    from src.source_census.phase10e import _depth_checkpoint_ids
    successful={9,3}; incremental=[{"doc_id":20},{"doc_id":10},{"doc_id":30}]
    depth4=_depth_checkpoint_ids(successful,incremental,4,5)
    depth5=_depth_checkpoint_ids(successful,incremental,5,5)
    assert depth4=={"target_depth":4,"new_ids_required":2,"target_ids":[3,9,20,10]}
    assert depth5["target_ids"]==[3,9,20,10,30]
    assert depth4["target_ids"]==depth5["target_ids"][:4]


def test_phase10f_medlatec_contract_prepares_two_depths_only():
    config=load_config("configs/phase10f_medlatec_10000.yaml")
    key,phase,sources,group,experiment=phase_contract(config)
    assert (key,group,experiment)==("phase10f_medlatec","MEDLATEC","phase10f_medlatec_10000")
    assert sources==["medlatec.vn"]
    assert phase["targets"]=={"medlatec.vn":10000}
    assert phase["depth_checkpoints"]==[7500,10000]
    assert set(phase["fixed_union_sources"])=={
        "v.familydoctor.com.cn","vinmec.com","tiemchunglongchau.com.vn","cancer.39.net",
        "suckhoedoisong.vn","benhviennhitrunguong.gov.vn","zydcd.com","hellobacsi.com",
    }
    assert all(preflight_contract_checks(config,8,sources).values())


def test_phase10e_union_includes_supplemental_previous_chunks(tmp_path):
    documents=tmp_path/"documents.parquet"; new_documents=tmp_path/"new_documents.parquet"
    old_chunks=tmp_path/"old_chunks.parquet"; supplemental=tmp_path/"supplemental.parquet"; new_chunks=tmp_path/"new_chunks.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"doc_id":1,"original_url":"https://example.org/one","extraction_status":"SUCCESS","normalized_text":"one"},
        {"doc_id":2,"original_url":"https://example.org/two","extraction_status":"SUCCESS","normalized_text":"two"},
    ]),documents)
    pq.write_table(pa.Table.from_pylist([
        {"doc_id":3,"original_url":"https://example.org/three","extraction_status":"SUCCESS","normalized_text":"three"},
    ]),new_documents)
    pq.write_table(pa.Table.from_pylist([{"doc_id":1,"chunk_id":"c1","chunk_index":0}]),old_chunks)
    pq.write_table(pa.Table.from_pylist([{"doc_id":2,"chunk_id":"c2","chunk_index":0}]),supplemental)
    pq.write_table(pa.Table.from_pylist([{"doc_id":3,"chunk_id":"c3","chunk_index":0}]),new_chunks)
    manifest=tmp_path/"manifest.json"
    manifest.write_text('{"sources":{"example.org":{"official_population":2,"target_ids":[1,2]}}}',encoding="utf-8")
    config={
        "focused_scaling":{"phase_key":"reserve","group":"X","experiment":"test"},
        "reserve":{"targets":{"example.org":2},"manifest":str(manifest),"previous_documents":[str(documents)],"previous_chunks":str(old_chunks),"supplemental_previous_chunks":[str(supplemental)],"new_documents":str(new_documents),"new_chunks":str(new_chunks)},
        "outputs":{"documents":str(tmp_path/"union_documents.parquet"),"chunks":str(tmp_path/"union_chunks.parquet"),"artifacts":str(tmp_path/"artifacts")},
    }
    result=assemble_union(config)
    assert result["documents"]==2
    assert result["chunks"]==2
    assert set(pq.read_table(config["outputs"]["chunks"])["chunk_id"].to_pylist())=={"c1","c2"}
