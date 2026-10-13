import json
import sqlite3
import zipfile
from pathlib import Path

from src.source_census.union_experiment import _stream_validate
from tools.phase13a_fusion_replay import (
    aggregate_documents,
    conservative_pubmed,
    document_aware_chunk_capacity,
    near_duplicate,
    rank_rrf,
    representative_chunks,
    select_submission_chunks,
    sha256,
    validate_existing_outputs,
)


def row(doc, chunk, score, source="A", text=None):
    domain = "pubmed.ncbi.nlm.nih.gov" if source == "PUBMED" else f"{source.lower()}.example"
    return {
        "doc_id": doc, "chunk_id": chunk, "score": score, "candidate_arms": [source],
        "canonical_domain": domain, "canonical_url": f"https://{domain}/{doc}", "chunk_text": text or chunk,
    }


def test_top_k_mean_uses_available_chunks_without_missing_penalty():
    docs = aggregate_documents([row(1, "a", 4), row(2, "b", 4), row(2, "c", 2)], "top3_mean")
    scores = {item["doc_id"]: item["score"] for item in docs}
    assert scores == {1: 4.0, 2: 3.0}


def test_robust_representatives_suppress_near_duplicates():
    rows = [row(1, "a", 5, text="alpha beta gamma delta"), row(1, "b", 4, text="alpha beta gamma delta"), row(1, "c", 3, text="other evidence")]
    selected, suppressed = representative_chunks(rows, top_k=3, robust=True)
    assert [item["chunk_id"] for item in selected] == ["a", "c"]
    assert suppressed == 1
    assert near_duplicate(rows[0]["chunk_text"], rows[1]["chunk_text"])


def test_rrf_is_rank_based_and_source_neutral_on_equal_ranks():
    docs = aggregate_documents([row(1, "a", 9, "A"), row(2, "b", 2, "B")], "max_chunk")
    ranked = rank_rrf(docs)
    assert {item["doc_id"] for item in ranked[:2]} == {1, 2}
    assert ranked[0]["fusion_score"] == ranked[1]["fusion_score"]


def test_same_canonical_document_in_pilot_and_domain_arm_merges_once():
    first = row(495914, "c1", 4, "ORIGINAL_PILOT")
    second = {**first, "candidate_arms": ["suckhoedoisong.vn"]}
    docs = aggregate_documents([first, second], "top2_mean")
    assert len(docs) == 1
    assert docs[0]["candidate_arms"] == ["ORIGINAL_PILOT", "suckhoedoisong.vn"]
    assert docs[0]["scored_chunk_count"] == 1
    assert docs[0]["score"] == 4


def test_conflicting_canonical_provenance_hard_fails():
    first = row(1, "a", 4, "A")
    second = {**row(1, "b", 3, "B"), "canonical_url": "https://conflict.example/1"}
    try:
        aggregate_documents([first, second], "max_chunk")
    except ValueError as error:
        assert "conflicting canonical provenance" in str(error)
    else:
        raise AssertionError("canonical provenance conflict was silently merged")


def test_multi_arm_rrf_uses_best_contribution_not_additive_bonus():
    multi = row(1, "a", 2, "A"); multi["candidate_arms"] = ["A", "B"]
    single = row(2, "b", 2, "A")
    docs = aggregate_documents([multi, single], "max_chunk")
    first = rank_rrf(docs); second = rank_rrf(list(reversed(docs)))
    assert first == second
    values = {item["doc_id"]: item["fusion_score"] for item in first}
    assert values[1] == 1 / 61
    assert values[1] < 2 / 61


def test_pubmed_cap_is_explicit_diagnostic_not_score_mutation():
    docs = aggregate_documents([row(i, str(i), 100 - i, "PUBMED" if i < 9 else "A") for i in range(12)], "max_chunk")
    ranked = conservative_pubmed(docs, limit=5)
    assert sum(item["canonical_domain"] == "pubmed.ncbi.nlm.nih.gov" for item in ranked) == 5
    assert all(item["score"] == 100 - item["doc_id"] for item in ranked)


def test_submission_chunks_use_full_pool_when_top_documents_supply_only_33():
    inside = [row((i % 20) + 1, f"in{i}", 100 - i, text=f"inside evidence number {i}") for i in range(33)]
    outside = [row(100 + i, f"out{i}", 60 - i, text=f"outside evidence number {i}") for i in range(20)]
    candidates = inside + outside
    for item in candidates:
        item["query_id"] = 7
    docs = aggregate_documents(inside, "max_chunk")
    assert len(docs) == 20
    assert document_aware_chunk_capacity(docs) == 33
    selected, stats = select_submission_chunks(candidates, docs, depth=40)
    assert len(selected) == 40
    assert len({item["chunk_id"] for item in selected}) == 40
    assert stats["chunk_owners_outside_top20"] > 0


def test_official_validator_allows_chunk_owner_outside_relevant_docs(tmp_path):
    database = tmp_path / "canonical.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE documents(doc_id INTEGER PRIMARY KEY,normalized_text TEXT NOT NULL)")
    connection.execute("CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY,doc_id INTEGER NOT NULL,raw_text TEXT NOT NULL,start_offset INTEGER NOT NULL,end_offset INTEGER NOT NULL)")
    connection.executemany("INSERT INTO documents VALUES (?,?)", [(1, "doc one"), (2, "evidence two")])
    connection.execute("INSERT INTO chunks VALUES (?,?,?,?,?)", ("c2", 2, "evidence two", 0, 12))
    connection.commit(); connection.close()
    payload = [{"id": 1, "relevant_docs": [1], "relevant_chunks": [{"doc_id": 2, "chunk_text": "evidence two"}]}]
    json_path = tmp_path / "submission.json"; json_path.write_text(json.dumps(payload), encoding="utf-8")
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(json_path, json_path.name)
    assert _stream_validate(json_path, zip_path, [1], database, document_depth=1, chunk_depth=1)["valid"] is True


def test_submission_chunks_fail_if_full_pool_is_genuinely_shallow():
    candidates = [dict(row(1, f"c{i}", 10 - i, text=f"unique {i}"), query_id=99) for i in range(3)]
    docs = aggregate_documents(candidates, "max_chunk")
    try:
        select_submission_chunks(candidates, docs, depth=4)
    except ValueError as error:
        assert "full eligible candidate pool" in str(error)
        assert "query_id=99" in str(error)
    else:
        raise AssertionError("shallow full pool was silently padded")


def test_chunk_ranking_is_deterministic_and_preserves_provenance_fields():
    candidates = [dict(row(2, "b", 1), query_id=1, source_url="u2"), dict(row(1, "a", 1), query_id=1, source_url="u1")]
    docs = aggregate_documents(candidates, "max_chunk")
    first, _ = select_submission_chunks(candidates, docs, 2)
    second, _ = select_submission_chunks(list(reversed(candidates)), docs, 2)
    assert [item["chunk_id"] for item in first] == ["a", "b"]
    assert first == second
    assert [item["source_url"] for item in first] == ["u1", "u2"]


def test_frozen_replay_checkpoint_hashes_still_match():
    audit = json.loads(Path("artifacts/phase13a_fusion/baseline_replay_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["phase11c"]["frozen_json_sha256"] == sha256(Path(audit["phase11c"]["frozen_json"]))
    assert audit["phase12"]["frozen_json_sha256"] == sha256(Path(audit["phase12"]["frozen_json"]))


def test_completed_report_set_is_ready_for_human_review():
    audit = validate_existing_outputs()
    assert audit["passed"] is True
    assert audit["status"] == "READY_FOR_HUMAN_REVIEW"
    assert audit["variant_count"] == 7
    assert audit["review_query_count"] == 60
    assert audit["candidate_score_contract"]["new_model_inference"] == 0
