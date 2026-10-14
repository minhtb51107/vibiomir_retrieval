from __future__ import annotations

import sqlite3
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.source_census.phase14_candidates import _risk_texts, load_config
from tools.phase14_rerank import (
    _valid_cache, diagnose_missing_canonical_documents, final_score_audit, runtime_config,
)
from tools.score_source_census_persistent import score_equivalence_control


def test_phase14_candidate_loader_preserves_fixed_policy() -> None:
    config = load_config("configs/phase14_medlatec_coverage_3000.yaml")
    assert config["candidate_cache"] == {
        "candidate_caps": [16], "dense_depth": 16, "sparse_depth": 16,
        "rrf_constant": 60, "bm25_k1": 1.2, "bm25_b": 0.75,
    }


def test_phase14_runtime_reuses_existing_new_official_pilot_pool() -> None:
    config = load_config("configs/phase14_medlatec_coverage_3000.yaml")
    runtime = __import__("yaml").safe_load(runtime_config(config).read_text(encoding="utf-8"))
    pilot = Path(runtime["inputs"]["pilot_rerank_pool"])
    assert pilot == Path("data/source_census/phase11b_new_corpus/round1/pilot_rerank_pool.new_official.parquet")
    assert pilot.exists()


def test_completed_phase14_score_cache_passes_terminal_read_only_audit() -> None:
    config = load_config("configs/phase14_medlatec_coverage_3000.yaml")
    result = final_score_audit(config)
    assert result["passed"] is True
    assert result["total_rows"] == result["unique_score_keys"] == 254400
    assert result["reused_scores"] == 250114
    assert result["newly_computed_scores"] == 4286


def test_phase14_missing_canonical_documents_are_exactly_recoverable() -> None:
    config = load_config("configs/phase14_medlatec_coverage_3000.yaml")
    ranked = Path(config["outputs"]["root"]) / "rankings/PHASE14/reranked_chunks.parquet"
    doc_ids = {
        int(value)
        for batch in pq.ParquetFile(ranked).iter_batches(columns=["doc_id"], batch_size=8192)
        for value in batch.column(0).to_pylist()
    }
    result = diagnose_missing_canonical_documents(config, ranked, doc_ids)
    assert result["passed"] is True
    assert result["missing_doc_ids"] == [
        175609, 177378, 177410, 178448, 178490, 179052, 179434, 180963,
        181723, 185743, 185988, 188108, 194121, 195389, 196041,
    ]
    assert all(row["official_new_corpus_member"] for row in result["entries"])
    assert all(row["candidate_url_matches_canonical"] for row in result["entries"])
    assert all(row["all_ranked_chunks_in_phase10f"] for row in result["entries"])


def test_risk_texts_requires_exact_duplicate_and_does_not_modify_rows(tmp_path: Path) -> None:
    path = tmp_path / "chunks.parquet"
    rows = [
        {"raw_text": "Tin cùng chuyên mục", "heading_path": [], "chunk_id": "a"},
        {"raw_text": "Tin cùng chuyên mục", "heading_path": [], "chunk_id": "b"},
        {"raw_text": "Đặt lịch thăm khám tại MEDLATEC", "heading_path": [], "chunk_id": "c"},
        {"raw_text": "Nội dung y khoa", "heading_path": [], "chunk_id": "d"},
        {"raw_text": "Nội dung y khoa", "heading_path": [], "chunk_id": "e"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    before = path.read_bytes()
    risk = _risk_texts(path)
    assert risk == {"Tin cùng chuyên mục"}
    assert path.read_bytes() == before


def test_candidate_checkpoint_schema_enforces_unique_chunk_identity(tmp_path: Path) -> None:
    database = tmp_path / "checkpoint.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE chunks(row_index INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE)")
    connection.execute("INSERT INTO chunks VALUES(0,'x')")
    try:
        connection.execute("INSERT INTO chunks VALUES(1,'x')")
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("duplicate chunk identity was accepted")
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_historical_cache_gate_rejects_incomplete_scores(tmp_path: Path) -> None:
    path = tmp_path / "scores.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE pairs(query_id INTEGER,chunk_id TEXT,rerank_score REAL)")
    connection.executemany("INSERT INTO pairs VALUES(?,?,?)", [(1, "a", 0.5), (1, "b", None)])
    connection.commit(); connection.close()
    assert _valid_cache(path) is None


def test_historical_cache_gate_accepts_complete_finite_scores(tmp_path: Path) -> None:
    path = tmp_path / "scores.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE pairs(query_id INTEGER,chunk_id TEXT,rerank_score REAL)")
    connection.executemany("INSERT INTO pairs VALUES(?,?,?)", [(1, "a", 0.5), (1, "b", -1.0)])
    connection.commit(); connection.close()
    assert _valid_cache(path)["rows"] == 2


def test_equivalence_control_replays_exact_historical_batch_without_mixing(tmp_path: Path) -> None:
    source = tmp_path / "historical.sqlite"
    destination = tmp_path / "current.sqlite"
    for path in (source, destination):
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE pairs(query_id INTEGER,chunk_id TEXT,query_text TEXT,chunk_text TEXT,rerank_score REAL)"
        )
        connection.commit(); connection.close()
    source_connection = sqlite3.connect(source)
    source_connection.executemany(
        "INSERT INTO pairs VALUES(?,?,?,?,?)",
        [(1, "partner", "q", "short", -1.0), (1, "target", "q", "target text", -2.0)],
    )
    source_connection.commit(); source_connection.close()
    destination_connection = sqlite3.connect(destination)
    destination_connection.execute("INSERT INTO pairs VALUES(?,?,?,?,?)", (1, "target", "q", "target text", -2.0))
    destination_connection.commit()

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    control = {"pairs": [{
        "query_id": 1, "chunk_id": "target", "reference_score": -2.0,
        "historical_batch": {
            "source_score_database": str(source), "target_index": 1,
            "pairs": [
                {"query_id": 1, "chunk_id": "partner", "query_text_sha256": digest("q"), "chunk_text_sha256": digest("short")},
                {"query_id": 1, "chunk_id": "target", "query_text_sha256": digest("q"), "chunk_text_sha256": digest("target text")},
            ],
        },
    }]}

    class FakeReranker:
        def __init__(self) -> None:
            self.calls = []

        def score_pairs(self, pairs):
            self.calls.append(list(pairs))
            return [-1.0, -2.0]

    reranker = FakeReranker()
    sample, observed, contexts = score_equivalence_control(reranker, destination_connection, control)
    destination_connection.close()
    assert len(sample) == 1 and observed == [-2.0]
    assert reranker.calls == [[("q", "short"), ("q", "target text")]]
    assert contexts[0]["batch_keys"] == ["1:partner", "1:target"]


def test_equivalence_control_rejects_changed_historical_text(tmp_path: Path) -> None:
    source = tmp_path / "historical.sqlite"
    destination = tmp_path / "current.sqlite"
    for path in (source, destination):
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE pairs(query_id INTEGER,chunk_id TEXT,query_text TEXT,chunk_text TEXT,rerank_score REAL)"
        )
        connection.execute("INSERT INTO pairs VALUES(1,'target','q','actual',-2.0)")
        connection.commit(); connection.close()
    control = {"pairs": [{
        "query_id": 1, "chunk_id": "target", "reference_score": -2.0,
        "historical_batch": {
            "source_score_database": str(source), "target_index": 0,
            "pairs": [
                {"query_id": 1, "chunk_id": "target", "query_text_sha256": hashlib.sha256(b"q").hexdigest(), "chunk_text_sha256": hashlib.sha256(b"different").hexdigest()},
                {"query_id": 1, "chunk_id": "target", "query_text_sha256": hashlib.sha256(b"q").hexdigest(), "chunk_text_sha256": hashlib.sha256(b"different").hexdigest()},
            ],
        },
    }]}
    connection = sqlite3.connect(destination)
    try:
        try:
            score_equivalence_control(object(), connection, control)
        except RuntimeError as error:
            assert "batch text changed" in str(error)
        else:
            raise AssertionError("changed historical control text was accepted")
    finally:
        connection.close()
