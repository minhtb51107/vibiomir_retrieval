from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.processing.chunking import LightweightUnicodeTokenizer, TokenSpan
from src.submission.calibration import (
    _HashingStream,
    generate_calibration_experiments,
    load_calibration_config,
    validate_calibration_outputs,
)
from src.submission.common import file_sha256, write_submission_stream
from src.submission.expansion import SourceWindowExpander, compute_window
from src.submission.validator import (
    SubmissionValidationError,
    canonical_provenance,
    expected_query_ids,
    load_source_documents,
    load_submission_payload,
    validate_submission,
)

REAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase10a_calibration.yaml"

DOC10 = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi"
DOC20 = "short doc here"
DOC30 = "third document text"
WORDS10 = DOC10.split()


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _span(text: str, first: int, last: int) -> tuple[int, int, str]:
    matches = list(re.finditer(r"\S+", text))
    start = matches[first].start()
    end = matches[last].end()
    return start, end, text[start:end]


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    queries = [{"id": 2, "query": "hai"}, {"id": 1, "query": "một"}]
    s1, e1, t1 = _span(DOC10, 0, 5)
    s2, e2, t2 = _span(DOC10, 6, 11)
    chunks = [
        {"chunk_id": "c1", "doc_id": 10, "raw_text": t1, "start_offset": s1, "end_offset": e1},
        {"chunk_id": "c2", "doc_id": 10, "raw_text": t2, "start_offset": s2, "end_offset": e2},
        {"chunk_id": "c3", "doc_id": 20, "raw_text": DOC20, "start_offset": 0, "end_offset": len(DOC20)},
        {"chunk_id": "c4", "doc_id": 30, "raw_text": DOC30, "start_offset": 0, "end_offset": len(DOC30)},
    ]
    _write(tmp_path / "queries.parquet", queries)
    _write(tmp_path / "chunks.parquet", chunks)
    _write(
        tmp_path / "docs.parquet",
        [
            {"doc_id": 10, "normalized_text": DOC10},
            {"doc_id": 20, "normalized_text": DOC20},
            {"doc_id": 30, "normalized_text": DOC30},
        ],
    )
    pool, selected, baseline_docs, hybrid_chunks, hybrid_docs = [], [], [], [], []
    for query_id in (2, 1):
        for rank, (chunk, score) in enumerate(zip(chunks, (5.0, 4.0, 3.0, 2.0)), start=1):
            row = {
                "query_id": query_id,
                "query_text": "q",
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "chunk_text": chunk["raw_text"],
                "source_url": "u",
                "rerank_score": score,
            }
            pool.append(row)
            if rank <= 2:
                selected.append({**row, "selection_rank": rank})
            hybrid_chunks.append(
                {
                    "query_id": query_id,
                    "rank": 5 - rank,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "chunk_text": chunk["raw_text"],
                }
            )
        baseline_docs.append({"query_id": query_id, "doc_id": 10, "rank": 1})
        hybrid_docs.extend(
            [
                {"query_id": query_id, "doc_id": 30, "rank": 1},
                {"query_id": query_id, "doc_id": 10, "rank": 2},
            ]
        )
    _write(tmp_path / "pool.parquet", pool)
    _write(tmp_path / "selected.parquet", selected)
    _write(tmp_path / "baseline_docs.parquet", baseline_docs)
    _write(tmp_path / "hybrid_chunks.parquet", hybrid_chunks)
    _write(tmp_path / "hybrid_docs.parquet", hybrid_docs)

    def exp(identifier, name, docs, chunks_selection, expand=None):
        row = {
            "id": identifier,
            "name": name,
            "hypothesis": "h",
            "difference_from_baseline": "d",
            "documents": docs,
            "chunks": chunks_selection,
        }
        if expand:
            row["expand"] = expand
        return row

    base_docs = {"source": "baseline", "depth": 1}
    base_chunks = {"source": "selected", "depth": 2}
    config = {
        "version": "test-v1",
        "organizer_schema": {"required_query_count": 2},
        "inputs": {
            "queries": str(tmp_path / "queries.parquet"),
            "canonical_chunks": str(tmp_path / "chunks.parquet"),
            "source_documents": str(tmp_path / "docs.parquet"),
            "tokenizer": {"name": "test", "local_files_only": True},
        },
        "output": {
            "directory": str(tmp_path / "out"),
            "artifact_directory": str(tmp_path / "artifacts"),
            "manifest": "manifest.json",
        },
        "sources": {
            "documents": {
                "baseline": {"kind": "ranked", "path": str(tmp_path / "baseline_docs.parquet")},
                "pool": {
                    "kind": "pool_best_chunk",
                    "path": str(tmp_path / "pool.parquet"),
                    "pool_depth": 100,
                },
                "hybrid": {"kind": "ranked", "path": str(tmp_path / "hybrid_docs.parquet")},
            },
            "chunks": {
                "selected": {
                    "kind": "ranked",
                    "path": str(tmp_path / "selected.parquet"),
                    "rank_column": "selection_rank",
                },
                "pool": {
                    "kind": "pool_pure",
                    "path": str(tmp_path / "pool.parquet"),
                    "pool_depth": 100,
                },
                "hybrid": {
                    "kind": "ranked",
                    "path": str(tmp_path / "hybrid_chunks.parquet"),
                    "rank_column": "rank",
                },
            },
        },
        "baseline": {
            "name": "phase9_base",
            "documents": base_docs,
            "chunks": base_chunks,
        },
        "experiments": [
            exp("E1", "phase10a_E1_docs_more", {"source": "pool", "depth": 5}, base_chunks),
            exp("E4", "phase10a_E4_chunks1", base_docs, {"source": "selected", "depth": 1}),
            exp("E5", "phase10a_E5_chunks_pool", base_docs, {"source": "pool", "depth": 10}),
            exp(
                "E6",
                "phase10a_E6_expand",
                base_docs,
                base_chunks,
                expand={"target_tokens": 10},
            ),
            exp(
                "E8",
                "phase10a_E8_hybrid",
                {"source": "hybrid", "depth": 5},
                {"source": "hybrid", "depth": 4},
            ),
        ],
    }
    path = tmp_path / "phase10a.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config


def _generate(tmp_path: Path, config: dict, config_path: Path, **kwargs):
    return generate_calibration_experiments(
        config,
        config_path=config_path,
        tokenizer_factory=LightweightUnicodeTokenizer,
        **kwargs,
    )


def _entry(manifest: dict, identifier: str) -> dict:
    return next(row for row in manifest["experiments"] if row["id"] == identifier)


# ------------------------------------------------------------------ depth override


def test_document_depth_override_extends_baseline_without_inventing_docs(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    manifest = _generate(tmp_path, config, config_path)
    entry = _entry(manifest, "E1")
    # Pool only contains documents 10, 20, 30 even though the cap is 5.
    assert entry["documents_per_query"]["min"] == 3
    assert entry["documents_per_query"]["max"] == 3
    assert entry["configuration"]["documents_depth_cap"] == 5
    payload = load_submission_payload(tmp_path / "out" / "phase10a_E1_docs_more.json")
    assert payload[0]["relevant_docs"] == [10, 20, 30]
    assert entry["versus_baseline"]["documents_per_query_relation"] == {
        "baseline_is_prefix_of_experiment": 2
    }
    assert entry["versus_baseline"]["underlying_chunks_per_query_relation"] == {"identical": 2}
    assert entry["unique_relevant_doc_ids"] == 3
    assert entry["invalid_doc_ids"] == 0


def test_chunk_depth_override_fewer_and_more(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    manifest = _generate(tmp_path, config, config_path)
    fewer = _entry(manifest, "E4")
    more = _entry(manifest, "E5")
    assert fewer["chunks_per_query"]["max"] == 1
    assert fewer["versus_baseline"]["underlying_chunks_per_query_relation"] == {
        "experiment_is_prefix_of_baseline": 2
    }
    assert more["chunks_per_query"]["min"] == 4  # data-limited: only 4 persisted chunks
    assert more["versus_baseline"]["underlying_chunks_per_query_relation"] == {
        "baseline_is_prefix_of_experiment": 2
    }
    payload = load_submission_payload(tmp_path / "out" / "phase10a_E5_chunks_pool.json")
    assert [row["doc_id"] for row in payload[0]["relevant_chunks"]] == [10, 10, 20, 30]


def test_hybrid_source_uses_plain_rank_column(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    manifest = _generate(tmp_path, config, config_path)
    payload = load_submission_payload(tmp_path / "out" / "phase10a_E8_hybrid.json")
    # hybrid rank = 5 - baseline rank, so c4 (doc 30) comes first.
    assert payload[0]["relevant_docs"] == [30, 10]
    assert payload[0]["relevant_chunks"][0]["doc_id"] == 30
    assert _entry(manifest, "E8")["chunks_per_query"]["max"] == 4


# ------------------------------------------------------------------ expansion


def test_compute_window_centres_clamps_and_spends_remaining_budget() -> None:
    assert compute_window(token_count=100, first_token=40, last_token_exclusive=50, target_tokens=30) == (30, 60)
    # Left edge clamped: remaining budget moves to the right side.
    assert compute_window(token_count=100, first_token=2, last_token_exclusive=12, target_tokens=30) == (0, 30)
    # Right edge clamped: remaining budget moves to the left side.
    assert compute_window(token_count=100, first_token=90, last_token_exclusive=98, target_tokens=30) == (70, 100)
    # Document smaller than target: whole document.
    assert compute_window(token_count=12, first_token=2, last_token_exclusive=6, target_tokens=50) == (0, 12)
    # Chunk already at least target size: unchanged.
    assert compute_window(token_count=100, first_token=10, last_token_exclusive=60, target_tokens=30) == (10, 60)


def test_expanded_windows_are_contiguous_verbatim_and_contain_original(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    manifest = _generate(tmp_path, config, config_path)
    entry = _entry(manifest, "E6")
    expansion = entry["expansion"]
    assert expansion["verbatim_failures"] == 0
    assert expansion["original_containment_failures"] == 0
    assert expansion["expanded_beyond_original"] > 0
    payload = load_submission_payload(tmp_path / "out" / "phase10a_E6_expand.json")
    texts = {10: DOC10, 20: DOC20, 30: DOC30}
    originals = {"c1": _span(DOC10, 0, 5)[2], "c2": _span(DOC10, 6, 11)[2]}
    chunks = payload[0]["relevant_chunks"]
    assert [chunk["doc_id"] for chunk in chunks] == [10, 10]
    for chunk, original in zip(chunks, originals.values()):
        assert chunk["chunk_text"] in texts[chunk["doc_id"]]
        assert original in chunk["chunk_text"]
        assert len(chunk["chunk_text"]) > len(original)
    assert chunks[0]["chunk_text"] == " ".join(WORDS10[0:10])
    assert chunks[1]["chunk_text"] == " ".join(WORDS10[4:14])


def test_expansion_snaps_outward_to_word_boundaries() -> None:
    class ThreeCharTokenizer:
        def spans(self, text: str) -> list[TokenSpan]:
            output = []
            position = 0
            for word in text.split(" "):
                for offset in range(0, len(word), 3):
                    piece = word[offset : offset + 3]
                    output.append(TokenSpan(piece, position + offset, position + offset + len(piece)))
                position += len(word) + 1
            return output

    text = "lorem ipsumdolor sitametconsectetur adipiscing elitseddo eiusmodtempor incididunt"
    start = text.index("sitamet")
    end = start + len("sitametconsectetur")
    expander = SourceWindowExpander({1: text}, ThreeCharTokenizer(), 8)
    span = expander.expand(doc_id=1, start=start, end=end, original_text=text[start:end])
    assert span.text == text[span.start : span.end]
    assert text[start:end] in span.text
    assert span.start == 0 or text[span.start - 1] == " "
    assert span.end == len(text) or text[span.end] == " "
    assert span.changed


def test_expander_rejects_offset_text_mismatch() -> None:
    expander = SourceWindowExpander({1: DOC10}, LightweightUnicodeTokenizer(), 10)
    with pytest.raises(ValueError, match="do not reproduce chunk text"):
        expander.expand(doc_id=1, start=0, end=5, original_text="wrong")


def test_whole_document_chunk_is_unchanged_when_doc_is_shorter_than_target() -> None:
    expander = SourceWindowExpander({20: DOC20}, LightweightUnicodeTokenizer(), 100)
    span = expander.expand(doc_id=20, start=0, end=len(DOC20), original_text=DOC20)
    assert span.text == DOC20
    assert span.covers_whole_document
    assert not span.changed


# ------------------------------------------------------------------ provenance


def test_expanded_submission_validates_only_with_source_span_check(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    manifest = _generate(tmp_path, config, config_path)
    results = validate_calibration_outputs(config, manifest, tmp_path / "out")
    assert all(row["valid"] for row in results)
    assert all(row["chunk_provenance_mismatches"] == 0 for row in results)
    assert all(row["zip_single_root_json"] and row["zip_member_sha256_matches_json"] for row in results)
    expanded = next(row for row in results if row["id"] == "E6")
    assert expanded["source_span_chunks_verified"] > 0
    documents, chunks = canonical_provenance(config["inputs"]["canonical_chunks"])
    queries = expected_query_ids(config["inputs"]["queries"])
    with pytest.raises(SubmissionValidationError, match="provenance mismatch"):
        validate_submission(
            tmp_path / "out" / "phase10a_E6_expand.zip",
            expected_queries=queries,
            valid_document_ids=documents,
            valid_chunks=chunks,
        )


def test_validator_rejects_rewritten_expansion(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    _generate(tmp_path, config, config_path)
    payload = load_submission_payload(tmp_path / "out" / "phase10a_E6_expand.json")
    payload[0]["relevant_chunks"][0]["chunk_text"] += " rewritten"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    documents, chunks = canonical_provenance(config["inputs"]["canonical_chunks"])
    sources = load_source_documents(config["inputs"]["source_documents"], documents)
    with pytest.raises(SubmissionValidationError, match="provenance mismatch"):
        validate_submission(
            tampered,
            expected_queries=expected_query_ids(config["inputs"]["queries"]),
            valid_document_ids=documents,
            valid_chunks=chunks,
            source_documents=sources,
        )


def test_zip_has_single_root_json(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    _generate(tmp_path, config, config_path)
    with zipfile.ZipFile(tmp_path / "out" / "phase10a_E1_docs_more.zip") as archive:
        assert archive.namelist() == ["phase10a_E1_docs_more.json"]


# ------------------------------------------------------------------ determinism


def test_generation_is_deterministic_including_expansion(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    first = _generate(tmp_path, config, config_path)
    second = _generate(tmp_path, config, config_path, output_directory=tmp_path / "second")
    assert [(r["name"], r["json_sha256"], r["zip_sha256"]) for r in first["experiments"]] == [
        (r["name"], r["json_sha256"], r["zip_sha256"]) for r in second["experiments"]
    ]
    for entry in first["experiments"]:
        assert len(entry["json_sha256"]) == len(entry["zip_sha256"]) == 64


def test_hashing_stream_matches_written_submission_bytes(tmp_path: Path) -> None:
    records = [{"id": 1, "relevant_docs": [3], "relevant_chunks": []}]
    path = tmp_path / "x.json"
    write_submission_stream(path, iter(records))
    assert _HashingStream().consume(iter(records)) == file_sha256(path)


# ------------------------------------------------------------------ naming / config


def test_real_config_declares_exactly_the_nine_requested_experiments() -> None:
    config = load_calibration_config(REAL_CONFIG)
    rows = {row["id"]: row for row in config["experiments"]}
    assert [row["name"] for row in config["experiments"]] == [
        "phase10a_E1_docs20",
        "phase10a_E2_docs50",
        "phase10a_E3_docs_max",
        "phase10a_E4_chunks20",
        "phase10a_E5_chunks_more",
        "phase10a_E6_chunk_expand_768",
        "phase10a_E7_chunk_expand_1024",
        "phase10a_E8_hybrid",
        "phase10a_E9_sparse",
    ]
    baseline = config["baseline"]
    assert (baseline["documents"]["depth"], baseline["chunks"]["depth"]) == (10, 50)
    assert rows["E1"]["documents"]["depth"] == 20 and rows["E2"]["documents"]["depth"] == 50
    assert rows["E3"]["documents"]["depth"] == 100
    assert rows["E4"]["chunks"]["depth"] == 20 and rows["E5"]["chunks"]["depth"] == 100
    assert rows["E6"]["expand"]["target_tokens"] == 768
    assert rows["E7"]["expand"]["target_tokens"] == 1024
    assert rows["E8"]["chunks"]["source"].startswith("phase6_hybrid")
    assert rows["E9"]["chunks"]["source"].startswith("phase6_sparse")
    # Single-hypothesis design: only E6/E7 expand text.
    assert {key for key, row in rows.items() if "expand" in row} == {"E6", "E7"}


def test_config_rejects_duplicate_ids_names_and_bad_naming(tmp_path: Path) -> None:
    config = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    duplicate = {**config, "experiments": [*config["experiments"], dict(config["experiments"][0])]}
    path = tmp_path / "dup.yaml"
    path.write_text(yaml.safe_dump(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="must be unique"):
        load_calibration_config(path)
    bad = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    bad["experiments"][0]["name"] = "wrong_name"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid experiment name"):
        load_calibration_config(path)
    mismatch = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    mismatch["experiments"][0]["name"] = "phase10a_E7_docs20"
    path.write_text(yaml.safe_dump(mismatch), encoding="utf-8")
    with pytest.raises(ValueError, match="must embed id"):
        load_calibration_config(path)
