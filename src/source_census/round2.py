from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from src.source_census.pipeline import (
    atomic_json,
    balanced_round2_groups,
    score_database_status,
)


def prepare_round2_groups(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifacts"])
    report = json.loads((artifacts / "round1_report.json").read_text(encoding="utf-8"))
    parent_groups = {parent: report["groups"][parent] for parent in ("A", "C")}
    parts_root = Path(config["outputs"]["root"]) / "round1/source_candidate_parts"
    metadata: dict[str, dict[str, Any]] = {}
    for parent, members in parent_groups.items():
        for source in members:
            marker_matches = []
            for marker_path in parts_root.glob("*.json"):
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if marker.get("source") == source:
                    marker_matches.append(marker)
            if len(marker_matches) > 1:
                raise ValueError(f"expected at most one candidate marker for {source}, got {len(marker_matches)}")
            metadata[source] = {
                **report["sources"][source],
                # A technically healthy extracted source can still yield zero
                # chunks.  It remains in its organizer-confirmed parent group;
                # recording zero preserves the 42-source experimental design.
                "chunk_count": int(marker_matches[0]["source_chunk_count"]) if marker_matches else 0,
                "candidate_rows": int(marker_matches[0]["candidate_rows"]) if marker_matches else 0,
                "parent": parent,
            }
    assignment = balanced_round2_groups(parent_groups, metadata)
    assignment["source_metadata"] = metadata
    assignment["source_count"] = 42
    assignment["new_crawling"] = 0
    assignment["new_embeddings"] = 0
    assignment["new_reranking"] = 0
    atomic_json(artifacts / "round2_groups.json", assignment)
    return assignment


def _query_groups(path: Path) -> Iterable[list[dict[str, Any]]]:
    current: int | None = None
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            if current is not None and query_id != current:
                yield rows
                rows = []
            current = query_id
            rows.append(row)
    if rows:
        yield rows


def finalize_round2(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifacts"])
    group_artifact = artifacts / "round2_groups.json"
    assignment = (
        json.loads(group_artifact.read_text(encoding="utf-8"))
        if group_artifact.exists()
        else prepare_round2_groups(config)
    )
    groups = {
        name: [str(row["source"]) for row in rows]
        for name, rows in assignment["groups"].items()
    }
    source_to_group = {
        source: group for group, members in groups.items() for source in members
    }
    if len(source_to_group) != 42 or "suckhoecongdongonline.vn" in source_to_group:
        raise ValueError("invalid Round-2 survivor membership")

    score_status = score_database_status(config, verify_integrity=True)
    if score_status != {"total": 582000, "done": 582000, "remaining": 0, "integrity": "ok"}:
        raise ValueError(f"source score cache is not complete: {score_status}")

    root = Path(config["outputs"]["root"]) / "round1"
    combined_chunks = root / "combined_chunks.parquet"
    combined_docs = root / "combined_documents.parquet"
    if not combined_chunks.exists() or not combined_docs.exists():
        raise FileNotFoundError("canonical Round-1 combined document/chunk artifacts are required")

    from src.indexing.tokenizer_validation import HuggingFaceOffsetTokenizer
    from src.source_probe.pipeline import _ParquetSink, _all_query_groups, _generate_submission
    from transformers import AutoTokenizer

    tokenizer = HuggingFaceOffsetTokenizer(
        AutoTokenizer.from_pretrained(
            config["models"]["embedder"]["name"],
            revision=config["models"]["embedder"]["revision"],
            local_files_only=True,
        )
    )
    parts_root = root / "source_candidate_parts"
    part_sources: dict[str, Path] = {}
    for marker_path in sorted(parts_root.glob("*.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        part_path = marker_path.with_suffix(".parquet")
        if part_path.exists():
            part_sources[str(marker["source"])] = part_path
    expected_parts = {
        source
        for source in source_to_group
        if int(assignment["source_metadata"][source].get("candidate_rows", 0)) > 0
    }
    if not expected_parts.issubset(part_sources):
        missing = sorted(expected_parts - set(part_sources))
        raise ValueError(f"one or more Round-2 sources lack durable candidate parts: {missing}")

    part_iterators = {source: iter(_query_groups(path)) for source, path in part_sources.items()}
    pilot_iterator = iter(_all_query_groups(Path(config["inputs"]["pilot_rerank_pool"])))
    query_ids = [
        int(value)
        for value in pq.read_table(config["inputs"]["queries"], columns=["id"]).column(0).to_pylist()
    ]
    source_scores = sqlite3.connect(root / "source_scores.sqlite")
    pilot_scores = sqlite3.connect(config["inputs"]["pilot_rerank_scores"])
    ranking_paths: dict[str, tuple[Path, Path]] = {}
    chunk_sinks: dict[str, Any] = {}
    document_sinks: dict[str, Any] = {}
    for group in sorted(groups):
        destination = Path(config["outputs"]["root"]) / f"round2/group_{group}"
        destination.mkdir(parents=True, exist_ok=True)
        chunk_path = destination / "reranked_chunks.parquet"
        document_path = destination / "reranked_documents.parquet"
        ranking_paths[group] = (chunk_path, document_path)
        chunk_sinks[group] = _ParquetSink(chunk_path)
        document_sinks[group] = _ParquetSink(document_path)

    try:
        for query_id in query_ids:
            pilot_rows = next(pilot_iterator, None)
            if not pilot_rows or int(pilot_rows[0]["query_id"]) != query_id:
                raise ValueError(f"pilot candidate pool missing query {query_id}")
            pilot_score_map = {
                str(chunk_id): float(score)
                for chunk_id, score in pilot_scores.execute(
                    "SELECT chunk_id,rerank_score FROM scores WHERE query_id=?", (query_id,)
                )
            }
            pilot_ranked = [
                {**row, "rerank_score": pilot_score_map[str(row["chunk_id"])]}
                for row in pilot_rows
                if str(row["chunk_id"]) in pilot_score_map
            ]
            rows_by_group: dict[str, list[dict[str, Any]]] = {group: [] for group in groups}
            source_score_map = {
                str(chunk_id): float(score)
                for chunk_id, score in source_scores.execute(
                    "SELECT chunk_id,rerank_score FROM pairs WHERE query_id=?", (query_id,)
                )
            }
            for source, iterator in part_iterators.items():
                rows = next(iterator, None)
                if not rows or int(rows[0]["query_id"]) != query_id:
                    raise ValueError(f"source candidate part {source} missing query {query_id}")
                group = source_to_group.get(source)
                if group is not None:
                    rows_by_group[group].extend(
                        {**row, "rerank_score": source_score_map[str(row["chunk_id"])]}
                        for row in rows
                    )
            for group in sorted(groups):
                ranked = [*pilot_ranked, *rows_by_group[group]]
                ranked.sort(key=lambda row: (-float(row["rerank_score"]), str(row["chunk_id"])))
                chunk_sinks[group].add(
                    {
                        "query_id": query_id,
                        "rerank_rank": rank,
                        "chunk_id": str(row["chunk_id"]),
                        "doc_id": int(row["doc_id"]),
                        "chunk_text": str(row["chunk_text"]),
                    }
                    for rank, row in enumerate(ranked, 1)
                )
                seen_docs: set[int] = set()
                document_rows: list[dict[str, int]] = []
                for row in ranked:
                    doc_id = int(row["doc_id"])
                    if doc_id in seen_docs:
                        continue
                    seen_docs.add(doc_id)
                    document_rows.append(
                        {"query_id": query_id, "rank": len(seen_docs), "doc_id": doc_id}
                    )
                document_sinks[group].add(document_rows)
    finally:
        source_scores.close()
        pilot_scores.close()
        for sink in chunk_sinks.values():
            sink.close()
        for sink in document_sinks.values():
            sink.close()

    if next(pilot_iterator, None) is not None:
        raise ValueError("pilot candidate pool contains unexpected queries")
    for source, iterator in part_iterators.items():
        if next(iterator, None) is not None:
            raise ValueError(f"source candidate part {source} contains unexpected queries")

    submissions: dict[str, Any] = {}
    for group in sorted(groups):
        chunk_path, document_path = ranking_paths[group]
        name = f"phase10d_R2_group_{group.removeprefix('G')}"
        kwargs = {
            "name": name,
            "documents_path": document_path,
            "chunks_path": chunk_path,
            "canonical_path": combined_chunks,
            "source_documents_path": combined_docs,
            "output": Path(config["outputs"]["submissions"]),
            "tokenizer": tokenizer,
        }
        first = _generate_submission(config, **kwargs)
        second = _generate_submission(config, **kwargs)
        deterministic = (
            first["json_sha256"] == second["json_sha256"]
            and first["zip_sha256"] == second["zip_sha256"]
        )
        if not deterministic:
            raise RuntimeError(f"submission regeneration was not deterministic: {group}")
        submissions[group] = {**second, "determinism_verified": True}

    report = {
        "terminal_state": "WAITING_FOR_LEADERBOARD",
        "experimental_control": {
            "only_variable": "source membership",
            "new_crawling": 0,
            "new_extraction": 0,
            "new_chunking": 0,
            "new_embeddings": 0,
            "new_reranking": 0,
            "candidate_cap_m": 8,
            "reused_score_database": str(root / "source_scores.sqlite"),
            "reused_scores": 582000,
        },
        "group_assignment": assignment,
        "groups": groups,
        "score_cache": score_status,
        "submissions": submissions,
    }
    atomic_json(artifacts / "round2_report.json", report)
    return report


def package_round2_group(config: dict[str, Any], group: str) -> dict[str, Any]:
    """Package one already-streamed Round-2 group in a fresh process."""
    if group not in {f"G{index}" for index in range(1, 8)}:
        raise ValueError(f"invalid Round-2 group: {group}")
    artifacts = Path(config["outputs"]["artifacts"])
    assignment = json.loads((artifacts / "round2_groups.json").read_text(encoding="utf-8"))
    if group not in assignment["groups"]:
        raise ValueError(f"group is absent from fixed assignment: {group}")
    score_status = score_database_status(config, verify_integrity=True)
    if score_status != {"total": 582000, "done": 582000, "remaining": 0, "integrity": "ok"}:
        raise ValueError(f"source score cache is not complete: {score_status}")

    root = Path(config["outputs"]["root"])
    chunk_path = root / f"round2/group_{group}/reranked_chunks.parquet"
    document_path = root / f"round2/group_{group}/reranked_documents.parquet"
    combined_chunks = root / "round1/combined_chunks.parquet"
    combined_docs = root / "round1/combined_documents.parquet"
    for path in (chunk_path, document_path, combined_chunks, combined_docs):
        if not path.exists():
            raise FileNotFoundError(path)

    from src.indexing.tokenizer_validation import HuggingFaceOffsetTokenizer
    from src.source_probe.pipeline import _generate_submission
    from transformers import AutoTokenizer

    tokenizer = HuggingFaceOffsetTokenizer(
        AutoTokenizer.from_pretrained(
            config["models"]["embedder"]["name"],
            revision=config["models"]["embedder"]["revision"],
            local_files_only=True,
        )
    )
    name = f"phase10d_R2_group_{group.removeprefix('G')}"
    kwargs = {
        "name": name,
        "documents_path": document_path,
        "chunks_path": chunk_path,
        "canonical_path": combined_chunks,
        "source_documents_path": combined_docs,
        "output": Path(config["outputs"]["submissions"]),
        "tokenizer": tokenizer,
    }
    first = _generate_submission(config, **kwargs)
    second = _generate_submission(config, **kwargs)
    deterministic = (
        first["json_sha256"] == second["json_sha256"]
        and first["zip_sha256"] == second["zip_sha256"]
    )
    if not deterministic:
        raise RuntimeError(f"submission regeneration was not deterministic: {group}")
    result = {**second, "determinism_verified": True}
    atomic_json(artifacts / f"round2_submission_{group}.json", result)
    return result


def finalize_round2_report(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifacts"])
    assignment = json.loads((artifacts / "round2_groups.json").read_text(encoding="utf-8"))
    submissions = {}
    for index in range(1, 8):
        group = f"G{index}"
        marker = artifacts / f"round2_submission_{group}.json"
        if not marker.exists():
            raise FileNotFoundError(marker)
        submissions[group] = json.loads(marker.read_text(encoding="utf-8"))
    groups = {
        name: [str(row["source"]) for row in rows]
        for name, rows in assignment["groups"].items()
    }
    score_status = score_database_status(config, verify_integrity=True)
    if score_status != {"total": 582000, "done": 582000, "remaining": 0, "integrity": "ok"}:
        raise ValueError(f"source score cache is not complete: {score_status}")
    report = {
        "terminal_state": "WAITING_FOR_LEADERBOARD",
        "experimental_control": {
            "only_variable": "source membership",
            "new_crawling": 0,
            "new_extraction": 0,
            "new_chunking": 0,
            "new_embeddings": 0,
            "new_reranking": 0,
            "candidate_cap_m": 8,
            "reused_score_database": str(Path(config["outputs"]["root"]) / "round1/source_scores.sqlite"),
            "reused_scores": 582000,
        },
        "group_assignment": assignment,
        "groups": groups,
        "score_cache": score_status,
        "submissions": submissions,
    }
    atomic_json(artifacts / "round2_report.json", report)
    atomic_json(
        Path("artifacts/runs/phase10d_round2/state.json"),
        {
            "run_id": "phase10d_round2",
            "stage": "WAITING_FOR_LEADERBOARD",
            "status": "WAITING_FOR_LEADERBOARD",
            "new_model_inference": 0,
        },
    )
    return report
