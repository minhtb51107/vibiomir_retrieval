from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from src.source_census.pipeline import atomic_json, score_database_status


def _require_complete_score_cache(config: dict[str, Any]) -> dict[str, int | str]:
    """Validate cache invariants without assuming a phase-specific row count."""
    parts_root = Path(config["outputs"]["root"]) / "round1/source_candidate_parts"
    expected_total = sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in sorted(parts_root.glob("*.parquet"))
    )
    status = score_database_status(config, verify_integrity=True)
    total = int(status.get("total", -1))
    done = int(status.get("done", -1))
    remaining = int(status.get("remaining", -1))
    integrity = str(status.get("integrity", ""))
    failures: list[str] = []
    if integrity != "ok":
        failures.append(f"integrity={integrity!r}")
    if remaining != 0:
        failures.append(f"remaining={remaining}")
    if done != total:
        failures.append(f"done={done} differs from total={total}")
    if expected_total <= 0:
        failures.append("no durable source candidate rows found")
    elif total != expected_total:
        failures.append(f"total={total} differs from expected candidate keys={expected_total}")
    score_path = Path(config["outputs"]["root"]) / "round1/source_scores.sqlite"
    if score_path.exists() and expected_total > 0:
        connection = sqlite3.connect(score_path)
        connection.execute(
            "CREATE TEMP TABLE expected_keys(query_id INTEGER, chunk_id TEXT, "
            "PRIMARY KEY(query_id,chunk_id)) WITHOUT ROWID"
        )
        candidate_rows = 0
        for path in sorted(parts_root.glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(columns=["query_id", "chunk_id"], batch_size=8192):
                rows = [(int(q), str(c)) for q, c in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True)]
                candidate_rows += len(rows)
                connection.executemany("INSERT OR IGNORE INTO expected_keys VALUES (?,?)", rows)
        unique_expected = int(connection.execute("SELECT COUNT(*) FROM expected_keys").fetchone()[0])
        missing_keys = int(connection.execute(
            "SELECT COUNT(*) FROM expected_keys AS e LEFT JOIN pairs AS p "
            "ON p.query_id=e.query_id AND p.chunk_id=e.chunk_id WHERE p.query_id IS NULL"
        ).fetchone()[0])
        unexpected_keys = int(connection.execute(
            "SELECT COUNT(*) FROM pairs AS p LEFT JOIN expected_keys AS e "
            "ON e.query_id=p.query_id AND e.chunk_id=p.chunk_id WHERE e.query_id IS NULL"
        ).fetchone()[0])
        connection.close()
        if unique_expected != candidate_rows:
            failures.append(f"duplicate candidate keys={candidate_rows-unique_expected}")
        if missing_keys:
            failures.append(f"missing score keys={missing_keys}")
        if unexpected_keys:
            failures.append(f"unexpected score keys={unexpected_keys}")
    if failures:
        raise ValueError(
            "source score cache is not complete: " + "; ".join(failures)
            + f"; status={status}"
        )
    return status


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


def validate_cached_sources(config: dict[str, Any], groups: dict[str, list[str]]) -> None:
    artifact_root = Path(config["outputs"]["artifacts"])
    round1_path = artifact_root / "round1_report.json"
    if round1_path.exists():
        round1 = json.loads(round1_path.read_text(encoding="utf-8"))
        known = set(round1["groups"]["A"]) | set(round1["groups"]["B"]) | set(round1["groups"]["C"])
    else:
        triage = json.loads((artifact_root / "technical_triage.json").read_text(encoding="utf-8"))
        known = set(triage["healthy_sources"])
    for name, sources in groups.items():
        if not sources or len(sources) != len(set(sources)):
            raise ValueError(f"{name} must contain unique sources")
        unknown = sorted(set(sources) - known)
        if unknown:
            raise ValueError(f"{name} contains sources outside the cached census: {unknown}")
        if "suckhoecongdongonline.vn" in sources:
            raise ValueError("S4 is not part of cached census subsets")
    _require_complete_score_cache(config)


def build_cached_subset_rankings(
    config: dict[str, Any], groups: dict[str, list[str]], output_root: str | Path
) -> dict[str, Any]:
    """Stream fixed cached candidates into one or more source-subset rankings."""
    validate_cached_sources(config, groups)
    from src.source_probe.pipeline import _ParquetSink, _all_query_groups

    root = Path(config["outputs"]["root"]) / "round1"
    parts_root = root / "source_candidate_parts"
    part_sources: dict[str, Path] = {}
    for marker_path in sorted(parts_root.glob("*.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        part_path = marker_path.with_suffix(".parquet")
        if part_path.exists():
            part_sources[str(marker["source"])] = part_path
    target_sources = set(source for sources in groups.values() for source in sources)
    # Sources with no produced chunks intentionally have no candidate part.
    required = set(target_sources)
    if not required.issubset(part_sources):
        raise ValueError(f"missing candidate parts: {sorted(required - set(part_sources))}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    part_iterators = {source: iter(_query_groups(path)) for source, path in part_sources.items()}
    pilot_iterator = iter(_all_query_groups(Path(config["inputs"]["pilot_rerank_pool"])))
    query_ids = [
        int(value)
        for value in pq.read_table(config["inputs"]["queries"], columns=["id"]).column(0).to_pylist()
    ]
    source_scores = sqlite3.connect(root / "source_scores.sqlite")
    pilot_scores = sqlite3.connect(config["inputs"]["pilot_rerank_scores"])
    ranking_paths: dict[str, dict[str, str]] = {}
    chunk_sinks: dict[str, Any] = {}
    document_sinks: dict[str, Any] = {}
    for name in sorted(groups):
        destination = output_root / name
        destination.mkdir(parents=True, exist_ok=True)
        chunk_path = destination / "reranked_chunks.parquet"
        document_path = destination / "reranked_documents.parquet"
        ranking_paths[name] = {"chunks": str(chunk_path), "documents": str(document_path)}
        chunk_sinks[name] = _ParquetSink(chunk_path)
        document_sinks[name] = _ParquetSink(document_path)

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
                for row in pilot_rows if str(row["chunk_id"]) in pilot_score_map
            ]
            rows_by_source: dict[str, list[dict[str, Any]]] = {}
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
                if source in target_sources:
                    rows_by_source[source] = [
                        {**row, "rerank_score": source_score_map[str(row["chunk_id"])]}
                        for row in rows
                    ]
            for name, sources in sorted(groups.items()):
                ranked = [*pilot_ranked]
                for source in sources:
                    ranked.extend(rows_by_source.get(source, []))
                ranked.sort(key=lambda row: (-float(row["rerank_score"]), str(row["chunk_id"])))
                chunk_sinks[name].add(
                    {
                        "query_id": query_id,
                        "rerank_rank": rank,
                        "chunk_id": str(row["chunk_id"]),
                        "doc_id": int(row["doc_id"]),
                        "chunk_text": str(row["chunk_text"]),
                    }
                    for rank, row in enumerate(ranked, 1)
                )
                seen: set[int] = set()
                documents: list[dict[str, int]] = []
                for row in ranked:
                    doc_id = int(row["doc_id"])
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    documents.append({"query_id": query_id, "rank": len(seen), "doc_id": doc_id})
                document_sinks[name].add(documents)
    finally:
        source_scores.close()
        pilot_scores.close()
        for sink in chunk_sinks.values():
            sink.close()
        for sink in document_sinks.values():
            sink.close()
    return {"groups": groups, "rankings": ranking_paths, "new_model_inference": 0}


def package_cached_subset(
    config: dict[str, Any], *, ranking_root: str | Path, group: str,
    submission_name: str, output_dir: str | Path, marker_path: str | Path,
) -> dict[str, Any]:
    """Generate and strictly validate one cached subset twice, deterministically."""
    _require_complete_score_cache(config)
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
    root = Path(config["outputs"]["root"])
    ranking = Path(ranking_root) / group
    kwargs = {
        "name": submission_name,
        "documents_path": ranking / "reranked_documents.parquet",
        "chunks_path": ranking / "reranked_chunks.parquet",
        "canonical_path": root / "round1/combined_chunks.parquet",
        "source_documents_path": root / "round1/combined_documents.parquet",
        "output": Path(output_dir),
        "tokenizer": tokenizer,
    }
    first = _generate_submission(config, **kwargs)
    second = _generate_submission(config, **kwargs)
    if first["json_sha256"] != second["json_sha256"] or first["zip_sha256"] != second["zip_sha256"]:
        raise RuntimeError(f"submission regeneration was not deterministic: {group}")
    result = {**second, "determinism_verified": True, "new_model_inference": 0,
              "readiness_status": "STRUCTURALLY_VALIDATED_NOT_SCIENTIFICALLY_READY"}
    atomic_json(marker_path, result)
    return result
