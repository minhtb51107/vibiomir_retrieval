"""Phase 10A public-leaderboard calibration submissions.

Builds several one-hypothesis variants of the Phase 9 baseline purely from
already-persisted Phase 4/6/7 artifacts. Nothing here crawls, uses the network,
embeds, reranks, trains, or fabricates relevance labels.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import pyarrow.parquet as pq
import yaml

from src.processing.chunking import Tokenizer
from src.reranking.selection import aggregate_reranked_documents, select_candidates

from .common import (
    atomic_write_json,
    canonical_json_bytes,
    file_sha256,
    write_deterministic_zip,
    write_submission_stream,
)
from .expansion import SourceWindowExpander
from .generator import _ordered_chunks, _ordered_documents
from .validator import (
    canonical_provenance,
    expected_query_ids,
    load_source_documents,
    validate_submission,
)

NAME_PATTERN = re.compile(r"^phase10a_E\d+_[A-Za-z0-9_]+$")
ID_PATTERN = re.compile(r"^E\d+$")

DOCUMENT_KINDS = {"ranked", "pool_best_chunk"}
CHUNK_KINDS = {"ranked", "pool_pure"}

RANKED_DOCUMENT_COLUMNS = ["query_id", "doc_id", "rank"]
POOL_COLUMNS = [
    "query_id",
    "query_text",
    "chunk_id",
    "doc_id",
    "chunk_text",
    "source_url",
    "rerank_score",
]


# --------------------------------------------------------------------------- config


def load_calibration_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("calibration configuration must be a mapping")
    required = {"version", "organizer_schema", "inputs", "output", "sources", "baseline", "experiments"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"calibration configuration missing: {sorted(missing)}")
    if int(config["organizer_schema"]["required_query_count"]) <= 0:
        raise ValueError("required_query_count must be positive")
    sources = config["sources"]
    for group, kinds in (("documents", DOCUMENT_KINDS), ("chunks", CHUNK_KINDS)):
        for name, spec in sources[group].items():
            if spec["kind"] not in kinds:
                raise ValueError(f"unsupported {group} source kind for {name}: {spec['kind']}")
    experiments = list(config["experiments"])
    if not experiments:
        raise ValueError("at least one calibration experiment is required")
    identifiers = [str(row["id"]) for row in experiments]
    names = [str(row["name"]) for row in experiments]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("experiment ids must be unique")
    if len(names) != len(set(names)):
        raise ValueError("experiment names must be unique")
    baseline = config["baseline"]
    if str(baseline["name"]) in names:
        raise ValueError("baseline name must differ from experiment names")
    for row in experiments:
        if not ID_PATTERN.match(str(row["id"])):
            raise ValueError(f"invalid experiment id: {row['id']}")
        if not NAME_PATTERN.match(str(row["name"])):
            raise ValueError(f"invalid experiment name: {row['name']}")
        if not str(row["name"]).startswith(f"phase10a_{row['id']}_"):
            raise ValueError(f"experiment name {row['name']} must embed id {row['id']}")
        for field in ("hypothesis", "difference_from_baseline"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"experiment {row['id']} missing {field}")
    for row in [baseline, *experiments]:
        _validate_selection(row, sources)
    return config


def _validate_selection(row: dict[str, Any], sources: dict[str, Any]) -> None:
    for group in ("documents", "chunks"):
        selection = row[group]
        if selection["source"] not in sources[group]:
            raise ValueError(f"unknown {group} source: {selection['source']}")
        if int(selection["depth"]) <= 0:
            raise ValueError(f"{group} depth must be positive")
    expand = row.get("expand")
    if expand is not None and int(expand["target_tokens"]) <= 0:
        raise ValueError("expand.target_tokens must be positive")


# --------------------------------------------------------------------------- inputs


def _group_rows(
    path: str | Path, columns: list[str]
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    table = pq.read_table(path, columns=columns)
    for batch in table.to_batches(max_chunksize=8192):
        for row in batch.to_pylist():
            grouped[int(row["query_id"])].append(row)
    return dict(grouped)


class _SelectionProvider:
    """Serve per-query document or chunk rows for one configured source."""

    def __init__(self, group: str, spec: dict[str, Any]) -> None:
        self.group = group
        self.spec = spec
        self.kind = str(spec["kind"])
        self.path = Path(spec["path"])
        if self.kind == "ranked" and group == "documents":
            columns = RANKED_DOCUMENT_COLUMNS
        elif self.kind == "ranked":
            columns = ["query_id", "chunk_id", "doc_id", "chunk_text", str(spec["rank_column"])]
        else:
            columns = POOL_COLUMNS
        self.rows = _group_rows(self.path, columns)
        self.pool_depth = int(spec.get("pool_depth", 100))

    def _pool_selection(self, query_id: int) -> list[dict[str, Any]]:
        selected, _ = select_candidates(self.rows.get(query_id, []), top_k=self.pool_depth)
        return selected

    def documents(self, query_id: int, depth: int) -> list[dict[str, Any]]:
        if self.kind == "ranked":
            return self.rows.get(query_id, [])
        return aggregate_reranked_documents(
            self._pool_selection(query_id), method="best_chunk", top_k_docs=depth
        )

    def chunks(self, query_id: int) -> list[dict[str, Any]]:
        if self.kind == "ranked":
            return self.rows.get(query_id, [])
        return self._pool_selection(query_id)


class _Inputs:
    """Query IDs, canonical chunk provenance, and optional processed documents."""

    def __init__(self, config: dict[str, Any]) -> None:
        inputs = config["inputs"]
        self.query_ids = _query_ids(
            inputs["queries"], int(config["organizer_schema"]["required_query_count"])
        )
        table = pq.read_table(
            inputs["canonical_chunks"],
            columns=["chunk_id", "doc_id", "raw_text", "start_offset", "end_offset"],
        )
        self.canonical: dict[str, tuple[int, str]] = {}
        self.offsets: dict[str, tuple[int, int]] = {}
        self.document_ids: set[int] = set()
        for row in table.to_pylist():
            chunk_id = str(row["chunk_id"])
            if chunk_id in self.canonical:
                raise ValueError(f"duplicate canonical chunk_id: {chunk_id}")
            self.canonical[chunk_id] = (int(row["doc_id"]), str(row["raw_text"]))
            self.offsets[chunk_id] = (int(row["start_offset"]), int(row["end_offset"]))
            self.document_ids.add(int(row["doc_id"]))
        self._source_documents: dict[int, str] | None = None
        self._source_path = inputs.get("source_documents")

    def source_documents(self) -> dict[int, str]:
        if self._source_documents is None:
            if not self._source_path:
                raise ValueError("inputs.source_documents is required for expansion")
            self._source_documents = load_source_documents(
                self._source_path, self.document_ids
            )
            missing = self.document_ids - set(self._source_documents)
            if missing:
                raise ValueError(f"processed source missing docs: {sorted(missing)[:10]}")
        return self._source_documents


def _query_ids(path: str | Path, required_count: int) -> list[int]:
    ids = expected_query_ids(path)
    if len(ids) != required_count:
        raise ValueError(f"query input has {len(ids)} rows; expected {required_count}")
    if len(ids) != len(set(ids)):
        raise ValueError("query input contains duplicate IDs")
    return ids


def _default_tokenizer_factory(config: dict[str, Any]) -> Callable[[], Tokenizer]:
    def factory() -> Tokenizer:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer

        from src.indexing.tokenizer_validation import HuggingFaceOffsetTokenizer

        spec = config["inputs"]["tokenizer"]
        tokenizer = AutoTokenizer.from_pretrained(
            spec["name"], use_fast=True, local_files_only=True
        )
        return HuggingFaceOffsetTokenizer(tokenizer)

    return factory


# --------------------------------------------------------------------------- stats


def _per_query_summary(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values, default=0),
        "median": statistics.median(values) if values else 0,
        "max": max(values, default=0),
        "mean": round(sum(values) / max(len(values), 1), 4),
        "total": sum(values),
    }


def _relation(experiment: list[Any], baseline: list[Any]) -> str:
    if experiment == baseline:
        return "identical"
    if baseline == experiment[: len(baseline)]:
        return "baseline_is_prefix_of_experiment"
    if experiment == baseline[: len(experiment)]:
        return "experiment_is_prefix_of_baseline"
    return "diverges"


class _Collector:
    def __init__(self) -> None:
        self.document_counts: list[int] = []
        self.chunk_counts: list[int] = []
        self.relevant_docs: set[int] = set()
        self.chunk_docs: set[int] = set()
        self.suppression: dict[str, int] = defaultdict(int)
        self.documents_by_query: dict[int, list[int]] = {}
        self.underlying_chunks_by_query: dict[int, list[str]] = {}
        self.expansion_token_counts: list[int] = []
        self.expansion = defaultdict(int)


def _expanded_chunks(
    chunks: list[dict[str, Any]],
    *,
    inputs: _Inputs,
    expander: SourceWindowExpander,
    cache: dict[str, Any],
    token_cache: dict[str, int],
    tokenizer: Tokenizer,
    collector: _Collector,
) -> list[dict[str, Any]]:
    documents = inputs.source_documents()
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        span = cache.get(chunk_id)
        if span is None:
            start, end = inputs.offsets[chunk_id]
            span = expander.expand(
                doc_id=int(chunk["doc_id"]),
                start=start,
                end=end,
                original_text=str(chunk["chunk_text"]),
            )
            cache[chunk_id] = span
            token_cache[chunk_id] = len(tokenizer.spans(span.text))
        # Independent verbatim / containment verification of every emitted span.
        if documents[span.doc_id][span.start : span.end] != span.text:
            collector.expansion["verbatim_failures"] += 1
        if str(chunk["chunk_text"]) not in span.text:
            collector.expansion["original_containment_failures"] += 1
        collector.expansion["emitted_before_collapse"] += 1
        collector.expansion["unchanged_by_expansion"] += int(not span.changed)
        collector.expansion["whole_document_windows"] += int(span.covers_whole_document)
        key = (span.doc_id, span.text)
        if key in seen:
            collector.expansion["collapsed_duplicate_windows"] += 1
            continue
        seen.add(key)
        collector.expansion_token_counts.append(token_cache[chunk_id])
        output.append(
            {
                "doc_id": span.doc_id,
                "chunk_text": span.text,
                "chunk_id": chunk_id,
            }
        )
    return output


def _record_iterator(
    *,
    inputs: _Inputs,
    selection: dict[str, Any],
    document_provider: _SelectionProvider,
    chunk_provider: _SelectionProvider,
    collector: _Collector,
    expander: SourceWindowExpander | None,
    tokenizer: Tokenizer | None,
) -> Iterator[dict[str, Any]]:
    document_depth = int(selection["documents"]["depth"])
    chunk_depth = int(selection["chunks"]["depth"])
    cache: dict[str, Any] = {}
    token_cache: dict[str, int] = {}
    invalid_documents: set[int] = set()
    for query_id in inputs.query_ids:
        documents = _ordered_documents(
            document_provider.documents(query_id, document_depth), document_depth
        )
        chunks, stats = _ordered_chunks(
            chunk_provider.chunks(query_id),
            depth=chunk_depth,
            canonical=inputs.canonical,
            with_chunk_id=True,
        )
        collector.underlying_chunks_by_query[query_id] = [row["chunk_id"] for row in chunks]
        if expander is not None and tokenizer is not None:
            chunks = _expanded_chunks(
                chunks,
                inputs=inputs,
                expander=expander,
                cache=cache,
                token_cache=token_cache,
                tokenizer=tokenizer,
                collector=collector,
            )
        invalid_documents.update(d for d in documents if d not in inputs.document_ids)
        invalid_documents.update(
            int(row["doc_id"]) for row in chunks if int(row["doc_id"]) not in inputs.document_ids
        )
        if invalid_documents:
            raise ValueError(f"invalid local document IDs: {sorted(invalid_documents)[:10]}")
        for key, value in stats.items():
            collector.suppression[key] += value
        collector.document_counts.append(len(documents))
        collector.chunk_counts.append(len(chunks))
        collector.relevant_docs.update(documents)
        collector.chunk_docs.update(int(row["doc_id"]) for row in chunks)
        collector.documents_by_query[query_id] = documents
        yield {
            "id": query_id,
            "relevant_docs": documents,
            "relevant_chunks": [
                {"doc_id": int(row["doc_id"]), "chunk_text": row["chunk_text"]}
                for row in chunks
            ],
        }


def _open_providers(
    config: dict[str, Any], selection: dict[str, Any]
) -> tuple[_SelectionProvider, _SelectionProvider]:
    documents = _SelectionProvider(
        "documents", config["sources"]["documents"][selection["documents"]["source"]]
    )
    chunks = _SelectionProvider(
        "chunks", config["sources"]["chunks"][selection["chunks"]["source"]]
    )
    return documents, chunks


def _expansion_summary(collector: _Collector, target: int) -> dict[str, Any]:
    counts = collector.expansion_token_counts
    emitted = collector.expansion["emitted_before_collapse"]
    return {
        "target_tokens": target,
        "tokenizer": "BAAI/bge-m3 (local cache, add_special_tokens=False)",
        "emitted_before_duplicate_collapse": emitted,
        "unchanged_by_expansion": collector.expansion["unchanged_by_expansion"],
        "expanded_beyond_original": emitted - collector.expansion["unchanged_by_expansion"],
        "whole_document_windows": collector.expansion["whole_document_windows"],
        "collapsed_duplicate_windows": collector.expansion["collapsed_duplicate_windows"],
        "emitted_window_tokens": {
            "min": min(counts, default=0),
            "median": statistics.median(counts) if counts else 0,
            "max": max(counts, default=0),
            "mean": round(sum(counts) / max(len(counts), 1), 2),
        },
        "verbatim_failures": collector.expansion["verbatim_failures"],
        "original_containment_failures": collector.expansion["original_containment_failures"],
    }


class _HashingStream:
    """Hash the exact bytes `write_submission_stream` would emit (no file)."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def consume(self, records: Iterator[dict[str, Any]]) -> str:
        self.digest.update(b"[")
        first = True
        for record in records:
            if not first:
                self.digest.update(b",")
            self.digest.update(canonical_json_bytes(record, trailing_newline=False))
            first = False
        self.digest.update(b"]\n")
        return self.digest.hexdigest()


# --------------------------------------------------------------------------- generation


def _baseline_signature(
    config: dict[str, Any],
    inputs: _Inputs,
    *,
    phase9_manifest_path: str | Path | None,
) -> dict[str, Any]:
    baseline = config["baseline"]
    documents, chunks = _open_providers(config, baseline)
    collector = _Collector()
    digest = _HashingStream().consume(
        _record_iterator(
            inputs=inputs,
            selection=baseline,
            document_provider=documents,
            chunk_provider=chunks,
            collector=collector,
            expander=None,
            tokenizer=None,
        )
    )
    reference = None
    if phase9_manifest_path and Path(phase9_manifest_path).exists():
        manifest = json.loads(Path(phase9_manifest_path).read_text(encoding="utf-8"))
        for row in manifest.get("variants", []):
            if row.get("name") == baseline.get("phase9_variant"):
                reference = row.get("json_sha256")
    return {
        "name": str(baseline["name"]),
        "documents_per_query": _per_query_summary(collector.document_counts),
        "chunks_per_query": _per_query_summary(collector.chunk_counts),
        "reproduced_json_sha256": digest,
        "phase9_recorded_json_sha256": reference,
        "reproduces_phase9_json": reference is not None and reference == digest,
        "_documents": collector.documents_by_query,
        "_chunks": collector.underlying_chunks_by_query,
    }


def _build_experiment(
    config: dict[str, Any],
    experiment: dict[str, Any],
    inputs: _Inputs,
    baseline: dict[str, Any],
    output: Path,
    tokenizer_factory: Callable[[], Tokenizer] | None,
    discard_outputs: bool,
) -> dict[str, Any]:
    documents, chunks = _open_providers(config, experiment)
    collector = _Collector()
    expand = experiment.get("expand")
    expander = tokenizer = None
    if expand is not None:
        tokenizer = (tokenizer_factory or _default_tokenizer_factory(config))()
        expander = SourceWindowExpander(
            inputs.source_documents(), tokenizer, int(expand["target_tokens"])
        )
    json_path = output / f"{experiment['name']}.json"
    zip_path = output / f"{experiment['name']}.zip"
    write_submission_stream(
        json_path,
        _record_iterator(
            inputs=inputs,
            selection=experiment,
            document_provider=documents,
            chunk_provider=chunks,
            collector=collector,
            expander=expander,
            tokenizer=tokenizer,
        ),
    )
    write_deterministic_zip(json_path, zip_path)
    document_relations: dict[str, int] = defaultdict(int)
    chunk_relations: dict[str, int] = defaultdict(int)
    for query_id in inputs.query_ids:
        document_relations[
            _relation(collector.documents_by_query[query_id], baseline["_documents"][query_id])
        ] += 1
        chunk_relations[
            _relation(
                collector.underlying_chunks_by_query[query_id],
                baseline["_chunks"][query_id],
            )
        ] += 1
    unique_all = collector.relevant_docs | collector.chunk_docs
    entry = {
        "id": str(experiment["id"]),
        "name": str(experiment["name"]),
        "json_filename": json_path.name,
        "zip_filename": zip_path.name,
        "json_sha256": file_sha256(json_path),
        "zip_sha256": file_sha256(zip_path),
        "json_bytes": json_path.stat().st_size,
        "zip_bytes": zip_path.stat().st_size,
        "hypothesis": str(experiment["hypothesis"]).strip(),
        "difference_from_baseline": str(experiment["difference_from_baseline"]).strip(),
        "configuration": {
            "documents_source": experiment["documents"]["source"],
            "documents_depth_cap": int(experiment["documents"]["depth"]),
            "chunks_source": experiment["chunks"]["source"],
            "chunks_depth_cap": int(experiment["chunks"]["depth"]),
            "expand": dict(expand) if expand else None,
        },
        "query_count": len(inputs.query_ids),
        "documents_per_query": _per_query_summary(collector.document_counts),
        "chunks_per_query": _per_query_summary(collector.chunk_counts),
        "unique_doc_ids_overall": len(unique_all),
        "unique_relevant_doc_ids": len(collector.relevant_docs),
        "unique_chunk_doc_ids": len(collector.chunk_docs),
        "invalid_doc_ids": 0,
        "duplicate_doc_ids": 0,
        "chunk_provenance": {
            "chunk_provenance_mismatches": 0,
            "source_rows_verified_against_canonical_chunk_id": collector.suppression[
                "source_rows_provenance_verified"
            ],
            "duplicate_chunk_ids_suppressed": collector.suppression[
                "duplicate_chunk_ids_suppressed"
            ],
            "duplicate_chunk_objects_suppressed": collector.suppression[
                "duplicate_chunk_objects_suppressed"
            ],
        },
        "expansion": _expansion_summary(collector, int(expand["target_tokens"]))
        if expand
        else None,
        "versus_baseline": {
            "documents_per_query_relation": dict(document_relations),
            "underlying_chunks_per_query_relation": dict(chunk_relations),
        },
        "source_artifacts": {
            "documents": str(documents.path),
            "chunks": str(chunks.path),
        },
    }
    if discard_outputs:
        json_path.unlink()
        zip_path.unlink()
    return entry


def _input_hashes(config: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in ("queries", "canonical_chunks", "source_documents"):
        path = config["inputs"].get(key)
        if path:
            hashes[f"{key}_sha256"] = file_sha256(path)
    seen: dict[str, str] = {}
    for group in ("documents", "chunks"):
        for name, spec in config["sources"][group].items():
            seen[f"{group}.{name}"] = str(spec["path"])
    for label, path in sorted(seen.items()):
        hashes[f"source[{label}]_sha256"] = file_sha256(path)
    return hashes


def generate_calibration_experiments(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    output_directory: str | Path | None = None,
    tokenizer_factory: Callable[[], Tokenizer] | None = None,
    discard_outputs: bool = False,
    phase9_manifest_path: str | Path | None = None,
    only: list[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_directory or config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    inputs = _Inputs(config)
    baseline = _baseline_signature(config, inputs, phase9_manifest_path=phase9_manifest_path)
    entries = []
    for experiment in config["experiments"]:
        if only and str(experiment["id"]) not in only:
            continue
        entries.append(
            _build_experiment(
                config,
                experiment,
                inputs,
                baseline,
                output,
                tokenizer_factory,
                discard_outputs,
            )
        )
    baseline_public = {k: v for k, v in baseline.items() if not k.startswith("_")}
    baseline_public["external_organizer_score"] = config["baseline"].get("organizer_score")
    return {
        "version": str(config["version"]),
        "generator_config_sha256": file_sha256(config_path),
        "quality_statement": (
            "QUALITY UNKNOWN UNTIL ORGANIZER LEADERBOARD RESULTS. No local relevance "
            "score was computed; no quality claim is made."
        ),
        "corpus": {
            "canonical_chunk_count": len(inputs.canonical),
            "canonical_document_count": len(inputs.document_ids),
        },
        "input_hashes": _input_hashes(config),
        "baseline": baseline_public,
        "experiments": entries,
    }


# --------------------------------------------------------------------------- validation


def validate_calibration_outputs(
    config: dict[str, Any], manifest: dict[str, Any], directory: str | Path
) -> list[dict[str, Any]]:
    directory = Path(directory)
    queries = expected_query_ids(config["inputs"]["queries"])
    valid_documents, valid_chunks = canonical_provenance(config["inputs"]["canonical_chunks"])
    source_documents = None
    results = []
    for entry in manifest["experiments"]:
        expanded = entry["expansion"] is not None
        if expanded and source_documents is None:
            source_documents = load_source_documents(
                config["inputs"]["source_documents"], valid_documents
            )
        zip_path = directory / entry["zip_filename"]
        result = validate_submission(
            zip_path,
            expected_queries=queries,
            valid_document_ids=valid_documents,
            valid_chunks=valid_chunks,
            require_exact_order=True,
            source_documents=source_documents if expanded else None,
        )
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            member_digest = hashlib.sha256()
            with archive.open(names[0]) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    member_digest.update(block)
        results.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "validator": (
                    "strict canonical-chunk validator + contiguous source-span check"
                    if expanded
                    else "strict canonical-chunk validator"
                ),
                "valid": bool(result["valid"]),
                "query_count": result["query_count"],
                "invalid_doc_ids": result["invalid_doc_ids"],
                "duplicate_doc_ids": result["duplicate_doc_ids"],
                "duplicate_chunk_objects": result["duplicate_chunk_objects"],
                "chunk_provenance_mismatches": result["chunk_provenance_mismatches"],
                "source_span_chunks_verified": result["source_span_chunks_verified"],
                "unique_doc_ids": result["unique_doc_ids"],
                "zip_root_files": names,
                "zip_single_root_json": names == [entry["json_filename"]],
                "zip_member_sha256_matches_json": member_digest.hexdigest()
                == entry["json_sha256"],
            }
        )
    return results


def build_calibration(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    verify_determinism: bool = True,
    tokenizer_factory: Callable[[], Tokenizer] | None = None,
    phase9_manifest_path: str | Path | None = None,
    write_manifest: bool = True,
    only: list[str] | None = None,
) -> dict[str, Any]:
    manifest = generate_calibration_experiments(
        config,
        config_path=config_path,
        tokenizer_factory=tokenizer_factory,
        phase9_manifest_path=phase9_manifest_path,
        only=only,
    )
    manifest["validation"] = validate_calibration_outputs(
        config, manifest, config["output"]["directory"]
    )
    determinism: dict[str, Any] = {"verified": False}
    if verify_determinism:
        with tempfile.TemporaryDirectory(prefix="phase10a_determinism_") as temporary:
            second = generate_calibration_experiments(
                config,
                config_path=config_path,
                output_directory=temporary,
                tokenizer_factory=tokenizer_factory,
                discard_outputs=True,
                phase9_manifest_path=phase9_manifest_path,
                only=only,
            )
        first_rows = [(r["name"], r["json_sha256"], r["zip_sha256"]) for r in manifest["experiments"]]
        second_rows = [(r["name"], r["json_sha256"], r["zip_sha256"]) for r in second["experiments"]]
        if first_rows != second_rows:
            raise RuntimeError("determinism verification failed: output hashes differ")
        determinism = {
            "verified": True,
            "byte_identical_json_and_zip": True,
            "experiment_count": len(first_rows),
            "method": "full independent regeneration (fresh source loads, fresh tokenizer)",
        }
    manifest["determinism"] = determinism
    if write_manifest:
        output = Path(config["output"]["directory"])
        artifact_directory = Path(config["output"]["artifact_directory"])
        atomic_write_json(artifact_directory / str(config["output"].get("manifest", "manifest.json")), manifest)
        lines = []
        for entry in manifest["experiments"]:
            lines.append(f"{entry['json_sha256']}  {entry['json_filename']}")
            lines.append(f"{entry['zip_sha256']}  {entry['zip_filename']}")
        (output / "SHA256SUMS_phase10a.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )
    return manifest
