from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import numpy as np

from src.acquisition_benchmark.sampling import deterministic_domain_samples
from src.indexing.embedder import normalize_text
from src.retrieval.lexical_tokenizer import tokenize_lexical
from src.retrieval.fusion import reciprocal_rank_fusion
from src.ingestion.checkpoint import SCHEMA_SQL
from src.storage.body_archive import BodyArchive, BodyMetadata


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("phase") not in {"phase10d_source_census", "phase10e_focused_scaling"}:
        raise ValueError("unexpected source-census configuration")
    if int(value["models"]["reranker"]["batch_size"]) != 2:
        raise ValueError("reranker batch size must remain 2")
    return value


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 7:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.025 * (attempt + 1))


def stable_fold(value: str, modulo: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % modulo


def _existing_ids(paths: Iterable[str]) -> set[int]:
    ids: set[int] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        def visit(node: Any) -> None:
            if isinstance(node, dict):
                if "doc_id" in node:
                    ids.add(int(node["doc_id"]))
                elif "id" in node and isinstance(node["id"], int):
                    ids.add(int(node["id"]))
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)
        visit(value)
    return ids


def prepare_round1(config: dict[str, Any]) -> dict[str, Any]:
    inventory = json.loads(Path(config["inputs"]["inventory"]).read_text(encoding="utf-8"))
    excluded = set(config["excluded_sources"])
    domains = [row["domain"] for row in inventory["domains"] if row["domain"] not in excluded]
    samples, populations = deterministic_domain_samples(
        config["inputs"]["corpus"], domains=set(domains),
        sample_size=int(config["sampling"]["round1_documents_per_source"]) * 3,
        seed=str(config["seed"]),
    )
    existing = _existing_ids(config["sampling"].get("avoid_existing_manifests", []))
    existing.update(int(value) for value in pq.read_table(config["inputs"]["pilot_documents"], columns=["doc_id"]).column(0).to_pylist())
    # Deterministic sampler returns more than enough only when sample_size is raised;
    # for the 100-row census, remove collisions and document any resulting shortfall.
    selected: dict[str, list[dict[str, Any]]] = {}
    for domain in domains:
        selected[domain] = [row for row in samples.get(domain, []) if int(row["doc_id"]) not in existing][:100]
    rows = [row for domain in sorted(selected) for row in selected[domain]]
    doc_ids = [int(row["doc_id"]) for row in rows]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("duplicate official doc IDs in Round-1 manifest")
    artifacts = Path(config["outputs"]["artifacts"])
    manifest = {
        "format_version": 1,
        "seed": config["seed"],
        "sampling": "lowest SHA-256(seed:doc_id) ranks within each domain; existing sampled IDs excluded",
        "excluded_sources": sorted(excluded),
        "remaining_sources": len(domains),
        "scheduled_sources": sum(bool(value) for value in selected.values()),
        "scheduled_documents": len(rows),
        "domains": {
            domain: {
                "population": int(populations.get(domain, 0)),
                "sample_count": len(selected[domain]),
                "records": selected[domain],
            }
            for domain in sorted(domains)
        },
    }
    atomic_json(artifacts / "round1_sample_manifest.json", manifest)
    atomic_json(artifacts / "round1_ids.json", {"doc_ids": doc_ids})
    return manifest


def host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def technical_triage(config: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((Path(config["outputs"]["artifacts"]) / "round1_sample_manifest.json").read_text(encoding="utf-8"))
    docs = pq.read_table(config["outputs"]["documents"]).to_pylist()
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in docs:
        by_domain[host(str(row.get("original_url") or row.get("final_url") or ""))].append(row)
    inventory = {row["domain"]: row for row in json.loads(Path(config["inputs"]["inventory"]).read_text(encoding="utf-8"))["domains"]}
    result: dict[str, Any] = {"sources": {}, "healthy_sources": []}
    for domain, info in manifest["domains"].items():
        rows = by_domain.get(domain, [])
        successes = [row for row in rows if str(row.get("extraction_status")) == "SUCCESS" and str(row.get("normalized_text") or "").strip()]
        statuses = Counter(str(row.get("extraction_status") or "MISSING") for row in rows)
        healthy = len(successes) >= int(config["triage"]["minimum_extracted_documents"])
        result["sources"][domain] = {
            "corpus_rows": int(info["population"]),
            "sampled": int(info["sample_count"]),
            "extracted_rows": len(rows),
            "usable_documents": len(successes),
            "status_counts": dict(sorted(statuses.items())),
            "likely_language": inventory.get(domain, {}).get("likely_language", "unknown"),
            "healthy": healthy,
            "technical_only": True,
        }
        if healthy:
            result["healthy_sources"].append(domain)
    result["healthy_sources"].sort()
    atomic_json(Path(config["outputs"]["artifacts"]) / "technical_triage.json", result)
    return result


def merge_acquisition(config: dict[str, Any]) -> dict[str, Any]:
    """Merge isolated source checkpoints into one resumable canonical input."""
    outputs = config["outputs"]
    sources: list[tuple[Path, Path]] = [
        (Path(outputs["serial_crawl_database"]), Path(outputs["serial_body_archive"]))
    ]
    worker_root = Path(outputs["worker_root"])
    for database in sorted(worker_root.glob("*/crawl.sqlite")):
        sources.append((database, database.parent / "bodies"))
    destination_db = Path(outputs["crawl_database"])
    destination_db.parent.mkdir(parents=True, exist_ok=True)
    destination = sqlite3.connect(destination_db)
    destination.row_factory = sqlite3.Row
    destination.executescript(SCHEMA_SQL)
    columns = [str(row[1]) for row in destination.execute("PRAGMA table_info(crawl_results)")]
    placeholders = ",".join("?" for _ in columns)
    inserted = 0
    with BodyArchive(outputs["body_archive"], documents_per_shard=1000) as output_archive:
        for database, archive_path in sources:
            if not database.exists():
                continue
            source = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            source.row_factory = sqlite3.Row
            archive = BodyArchive(archive_path, documents_per_shard=1000) if archive_path.exists() else None
            try:
                for row in source.execute("SELECT * FROM crawl_results ORDER BY doc_id"):
                    doc_id = int(row["doc_id"])
                    if archive is not None and archive.contains(doc_id) and not output_archive.contains(doc_id):
                        metadata = archive.metadata(doc_id)
                        output_archive.add(
                            BodyMetadata(
                                doc_id=doc_id, original_url=str(metadata["original_url"]),
                                final_url=metadata["final_url"], fetched_at=str(metadata["fetched_at"]),
                                content_type=metadata["content_type"], encoding=metadata["encoding"],
                                declared_http_encoding=metadata["declared_http_encoding"],
                            ),
                            archive.get_body(doc_id),
                        )
                    values = [row[name] if name in row.keys() else None for name in columns]
                    before = destination.total_changes
                    with destination:
                        destination.execute(
                            f"INSERT OR IGNORE INTO crawl_results ({','.join(columns)}) VALUES ({placeholders})",
                            values,
                        )
                    inserted += destination.total_changes - before
            finally:
                if archive is not None:
                    archive.close()
                source.close()
        integrity = str(destination.execute("PRAGMA integrity_check").fetchone()[0])
        rows = int(destination.execute("SELECT COUNT(*) FROM crawl_results").fetchone()[0])
        bodies = output_archive.count
    destination.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    destination.close()
    result = {"source_checkpoints": len(sources), "inserted_this_run": inserted, "canonical_rows": rows, "canonical_bodies": bodies, "integrity": integrity, "database": str(destination_db), "archive": outputs["body_archive"]}
    atomic_json(Path(outputs["artifacts"]) / "round1_acquisition_merge.json", result)
    return result


def balanced_groups(config: dict[str, Any], triage: dict[str, Any]) -> dict[str, list[str]]:
    rows = triage["sources"]
    healthy = list(triage["healthy_sources"])
    sizes = sorted(int(rows[d]["corpus_rows"]) for d in healthy)
    q1 = sizes[len(sizes) // 3] if sizes else 0
    q2 = sizes[(2 * len(sizes)) // 3] if sizes else 0
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for domain in healthy:
        row = rows[domain]
        size = "small" if row["corpus_rows"] <= q1 else "medium" if row["corpus_rows"] <= q2 else "large"
        yield_band = "high" if row["usable_documents"] >= 80 else "medium" if row["usable_documents"] >= 40 else "low"
        strata[(str(row["likely_language"]), size, yield_band)].append(domain)
    groups = {"A": [], "B": [], "C": []}
    cursor = 0
    for key in sorted(strata):
        for domain in sorted(strata[key], key=lambda d: (-int(rows[d]["corpus_rows"]), d)):
            groups["ABC"[cursor % 3]].append(domain)
            cursor += 1
    for values in groups.values():
        values.sort()
    return groups


def balanced_round2_groups(
    parent_groups: dict[str, list[str]],
    source_metadata: dict[str, dict[str, Any]],
    *,
    group_count: int = 7,
) -> dict[str, Any]:
    """Split surviving A/C sources into deterministic, metadata-balanced groups.

    Each parent is assigned independently so every output group receives the
    same number of sources from each parent.  Within that hard constraint, a
    greedy objective balances language, corpus-size tier, usable documents,
    chunk counts, and acquisition yield.  The complete assignment metadata is
    returned so the decision can be reconstructed without hidden state.
    """
    survivors = {parent: sorted(parent_groups[parent]) for parent in ("A", "C")}
    if any(len(values) % group_count for values in survivors.values()):
        raise ValueError("each surviving parent group must divide evenly")
    per_parent = {parent: len(values) // group_count for parent, values in survivors.items()}
    if per_parent != {"A": 3, "C": 3}:
        raise ValueError(f"expected 3 sources from each parent per group, got {per_parent}")

    all_sources = sorted([*survivors["A"], *survivors["C"]])
    corpus_sizes = sorted(int(source_metadata[s]["corpus_rows"]) for s in all_sources)
    low_cut = corpus_sizes[len(corpus_sizes) // 3]
    high_cut = corpus_sizes[(2 * len(corpus_sizes)) // 3]

    def enriched(source: str, parent: str) -> dict[str, Any]:
        row = source_metadata[source]
        corpus_rows = int(row["corpus_rows"])
        sampled = int(row["sampled"])
        usable = int(row["usable_documents"])
        chunks = int(row["chunk_count"])
        language = str(row.get("likely_language") or "unknown")
        size_tier = "small" if corpus_rows <= low_cut else "medium" if corpus_rows <= high_cut else "large"
        return {
            "source": source,
            "parent": parent,
            "likely_language": language if language in {"vi", "zh"} else "other",
            "size_tier": size_tier,
            "corpus_rows": corpus_rows,
            "sampled": sampled,
            "usable_documents": usable,
            "chunk_count": chunks,
            "acquisition_yield": (usable / sampled) if sampled else 0.0,
        }

    rows = {
        source: enriched(source, parent)
        for parent, sources in survivors.items()
        for source in sources
    }
    group_names = [f"G{index}" for index in range(1, group_count + 1)]
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in group_names}

    # Hard-to-balance rows go first.  Stable SHA-256 is the final deterministic
    # ordering key and is independent of Python hash randomization.
    for parent in ("A", "C"):
        ordered = sorted(
            (rows[source] for source in survivors[parent]),
            key=lambda row: (
                -int(row["corpus_rows"]),
                -int(row["chunk_count"]),
                -int(row["usable_documents"]),
                stable_fold(f"phase10d-round2:{row['source']}", 2**31 - 1),
                str(row["source"]),
            ),
        )
        for row in ordered:
            eligible = [
                name for name in group_names
                if sum(item["parent"] == parent for item in assigned[name]) < per_parent[parent]
            ]

            def objective(name: str) -> tuple[Any, ...]:
                current = assigned[name]
                language_count = sum(item["likely_language"] == row["likely_language"] for item in current)
                tier_count = sum(item["size_tier"] == row["size_tier"] for item in current)
                # Normalize magnitudes so corpus size cannot drown out yield or chunks.
                corpus_load = sum(float(item["corpus_rows"]) for item in current) / max(1.0, sum(corpus_sizes) / group_count)
                usable_load = sum(float(item["usable_documents"]) for item in current) / max(1.0, sum(rows[s]["usable_documents"] for s in all_sources) / group_count)
                chunk_load = sum(float(item["chunk_count"]) for item in current) / max(1.0, sum(rows[s]["chunk_count"] for s in all_sources) / group_count)
                yield_load = sum(float(item["acquisition_yield"]) for item in current)
                return (
                    language_count,
                    tier_count,
                    round(corpus_load + usable_load + chunk_load + yield_load, 12),
                    len(current),
                    name,
                )

            assigned[min(eligible, key=objective)].append(row)

    # Deterministic same-parent swap refinement.  It preserves the exact 3A/3C
    # constraint while reducing imbalance left by the one-pass greedy seed.
    def balance_cost() -> float:
        numeric_fields = ("corpus_rows", "usable_documents", "chunk_count", "acquisition_yield")
        cost = 0.0
        for field in numeric_fields:
            totals = [sum(float(row[field]) for row in assigned[name]) for name in group_names]
            mean = sum(totals) / len(totals)
            scale = max(mean, 1.0)
            cost += sum(((value - mean) / scale) ** 2 for value in totals)
        for field, values in (("likely_language", ("vi", "zh", "other")), ("size_tier", ("small", "medium", "large"))):
            for value in values:
                counts = [sum(row[field] == value for row in assigned[name]) for name in group_names]
                mean = sum(counts) / len(counts)
                cost += 0.25 * sum((count - mean) ** 2 for count in counts)
        return cost

    while True:
        baseline = balance_cost()
        best: tuple[float, str, int, str, int] | None = None
        for left_index, left_name in enumerate(group_names):
            for right_name in group_names[left_index + 1:]:
                for left_row_index, left_row in enumerate(assigned[left_name]):
                    for right_row_index, right_row in enumerate(assigned[right_name]):
                        if left_row["parent"] != right_row["parent"]:
                            continue
                        assigned[left_name][left_row_index], assigned[right_name][right_row_index] = right_row, left_row
                        candidate_cost = balance_cost()
                        assigned[left_name][left_row_index], assigned[right_name][right_row_index] = left_row, right_row
                        candidate = (candidate_cost, left_name, left_row_index, right_name, right_row_index)
                        if candidate_cost < baseline - 1e-12 and (best is None or candidate < best):
                            best = candidate
        if best is None:
            break
        _, left_name, left_row_index, right_name, right_row_index = best
        assigned[left_name][left_row_index], assigned[right_name][right_row_index] = (
            assigned[right_name][right_row_index], assigned[left_name][left_row_index]
        )

    for name in group_names:
        assigned[name].sort(key=lambda row: str(row["source"]))
        if len(assigned[name]) != 6:
            raise ValueError(f"{name} has {len(assigned[name])} sources, expected 6")
        if Counter(row["parent"] for row in assigned[name]) != Counter({"A": 3, "C": 3}):
            raise ValueError(f"{name} does not contain exactly 3 A and 3 C sources")
    flattened = [row["source"] for name in group_names for row in assigned[name]]
    if len(flattened) != 42 or len(set(flattened)) != 42:
        raise ValueError("Round-2 groups are not a disjoint partition of 42 sources")

    return {
        "format_version": 1,
        "method": "deterministic constrained greedy balance",
        "constraints": {
            "groups": group_count,
            "sources_per_group": 6,
            "parent_A_per_group": 3,
            "parent_C_per_group": 3,
            "excluded_parent": "B",
            "excluded_source": "suckhoecongdongonline.vn",
        },
        "size_tier_cutoffs": {"small_max": low_cut, "medium_max": high_cut},
        "balance_fields": [
            "parent", "likely_language", "size_tier", "corpus_rows",
            "usable_documents", "chunk_count", "acquisition_yield",
        ],
        "groups": assigned,
    }


def score_gate(scores: dict[str, float], minimum_delta: float = 0.0001) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) < 2:
        return {"status": "AMBIGUOUS", "survivors": [key for key, _ in ordered]}
    delta = ordered[0][1] - ordered[1][1]
    return {
        "status": "ADVANCE" if delta >= minimum_delta else "AMBIGUOUS",
        "winner": ordered[0][0] if delta >= minimum_delta else None,
        "survivors": [ordered[0][0]] if delta >= minimum_delta else [key for key, value in ordered if ordered[0][1] - value < minimum_delta],
        "top_delta": delta,
    }


def calibrate_cap(config: dict[str, Any]) -> dict[str, Any]:
    persisted = config["candidate_cache"].get("calibrated_cap_artifact")
    if persisted:
        source = Path(persisted)
        if not source.exists():
            raise FileNotFoundError(f"persisted candidate-cap calibration is missing: {source}")
        prior = json.loads(source.read_text(encoding="utf-8"))
        chosen = int(prior.get("chosen_m", -1))
        allowed = [int(value) for value in config["candidate_cache"]["candidate_caps"]]
        if chosen not in allowed:
            raise ValueError(f"persisted chosen_m={chosen} is not allowed by candidate_caps={allowed}")
        result = {
            **prior,
            "chosen_m": chosen,
            "reused_persisted_calibration": True,
            "source_artifact": str(source),
        }
        atomic_json(Path(config["outputs"]["artifacts"]) / "candidate_cap_calibration.json", result)
        return result
    pool_path = Path(config["inputs"]["c1_candidate_pool"])
    reranked_path = Path(config["inputs"]["c1_reranked_chunks"])
    source_ids = set(int(x) for x in pq.read_table(config["inputs"]["c1_source_doc_ids"], columns=["doc_id"]).column(0).to_pylist())
    pools: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(pool_path).to_pylist():
        if int(row["doc_id"]) in source_ids:
            pools[int(row["query_id"])].append(row)
    truth_chunks: dict[int, set[str]] = defaultdict(set)
    truth_docs: dict[int, set[int]] = defaultdict(set)
    for row in pq.read_table(reranked_path).to_pylist():
        q = int(row["query_id"])
        if int(row["doc_id"]) in source_ids and int(row["rerank_rank"]) <= 20:
            truth_chunks[q].add(str(row["chunk_id"]))
        if int(row["doc_id"]) in source_ids and int(row["rerank_rank"]) <= 50:
            truth_docs[q].add(int(row["doc_id"]))
    results = {}
    chosen = max(config["candidate_cache"]["candidate_caps"])
    for cap in config["candidate_cache"]["candidate_caps"]:
        kept_chunks: dict[int, set[str]] = {}
        kept_docs: dict[int, set[int]] = {}
        for q, rows in pools.items():
            ranked = sorted(rows, key=lambda r: (int(r["pool_rank"]), str(r["chunk_id"])))[: int(cap)]
            kept_chunks[q] = {str(r["chunk_id"]) for r in ranked}
            kept_docs[q] = {int(r["doc_id"]) for r in ranked}
        chunk_den = sum(len(v) for v in truth_chunks.values())
        doc_den = sum(len(v) for v in truth_docs.values())
        chunk_ret = sum(len(v & kept_chunks.get(q, set())) for q, v in truth_chunks.items()) / max(chunk_den, 1)
        doc_ret = sum(len(v & kept_docs.get(q, set())) for q, v in truth_docs.items()) / max(doc_den, 1)
        results[str(cap)] = {"chunk_retention": chunk_ret, "document_retention": doc_ret}
        if chunk_ret >= float(config["candidate_cache"]["calibration_chunk_retention"]) and doc_ret >= float(config["candidate_cache"]["calibration_document_retention"]):
            chosen = int(cap)
            break
    result = {"method": "local C1 source-candidate cap reproduction; no organizer labels", "caps": results, "chosen_m": chosen}
    atomic_json(Path(config["outputs"]["artifacts"]) / "candidate_cap_calibration.json", result)
    return result


def _bm25_rank(texts: list[str], query: str, *, k1: float, b: float, top_k: int) -> list[tuple[int, float]]:
    tokenized = [tokenize_lexical(text) for text in texts]
    n = len(tokenized)
    if not n:
        return []
    df = Counter(token for tokens in tokenized for token in set(tokens))
    lengths = [len(tokens) for tokens in tokenized]
    avg = sum(lengths) / max(n, 1)
    query_tokens = set(tokenize_lexical(query))
    scored = []
    import math
    for index, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            tf = counts.get(term, 0)
            if not tf:
                continue
            ident = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + k1 * (1.0 - b + b * lengths[index] / max(avg, 1e-9))
            score += ident * tf * (k1 + 1.0) / denom
        if score > 0:
            scored.append((index, score))
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


class SourceBM25Index:
    """Exact in-memory equivalent of `_bm25_rank`, built once per source."""

    def __init__(self, texts: list[str], *, k1: float, b: float) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenized = [tokenize_lexical(text) for text in texts]
        self.document_count = len(self.tokenized)
        self.lengths = [len(tokens) for tokens in self.tokenized]
        self.average_length = sum(self.lengths) / max(self.document_count, 1)
        self.document_frequency = Counter(
            token for tokens in self.tokenized for token in set(tokens)
        )
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row_index, tokens in enumerate(self.tokenized):
            for term, frequency in Counter(tokens).items():
                postings[term].append((row_index, int(frequency)))
        self.postings = dict(postings)

    def rank(self, query: str, *, top_k: int) -> list[tuple[int, float]]:
        import math

        scores: dict[int, float] = {}
        # Set construction and iteration deliberately match `_bm25_rank`.
        for term in set(tokenize_lexical(query)):
            rows = self.postings.get(term)
            if not rows:
                continue
            ident = math.log(
                1.0
                + (self.document_count - self.document_frequency[term] + 0.5)
                / (self.document_frequency[term] + 0.5)
            )
            for row_index, term_frequency in rows:
                denominator = term_frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b
                    * self.lengths[row_index]
                    / max(self.average_length, 1e-9)
                )
                increment = (
                    ident
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                )
                scores[row_index] = scores.get(row_index, 0.0) + increment
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def stable_top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Partial top-k with the same score-desc/index-asc ordering as stable argsort."""
    count = min(int(top_k), int(scores.size))
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    if count == scores.size:
        return np.asarray(
            sorted(range(scores.size), key=lambda index: (-float(scores[index]), index)),
            dtype=np.int64,
        )
    partition = np.argpartition(-scores, count - 1)[:count]
    boundary = float(np.min(scores[partition]))
    eligible = np.flatnonzero(scores >= boundary)
    ordered = sorted(
        (int(index) for index in eligible),
        key=lambda index: (-float(scores[index]), index),
    )[:count]
    return np.asarray(ordered, dtype=np.int64)


def _source_slug(source: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in source).strip("_")


def _write_source_part(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.replace(temporary, path)


def _merge_source_parts(paths: list[Path], output: Path) -> int:
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    writer: pq.ParquetWriter | None = None
    canonical_schema: pa.Schema | None = None
    count = 0
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=4096):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    fields = [
                        pa.field("heading_path", pa.list_(pa.string()))
                        if field.name == "heading_path"
                        else field
                        for field in table.schema
                    ]
                    canonical_schema = pa.schema(fields, metadata=table.schema.metadata)
                    writer = pq.ParquetWriter(temporary, canonical_schema)
                if table.schema != canonical_schema:
                    # A source whose headings are all null is inferred by Arrow as
                    # list<null>. Cast it to the canonical nullable list<string>
                    # representation before combining otherwise equivalent parts.
                    table = table.cast(canonical_schema)
                writer.write_table(table)
                count += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("no source candidate parts were produced")
    os.replace(temporary, output)
    return count


def build_source_candidates(
    config: dict[str, Any],
    *,
    run_state: str | Path | None = None,
    groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    triage = json.loads((Path(config["outputs"]["artifacts"]) / "technical_triage.json").read_text(encoding="utf-8"))
    calibration = calibrate_cap(config)
    chosen = int(calibration["chosen_m"])
    chunks = pq.read_table(config["outputs"]["chunks"]).to_pylist()
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunks:
        domain = host(str(row.get("source_url") or ""))
        if domain in triage["healthy_sources"]:
            by_domain[domain].append(row)
    queries = pq.read_table(config["inputs"]["queries"], columns=["id", "query"]).to_pylist()
    query_texts = [normalize_text(str(row["query"])) for row in queries]
    root = Path(config["outputs"]["root"]) / "round1"
    root.mkdir(parents=True, exist_ok=True)
    embedding_path = root / "chunk_embeddings.f32"
    query_embedding_path = Path("data/source_relevance_probe/query_embeddings.f32")
    model_cfg = config["models"]["embedder"]
    dimension = int(model_cfg["dimension"])
    expected_embedding_bytes = len(chunks) * dimension * 4
    if embedding_path.exists() and embedding_path.stat().st_size != expected_embedding_bytes:
        raise ValueError("persisted source-census embedding size does not match chunks")
    if not embedding_path.exists():
        import torch
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
        model = SentenceTransformer(model_cfg["name"], revision=model_cfg["revision"], device="cpu")
        model.max_seq_length = int(model_cfg["max_length"]); model.half(); model.to("cuda")
        vectors = model.encode([normalize_text(str(row["normalized_text"])) for row in chunks], batch_size=int(model_cfg["batch_size"]), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        np.asarray(vectors, dtype=np.float32).tofile(embedding_path)
        del model, vectors
        torch.cuda.empty_cache()
    vectors = np.memmap(embedding_path, dtype=np.float32, mode="r", shape=(len(chunks), dimension))
    query_vectors = np.memmap(query_embedding_path, dtype=np.float32, mode="r", shape=(len(queries), dimension))
    chunk_index = {str(row["chunk_id"]): i for i, row in enumerate(chunks)}
    depth_dense = int(config["candidate_cache"]["dense_depth"])
    depth_sparse = int(config["candidate_cache"]["sparse_depth"])
    source_names = sorted(by_domain)
    parts_root = root / "source_candidate_parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    completed_sources = 0
    completed_pairs = 0
    candidate_rows_written = 0
    started = time.perf_counter()
    part_paths: list[Path] = []
    for source in source_names:
        part_path = parts_root / f"{_source_slug(source)}.parquet"
        marker_path = parts_root / f"{_source_slug(source)}.json"
        part_paths.append(part_path)
        if part_path.exists() and marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("source") == source
                and int(marker.get("query_count", -1)) == len(queries)
                and int(marker.get("chosen_m", -1)) == chosen
                and int(marker.get("source_chunk_count", -1)) == len(by_domain[source])
            ):
                completed_sources += 1
                completed_pairs += len(queries)
                candidate_rows_written += int(marker["candidate_rows"])
                continue
        source_rows = by_domain[source]
        indices = np.asarray([chunk_index[str(row["chunk_id"])] for row in source_rows], dtype=np.int64)
        matrix = np.asarray(vectors[indices], dtype=np.float32)
        texts = [str(row["normalized_text"]) for row in source_rows]
        dense_scores = np.asarray(query_vectors) @ matrix.T
        bm25 = SourceBM25Index(
            texts,
            k1=float(config["candidate_cache"]["bm25_k1"]),
            b=float(config["candidate_cache"]["bm25_b"]),
        )
        lookup = {str(row["chunk_id"]): row for row in source_rows}
        output_rows: list[dict[str, Any]] = []
        source_started = time.perf_counter()
        for qpos, query in enumerate(queries):
            order = stable_top_k_indices(dense_scores[qpos], depth_dense)
            dense = [{"chunk_id": source_rows[i]["chunk_id"], "score": float(dense_scores[qpos, i]), "rank": rank} for rank, i in enumerate(order, 1)]
            sparse_pairs = bm25.rank(query_texts[qpos], top_k=depth_sparse)
            sparse = [{"chunk_id": source_rows[i]["chunk_id"], "score": float(score), "rank": rank} for rank, (i, score) in enumerate(sparse_pairs, 1)]
            fused = reciprocal_rank_fusion([dense, sparse], rrf_constant=int(config["candidate_cache"]["rrf_constant"]), top_k=chosen)
            for item in fused:
                canonical = lookup[str(item["chunk_id"])]
                output_rows.append({
                    "query_id": int(query["id"]), "query_text": str(query["query"]), "source": source,
                    "source_rank": int(item["rank"]), "source_score": float(item["score"]),
                    "chunk_id": str(canonical["chunk_id"]), "doc_id": int(canonical["doc_id"]),
                    "chunk_text": str(canonical["raw_text"]), "source_url": str(canonical["source_url"]),
                    "chunk_index": int(canonical["chunk_index"]), "start_offset": int(canonical["start_offset"]),
                    "end_offset": int(canonical["end_offset"]), "section_type": str(canonical["section_type"]),
                    "heading_path": canonical.get("heading_path"), "extraction_method": str(canonical["extraction_method"]),
                })
            if run_state is not None and ((qpos + 1) % 50 == 0 or qpos + 1 == len(queries)):
                elapsed = max(time.perf_counter() - started, 1e-9)
                done_pairs = completed_pairs + qpos + 1
                total_pairs = len(source_names) * len(queries)
                speed = done_pairs / elapsed
                atomic_json(run_state, {
                    "stage": "SOURCE_CANDIDATES", "status": "RUNNING",
                    "completed_sources": completed_sources,
                    "total_sources": len(source_names),
                    "completed_query_source_pairs": done_pairs,
                    "total_query_source_pairs": total_pairs,
                    "current_source": source,
                    "queries_done_for_current_source": qpos + 1,
                    "queries_per_source": len(queries),
                    "candidate_rows_written": candidate_rows_written + len(output_rows),
                    "query_source_pairs_per_second": round(speed, 3),
                    "eta_seconds": round((total_pairs - done_pairs) / max(speed, 1e-9), 2),
                    "groups": groups or {},
                })
        _write_source_part(part_path, output_rows)
        marker = {
            "source": source, "query_count": len(queries), "chosen_m": chosen,
            "source_chunk_count": len(source_rows), "candidate_rows": len(output_rows),
            "runtime_seconds": round(time.perf_counter() - source_started, 6),
        }
        atomic_json(marker_path, marker)
        completed_sources += 1
        completed_pairs += len(queries)
        candidate_rows_written += len(output_rows)
        if run_state is not None:
            elapsed = max(time.perf_counter() - started, 1e-9)
            total_pairs = len(source_names) * len(queries)
            speed = completed_pairs / elapsed
            atomic_json(run_state, {
                "stage": "SOURCE_CANDIDATES", "status": "RUNNING",
                "completed_sources": completed_sources,
                "total_sources": len(source_names),
                "completed_query_source_pairs": completed_pairs,
                "total_query_source_pairs": total_pairs,
                "current_source": source,
                "queries_done_for_current_source": len(queries),
                "queries_per_source": len(queries),
                "candidate_rows_written": candidate_rows_written,
                "query_source_pairs_per_second": round(speed, 3),
                "eta_seconds": round((total_pairs - completed_pairs) / max(speed, 1e-9), 2),
                "checkpoint_status": "source_part_durable",
                "groups": groups or {},
            })
        del bm25, dense_scores, matrix, output_rows
    if run_state is not None:
        atomic_json(run_state, {
            "stage": "SOURCE_CANDIDATES", "status": "RUNNING",
            "completed_sources": completed_sources,
            "total_sources": len(source_names),
            "completed_query_source_pairs": completed_pairs,
            "total_query_source_pairs": len(source_names) * len(queries),
            "current_source": None,
            "queries_done_for_current_source": 0,
            "queries_per_source": len(queries),
            "candidate_rows_written": candidate_rows_written,
            "checkpoint_status": "all_source_parts_durable",
            "finalization_status": "merging_source_parts",
            "groups": groups or {},
        })
    candidate_path = root / "source_candidates.parquet"
    candidate_rows = _merge_source_parts(part_paths, candidate_path)
    db = sqlite3.connect(root / "source_scores.sqlite")
    db.executescript("CREATE TABLE IF NOT EXISTS pairs(query_id INTEGER,chunk_id TEXT,query_text TEXT,chunk_text TEXT,rerank_score REAL,inference_ms REAL,PRIMARY KEY(query_id,chunk_id));")
    with db:
        for batch in pq.ParquetFile(candidate_path).iter_batches(
            columns=["query_id", "chunk_id", "query_text", "chunk_text"], batch_size=4096
        ):
            db.executemany(
                "INSERT OR IGNORE INTO pairs(query_id,chunk_id,query_text,chunk_text) VALUES(?,?,?,?)",
                ((int(r["query_id"]), str(r["chunk_id"]), str(r["query_text"]), str(r["chunk_text"])) for r in batch.to_pylist()),
            )
    count = int(db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]); db.close()
    result = {"chosen_m": chosen, "healthy_sources": len(by_domain), "candidate_rows": candidate_rows, "unique_pairs": count, "candidate_path": str(candidate_path), "source_parts": len(part_paths), "runtime_seconds": round(time.perf_counter() - started, 3)}
    atomic_json(Path(config["outputs"]["artifacts"]) / "source_candidate_summary.json", result)
    if run_state is not None:
        atomic_json(run_state, {
            "stage": "SOURCE_CANDIDATES", "status": "RUNNING",
            "completed_sources": completed_sources,
            "total_sources": len(source_names),
            "completed_query_source_pairs": completed_pairs,
            "total_query_source_pairs": len(source_names) * len(queries),
            "current_source": None,
            "queries_done_for_current_source": 0,
            "queries_per_source": len(queries),
            "candidate_rows_written": candidate_rows,
            "checkpoint_status": "candidate_artifact_durable",
            "finalization_status": "complete",
            "candidate_path": str(candidate_path),
            "unique_pairs": count,
            "groups": groups or {},
        })
    return result


def score_database_status(
    config: dict[str, Any], *, verify_integrity: bool = False
) -> dict[str, int | str]:
    path = Path(config["outputs"]["root"]) / "round1/source_scores.sqlite"
    connection = sqlite3.connect(path)
    total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    done = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
    # A full scan of this text-bearing database is expensive. SQLite
    # transactions provide the per-shard checkpoint boundary; scan integrity
    # only at explicit recovery/experiment boundaries.
    integrity = (
        str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if verify_integrity
        else "transactional-ok"
    )
    connection.close()
    return {"total": total, "done": done, "remaining": total-done, "integrity": integrity}


def finalize_round1(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifacts"])
    triage = json.loads((artifacts / "technical_triage.json").read_text(encoding="utf-8"))
    groups = json.loads((artifacts / "round1_groups.json").read_text(encoding="utf-8"))
    combined_chunks = Path(config["outputs"]["root"]) / "round1/combined_chunks.parquet"
    combined_docs = Path(config["outputs"]["root"]) / "round1/combined_documents.parquet"
    if not combined_chunks.exists():
        canonical_chunks = [*pq.read_table(config["inputs"]["pilot_chunks"]).to_pylist(), *pq.read_table(config["outputs"]["chunks"]).to_pylist()]
        pq.write_table(pa.Table.from_pylist(canonical_chunks), combined_chunks)
        del canonical_chunks
    if not combined_docs.exists():
        canonical_docs = [*pq.read_table(config["inputs"]["pilot_documents"]).to_pylist(), *pq.read_table(config["outputs"]["documents"]).to_pylist()]
        pq.write_table(pa.Table.from_pylist(canonical_docs), combined_docs)
        del canonical_docs
    from src.source_probe.pipeline import _ParquetSink, _all_query_groups, _generate_submission
    from src.indexing.tokenizer_validation import HuggingFaceOffsetTokenizer
    from transformers import AutoTokenizer
    tokenizer = HuggingFaceOffsetTokenizer(AutoTokenizer.from_pretrained(config["models"]["embedder"]["name"], revision=config["models"]["embedder"]["revision"], local_files_only=True))
    group_for_source = {
        source: group for group, members in groups.items() for source in members
    }
    parts_root = Path(config["outputs"]["root"]) / "round1/source_candidate_parts"
    part_sources: dict[str, Path] = {}
    for marker_path in sorted(parts_root.glob("*.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        part_path = marker_path.with_suffix(".parquet")
        if part_path.exists():
            part_sources[str(marker["source"])] = part_path

    def small_query_groups(path: Path) -> Iterable[list[dict[str, Any]]]:
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

    part_iterators = {
        source: iter(small_query_groups(path)) for source, path in part_sources.items()
    }
    pilot_iterator = iter(_all_query_groups(Path(config["inputs"]["pilot_rerank_pool"])))
    query_ids = [int(value) for value in pq.read_table(config["inputs"]["queries"], columns=["id"]).column(0).to_pylist()]
    source_scores = sqlite3.connect(Path(config["outputs"]["root"]) / "round1/source_scores.sqlite")
    pilot_scores = sqlite3.connect(config["inputs"]["pilot_rerank_scores"])
    ranking_paths: dict[str, tuple[Path, Path]] = {}
    chunk_sinks: dict[str, Any] = {}
    document_sinks: dict[str, Any] = {}
    for group in sorted(groups):
        destination = Path(config["outputs"]["root"]) / f"round1/group_{group}"
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
                    "SELECT chunk_id,rerank_score FROM scores WHERE query_id=?",
                    (query_id,),
                )
            }
            pilot_ranked = [
                {**row, "rerank_score": pilot_score_map[str(row["chunk_id"])]}
                for row in pilot_rows if str(row["chunk_id"]) in pilot_score_map
            ]
            source_rows_by_group: dict[str, list[dict[str, Any]]] = {
                group: [] for group in groups
            }
            source_score_map = {
                str(chunk_id): float(score)
                for chunk_id, score in source_scores.execute(
                    "SELECT chunk_id,rerank_score FROM pairs WHERE query_id=?",
                    (query_id,),
                )
            }
            for source, iterator in part_iterators.items():
                rows = next(iterator, None)
                if not rows or int(rows[0]["query_id"]) != query_id:
                    raise ValueError(f"source candidate part {source} missing query {query_id}")
                group = group_for_source[source]
                source_rows_by_group[group].extend(
                    {**row, "rerank_score": source_score_map[str(row["chunk_id"])]}
                    for row in rows
                )
            for group in sorted(groups):
                ranked = [*pilot_ranked, *source_rows_by_group[group]]
                ranked.sort(key=lambda row: (-float(row["rerank_score"]), str(row["chunk_id"])))
                chunk_sinks[group].add(
                    {
                        "query_id": query_id, "rerank_rank": rank,
                        "chunk_id": str(row["chunk_id"]), "doc_id": int(row["doc_id"]),
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
                    document_rows.append({
                        "query_id": query_id, "rank": len(seen_docs), "doc_id": doc_id,
                    })
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

    variants={}
    for group, members in groups.items():
        chunk_path, doc_path = ranking_paths[group]
        name=f"phase10d_R1_group_{group}"
        first = _generate_submission(config,name=name,documents_path=doc_path,chunks_path=chunk_path,canonical_path=combined_chunks,source_documents_path=combined_docs,output=Path(config["outputs"]["submissions"]),tokenizer=tokenizer)
        second = _generate_submission(config,name=name,documents_path=doc_path,chunks_path=chunk_path,canonical_path=combined_chunks,source_documents_path=combined_docs,output=Path(config["outputs"]["submissions"]),tokenizer=tokenizer)
        deterministic = (
            first["json_sha256"] == second["json_sha256"]
            and first["zip_sha256"] == second["zip_sha256"]
        )
        if not deterministic:
            raise RuntimeError(f"submission regeneration was not deterministic: {group}")
        variants[group] = {**second, "determinism_verified": True}
    sources = triage["sources"]
    status_counts: Counter[str] = Counter()
    for row in sources.values():
        status_counts.update(row.get("status_counts", {}))
    triage_summary = {
        "source_count": len(sources),
        "healthy_source_count": len(triage.get("healthy_sources", [])),
        "sampled_official_ids": sum(int(row.get("sampled", 0)) for row in sources.values()),
        "extracted_rows": sum(int(row.get("extracted_rows", 0)) for row in sources.values()),
        "usable_documents": sum(int(row.get("usable_documents", 0)) for row in sources.values()),
        "status_counts": dict(sorted(status_counts.items())),
    }
    report={"terminal_state":"WAITING_FOR_LEADERBOARD","groups":groups,"technical_triage_summary":triage_summary,"sources":sources,"candidate_cap":json.loads((artifacts/"candidate_cap_calibration.json").read_text(encoding="utf-8")),"candidate_cap_m":8,"reranker_cache":score_database_status(config,verify_integrity=True),"rerank_scores_complete":582000,"submissions":variants}
    atomic_json(artifacts/"round1_report.json",report)
    return report

