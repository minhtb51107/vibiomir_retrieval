from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit

import psutil
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from src.source_census.pipeline import atomic_json


PHASE14_NAME = "phase14_medlatec_coverage"
PHASE14_RUN_ID = "phase14_medlatec_coverage_3000"


def load_acquisition_config(path: str | Path) -> dict:
    """Load only the bounded Phase 14 acquisition contract.

    Phase 14 intentionally does not pass through the Phase 10D/E shared
    loader. This keeps the legacy validator closed instead of teaching it to
    accept unrelated future phases.
    """
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("phase") != PHASE14_NAME or value.get("run_id") != PHASE14_RUN_ID:
        raise ValueError("unexpected Phase 14 Medlatec acquisition configuration")
    if value.get("focused_scaling", {}).get("phase_key") != "phase14_medlatec":
        raise ValueError("Phase 14 phase_key mismatch")
    phase = value.get("phase14_medlatec", {})
    if phase.get("sampling_seed") != "vibiomir-phase14-medlatec-coverage-v1":
        raise ValueError("Phase 14 sampling seed mismatch")
    if int(phase.get("maximum_new_ids", 0)) != 3000:
        raise ValueError("Phase 14 acquisition must remain bounded at exactly 3000 selected IDs")
    if int(value.get("acquisition", {}).get("minimum_free_gib", 0)) != 20:
        raise ValueError("Phase 14 disk floor must remain 20 GiB")
    if int(value.get("acquisition", {}).get("global_concurrency", 0)) != 1:
        raise ValueError("Phase 14 requires a single acquisition owner")
    if int(value.get("models", {}).get("reranker", {}).get("batch_size", 0)) != 2:
        raise ValueError("frozen reranker contract changed")
    return value


def acquisition_manifest(config: dict) -> tuple[Path, list[dict], str]:
    phase = config["phase14_medlatec"]
    path = Path(phase["manifest"])
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment") != config["run_id"] or int(value.get("total_incremental_ids", 0)) != 3000:
        raise ValueError("Phase 14 durable manifest identity/count mismatch")
    records = value.get("sources", {}).get("medlatec.vn", {}).get("incremental_records", [])
    normalized = [
        {"doc_id": int(row["doc_id"]), "original_url": str(row.get("original_url") or row["url"])}
        for row in records
    ]
    ids = [row["doc_id"] for row in normalized]
    if len(ids) != 3000 or len(set(ids)) != 3000:
        raise ValueError("Phase 14 selection is not exactly 3000 unique official IDs")
    digest = hashlib.sha256("\n".join(map(str, sorted(ids))).encode()).hexdigest()
    preflight = json.loads(Path(phase["preflight"]).read_text(encoding="utf-8"))
    if digest != preflight.get("selected_id_sha256"):
        raise ValueError("Phase 14 selection differs from the durable preflight")
    return path, normalized, digest


def crawl_checkpoint_audit(database: str | Path, selected_records: list[dict]) -> dict:
    path = Path(database)
    selected = {int(row["doc_id"]): str(row["original_url"]) for row in selected_records}
    if not path.exists():
        return {"passed": True, "integrity": "not_created", "rows": 0, "success": 0,
                "status_counts": {}, "remaining": len(selected), "unexpected_ids": 0,
                "url_mismatches": 0}
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=10)
    try:
        connection.execute("PRAGMA query_only=ON")
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        rows = connection.execute("SELECT doc_id,original_url,status FROM crawl_results").fetchall()
    finally:
        connection.close()
    ids = [int(row[0]) for row in rows]
    unexpected = [doc_id for doc_id in ids if doc_id not in selected]
    mismatches = [int(doc_id) for doc_id, url, _ in rows if selected.get(int(doc_id)) != str(url)]
    counts: dict[str, int] = {}
    for _, _, status in rows:
        counts[str(status)] = counts.get(str(status), 0) + 1
    passed = integrity == "ok" and len(ids) == len(set(ids)) and not unexpected and not mismatches
    return {
        "passed": passed, "integrity": integrity, "rows": len(ids),
        "success": counts.get("SUCCESS", 0), "status_counts": counts,
        "remaining": len(selected) - len(ids), "unexpected_ids": len(unexpected),
        "url_mismatches": len(mismatches),
    }


def acquisition_progress(total: int, done: int, *, rate: float, database: Path) -> dict:
    remaining = max(0, total - done)
    return {
        "stage": "ACQUIRE", "status": "RUNNING",
        "selected": total, "completed": done,
        "total": total, "done": done, "remaining": remaining,
        "progress": {"stage": "ACQUIRE", "total": total, "done": done, "remaining": remaining},
        "urls_per_minute": round(rate, 3),
        "eta_minutes": round(remaining / rate, 2) if rate else None,
        "checkpoint": {"database": str(database), "committed_rows": done},
        "last_successful_checkpoint": done,
    }


def host(url: str) -> str:
    return (urlsplit(str(url)).hostname or "").lower().removeprefix("www.")


def deterministic_new_records(records: list[dict], excluded: set[int], count: int, seed: str) -> list[dict]:
    eligible = [row for row in records if int(row["doc_id"]) not in excluded]
    ranked = sorted(
        eligible,
        key=lambda row: (hashlib.sha256(f"{seed}:{int(row['doc_id'])}".encode()).hexdigest(), int(row["doc_id"])),
    )
    if len(ranked) < count:
        raise ValueError(f"only {len(ranked)} unacquired official rows available; need {count}")
    return ranked[:count]


def _official_medlatec(path: Path) -> list[dict]:
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["id", "url"], batch_size=65536):
        for doc_id, url in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            if host(str(url)) == "medlatec.vn":
                rows.append({"doc_id": int(doc_id), "url": str(url)})
    return rows


def _registry_ids(path: Path) -> set[int]:
    table = pq.read_table(path, columns=["doc_id", "canonical_domain", "content_available"])
    mask = pc.and_(pc.equal(table["canonical_domain"], "medlatec.vn"), table["content_available"])
    return {int(value) for value in table.filter(mask)["doc_id"].to_pylist()}


def _crawl_ids(paths: list[Path]) -> tuple[set[int], set[int]]:
    attempted: set[int] = set()
    successful: set[int] = set()
    for path in paths:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        for doc_id, status in connection.execute(
            "SELECT doc_id,status FROM crawl_results WHERE domain='medlatec.vn'"
        ):
            attempted.add(int(doc_id))
            if status == "SUCCESS":
                successful.add(int(doc_id))
        connection.close()
    return attempted, successful


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase = config[config["focused_scaling"]["phase_key"]]
    inputs = {key: Path(value) for key, value in config["inputs"].items() if isinstance(value, str)}
    required = [config_path, inputs["corpus"], inputs["queries"], inputs["canonical_registry"],
                inputs["old_medlatec_manifest"], inputs["old_medlatec_documents"],
                inputs["old_medlatec_chunks"], inputs["old_medlatec_embeddings"],
                inputs["old_medlatec_scores"], inputs["old_medlatec_candidate_part"],
                inputs["phase11c_audit"], Path(config["embedding_recovery"]["profile_artifact"]),
                *map(Path, phase["previous_crawl_databases"])]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase14 dependencies missing: {missing}")

    official = _official_medlatec(inputs["corpus"])
    official_ids = {int(row["doc_id"]) for row in official}
    local_ids = _registry_ids(inputs["canonical_registry"])
    attempted, successful = _crawl_ids([Path(value) for value in phase["previous_crawl_databases"]])
    if not local_ids <= official_ids or not attempted <= official_ids:
        raise ValueError("local/crawl Medlatec IDs are not a subset of the NEW official corpus")

    count = int(phase["maximum_new_ids"])
    selected = deterministic_new_records(official, local_ids | attempted, count, phase["sampling_seed"])
    selected_ids = {int(row["doc_id"]) for row in selected}
    old_manifest = json.loads(inputs["old_medlatec_manifest"].read_text(encoding="utf-8"))
    old_target = {int(value) for value in old_manifest["sources"]["medlatec.vn"]["target_ids"]}
    if len(old_target) != 10000 or old_target & selected_ids:
        raise ValueError("Phase14 old/new target boundary is invalid")

    measurement = phase["historical_measurement"]
    scale = count / float(measurement["attempted_ids"])
    projected_usable = round(count * measurement["usable_documents"] / measurement["attempted_ids"])
    projected_chunks = round(count * measurement["chunks"] / measurement["attempted_ids"])
    projected_clean = round(count * measurement["clean_characters"] / measurement["attempted_ids"])
    combined_chunks = pq.ParquetFile(inputs["old_medlatec_chunks"]).metadata.num_rows + projected_chunks
    embedding_bytes = combined_chunks * int(config["models"]["embedder"]["dimension"]) * 4
    # Based on the retained Phase10F crawl/process footprint, then add the full
    # union embedding matrix, a second transactional vector file, and 15% contingency.
    incremental_crawl_process = round((0.235 + 0.004 + 0.229 + 0.040 + 0.042) * 2**30 * scale)
    retained = round((incremental_crawl_process + embedding_bytes + 0.20 * 2**30) * 1.15)
    transient = retained + embedding_bytes + round(0.50 * 2**30)
    disk = shutil.disk_usage(Path.cwd())
    memory = psutil.virtual_memory()
    control_json = Path(config["focused_scaling"]["control_submission_json"])
    control_zip = Path(config["focused_scaling"]["control_submission_zip"])
    hashes = {
        "json": _sha256(control_json),
        "zip": _sha256(control_zip),
    }
    expected_hashes = {
        "json": config["focused_scaling"]["control_json_sha256"],
        "zip": config["focused_scaling"]["control_zip_sha256"],
    }
    projection = {
        "selected_new_official_ids": count,
        "projected_usable_new_documents": projected_usable,
        "projected_new_chunks": projected_chunks,
        "projected_new_embedding_rows": projected_chunks,
        "projected_incremental_clean_characters": projected_clean,
        "projected_final_focused_medlatec_chunks": combined_chunks,
        "projected_retained_bytes": retained,
        "projected_transient_peak_bytes": transient,
        "free_disk_before_bytes": disk.free,
        "projected_minimum_free_bytes": disk.free - transient,
        "disk_floor_bytes": 20 * 2**30,
        "disk_gate_passed": disk.free - transient >= 20 * 2**30,
        "available_physical_mib": memory.available / 2**20,
        "required_embedding_physical_mib": float(measurement["required_embedding_physical_mib"]),
        "embedding_memory_gate_passed_now": memory.available / 2**20 >= float(measurement["required_embedding_physical_mib"]),
        "embedding_policy": "acquisition may run independently; model load must wait for the established memory gate",
        "runtime_seconds": {
            "acquisition": count / float(measurement["acquisition_urls_per_minute"]) * 60,
            "extraction": float(measurement["extraction_seconds"]) * scale,
            "chunking": float(measurement["chunking_seconds"]) * scale,
            "embedding": float(measurement["embedding_seconds"]) * projected_chunks / float(measurement["chunks"]),
            "medlatec_m16_candidate_recompute": float(measurement["medlatec_m16_candidate_seconds"]) * combined_chunks / 196653,
            "reranking": "UNKNOWN until exact new candidate keys are reconciled",
        },
    }
    manifest = {
        "format_version": 1,
        "experiment": config["run_id"],
        "selection": "lowest SHA256(seed:doc_id) among NEW-official Medlatec IDs absent from canonical registry and every prior Medlatec crawl cache",
        "sources": {"medlatec.vn": {
            "official_population": len(official),
            "current_local_usable_documents": len(local_ids),
            "already_attempted": len(attempted),
            "already_successful_fetches": len(successful),
            "successful_fetches_not_usable": len(successful - local_ids),
            "excluded_ids": len(local_ids | attempted),
            "new_ids_required": count,
            "target_depth": len(old_target) + count,
            "target_ids": sorted(old_target | selected_ids),
            "incremental_records": selected,
        }},
        "total_incremental_ids": count,
        "resource_projection": projection,
    }
    manifest_path = Path(phase["manifest"])
    atomic_json(manifest_path, manifest)
    preflight = {
        "run_id": config["run_id"],
        "passed": bool(projection["disk_gate_passed"] and hashes == expected_hashes),
        "acquisition_launch_allowed": bool(projection["disk_gate_passed"] and hashes == expected_hashes),
        "embedding_start_allowed_now": bool(projection["embedding_memory_gate_passed_now"]),
        "frozen_phase11c_hashes": hashes,
        "expected_phase11c_hashes": expected_hashes,
        "official_medlatec_ids": len(official),
        "current_local_usable_documents": len(local_ids),
        "prior_attempted_ids": len(attempted),
        "selected_new_ids": count,
        "selected_id_sha256": hashlib.sha256("\n".join(map(str, sorted(selected_ids))).encode()).hexdigest(),
        "robots_and_rate_limit": {
            "robots_enabled": True,
            "per_domain_concurrency": int(config["acquisition"]["per_domain_concurrency"]),
            "default_delay_seconds": 1.0,
            "historical_successful_fetches": int(measurement["attempted_ids"]) - 11,
            "historical_attempts": int(measurement["attempted_ids"]),
            "historical_failed_sources": 0,
        },
        "resource_projection": projection,
        "scientific_contract": {
            "control": "immutable Phase11C Docs20/Chunks40",
            "candidate_cap": 16,
            "ranking_formula_changed": False,
            "pubmed_or_translation_branch": False,
            "only_intended_variable": "addition of deterministically selected new official Medlatec documents",
        },
    }
    atomic_json(Path(phase["preflight"]), preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"Phase14 preflight failed: {preflight}")
    return {"manifest": manifest, "preflight": preflight}
