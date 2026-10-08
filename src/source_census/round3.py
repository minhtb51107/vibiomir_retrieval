from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.acquisition_benchmark.sampling import deterministic_domain_samples
from src.source_census.pipeline import atomic_json, score_database_status


ROUND3_GROUPS = {
    "G1A": ["medlatec.vn", "v.familydoctor.com.cn", "vinmec.com"],
    "G1B": ["vov2.vov.vn", "familydoctor.cn", "suckhoeviet.org.vn"],
    "G5A": ["benhviennhitrunguong.gov.vn", "zydcd.com", "hellobacsi.com"],
    "G5B": ["tuoitre.vn", "qy.familydoctor.com.cn", "tnb.39.net"],
    "G6A": ["baoquangninh.vn", "jb39.com", "baochinhphu.vn"],
    "G6B": ["tiemchunglongchau.com.vn", "cancer.39.net", "suckhoedoisong.vn"],
}


def prepare_round3_groups(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifacts"])
    round2 = json.loads((artifacts / "round2_groups.json").read_text(encoding="utf-8"))
    round2_map = {
        str(row["source"]): {**row, "round2_group": group}
        for group, rows in round2["groups"].items()
        for row in rows
    }
    expected = set(round2_map)
    survivors = set(ROUND3_GROUPS["G1A"] + ROUND3_GROUPS["G1B"] + ROUND3_GROUPS["G5A"] + ROUND3_GROUPS["G5B"] + ROUND3_GROUPS["G6A"] + ROUND3_GROUPS["G6B"])
    round2_survivors = {
        source for group in ("G1", "G5", "G6")
        for source in [str(row["source"]) for row in round2["groups"][group]]
    }
    if survivors != round2_survivors or not survivors.issubset(expected) or len(survivors) != 18:
        raise ValueError("fixed Round-3 groups do not exactly partition Round-2 G1/G5/G6")
    result = {
        "format_version": 1,
        "method": "organizer-directed sibling split of Round-2 G1/G5/G6",
        "experimental_control": {
            "only_variable": "source membership",
            "source_depth": "unchanged Round-1 sample depth",
            "candidate_cap_m": 8,
            "new_crawling": 0,
            "new_extraction": 0,
            "new_chunking": 0,
            "new_embeddings": 0,
            "new_reranking": 0,
            "reused_scores": 582000,
        },
        "groups": {
            group: [round2_map[source] for source in sources]
            for group, sources in ROUND3_GROUPS.items()
        },
    }
    atomic_json(artifacts / "round3_groups.json", result)
    return result


def _worker_evidence(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    root = Path(config["outputs"]["worker_root"])
    evidence: dict[str, dict[str, float]] = {}
    for path in root.glob("*/summary.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            domains = value.get("run", {}).get("by_domain", {})
            if len(domains) != 1:
                continue
            source = next(iter(domains))
            completed = int(value["run"].get("completed", 0))
            directory_bytes = sum(item.stat().st_size for item in path.parent.rglob("*") if item.is_file())
            evidence[source] = {
                "completed": completed,
                "wall_seconds": float(value["run"].get("wall_seconds", 0.0)),
                "downloaded_bytes": int(value["run"].get("downloaded_bytes", 0)),
                "retained_worker_bytes": directory_bytes,
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return evidence


def prepare_depth_manifests(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifacts"])
    group_artifact = json.loads((artifacts / "round3_groups.json").read_text(encoding="utf-8"))
    sources = sorted({str(row["source"]) for rows in group_artifact["groups"].values() for row in rows})
    round1 = json.loads((artifacts / "round1_sample_manifest.json").read_text(encoding="utf-8"))
    current = {
        source: list(round1["domains"][source]["records"])
        for source in sources
    }
    samples, populations = deterministic_domain_samples(
        config["inputs"]["corpus"],
        domains=set(sources),
        sample_size=1100,
        seed="vibiomir-phase10d-survivor-depth-v1",
    )
    worker = _worker_evidence(config)
    manifests: dict[str, Any] = {}
    summary: dict[str, Any] = {"sources": sources, "targets": {}}
    for target in (300, 1000):
        domains: dict[str, Any] = {}
        total_incremental = 0
        total_disk = 0.0
        total_seconds = 0.0
        insufficient: list[str] = []
        for source in sources:
            current_ids = {int(row["doc_id"]) for row in current[source]}
            population = int(populations.get(source, 0))
            achievable = min(target, population)
            needed = max(0, achievable - len(current_ids))
            candidates = [row for row in samples[source] if int(row["doc_id"]) not in current_ids]
            incremental = candidates[:needed]
            if len(incremental) != needed:
                raise ValueError(f"insufficient deterministic sample window for {source} depth {target}")
            observed = worker.get(source, {})
            completed = int(observed.get("completed", 0))
            seconds_per_url = float(observed.get("wall_seconds", 0.0)) / completed if completed else None
            bytes_per_url = float(observed.get("retained_worker_bytes", 0.0)) / completed if completed else None
            projected_seconds = needed * seconds_per_url if seconds_per_url is not None else None
            projected_bytes = needed * bytes_per_url if bytes_per_url is not None else None
            total_incremental += needed
            total_seconds += projected_seconds or 0.0
            total_disk += projected_bytes or 0.0
            if population < target:
                insufficient.append(source)
            domains[source] = {
                "official_population": population,
                "current_acquired_ids": len(current_ids),
                "target_depth": target,
                "achievable_depth": achievable,
                "incremental_ids_required": needed,
                "enough_official_ids": population >= target,
                "records": incremental,
                "projection": {
                    "basis": "linear extrapolation from the source's Round-1 worker summary",
                    "observed_completed": completed,
                    "seconds_per_url": seconds_per_url,
                    "retained_bytes_per_url": bytes_per_url,
                    "incremental_wall_seconds_if_serial": projected_seconds,
                    "incremental_retained_bytes": projected_bytes,
                },
            }
        manifest = {
            "format_version": 1,
            "seed": "vibiomir-phase10d-survivor-depth-v1",
            "sampling": "lowest SHA-256(seed:doc_id) ranks among official source URLs after excluding Round-1 acquired IDs",
            "target_depth": target,
            "new_network_requests": 0,
            "total_incremental_ids": total_incremental,
            "sources_without_target_population": insufficient,
            "projection": {
                "aggregate_retained_bytes": total_disk,
                "aggregate_retained_gib": total_disk / (1024**3),
                "aggregate_source_serial_seconds": total_seconds,
                "aggregate_source_serial_hours": total_seconds / 3600,
                "warning": "Per-source linear engineering projection, not a guaranteed concurrent wall time or content yield.",
            },
            "domains": domains,
        }
        path = artifacts / f"round3_depth{target}_manifest.json"
        atomic_json(path, manifest)
        manifests[str(target)] = {"path": str(path), **manifest}
        summary["targets"][str(target)] = {
            "total_incremental_ids": total_incremental,
            "sources_without_target_population": insufficient,
            **manifest["projection"],
        }
    atomic_json(artifacts / "round3_depth_manifest_summary.json", summary)
    return manifests


def finalize_round3_report(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifacts"])
    assignment = json.loads((artifacts / "round3_groups.json").read_text(encoding="utf-8"))
    submissions = {}
    for group in ROUND3_GROUPS:
        marker = artifacts / f"round3_submission_{group}.json"
        if not marker.exists():
            raise FileNotFoundError(marker)
        submissions[group] = json.loads(marker.read_text(encoding="utf-8"))
    status = score_database_status(config, verify_integrity=True)
    if status != {"total": 582000, "done": 582000, "remaining": 0, "integrity": "ok"}:
        raise ValueError(f"source score cache is not complete: {status}")
    report = {
        "terminal_state": "WAITING_FOR_LEADERBOARD",
        "groups": {group: sources for group, sources in ROUND3_GROUPS.items()},
        "group_assignment": assignment,
        "experimental_control": assignment["experimental_control"],
        "score_cache": status,
        "submissions": submissions,
    }
    atomic_json(artifacts / "round3_report.json", report)
    return report
