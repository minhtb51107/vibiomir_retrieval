from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.acquisition_benchmark.metrics import tier_for
from src.indexing.embedder import file_sha256


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    root = Path("artifacts/phase10b2_acquisition")
    metrics = json.loads((root / "measured_sources.json").read_text(encoding="utf-8"))["sources"]
    config = yaml.safe_load(Path("configs/source_acquisition_benchmark.yaml").read_text(encoding="utf-8"))
    healthy_costs = [
        (
            row["performance"]["modeled_crawl_seconds"] / max(row["usable_documents"], 1),
            row["storage"]["mib_per_1000_usable_docs"],
        )
        for row in metrics if not row["early_stop"]
    ]
    median_seconds = statistics.median(value[0] for value in healthy_costs)
    median_storage = statistics.median(value[1] for value in healthy_costs)
    disk_free_bytes = shutil.disk_usage(ROOT).free
    for row in metrics:
        seconds_per_usable = row["performance"]["modeled_crawl_seconds"] / max(row["usable_documents"], 1)
        resource_cost = 0.5 * seconds_per_usable / median_seconds + 0.5 * row["storage"]["mib_per_1000_usable_docs"] / median_storage
        row["normalized_resource_cost"] = round(resource_cost, 6)
        row["source_efficiency"] = round(row["corpus_share"] * row["usable_doc_rate"] / max(resource_cost, 1e-9), 8)
        projected_50k = int(row["projections"]["50k"]["retained_bytes"])
        fits = projected_50k * 1.5 < disk_free_bytes - 20 * 1024 ** 3
        throughput_ok = row["performance"]["urls_per_minute"] >= 20
        row["tier"] = tier_for(
            usable_rate=float(row["usable_doc_rate"]), access_rate=float(row["robots_access_rate"]),
            projected_50k_fits=fits, throughput_acceptable=throughput_ok,
        )
        if row["early_stop"]:
            row["tier"] = "C" if row["robots_access_rate"] < 0.8 else "BLOCKED"
        row["projected_50k_fits_disk_guard"] = fits
        row["main_blocker"] = (
            row["early_stop_reason"] if row["early_stop"] else
            "slow network/retries" if row["performance"]["urls_per_minute"] < 40 else
            "higher storage/chunk density" if row["storage"]["mib_per_1000_usable_docs"] > median_storage * 1.5 else
            "none observed in bounded benchmark"
        )
    ranked = sorted((row for row in metrics if not row["early_stop"]), key=lambda row: (-row["source_efficiency"], row["domain"]))
    safest = sorted(
        (row for row in metrics if row["tier"] == "A"),
        key=lambda row: (-row["performance"]["usable_docs_per_minute"], row["storage"]["mib_per_1000_usable_docs"], row["domain"]),
    )
    new_urls = sum(int(row["performance"]["observed_new_rows"]) for row in metrics)
    summary = {
        "phase": "10B2",
        "interpretation": "acquisition engineering only; no relevance claim",
        "config_sha256": file_sha256("configs/source_acquisition_benchmark.yaml"),
        "sample_manifest_sha256": file_sha256(root / "sample_manifest.json"),
        "sampled_rows_total": sum(row["attempted"] for row in metrics),
        "new_urls_fetched": new_urls,
        "reused_phase8_rows": sum(row["attempted"] for row in metrics) - new_urls,
        "disk_free_bytes_at_finalization": disk_free_bytes,
        "early_stops": [
            {"domain": row["domain"], "attempted": row["attempted"], "reason": row["early_stop_reason"]}
            for row in metrics if row["early_stop"]
        ],
        "source_efficiency_formula": config["efficiency_score"],
        "sources": metrics,
        "source_efficiency_ranking": [row["domain"] for row in ranked],
        "safest_operational_ranking": [row["domain"] for row in safest],
        "recommended_first": safest[0]["domain"] if safest else None,
        "recommended_second": safest[1]["domain"] if len(safest) > 1 else None,
    }
    projections = {
        "basis": "linear engineering projections from bounded measured bytes and modeled source-exclusive time",
        "sources": {
            row["domain"]: {
                "attempted_basis": row["attempted"], "usable_rate": row["usable_doc_rate"],
                "chunks_per_usable_document": row["chunks"]["per_usable_mean"],
                "projected_chunks_per_10000_usable_docs": row["chunks"]["projected_per_10000_usable_docs"],
                "10k": row["projections"]["10k"], "50k": row["projections"]["50k"],
                "full_source": row["projections"]["full_source"],
            }
            for row in metrics
        },
    }
    atomic_json(root / "source_summary.json", summary)
    atomic_json(root / "source_projection.json", projections)
    print(json.dumps({
        "source_efficiency_ranking": summary["source_efficiency_ranking"],
        "safest_operational_ranking": summary["safest_operational_ranking"],
        "early_stops": summary["early_stops"],
        "tiers": {row["domain"]: row["tier"] for row in metrics},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
