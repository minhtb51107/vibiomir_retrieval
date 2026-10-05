from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.acquisition_benchmark.sampling import annotate_phase8_reuse, deterministic_domain_samples, records_manifest
from src.indexing.embedder import file_sha256


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    config_path = Path("configs/source_acquisition_benchmark.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    if file_sha256(source["corpus"]) != source["corpus_sha256"]:
        raise ValueError("official corpus hash mismatch")
    if file_sha256(source["phase8_manifest"]) != source["phase8_manifest_sha256"]:
        raise ValueError("Phase 8 manifest hash mismatch")
    maximum = int(config["sampling"]["maximum_urls_per_source"])
    checkpoint = int(config["sampling"]["checkpoint_urls_per_source"])
    samples, populations = deterministic_domain_samples(
        source["corpus"], domains=set(config["domains"]), sample_size=maximum,
        seed=source["sample_seed"],
    )
    phase8 = json.loads(Path(source["phase8_manifest"]).read_text(encoding="utf-8"))
    reusable = annotate_phase8_reuse(samples, phase8)
    data_root = Path(config["pipeline"]["output_root"]) / "manifests"
    for domain, rows in samples.items():
        slug = domain.replace(".", "_")
        atomic_json(data_root / f"{slug}_checkpoint200.json", records_manifest(rows, checkpoint))
        atomic_json(data_root / f"{slug}_full1000.json", records_manifest(rows, maximum))
    payload = {
        "format_version": 1,
        "source_corpus": source["corpus"],
        "source_sha256": source["corpus_sha256"],
        "sampling_method": "lowest SHA-256(seed:doc_id) ranks within each domain",
        "seed": source["sample_seed"],
        "checkpoint_size": checkpoint,
        "maximum_size": maximum,
        "phase8_manifest_sha256": source["phase8_manifest_sha256"],
        "domains": {
            domain: {
                "population": populations[domain],
                "phase8_reusable": reusable[domain],
                "records": rows,
            }
            for domain, rows in samples.items()
        },
    }
    destination = Path("artifacts/phase10b2_acquisition/sample_manifest.json")
    atomic_json(destination, payload)
    print(json.dumps({
        "domains": len(samples), "records": sum(len(rows) for rows in samples.values()),
        "phase8_reusable": reusable, "manifest_sha256": file_sha256(destination),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
