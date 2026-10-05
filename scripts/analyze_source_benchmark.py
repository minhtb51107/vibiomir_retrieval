from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.acquisition_benchmark.metrics import analyze_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, help="stage200, final1000, or auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/source_acquisition_benchmark.yaml").read_text(encoding="utf-8"))
    crawler = yaml.safe_load(Path(config["pipeline"]["crawler_config"]).read_text(encoding="utf-8"))
    default_concurrency = int(crawler["concurrency"]["per_domain_limit"])
    default_delay = float(crawler["concurrency"]["default_domain_delay_seconds"])
    overrides = crawler["concurrency"].get("domain_overrides", {})
    results = []
    for domain, metadata in config["domains"].items():
        policy = overrides.get(domain, {})
        stage = args.stage
        if stage == "auto":
            candidate = Path(config["pipeline"]["output_root"]) / domain / "final1000" / "documents.parquet"
            stage = "final1000" if candidate.exists() else "stage200"
        results.append(analyze_source(
            domain=domain, corpus_rows=int(metadata["corpus_rows"]), corpus_share=float(metadata["corpus_share"]),
            root=Path(config["pipeline"]["output_root"]) / domain, stage=stage,
            concurrency=int(policy.get("concurrency", default_concurrency)),
            delay_seconds=float(policy.get("delay_seconds", default_delay)),
        ))
    payload = {"stage": args.stage, "sources": results}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({row["domain"]: {"usable_rate": row["usable_doc_rate"], "early_stop": row["early_stop"], "reason": row["early_stop_reason"]} for row in results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
