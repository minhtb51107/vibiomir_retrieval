#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.remote_embedding_poc import duplicate_chunk_analysis, export_package, verify_results
from src.runtime.state_file import write_json


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Phase 14 duplicate audit and bounded Colab embedding PoC")
    commands = parser.add_subparsers(dest="command", required=True)
    duplicate = commands.add_parser("analyze-duplicates")
    duplicate.add_argument("--chunks", type=Path, default=Path("data/source_census/phase14_medlatec_coverage_3000/chunks.parquet"))
    duplicate.add_argument("--output", type=Path, default=Path("artifacts/phase14_medlatec_coverage_3000/duplicate_chunk_analysis.json"))
    export = commands.add_parser("export")
    export.add_argument("--chunks", type=Path, default=Path("data/source_census/phase10e_g1a_10010/chunks.parquet"))
    export.add_argument("--embeddings", type=Path, default=Path("data/source_census/phase10e_g1a_10010/round1/chunk_embeddings.f32"))
    export.add_argument("--previous-chunks", type=Path, default=Path("data/source_census/depth1000/chunks.parquet"))
    export.add_argument("--notebook", type=Path, default=Path("notebooks/colab_embedding_poc.py"))
    export.add_argument("--output", type=Path, default=Path("artifacts/phase14_embedding_poc/phase14_bge_m3_embedding_poc_200.zip"))
    export.add_argument("--report", type=Path, default=Path("artifacts/phase14_embedding_poc/export_report.json"))
    export.add_argument("--count", type=int, default=200)
    export.add_argument("--seed", default="vibiomir-colab-embedding-poc-v1")
    verify = commands.add_parser("verify")
    verify.add_argument("--package", type=Path, default=Path("artifacts/phase14_embedding_poc/phase14_bge_m3_embedding_poc_200.zip"))
    verify.add_argument("--results", type=Path, required=True)
    verify.add_argument("--report", type=Path, default=Path("artifacts/phase14_embedding_poc/equivalence_report.json"))
    args = parser.parse_args()
    if args.command == "analyze-duplicates":
        report = duplicate_chunk_analysis(args.chunks)
        write_json(args.output, report)
        code = 0
    elif args.command == "export":
        report = export_package(
            args.chunks, args.embeddings, args.notebook, args.output,
            previous_chunks_path=args.previous_chunks, count=args.count, seed=args.seed,
        )
        write_json(args.report, report)
        code = 0
    else:
        report = verify_results(args.package, args.results)
        write_json(args.report, report)
        code = 0 if report["passed"] else 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
