#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reranking.remote_poc import export_package, verify_results, write_staging


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded Windows/Colab reranker equivalence proof of concept")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--database", type=Path, default=Path("data/pubmed_phase12/retrieval/round1/source_scores.sqlite"))
    export.add_argument("--candidate-parts", type=Path, default=Path("data/pubmed_phase12/retrieval/round1/source_candidate_parts"))
    export.add_argument("--output", type=Path, default=Path("artifacts/colab_reranker_poc/reranker_poc_200.zip"))
    export.add_argument("--seed", default="vibiomir-colab-reranker-poc-v1")
    export.add_argument("--count", type=int, default=200)
    export.add_argument("--report", type=Path, default=Path("artifacts/colab_reranker_poc/export_report.json"))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--package", type=Path, default=Path("artifacts/colab_reranker_poc/reranker_poc_200.zip"))
    verify.add_argument("--results", type=Path, required=True)
    verify.add_argument("--report", type=Path, default=Path("artifacts/colab_reranker_poc/equivalence_report.json"))
    verify.add_argument("--write-staging", type=Path, help="optional new staging DB; production caches are never modified")
    args = parser.parse_args()
    if args.command == "export":
        report = export_package(args.database, args.candidate_parts, args.output, seed=args.seed, count=args.count)
    else:
        report, comparisons = verify_results(args.package, args.results)
        if args.write_staging:
            write_staging(args.write_staging, report, comparisons)
            report["staging_database"] = str(args.write_staging)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
