#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.submission.common import atomic_write_json
from src.submission.generator import load_submission_config
from src.submission.validator import (
    SubmissionValidationError,
    canonical_provenance,
    expected_query_ids,
    load_source_documents,
    validate_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strictly validate Phase 9 organizer JSON or ZIP submissions."
    )
    parser.add_argument("paths", nargs="+", help="Submission JSON/ZIP files")
    parser.add_argument("--config", default="configs/submission.yaml")
    parser.add_argument("--report")
    parser.add_argument(
        "--source-documents",
        help=(
            "Processed documents Parquet. When given, non-canonical chunk text is "
            "accepted only as a verbatim contiguous span of its source document "
            "(Phase 10A expanded windows)."
        ),
    )
    args = parser.parse_args()
    config = load_submission_config(args.config)
    query_ids = expected_query_ids(config["inputs"]["queries"])
    required_query_count = int(config["organizer_schema"]["required_query_count"])
    if len(query_ids) != required_query_count:
        print(
            f"INVALID: official query input has {len(query_ids)} rows; "
            f"expected {required_query_count}",
            file=sys.stderr,
        )
        return 1
    if len(query_ids) != len(set(query_ids)):
        print("INVALID: official query input has duplicate IDs", file=sys.stderr)
        return 1
    valid_documents, valid_chunks = canonical_provenance(
        config["inputs"]["canonical_chunks"]
    )
    source_documents = (
        load_source_documents(args.source_documents, valid_documents)
        if args.source_documents
        else None
    )
    try:
        results = [
            validate_submission(
                path,
                expected_queries=query_ids,
                valid_document_ids=valid_documents,
                valid_chunks=valid_chunks,
                require_exact_order=bool(
                    config["validation"]["require_exact_query_order"]
                ),
                source_documents=source_documents,
            )
            for path in args.paths
        ]
    except (SubmissionValidationError, OSError, json.JSONDecodeError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    report = {"valid": True, "submission_count": len(results), "results": results}
    if args.report:
        atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
