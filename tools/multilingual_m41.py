from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from src.multilingual_retrieval.m41 import retrieve_by_text, semantic_preflight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4.1 query-text-only Chinese evidence retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    query = sub.add_parser("query")
    query.add_argument("query_vi")
    query.add_argument("--top-k", type=int, default=20)
    query.add_argument("--include-candidates", action="store_true")
    sub.add_parser("semantic-preflight")
    sub.add_parser("evaluate-development")
    args = parser.parse_args()
    if args.command == "query":
        result = retrieve_by_text(args.query_vi, args.top_k)
        if not args.include_candidates:
            result["candidate_count"] = len(result.pop("candidates"))
    elif args.command == "semantic-preflight":
        result = semantic_preflight()
    else:
        from src.multilingual_retrieval.m41_evaluation import evaluate
        result = evaluate(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
