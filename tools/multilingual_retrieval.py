#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multilingual_retrieval.evaluation import evaluate, prepare, render_review
from src.multilingual_retrieval.review_server import serve_review


def main() -> int:
    parser = argparse.ArgumentParser(description="ViBioMIR multilingual retrieval benchmark entrypoint")
    parser.add_argument("command", choices=("prepare", "evaluate", "review", "serve-review", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/multilingual_retrieval.yaml"))
    parser.add_argument("--split", choices=("development", "held_out"), default="development")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    if args.command in {"prepare", "run"}:
        print(json.dumps(prepare(config), ensure_ascii=False, indent=2))
    if args.command in {"evaluate", "run"}:
        print(json.dumps(evaluate(config), ensure_ascii=False, indent=2))
    if args.command == "review":
        print(json.dumps(render_review(config, args.split), ensure_ascii=False, indent=2))
    if args.command == "serve-review":
        serve_review(config, args.split, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
