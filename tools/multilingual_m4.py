#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="strict")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multilingual_retrieval.m4 import freeze_target_manifest, load_lexicon, run_m4, understand_vi_query


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4 bounded VI-to-ZH answer retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    sub.add_parser("run")
    understand = sub.add_parser("understand")
    understand.add_argument("query")
    args = parser.parse_args()
    if args.command == "manifest":
        value = freeze_target_manifest(ROOT, ROOT / "artifacts/multilingual_retrieval_m4/targeted_manifest.json")
    elif args.command == "understand":
        value = understand_vi_query(args.query, load_lexicon(ROOT / "configs/m4_vi_zh_medical_lexicon.json"))
    else:
        value = run_m4(ROOT)
    print(json.dumps(value, ensure_ascii=False, indent=2))
