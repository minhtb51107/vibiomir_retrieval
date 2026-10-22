from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,"reconfigure"):
        stream.reconfigure(encoding="utf-8")

from src.multilingual_retrieval.m42 import import_review, prepare, require_idle

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="M4.2 bounded semantic pilot preparation; no model loading")
    parser.add_argument("command",choices=["prepare","import-review","inference-preflight"])
    args=parser.parse_args()
    if args.command=="prepare":
        result=prepare(ROOT)
    elif args.command=="import-review":
        out=ROOT/"artifacts/multilingual_retrieval_m42"
        out.mkdir(parents=True,exist_ok=True)
        result=import_review(ROOT,out)
    else:
        result=require_idle(ROOT)
    print(json.dumps(result,ensure_ascii=False,indent=2))
