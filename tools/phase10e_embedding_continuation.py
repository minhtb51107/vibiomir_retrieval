#!/usr/bin/env python3
"""Lightweight EMBEDDINGS boundary: no PyArrow/NumPy/model imports in parent."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def write_state(path: Path,value: dict) -> None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--run-state",required=True); parser.add_argument("--max-native-restarts",type=int,default=5); args=parser.parse_args()
    state=Path(args.run_state); python=sys.executable
    prepare=[python,"tools/phase10e_embed_streaming.py","prepare","--config",args.config]
    subprocess.run(prepare,cwd=ROOT,check=True)
    encode=[python,"tools/phase10e_embed_streaming.py","encode","--config",args.config,"--run-state",args.run_state]
    restarts=0
    while True:
        completed=subprocess.run(encode,cwd=ROOT)
        if completed.returncode==0: break
        # Windows STATUS_ACCESS_VIOLATION. All completed rows are durable, so a
        # fresh isolated model-load retry is safe and does not repeat them.
        if completed.returncode not in {3221225477,-1073741819} or restarts>=args.max_native_restarts:
            return completed.returncode
        restarts+=1
        write_state(state,{"stage":"EMBEDDINGS","status":"RUNNING","native_model_load_restarts":restarts,"last_exit_code":completed.returncode,"checkpoint_preserved":True})
        time.sleep(5)
    downstream=[python,"tools/phase10e_worker.py","--config",args.config,"--run-state",args.run_state,"--resume-from","source_candidates"]
    return subprocess.run(downstream,cwd=ROOT).returncode


if __name__=="__main__": raise SystemExit(main())
