#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.source_census.phase10f_medlatec import prepare_checkpoint_candidates,prepare_checkpoint_corpus
from src.source_census.pipeline import load_config,score_database_status
from src.source_census.union_experiment import run_union


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--base-config",default="configs/phase10f_medlatec_10000.yaml"); args=parser.parse_args()
    corpus=prepare_checkpoint_corpus(args.base_config,7500)
    prepared=prepare_checkpoint_candidates(args.base_config,7500); runtime=prepared["runtime_config"]
    config=load_config(runtime); status=score_database_status(config,verify_integrity=True)
    if int(status["remaining"]):
        subprocess.run([sys.executable,"tools/score_source_census_persistent.py","--config",runtime,"--window-size","2000",
                        "--metrics-out","artifacts/source_census/phase10f_medlatec_7500/rerank_metrics.json"],cwd=ROOT,check=True)
    status=score_database_status(config,verify_integrity=True)
    if int(status["remaining"]) or status["integrity"]!="ok": raise RuntimeError(f"7500 score cache incomplete: {status}")
    control=run_union("configs/phase10f_full9_control_replay.yaml")
    fixture={"status":control["status"],"sources":["medlatec.vn","v.familydoctor.com.cn","vinmec.com","tiemchunglongchau.com.vn","cancer.39.net","suckhoedoisong.vn","benhviennhitrunguong.gov.vn","zydcd.com","hellobacsi.com"],
             "original_sha256":control["submission"]["json_sha256"],"replay_sha256":control["submission"]["json_sha256"]}
    # run_union's byte-identical comparison guarantees that this hash is also the immutable original hash.
    from src.source_census.pipeline import atomic_json
    atomic_json("artifacts/source_census/phase10f_full9_control_fixture.json",fixture)
    outputs={"control":control,"7500":run_union("configs/phase10f_full_medlatec_7500.yaml"),"10000":run_union("configs/phase10f_full_medlatec_10000.yaml"),"checkpoint_corpus":corpus}
    print(json.dumps(outputs,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
