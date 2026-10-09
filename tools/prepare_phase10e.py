#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.source_census.pipeline import load_config
from src.source_census.phase10e import preflight
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/phase10e_g5a_11200.yaml'); args=parser.parse_args()
    print(json.dumps(preflight(load_config(args.config)),ensure_ascii=False,indent=2))
