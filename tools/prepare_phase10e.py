#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.source_census.pipeline import load_config
from src.source_census.phase10e import prepare_manifest
if __name__=='__main__': print(json.dumps(prepare_manifest(load_config('configs/phase10e_g5a_11200.yaml')),ensure_ascii=False,indent=2))
