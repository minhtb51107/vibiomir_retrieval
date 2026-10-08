#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.source_census.pipeline import atomic_json,load_config
from src.validation.pre_submission import audit_depth1000
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/source_census_depth1000.yaml'); p.add_argument('--submissions-manifest',required=True); p.add_argument('--output',required=True); a=p.parse_args()
 cfg=load_config(a.config); submissions=json.loads(Path(a.submissions_manifest).read_text(encoding='utf-8'))
 result=audit_depth1000(cfg,submissions,Path(cfg['outputs']['root'])/'rankings_fixed'); atomic_json(Path(a.output),result); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0 if result['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
