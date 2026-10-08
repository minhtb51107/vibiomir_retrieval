#!/usr/bin/env python3
"""Upgrade a completed legacy repair run only after the scientific gate passes."""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.source_census.pipeline import atomic_json,load_config
from src.validation.pre_submission import audit_depth1000

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/source_census_depth1000.yaml'); p.add_argument('--run-id',required=True); a=p.parse_args()
    cfg=load_config(a.config); run=Path('artifacts/runs')/a.run_id; final=run/'final_report.json'
    while not final.exists(): time.sleep(5)
    terminal=json.loads(final.read_text(encoding='utf-8'))
    if terminal.get('status') != 'WAITING_FOR_LEADERBOARD': return 2
    markers={}
    for group in ('G1A','G5A','G6B'):
        path=Path(cfg['outputs']['artifacts'])/f'submission_FIXED_{group}.json'
        if not path.exists(): raise RuntimeError(f'missing fixed submission marker: {path}')
        markers[group]=json.loads(path.read_text(encoding='utf-8'))
    manifest=Path(cfg['outputs']['artifacts'])/'fixed_submission_manifest.json'; atomic_json(manifest,markers)
    result=audit_depth1000(cfg,markers,Path(cfg['outputs']['root'])/'rankings_fixed')
    output=Path('artifacts/validation/phase10d_depth1000_fixed_pre_submit_audit.json'); atomic_json(output,result)
    state={**terminal,'stage':result['status'],'status':result['status'],'pre_submit_audit':str(output)}
    atomic_json(run/'state.json',state); atomic_json(run/'scientific_final_report.json',state)
    return 0 if result['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
