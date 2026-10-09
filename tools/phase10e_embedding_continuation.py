#!/usr/bin/env python3
"""Lightweight EMBEDDINGS boundary: no PyArrow/NumPy/model imports in parent."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]


def system_memory() -> dict[str,float]:
    if os.name!="nt":
        return {"available_physical_mib":float("inf"),"available_commit_mib":float("inf")}
    class MemoryStatus(ctypes.Structure):
        _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
    value=MemoryStatus(); value.dwLength=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    return {"available_physical_mib":value.ullAvailPhys/2**20,"available_commit_mib":value.ullAvailPageFile/2**20}


def memory_gate_status(profile: dict,current: dict[str,float]) -> dict:
    gate=profile["memory_gate"]
    peak_rss=float(gate["observed_peak_rss_mib"])
    model_loaded=next(
        (float(item["rss_mib"]) for item in profile["measurements"] if item.get("phase")=="model_loaded"),
        None,
    )
    if model_loaded is None or model_loaded>peak_rss:
        raise ValueError("validated embedding profile lacks a usable model-loaded RSS measurement")
    # Reserve one additional measured model-loaded-to-peak excursion. This is
    # the observed tokenizer/first-batch warmup demand, not an arbitrary fixed
    # machine-memory threshold.
    physical_safety_headroom=peak_rss-model_loaded
    required_physical=peak_rss+physical_safety_headroom
    required_commit=float(gate["projected_peak_private_mib"])
    return {**current,"measured_peak_rss_mib":peak_rss,"physical_safety_headroom_mib":physical_safety_headroom,"required_physical_mib":required_physical,"required_commit_mib":required_commit,"passed":current["available_physical_mib"]>=required_physical and current["available_commit_mib"]>=required_commit,"rule":"available physical memory must cover the measured peak RSS plus one measured model-load-to-peak warmup excursion; available commit must cover the measured projected peak private bytes"}


def wait_for_memory(config: dict,state: Path) -> dict:
    profile=json.loads(Path(config["embedding_recovery"]["profile_artifact"]).read_text(encoding="utf-8"))
    while True:
        result=memory_gate_status(profile,system_memory())
        if result["passed"]: return result
        write_state(state,{"stage":"EMBEDDINGS_MEMORY_WAIT","status":"RUNNING","checkpoint_preserved":True,"memory_gate":result})
        time.sleep(30)


def write_state(path: Path,value: dict) -> None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--run-state",required=True); parser.add_argument("--max-native-restarts",type=int,default=5); args=parser.parse_args()
    state=Path(args.run_state); python=sys.executable
    config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    memory_gate=wait_for_memory(config,state)
    write_state(state,{"stage":"EMBEDDINGS","status":"RUNNING","checkpoint_preserved":True,"startup_memory_gate":memory_gate})
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
