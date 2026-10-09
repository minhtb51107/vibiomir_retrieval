#!/usr/bin/env python3
"""Local unattended runner with sleep prevention, locking and incident bundles."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.025 * (attempt + 1))


def event(path: Path, kind: str, **fields: Any) -> None:
    record = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "event": kind, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prevent_sleep(active: bool) -> None:
    if sys.platform == "win32":
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if active else ES_CONTINUOUS
        if not ctypes.windll.kernel32.SetThreadExecutionState(flags):
            raise ctypes.WinError()


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok and code.value == 259)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def recover_stale_run(run: Path, run_id: str) -> dict[str, Any] | None:
    """Mark a dead lock owner as interrupted without touching checkpoints."""
    lock=run/"run.lock"
    state_path=run/"state.json"; events=run/"events.jsonl"
    try:
        prior=json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        prior={}
    if lock.exists():
        raw=lock.read_text(encoding="utf-8",errors="replace")
        try:
            pid=int(next(part.split("=",1)[1] for part in raw.split() if part.startswith("pid=")))
        except (StopIteration,ValueError,IndexError):
            pid=-1
        if process_alive(pid):
            raise FileExistsError(f"live run owner pid={pid}: {lock}")
    elif prior.get("status")!="RUNNING":
        return None
    else:
        pid=None
    recovered={
        **prior,"run_id":run_id,"status":"STALE_RUNNING",
        "interrupted_pid":pid,"checkpoint_preserved":True,
        "stale_detected_at":dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(state_path,recovered)
    event(events,"STALE_RUN_DETECTED",interrupted_pid=pid,checkpoint_preserved=True)
    lock.unlink(missing_ok=True)
    return recovered


def gpu_state() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except Exception as error:
        return f"unavailable: {error}"


def crawl_count(path: Path) -> int | None:
    if not path.exists():
        return 0
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
        for table in ("crawl_results", "results"):
            try:
                value = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                connection.close(); return value
            except sqlite3.Error:
                continue
        connection.close()
    except sqlite3.Error:
        pass
    return None


def create_incident(root: Path, run_id: str, command: list[str], exit_code: int, log_path: Path, state: dict[str, Any]) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = root / "artifacts" / "incidents" / f"{stamp}_{run_id}"
    destination.mkdir(parents=True, exist_ok=False)
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-300:] if log_path.exists() else []
    (destination / "last_log_lines.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (destination / "gpu_state.txt").write_text(gpu_state() + "\n", encoding="utf-8")
    (destination / "reproduction_command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    atomic_json(destination / "incident.json", {"run_id": run_id, "failed_stage": state.get("stage"), "exit_code": exit_code, "command": command, "retry_history": state.get("retry_history", []), "last_successful_checkpoint": state.get("last_successful_checkpoint"), "remaining_work": state.get("remaining"), "artifact_paths": state})
    atomic_json(destination / "system_state.json", {"free_disk_gib": shutil.disk_usage(root).free / 2**30, "platform": sys.platform, "python": sys.version, "pid": os.getpid()})
    atomic_json(destination / "checkpoint_state.json", state)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--": args.command = args.command[1:]
    if not args.command: raise ValueError("worker command is required after --")
    root = Path(__file__).resolve().parents[1]
    run = root / "artifacts" / "runs" / args.run_id
    logs = run / "logs"; logs.mkdir(parents=True, exist_ok=True)
    lock = run / "run.lock"; state_path = run / "state.json"; events = run / "events.jsonl"
    try:
        recovered=recover_stale_run(run,args.run_id)
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"duplicate run refused: {lock}", file=sys.stderr); return 73
    os.write(descriptor, f"pid={os.getpid()}\nrun_id={args.run_id}\n".encode()); os.close(descriptor)
    log_path = logs / "worker.log"
    started = time.time()
    state = {"run_id": args.run_id, "status": "RUNNING", "stage": "STARTING", "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "command": args.command}
    if recovered:
        state["recovered_from"]={"status":"STALE_RUNNING","interrupted_pid":recovered.get("interrupted_pid"),"last_stage":recovered.get("stage"),"last_successful_checkpoint":recovered.get("last_successful_checkpoint")}
    atomic_json(state_path, state); atomic_json(run / "manifest.json", state); event(events, "RUN_STARTED", command=args.command)
    prevent_sleep(True)
    exit_code = -1
    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(args.command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            last_stage = None
            while process.poll() is None:
                try: state = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception: pass
                free = shutil.disk_usage(root).free / 2**30
                if free < args.minimum_free_gib:
                    process.terminate(); state.update(status="FAILED_SAFE", stage="DISK_GUARD", free_disk_gib=free); atomic_json(state_path,state); event(events,"DISK_GUARD",free_disk_gib=free); break
                if state.get("stage") != last_stage:
                    event(events,"STAGE",stage=state.get("stage")); last_stage=state.get("stage")
                if state.get("stage") == "SOURCE_CANDIDATES":
                    total = state.get("total_query_source_pairs")
                    done = state.get("completed_query_source_pairs")
                else:
                    total = state.get("total") or state.get("selected")
                    done = state.get("done") or state.get("completed")
                progress = f"{100*done/total:6.2f}% ({done}/{total})" if total and done is not None else "denominator pending"
                elapsed=max(time.time()-started,1); speed=(done/elapsed if done is not None else None)
                os.system("cls")
                print(f"RUN: {args.run_id}\nSTATUS: {state.get('status','RUNNING')}\nSTAGE: {state.get('stage','UNKNOWN')}\n\nProgress: {progress}\nSpeed: {speed:.2f}/s" if speed is not None else f"RUN: {args.run_id}\nSTATUS: {state.get('status','RUNNING')}\nSTAGE: {state.get('stage','UNKNOWN')}\n\nProgress: {progress}")
                if state.get("stage") == "ACQUIRE":
                    active_names = state.get("active_sources", [])
                    print(
                        f"Active sources: {len(active_names)} ({', '.join(active_names)})\n"
                        f"Completed sources: {state.get('completed_sources', 0)}/{state.get('total_sources', 0)}\n"
                        f"Failed sources: {len(state.get('failed_sources', {}))}\n"
                        f"Aggregate: {state.get('urls_per_minute', 0)} URLs/min\n"
                        f"ETA: {state.get('eta_minutes', 'pending')} minutes"
                    )
                elif state.get("stage") == "SOURCE_CANDIDATES":
                    print(
                        f"Completed sources: {state.get('completed_sources', 0)}/{state.get('total_sources', 0)}\n"
                        f"Current source: {state.get('current_source', 'pending')}\n"
                        f"Current source queries: {state.get('queries_done_for_current_source', 0)}/1200\n"
                        f"Candidate rows: {state.get('candidate_rows_written', 0)}\n"
                        f"Stage speed: {state.get('query_source_pairs_per_second', 0):.2f} query-source pairs/s\n"
                        f"ETA: {state.get('eta_seconds', 'pending')} seconds"
                    )
                elif state.get("stage") == "RERANK":
                    print(
                        f"Active scoring: {state.get('active_pairs_per_second', 0):.2f} pairs/s\n"
                        f"Effective: {state.get('effective_pairs_per_second', 0):.2f} pairs/s\n"
                        f"Model load: {state.get('model_load_seconds', 'pending')} seconds\n"
                        f"Checkpoint window: {state.get('current_checkpoint_window', 'loading')} "
                        f"(size {state.get('shard_size', 'pending')})\n"
                        f"Model loads: {state.get('model_load_count', 0)}; "
                        f"restarts: {state.get('scorer_restarts', 0)}\n"
                        f"RSS/private: {state.get('rss_mib', 'pending')}/"
                        f"{state.get('private_mib', 'pending')} MiB\n"
                        f"GPU allocated/reserved: {state.get('gpu_allocated_mib', 'pending')}/"
                        f"{state.get('gpu_reserved_mib', 'pending')} MiB\n"
                        f"ETA: {state.get('eta_seconds', 'pending')} seconds"
                    )
                print(f"Free disk: {free:.2f} GiB\nGPU: {gpu_state()}\nCheckpoint: {state.get('integrity','pending')}\nCurrent shard: {state.get('shard_size','n/a')}\nRetry count: {len(state.get('retry_history',[]))}\nLast checkpoint: {state.get('last_successful_checkpoint','n/a')}", flush=True)
                atomic_json(run / "metrics.json", {"updated_at":dt.datetime.now(dt.timezone.utc).isoformat(),"stage":state.get("stage"),"free_disk_gib":free,"gpu":gpu_state(),"elapsed_seconds":time.time()-started})
                time.sleep(5)
            exit_code = process.wait()
        try: state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception: state = {"stage":"UNKNOWN"}
        if state.get("status") == "FAILED_SAFE": terminal="FAILED_SAFE"
        elif exit_code == 0 and state.get("status") in {"WAITING_FOR_LEADERBOARD", "READY_FOR_LEADERBOARD"}: terminal=state["status"]
        else:
            terminal="NEEDS_AGENT"; incident=create_incident(root,args.run_id,args.command,exit_code,log_path,state); state["incident_bundle"]=str(incident)
        state.update(status=terminal, exit_code=exit_code, completed_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(state_path,state); atomic_json(run/"final_report.json",state); event(events,"RUN_TERMINAL",status=terminal,exit_code=exit_code)
        return 0 if terminal in {"WAITING_FOR_LEADERBOARD", "READY_FOR_LEADERBOARD"} else 2
    finally:
        prevent_sleep(False)
        lock.unlink(missing_ok=True)


if __name__ == "__main__": raise SystemExit(main())

