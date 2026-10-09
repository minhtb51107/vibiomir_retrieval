from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.indexing.embedder import normalize_text


def process_memory() -> dict[str, float]:
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    if os.name != "nt":
        return {"rss_mib":-1.0,"private_mib":-1.0,"page_faults":-1}
    get_process=ctypes.windll.kernel32.GetCurrentProcess; get_process.restype=ctypes.c_void_p
    get_memory=ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]; get_memory.restype=ctypes.c_int
    ok=get_memory(get_process(),ctypes.byref(counters),ctypes.sizeof(counters))
    if not ok: return {"rss_mib":-1.0,"private_mib":-1.0,"page_faults":-1}
    return {"rss_mib":round(counters.WorkingSetSize/2**20,2),"private_mib":round(counters.PrivateUsage/2**20,2),"page_faults":int(counters.PageFaultCount)}


def file_sha256(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def recovery_paths(config: dict[str,Any]) -> dict[str,Path]:
    root=Path(config["outputs"]["root"])/"round1"; root.mkdir(parents=True,exist_ok=True)
    return {
        "final":root/"chunk_embeddings.f32",
        "partial":root/"chunk_embeddings.streaming.partial",
        "status":root/"chunk_embeddings.streaming.status.u8",
        "checkpoint":root/"chunk_embeddings.streaming.checkpoint.json",
    }


def _atomic_json(path: Path,value: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,path)


def prepare_streaming_cache(config: dict[str,Any]) -> dict[str,Any]:
    from src.source_census.phase10e import phase_contract
    _,phase,_,_,experiment=phase_contract(config); paths=recovery_paths(config)
    chunks_path=Path(config["outputs"]["chunks"]); old_chunks_path=Path(phase["previous_chunks"])
    dimension=int(config["models"]["embedder"]["dimension"])
    rows=int(pq.ParquetFile(chunks_path).metadata.num_rows); old_rows=int(pq.ParquetFile(old_chunks_path).metadata.num_rows)
    signature={"experiment":experiment,"chunks_sha256":file_sha256(chunks_path),"old_chunks_sha256":file_sha256(old_chunks_path),"rows":rows,"dimension":dimension,"dtype":"float32","model":config["models"]["embedder"]}
    if paths["checkpoint"].exists() and paths["partial"].exists() and paths["status"].exists():
        checkpoint=json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
        if checkpoint.get("signature")!=signature: raise ValueError("streaming embedding checkpoint signature mismatch")
        if paths["partial"].stat().st_size!=rows*dimension*4 or paths["status"].stat().st_size!=rows: raise ValueError("streaming embedding checkpoint size mismatch")
        return checkpoint
    started=time.perf_counter(); memory=[{"phase":"process_start",**process_memory()}]
    old_ids=[]
    for batch in pq.ParquetFile(old_chunks_path).iter_batches(batch_size=8192,columns=["chunk_id"]): old_ids.extend(str(value) for value in batch.column(0).to_pylist())
    old_lookup={chunk_id:index for index,chunk_id in enumerate(old_ids)}; del old_ids
    memory.append({"phase":"old_key_map",**process_memory()})
    old_vectors=np.memmap(phase["previous_embeddings"],dtype=np.float32,mode="r",shape=(old_rows,dimension))
    vectors=np.memmap(paths["partial"],dtype=np.float32,mode="w+",shape=(rows,dimension))
    status=np.memmap(paths["status"],dtype=np.uint8,mode="w+",shape=(rows,)); status[:]=0
    position=0; reused=0
    for batch in pq.ParquetFile(chunks_path).iter_batches(batch_size=2048,columns=["chunk_id"]):
        ids=[str(value) for value in batch.column(0).to_pylist()]
        for local,chunk_id in enumerate(ids):
            old_index=old_lookup.get(chunk_id)
            if old_index is not None:
                vectors[position+local]=old_vectors[old_index]; status[position+local]=1; reused+=1
        position+=len(ids)
    vectors.flush(); status.flush(); del vectors,status,old_vectors,old_lookup; gc.collect()
    checkpoint={"signature":signature,"prepared":True,"complete":False,"reused_rows":reused,"embedded_rows":0,"completed_rows":reused,"missing_rows":rows-reused,"last_scanned_row":0,"paths":{key:str(value) for key,value in paths.items()},"prepare_runtime_seconds":time.perf_counter()-started,"prepare_memory":memory+[{"phase":"prepare_complete",**process_memory()}]}
    _atomic_json(paths["checkpoint"],checkpoint); return checkpoint


def validate_profile(profile: dict[str,Any],expected_rows: int) -> dict[str,Any]:
    samples=[row for row in profile.get("checkpoints",[]) if float(row.get("rss_mib",-1))>=0 and float(row.get("private_mib",-1))>=0]
    tail=samples[len(samples)//2:]; tail_growth=(float(tail[-1]["private_mib"])-float(tail[0]["private_mib"])) if len(tail)>=2 else float("inf")
    loaded=next((float(row["private_mib"]) for row in profile.get("measurements",[]) if row.get("phase")=="model_loaded"),0.0)
    # Streaming retains no row data between checkpoints. Treat the measured
    # second-half growth as cache warm-up only when it is below one percent of
    # the measured loaded-model commitment; reserve one more such interval.
    warmup_limit=loaded*0.01; stable=len(samples)>=8 and tail_growth<=warmup_limit
    projected_private=max((float(row["private_mib"]) for row in samples),default=-1.0)+max(0.0,tail_growth)
    return {"passed":bool(profile.get("completed")) and stable,"measured_rows":int(profile.get("rows",0)),"checkpoint_count":len(samples),"tail_private_growth_mib":tail_growth,"measured_model_private_mib":loaded,"derived_warmup_limit_mib":warmup_limit,"observed_peak_rss_mib":max((float(row["rss_mib"]) for row in samples),default=-1.0),"observed_peak_private_mib":max((float(row["private_mib"]) for row in samples),default=-1.0),"projected_peak_private_mib":projected_private,"expected_rows":expected_rows,"rule":"at least 8 checkpoints; second-half private-memory growth must remain below 1% of the measured loaded-model commitment; projected peak reserves one additional measured tail-growth interval"}

