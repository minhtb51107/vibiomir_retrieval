"""Executable frozen semantic trial. No ML runtime imports before resource gates."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil

from src.multilingual_retrieval.m42 import EMBEDDING, digest, require_idle
from src.multilingual_retrieval.m2 import sha256_file
from src.multilingual_retrieval.m3 import _atomic_json

PROFILE = 'artifacts/source_census/phase10e_g1a_10010/embedding_memory_profile/profile.json'
EQUIVALENCE = 'artifacts/source_census/phase10e_g1a_10010/embedding_equivalence.json'


def commit_headroom():
    if os.name != 'nt':
        return (psutil.virtual_memory().available + psutil.swap_memory().free)/2**20
    class Memory(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [(n,ctypes.c_ulonglong) for n in
            ('total','available','commit_limit','commit_free','virtual_total','virtual_free','extended')]
    state = Memory(); state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise OSError('GlobalMemoryStatusEx failed')
    return state.commit_free / 2**20


def measured_limits(profile):
    rows = {r['phase']:r for r in profile['measurements']}
    start, before, peak = (rows[k] for k in ('process_start','before_model_load','first_batch_written'))
    # Keep original safety factors. Include runtime import cost omitted by warm-load gate.
    return {'required_cold_ram_mib':(peak['rss_mib']-start['rss_mib'])*1.25,
            'required_warm_ram_mib':(peak['rss_mib']-before['rss_mib'])*1.25,
            'required_commit_mib':(peak['private_mib']-start['private_mib'])*1.10,
            'required_vram_mib':(peak['gpu_allocated_mib']-before['gpu_allocated_mib'])*1.25}


def preflight(root):
    idle = require_idle(root)
    limits = measured_limits(json.loads((root/PROFILE).read_text(encoding='utf-8')))
    gpu = subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True,timeout=10).stdout.strip().splitlines()
    if len(gpu)!=1: raise ValueError('expected one pinned CUDA device')
    free, used = map(float,gpu[0].split(','))
    telemetry = {'available_ram_mib':psutil.virtual_memory().available/2**20,
                 'commit_headroom_mib':commit_headroom(), 'gpu_free_mib':free,'gpu_used_mib':used,
                 'disk_free_gib':shutil.disk_usage(root).free/2**30}
    failures = []
    for key, requirement in (('available_ram_mib','required_cold_ram_mib'),
                             ('commit_headroom_mib','required_commit_mib'),('gpu_free_mib','required_vram_mib')):
        if telemetry[key] < limits[requirement]: failures.append(key)
    if telemetry['disk_free_gib'] < 20: failures.append('disk_free_gib')
    # No concurrent CUDA owner is expected for this isolated pilot.
    competitors=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=' '.join(p.info['cmdline'] or []).lower()
            if p.pid!=os.getpid() and any(x in cmd for x in ('llama-server','ollama serve','phase14_embed.py','score_source_census_persistent.py')):
                competitors.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    if competitors: failures.append('active_model_processes')
    return {'status':'PASS' if not failures else 'BLOCKED_RESOURCES', 'failures':failures,
            'model_loaded':False,'phase27_pids':idle['phase27_pids'],'competitor_pids':competitors,
            **telemetry,**limits,'profile_sha256':sha256_file(root/PROFILE),
            'basis':'Historical process-start to first-batch deltas; original 1.25 physical/VRAM and 1.10 commit multipliers'}


def validate_manifest(root):
    out=root/'artifacts/multilingual_retrieval_m42'
    manifest=json.loads((out/'frozen_manifest.json').read_text(encoding='utf-8'))
    if manifest['embedding'] != EMBEDDING or digest(manifest['trial20']) != manifest['trial20_sha256']:
        raise ValueError('frozen trial/model contract changed')
    for relative, expected in manifest['inputs_sha256'].items():
        if sha256_file(root/relative)!=expected: raise ValueError(f'frozen input changed: {relative}')
    if len(manifest['trial20']) != 20: raise ValueError('trial cardinality')
    return manifest


def trial(root):
    import numpy as np
    out=root/'artifacts/multilingual_retrieval_m42'
    manifest=validate_manifest(root)
    gate=preflight(root)
    _atomic_json(out/'execution_resource_preflight.json',gate)
    if gate['status']!='PASS': return gate
    historical=json.loads((root/EQUIVALENCE).read_text(encoding='utf-8'))
    expected={'model':EMBEDDING['name'],'revision':EMBEDDING['revision'],'dimension':1024,
              'batch_size':4,'max_length':512,'normalization':True,'precision':'FP16 CUDA'}
    if historical.get('passed') is not True or historical['contract']!=expected:
        raise ValueError('historical exact contract equivalence failed')
    # Runtime import and FP16-at-load match tools/phase14_embed.py; no FP32 GPU load.
    started=time.perf_counter()
    import torch
    from sentence_transformers import SentenceTransformer
    runtime_seconds=time.perf_counter()-started
    if psutil.virtual_memory().available/2**20 < gate['required_warm_ram_mib']:
        raise RuntimeError('RAM gate failed after runtime import; model not loaded')
    measurements=[]; stop=threading.Event()
    def sample():
        process=psutil.Process()
        while not stop.is_set():
            info=process.memory_info()
            measurements.append({'rss_mib':info.rss/2**20,'private_mib':getattr(info,'private',info.vms)/2**20,
                'available_mib':psutil.virtual_memory().available/2**20,'commit_headroom_mib':commit_headroom(),
                'gpu_allocated_mib':torch.cuda.memory_allocated()/2**20,
                'gpu_reserved_mib':torch.cuda.memory_reserved()/2**20})
            stop.wait(.25)
    monitor=threading.Thread(target=sample,daemon=True); monitor.start()
    try:
        torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
        started=time.perf_counter()
        model=SentenceTransformer(EMBEDDING['name'],revision=EMBEDDING['revision'],device='cuda',
                                  model_kwargs={'torch_dtype':torch.float16},local_files_only=True)
        model.max_seq_length=512
        load_seconds=time.perf_counter()-started
        from src.indexing.embedder import normalize_text
        texts=[normalize_text(r['text']) for r in manifest['trial20']]
        def encode(values):
            # Recheck resources at every permitted four-text batch; no automatic retry.
            batches=[]
            for i in range(0,len(values),4):
                if psutil.virtual_memory().available/2**20 < gate['required_warm_ram_mib']:
                    raise RuntimeError('memory pressure during semantic trial; stop without retry')
                a=model.encode(values[i:i+4],batch_size=4,show_progress_bar=False,
                               convert_to_numpy=True,normalize_embeddings=True)
                batches.append(np.asarray(a,dtype=np.float32))
            return np.concatenate(batches)
        started=time.perf_counter(); vectors=encode(texts); inference_seconds=time.perf_counter()-started
        # Reverse order changes batch context, not the pinned batch size or precision.
        equivalence_started=time.perf_counter()
        control=encode(texts[::-1])[::-1]
        equivalence_seconds=time.perf_counter()-equivalence_started
        maximum=float(np.max(np.abs(vectors-control)))
        cos=np.sum(vectors*control,axis=1)/(np.linalg.norm(vectors,axis=1)*np.linalg.norm(control,axis=1))
        passed=(vectors.shape==(20,1024) and np.isfinite(vectors).all() and
                np.max(np.abs(np.linalg.norm(vectors,axis=1)-1)) <= .001 and
                maximum<=historical['absolute_tolerance'] and float(cos.min())>=historical['minimum_cosine_required'])
        report={'status':'PASS' if passed else 'FAILED_EQUIVALENCE','passed':bool(passed),
                'trial20_sha256':manifest['trial20_sha256'],'runtime_import_seconds':runtime_seconds,
                'model_load_seconds':load_seconds,'inference_seconds':inference_seconds,
                'equivalence_seconds':equivalence_seconds,
                'tokens_trial20':sum(len(model.tokenizer.encode(t,truncation=True,max_length=512)) for t in texts),
                'tokens_including_equivalence':2*sum(len(model.tokenizer.encode(t,truncation=True,max_length=512)) for t in texts),
                'shape':list(vectors.shape),'maximum_absolute_difference':maximum,'minimum_cosine':float(cos.min()),
                'equivalence_kind':'Pinned local FP16 batch-context consistency; not remote-vector equivalence',
                'historical_equivalence_sha256':sha256_file(root/EQUIVALENCE),'resource_gate':gate}
    finally:
        stop.set(); monitor.join(timeout=2)
    report['telemetry']={'peak_'+k:max(r[k] for r in measurements) for k in
                         ('rss_mib','private_mib','gpu_allocated_mib','gpu_reserved_mib')}
    report['telemetry'].update({'minimum_'+k:min(r[k] for r in measurements) for k in ('available_mib','commit_headroom_mib')})
    _atomic_json(out/'trial20_execution.json',report)
    if passed: np.save(out/'trial20_vectors.npy',vectors)
    return report
