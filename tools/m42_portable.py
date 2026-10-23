"""Private, weights-free Windows execution bundle for the frozen M4.2 pilot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PACKAGES=('torch','sentence-transformers','transformers','numpy','pyarrow','psutil',
          'PyYAML','huggingface-hub','tokenizers','safetensors')
MODELS={'BAAI/bge-m3':'5617a9f61b028005a4858fdac845db406aefb181',
        'BAAI/bge-reranker-v2-m3':'953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'}


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def safe_path(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('unsafe bundle path')
    result=(root/Path(*p.parts)).resolve()
    if not result.is_relative_to(root.resolve()): raise ValueError('bundle path escaped root')
    return result


def cache_root(root):
    return root/'.cache/huggingface'


def verify(root, *, runtime=True, models=True):
    manifest=json.loads((root/'portable_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        if file_hash(safe_path(root,name))!=expected: raise ValueError(f'bundle checksum mismatch: {name}')
    if runtime:
        if platform.system()!='Windows' or platform.python_version_tuple()[:2]!=tuple(manifest['python_major_minor']):
            raise ValueError('this bundle is pinned for Windows and the exported Python major/minor')
        for name,version in manifest['packages'].items():
            if importlib.metadata.version(name)!=version: raise ValueError(f'runtime package mismatch: {name}')
    if models:
        for repo,detail in manifest['models'].items():
            snapshot=cache_root(root)/('models--'+repo.replace('/','--'))/'snapshots'/detail['revision']
            for name,expected in detail['files'].items():
                pure=PurePosixPath(name)
                if pure.is_absolute() or '..' in pure.parts or '\\' in name or ':' in name:
                    raise ValueError('unsafe model path')
                model_file=(snapshot/Path(*pure.parts)).resolve()
                # Normal HF snapshots symlink to the cache's sibling blobs directory.
                if not model_file.is_relative_to(cache_root(root).resolve()): raise ValueError('model escaped cache')
                if file_hash(model_file)!=expected: raise ValueError(f'model/tokenizer checksum mismatch: {repo}/{name}')
    return {'status':'VERIFIED','files':len(manifest['files']),'models_checked':models,
            'runtime_checked':runtime,'score_reuse':'DISABLED_PENDING_TARGET_HOST_19_REFERENCE_EQUIVALENCE',
            'manifest_sha256':file_hash(root/'portable_manifest.json')}


def export(root,destination):
    from src.multilingual_retrieval.m42_execution import validate_manifest,PROFILE,EQUIVALENCE
    frozen=validate_manifest(root)
    files={name.replace('\\','/') for name in frozen['inputs_sha256']}
    files.update(('tools/m42_portable.py','tools/multilingual_m42.py',
        'src/indexing/embedder.py','src/reranking/equivalence.py','src/retrieval/pubmed_lexical.py',
        'src/runtime/state_file.py','tools/competition_supervisor.py',
        PROFILE,EQUIVALENCE,
        'artifacts/multilingual_retrieval_m42/frozen_manifest.json',
        'artifacts/multilingual_retrieval_m42/tokenizer_contract.json',
        'artifacts/multilingual_retrieval_m42/exact_reference_verification.csv',
        'artifacts/multilingual_retrieval_m4/candidate_trace_top100.parquet',
        'artifacts/multilingual_retrieval_m41/partial_retention.csv',
        'artifacts/multilingual_retrieval_m41/candidate_failure_taxonomy.json',
        'artifacts/multilingual_retrieval_m41/SEMANTIC_PILOT_PLAN.md',
        'artifacts/multilingual_retrieval_m42/PORTABLE_EXECUTION.md'))
    files.update(str(p.relative_to(root)).replace('\\','/') for p in (root/'src/multilingual_retrieval').glob('*.py'))
    # Only explicit fixed-development inputs: no held-out review, credentials,
    # frozen submission ZIP, whole-corpus index, model weights or score DB.
    versions={name:importlib.metadata.version(name) for name in PACKAGES}
    models={}
    for repo,revision in MODELS.items():
        rel=Path('models--'+repo.replace('/','--'))/'snapshots'/revision
        candidates=[root/'.cache/huggingface'/rel,Path.home()/'.cache/huggingface/hub'/rel]
        snapshot=next((p for p in candidates if p.exists()),None)
        if snapshot is None: raise ValueError(f'pinned local snapshot absent: {repo}')
        hashes={str(p.relative_to(snapshot)).replace('\\','/'):file_hash(p)
                for p in sorted(snapshot.rglob('*')) if p.is_file()}
        if not hashes or not any(n.endswith(('.bin','.safetensors')) for n in hashes):
            raise ValueError(f'no pinned weight evidence: {repo}')
        models[repo]={'revision':revision,'files':hashes}
    manifest={'version':'m42-windows-portable-v1','source_commit':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_code_state':'Exact bundled file hashes are authoritative; includes scoped working-tree implementation',
        'files':{name:file_hash(root/name) for name in sorted(files)},
        'python_major_minor':list(platform.python_version_tuple()[:2]),'packages':versions,'models':models,
        'trial20_sha256':frozen['trial20_sha256'],
        'scope':'12 frozen development queries; no held-out or new source acquisition',
        'embedding_reuse':'No source-machine M4.2 vectors exist; recompute target vectors after its trial PASS',
        'reranker_reuse':'19 reference scores are evidence only until exact target-host replay PASS; no scores imported automatically'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): raise ValueError('portable archive already exists; not overwritten')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files): archive.write(root/name,name)
        archive.writestr('portable_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        # CUDA wheel source must provide the exact captured local build; no silent CPU substitution.
        archive.writestr('requirements.runtime.txt','\n'.join(f'{n}=={v}' for n,v in versions.items())+'\n')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None: raise ValueError('bundle ZIP CRC failed')
        for name in archive.namelist(): safe_path(root,name)
    report={'status':'PORTABLE_BUNDLE_CREATED_CRC_PASS','path':str(destination),'bytes':destination.stat().st_size,
            'sha256':file_hash(destination),'files':len(files),'weights_included':False,
            'target_execution':'NOT_RUN_NO_REMOTE_SESSION','score_reuse_enabled':False}
    (destination.parent/'bundle_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['export','verify-files','verify','download-models','trial20','dense'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/multilingual_retrieval_m42/portable/m42_rtx3060.zip')
    parser.add_argument('--state',type=Path)
    args=parser.parse_args()
    if args.state:
        if not args.state.resolve().is_relative_to(ROOT): raise ValueError('state must stay within extracted bundle')
        os.environ['M42_RUN_STATE']=str(args.state.resolve())
    if args.command=='export': return export(ROOT,args.output)
    if args.command=='verify-files': return verify(ROOT,runtime=False,models=False)
    os.environ['HF_HUB_CACHE']=str(cache_root(ROOT))
    if args.command=='download-models':
        verify(ROOT,models=False)
        from huggingface_hub import snapshot_download
        manifest=json.loads((ROOT/'portable_manifest.json').read_text(encoding='utf-8'))
        for repo,detail in manifest['models'].items():
            snapshot_download(repo,revision=detail['revision'],cache_dir=cache_root(ROOT),allow_patterns=list(detail['files']))
        return verify(ROOT)
    verified=verify(ROOT)
    if args.command=='verify': return verified
    # No environment token is printed or copied. All inference is local/offline.
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.command=='trial20':
        from src.multilingual_retrieval.m42_execution import trial
        result=trial(ROOT)
        if result.get('passed') is not True: raise RuntimeError(f'trial stopped: {result}')
        return result
    from src.multilingual_retrieval.m42_dense import run
    return run(ROOT)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        print(json.dumps(main(),ensure_ascii=False,indent=2))
    except Exception as exc:
        if os.environ.get('M42_RUN_STATE'):
            import datetime
            state=Path(os.environ['M42_RUN_STATE']); state.parent.mkdir(parents=True,exist_ok=True)
            value={'status':'NEEDS_AGENT','stage':'STOPPED_SAFELY','error':f'{type(exc).__name__}: {exc}',
                   'heartbeat_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            temporary=state.with_suffix('.tmp'); temporary.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8'); temporary.replace(state)
        raise
