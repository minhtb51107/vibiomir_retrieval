"""Final read-only inventory and local recovery snapshots. No deletion/move/crawl.

Backups remain unapproved until an independent destination is confirmed and
every copied file is hash-verified. Local snapshots are NOT independent backups.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final_freeze_20261027'
RUN=ROOT/'artifacts/runs/final_freeze_20261027'
sys.path.insert(0,str(ROOT))
from src.runtime.state_file import write_json, update_json

BEST='submissions/00_READY_TO_UPLOAD/p27_sdso20749_cap16.zip'
BEST_SHA='b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e'


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def classify(path):
    p=PurePosixPath(path);parts=p.parts;name=p.name.lower()
    if name in {'.env','.env.local','.env.production'} or name in {'credentials.json','secrets.json'}:
        return 'D','SENSITIVE_PRIVATE_REQUIRES_ENCRYPTED_BACKUP_REVIEW'
    if parts and parts[0]=='.git':return 'A','GIT_HISTORY_AND_LOCAL_REFS'
    if any(x in {'.venv','venv','node_modules','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','test_tmp','test_tmp_preserve'} for x in parts):
        return 'C','REGENERABLE_ENVIRONMENT_OR_TEST_BYTECODE_CACHE_AFTER_MANIFEST_BACKUP'
    if name.endswith(('.pyc','.pyo')) or name in {'.coverage','coverage.xml'}:
        return 'C','REGENERABLE_TEST_OR_BYTECODE_OUTPUT'
    if name.endswith(('.tmp','.part','.lock','-wal','-shm')):
        return 'D','POSSIBLE_CHECKPOINT_OR_INFLIGHT_STATE_DO_NOT_DELETE_BY_EXTENSION'
    if parts and parts[0] in {'data','artifacts','submissions','models','checkpoints'}:
        return 'B','SCIENTIFIC_DATA_PROVENANCE_SCORES_MODELS_OR_OUTPUT_ARCHIVE'
    if parts and parts[0] in {'src','tools','scripts','tests','configs','docs','requirements'}:
        return 'A','SOURCE_CONFIG_TESTS_OR_DOCUMENTATION'
    if len(parts)==1 and (p.suffix.lower() in {'.py','.md','.toml','.yaml','.yml','.txt','.ps1','.json'} or name in {'.gitignore','.gitattributes'}):
        return 'A','PROJECT_HARNESS_OR_DOCUMENTATION'
    return 'D','UNKNOWN_PRESERVE_UNTIL_REVIEW'


def git_output(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def cleanup_proposal(db, folders, root):
    """Only C-only directory trees qualify; links/unknown descendants protect parents."""
    protected=[row[0] for row in db.execute("SELECT path FROM files WHERE class!='C'")]
    roots=[]
    for name in sorted(folders,key=lambda p:(p.count('/'),p)):
        if classify(name)[0]!='C':continue
        if any(p==name or p.startswith(name+'/') for p in protected):continue
        if not any(name==r or name.startswith(r+'/') for r in roots):roots.append(name)
    candidates=[]
    for name in roots:
        candidates.append(dict(path=str(root/Path(name)),relative_path=name,kind='DIRECTORY',logical_bytes=folders[name][0],
            status='PROPOSED_ONLY_BACKUP_AND_EXPLICIT_APPROVAL_REQUIRED'))
    for name,size in db.execute("SELECT path,bytes FROM files WHERE class='C' ORDER BY path"):
        if not any(name==r or name.startswith(r+'/') for r in roots):
            candidates.append(dict(path=str(root/Path(name)),relative_path=name,kind='FILE',logical_bytes=size,
                status='PROPOSED_ONLY_BACKUP_AND_EXPLICIT_APPROVAL_REQUIRED'))
    return dict(targets=candidates,deleted=0,moved=0,logical_recoverable_bytes=sum(r['logical_bytes'] for r in candidates),
        disk_allocated_recoverable_bytes='UNKNOWN_LOGICAL_SIZES_ONLY',approval=False)


def inventory():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    if (OUT/'inventory.sqlite').exists():
        raise RuntimeError('Inventory already exists: preserved; use existing snapshot or explicit new version')
    started=time.perf_counter();start=stamp()
    tracked={x.decode('utf-8') for x in git_output('ls-files','-z','--cached').split(b'\0') if x}
    untracked={x.decode('utf-8') for x in git_output('ls-files','-z','--others','--exclude-standard').split(b'\0') if x}
    # Private recovery records, never published or interpreted as a verified backup.
    for name,args in [('worktree_before_finalization.patch',['diff','--binary','HEAD']),
                      ('index_before_finalization.patch',['diff','--cached','--binary']),
                      ('git_status_before_finalization.z',['status','--porcelain=v1','-z','--untracked-files=all'])]:
        with (OUT/name).open('wb') as stream:subprocess.run(['git',*args],cwd=ROOT,stdout=stream,check=True)
    db=sqlite3.connect(OUT/'inventory.sqlite');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA cache_size=-2048')
    db.execute('CREATE TABLE files(path TEXT PRIMARY KEY,bytes INTEGER,mtime_ns INTEGER,git_state TEXT,class TEXT,reason TEXT,link_target TEXT)')
    db.execute('CREATE TABLE errors(path TEXT,error TEXT)')
    folders=collections.defaultdict(lambda:[0,0]);counts=collections.Counter();totals=collections.Counter();states=collections.Counter()
    stack=[ROOT];files=0;links=0;errors=0
    excluded={OUT.resolve(),RUN.resolve()}
    with (OUT/'file_inventory.jsonl').open('w',encoding='utf-8') as listing:
        while stack:
            directory=stack.pop();relative_dir=directory.relative_to(ROOT).as_posix()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        path=Path(entry.path);relative=path.relative_to(ROOT).as_posix()
                        if path.resolve() in excluded:continue
                        try:
                            info=entry.stat(follow_symlinks=False)
                            reparse=entry.is_symlink() or bool(getattr(info,'st_file_attributes',0)&0x400)
                            if entry.is_dir(follow_symlinks=False) and not reparse:
                                folders.setdefault(relative,[0,0]);stack.append(path);continue
                            kind,reason=classify(relative)
                            target=os.readlink(path) if reparse else None
                            if reparse:kind,reason='D','REPARSE_LINK_NOT_TRAVERSED_SHARED_TARGET_UNKNOWN';links+=1
                            status='GIT_INTERNAL' if relative.startswith('.git/') else 'TRACKED' if relative in tracked else 'UNTRACKED' if relative in untracked else 'IGNORED'
                            record=dict(path=relative,absolute_path=str(path),bytes=info.st_size,mtime_ns=info.st_mtime_ns,
                                        git_state=status,category=kind,reason=reason,link_target=target)
                            listing.write(json.dumps(record,ensure_ascii=False)+'\n')
                            db.execute('INSERT INTO files VALUES(?,?,?,?,?,?,?)',(relative,info.st_size,info.st_mtime_ns,status,kind,reason,target))
                            folders[relative_dir][0]+=info.st_size;folders[relative_dir][1]+=1
                            counts[kind]+=1;totals[kind]+=info.st_size;states[status]+=1;files+=1
                            if files%2000==0:
                                db.commit();update_json(RUN/'state.json',dict(status='RUNNING',stage='FILE_INVENTORY',
                                    done=files,total=None,heartbeat_at=stamp(),checkpoint=str(OUT/'inventory.sqlite')))
                        except OSError as exc:
                            errors+=1;db.execute('INSERT INTO errors VALUES(?,?)',(relative,type(exc).__name__))
            except OSError as exc:
                errors+=1;db.execute('INSERT INTO errors VALUES(?,?)',(relative_dir,type(exc).__name__))
    db.commit()
    direct={k:list(v) for k,v in folders.items()}
    for name in sorted(folders,key=lambda p:p.count('/'),reverse=True):
        if name=='.':continue
        parent=PurePosixPath(name).parent.as_posix()
        folders[parent][0]+=folders[name][0];folders[parent][1]+=folders[name][1]
    with (OUT/'folder_inventory.jsonl').open('w',encoding='utf-8') as stream:
        for name,values in sorted(folders.items()):
            stream.write(json.dumps(dict(path=name,logical_bytes_recursive=values[0],files_recursive=values[1],
                direct_bytes=direct.get(name,[0,0])[0],direct_files=direct.get(name,[0,0])[1]))+'\n')
    # Exact non-overlapping C-only targets; never candidate datasets/checkpoints.
    proposal=cleanup_proposal(db,folders,ROOT);candidates=proposal['targets']
    write_json(OUT/'cleanup_candidates.json',proposal)
    critical={BEST:BEST_SHA,'data/raw/links_corpus_vi_en_zh.parquet':'9e34cb4b7d6424b80ed2634ada581ea9e2d97da69c245d80bdbb31413a78e8ce',
              'data/raw/query.parquet':'ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08'}
    hashes={}
    for name,expected in critical.items():
        actual=sha(ROOT/name);hashes[name]=dict(sha256=actual,expected=expected,passed=actual==expected)
    if not all(r['passed'] for r in hashes.values()):raise ValueError('critical immutable file hash mismatch')
    write_json(OUT/'critical_hashes.json',hashes)
    packages=sorted({(d.metadata.get('Name','UNKNOWN'),d.version) for d in importlib.metadata.distributions()})
    write_json(OUT/'installed_versions_private.json',dict(python=sys.version,packages=packages))
    check=db.execute('PRAGMA integrity_check').fetchone()[0];db.close()
    import psutil
    info=psutil.Process().memory_info()
    summary=dict(status='COMPLETE' if not errors else 'PARTIAL',snapshot_started=start,snapshot_completed=stamp(),
        root=str(ROOT),files=files,logical_bytes=sum(totals.values()),categories={k:dict(files=counts[k],bytes=totals[k]) for k in 'ABCD'},
        git_states=dict(states),top_level={k:dict(bytes=v[0],files=v[1]) for k,v in folders.items() if '/' not in k and k!='.'},
        inventory_errors=errors,reparse_links_not_traversed=links,sqlite_integrity=check,
        excluded_audit_directories=[str(x) for x in excluded],logical_not_allocated_sizes=True,
        cleanup_candidate_bytes=sum(r['logical_bytes'] for r in candidates),
        external_backup_created=False,external_backup_verified=False,destination_confirmed=False,
        runtime_seconds=time.perf_counter()-started,peak_working_set_mib=getattr(info,'peak_wset',info.rss)/2**20,
        note='Phase A inventory precedes documentation finalization; record later edits separately. No reparse targets traversed.')
    write_json(OUT/'inventory_summary.json',summary)
    update_json(RUN/'state.json',dict(status=summary['status'],stage='INVENTORY_COMPLETE',done=files,total=files,
        checkpoint=str(OUT/'inventory.sqlite'),heartbeat_at=stamp()))
    print(json.dumps(summary,ensure_ascii=False));return 0


def safe_relative(path):
    normal=path.replace('\\','/');p=PurePosixPath(normal)
    if p.is_absolute() or '..' in p.parts or ':' in normal or not normal:raise ValueError('unsafe archive path')
    return p


def verify_manifest(manifest,mirror):
    mirror=Path(mirror).resolve();checked=0;total=0;seen=set()
    with Path(manifest).open(encoding='utf-8') as stream:
        for line in stream:
            record=json.loads(line);relative=safe_relative(record['path']);path=mirror/Path(relative)
            if relative in seen:raise ValueError('duplicate manifest path')
            seen.add(relative)
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(mirror):
                raise ValueError('missing/unsafe backup file: '+record['path'])
            if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
                raise ValueError('backup checksum mismatch: '+record['path'])
            checked+=1;total+=record['bytes']
    if not checked:raise ValueError('empty verification manifest')
    return dict(files_verified=checked,bytes_verified=total,source_destination_deletion_authorized=False,
                note='Manifest-listed files verified only; separately inspect extras, archive completeness and restore tests')


def hash_source_manifest(destination):
    """Explicit future operation: hash stable sources, never copy or delete them."""
    destination=Path(destination).resolve()
    if destination.is_relative_to(ROOT) or destination.exists():
        raise ValueError('new manifest must be outside source tree; existing files are never overwritten')
    if not destination.parent.is_dir():raise ValueError('confirmed manifest directory must already exist')
    count=total=0;pending=[ROOT]
    with destination.open('x',encoding='utf-8') as output:
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    path=Path(entry.path);before=entry.stat(follow_symlinks=False)
                    if entry.is_symlink() or getattr(before,'st_file_attributes',0)&0x400:
                        raise ValueError('reparse links require explicit separate archival review: '+str(path))
                    if entry.is_dir(follow_symlinks=False):pending.append(path);continue
                    digest=sha(path);after=path.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                        raise ValueError('source changed during hashing: '+str(path))
                    output.write(json.dumps(dict(path=path.relative_to(ROOT).as_posix(),bytes=after.st_size,sha256=digest))+'\n')
                    count+=1;total+=after.st_size
    return dict(files=count,bytes=total,manifest_sha256=sha(destination),
                source_quiescence_required=True,backup_created=False,deletion_authorized=False)


def local_snapshot():
    OUT.mkdir(parents=True,exist_ok=True)
    head=git_output('rev-parse','HEAD').decode().strip();bundle=OUT/f'git_history_{head[:12]}.bundle'
    if not bundle.exists():subprocess.run(['git','-c','pack.threads=1','-c','pack.windowMemory=16m',
        '-c','pack.deltaCacheSize=16m','bundle','create',str(bundle),'--all'],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    names={x.decode('utf-8') for x in git_output('ls-files','-z','--cached','--others','--exclude-standard').split(b'\0') if x}
    archive=OUT/f'local_working_source_{head[:12]}.tar'
    if archive.exists():raise ValueError('local source snapshot already exists; preserved without overwrite')
    count=0
    with tarfile.open(archive,'w',dereference=False) as tar:
        for name in sorted(names):
            path=ROOT/Path(name)
            category,reason=classify(name)
            if category!='A' or name.startswith('.git/') or not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):continue
            tar.add(path,arcname=name,recursive=False);count+=1
    with tarfile.open(archive,'r') as tar:
        for member in tar:
            relative=safe_relative(member.name)
            if not member.isfile():raise ValueError('unexpected non-file source snapshot member')
            payload=tar.extractfile(member);actual=hashlib.sha256()
            for block in iter(lambda:payload.read(65536),b''):actual.update(block)
            if actual.hexdigest()!=sha(ROOT/Path(relative)):raise ValueError('source snapshot roundtrip mismatch')
    result=dict(status='LOCAL_RECOVERY_SNAPSHOT_ONLY',head=head,bundle=str(bundle),bundle_sha256=sha(bundle),
        source_archive=str(archive),source_archive_sha256=sha(archive),source_files=count,
        source_tar_roundtrip_verified=True,git_bundle_reachable_refs_only=True,
        external_backup=False,physical_disk_independent=False,
        warning='No dataset backup. Recognized sensitive names and unknown-category files are excluded; this is not a comprehensive secret scan. Private recovery archive only.')
    write_json(OUT/f'local_snapshot_{head[:12]}.json',result);print(json.dumps(result));return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['inventory','snapshot','verify','hash-source','supervise'])
    parser.add_argument('--manifest');parser.add_argument('--mirror')
    parser.add_argument('--confirmed-destination',action='store_true');args=parser.parse_args()
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    if args.mode=='inventory':return inventory()
    if args.mode=='snapshot':return local_snapshot()
    if args.mode=='hash-source':
        if not args.manifest or not args.confirmed_destination:parser.error('explicit confirmed destination/manifest required')
        print(json.dumps(hash_source_manifest(args.manifest)));return 0
    if args.mode=='verify':
        if not args.manifest or not args.mirror:parser.error('explicit SHA manifest and confirmed mirror path required')
        print(json.dumps(verify_manifest(args.manifest,args.mirror)));return 0
    from tools import competition_supervisor as supervisor
    (RUN/'logs').mkdir(parents=True,exist_ok=True)
    sys.stdout=(RUN/'logs/supervisor.log').open('a',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    supervisor.prevent_sleep=lambda active:None
    supervisor.SUCCESS_TERMINAL_STATES.update({'COMPLETE','PARTIAL'})
    sys.argv=[str(Path(__file__)),'--run-id','final_freeze_20261027','--minimum-free-gib','20','--',
              sys.executable,'-X','utf8',str(Path(__file__)),'inventory']
    return supervisor.main()


if __name__=='__main__':raise SystemExit(main())
