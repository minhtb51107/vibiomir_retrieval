#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re,shutil,sqlite3,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.source_census.pipeline import atomic_json,load_config
from src.source_census.phase10e import phase_contract
def slug(s): return re.sub(r'[^a-z0-9]+','_',s.lower()).strip('_')
def count(p):
 if not p.exists(): return 0
 try:
  c=sqlite3.connect(f'file:{p.resolve()}?mode=ro',uri=True,timeout=1); n=int(c.execute('select count(*) from crawl_results').fetchone()[0]); c.close(); return n
 except sqlite3.Error:return 0
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-state',required=True);a=p.parse_args();cfg=load_config(a.config); out=cfg['outputs']; _,phase,sources,_,_=phase_contract(cfg); man=json.loads(Path(phase['manifest']).read_text(encoding='utf-8')); total=man['total_incremental_ids']; root=Path(out['worker_root']);root.mkdir(parents=True,exist_ok=True);logs=Path(a.run_state).parent/'logs/sources';logs.mkdir(parents=True,exist_ok=True);jobs=[]
 for source,row in sorted(man['sources'].items()):
  r=root/slug(source);r.mkdir(parents=True,exist_ok=True);m=r/'manifest.json';atomic_json(m,{'domain':source,'records':row['incremental_records']});jobs.append({'source':source,'root':r,'manifest':m,'selected':len(row['incremental_records'])})
 pending=[j for j in jobs if count(j['root']/'crawl.sqlite')<j['selected']]; active={};complete={j['source'] for j in jobs if count(j['root']/'crawl.sqlite')>=j['selected']};failed={};retries={};start=time.monotonic()
 def launch(j):
  log=(logs/f"{slug(j['source'])}.log").open('a',encoding='utf-8');cmd=[sys.executable,'scripts/crawl_corpus.py','--config',cfg['acquisition']['crawler_config'],'--ids-file',str(j['manifest']),'--full','--output-db',str(j['root']/'crawl.sqlite'),'--summary-out',str(j['root']/'summary.json'),'--body-archive-dir',str(j['root']/'bodies'),'--archive-documents-per-shard',str(cfg['acquisition']['body_documents_per_shard'])];active[j['source']]=(subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log,j)
 limit=min(int(cfg['acquisition'].get('global_concurrency',3)),len(sources))
 while pending or active:
  while pending and len(active)<limit:launch(pending.pop(0))
  time.sleep(2)
  for s,(proc,log,j) in list(active.items()):
   code=proc.poll()
   if code is None:continue
   log.close();del active[s]
   if code==0:complete.add(s)
   else:
    retries[s]=retries.get(s,0)+1
    if retries[s]<=2:pending.append(j)
    else:failed[s]=code
  done=sum(count(j['root']/'crawl.sqlite') for j in jobs);mins=max((time.monotonic()-start)/60,1e-6);rate=done/mins;atomic_json(a.run_state,{'stage':'ACQUIRE','status':'RUNNING','selected':total,'completed':done,'remaining':total-done,'active_sources':sorted(active),'active_source_count':len(active),'completed_sources':len(complete),'total_sources':len(sources),'failed_sources':failed,'retry_history':retries,'urls_per_minute':round(rate,3),'eta_minutes':round((total-done)/rate,2) if rate else None,'free_disk_gib':round(shutil.disk_usage(ROOT).free/2**30,3),'last_successful_checkpoint':done})
  if shutil.disk_usage(ROOT).free/2**30<20:
   for proc,_,_ in active.values():proc.terminate()
   raise RuntimeError('20 GiB Phase10E disk floor reached')
 result={'selected':total,'completed':sum(count(j['root']/'crawl.sqlite') for j in jobs),'failed_sources':failed,'retry_counts':retries,'maximum_concurrent_source_workers':limit};atomic_json(Path(out['artifacts'])/'parallel_acquisition.json',result);return 0
if __name__=='__main__':raise SystemExit(main())
