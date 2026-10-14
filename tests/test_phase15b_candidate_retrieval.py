import datetime as dt
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

from src.retrieval.vietnamese_lexical import add_documents, content_hash, open_index, search
from src.runtime.progress_contract import StageProgress, progress_snapshot
from tools.phase15b_candidate_retrieval import STAGES, stratified_queries


def test_vietnamese_fts_is_deterministic_and_resume_safe(tmp_path: Path):
    db=tmp_path/"index.sqlite"; c=open_index(db)
    rows=[{"doc_id":1,"official_url":"https://x/1","canonical_domain":"x","title":"Viêm dạ dày",
           "body":"Điều trị viêm dạ dày", "content_hash":content_hash("Viêm dạ dày","Điều trị viêm dạ dày"),"source_reference":"p"},
          {"doc_id":2,"official_url":"https://x/2","canonical_domain":"x","title":"Đau đầu",
           "body":"Nguyên nhân đau đầu", "content_hash":content_hash("Đau đầu","Nguyên nhân đau đầu"),"source_reference":"p"}]
    assert add_documents(c,rows)==2
    assert add_documents(c,rows)==0
    one=search(c,"viêm dạ dày",2); two=search(c,"viêm dạ dày",2)
    assert one==two and one[0]["doc_id"]==1
    assert c.execute("PRAGMA integrity_check").fetchone()[0]=="ok"


def test_progress_contract_covers_every_phase15b_stage():
    now=dt.datetime.now(dt.timezone.utc)
    for index,stage in enumerate(STAGES):
        value=progress_snapshot(StageProgress(stage,"op",1,2,now,now,True),ordered_stages=STAGES)
        assert value["stage_percent"]==50.0 and value["overall_percent"]<100
        if index: assert value["overall_percent"]>0


def test_stratified_sample_is_exact_and_deterministic(tmp_path: Path):
    import pyarrow as pa, pyarrow.parquet as pq
    path=tmp_path/"q.parquet"
    texts=["Đau đầu và sốt?","Thuốc điều trị bệnh gì?","Phẫu thuật ung thư?","Câu hỏi sức khỏe khác?"]
    pq.write_table(pa.Table.from_pylist([{"id":i+1,"query":texts[i%4]} for i in range(1200)]),path)
    a=stratified_queries(path,"seed",120); b=stratified_queries(path,"seed",120)
    assert a==b and len(a)==120 and len({r["query_id"] for r in a})==120
    assert {r["topic_proxy"] for r in a} >= {"medication","treatment","symptom","other"}


def test_powershell_monitor_accepts_terminal_progress_contract(tmp_path: Path):
    shell=shutil.which("pwsh") or shutil.which("powershell")
    if not shell: return
    state=tmp_path/"state.json"; state.write_text(json.dumps({"status":"READY_FOR_NEXT_STAGE","stage":"PACKAGING","overall_percent":100.0,
        "stage_percent":100.0,"current_operation":"NOT_APPLICABLE","done":1,"total":1}),encoding="utf-8")
    result=subprocess.run([shell,"-NoProfile","-File",str(Path("tools/watch_run_progress.ps1").resolve()),"-StatePath",str(state),"-RefreshSeconds","1"],capture_output=True,text=True,timeout=15)
    assert result.returncode==0
    assert "READY_FOR_NEXT_STAGE" in result.stdout and '"overall_percent": 100.0' in result.stdout
