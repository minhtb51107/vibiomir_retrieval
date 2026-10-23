"""Cached, bounded pre-submission audit; never changes frozen submissions."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.multilingual_retrieval.m2 import sha256_file


def records(reader):
    decoder=json.JSONDecoder(); buffer=""; eof=False
    while not buffer.strip():
        buffer=reader.read(2**20)
        if not buffer:
            raise ValueError("empty submission")
    if not buffer.lstrip().startswith("["):
        raise ValueError("expected submission array")
    buffer=buffer.lstrip()[1:]
    while True:
        buffer=buffer.lstrip()
        if buffer.startswith("]"):
            return
        try:
            value,end=decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if eof: raise
            block=reader.read(2**20); eof=not block; buffer+=block
            continue
        yield value
        buffer=buffer[end:].lstrip()
        if buffer.startswith(","): buffer=buffer[1:]


def chunk_key(row):
    return int(row["doc_id"]),row["chunk_text"]


def ngrams(text):
    words=text.casefold().split()
    return set(zip(*(words[i:] for i in range(5)))) if len(words)>=5 else set()


def redundancy(chunks):
    grams=[ngrams(r["chunk_text"]) for r in chunks]
    count=0
    for i in range(len(chunks)):
        if any(grams[i] and grams[j] and len(grams[i]&grams[j])/min(len(grams[i]),len(grams[j]))>=0.8 for j in range(i)):
            count+=1
    return count


def main():
    out=ROOT/"artifacts/phase27_final_relevance_check"; out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((ROOT/"artifacts/phase27_suckhoedoisong_remaining_20749/PHASE25_FINAL_AUDIT.json").read_text(encoding="utf-8"))
    package=audit["cap16"]["package"]
    baseline=ROOT/"submissions/00_READY_TO_UPLOAD/p29_bachmai2679_cap16.zip"
    challenger=ROOT/package["zip_path"]
    expected="e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04"
    if sha256_file(baseline)!=expected or sha256_file(challenger)!=package["zip_sha256"]:
        raise ValueError("frozen ZIP hash mismatch")
    devrows=ds.dataset(ROOT/"artifacts/multilingual_retrieval_m1/review_queue.parquet").to_table(
        columns=["query_id","query_text"],filter=ds.field("split")=="development").to_pylist()
    development={int(r["query_id"]):r["query_text"] for r in devrows}
    manifest=json.loads((ROOT/"artifacts/phase27_suckhoedoisong_remaining_20749/suckhoedoisong_remaining_20749_manifest.json").read_text(encoding="utf-8"))
    newids={int(r["doc_id"]):r["original_url"] for r in manifest["records"]}
    totals=Counter(); eligible={}; compact={}
    with zipfile.ZipFile(baseline) as za,zipfile.ZipFile(challenger) as zb:
        with io.TextIOWrapper(za.open(za.namelist()[0]),encoding="utf-8") as fa,io.TextIOWrapper(zb.open(zb.namelist()[0]),encoding="utf-8") as fb:
            for a,b in zip(records(fa),records(fb),strict=True):
                qid=int(a["id"])
                if qid!=int(b["id"]): raise ValueError("query alignment mismatch")
                ad,bd=a["relevant_docs"],b["relevant_docs"]
                ac,bc=a["relevant_chunks"],b["relevant_chunks"]
                changed_docs=ad!=bd; changed_chunks=ac!=bc
                totals["queries"]+=1; totals["docs_changed_queries"]+=changed_docs; totals["chunks_changed_queries"]+=changed_chunks
                totals["both_changed_queries"]+=changed_docs and changed_chunks
                totals["doc_position_differences"]+=sum(x!=y for x,y in zip(ad,bd))
                totals["chunk_position_differences"]+=sum(x!=y for x,y in zip(ac,bc))
                added=[int(doc) for doc in bd if doc not in ad]
                totals["inserted_doc_slots"]+=len(added)
                totals["inserted_new_tranche_doc_slots"]+=sum(doc in newids for doc in added)
                aset={chunk_key(r) for r in ac}
                totals["inserted_chunk_objects"]+=sum(chunk_key(r) not in aset for r in bc)
                totals["new_tranche_chunk_slots"]+=sum(int(r["doc_id"]) in newids for r in bc)
                compact[qid]={"a_docs":ad,"b_docs":bd}
                if qid in development and (changed_docs or changed_chunks):
                    stratum="NEW_TRANCHE_DOCUMENT" if any(doc in newids for doc in added) else "OTHER_CHANGED_OUTPUT"
                    eligible[qid]={"query_id":qid,"query_vi":development[qid],"stratum":stratum,"phase29":a,"phase27":b}
    sample=[]
    for stratum in ("NEW_TRANCHE_DOCUMENT","OTHER_CHANGED_OUTPUT"):
        pool=[r for r in eligible.values() if r["stratum"]==stratum]
        pool.sort(key=lambda r:hashlib.sha256(f"p27-final-review-v1:{r['query_id']}".encode()).hexdigest())
        sample.extend(pool[:6])
    sample.sort(key=lambda r:r["query_id"])
    sample_ids=[r["query_id"] for r in sample]
    official={}
    needed={int(d) for r in sample for variant in ("phase29","phase27") for d in r[variant]["relevant_docs"]}
    needed|={int(c["doc_id"]) for r in sample for variant in ("phase29","phase27") for c in r[variant]["relevant_chunks"]}
    for batch in ds.dataset(ROOT/"data/raw/links_corpus_vi_en_zh.parquet").scanner(columns=["id","url"],filter=ds.field("id").isin(sorted(needed)),batch_size=512,use_threads=False).to_batches():
        for row in batch.to_pylist(): official[int(row["id"])]=row["url"]
    for r in sample:
        for variant in ("phase29","phase27"):
            rows=r[variant]["relevant_chunks"]
            r[variant]["near_duplicate_chunks_80pct_5gram"]=redundancy(rows)
        ad=set(r["phase29"]["relevant_docs"]); bd=set(r["phase27"]["relevant_docs"])
        ac={chunk_key(c) for c in r["phase29"]["relevant_chunks"]}; bc={chunk_key(c) for c in r["phase27"]["relevant_chunks"]}
        r["new_docs"]=[d for d in r["phase27"]["relevant_docs"] if d not in ad]
        r["displaced_docs"]=[d for d in r["phase29"]["relevant_docs"] if d not in bd]
        r["new_chunks"]=[{**c,"rank":i} for i,c in enumerate(r["phase27"]["relevant_chunks"],1) if chunk_key(c) not in ac]
        r["lost_chunks"]=[{**c,"rank":i} for i,c in enumerate(r["phase29"]["relevant_chunks"],1) if chunk_key(c) not in bc]
        r["urls"]={str(d):official.get(d,"UNKNOWN") for d in set(r["new_docs"]+r["displaced_docs"])}
    known=[]
    for relative in ("artifacts/multilingual_retrieval_m1/ai_judgments_import.csv","artifacts/multilingual_retrieval_m2/ai_judgments_m2.csv"):
        path=ROOT/relative
        with path.open(encoding="utf-8-sig",newline="") as stream:
            for row in csv.DictReader(stream):
                qid=int(row["query_id"])
                if qid in development and row["judgment"] in {"1","2"}:
                    doc=int(row["doc_id"])
                    known.append({"query_id":qid,"doc_id":doc,"label":row["judgment"],"status":row["verification_status"],
                                  "phase29_present":doc in compact[qid]["a_docs"],"phase27_present":doc in compact[qid]["b_docs"]})
    summary={"audit_status":audit["status"],"phase27_zip":str(challenger),"phase27_sha256":package["zip_sha256"],
             "phase27_bytes":challenger.stat().st_size,"phase29_sha256":expected,"global_comparison":dict(totals),
             "development_pool":len(development),"changed_development_pool":len(eligible),"sample_ids":sample_ids,
             "selection":"first six SHA256(seed:query_id) per stratum; no labels or content used",
             "sample_strata":dict(Counter(r["stratum"] for r in sample)),"known_silver_positive_pairs":known,
             "sealed_labels_opened":False,"AI_review_is_provisional":True}
    (out/"sample_evidence.json").write_text(json.dumps(sample,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="known_silver_positive_pairs"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
