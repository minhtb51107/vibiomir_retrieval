#!/usr/bin/env python3
"""Manifest and resumable public-data acquisition for Phase 12 PubMed."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import http.client
import json
import os
import random
import re
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from src.source_census.pipeline import atomic_json


PUBMED_HOST="pubmed.ncbi.nlm.nih.gov"
PMID_PATH=re.compile(r"^/(\d+)/?$")
SECTION_TYPE=pa.struct([pa.field("label",pa.string()),pa.field("nlm_category",pa.string()),pa.field("text",pa.string())])
RECORD_SCHEMA=pa.schema([
    pa.field("official_doc_id",pa.int64()),pa.field("official_url",pa.string()),pa.field("pmid",pa.int64()),
    pa.field("title",pa.string()),pa.field("abstract_sections",pa.list_(SECTION_TYPE)),pa.field("abstract_text",pa.string()),
    pa.field("mesh_terms",pa.list_(pa.string())),pa.field("publication_types",pa.list_(pa.string())),
    pa.field("journal",pa.string()),pa.field("publication_year",pa.string()),pa.field("article_languages",pa.list_(pa.string())),
    pa.field("retrieval_text",pa.string()),pa.field("original_evidence_text",pa.string()),
    pa.field("acquisition_status",pa.string()),pa.field("source",pa.string()),pa.field("batch_index",pa.int32()),
    pa.field("raw_xml_sha256",pa.string()),pa.field("fetched_at_utc",pa.string()),
])


def load(path:Path)->dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_bytes(value:bytes)->str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def text(element:ET.Element|None)->str:
    if element is None:return ""
    return "".join(element.itertext()).strip()


def pmid_from_url(url:str)->int|None:
    parsed=urllib.parse.urlsplit(str(url));host=(parsed.hostname or "").lower().removeprefix("www.")
    if host!=PUBMED_HOST or parsed.query or parsed.fragment:return None
    matched=PMID_PATH.fullmatch(parsed.path)
    return int(matched.group(1)) if matched else None


def build_manifest(config:dict)->dict:
    corpus=ROOT/config["inputs"]["official_corpus"];destination=ROOT/config["outputs"]["manifest"];destination.parent.mkdir(parents=True,exist_ok=True)
    rows=[];malformed=[]
    for batch in pq.ParquetFile(corpus).iter_batches(columns=["id","url"],batch_size=65536):
        for doc_id,url in zip(batch.column(0).to_pylist(),batch.column(1).to_pylist(),strict=True):
            parsed=urllib.parse.urlsplit(str(url));host=(parsed.hostname or "").lower().removeprefix("www.")
            if host!=PUBMED_HOST:continue
            pmid=pmid_from_url(str(url))
            if pmid is None:malformed.append({"doc_id":int(doc_id),"url":str(url)})
            else:rows.append({"official_doc_id":int(doc_id),"official_url":str(url),"pmid":pmid})
    doc_ids=[row["official_doc_id"] for row in rows];pmids=[row["pmid"] for row in rows];urls=[row["official_url"] for row in rows]
    result={"target_rows":len(rows)+len(malformed),"valid_mappings":len(rows),"malformed_urls":len(malformed),"unique_doc_ids":len(set(doc_ids)),"unique_urls":len(set(urls)),"unique_pmids":len(set(pmids)),"duplicate_doc_ids":len(doc_ids)-len(set(doc_ids)),"duplicate_urls":len(urls)-len(set(urls)),"duplicate_pmids":len(pmids)-len(set(pmids)),"one_to_one":len(rows)==len(set(doc_ids))==len(set(urls))==len(set(pmids))==240000 and not malformed,"malformed_sample":malformed[:20]}
    if not result["one_to_one"]:raise ValueError(f"official PubMed mapping ambiguity: {result}")
    rows.sort(key=lambda row:(row["pmid"],row["official_doc_id"]))
    temporary=destination.with_suffix(".parquet.tmp");pq.write_table(pa.Table.from_pylist(rows,schema=pa.schema([pa.field("official_doc_id",pa.int64()),pa.field("official_url",pa.string()),pa.field("pmid",pa.int64())])),temporary,compression="zstd");os.replace(temporary,destination)
    result.update({"path":str(destination),"sha256":sha256_file(destination)})
    artifact=ROOT/config["outputs"]["artifacts"]/"official_pubmed_manifest.json";atomic_json(artifact,result);return result


def _article_year(article:ET.Element)->str:
    for path in ("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year","./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate","./PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year"):
        value=text(article.find(path))
        if value:return value[:4] if len(value)>=4 else value
    return ""


def _book_year(article:ET.Element)->str:
    for path in ("./BookDocument/Book/PubDate/Year","./BookDocument/Book/BeginningDate/Year","./PubmedBookData/History/PubMedPubDate[@PubStatus='pubmed']/Year"):
        value=text(article.find(path))
        if value:return value[:4] if len(value)>=4 else value
    return ""


def parse_pubmed_xml(payload:bytes)->dict[int,dict]:
    root=ET.fromstring(payload);output={}
    for article in [*root.findall(".//PubmedArticle"),*root.findall(".//PubmedBookArticle")]:
        is_book=article.tag.endswith("PubmedBookArticle")
        prefix="./BookDocument" if is_book else "./MedlineCitation/Article"
        pmid_text=text(article.find("./BookDocument/PMID" if is_book else "./MedlineCitation/PMID"))
        if not pmid_text.isdigit():continue
        pmid=int(pmid_text)
        if pmid in output:raise ValueError(f"duplicate PMID in EFetch response: {pmid}")
        title=text(article.find(prefix+"/ArticleTitle"))
        sections=[]
        for node in article.findall(prefix+"/Abstract/AbstractText"):
            value=text(node)
            if value:sections.append({"label":str(node.attrib.get("Label","")).strip(),"nlm_category":str(node.attrib.get("NlmCategory","")).strip(),"text":value})
        abstract_text="\n".join(section["text"] for section in sections)
        mesh=[]
        for heading in article.findall("./MedlineCitation/MeshHeadingList/MeshHeading") if not is_book else []:
            descriptor=text(heading.find("./DescriptorName"));qualifiers=[text(node) for node in heading.findall("./QualifierName") if text(node)]
            if descriptor:mesh.append(descriptor+(" / "+" / ".join(qualifiers) if qualifiers else ""))
        publication_types=[text(node) for node in article.findall(prefix+("/PublicationType" if is_book else "/PublicationTypeList/PublicationType")) if text(node)]
        journal=(text(article.find("./BookDocument/Book/BookTitle")) if is_book else text(article.find("./MedlineCitation/Article/Journal/Title")) or text(article.find("./MedlineCitation/MedlineJournalInfo/MedlineTA")))
        languages=[text(node) for node in article.findall(prefix+"/Language") if text(node)]
        year=_book_year(article) if is_book else _article_year(article)
        evidence=[]
        if title:evidence.append(title)
        for section in sections:evidence.append((section["label"]+": " if section["label"] else "")+section["text"])
        retrieval=[*evidence]
        if mesh:retrieval.append("MeSH: "+"; ".join(mesh))
        metadata="; ".join(value for value in [journal,year,"; ".join(publication_types)] if value)
        if metadata:retrieval.append("Publication: "+metadata)
        status="ABSTRACT" if abstract_text else ("TITLE_ONLY" if title else "METADATA_ONLY")
        output[pmid]={"title":title,"abstract_sections":sections,"abstract_text":abstract_text,"mesh_terms":mesh,"publication_types":publication_types,"journal":journal,"publication_year":year,"article_languages":languages,"retrieval_text":"\n".join(retrieval),"original_evidence_text":"\n".join(evidence),"acquisition_status":status}
    return output


def _write_gzip(path:Path,payload:bytes)->None:
    temporary=path.with_suffix(path.suffix+".tmp");path.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,compresslevel=6,mtime=0) as handle:handle.write(payload)
    os.replace(temporary,path)


def _batch_paths(config:dict,index:int)->tuple[Path,Path,Path]:
    records=ROOT/config["outputs"]["records"];raw=ROOT/config["outputs"]["raw_xml"]
    return records/f"part-{index:06d}.parquet",raw/f"batch-{index:06d}.xml.gz",records/f"part-{index:06d}.done.json"


class RecoverableAcquisitionError(RuntimeError):
    """A batch exhausted bounded retries without committing output."""


class RequestPacer:
    def __init__(self,requests_per_second:float,*,clock=time.monotonic,sleep=time.sleep)->None:
        if not 0<requests_per_second<=2.5:raise ValueError("NCBI request rate must be in (0, 2.5]")
        self.interval=1.0/requests_per_second;self.clock=clock;self.sleep=sleep;self.last_start:float|None=None

    def wait(self)->float:
        now=self.clock()
        if self.last_start is not None:
            delay=self.interval-(now-self.last_start)
            if delay>0:self.sleep(delay);now=self.clock()
        self.last_start=now;return now


def _request(config:dict,pmids:list[int],pacer:RequestPacer,*,opener=urllib.request.urlopen)->bytes:
    acquisition=config["acquisition"];pacer.wait();data=urllib.parse.urlencode({"db":acquisition["database"],"id":",".join(map(str,pmids)),"retmode":"xml","rettype":"abstract","tool":acquisition["tool"]}).encode("ascii")
    request=urllib.request.Request(acquisition["endpoint"],data=data,headers={"User-Agent":acquisition["user_agent"],"Accept":"application/xml"},method="POST")
    with opener(request,timeout=int(acquisition["request_timeout_seconds"])) as response:return response.read()


def _retryable(exc:BaseException)->bool:
    if isinstance(exc,urllib.error.HTTPError):return exc.code==429 or exc.code==408 or 500<=exc.code<=599
    return isinstance(exc,(http.client.IncompleteRead,http.client.RemoteDisconnected,ConnectionResetError,ConnectionAbortedError,BrokenPipeError,TimeoutError,socket.timeout,urllib.error.URLError,ET.ParseError))


def _fetch_with_retries(config:dict,pmids:list[int],pacer:RequestPacer,*,request_fn=_request,sleep=time.sleep,jitter=random.random)->tuple[bytes,dict[int,dict],list[dict]]:
    attempts=int(config["acquisition"]["maximum_attempts"]);history=[]
    for attempt in range(1,attempts+1):
        try:
            payload=request_fn(config,pmids,pacer);parsed=parse_pubmed_xml(payload);return payload,parsed,history
        except BaseException as exc:
            if not _retryable(exc):raise
            retry_after=0.0
            if isinstance(exc,urllib.error.HTTPError) and exc.headers:
                try:retry_after=float(exc.headers.get("Retry-After") or 0.0)
                except (TypeError,ValueError):retry_after=0.0
            base=min(60.0,2.0**attempt);delay=max(retry_after,base+min(1.0,base*0.25)*float(jitter()))
            history.append({"attempt":attempt,"error_type":type(exc).__name__,"error":str(exc),"retry_after":retry_after,"backoff_seconds":round(delay,3)})
            if attempt==attempts:raise RecoverableAcquisitionError(f"PubMed batch retry exhaustion after {attempts} attempts; first_pmid={pmids[0]} count={len(pmids)}; last={type(exc).__name__}: {exc}") from exc
            sleep(delay)


def audit_completed(config:dict,rows:list[dict],batch_size:int)->dict:
    batches=(len(rows)+batch_size-1)//batch_size;counts=Counter();completed=[];seen_pmids=set();missing_by_batch={}
    records_root=ROOT/config["outputs"]["records"];raw_root=ROOT/config["outputs"]["raw_xml"]
    temporaries=[str(path) for root in (records_root,raw_root) if root.exists() for path in root.glob("*.tmp")]
    if temporaries:raise ValueError(f"uncommitted PubMed temporary files require review: {temporaries[:10]}")
    for index in range(batches):
        part,raw,marker=_batch_paths(config,index);present=(part.exists(),raw.exists(),marker.exists())
        if not any(present):continue
        if not all(present):raise ValueError(f"incomplete committed batch triple {index}: part/raw/marker={present}")
        expected=rows[index*batch_size:(index+1)*batch_size];expected_pmids=[int(row["pmid"]) for row in expected];expected_docs=[int(row["official_doc_id"]) for row in expected];expected_urls=[str(row["official_url"]) for row in expected]
        checkpoint=json.loads(marker.read_text(encoding="utf-8"));expected_sha=sha256_bytes("\n".join(map(str,expected_pmids)).encode("ascii"))
        if checkpoint["pmid_order_sha256"]!=expected_sha or checkpoint["target_count"]!=len(expected):raise ValueError(f"batch {index} checkpoint no longer matches official manifest")
        if sha256_file(part)!=checkpoint["part_sha256"] or sha256_file(raw)!=checkpoint["raw_gzip_sha256"]:raise ValueError(f"completed PubMed batch integrity drift: {index}")
        with gzip.open(raw,"rb") as handle:payload=handle.read()
        if sha256_bytes(payload)!=checkpoint["raw_payload_sha256"]:raise ValueError(f"batch {index} raw XML payload hash mismatch")
        table=pq.read_table(part,columns=["official_doc_id","official_url","pmid","acquisition_status"]);values=table.to_pylist()
        observed_pmids=[int(row["pmid"]) for row in values]
        if observed_pmids!=expected_pmids or [int(row["official_doc_id"]) for row in values]!=expected_docs or [str(row["official_url"]) for row in values]!=expected_urls:raise ValueError(f"batch {index} parsed rows do not exactly match official manifest")
        duplicates=set(observed_pmids)&seen_pmids
        if duplicates:raise ValueError(f"duplicate completed PMIDs across batches: {sorted(duplicates)[:10]}")
        seen_pmids.update(observed_pmids);completed.append(index);counts.update(str(row["acquisition_status"]) for row in values)
        missing=[int(row["pmid"]) for row in values if row["acquisition_status"]=="MISSING"]
        if missing:missing_by_batch[str(index)]=missing
    if completed!=list(range(len(completed))):raise ValueError(f"completed PubMed batches are not a contiguous prefix: tail={completed[-10:]}")
    return {"completed_batches":len(completed),"completed_ids":len(seen_pmids),"completed_batch_indexes":completed,"status_counts":dict(counts),"missing_by_batch":missing_by_batch,"duplicate_completed_pmids":0,"temporary_files":[]}


def audit_missing_records(config:dict,audit:dict)->dict:
    classifications=Counter();samples={}
    for key,pmids in audit["missing_by_batch"].items():
        index=int(key);_,raw,_=_batch_paths(config,index)
        with gzip.open(raw,"rb") as handle:payload=handle.read()
        parsed=parse_pubmed_xml(payload)
        for pmid in pmids:
            if pmid in parsed:category="PARSER_MISSED_VALID_RECORD"
            elif re.search(rb"<PMID(?:\s[^>]*)?>\s*"+str(pmid).encode("ascii")+rb"\s*</PMID>",payload):category="PARSER_MISSED_PMID_ELEMENT"
            else:category="PUBMED_RETURNED_NO_MATCHING_RECORD"
            classifications[category]+=1;samples.setdefault(category,[])
            if len(samples[category])<20:samples[category].append(pmid)
    result={"missing_total":sum(classifications.values()),"classifications":dict(classifications),"samples":samples,"transport_failures_committed_as_missing":0,"official_mapping_ambiguities":0,"targeted_retry_required":bool(classifications.get("PUBMED_RETURNED_NO_MATCHING_RECORD",0)),"note":"Complete parseable raw XML exists for every classified row; transport failures cannot reach a .done.json checkpoint."}
    atomic_json(ROOT/config["outputs"]["artifacts"]/"missing_record_audit.json",result);return result


def _repair_missing_rows(config:dict,recovered:dict[int,dict],raw_sha_by_pmid:dict[int,str],*,reason:str,supplemental_raw:str|None=None)->int:
    repaired=0
    for part in sorted((ROOT/config["outputs"]["records"]).glob("part-*.parquet")):
        rows=pq.read_table(part).to_pylist();changed=[]
        for row in rows:
            pmid=int(row["pmid"])
            if row["acquisition_status"]=="MISSING" and pmid in recovered:
                row.update(recovered[pmid]);row["raw_xml_sha256"]=raw_sha_by_pmid[pmid];changed.append(pmid);repaired+=1
        if not changed:continue
        temporary=part.with_suffix(".parquet.tmp");pq.write_table(pa.Table.from_pylist(rows,schema=RECORD_SCHEMA),temporary,compression="zstd");os.replace(temporary,part)
        index=int(part.stem.split("-")[1]);_,_,marker=_batch_paths(config,index);checkpoint=json.loads(marker.read_text(encoding="utf-8"));checkpoint["part_sha256"]=sha256_file(part);checkpoint["status_counts"]=dict(Counter(str(row["acquisition_status"]) for row in rows));checkpoint.setdefault("recoveries",[]).append({"reason":reason,"pmids":changed,"supplemental_raw":supplemental_raw});atomic_json(marker,checkpoint)
    return repaired


def recover_missing_records(config:dict,audit:dict,pacer:RequestPacer)->dict:
    # First repair the historical parser omission directly from already-retained raw XML.
    locally_recovered={};local_sha={}
    for key,pmids in audit["missing_by_batch"].items():
        _,raw,_=_batch_paths(config,int(key))
        with gzip.open(raw,"rb") as handle:payload=handle.read()
        parsed=parse_pubmed_xml(payload);payload_sha=sha256_bytes(payload)
        for pmid in pmids:
            if pmid in parsed:locally_recovered[pmid]=parsed[pmid];local_sha[pmid]=payload_sha
    local_count=_repair_missing_rows(config,locally_recovered,local_sha,reason="expanded parser support for valid PubmedBookArticle in original complete response")
    refreshed=audit_completed(config,pq.read_table(ROOT/config["outputs"]["manifest"]).to_pylist(),int(config["acquisition"]["batch_size"]))
    remaining=[pmid for values in refreshed["missing_by_batch"].values() for pmid in values]
    network_count=0;retry_history=[]
    if remaining:
        payload,parsed,retry_history=_fetch_with_retries(config,remaining,pacer);raw=ROOT/config["outputs"]["raw_xml"]/"missing-targeted-retry.xml.gz";_write_gzip(raw,payload);payload_sha=sha256_bytes(payload)
        recovered={pmid:parsed[pmid] for pmid in remaining if pmid in parsed}
        network_count=_repair_missing_rows(config,recovered,{pmid:payload_sha for pmid in recovered},reason="targeted post-acquisition retry of PubMed response omission",supplemental_raw=str(raw))
    final=audit_completed(config,pq.read_table(ROOT/config["outputs"]["manifest"]).to_pylist(),int(config["acquisition"]["batch_size"]))
    result={"parser_recovered":local_count,"targeted_retry_requested":len(remaining),"targeted_retry_recovered":network_count,"confirmed_missing_after_retry":int(final["status_counts"].get("MISSING",0)),"retry_history":retry_history}
    atomic_json(ROOT/config["outputs"]["artifacts"]/"missing_record_recovery.json",result);return result


def acquire(config:dict,run_state:Path)->dict:
    manifest_path=ROOT/config["outputs"]["manifest"]
    if not manifest_path.exists():build_manifest(config)
    rows=pq.read_table(manifest_path).to_pylist();batch_size=int(config["acquisition"]["batch_size"]);batches=(len(rows)+batch_size-1)//batch_size
    (ROOT/config["outputs"]["records"]).mkdir(parents=True,exist_ok=True);(ROOT/config["outputs"]["raw_xml"]).mkdir(parents=True,exist_ok=True)
    initial=audit_completed(config,rows,batch_size);missing_audit=audit_missing_records(config,initial);completed=set(initial["completed_batch_indexes"]);counts=Counter(initial["status_counts"]);completed_ids=int(initial["completed_ids"]);started=time.monotonic();new_ids=0;pacer=RequestPacer(float(config["acquisition"]["requests_per_second"]));retry_history=[]
    for index in range(batches):
        subset=rows[index*batch_size:(index+1)*batch_size];part,raw,marker=_batch_paths(config,index)
        expected_pmids=[int(row["pmid"]) for row in subset];pmid_sha=sha256_bytes("\n".join(map(str,expected_pmids)).encode("ascii"))
        if index in completed:
            done=json.loads(marker.read_text(encoding="utf-8"))
            if done["pmid_order_sha256"]!=pmid_sha:raise ValueError(f"batch manifest drift: {index}")
            continue
        payload,parsed,batch_retries=_fetch_with_retries(config,expected_pmids,pacer);retry_history.extend({"batch_index":index,**row} for row in batch_retries)
        unexpected=sorted(set(parsed)-set(expected_pmids))
        if unexpected:raise ValueError(f"EFetch returned unexpected PMIDs in batch {index}: {unexpected[:10]}")
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat();raw_sha=sha256_bytes(payload);output=[]
        for source in subset:
            pmid=int(source["pmid"]);record=parsed.get(pmid)
            if record is None:record={"title":"","abstract_sections":[],"abstract_text":"","mesh_terms":[],"publication_types":[],"journal":"","publication_year":"","article_languages":[],"retrieval_text":"","original_evidence_text":"","acquisition_status":"MISSING"}
            output.append({"official_doc_id":int(source["official_doc_id"]),"official_url":str(source["official_url"]),"pmid":pmid,**record,"source":"NCBI PubMed EFetch public data","batch_index":index,"raw_xml_sha256":raw_sha,"fetched_at_utc":fetched_at})
        statuses=Counter(row["acquisition_status"] for row in output);part.parent.mkdir(parents=True,exist_ok=True);temporary=part.with_suffix(".parquet.tmp");pq.write_table(pa.Table.from_pylist(output,schema=RECORD_SCHEMA),temporary,compression="zstd");os.replace(temporary,part);_write_gzip(raw,payload)
        checkpoint={"batch_index":index,"target_count":len(subset),"returned_records":len(parsed),"pmid_order_sha256":pmid_sha,"raw_payload_sha256":raw_sha,"raw_gzip_sha256":sha256_file(raw),"part_sha256":sha256_file(part),"status_counts":dict(statuses),"completed_at":fetched_at,"retry_history":batch_retries}
        atomic_json(marker,checkpoint);new_ids+=len(subset);completed_ids+=len(subset);counts.update(statuses);completed.add(index)
        elapsed=time.monotonic()-started;rate=new_ids/max(elapsed,1e-9)*60;remaining=len(rows)-completed_ids;eta=remaining/max(rate/60,1e-9)
        state={"stage":"ACQUIRE_PUBMED","status":"RUNNING","total":len(rows),"done":completed_ids,"remaining":remaining,"completed_batches":len(completed),"total_batches":batches,"current_batch":index,"batch_size":batch_size,"records_per_minute":round(rate,2),"eta_seconds":round(eta,1),"records_with_abstract":int(counts.get("ABSTRACT",0)),"title_only_records":int(counts.get("TITLE_ONLY",0)),"metadata_only_records":int(counts.get("METADATA_ONLY",0)),"missing_records":int(counts.get("MISSING",0)),"retry_count":len(retry_history),"last_retry":retry_history[-1] if retry_history else None,"checkpoint":str(marker),"manifest":str(manifest_path),"requests_per_second_limit":float(config["acquisition"]["requests_per_second"])}
        atomic_json(run_state,{**(json.loads(run_state.read_text(encoding="utf-8")) if run_state.exists() else {}),**state})
    before_recovery=audit_completed(config,rows,batch_size);missing_audit=audit_missing_records(config,before_recovery);missing_recovery=recover_missing_records(config,before_recovery,pacer);final=audit_completed(config,rows,batch_size);summary={"status":"ACQUISITION_COMPLETE","target":len(rows),**{key:value for key,value in final.items() if key not in {"completed_batch_indexes","missing_by_batch"}},"missing_record_audit":missing_audit,"missing_record_recovery":missing_recovery,"manifest":str(manifest_path),"records_root":config["outputs"]["records"],"raw_xml_root":config["outputs"]["raw_xml"],"batch_size":batch_size,"requests_per_second_limit":float(config["acquisition"]["requests_per_second"]),"retry_history":retry_history,"provenance":"exact official PMID mapping fetched from NCBI PubMed EFetch; raw batch XML retained","no_relevance_ranking_used":True}
    atomic_json(ROOT/config["outputs"]["artifacts"]/"acquisition_summary.json",summary);atomic_json(run_state,{**(json.loads(run_state.read_text(encoding="utf-8")) if run_state.exists() else {}),"stage":"ACQUIRE_PUBMED_COMPLETE","status":"SUCCESS","total":len(rows),"done":len(rows),"remaining":0,"checkpoint":str(ROOT/config["outputs"]["artifacts"]/"acquisition_summary.json")});return summary


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["manifest","acquire"]);parser.add_argument("--config",default="configs/phase12_pubmed.yaml");parser.add_argument("--run-state");args=parser.parse_args();config=load(ROOT/args.config)
    if args.command=="manifest":result=build_manifest(config)
    else:
        if not args.run_state:raise ValueError("--run-state is required for acquisition")
        result=acquire(config,ROOT/args.run_state)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
