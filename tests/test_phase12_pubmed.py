import gzip
import http.client
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools import phase12_pubmed as pubmed
from tools.phase12_pubmed import parse_pubmed_xml, pmid_from_url
from tools import phase12_worker
from tools.phase12_worker import process_record


def test_pubmed_url_mapping_is_strict():
    assert pmid_from_url("https://pubmed.ncbi.nlm.nih.gov/12345/")==12345
    assert pmid_from_url("https://pubmed.ncbi.nlm.nih.gov/12345/?x=1") is None
    assert pmid_from_url("https://example.org/12345/") is None


def test_pubmed_xml_preserves_structured_abstract_and_metadata():
    payload=b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID><Article><ArticleTitle>A <i>nested</i> title</ArticleTitle><Abstract><AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">No benefit was observed.</AbstractText><AbstractText Label="RESULTS">Dose was 5 mg.</AbstractText></Abstract><Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue><Title>Journal X</Title></Journal><Language>eng</Language><PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList></Article><MeshHeadingList><MeshHeading><DescriptorName>Hypertension</DescriptorName><QualifierName>therapy</QualifierName></MeshHeading></MeshHeadingList></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    row=parse_pubmed_xml(payload)[12345]
    assert row["title"]=="A nested title"
    assert row["abstract_sections"][0]=={"label":"BACKGROUND","nlm_category":"BACKGROUND","text":"No benefit was observed."}
    assert "Dose was 5 mg." in row["original_evidence_text"]
    assert row["mesh_terms"]==["Hypertension / therapy"]
    assert row["publication_year"]=="2024"
    assert row["acquisition_status"]=="ABSTRACT"


def test_pubmed_xml_never_fabricates_missing_abstract():
    payload=b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>9</PMID><Article><ArticleTitle>Title only</ArticleTitle></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    row=parse_pubmed_xml(payload)[9]
    assert row["abstract_sections"]==[]
    assert row["abstract_text"]==""
    assert row["original_evidence_text"]=="Title only"
    assert row["acquisition_status"]=="TITLE_ONLY"


def test_pubmed_book_article_is_not_misclassified_as_missing():
    payload=b"<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>20301282</PMID><Book><BookTitle>GeneReviews</BookTitle><PubDate><Year>1993</Year></PubDate></Book><ArticleTitle>Primary Trimethylaminuria</ArticleTitle><Language>eng</Language><PublicationType>Review</PublicationType><Abstract><AbstractText Label='SUMMARY'>Original book abstract.</AbstractText></Abstract></BookDocument></PubmedBookArticle></PubmedArticleSet>"
    row=parse_pubmed_xml(payload)[20301282]
    assert row["title"]=="Primary Trimethylaminuria"
    assert row["abstract_text"]=="Original book abstract."
    assert row["journal"]=="GeneReviews"
    assert row["publication_year"]=="1993"
    assert row["acquisition_status"]=="ABSTRACT"


def _processed_row(status="ABSTRACT"):
    return {
        "official_doc_id":77,"official_url":"https://pubmed.ncbi.nlm.nih.gov/9/","pmid":9,
        "title":"Original title","abstract_sections":[{"label":"METHODS","nlm_category":"METHODS","text":"Original evidence sentence."}],
        "abstract_text":"Original evidence sentence.","mesh_terms":["Private metadata term"],"publication_types":["Journal Article"],
        "journal":"Journal","publication_year":"2024","article_languages":["eng"],
        "retrieval_text":"Original title\nMETHODS: Original evidence sentence.\nMeSH: Private metadata term",
        "original_evidence_text":"Original title\nMETHODS: Original evidence sentence.","acquisition_status":status,
        "batch_index":0,"raw_xml_sha256":"a"*64,
    }


def test_phase12_processing_separates_retrieval_metadata_from_evidence():
    document,chunks=process_record(_processed_row())
    assert document is not None
    assert "Private metadata term" in document["retrieval_text"]
    assert "Private metadata term" not in document["normalized_text"]
    assert chunks
    for chunk in chunks:
        assert document["normalized_text"][chunk["start_offset"]:chunk["end_offset"]]==chunk["raw_text"]


def test_phase12_processing_has_deterministic_chunk_ids():
    first=process_record(_processed_row())[1]
    second=process_record(_processed_row())[1]
    assert [item["chunk_id"] for item in first]==[item["chunk_id"] for item in second]
    assert len({item["chunk_id"] for item in first})==len(first)


def test_phase12_processing_does_not_fabricate_missing_record():
    document,chunks=process_record(_processed_row("MISSING"))
    assert document is None
    assert chunks==[]


def test_phase12_dependency_preflight_fails_before_long_worker(monkeypatch):
    monkeypatch.setattr(phase12_worker.importlib.util,"find_spec",lambda name: None if name=="psutil" else object())
    with pytest.raises(ModuleNotFoundError,match="psutil"):
        phase12_worker.dependency_preflight({},verify_runtime=False)


def test_phase12_dependency_preflight_imports_every_remaining_stage_module(monkeypatch):
    class Available:
        __version__="test"
    seen=[]
    monkeypatch.setattr(phase12_worker.importlib.util,"find_spec",lambda name: object())
    monkeypatch.setattr(phase12_worker.importlib,"import_module",lambda name: seen.append(name) or Available())
    result=phase12_worker.dependency_preflight({},verify_runtime=False)
    assert result["passed"] is True
    assert seen==list(phase12_worker.REMAINING_STAGE_MODULES)


def _config(tmp_path:Path,attempts:int=3)->dict:
    return {
        "outputs":{"records":"records","raw_xml":"raw","artifacts":"artifacts"},
        "acquisition":{"maximum_attempts":attempts,"requests_per_second":2.5},
    }


def _xml(pmid:int)->bytes:
    return f"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>T</ArticleTitle></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>".encode()


def test_incomplete_read_retries_exact_same_batch(tmp_path):
    config=_config(tmp_path);calls=[]
    def request(_config,pmids,_pacer):
        calls.append(list(pmids))
        if len(calls)==1:raise http.client.IncompleteRead(b"<partial")
        return _xml(pmids[0])
    payload,parsed,history=pubmed._fetch_with_retries(config,[9],object(),request_fn=request,sleep=lambda _:None,jitter=lambda:0)
    assert calls==[[9],[9]]
    assert parsed.keys()=={9}
    assert history[0]["error_type"]=="IncompleteRead"
    assert payload==_xml(9)


def test_partial_xml_cannot_be_committed(tmp_path):
    config=_config(tmp_path,attempts=2)
    def request(*_):return b"<PubmedArticleSet><PubmedArticle>"
    with pytest.raises(pubmed.RecoverableAcquisitionError):
        pubmed._fetch_with_retries(config,[9],object(),request_fn=request,sleep=lambda _:None,jitter=lambda:0)
    assert list(tmp_path.rglob("*.done.json"))==[]
    assert list(tmp_path.rglob("*.parquet"))==[]


def test_retry_exhaustion_is_explicitly_recoverable(tmp_path):
    config=_config(tmp_path,attempts=2)
    def request(*_):raise ConnectionResetError("reset")
    with pytest.raises(pubmed.RecoverableAcquisitionError,match="retry exhaustion"):
        pubmed._fetch_with_retries(config,[9],object(),request_fn=request,sleep=lambda _:None,jitter=lambda:0)


def test_request_pacer_enforces_rate_limit():
    now=[0.0];sleeps=[]
    def clock():return now[0]
    def sleep(delay):sleeps.append(delay);now[0]+=delay
    pacer=pubmed.RequestPacer(2.5,clock=clock,sleep=sleep)
    pacer.wait();pacer.wait();pacer.wait()
    assert sleeps==pytest.approx([0.4,0.4])


def _commit_one(tmp_path:Path,config:dict,manifest_row:dict,index:int=0)->None:
    part,raw,marker=pubmed._batch_paths(config,index);part.parent.mkdir(parents=True,exist_ok=True);raw.parent.mkdir(parents=True,exist_ok=True)
    payload=_xml(manifest_row["pmid"]);parsed=pubmed.parse_pubmed_xml(payload)[manifest_row["pmid"]]
    row={"official_doc_id":manifest_row["official_doc_id"],"official_url":manifest_row["official_url"],"pmid":manifest_row["pmid"],**parsed,"source":"NCBI PubMed EFetch public data","batch_index":index,"raw_xml_sha256":pubmed.sha256_bytes(payload),"fetched_at_utc":"x"}
    pq.write_table(pa.Table.from_pylist([row],schema=pubmed.RECORD_SCHEMA),part)
    pubmed._write_gzip(raw,payload)
    checkpoint={"batch_index":index,"target_count":1,"returned_records":1,"pmid_order_sha256":pubmed.sha256_bytes(str(manifest_row["pmid"]).encode()),"raw_payload_sha256":pubmed.sha256_bytes(payload),"raw_gzip_sha256":pubmed.sha256_file(raw),"part_sha256":pubmed.sha256_file(part),"status_counts":{"TITLE_ONLY":1},"completed_at":"x","retry_history":[]}
    pubmed.atomic_json(marker,checkpoint)


def test_completed_batch_is_reused_and_exactly_accounted(tmp_path,monkeypatch):
    monkeypatch.setattr(pubmed,"ROOT",tmp_path);config=_config(tmp_path);row={"official_doc_id":1,"official_url":"https://pubmed.ncbi.nlm.nih.gov/9/","pmid":9}
    _commit_one(tmp_path,config,row)
    audit=pubmed.audit_completed(config,[row],1)
    assert audit["completed_batches"]==1
    assert audit["completed_ids"]==1
    assert audit["duplicate_completed_pmids"]==0


def test_completed_batch_membership_drift_cannot_duplicate_records(tmp_path,monkeypatch):
    monkeypatch.setattr(pubmed,"ROOT",tmp_path);config=_config(tmp_path);row={"official_doc_id":1,"official_url":"https://pubmed.ncbi.nlm.nih.gov/9/","pmid":9}
    _commit_one(tmp_path,config,row)
    changed={"official_doc_id":2,"official_url":"https://pubmed.ncbi.nlm.nih.gov/10/","pmid":10}
    with pytest.raises(ValueError,match="checkpoint no longer matches official manifest"):
        pubmed.audit_completed(config,[changed],1)


def test_permanent_http_error_is_not_retried(tmp_path):
    import urllib.error
    config=_config(tmp_path);calls=[]
    def request(*_):
        calls.append(1);raise urllib.error.HTTPError("x",400,"bad",{},None)
    with pytest.raises(urllib.error.HTTPError):
        pubmed._fetch_with_retries(config,[9],object(),request_fn=request,sleep=lambda _:None,jitter=lambda:0)
    assert len(calls)==1
