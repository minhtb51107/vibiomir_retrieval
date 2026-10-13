#!/usr/bin/env python3
"""Milestone-1 streaming audit. Reads durable artifacts; performs no model/network work."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.evaluation.silver import normalize_query, validate_benchmark
from src.submission.common import atomic_write_json, file_sha256

OUT=ROOT/"artifacts/system_upgrade_m1"
OFFICIAL=ROOT/"data/raw/links_corpus_vi_en_zh.parquet"
BROAD=ROOT/"data/source_census/phase11a_bottleneck/broad/documents.parquet"
PUBMED=ROOT/"data/pubmed_phase12/processed/documents.parquet"
P11=ROOT/"submissions/00_READY_TO_UPLOAD/phase11c_BROAD_EXISTING_NEW_DOCS20_CHUNKS40.json"
P12=ROOT/"submissions/00_READY_TO_UPLOAD/phase12_FULL_PUBMED_DENSE_DOCS20_CHUNKS40.json"


def domain(url:str)->str: return (urlsplit(str(url)).hostname or "unknown").lower().removeprefix("www.")
def digest_int(value:str)->int: return int(hashlib.sha256(value.encode()).hexdigest(),16)
def write_text(name:str,text:str)->None: (OUT/name).write_text(text.rstrip()+"\n",encoding="utf-8")


def official_inventory()->tuple[set[int],Counter]:
    ids=set(); domains=Counter()
    for batch in pq.ParquetFile(OFFICIAL).iter_batches(columns=["id","url"],batch_size=131072):
        for doc,url in zip(*(c.to_pylist() for c in batch.columns),strict=True): ids.add(int(doc));domains[domain(url)]+=1
    return ids,domains


def parquet_meta(path:Path)->dict:
    pf=pq.ParquetFile(path);return {"path":str(path.relative_to(ROOT)),"rows":pf.metadata.num_rows,"size_bytes":path.stat().st_size,"sha256":file_sha256(path)}


def candidate_parts(root:Path)->dict:
    result={}
    for marker in sorted(root.glob("*.json")):
        part=marker.with_suffix(".parquet")
        if part.exists():
            meta=json.loads(marker.read_text(encoding="utf-8")); result[str(meta.get("source",marker.stem))]=parquet_meta(part)
    return result


def build_inventory(official_ids:set[int],domains:Counter)->dict:
    migration=json.loads((ROOT/"artifacts/phase11b_new_corpus/migration_audit.json").read_text(encoding="utf-8"))
    funnel=json.loads((ROOT/"artifacts/phase13_diagnosis/coverage_funnel.json").read_text(encoding="utf-8"))
    p11parts=candidate_parts(ROOT/"data/source_census/phase11c_broad_existing_new/round1/source_candidate_parts")
    p12parts=candidate_parts(ROOT/"data/pubmed_phase12/retrieval/round1/source_candidate_parts")
    def db(path,table):
        con=sqlite3.connect(f"file:{path.resolve()}?mode=ro",uri=True); n=int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]);con.close();return n
    inventory={
      "generated_at":dt.datetime.now(dt.timezone.utc).isoformat(),
      "count_semantics":"Each stage is separate: official URL != acquired != extracted/usable != indexed != candidate != reranked != emitted.",
      "official":{"documents":len(official_ids),"unique_ids":len(official_ids),"sha256":file_sha256(OFFICIAL),
        "organizer_declared_languages":{"vi":714469,"zh":240000,"en":240000},"domains":len(domains),"top_domains":dict(domains.most_common(40)),
        "url_domain_language_is_not_authoritative":True},
      "local_content":{"broad_existing":{"usable_documents":35690,"chunks":523482,"domains":65,"source":"verified migration audit"},
        "pubmed":{"official_rows":240000,"acquired_usable_documents":239896,"unavailable":104,"chunks":744861},
        "deduplicated_union":{"usable_documents":275586,"chunks":1268343,"official_gap_documents":len(official_ids)-275586,
          "identified_language_counts":{"vi":29396,"zh":592,"en":239896,"unknown":5702}}},
      "indexes":{"focused":{"documents":26639,"chunks":444411,"method":"per-source BGE-M3 dense + BM25/RRF; M16 candidates"},
        "broad_extra":{"documents":9051,"chunks":79071,"method":"global BM25 only"},
        "pubmed":{"documents":239896,"chunks":744861,"dense_document_embeddings":239896,"embedding_dimension":1024,
          "embedding_file_bytes":(ROOT/"data/pubmed_phase12/retrieval/pubmed_document_embeddings.f32").stat().st_size,"method":"BGE-M3 document dense blockwise exact search"},
        "single_unified_index_exists":False},
      "phase11c":{"candidate_parts":p11parts,"reranker_score_keys":db(ROOT/"data/source_census/phase11c_broad_existing_new/round1/source_scores.sqlite","pairs"),
        "final_document_slots":24000,"final_chunk_slots":48000,"organizer_final":0.0492},
      "phase12":{"candidate_parts":p12parts,"reranker_score_keys":db(ROOT/"data/pubmed_phase12/retrieval/round1/source_scores.sqlite","pairs"),
        "frozen_full_unique_score_keys":695539,"final_document_slots":24000,"final_chunk_slots":48000,"organizer_final":0.0451},
      "retrieval_funnel_reference":funnel["logical_candidate_funnel"],
      "provenance":{"migration_audit_sha256":file_sha256(ROOT/"artifacts/phase11b_new_corpus/migration_audit.json"),
        "coverage_funnel_sha256":file_sha256(ROOT/"artifacts/phase13_diagnosis/coverage_funnel.json")},
    }
    atomic_write_json(OUT/"corpus_inventory.json",inventory);return inventory


def flags(row:dict)->tuple[list[str],dict]:
    text=str(row.get("normalized_text") or "");title=str(row.get("title") or ""); low=text.casefold()
    paragraphs=[" ".join(p.split()) for p in re.split(r"\n\s*\n",text) if p.strip()]
    repeats=len(paragraphs)-len(set(paragraphs)); nav=sum(low.count(x) for x in ("đăng nhập","menu","cookie","copyright","xem thêm","related articles","privacy policy","javascript"))
    near_repeats=0
    for left,right in zip(paragraphs,paragraphs[1:]):
        a,b=set(left.casefold().split()),set(right.casefold().split())
        if a and b and len(a&b)/len(a|b)>=.9: near_repeats+=1
    found=[]
    if not text.strip():found.append("empty")
    if not title.strip():found.append("missing_title")
    if 0<len(text)<200:found.append("short_lt_200_review_not_reject")
    if re.search(r"<\/?(?:script|div|span|p|html)|&nbsp;|\{\s*[\"']",text,re.I):found.append("html_or_js_remnant")
    if nav>=3:found.append("navigation_or_boilerplate")
    if paragraphs and repeats/max(len(paragraphs),1)>=.2:found.append("repeated_paragraphs")
    if near_repeats:found.append("near_duplicate_adjacent_paragraphs")
    if int(row.get("raw_byte_length") or 0)>=2*1024*1024:found.append("possible_download_cap_truncation")
    if len(text)>=100000:found.append("possible_unrelated_concatenation_review")
    signal=str(row.get("language_signal") or ""); han=sum("\u4e00"<=c<="\u9fff" for c in text)/max(len(text),1)
    if signal.startswith("zh") and han<.1:found.append("possible_wrong_language")
    if signal.startswith("en") and han>.05:found.append("possible_wrong_language")
    if int(row.get("replacement_character_count") or 0)>0:found.append("replacement_characters")
    return found,{"characters":len(text),"paragraphs":len(paragraphs),"repeated_paragraphs":repeats,"near_duplicate_adjacent_paragraphs":near_repeats,"nav_markers":nav}


def data_quality(official:set[int])->dict:
    high=set();low=set()
    ranking=ROOT/"data/pubmed_phase12/retrieval/rankings/FULL/reranked_documents.parquet"
    for b in pq.ParquetFile(ranking).iter_batches(columns=["rank","doc_id"],batch_size=32768):
        for rank,doc in zip(*(c.to_pylist() for c in b.columns),strict=True):
            (high if int(rank)<=20 else low if int(rank)>=200 else set()).add(int(doc))
    totals={}; samples={}; content_hashes=Counter(); examples=[]
    for label,path in (("broad_existing",BROAD),("pubmed",PUBMED)):
        counters=Counter(); eligible=0
        columns=["doc_id","original_url","title","language_signal","normalized_text","extraction_method","extraction_status","replacement_character_count","raw_byte_length"]
        if label=="pubmed": columns.append("abstract_text")
        for b in pq.ParquetFile(path).iter_batches(columns=columns,batch_size=512):
            for row in b.to_pylist():
                doc=int(row["doc_id"])
                if doc not in official:continue
                eligible+=1; found,measure=flags(row);counters.update(found)
                if label=="pubmed" and not str(row.get("abstract_text") or "").strip(): counters["missing_abstract_title_only"]+=1
                text=" ".join(str(row.get("normalized_text") or "").casefold().split())
                if text:content_hashes[hashlib.sha256(text.encode()).hexdigest()]+=1
                lang="en" if label=="pubmed" else "zh" if str(row.get("language_signal")).startswith("zh") else "vi" if str(row.get("language_signal")).startswith("vi") else "unknown"
                rank_band="high" if doc in high else "low" if doc in low else "not_ranked_or_middle"
                stratum=f"{lang}:{rank_band}:{row.get('extraction_method')}"
                key=(digest_int(f"m1-quality:{doc}"),row,found,measure,stratum)
                current=samples.get(stratum)
                if current is None or key[0]<current[0]:samples[stratum]=key
        totals[label]={"official_usable_documents":eligible,"flag_counts":dict(counters),"flag_rates":{k:v/eligible for k,v in counters.items()}}
    for _,row,found,measure,stratum in sorted(samples.values())[:120]:
        examples.append({"stratum":stratum,"doc_id":int(row["doc_id"]),"source_domain":domain(row["original_url"]),"source_url":row["original_url"],
          "language_signal":row["language_signal"],"extraction_method":row["extraction_method"],"flags":"|".join(found),
          **measure,"title":str(row.get("title") or "")[:300],"text_excerpt":str(row.get("normalized_text") or "")[:800].replace("\n"," ")})
    with (OUT/"data_quality_examples.csv").open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(examples[0]));w.writeheader();w.writerows(examples)
    report={"method":"deterministic one-min-hash sample per language/rank-band/extraction-method stratum plus corpus-wide lightweight flags",
      "heuristic_warning":"Flags (including truncation, concatenation, wrong-language and near-duplicate heuristics) require human review; they are not automatic rejection. Short medical answers remain valid candidates.","collections":totals,
      "exact_duplicate_document_instances":sum(v for v in content_hashes.values() if v>1),"exact_duplicate_content_groups":sum(v>1 for v in content_hashes.values()),
      "sample_rows":len(examples),"examples_path":str((OUT/"data_quality_examples.csv").relative_to(ROOT))}
    atomic_write_json(OUT/"data_quality_audit.json",report);return report


def silver(official:set[int])->dict:
    rows=[]
    for batch in pq.ParquetFile(BROAD).iter_batches(columns=["doc_id","original_url","title","language_signal","sections","extraction_method","extraction_status"],batch_size=512):
        for row in batch.to_pylist():
            doc=int(row["doc_id"]);title=str(row.get("title") or "").strip();sections=row.get("sections") or []
            answers=[str(s.get("text") or "").strip() for s in sections if s.get("section_type")=="answer"]
            if doc not in official or not answers or not ("?" in title or "？" in title) or not (10<=len(title)<=300) or max(map(len,answers),default=0)<80:continue
            lang="zh" if str(row.get("language_signal")).startswith("zh") else "vi" if str(row.get("language_signal")).startswith("vi") else "unknown"
            url=str(row["original_url"]);norm=normalize_query(title)
            rows.append({"silver_query_id":f"silver_{lang}_{doc}","query_text":title,"positive_doc_id":doc,"source_url":url,
              "source_domain":domain(url),"query_language":lang,"document_language":lang,"split":"pending",
              "provenance_type":"source_title_question_with_extracted_answer","verbatim_source_query":True,
              "normalized_query_sha256":hashlib.sha256(norm.encode()).hexdigest(),"answer_section_count":len(answers),
              "extraction_method":row["extraction_method"]})
    # Exact/near duplicate token-set clusters are assigned wholly to one split.
    parent=list(range(len(rows)))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    tokens=[set(normalize_query(r["query_text"]).split()) for r in rows]
    for i in range(len(rows)):
        for j in range(i):
            if tokens[i] and len(tokens[i]&tokens[j])/len(tokens[i]|tokens[j])>=.85:union(i,j)
    for i,row in enumerate(rows):row["split"]="held_out" if digest_int(f"m1-split:{find(i)}")%5==0 else "development"
    rows.sort(key=lambda r:r["silver_query_id"])
    path=OUT/"silver_benchmark.parquet";pq.write_table(pa.Table.from_pylist(rows),path,compression="zstd")
    validation=validate_benchmark(path,official)
    languages=Counter(r["query_language"] for r in rows);splits=Counter(r["split"] for r in rows)
    manifest={"status":"INSUFFICIENT_FOR_ARCHITECTURE_SELECTION","validation":validation,"rows":len(rows),"languages":dict(languages),"splits":dict(splits),
      "benchmark_sha256":file_sha256(path),"positive_semantics":"Known source-document relation from a question-like source title plus extracted answer section.",
      "limitations":["Every query is copied from its positive source title: useful for source-lookup/index regression, not generalization.",
        "No trustworthy English or Vietnamese-to-English positives were found locally.","No organizer hidden gold; official Recall@K remains UNKNOWN."],
      "tracks":{"vi_source_lookup":"available" if languages['vi'] else "insufficient","zh_source_lookup":"available" if languages['zh'] else "insufficient",
        "en":"INSUFFICIENT","vi_to_en":"INSUFFICIENT"},"no_official_query_scoring":True,
      "evaluator":"tools/offline_evaluator.py"}
    atomic_write_json(OUT/"silver_benchmark_manifest.json",manifest);return manifest


def retrieval_reports(inventory:dict)->dict:
    funnel=json.loads((ROOT/"artifacts/phase13_diagnosis/coverage_funnel.json").read_text(encoding="utf-8"))
    score=json.loads((ROOT/"artifacts/phase13_diagnosis/source_score_analysis.json").read_text(encoding="utf-8"))
    displacement=json.loads((ROOT/"artifacts/phase13_diagnosis/ranking_displacement.json").read_text(encoding="utf-8"))
    ownership=json.loads((ROOT/"artifacts/phase13a_fusion/candidate_ownership_audit.json").read_text(encoding="utf-8"))
    result={"no_gold_relevance_claims":True,"stage_definitions":inventory["count_semantics"],"arms":funnel["logical_candidate_funnel"],
      "candidate_overlap":{"documents_with_multiple_arms":ownership["documents_with_multiple_candidate_arms"],
        "query_document_pairs_with_multiple_arms":ownership["query_document_pairs_with_multiple_candidate_arms"],
        "chunks_with_multiple_arms":ownership["chunks_present_in_multiple_candidate_arms"]},
      "score_distributions":score["by_candidate_arm"],"ranking_semantics":score["ranking_semantics"],
      "displacement":displacement["aggregate"],
      "bottleneck_partition":{"absent_from_local_content":{"documents":918883,"relevance":"UNKNOWN"},
        "present_but_not_retrieved":{"measurable_per_arm":True,"relevance":"UNKNOWN; see arms indexed vs unique candidates"},
        "retrieved_but_ranked_below_output":{"measurable_counts":True,"relevance":"UNKNOWN without labels"}}}
    atomic_write_json(OUT/"retrieval_funnel.json",result);return result


def markdowns(inv,dq,silver_manifest,retrieval):
    write_text("architecture_inventory.md",f'''# Kiểm kê kiến trúc hiện tại

## Sự thật triển khai

- Official: **{inv['official']['documents']:,}** URL/ID, không có text trong manifest.
- Local usable union: **275.586 documents / 1.268.343 chunks**; gồm broad-existing 35.690 và PubMed 239.896, không trùng ID.
- Không có một unified index. Focused dùng dense+BM25 theo source; broad-extra dùng global BM25; PubMed dùng dense BGE-M3 document-level.
- Phase 12 rerank cache có 603.587 key trong experiment DB; frozen FULL universe có 695.539 exact query/chunk scores sau reuse pilot.
- Đầu ra luôn chỉ 24.000 doc slots và 48.000 chunk slots (1.200 × Docs20/Chunks40).

## Luồng thật

`official URL -> acquire -> extract/clean -> chunk -> arm-specific index -> candidate -> exact-key reranker -> global max-chunk document ranking -> independent docs/chunks packaging`

Các count ở mỗi mũi tên không đồng nghĩa. `combined_documents.parquet` Phase 12 (111.812 rows) là provenance closure của candidate/ranking, không phải toàn bộ 239.896 PubMed document index.

## Thành phần đã có / còn thiếu

| Thành phần | Trạng thái |
|---|---|
| Resumable acquisition/extraction/chunking | Có, theo experiment/source |
| BGE-M3 dense | Có cho focused chunks và toàn bộ PubMed documents |
| BM25 | Có theo focused source và broad-extra |
| Unified disk-backed multilingual document index | **Chưa có** |
| Exact-key reranker cache | Có |
| Provenance/official/package gates | Có |
| Offline relevance evaluator | Có từ M1, nhưng silver benchmark bị giới hạn |
| Trustworthy VI→EN held-out relevance labels | **Chưa có** |
''')
    write_text("ranking_failure_diagnostics.md",f'''# Chẩn đoán bottleneck retrieval

## Đã xác nhận

1. **Coverage local còn thiếu 918.883 official documents.** English PubMed gần đủ; VI/ZH thiếu nhiều.
2. **Candidate-volume không cân bằng:** PubMed đưa 349.187 query–chunk pairs, mỗi focused source khoảng 19.200. Phase 12 giảm FINAL 0.0492 → 0.0451.
3. **Document rank dùng max chunk:** nguồn/document có nhiều candidate chunks có nhiều cơ hội tạo extreme score. Đây là volume sensitivity, chưa phải bằng chứng calibration sai.
4. PubMed chiếm 5.913 doc slots và 11.592 chunk slots; mọi leaderboard metric giảm. Displacement có hại tổng thể, nhưng relevance từng item vẫn UNKNOWN.
5. Docs và chunks được chọn độc lập hợp lệ theo schema; ownership consistency không đồng nghĩa relevance.

## Chưa thể xác nhận

- Bao nhiêu gold documents vắng khỏi local corpus: **UNKNOWN**.
- Bao nhiêu gold có local nhưng candidate retrieval bỏ sót: **UNKNOWN**.
- Official Recall@20/100 offline: **UNKNOWN** vì không có hidden gold.
- Dense VI→EN hay reranker VI→EN là lỗi chính: **UNKNOWN**; cần held-out bilingual judgments.
''')
    write_text("offline_evaluation_protocol.md",f'''# Protocol đánh giá offline

Silver benchmark có **{silver_manifest['rows']}** known-source pairs, status **{silver_manifest['status']}**.

## Metric hợp lệ

- Known-positive Recall@20/@100, MRR, candidate coverage, source diversity.
- Chỉ chạy bằng `tools/offline_evaluator.py` trên query IDs `silver_*`.

## Không được diễn giải

- Không gọi metric này là official recall/F2.
- Query được lấy nguyên title nguồn (`verbatim_source_query=true`), nên chỉ đo source lookup/index regression, không chứng minh generalization.
- EN và VI→EN tracks hiện **INSUFFICIENT**.
- Unjudged documents là UNKNOWN, không phải irrelevant.

Split theo cluster exact/near-duplicate; một cluster không xuất hiện ở cả development và held-out. Positive phải là NEW-official doc ID và giữ URL/provenance.
''')
    write_text("TARGET_ARCHITECTURE.md",'''# Target architecture — thiết kế sau Milestone 1

1. **Registry chất lượng/corpus:** một bảng doc-level authoritative ghi official membership, acquisition, extraction quality, language, hashes và provenance.
2. **Disk-backed document retrieval:** BM25 sharded + BGE-M3 dense/memmap/ANN có version manifest; không materialize corpus text vào RAM.
3. **Candidate union theo canonical doc ID:** dense, sparse và terminology expansion tạo ranks riêng; deduplicate trước khi lấy bounded representative chunks.
4. **Source-neutral fusion:** RRF/rank features ở doc level; source/arm là feature/audit, không phải magic boost. Candidate volume không được tự tạo lợi thế.
5. **Candidate-only rerank:** exact `(query_id, chunk_id)` cache; giới hạn chunks/document trước inference.
6. **Hai output độc lập hợp schema:** Docs và chunks có ranking riêng, bounded evidence/document, exact source text và provenance closure.
7. **Evaluation-first:** silver regression + human pooled judgments + frozen controls trước model/crawl/leaderboard.

Reuse bắt buộc: Phase 11C control, PubMed documents/embeddings, exact reranker scores, canonical provenance.
''')
    write_text("IMPLEMENTATION_BACKLOG.md",'''# Implementation backlog (xếp theo impact/cost)

| Ưu tiên | Thay đổi | Evidence/benefit | Cost | Acceptance | Rollback |
|---:|---|---|---|---|---|
| 1 | Unified document registry + offline evaluator CI | Count/provenance hiện phân tán; ngăn đánh giá sai | CPU thấp, <1 GiB | silver validation + frozen controls pass; không leakage | giữ artifacts cũ, feature flag |
| 2 | Document-first candidate ledger, cap representative chunks/doc | PubMed 349k pairs vs 19.2k/source; max-chunk volume bias | CPU, reuse scores | arm duplication không tăng score; measured candidate coverage; provenance 100% | quay lại frozen candidate pool |
| 3 | Unified disk-backed BM25 over all 275.586 local docs | broad BM25 và source BM25 đang tách rời | vài GiB disk, giờ CPU; cần approval | deterministic Recall@K trên silver + resource gate | giữ index version cũ |
| 4 | Unified multilingual dense document index reusing vectors | Không có unified dense index; broad-extra thiếu dense | index build CPU/disk; embedding mới chỉ cho thiếu vectors sau approval | silver held-out improves; memory/disk bounds | disable dense arm |
| 5 | Versioned VI/EN/ZH medical terminology expansion | lexical cross-language gap | CPU thấp; dictionary provenance required | improves judged/silver coverage without leakage | disable expansion |
| 6 | Vietnamese acquisition quality-gated expansion | VI local identified chỉ 29.396/714.469 | network/disk/GPU cao; separate milestone | extraction success + offline candidate gain | stop per-domain worker |

Không triển khai source boost/quota tùy ý. Threshold chỉ được chọn trên development và kiểm tra held-out/human labels.
''')


def main()->int:
    OUT.mkdir(parents=True,exist_ok=True)
    official,domains=official_inventory();inv=build_inventory(official,domains);dq=data_quality(official)
    sm=silver(official);rf=retrieval_reports(inv);markdowns(inv,dq,sm,rf)
    summary={"status":"MILESTONE_1_COMPLETE","no_network":True,"no_model_inference":True,"no_index_rebuild":True,
      "inventory_documents":inv["local_content"]["deduplicated_union"]["usable_documents"],"silver_status":sm["status"],
      "outputs":sorted(p.name for p in OUT.iterdir() if p.is_file())}
    atomic_write_json(OUT/"milestone1_summary.json",summary);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
