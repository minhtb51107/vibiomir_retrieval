#!/usr/bin/env python3
"""Cache-only PubMed comparisons and approval-gated multilingual judging.

The reconstruction path never calls a model. The pilot uses Hugging Face's
OpenAI-compatible router only after explicit free-credit and policy gates.
Secrets are read from the process environment at runtime and never persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.union_experiment import _iter_json_array, _query_groups

BEST_JSON = ROOT / "submissions/00_READY_TO_UPLOAD/phase16_S4_FULL_MEDLATEC_FULL_DOCS20_CHUNKS40.json"
BEST_RANK = ROOT / "data/source_census/phase16_72h_coverage/overnight/union_experiment/rankings/UNION/reranked_chunks.parquet"
BEST_SCORES = ROOT / "data/source_census/phase16_72h_coverage/overnight/union_experiment/round1/source_scores.sqlite"
BEST_CANONICAL = ROOT / "data/source_census/phase16_72h_coverage/overnight/union_experiment/round1/canonical_text.sqlite"
PILOT_SCORES = ROOT / "data/retrieval/phase7_reranking/rerank_scores.sqlite"
GUARDED = ROOT / "artifacts/phase16_pubmed_guarded_fusion/guarded_documents.parquet"
PUBMED_RANK = ROOT / "data/pubmed_phase12/retrieval/rankings/FULL/reranked_chunks.parquet"
PUBMED_SCORES = ROOT / "data/pubmed_phase12/retrieval/round1/source_scores.sqlite"
PUBMED_DOCS = ROOT / "data/pubmed_phase12/processed/documents.parquet"
PUBMED_CANONICAL = ROOT / "data/pubmed_phase12/retrieval/round1/canonical_text.sqlite"
QUERIES = ROOT / "data/raw/query.parquet"
OFFICIAL = ROOT / "data/raw/links_corpus_vi_en_zh.parquet"
OUT = ROOT / "artifacts/phase16_pubmed_semantic_recovery"
COMPARISONS = OUT / "candidate_comparisons.parquet"
MANIFEST = OUT / "comparison_manifest.json"
JUDGE_CACHE = OUT / "semantic_judgments.sqlite"
PROMPT_VERSION = "pubmed_comparative_v1"
HF_ENDPOINT = "https://router.huggingface.co/v1"
HF_WHOAMI = "https://huggingface.co/api/whoami-v2"
HF_MODEL_API = "https://huggingface.co/api/models/Qwen/Qwen3-14B?expand%5B%5D=inferenceProviderMapping"
MODEL = "Qwen/Qwen3-14B"
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
PROVIDER = "deepinfra"
ROUTED_MODEL = f"{MODEL}:{PROVIDER}"
APPROVAL = "USE_FREE_HF_CREDIT_QWEN3_14B_DEEPINFRA"
INPUT_USD_PER_MILLION = 0.12
OUTPUT_USD_PER_MILLION = 0.24
MAX_PILOT_CALLS = 60
POLICY_URLS = {
    "hf_pricing": "https://huggingface.co/docs/inference-providers/en/pricing",
    "hf_security": "https://huggingface.co/docs/inference-providers/main/en/security",
    "hf_deepinfra": "https://huggingface.co/docs/inference-providers/providers/deepinfra",
    "organizer_rules": "https://leaderboard.aiguru.com.vn/api/competitions/15/",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ro(path: Path, immutable: bool = True) -> sqlite3.Connection:
    suffix = "&immutable=1" if immutable else ""
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro{suffix}", uri=True)


def language_signal(text: str) -> str:
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return "zh"
    lower = f" {text.lower()} "
    vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    if any(c in lower for c in vi_chars):
        return "vi"
    return "en" if any("a" <= c <= "z" for c in lower) else "unknown"


def score_rows(db: sqlite3.Connection, query_id: int) -> dict[str, tuple[str | None, str | None, float]]:
    return {str(c): (str(q), str(t), float(s)) for c, q, t, s in db.execute(
        "SELECT chunk_id,query_text,chunk_text,rerank_score FROM pairs WHERE query_id=?", (query_id,)
    )}


def pilot_rows(db: sqlite3.Connection, query_id: int) -> dict[str, tuple[None, None, float]]:
    return {str(c): (None, None, float(s)) for c, s in db.execute(
        "SELECT chunk_id,rerank_score FROM scores WHERE query_id=?", (query_id,)
    )}


def exact_support(rows: list[dict], doc_id: int, query: str,
                  scores: dict[str, tuple[str | None, str | None, float]], query_id: int) -> dict:
    best = None
    for row in rows:
        if int(row["doc_id"]) != doc_id:
            continue
        chunk_id = str(row["chunk_id"]); cached = scores.get(chunk_id)
        if cached is None:
            raise ValueError(f"missing score {query_id}:{chunk_id}")
        if cached[0] is not None and (cached[0] != query or cached[1] != str(row["chunk_text"])):
            raise ValueError(f"exact text mismatch {query_id}:{chunk_id}")
        if not math.isfinite(cached[2]):
            raise ValueError(f"non-finite score {query_id}:{chunk_id}")
        candidate = (cached[2], -int(row["rerank_rank"]), chunk_id, str(row["chunk_text"]))
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise ValueError(f"no supporting chunk query={query_id} doc={doc_id}")
    return {"score": best[0], "chunk_id": best[2], "chunk_text": best[3]}


def load_pubmed_documents(wanted: set[int]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    columns = ["doc_id", "original_url", "title", "language_signal", "normalized_text", "abstract_text", "pmid"]
    for batch in pq.ParquetFile(PUBMED_DOCS).iter_batches(columns=columns, batch_size=32768):
        for row in pa.Table.from_batches([batch]).to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in wanted:
                result[doc_id] = row
    if set(result) != wanted:
        raise ValueError(f"PubMed document closure failed: missing={sorted(wanted-set(result))[:10]}")
    return result


def load_official(wanted: set[int]) -> dict[int, str]:
    result: dict[int, str] = {}
    pf = pq.ParquetFile(OFFICIAL)
    names = set(pf.schema.names); id_col = "id" if "id" in names else "doc_id"
    url_col = "url" if "url" in names else "official_url"
    for batch in pf.iter_batches(columns=[id_col, url_col], batch_size=65536):
        for doc_id, url in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            value = int(doc_id)
            if value in wanted:
                result[value] = str(url)
    if set(result) != wanted:
        raise ValueError(f"official ID closure failed: missing={sorted(wanted-set(result))[:10]}")
    return result


def canonical_check(db: sqlite3.Connection, support: dict, doc_id: int) -> None:
    row = db.execute("SELECT doc_id,raw_text FROM chunks WHERE chunk_id=?", (support["chunk_id"],)).fetchone()
    if row is None or int(row[0]) != doc_id or str(row[1]) != support["chunk_text"]:
        raise ValueError(f"canonical chunk mismatch {support['chunk_id']}")
    doc = db.execute("SELECT normalized_text FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    if doc is None or support["chunk_text"] not in str(doc[0]):
        raise ValueError(f"supporting evidence absent from canonical document {doc_id}")


def reconstruct() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    query_table = pq.read_table(QUERIES, columns=["id", "query"])
    query_text = dict(zip(map(int, query_table.column(0).to_pylist()), map(str, query_table.column(1).to_pylist()), strict=True))
    frozen = {int(row["id"]): list(map(int, row["relevant_docs"])) for row in _iter_json_array(BEST_JSON)}
    guarded_groups = {int(g[0]["query_id"]): sorted(g, key=lambda x: int(x["rank"])) for g in _query_groups(GUARDED)}
    pairs = []
    for query_id, rows in guarded_groups.items():
        old = frozen[query_id]
        inserted = [r for r in rows if str(r["source"]) == "PUBMED"]
        displaced = [(rank, doc) for rank, doc in enumerate(old, 1) if doc not in {int(r["doc_id"]) for r in rows}]
        if len(inserted) != len(displaced):
            raise ValueError(f"replacement cardinality mismatch query={query_id}")
        for replacement_index, (new, (old_rank, old_doc)) in enumerate(zip(inserted, displaced, strict=True), 1):
            pairs.append({"query_id": query_id, "query_text": query_text[query_id],
                          "query_language": language_signal(query_text[query_id]),
                          "replacement_index": replacement_index, "pubmed_rank": int(new["rank"]),
                          "pubmed_doc_id": int(new["doc_id"]), "guarded_document_score": float(new["score"]),
                          "best_original_rank": old_rank, "best_doc_id": old_doc})
    if len(pairs) != 1508 or len({p["query_id"] for p in pairs}) != 835:
        raise ValueError(f"guarded comparison counts changed pairs={len(pairs)} queries={len({p['query_id'] for p in pairs})}")

    pub_docs = load_pubmed_documents({p["pubmed_doc_id"] for p in pairs})
    official = load_official({p["pubmed_doc_id"] for p in pairs} | {p["best_doc_id"] for p in pairs})
    by_query = {}
    for pair in pairs:
        by_query.setdefault(pair["query_id"], []).append(pair)
    best_db, pub_db, pilot_db = ro(BEST_SCORES), ro(PUBMED_SCORES), ro(PILOT_SCORES)
    best_can, pub_can = ro(BEST_CANONICAL), ro(PUBMED_CANONICAL)
    best_groups, pub_groups = iter(_query_groups(BEST_RANK)), iter(_query_groups(PUBMED_RANK))
    completed = []
    try:
        for best_rows, pub_rows in zip(best_groups, pub_groups, strict=True):
            query_id = int(best_rows[0]["query_id"])
            if query_id != int(pub_rows[0]["query_id"]):
                raise ValueError("ranking query order mismatch")
            if query_id not in by_query:
                continue
            best_scores = pilot_rows(pilot_db, query_id)
            for key, value in score_rows(best_db, query_id).items():
                if key in best_scores and best_scores[key][2] != value[2]:
                    raise ValueError(f"conflicting BEST caches {query_id}:{key}")
                best_scores[key] = value
            pub_scores = score_rows(pub_db, query_id)
            for pair in by_query[query_id]:
                bs = exact_support(best_rows, pair["best_doc_id"], pair["query_text"], best_scores, query_id)
                ps = exact_support(pub_rows, pair["pubmed_doc_id"], pair["query_text"], pub_scores, query_id)
                canonical_check(best_can, bs, pair["best_doc_id"]); canonical_check(pub_can, ps, pair["pubmed_doc_id"])
                document = pub_docs[pair["pubmed_doc_id"]]
                if abs(ps["score"] - pair["guarded_document_score"]) > 1e-12:
                    raise ValueError(f"PubMed document score mismatch query={query_id} doc={pair['pubmed_doc_id']}")
                completed.append({**pair,
                    "pubmed_url": official[pair["pubmed_doc_id"]], "pubmed_domain": urlparse(official[pair["pubmed_doc_id"]]).netloc.lower(),
                    "pubmed_title": str(document.get("title") or ""), "pubmed_abstract": str(document.get("abstract_text") or ""),
                    "pubmed_language": str(document.get("language_signal") or "unknown"), "pubmed_chunk_id": ps["chunk_id"],
                    "pubmed_chunk_text": ps["chunk_text"], "pubmed_exact_score": ps["score"],
                    "best_url": official[pair["best_doc_id"]], "best_domain": urlparse(official[pair["best_doc_id"]]).netloc.lower(),
                    "best_language": language_signal(bs["chunk_text"]), "best_chunk_id": bs["chunk_id"],
                    "best_chunk_text": bs["chunk_text"], "best_exact_score": bs["score"]})
    finally:
        for connection in (best_db, pub_db, pilot_db, best_can, pub_can): connection.close()
    completed.sort(key=lambda x: (x["query_id"], x["replacement_index"]))
    tmp = COMPARISONS.with_name(f".{COMPARISONS.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(completed), tmp, compression="zstd"); os.replace(tmp, COMPARISONS)
    report = {"passed": True, "pairs": len(completed), "changed_queries": len({r['query_id'] for r in completed}),
              "query_languages": dict(Counter(r["query_language"] for r in completed)),
              "all_supporting_chunks_canonical": True, "all_scores_exact": True,
              "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in (BEST_JSON, GUARDED, BEST_RANK, PUBMED_RANK, PUBMED_DOCS)},
              "output": str(COMPARISONS.relative_to(ROOT)), "output_sha256": sha256(COMPARISONS),
              "prompt_version": PROMPT_VERSION, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    MANIFEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def prompt_for(row: dict, swap: bool = False) -> str:
    pubmed = {"label": "candidate", "title": row["pubmed_title"], "abstract": row["pubmed_abstract"], "supporting_chunk": row["pubmed_chunk_text"]}
    best = {"label": "candidate", "url": row["best_url"], "supporting_chunk": row["best_chunk_text"]}
    first, second = (best, pubmed) if swap else (pubmed, best)
    return json.dumps({"task": "Compare which candidate better answers the original biomedical question. Judge only supplied evidence; abstain when insufficient.",
        "checks": ["same disease/topic", "same population", "same intent type", "actual answer evidence", "which is better"],
        "query": row["query_text"], "candidate_A": first, "candidate_B": second}, ensure_ascii=False, separators=(",", ":"))


def conservative_token_estimate(text: str) -> int:
    # Conservative without adding a tokenizer dependency; Vietnamese UTF-8 and
    # JSON punctuation commonly exceed the English 4-char/token heuristic.
    return math.ceil(len(text) / 2.5)


def estimate() -> dict:
    rows = pq.read_table(COMPARISONS).to_pylist()
    tokens = [conservative_token_estimate(prompt_for(row)) for row in rows]
    output_tokens_each = 420
    input_tokens = sum(tokens); output_tokens = len(rows) * output_tokens_each
    # Conservative local ceiling. HF routed requests use the provider price
    # without HF markup, while remaining credit is visible only in Billing
    # Settings; runtime therefore requires operator-confirmed free credit.
    standard_usd = input_tokens / 1e6 * INPUT_USD_PER_MILLION + output_tokens / 1e6 * OUTPUT_USD_PER_MILLION
    pilot_n = min(40, len(rows)); swap_n = min(20, pilot_n)
    pilot_input = sum(tokens[:pilot_n]) + sum(tokens[:swap_n])
    pilot_output = (pilot_n + swap_n) * output_tokens_each
    pilot_usd = pilot_input / 1e6 * INPUT_USD_PER_MILLION + pilot_output / 1e6 * OUTPUT_USD_PER_MILLION
    result = {"router":"Hugging Face Inference Providers", "endpoint":HF_ENDPOINT,
              "endpoint_provider":PROVIDER, "routed_model":ROUTED_MODEL, "model": MODEL, "model_revision":MODEL_REVISION,
              "model_parameters":14768307200, "license":"apache-2.0", "release":"2025-04-28",
              "competition_eligibility_basis":"public weights; 14.768B <= 15B; released before 2026-08-01",
              "policy_urls":POLICY_URLS,
              "data_handling_summary":"HF states request/response bodies are not stored while routing and are not used for training; provider policy also applies; TLS in transit.",
              "credit_constraint":"No documented balance API; operator confirmation is mandatory and the client stops at the lower of confirmed balance and local cap.",
              "provider_controls":{"fallbacks":False,"provider_pinned_by_model_suffix":PROVIDER},
              "pricing_verified_date": "2026-10-18", "pricing_ceiling_usd_per_million": {"input": INPUT_USD_PER_MILLION, "output": OUTPUT_USD_PER_MILLION},
              "method": "conservative character estimate (ceil(chars/2.5)); no API calls", "pairs": len(rows),
              "projected_input_tokens": input_tokens, "projected_output_tokens": output_tokens,
              "projected_standard_usd": round(standard_usd, 4), "projected_standard_vnd_at_27000": round(standard_usd * 27000),
              "reliability_pilot": {"primary_pairs": pilot_n, "order_swaps": swap_n, "projected_usd": round(pilot_usd, 4), "projected_vnd_at_27000": round(pilot_usd * 27000)},
              "full_run_allowed":False, "paid_calls_made": 0}
    (OUT / "api_cost_preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


JUDGE_SCHEMA = {"type":"object","properties":{
    "preference":{"type":"string","enum":["A","B","TIE","ABSTAIN"]}, "confidence":{"type":"number","minimum":0,"maximum":1},
    "topic_match":{"type":"string","enum":["MATCH","MISMATCH","UNKNOWN"]},
    "population_match":{"type":"string","enum":["MATCH","MISMATCH","UNKNOWN"]},
    "intent_match":{"type":"string","enum":["MATCH","MISMATCH","UNKNOWN"]},
    "critical_mismatch":{"type":"boolean"},
    "evidence_spans":{"type":"array","items":{"type":"object","properties":{"source":{"type":"string","enum":["QUERY","A","B"]},"exact_quote":{"type":"string"}},"required":["source","exact_quote"],"additionalProperties":False}},
    "mismatch_reasons":{"type":"array","items":{"type":"string"}}, "abstain":{"type":"boolean"}},
    "required":["preference","confidence","topic_match","population_match","intent_match","critical_mismatch","evidence_spans","mismatch_reasons","abstain"],"additionalProperties":False}


def validate_spans(result: dict, prompt_payload: dict) -> bool:
    sources = {"QUERY": prompt_payload["query"], "A": json.dumps(prompt_payload["candidate_A"], ensure_ascii=False), "B": json.dumps(prompt_payload["candidate_B"], ensure_ascii=False)}
    cited = {span["source"] for span in result["evidence_spans"]}
    required = set() if result.get("abstain") else {"A", "B"}
    return required <= cited and all(span["exact_quote"] and span["exact_quote"] in sources[span["source"]] for span in result["evidence_spans"])


def _request_json(url: str, token: str, *, payload: dict | None = None,
                  timeout: int = 180, retries: int = 2) -> dict:
    """Issue a bounded HF request without returning or logging credentials."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 402, 403):
                raise RuntimeError(f"HF request rejected with HTTP {exc.code}") from None
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"HF request failed with HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            if attempt == retries:
                raise RuntimeError(f"HF network request failed: {type(exc.reason).__name__}") from None
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _provider_is_live(metadata: dict, provider: str) -> bool:
    mapping = metadata.get("inferenceProviderMapping") or {}
    if isinstance(mapping, list):
        for item in mapping:
            if not isinstance(item, dict) or str(item.get("provider")) != provider:
                continue
            return str(item.get("status", "live")).lower() in {"live", "available", "ready"}
        return False
    value = mapping.get(provider) if isinstance(mapping, dict) else None
    if value is None:
        return False
    if not isinstance(value, dict):
        return True
    return str(value.get("status", "live")).lower() in {"live", "available", "ready"}


def online_preflight(token: str, confirmed_free_credit_usd: float,
                     projected_usd: float, accepted_policies: bool) -> dict:
    if not accepted_policies:
        raise SystemExit("policy gate failed; pass --accept-policies after reviewing the recorded URLs")
    if confirmed_free_credit_usd <= 0:
        raise SystemExit("credit gate failed; confirm remaining free credit in HF Billing Settings")
    if projected_usd > confirmed_free_credit_usd:
        raise SystemExit(f"credit gate failed projected_usd={projected_usd:.6f} confirmed_free_credit_usd={confirmed_free_credit_usd:.6f}")
    account = _request_json(HF_WHOAMI, token, timeout=30)
    metadata = _request_json(HF_MODEL_API, token, timeout=30)
    if metadata.get("sha") != MODEL_REVISION:
        raise SystemExit(f"model revision gate failed expected={MODEL_REVISION} actual={metadata.get('sha')}")
    if not _provider_is_live(metadata, PROVIDER):
        raise SystemExit(f"provider gate failed: {PROVIDER} is not live for {MODEL}")
    return {
        "authenticated": True,
        "account": str(account.get("name") or account.get("fullname") or "authenticated"),
        "endpoint": HF_ENDPOINT,
        "routed_model": ROUTED_MODEL,
        "provider_pinned": PROVIDER,
        "fallback_allowed": False,
        "model_revision": metadata.get("sha"),
        "confirmed_free_credit_usd": confirmed_free_credit_usd,
        "projected_pilot_usd": projected_usd,
        "policy_urls": POLICY_URLS,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _parse_json_content(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("provider returned no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise RuntimeError("provider JSON is not an object")
    return value


def call_provider(prompt: str, token: str) -> tuple[dict, dict]:
    payload = {"model": ROUTED_MODEL, "temperature":0, "seed":1600694, "max_tokens":900,
               "messages": [{"role":"system","content":"You are a conservative multilingual biomedical retrieval judge. Use only supplied text. Prefer ABSTAIN when evidence is insufficient. Output exact evidence quotes only."}, {"role":"user","content":prompt}],
               "response_format":{"type":"json_schema","json_schema":{"name":"pubmed_comparison","strict":True,"schema":JUDGE_SCHEMA}}}
    body = _request_json(f"{HF_ENDPOINT}/chat/completions", token, payload=payload)
    choices=body.get("choices",[])
    if len(choices)!=1 or not choices[0].get("message",{}).get("content"):
        raise RuntimeError(f"unexpected provider output id={body.get('id')}")
    return _parse_json_content(choices[0]["message"]["content"]), body.get("usage", {})


def connection_smoke(token: str) -> dict:
    payload = {"model": ROUTED_MODEL, "temperature":0, "max_tokens":8,
               "messages":[{"role":"user","content":"Reply with exactly READY."}]}
    body = _request_json(f"{HF_ENDPOINT}/chat/completions", token, payload=payload)
    choices = body.get("choices", [])
    content = choices[0].get("message", {}).get("content", "") if len(choices) == 1 else ""
    if "READY" not in content.upper():
        raise RuntimeError("HF connectivity smoke returned an unexpected response")
    return body.get("usage", {})


def init_cache() -> sqlite3.Connection:
    db = sqlite3.connect(JUDGE_CACHE); db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS judgments(cache_key TEXT PRIMARY KEY,query_id INTEGER,pubmed_doc_id INTEGER,best_doc_id INTEGER,model TEXT,prompt_version TEXT,swap INTEGER,result_json TEXT,input_tokens INTEGER,output_tokens INTEGER,created_at TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value_json TEXT NOT NULL)")
    return db


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1e6 * INPUT_USD_PER_MILLION + output_tokens / 1e6 * OUTPUT_USD_PER_MILLION


def _validate_result_shape(result: dict) -> bool:
    return set(JUDGE_SCHEMA["required"]) <= set(result) and result.get("preference") in {"A", "B", "TIE", "ABSTAIN"}


def judge(sample_size: int, swaps: int, approval: str, max_cost_usd: float,
          confirmed_free_credit_usd: float, accepted_policies: bool) -> dict:
    if approval != APPROVAL:
        raise SystemExit(f"inference calls disabled; pass --approve-free-credit {APPROVAL}")
    if sample_size != 40 or swaps != 20:
        raise SystemExit("bounded pilot contract requires exactly 40 primary pairs and 20 A/B swaps")
    if sample_size + swaps > MAX_PILOT_CALLS:
        raise SystemExit(f"pilot call cap exceeded: {sample_size + swaps}>{MAX_PILOT_CALLS}")
    preflight = estimate()
    projected_usd = float(preflight["reliability_pilot"]["projected_usd"])
    if max_cost_usd <= 0 or projected_usd > max_cost_usd:
        raise SystemExit(f"local cost gate failed projected_usd={projected_usd:.6f} max_cost_usd={max_cost_usd:.6f}")
    try:
        token = os.environ["HF_TOKEN"]  # Runtime-only: never persist or print.
    except KeyError:
        raise SystemExit("HF_TOKEN is not set in this Python process") from None
    if not token.strip():
        raise SystemExit("HF_TOKEN is empty in this Python process")
    runtime_preflight = online_preflight(token, confirmed_free_credit_usd,
                                         projected_usd, accepted_policies)
    _atomic_json(OUT / "hf_runtime_preflight.json", runtime_preflight)
    rows = pq.read_table(COMPARISONS).to_pylist(); rng = random.Random(1600694); rng.shuffle(rows); rows = rows[:sample_size]
    work = [(row, False) for row in rows] + [(row, True) for row in rows[:swaps]]
    db = init_cache(); made = hits = invalid = input_tokens = output_tokens = 0; selected_keys=[]
    try:
        smoke_key = f"smoke:{HF_ENDPOINT}:{ROUTED_MODEL}:{MODEL_REVISION}"
        if db.execute("SELECT 1 FROM metadata WHERE key=?", (smoke_key,)).fetchone() is None:
            usage = connection_smoke(token)
            input_tokens += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
            output_tokens += int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
            value = {"passed":True,"usage":usage,"created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
            db.execute("INSERT INTO metadata(key,value_json) VALUES(?,?)", (smoke_key,json.dumps(value,separators=(",",":"))))
            db.commit()
        for row, swap in work:
            prompt = prompt_for(row, swap)
            key = hashlib.sha256((HF_ENDPOINT+ROUTED_MODEL+MODEL_REVISION+PROMPT_VERSION+prompt).encode()).hexdigest()
            selected_keys.append((key,row,swap))
            if db.execute("SELECT 1 FROM judgments WHERE cache_key=?",(key,)).fetchone(): hits += 1; continue
            request_ceiling = _cost_usd(conservative_token_estimate(prompt), 900)
            if _cost_usd(input_tokens, output_tokens) + request_ceiling > min(max_cost_usd, confirmed_free_credit_usd):
                raise RuntimeError("runtime credit cap cannot safely cover the next request")
            result, usage = call_provider(prompt, token); payload = json.loads(prompt)
            if not _validate_result_shape(result):
                raise RuntimeError(f"invalid structured result query={row['query_id']}")
            if not validate_spans(result, payload): result = {**result,"preference":"ABSTAIN","confidence":0.0,"abstain":True,"mismatch_reasons":list(result.get("mismatch_reasons",[]))+["INVALID_EVIDENCE_SPAN"]}; invalid += 1
            it=int(usage.get("input_tokens",usage.get("prompt_tokens",0))); ot=int(usage.get("output_tokens",usage.get("completion_tokens",0))); input_tokens+=it; output_tokens+=ot
            db.execute("INSERT INTO judgments VALUES(?,?,?,?,?,?,?,?,?,?,?)",(key,int(row["query_id"]),int(row["pubmed_doc_id"]),int(row["best_doc_id"]),ROUTED_MODEL,PROMPT_VERSION,int(swap),json.dumps(result,ensure_ascii=False),it,ot,time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))); db.commit(); made += 1
            _atomic_json(OUT/"pilot_progress.json", {"status":"RUNNING","done":made+hits,"total":len(work),"api_calls_this_run":made,"cache_hits":hits,"last_query_id":int(row["query_id"]),"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
        results={key:json.loads(db.execute("SELECT result_json FROM judgments WHERE cache_key=?",(key,)).fetchone()[0]) for key,_,_ in selected_keys}
    finally: db.close()
    def normalized(result, swap):
        preference=result["preference"]
        if preference in ("TIE","ABSTAIN"): return preference
        return "PUBMED" if (preference=="B" if swap else preference=="A") else "BEST"
    primary={(int(row["query_id"]),int(row["pubmed_doc_id"]),int(row["best_doc_id"])):normalized(results[key],False) for key,row,swap in selected_keys if not swap}
    controls=[]
    for key,row,swap in selected_keys:
        if swap:
            identity=(int(row["query_id"]),int(row["pubmed_doc_id"]),int(row["best_doc_id"])); controls.append(primary[identity]==normalized(results[key],True))
    agreement=sum(controls)/len(controls) if controls else None
    classifications = Counter(); mismatch_counts = Counter(); confidences = []
    for key,row,swap in selected_keys:
        if swap: continue
        result=results[key]; confidences.append(float(result.get("confidence",0)))
        for field,name in (("topic_match","topic"),("population_match","population"),("intent_match","intent")):
            if result.get(field)=="MISMATCH": mismatch_counts[name]+=1
        if result.get("critical_mismatch"): mismatch_counts["critical"]+=1
        preference=normalized(result,False)
        accepted=(preference=="PUBMED" and float(result.get("confidence",0))>=0.85
                  and result.get("topic_match")=="MATCH" and result.get("intent_match")=="MATCH"
                  and result.get("population_match")!="MISMATCH" and not result.get("critical_mismatch")
                  and not result.get("abstain"))
        if accepted: classifications["ACCEPT"]+=1
        elif preference in {"TIE","ABSTAIN"} or result.get("abstain"): classifications["ABSTAIN"]+=1
        else: classifications["REJECT"]+=1
    invalid_fraction=invalid/max(1,len(selected_keys))
    stable=agreement is not None and agreement>=0.90 and invalid_fraction<=0.05
    actual_usd=_cost_usd(input_tokens,output_tokens)
    report={"status":"PASS_RELIABILITY_PILOT" if stable else "BLOCKED_UNSTABLE_JUDGE", "calls_made":made,"cache_hits":hits,
            "invalid_spans_forced_to_abstain":invalid,"input_tokens":input_tokens,"output_tokens":output_tokens,
            "actual_cost_usd":round(actual_usd,6),"actual_cost_vnd_at_27000":round(actual_usd*27000),
            "primary_pairs":sample_size,"order_controls":len(controls),"order_agreement":agreement,"invalid_span_fraction":invalid_fraction,
            "classifications":dict(classifications),"medical_mismatches":dict(mismatch_counts),
            "query_language_distribution":dict(Counter(str(r["query_language"]) for r in rows)),
            "cross_language_scope":"All sampled competition queries are VI and PubMed evidence is EN; this diagnoses judge bias, not relevance ground truth.",
            "mean_confidence":sum(confidences)/len(confidences),
            "acceptance":{"minimum_order_agreement":0.90,"maximum_invalid_span_fraction":0.05},
            "full_1508_run_allowed":False,"router":HF_ENDPOINT,"routed_model":ROUTED_MODEL,"provider_fallback":False}
    _atomic_json(OUT/"judge_pilot_report.json",report)
    _atomic_json(OUT/"pilot_progress.json", {"status":report["status"],"done":len(work),"total":len(work),"api_calls_this_run":made,"cache_hits":hits,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    return report


def main() -> int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("reconstruct"); sub.add_parser("estimate")
    run=sub.add_parser("judge-pilot"); run.add_argument("--sample-size",type=int,default=40); run.add_argument("--order-swaps",type=int,default=20)
    run.add_argument("--approve-free-credit",default=""); run.add_argument("--confirmed-free-credit-usd",type=float,default=0.0)
    run.add_argument("--max-cost-usd",type=float,default=0.0); run.add_argument("--accept-policies",action="store_true")
    args=parser.parse_args()
    result=reconstruct() if args.command=="reconstruct" else estimate() if args.command=="estimate" else judge(args.sample_size,args.order_swaps,args.approve_free_credit,args.max_cost_usd,args.confirmed_free_credit_usd,args.accept_policies)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
