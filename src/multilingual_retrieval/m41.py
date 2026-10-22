from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psutil
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
POLICY = "m41-relation-qualified-lexical-v1"


def fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def understand(query: str, root: Path = ROOT) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query_vi must be nonempty text")
    base = json.loads((root / "configs/m4_vi_zh_medical_lexicon.json").read_text(encoding="utf-8"))
    extra = json.loads((root / "configs/m41_query_terms.json").read_text(encoding="utf-8"))
    normalized_query = unicodedata.normalize("NFC",query)
    text = fold(normalized_query)
    matches = []
    # Longest non-overlapping phrases prevent 'sốt' inside 'sốt rét',
    # 'hp' inside unrelated words and 'ăn' inside a longer Vietnamese word.
    for category in ("entities", "context", "relations"):
        for phrase, terms in {**base[category], **extra.get(category, {})}.items():
            for match in re.finditer(r"(?<!\w)" + re.escape(fold(phrase)) + r"(?!\w)", text):
                matches.append({"vi": phrase, "zh": terms, "category": category,
                                "start": match.start(), "end": match.end()})
    occupied: set[int] = set()
    selected = []
    for item in sorted(matches, key=lambda x: (-(x["end"] - x["start"]), x["start"], x["category"])):
        positions = set(range(item["start"], item["end"]))
        if positions & occupied:
            continue
        occupied |= positions
        selected.append(item)
    selected.sort(key=lambda x: x["start"])
    # A polar-question suffix is not a negative statement. Any earlier
    # negation remains explicit with its source span, never globally erased.
    suffix = re.search(r"\bkhông\s*[?!.]*$", text)
    polar = bool(suffix and re.search(r"\bcó\b", text[:suffix.start()]))
    negations = []
    for match in re.finditer(r"(?<!\w)(không phải|không|chưa)(?!\w)", text):
        clause_start = max((text.rfind(mark,0,match.start()) for mark in ".!?;\n"),default=-1)+1
        prefix = text[clause_start:match.start()]
        tail = text[match.end():]
        question_end = bool(re.match(r"\s*(?:ạ\s*)?(?:[?!.,;]|$)",tail))
        question_continuation = bool(re.match(r"\s+và\s+(?:điều trị|nên|cách|phải)",tail))
        if match.group() == "không" and re.search(r"\bcó\b",prefix) and (question_end or question_continuation):
            continue
        informational = bool(re.match(r"(?:chưa|không)\s+(?:hiểu|rõ|biết)",text[match.start():]))
        negations.append({"text": normalized_query[match.start():match.end()], "start": match.start(),
                          "end": match.end(), "scope_kind":"INFORMATIONAL" if informational else "CLINICAL_OR_UNRESOLVED",
                          "scope": normalized_query[match.start():min(len(normalized_query), match.end()+45)]})
    # Preserve unhandled words verbatim. They are coverage warnings, not
    # a claim that every residual word is a medical term.
    residual = "".join(" " if i in occupied else char for i, char in enumerate(text))
    stop = set("tôi em bị có là thì và của với cho hỏi ở khi một những các này đó gì thế nào thường xuyên được cần mới nhưng đang liệu không phải ạ hay người mắc lúc ơi vì sao vậy gần nhận kết quả bên mình hiểu chưa".split())
    unknown = list(dict.fromkeys(w for w in re.findall(r"\w+", residual) if w not in stop))
    grouped = {category: [x["zh"] for x in selected if x["category"] == category]
               for category in ("entities", "context", "relations")}
    terms_zh = list(dict.fromkeys(t for x in selected for t in x["zh"]))
    unresolved_negation = any(
        n["scope_kind"]!="INFORMATIONAL" and not any(
            x["vi"].startswith(("không ","chưa ")) and x["start"] <= n["start"] and n["end"] <= x["end"]
            for x in selected)
        for n in negations)
    if unresolved_negation:
        # Untranslated scope is exposed; never attach a global Chinese
        # negation to every entity in a multi-clause medical question.
        unknown = list(dict.fromkeys([*unknown, "UNRESOLVED_NEGATION_SCOPE"]))
    return {"query_vi": query, "query_nfc":normalized_query, "span_basis":"NFC_QUERY", "query_zh": " ".join(terms_zh),
            "terms": selected, "entity_groups": grouped["entities"], "relation_groups": grouped["relations"],
            "qualifier_groups": grouped["context"], "negations": negations, "polar_question": polar,
            "unmapped_query_tokens": unknown, "unsupported_or_ambiguous": unknown,
            "ambiguous": not grouped["entities"] or not grouped["relations"] or "UNRESOLVED_NEGATION_SCOPE" in unknown,
            "coverage": "LIMITED_TERMINOLOGY_UNMAPPED_SPANS" if unknown else "DICTIONARY_TERMS_COVERED",
            "version": extra["version"], "is_general_translation_model": False}


def group_hits(text: str, groups: list[list[str]]) -> int:
    text = fold(text)
    def present(term: str) -> bool:
        value = fold(term)
        if re.search(r"[a-z0-9]", value):
            return bool(re.search(r"(?<![a-z0-9])" + re.escape(value) + r"(?![a-z0-9])", text))
        return value in text
    return sum(any(present(term) for term in group) for group in groups)


def navigation_fragment(text: str) -> bool:
    lines = [part.strip() for part in text.splitlines() if part.strip()]
    return (len(lines) >= 12 and sum(len(line) <= 35 for line in lines)/len(lines) >= 0.75
            and sum(text.count(mark) for mark in ("。", "；", "？")) < 3)


def bigrams(text: str) -> Counter[str]:
    tokens = Counter()
    # Do not manufacture bigrams across unrelated translated concepts.
    for part in re.findall(r"[\u3400-\u9fff]+|[a-z0-9γ-]+", fold(text)):
        tokens.update([part] if len(part) < 2 else [part[i:i+2] for i in range(len(part)-1)])
    return tokens


class ChineseRetriever:
    """Bounded existing-corpus lexical retriever, independent of query IDs/labels."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for index, row in enumerate(rows):
            counts = bigrams(row["evidence_text"])
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((index, count))
        self.average = sum(self.lengths) / max(1, len(rows))

    @classmethod
    def from_root(cls, root: Path = ROOT) -> "ChineseRetriever":
        path = root / "artifacts/multilingual_retrieval_m4/candidate_corpus.parquet"
        rows = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            rows.extend(batch.to_pylist())
        return cls(rows)

    def retrieve(self, query_vi: str, top_k: int = 20, root: Path = ROOT, *, rank_qualifiers: bool = True) -> dict[str, Any]:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        start = time.perf_counter()
        parsed = understand(query_vi, root)
        scores: dict[int, float] = defaultdict(float)
        for term in bigrams(parsed["query_zh"]):
            postings = self.postings.get(term, [])
            inverse = math.log(1 + (len(self.rows)-len(postings)+0.5)/(len(postings)+0.5))
            for index, count in postings:
                scores[index] += inverse * count * 2.2 / (count + 1.2*(0.25+0.75*self.lengths[index]/max(1,self.average)))
        candidates = []
        for index, row in enumerate(self.rows):
            entity = group_hits(row["evidence_text"], parsed["entity_groups"])
            # Core concept matches are retained even if the lexical score is zero.
            if not entity and index not in scores:
                continue
            relation = group_hits(row["evidence_text"], parsed["relation_groups"])
            qualifier = group_hits(row["evidence_text"], parsed["qualifier_groups"])
            candidates.append({**row, "entity_hits": entity, "entity_total": len(parsed["entity_groups"]),
                               "relation_hits": relation, "qualifier_hits": qualifier,
                               "lexical_score": scores.get(index, 0.0), "branch": POLICY,
                               "answer_relevance": "UNVERIFIED"})
        candidates.sort(key=lambda r: (-r["entity_hits"]/max(1,r["entity_total"]),
                                      -int(r["page_type"] == "MEDICAL_ARTICLE"),
                                      -r["qualifier_hits"] if rank_qualifiers else 0, -int(r["relation_hits"] > 0),
                                      -r["lexical_score"], int(r["doc_id"]), str(r["chunk_id"])))
        docs, trace, final = set(), [], []
        for row in candidates:
            if int(row["doc_id"]) in docs:
                continue
            docs.add(int(row["doc_id"]))
            reasons = []
            if parsed["ambiguous"]:
                reasons.append("QUERY_UNDERSTANDING_INCOMPLETE")
            if not row["entity_total"] or row["entity_hits"] < row["entity_total"]:
                reasons.append("MISSING_REQUIRED_ENTITY")
            if not row["relation_hits"]:
                reasons.append("NO_MATCHED_RELATION")
            if row["page_type"] != "MEDICAL_ARTICLE":
                reasons.append("PAGE_TYPE_NOT_ARTICLE")
            if navigation_fragment(row["evidence_text"]):
                reasons.append("NAVIGATION_OR_LIST_FRAGMENT")
            record = {**row, "candidate_rank": len(trace)+1, "admission_reasons": reasons}
            trace.append(record)
            if not reasons and len(final) < top_k:
                final.append({**record, "rank": len(final)+1})
        return {"understanding": parsed, "candidates": trace, "results": final,
                "elapsed_seconds": time.perf_counter()-start, "policy": POLICY,
                "admission_claim": "lexical compatibility screen; answer relevance requires review"}


def retrieve_by_text(query_vi: str, top_k: int = 20, *, root: Path = ROOT,
                     retriever: ChineseRetriever | None = None) -> dict[str, Any]:
    return (retriever or ChineseRetriever.from_root(root)).retrieve(query_vi, top_k, root)


def semantic_preflight(root: Path = ROOT) -> dict[str, Any]:
    active = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(proc.info["cmdline"] or [])
            if (proc.info["name"] or "").lower().startswith("python") and re.search(r"tools[/\\]phase27_", command):
                active.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"status": "DEFERRED_PHASE27_ACTIVE" if active else "REQUIRES_SEMANTIC_RESOURCE_TRIAL",
            "phase27_pids": active, "available_mib": psutil.virtual_memory().available/2**20,
            "free_disk_gib": psutil.disk_usage(root.anchor).free/2**30,
            "model_loaded": False, "inference_started": False}
