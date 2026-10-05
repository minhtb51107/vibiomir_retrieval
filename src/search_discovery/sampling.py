from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold()).replace("đ", "d")
    return "".join(ch for ch in value if not unicodedata.combining(ch))


CATEGORY_TERMS = {
    "drug": ("thuoc", "lieu", "uong", "tiem", "khang sinh", "vitamin"),
    "symptom": ("dau", "sot", "ho", "ngua", "met", "chay mau", "trieu chung"),
    "diagnosis_test": ("xet nghiem", "chup", "sieu am", "chan doan", "mri", "ct"),
    "treatment": ("dieu tri", "chua", "phau thuat", "tap luyen", "quan ly"),
    "anatomy_physiology": ("tim", "gan", "than", "phoi", "nao", "xuong", "mau"),
    "international_term": ("covid", "hiv", "hpv", "mri", "ct", "pcr", "sars"),
    "disease": ("benh", "viem", "ung thu", "tieu duong", "tang huyet ap"),
}


def category_for(text: str) -> str:
    folded = _fold(text)
    for category in (
        "drug", "diagnosis_test", "treatment", "symptom",
        "international_term", "anatomy_physiology", "disease",
    ):
        if any(term in folded for term in CATEGORY_TERMS[category]):
            return category
    return "general"


def _evidence_proxy(
    dense_path: str | Path, sparse_path: str | Path, *, top_k: int
) -> dict[int, dict[str, Any]]:
    methods = []
    for path in (dense_path, sparse_path):
        table = pq.read_table(path, columns=["query_id", "rank", "doc_id"])
        by_query: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row in table.to_pylist():
            if int(row["rank"]) <= top_k:
                by_query[int(row["query_id"])].append((int(row["rank"]), int(row["doc_id"])))
        methods.append(by_query)
    output = {}
    for query_id in set(methods[0]) | set(methods[1]):
        dense = sorted(methods[0].get(query_id, []))
        sparse = sorted(methods[1].get(query_id, []))
        dense_docs = {doc for _, doc in dense}
        sparse_docs = {doc for _, doc in sparse}
        overlap = len(dense_docs & sparse_docs)
        top1_same = bool(dense and sparse and dense[0][1] == sparse[0][1])
        label = "strong" if top1_same or overlap >= 3 else "weak" if overlap == 0 else "mixed"
        output[query_id] = {"label": label, "top10_doc_overlap": overlap, "top1_same": top1_same}
    return output


def _stable_order(query_id: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest()


def _round_robin(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["length_quantile"], row["category"], row["evidence_proxy"])
        buckets[key].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: _stable_order(int(row["query_id"]), seed))
    selected = []
    keys = sorted(
        buckets,
        key=lambda key: hashlib.sha256(f"bucket:{seed}:{key!r}".encode()).hexdigest(),
    )
    while len(selected) < min(count, len(rows)):
        progress = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
                progress = True
        if not progress:
            break
    return selected


def build_stratified_sample(config: dict[str, Any]) -> dict[str, Any]:
    sample = config["sampling"]
    query_table = pq.read_table(config["inputs"]["queries"], columns=["id", "query"])
    queries = [(int(row["id"]), str(row["query"])) for row in query_table.to_pylist()]
    ordered_lengths = sorted(len(text) for _, text in queries)
    quantiles = int(sample["length_quantiles"])
    evidence = _evidence_proxy(
        config["inputs"]["dense_candidates"],
        config["inputs"]["sparse_candidates"],
        top_k=int(sample["evidence_top_k"]),
    )
    rows = []
    for query_id, text in queries:
        length = len(text)
        position = sum(value < length for value in ordered_lengths)
        length_bin = min(quantiles - 1, position * quantiles // len(queries))
        proxy = evidence.get(query_id, {"label": "unknown", "top10_doc_overlap": 0, "top1_same": False})
        rows.append(
            {
                "query_id": query_id,
                "query_text": text,
                "query_characters": length,
                "query_tokens": len(re.findall(r"\w+", text, re.UNICODE)),
                "length_quantile": length_bin + 1,
                "category": category_for(text),
                "multi_part": text.count("?") > 1 or length >= 200,
                "evidence_proxy": proxy["label"],
                "pilot_top10_doc_overlap": proxy["top10_doc_overlap"],
                "pilot_top1_doc_same": proxy["top1_same"],
            }
        )
    stage_b = _round_robin(rows, int(sample["stage_b_queries"]), int(sample["seed"]))
    stage_a = _round_robin(stage_b, int(sample["stage_a_queries"]), int(sample["seed"]) + 1)
    stage_a_ids = {row["query_id"] for row in stage_a}
    return {
        "method": "deterministic round-robin over length-quantile × keyword-proxy category × pilot dense/sparse evidence proxy",
        "seed": int(sample["seed"]),
        "stage_a_query_ids": [row["query_id"] for row in stage_a],
        "stage_b_query_ids": [row["query_id"] for row in stage_b],
        "queries": [
            {**row, "stage_a": row["query_id"] in stage_a_ids, "stage_b": True}
            for row in stage_b
        ],
    }
