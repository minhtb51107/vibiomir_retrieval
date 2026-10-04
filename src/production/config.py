from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


EXPECTED_DENSE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EXPECTED_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def load_production_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if int(config["version"]) != 1 or int(config["phase"]) != 8:
        raise ValueError("unsupported production pipeline config version")
    if int(config["source"]["rows"]) != 4_394_718:
        raise ValueError("production config must target the official corpus row count")
    chunking = config["chunking"]
    if (
        chunking["tokenizer_model"] != "BAAI/bge-m3"
        or chunking["tokenizer_revision"] != EXPECTED_DENSE_REVISION
        or int(chunking["content_token_budget"]) != 508
        or int(chunking["maximum_tokens_with_specials"]) != 512
        or int(chunking["overlap_tokens"]) != 64
    ):
        raise ValueError("production chunking must preserve the validated BGE-M3 508/512/64 policy")
    dense = config["dense"]
    if dense["model"] != "BAAI/bge-m3" or dense["revision"] != EXPECTED_DENSE_REVISION:
        raise ValueError("production dense model/revision must match Phase 4")
    if int(dense["dimension"]) != 1024 or dense["output_dtype"] != "float32":
        raise ValueError("production dense representation must remain float32/1024")
    reranker = config["reranking"]
    if (
        reranker["model"] != "BAAI/bge-reranker-v2-m3"
        or reranker["revision"] != EXPECTED_RERANKER_REVISION
        or int(reranker["batch_size"]) != 2
    ):
        raise ValueError("production reranker must match the Phase 7 model and safe batch")
    safety = config["safety"]
    if int(safety["maximum_bounded_crawl_records"]) <= 0:
        raise ValueError("maximum bounded crawl size must be positive")
    if float(safety["projected_growth_headroom_multiplier"]) < 1:
        raise ValueError("projected growth headroom multiplier must be at least one")
    return config
