from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.processing.chunking import ChunkConfig, TokenSpan, chunk_document
from src.processing.models import CleanDocument, DocumentSection, ExtractionStatus


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def token_statistics(tokenizer: Any, texts: list[str], max_tokens: int) -> dict[str, Any]:
    lengths = [
        len(
            tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )["input_ids"]
        )
        for text in texts
    ]
    exceeding = sum(length > max_tokens for length in lengths)
    return {
        "count": len(lengths),
        "median": statistics.median(lengths) if lengths else 0,
        "p95": percentile(lengths, 0.95),
        "max": max(lengths, default=0),
        "exceeding_limit": exceeding,
        "truncation_rate": round(exceeding / max(len(lengths), 1), 6),
    }


class HuggingFaceOffsetTokenizer:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def spans(self, text: str) -> list[TokenSpan]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        return [
            TokenSpan(text[start:end], int(start), int(end))
            for start, end in encoded["offset_mapping"]
            if end > start
        ]


def _document(row: dict[str, Any]) -> CleanDocument:
    sections = [DocumentSection(**section) for section in row["sections"]]
    return CleanDocument(
        **{
            **row,
            "sections": sections,
            "extraction_status": ExtractionStatus(row["extraction_status"]),
        }
    )


def validate_and_maybe_regenerate(
    *,
    tokenizer: Any,
    source_chunks: str | Path,
    source_documents: str | Path,
    validated_chunks: str | Path,
    max_tokens: int,
    overlap_tokens: int,
    material_exceedance_rate: float,
) -> dict[str, Any]:
    source_table = pq.read_table(source_chunks)
    source_rows = source_table.to_pylist()
    before = token_statistics(
        tokenizer, [row["normalized_text"] for row in source_rows], max_tokens
    )
    should_regenerate = before["truncation_rate"] > material_exceedance_rate
    output_path = Path(validated_chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    special_tokens = len(
        tokenizer("", add_special_tokens=True, truncation=False)["input_ids"]
    )
    initial_content_budget = max_tokens - special_tokens
    content_budget = initial_content_budget
    passes = 0
    if should_regenerate:
        adapter = HuggingFaceOffsetTokenizer(tokenizer)
        documents = [
            _document(row)
            for row in pq.read_table(source_documents).to_pylist()
            if row["extraction_status"] == ExtractionStatus.SUCCESS.value
        ]
        while True:
            passes += 1
            config = ChunkConfig(
                name=f"bge_m3_{max_tokens}_overlap_{overlap_tokens}",
                target_tokens=content_budget,
                overlap_tokens=overlap_tokens,
            )
            rows = [
                chunk.as_dict()
                for document in documents
                for chunk in chunk_document(document, config, adapter)
            ]
            after = token_statistics(
                tokenizer, [row["normalized_text"] for row in rows], max_tokens
            )
            if after["exceeding_limit"] == 0:
                break
            overshoot = max(2, int(after["max"]) - max_tokens)
            content_budget -= overshoot
            if content_budget <= overlap_tokens or passes >= 10:
                raise RuntimeError("unable to produce truncation-free BGE-M3 chunks")
        pq.write_table(pa.Table.from_pylist(rows), output_path)
    else:
        pq.write_table(source_table, output_path)
        rows = source_rows
        after = before
    return {
        "model_name": "BAAI/bge-m3",
        "tokenizer_name": getattr(tokenizer, "name_or_path", "BAAI/bge-m3"),
        "intended_max_tokens": max_tokens,
        "special_tokens_per_sequence": special_tokens,
        "initial_content_token_budget": initial_content_budget,
        "final_content_token_budget": content_budget,
        "regeneration_passes": passes,
        "material_exceedance_rate": material_exceedance_rate,
        "regenerated": should_regenerate,
        "source_chunks": str(source_chunks),
        "validated_chunks": str(output_path),
        "before": before,
        "after": after,
    }
