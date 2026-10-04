#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import unicodedata
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.url_reduction.index import build_query_conditioned_index
from src.url_reduction.pipeline import (
    candidate_diagnostics,
    candidate_retention,
    evaluate_viability,
    retrieve_all_queries,
)
from src.url_reduction.url_text import accent_fold


class _OfflineSocket(socket.socket):
    def connect(self, address: object) -> None:
        raise RuntimeError(f"network access is disabled for URL reduction: {address}")

    def connect_ex(self, address: object) -> int:
        raise RuntimeError(f"network access is disabled for URL reduction: {address}")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if int(config["version"]) != 1 or config["phase"] != "8C_url_reduction_feasibility":
        raise ValueError("unsupported URL reduction configuration")
    if bool(config["network_allowed"]):
        raise ValueError("URL reduction must remain offline")
    depths = [int(value) for value in config["candidates"]["depths"]]
    if depths != [50, 100, 200, 500, 1000]:
        raise ValueError("required candidate depths are 50/100/200/500/1000")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate offline query-conditioned URL reduction.")
    parser.add_argument("--config", default="configs/url_candidate_reduction.yaml")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--reuse-candidates", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    inputs = config["inputs"]
    outputs = config["outputs"]
    artifact_dir = Path(outputs["artifacts"])
    original_socket = socket.socket
    socket.socket = _OfflineSocket
    total_started = time.perf_counter()
    try:
        stopwords = {
            unicodedata.normalize("NFC", str(value)).casefold()
            for value in config["tokenization"]["stopwords"]
        }
        stopwords.update(accent_fold(value).casefold() for value in list(stopwords))
        if args.reuse_index:
            index_summary = json.loads(
                (artifact_dir / "index_summary.json").read_text(encoding="utf-8")
            )
        else:
            index_summary = build_query_conditioned_index(
                corpus_path=inputs["corpus"],
                query_path=inputs["queries"],
                output_path=outputs["index"],
                hash_path=outputs["url_text_hashes"],
                stopwords=frozenset(stopwords),
                batch_rows=int(config["index"]["batch_rows"]),
                fallback_rows_per_domain=int(config["index"]["fallback_rows_per_domain"]),
                fallback_seed=config["candidates"]["fallback_seed"],
                sqlite_cache_mib=int(config["index"]["sqlite_cache_mib"]),
                maximum_url_characters=int(config["tokenization"]["maximum_url_characters"]),
            )
            _write_json(artifact_dir / "index_summary.json", index_summary)
        if args.reuse_candidates:
            retrieval_summary = json.loads(
                (artifact_dir / "retrieval_summary.json").read_text(encoding="utf-8")
            )
        else:
            candidate_config = config["candidates"]
            retrieval_summary = retrieve_all_queries(
                index_path=outputs["index"],
                output_path=outputs["candidates"],
                depths=[int(value) for value in candidate_config["depths"]],
                lexical_pool=int(candidate_config["lexical_pool"]),
                k1=float(config["index"]["bm25_k1"]),
                b=float(config["index"]["bm25_b"]),
                weak_best_score=float(candidate_config["weak_query_best_score"]),
                weak_minimum_candidates=int(
                    candidate_config["weak_query_minimum_lexical_candidates"]
                ),
                maximum_domain_share=float(candidate_config["maximum_domain_share"]),
                minimum_domains=int(candidate_config["minimum_domains"]),
                weak_fallback_fraction=float(
                    candidate_config["weak_query_fallback_fraction"]
                ),
                maximum_query_terms=int(candidate_config["maximum_query_terms_by_idf"]),
            )
            _write_json(artifact_dir / "retrieval_summary.json", retrieval_summary)
        depths = [int(value) for value in config["candidates"]["depths"]]
        diagnostics = candidate_diagnostics(
            index_path=outputs["index"],
            candidate_path=outputs["candidates"],
            depths=depths,
            full_signal_stats=index_summary["full_corpus"],
        )
        _write_json(artifact_dir / "diversity_diagnostics.json", diagnostics)
        retention = candidate_retention(
            candidate_path=outputs["candidates"],
            sources={
                "dense": inputs["dense_candidates"],
                "sparse": inputs["sparse_candidates"],
                "hybrid": inputs["hybrid_candidates"],
                "reranked": inputs["reranked_candidates"],
            },
            depths=depths,
        )
        _write_json(artifact_dir / "pilot_candidate_retention.json", retention)
        viability = evaluate_viability(
            index_summary=index_summary,
            retrieval_summary=retrieval_summary,
            diagnostics=diagnostics,
            retention=retention,
            gates=config["viability_gates"],
        )
        _write_json(artifact_dir / "viability_verdict.json", viability)
        current_command_seconds = round(time.perf_counter() - total_started, 6)
        prior_benchmark_path = artifact_dir / "benchmark.json"
        prior_benchmark = (
            json.loads(prior_benchmark_path.read_text(encoding="utf-8"))
            if prior_benchmark_path.exists()
            else {}
        )
        index_build_seconds = float(index_summary["runtime"]["total_seconds"])
        if args.reuse_candidates:
            candidate_and_diagnostics_seconds = prior_benchmark.get(
                "candidate_and_diagnostics_seconds"
            )
        elif args.reuse_index:
            candidate_and_diagnostics_seconds = current_command_seconds
        else:
            candidate_and_diagnostics_seconds = round(
                current_command_seconds - index_build_seconds, 6
            )
        benchmark = {
            "zero_network_guard": True,
            "network_requests_made": 0,
            "preprocessing_seconds": index_summary["runtime"]["preprocessing_scan_seconds"],
            "index_finalize_seconds": index_summary["runtime"]["index_finalize_seconds"],
            "index_build_total_seconds": index_build_seconds,
            "retrieval_seconds": retrieval_summary["retrieval_seconds"],
            "diagnostic_and_global_dedupe_seconds": diagnostics["dedupe_diagnostics_seconds"],
            "candidate_and_diagnostics_seconds": candidate_and_diagnostics_seconds,
            "measured_pipeline_seconds": round(
                index_build_seconds + float(candidate_and_diagnostics_seconds), 6
            )
            if candidate_and_diagnostics_seconds is not None
            else None,
            "current_command_seconds": current_command_seconds,
            "current_command_reused_index": bool(args.reuse_index),
            "current_command_reused_candidates": bool(args.reuse_candidates),
            "peak_process_rss_bytes": index_summary["runtime"]["peak_process_rss_bytes"],
            "index_size_bytes": index_summary["storage"]["index_size_bytes"],
            "candidate_database_size_bytes": Path(outputs["candidates"]).stat().st_size,
        }
        _write_json(artifact_dir / "benchmark.json", benchmark)
        print(json.dumps({"viability": viability, "benchmark": benchmark}, indent=2))
    finally:
        socket.socket = original_socket
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
