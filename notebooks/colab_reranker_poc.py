"""ViBioMIR bounded Colab A100 reranker proof of concept.

In a Colab cell, upload this script and reranker_poc_200.zip, then run:
    !pip -q install transformers==4.48.3 psutil
    !python colab_reranker_poc.py --package reranker_poc_200.zip --output colab_scores.json

Download colab_scores.json and verify it on Windows. Production-equivalence
scores are always produced with batch=2. Optional larger batches are benchmark
telemetry only and never enter the returned pair scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
import zipfile
from pathlib import Path


MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
CONTRACT = {
    "model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION,
    "batch_size": 2, "max_length": 512, "precision": "float16", "eval_mode": True,
    "padding": True, "truncation": "longest_first",
    "score_extraction": "logits.reshape(-1).float().cpu().numpy()", "input_order": "query, chunk",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_package(path: Path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("input ZIP CRC failed")
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("pairs.jsonl")
    if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
        raise RuntimeError("input payload hash mismatch")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if len(rows) != 200 or len({(row["query_id"], row["chunk_id"]) for row in rows}) != 200:
        raise RuntimeError("proof-of-concept package must contain exactly 200 unique pairs")
    for row in rows:
        if hashlib.sha256(row["query_text"].encode()).hexdigest() != row["query_text_sha256"]:
            raise RuntimeError("query hash mismatch")
        if hashlib.sha256(row["chunk_text"].encode()).hexdigest() != row["chunk_text_sha256"]:
            raise RuntimeError("chunk hash mismatch")
    return manifest, rows


def score(model, tokenizer, torch, rows, batch_size: int):
    values = []
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        inputs = tokenizer(
            [(row["query_text"], row["chunk_text"]) for row in batch],
            padding=True, truncation="longest_first", max_length=512, return_tensors="pt",
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, return_dict=True).logits.reshape(-1)
        values.extend(float(value) for value in logits.float().cpu().numpy())
    torch.cuda.synchronize()
    return values, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("colab_scores.json"))
    parser.add_argument("--benchmark-larger-batches", action="store_true")
    args = parser.parse_args()
    import numpy
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this proof of concept")
    manifest, rows = load_package(args.package)
    if manifest["contract"] != CONTRACT:
        raise RuntimeError("package contract differs from notebook production contract")
    process = psutil.Process()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    # Warm-up is separate from measured production-equivalence scoring.
    warm = tokenizer([("warmup", "warmup")], padding=True, truncation="longest_first", max_length=512, return_tensors="pt")
    with torch.inference_mode():
        model(**{key: value.to("cuda") for key, value in warm.items()}, return_dict=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rss_before = process.memory_info().rss
    scores, elapsed = score(model, tokenizer, torch, rows, 2)
    production_peak_vram = torch.cuda.max_memory_allocated()
    telemetry = []
    if args.benchmark_larger_batches:
        for batch_size in (4, 8, 16):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                _, seconds = score(model, tokenizer, torch, rows[:32], batch_size)
                telemetry.append({"batch_size": batch_size, "pairs": 32, "seconds": seconds,
                                  "pairs_per_second": 32 / seconds,
                                  "peak_vram_bytes": torch.cuda.max_memory_allocated(), "scores_used": False})
            except torch.cuda.OutOfMemoryError as error:
                telemetry.append({"batch_size": batch_size, "status": "OOM", "error": str(error), "scores_used": False})
                torch.cuda.empty_cache()
    result = {
        "schema_version": 1, "input_package_sha256": file_sha256(args.package), "contract": CONTRACT,
        "pairs": [
            {"query_id": row["query_id"], "chunk_id": row["chunk_id"],
             "query_text_sha256": row["query_text_sha256"], "chunk_text_sha256": row["chunk_text_sha256"],
             "score": value}
            for row, value in zip(rows, scores, strict=True)
        ],
        "runtime": {"model_load_seconds": load_seconds, "batch2_scoring_seconds": elapsed,
                    "pairs_per_second": len(rows) / elapsed, "rss_before_bytes": rss_before,
                    "rss_after_bytes": process.memory_info().rss, "peak_vram_bytes": production_peak_vram,
                    "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                    "larger_batch_telemetry": telemetry},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
                     "python": platform.python_version(), "platform": platform.platform(),
                     "torch": torch.__version__, "transformers": transformers.__version__, "numpy": numpy.__version__,
                     "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG")},
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pairs": len(rows), "batch2_pairs_per_second": len(rows) / elapsed,
                      "peak_vram_gib": production_peak_vram / 2**30}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
