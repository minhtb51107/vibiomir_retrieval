"""Bounded BGE-M3 embedding equivalence runner for Colab L4/A100.

Colab commands:
  !pip -q install sentence-transformers==5.7.0 transformers==4.48.3 psutil
  !python colab_embedding_poc.py --package phase14_bge_m3_embedding_poc_200.zip \
      --output colab_embedding_results.zip --benchmark-larger-batches

Only the batch=4 vectors are exported for equivalence. Larger batches are
performance telemetry and must never be imported as production vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
import unicodedata
import zipfile
from pathlib import Path


MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
CONTRACT = {
    "model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION,
    "dimension": 1024, "batch_size": 4, "max_length": 512,
    "precision": "float16_cuda", "normalization": "sentence_transformers_L2",
    "output_dtype": "float32", "text_normalization": "NFC_then_split_join",
    "input_order": "package_row_order",
}


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_package(path: Path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("input package CRC failed")
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("chunks.jsonl")
    if manifest.get("contract") != CONTRACT:
        raise RuntimeError("input package model contract mismatch")
    if sha256_bytes(payload) != manifest.get("chunks_payload_sha256"):
        raise RuntimeError("input chunk payload hash mismatch")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if len(rows) != manifest.get("row_count"):
        raise RuntimeError("input row count mismatch")
    for index, row in enumerate(rows):
        text = normalize_text(str(row["normalized_text"]))
        if int(row["sample_index"]) != index or sha256_bytes(text.encode("utf-8")) != row["text_sha256"]:
            raise RuntimeError(f"input row identity/hash mismatch at {index}")
        if int(row["reference_batch_offset"]) != index % 4:
            raise RuntimeError("input rows do not preserve the historical batch=4 context")
    return manifest, rows


def encode(model, texts, batch_size: int):
    import numpy as np
    import torch
    started = time.perf_counter()
    vectors = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    torch.cuda.synchronize()
    return np.asarray(vectors, dtype=np.float32), time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("colab_embedding_results.zip"))
    parser.add_argument("--benchmark-larger-batches", action="store_true")
    args = parser.parse_args()
    import numpy
    import psutil
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; select a Colab L4 or A100 runtime")
    manifest, rows = load_package(args.package)
    texts = [normalize_text(str(row["normalized_text"])) for row in rows]
    process = psutil.Process()
    rss_before = process.memory_info().rss
    load_started = time.perf_counter()
    model = SentenceTransformer(
        MODEL, revision=REVISION, device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = 512
    load_seconds = time.perf_counter() - load_started
    # Warm-up is excluded and never included in equivalence output.
    model.encode(["warmup"], batch_size=1, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    vectors, elapsed = encode(model, texts, 4)
    if vectors.shape != (len(rows), 1024) or not numpy.isfinite(vectors).all():
        raise RuntimeError(f"invalid candidate vectors: {vectors.shape}")
    peak_vram = torch.cuda.max_memory_allocated()
    telemetry = []
    if args.benchmark_larger_batches:
        for batch_size in (8, 16, 32):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                _, seconds = encode(model, texts, batch_size)
                telemetry.append({
                    "batch_size": batch_size, "rows": len(rows), "seconds": seconds,
                    "rows_per_second": len(rows) / seconds,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                    "vectors_used_for_equivalence": False,
                })
            except torch.cuda.OutOfMemoryError as error:
                telemetry.append({"batch_size": batch_size, "status": "OOM", "error": str(error), "vectors_used_for_equivalence": False})
                torch.cuda.empty_cache()
    vector_bytes = vectors.tobytes(order="C")
    result = {
        "schema_version": 1,
        "input_package_sha256": file_sha256(args.package),
        "contract": CONTRACT,
        "rows": [
            {"sample_index": row["sample_index"], "chunk_id": row["chunk_id"], "text_sha256": row["text_sha256"]}
            for row in rows
        ],
        "candidate_vectors_sha256": sha256_bytes(vector_bytes),
        "runtime": {
            "model_load_seconds": load_seconds, "batch4_embedding_seconds": elapsed,
            "rows_per_second": len(rows) / elapsed, "rss_before_bytes": rss_before,
            "rss_after_bytes": process.memory_info().rss, "peak_vram_bytes": peak_vram,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "larger_batch_telemetry": telemetry,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__, "numpy": numpy.__version__,
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        },
    }
    result_bytes = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("result_manifest.json", result_bytes)
        archive.writestr("candidate_vectors.f32", vector_bytes)
    print(json.dumps({
        "output": str(args.output), "rows": len(rows), "batch4_rows_per_second": len(rows) / elapsed,
        "peak_vram_gib": peak_vram / 2**30, "production_import_allowed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
