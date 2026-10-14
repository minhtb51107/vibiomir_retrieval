"""Bounded BGE-M3 embedding equivalence runner for Colab L4/A100.

Colab commands:
  !pip -q install sentence-transformers==5.7.0 transformers==4.48.3 psutil
  !python colab_embedding_poc.py --package phase14_bge_m3_embedding_poc_200.zip \
      --output colab_embedding_results.zip

Every historical group of four is encoded in a separate ``model.encode`` call.
Groups are never combined, because that would change SentenceTransformers'
length-sorted batch membership and invalidate equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
    "input_order": "historical_batch_groups_of_4",
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
    reconstruction = manifest.get("historical_batch_reconstruction") or {}
    if (
        reconstruction.get("sentence_transformers_version") != "5.7.0"
        or reconstruction.get("transformers_version") != "4.48.3"
        or reconstruction.get("window_rows") != 128
        or reconstruction.get("batch_size") != 4
        or reconstruction.get("flatten_interleave_enabled") is not False
    ):
        raise RuntimeError("historical batch reconstruction contract mismatch")
    if sha256_bytes(payload) != manifest.get("chunks_payload_sha256"):
        raise RuntimeError("input chunk payload hash mismatch")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if len(rows) != manifest.get("row_count"):
        raise RuntimeError("input row count mismatch")
    for index, row in enumerate(rows):
        text = normalize_text(str(row["normalized_text"]))
        if int(row["sample_index"]) != index or sha256_bytes(text.encode("utf-8")) != row["text_sha256"]:
            raise RuntimeError(f"input row identity/hash mismatch at {index}")
    group_rows(rows)
    return manifest, rows


def group_rows(rows):
    if len(rows) % 4:
        raise RuntimeError("input does not contain complete historical groups")
    groups = []
    seen = set()
    for start in range(0, len(rows), 4):
        group = rows[start:start + 4]
        batch_ids = {row.get("historical_batch_id") for row in group}
        offsets = [int(row.get("historical_batch_offset", -1)) for row in group]
        if len(batch_ids) != 1 or None in batch_ids or offsets != [0, 1, 2, 3]:
            raise RuntimeError(f"historical group membership/order mismatch at row {start}")
        batch_id = next(iter(batch_ids))
        if batch_id in seen:
            raise RuntimeError(f"historical group repeated: {batch_id}")
        seen.add(batch_id)
        groups.append(group)
    return groups


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


def encode_historical_groups(model, rows, *, encode_fn=encode):
    import numpy as np
    group_vectors = []
    group_telemetry = []
    started = time.perf_counter()
    for group in group_rows(rows):
        texts = [normalize_text(str(row["normalized_text"])) for row in group]
        vectors_here, seconds = encode_fn(model, texts, 4)
        vectors_here = np.asarray(vectors_here, dtype=np.float32)
        if vectors_here.shape != (4, 1024):
            raise RuntimeError(f"invalid historical-group vector shape: {vectors_here.shape}")
        group_vectors.append(vectors_here)
        group_telemetry.append({
            "historical_batch_id": group[0]["historical_batch_id"],
            "rows": 4,
            "seconds": seconds,
            "member_chunk_ids": [row["chunk_id"] for row in group],
        })
    return np.concatenate(group_vectors, axis=0), time.perf_counter() - started, group_telemetry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("colab_embedding_results.zip"))
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
    groups = group_rows(rows)
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
    vectors, elapsed, group_telemetry = encode_historical_groups(model, rows)
    if vectors.shape != (len(rows), 1024) or not numpy.isfinite(vectors).all():
        raise RuntimeError(f"invalid candidate vectors: {vectors.shape}")
    peak_vram = torch.cuda.max_memory_allocated()
    try:
        import resource
        peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except ImportError:
        peak_rss_bytes = int(process.memory_info().peak_wset)
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
            "peak_rss_bytes": peak_rss_bytes,
            "historical_group_calls": group_telemetry,
            "historical_group_count": len(groups),
            "combined_encode_call_used": False,
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
