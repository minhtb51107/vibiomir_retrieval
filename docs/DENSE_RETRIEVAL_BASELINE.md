# Phase 4 Dense Retrieval Baseline

## Scope

Phase 4 implements a correctness-first dense baseline from the Phase 3 pilot
chunks and all 1,200 official queries. It does not implement BM25, sparse
retrieval, query expansion, reranking, an official evaluator, or submission
generation. Large model, embedding, FAISS, metadata, and retrieval outputs are
kept under ignored directories.

## Model and runtime

The baseline uses exactly `BAAI/bge-m3`, resolved from the local Hugging Face
cache at revision `5617a9f61b028005a4858fdac845db406aefb181`. No fallback
model or hosted API is used.

| Property | Value |
|---|---|
| Model/tokenizer | `BAAI/bge-m3` |
| Embedding dimension | 1,024 |
| Native model sequence limit | 8,192 tokens |
| Baseline operational limit | 512 tokens |
| Pooling | CLS token, from the model's SentenceTransformers pooling config |
| Model inference precision | FP16 on CUDA |
| Persisted embedding precision | normalized float32 |
| Similarity | cosine-equivalent inner product after L2 normalization |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB |
| PyTorch/CUDA | 2.5.1+cu121 / CUDA 12.1 |

The config pins Transformers below 4.49 because newer installed Transformers
rejected the model's cached PyTorch `.bin` weights under the fixed PyTorch
2.5.1 runtime. This changes neither model weights nor model identity.

Random seeds are set for Python, NumPy, and PyTorch. Exact bitwise equality
across different devices or library/CUDA versions is not promised, but row
order, text normalization, index construction, and ranking tie-breaks are
deterministic for a fixed runtime.

## Actual tokenizer validation

Phase 3 used an offline proxy tokenizer. The BGE-M3 tokenizer found a material
mismatch in the recommended 512/64 chunk set:

| Metric | Phase 3 chunks under BGE-M3 tokenizer | Validated Phase 4 chunks |
|---|---:|---:|
| Chunks | 5,668 | 5,643 |
| Median tokens including special tokens | 282.5 | 292 |
| p95 | 569 | 510 |
| Maximum | 1,358 | 512 |
| Above 512 | 1,152 (20.32%) | 0 |

Because the exceedance rate was above the configured 1% materiality gate,
the bounded set was regenerated once from the Phase 3 cleaned documents using
the real fast-tokenizer offsets. Boundary retokenization required two
deterministic passes. The final content budget is 508 tokens plus two special
tokens, with the original 64-token overlap and all Phase 3 section/Q&A rules.
The validated Parquet SHA-256 is
`9f875e0229b9c3f911ad502162efd84c574462d05b57648245d015177e921c3e`.

All 1,200 queries were also measured: median 25 tokens, p95 103.05, maximum
350, and zero above the operational 512-token limit.

## Embedding and cache design

`src/indexing/embedder.py` provides a mockable embedding interface and the
local SentenceTransformers implementation. Both query and chunk input apply
the same NFC and whitespace normalization. Retrieved `chunk_text` remains the
unaltered, source-derived `raw_text` stored in Phase 3 metadata.

Corpus embeddings are written by row to a float32 memory-mapped file. An
atomically replaced JSON checkpoint records the source checksum, row count,
dimension, model name, and completed row count. A restart validates those
fields and resumes after the last completed batch. The chunk source is read in
bounded Parquet batches.

The interrupted CPU run had completed 640 rows in FP32. Because production GPU
inference uses FP16 model weights, mixing that prefix with GPU embeddings could
introduce small cross-precision inconsistencies. The CPU checkpoint was
preserved under `data/embeddings/phase4_dense_cpu_backup/`; only the embedding
artifact was restarted. The validated chunks and tokenizer statistics were not
regenerated.

## GPU benchmark and index build

The batch benchmark used the 64 longest validated chunks (509–510 content
tokens), after one warm-up forward pass:

| Batch | Result | Seconds | Chunks/s | Peak reserved VRAM | Headroom |
|---:|---|---:|---:|---:|---:|
| 1 | Success | 2.6947 | 23.75 | 2,225,078,272 B | 2,069,364,736 B |
| 2 | Success | 2.0098 | 31.84 | 2,218,786,816 B | 2,075,656,192 B |
| 4 | Success | 1.7951 | 35.65 | 2,218,786,816 B | 2,075,656,192 B |

Batch 4 was the largest safely tested size. It used 51.67% of reported VRAM,
so no higher or repeated OOM-search batches were attempted.

The complete build encoded 5,643 chunks in 115.144 seconds at 49.01 chunks/s.
Peak allocated GPU memory was 1,196,663,808 bytes and peak reserved memory was
2,208,301,056 bytes. The normalized float32 embedding file is 23,113,728 bytes.

## FAISS and metadata mapping

The exact baseline is `faiss.IndexFlatIP`. No IVF, HNSW, or product
quantization is used. The persisted FAISS index is 23,113,773 bytes.

A separate SQLite metadata store maps `faiss_row INTEGER PRIMARY KEY` to the
chunk ID, document ID, source-derived text, URL, offsets, section/heading
provenance, and extraction method. Construction follows Parquet row order.
Verification compared all 5,643 source rows with the SQLite mapping:

- 5,643 unique chunk IDs;
- 1,097 distinct documents;
- zero row/chunk/document mismatches;
- SQLite `integrity_check = ok`;
- embedding norms ranged from 0.99999988 to 1.00000012.

## Retrieval and document aggregation

For every query, the pipeline returns the top 50 individual chunks and two
deduplicated top-10 document rankings:

1. `best_chunk`: the maximum retrieved chunk score for each document.
2. `top_n_mean`: the mean of up to the top three retrieved chunk scores for
   each document.

Both use deterministic score-descending/document-ID tie ordering. Different
`doc_id` values are never merged. Output chunk rows retain query ID/text,
rank, score, chunk ID, document ID, exact source-derived chunk text, source
URL, FAISS row, offsets, section type, heading path, and extraction method.

Across all 1,200 queries, the two aggregations agreed at rank 1 for 895 queries
(74.58%). Their mean top-10 Jaccard overlap was 0.8736. `best_chunk` is the
recommended Phase 5 baseline because it is the simplest unlearned rule and
does not penalize documents that contribute fewer chunks to the retrieved
pool. The top-three mean result remains available for evaluation.

## Retrieval performance

Query embeddings were generated on GPU before loading CPU FAISS. This avoids a
Windows CUDA/FAISS initialization conflict observed when FAISS was loaded
first and cleanly separates timing components.

| Measurement | Result |
|---|---:|
| Queries | 1,200 |
| Chunk ranking rows | 60,000 |
| Mean end-to-end latency/query | 16.207 ms |
| p50 latency/query | 11.064 ms |
| p95 latency/query | 16.309 ms |
| Mean embedding latency/query | 12.905 ms |
| Mean exact-search + mapping latency/query | 3.302 ms |

Timings are batch-amortized at query batch size 4 on this machine and exclude
one-time model loading.

## Multilingual sanity check — not evaluation

Thirty queries were selected evenly across query character-length order. This
is explicitly a **SANITY CHECK**, not an official relevance evaluation.
Among their 300 top-10 chunks, lightweight content signals counted 161
Vietnamese, 114 Han-script, and 25 unknown. Twenty-eight queries returned at
least one Vietnamese chunk and 24 returned at least one Chinese chunk. This
confirms raw cross-lingual retrieval occurs without translation or query
expansion; it does not prove relevance quality.

Manual inspection found mixed results. Queries 1177, 506, 816, and 964 showed
useful or partially useful Chinese/Vietnamese semantic matches. Queries 3, 33,
360, 833, 842, and 894 exposed clear mismatches or overly generic heading
hits. Short queries were vulnerable to title-only Chinese chunks such as
“treatment” or “examination”; long queries were also mixed, so no monotonic
query-length effect was established. Specific titles helped, while generic
headings sometimes ranked too highly.

Chunk-level document concentration is material: the median sanity query had
24 unique documents among its top 50 chunks, but the range was 4–43; one
document contributed as many as 44 chunks. Twenty-four of 30 queries had at
least 10 chunks from one document. Document aggregation is therefore necessary
for downstream evaluation.

## Reproduction

```powershell
# Only needed when tokenizer validation has not already been completed
.venv\Scripts\python.exe scripts\validate_bge_tokenizer.py

.venv\Scripts\python.exe scripts\benchmark_bge_gpu.py
.venv\Scripts\python.exe scripts\build_dense_index.py
.venv\Scripts\python.exe scripts\retrieve_dense.py
```

Operational values are in `configs/dense_retrieval.yaml`. Small benchmark and
sanity JSON files are under `artifacts/phase4_dense/`. Generated files remain
under `data/chunks/`, `data/embeddings/`, `data/indexes/`, and
`data/retrieval/` and are not committed.

## Recommended Phase 5 starting settings

- exact `BAAI/bge-m3` revision recorded above;
- real-tokenizer 508-content-token / 64-overlap chunks, maximum 512 including
  special tokens;
- normalized 1,024-dimensional float32 embeddings;
- CUDA batch size 4 on this 4 GB GPU;
- exact `IndexFlatIP`;
- top 50 chunks, top 10 documents;
- `best_chunk` document aggregation.

These are reproducible baseline settings, not a quality winner. Phase 5 must
implement the official evaluator before any quality claim or tuning decision.

## Limitations and deferred work

- The index covers only 1,097 successfully processed pilot documents, so many
  official queries have no close in-pilot answer.
- The sanity check is subjective and not a relevance metric.
- Model load time and full-corpus scaling are not included in query latency.
- No BM25, sparse retrieval, fusion, reranking, query expansion, official
  evaluation, or submission was implemented.
