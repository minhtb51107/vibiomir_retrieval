# TASKS.md

## Phase 1 — Dataset Understanding

| Task | Status | Date |
|---|---|---|
| Inspect query.parquet (schema, counts, stats, Unicode) | COMPLETE | 2026-10-03 |
| Inspect links_corpus.parquet (schema, IDs, URLs, domains, extensions) | COMPLETE | 2026-10-03 |
| Analyze corpus crawling risks | COMPLETE | 2026-10-03 |
| Check ID integrity | COMPLETE | 2026-10-03 |
| Create scripts/inspect_dataset.py | COMPLETE | 2026-10-03 |
| Run script against real files | COMPLETE | 2026-10-03 |
| Create docs/DATASET_REPORT.md | COMPLETE | 2026-10-03 |

**Phase 1 status: COMPLETE**

---

## Phase 2A — Corpus Acquisition Core

| Task | Status | Date |
|---|---|---|
| Implement streaming corpus loader and fragment-safe URL handling | COMPLETE | 2026-10-03 |
| Implement pooled async HTTP fetcher with retry/backoff | COMPLETE | 2026-10-03 |
| Implement global/per-domain concurrency and delays | COMPLETE | 2026-10-03 |
| Implement cached robots policy | COMPLETE | 2026-10-03 |
| Implement transactional SQLite checkpoint/resume | COMPLETE | 2026-10-03 |
| Add safe pilot CLI and explicit `--full` gate | COMPLETE | 2026-10-03 |
| Add offline tests for URL, retry, resume, status, robots, and CLI safety | COMPLETE | 2026-10-03 |
| Run mixed-domain pilot (10 URLs) | COMPLETE | 2026-10-03 |
| Demonstrate second-run checkpoint skip (10/10 skipped) | COMPLETE | 2026-10-03 |
| Document crawler architecture and deferred work | COMPLETE | 2026-10-03 |

**Phase 2A status: COMPLETE**

---

## Phase 2B — Source Probing and Extraction Strategy

| Task | Status | Date |
|---|---|---|
| Build reproducible balanced source sampler | COMPLETE | 2026-10-03 |
| Probe 118 unique URLs across major and special categories | COMPLETE | 2026-10-03 |
| Measure response, encoding, language, DOM, and anti-bot signals | COMPLETE | 2026-10-03 |
| Compare semantic-container and density-scored extraction | COMPLETE | 2026-10-03 |
| Persist metadata-only JSON/CSV probe artifacts | COMPLETE | 2026-10-03 |
| Add fixture-based offline extraction and encoding tests | COMPLETE | 2026-10-03 |
| Document evidence and extraction policy | COMPLETE | 2026-10-03 |

**Phase 2B status: COMPLETE**

---

## Phase 2C — Production Crawl Readiness and Corpus Acquisition

| Task | Status | Date |
|---|---|---|
| Add bounded production batches and two-step full-crawl gate | COMPLETE | 2026-10-03 |
| Add bounded-memory SQLite telemetry export | COMPLETE | 2026-10-03 |
| Run balanced 1,225-URL readiness benchmark | COMPLETE | 2026-10-03 |
| Demonstrate interruption integrity and process-restart resume | COMPLETE | 2026-10-03 |
| Demonstrate second-run 1,225/1,225 checkpoint skip | COMPLETE | 2026-10-03 |
| Measure throughput, response sizes, retries, redirects, and SQLite growth | COMPLETE | 2026-10-03 |
| Produce full-corpus time, network, and disk estimates | COMPLETE | 2026-10-03 |
| Document scalable body-storage decision | COMPLETE | 2026-10-03 |

**Phase 2C status: COMPLETE — unrestricted full crawl was not started.**

---

## Phase 3 — Cleaning, Extraction, and Chunking

| Task | Status | Date |
|---|---|---|
| Implement crash-safe compressed body shards and indexed reader | COMPLETE | 2026-10-03 |
| Integrate bounded body capture without SQLite BLOB storage | COMPLETE | 2026-10-03 |
| Implement evidence-based decoding with NFC normalization | COMPLETE | 2026-10-03 |
| Implement semantic/density hybrid extraction | COMPLETE | 2026-10-03 |
| Preserve structured Q&A sections and explicit unusable statuses | COMPLETE | 2026-10-03 |
| Implement deterministic paragraph/heading/Q&A-aware chunking | COMPLETE | 2026-10-03 |
| Verify exact offsets, provenance, checksums, and resume | COMPLETE | 2026-10-03 |
| Process bounded 1,225-row / 1,148-body pilot | COMPLETE | 2026-10-03 |
| Compare 256/32, 512/64, and 768/96 chunk configurations | COMPLETE | 2026-10-03 |
| Document pipeline, measured results, and Phase 4 recommendation | COMPLETE | 2026-10-03 |

**Phase 3 status: COMPLETE — Phase 4 retrieval work was not started.**

---

## Phase 4 — Dense Retrieval Baseline

| Task | Status | Date |
|---|---|---|
| Validate Phase 3 chunks with the actual BGE-M3 tokenizer | COMPLETE | 2026-10-03 |
| Regenerate bounded chunks to eliminate 512-token overflow | COMPLETE | 2026-10-03 |
| Benchmark safe CUDA batch sizes on longest chunks | COMPLETE | 2026-10-03 |
| Build resumable normalized BGE-M3 embeddings | COMPLETE | 2026-10-03 |
| Build exact FAISS `IndexFlatIP` and SQLite metadata mapping | COMPLETE | 2026-10-03 |
| Verify all FAISS row/chunk/document mappings | COMPLETE | 2026-10-03 |
| Retrieve all 1,200 official queries against the pilot index | COMPLETE | 2026-10-03 |
| Compare best-chunk and top-three-mean document aggregation | COMPLETE | 2026-10-03 |
| Run multilingual/manual sanity checks (not evaluation) | COMPLETE | 2026-10-03 |
| Record index/retrieval benchmarks and baseline recommendation | COMPLETE | 2026-10-03 |

**Phase 4 status: COMPLETE — Phase 5 evaluator work was not started.**

---

## Phase 5 — Local Evaluator

**Status: DEFERRED — pending leaderboard/submission evidence.**

No trustworthy local ground truth exists. No local official evaluator or
fabricated relevance metric was introduced.

---

## Phase 6 — Hybrid Retrieval Baseline

| Task | Status | Date |
|---|---|---|
| Profile multilingual chunk text and choose deterministic lexical tokenization | COMPLETE | 2026-10-03 |
| Build atomic SQLite BM25 index over all 5,643 validated chunks | COMPLETE | 2026-10-03 |
| Verify sparse row/chunk/document mapping against Phase 4 metadata | COMPLETE | 2026-10-03 |
| Retrieve sparse top-100 chunks for all 1,200 queries | COMPLETE | 2026-10-03 |
| Reuse existing Phase 4 dense top-50 results unchanged | COMPLETE | 2026-10-03 |
| Implement deterministic chunk- and document-level RRF | COMPLETE | 2026-10-03 |
| Apply best-chunk and top-three-mean document aggregation | COMPLETE | 2026-10-03 |
| Measure latency, overlap, duplicates, and document concentration | COMPLETE | 2026-10-03 |
| Run 40-query multilingual side-by-side sanity check | COMPLETE | 2026-10-03 |
| Add offline tokenizer, BM25, mapping, fusion, and determinism tests | COMPLETE | 2026-10-03 |

**Phase 6 status: COMPLETE — diagnostics are descriptive only; Phase 7 was not started.**

---

## Phase 7 — Reranking and Candidate Selection

| Task | Status | Date |
|---|---|---|
| Build source-aware unified dense/sparse/hybrid candidate pools | COMPLETE | 2026-10-04 |
| Pin and benchmark the multilingual BGE reranker on CUDA | COMPLETE | 2026-10-04 |
| Select safe batch size 2 after recording batch-4 OOM | COMPLETE | 2026-10-04 |
| Score all 120,000 pairs with transactional checkpoint/resume | COMPLETE | 2026-10-04 |
| Verify checkpoint signature, integrity, and exact pool-key compatibility | COMPLETE | 2026-10-04 |
| Materialize reranked chunk outputs at depths 20, 50, and 100 | COMPLETE | 2026-10-04 |
| Produce best-chunk and top-three-mean document rankings | COMPLETE | 2026-10-04 |
| Measure exact-text, per-document, and boilerplate controls | COMPLETE | 2026-10-04 |
| Run 40-query multilingual/source-composition sanity checks | COMPLETE | 2026-10-04 |
| Add offline candidate, checkpoint, selection, aggregation, and provenance tests | COMPLETE | 2026-10-04 |

**Phase 7 status: COMPLETE — diagnostics are descriptive only; Phase 8 was not started.**

---

## Phase 8 and beyond

_Not started._
