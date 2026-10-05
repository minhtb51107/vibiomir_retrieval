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

## Phase 8 — Full-Corpus Scaling / Production Retrieval Build

| Task | Status | Date |
|---|---|---|
| Audit local disk and model full-pipeline storage/time capacity | COMPLETE | 2026-10-04 |
| Add explicit bounded/full execution and free-space gates | COMPLETE | 2026-10-04 |
| Add atomic stage manifests and transactional partition checkpoints | COMPLETE | 2026-10-04 |
| Add streaming real-BGE-tokenizer chunk partitions | COMPLETE | 2026-10-04 |
| Run deterministic bounded 5,000-URL acquisition/extraction proof | COMPLETE | 2026-10-04 |
| Benchmark bounded dense alternatives and SQLite sparse scaling | COMPLETE | 2026-10-04 |
| Demonstrate crawl/archive/chunk resume and checksum integrity | COMPLETE | 2026-10-04 |
| Document stage plan, retention policy, and production blockers | COMPLETE | 2026-10-04 |

**Phase 8 status: COMPLETE — production architecture and bounded gates are ready; unrestricted production remains disabled and Phase 9 was not started.**

---

## Phase 8C — Query-Conditioned URL Reduction Feasibility

| Task | Status | Date |
|---|---|---|
| Process all 4,394,718 URLs using local URL/domain/path metadata only | COMPLETE | 2026-10-05 |
| Build deterministic multilingual URL BM25 and domain-balanced fallback | COMPLETE | 2026-10-05 |
| Retrieve all 1,200 queries at depths 50/100/200/500/1,000 | COMPLETE | 2026-10-05 |
| Measure global reduction, domain/script diversity, and weak-query behavior | COMPLETE | 2026-10-05 |
| Compare URL selections with dense/sparse/hybrid/reranked pilot candidates | COMPLETE | 2026-10-05 |
| Apply predeclared viability gates and record an explicit verdict | COMPLETE — NOT VIABLE | 2026-10-05 |
| Verify zero network requests and decline the proposed 10k crawl | COMPLETE | 2026-10-05 |

**Phase 8C status: COMPLETE — URL-only reduction is NOT VIABLE; no candidate crawl was started, no submission was generated, and Phase 9 was not started.**

---

## Phase 9 — Valid Competition Submission Generation

| Task | Status | Date |
|---|---|---|
| Transcribe and document the organizer JSON/ZIP contract | COMPLETE | 2026-10-05 |
| Preserve all 1,200 official query IDs and source order | COMPLETE | 2026-10-05 |
| Generate four bounded Phase 7 submission variants | COMPLETE | 2026-10-05 |
| Verify every emitted chunk against canonical source-derived text | COMPLETE | 2026-10-05 |
| Reject invalid IDs and duplicate document/chunk objects | COMPLETE | 2026-10-05 |
| Add strict offline JSON/ZIP validator and negative tests | COMPLETE | 2026-10-05 |
| Prove byte-identical JSON and ZIP generation across two runs | COMPLETE | 2026-10-05 |
| Persist compact manifests, hashes, and validation evidence | COMPLETE | 2026-10-05 |

**Phase 9 implementation status: COMPLETE — submissions are ready for manual upload; QUALITY UNKNOWN UNTIL ORGANIZER SCORE. No upload, crawl, or leaderboard calibration was started.**

---

## Phase 10A — Public Leaderboard Calibration Submissions

| Task | Status | Date |
|---|---|---|
| Reproduce the Phase 9 A baseline hash from persisted artifacts | COMPLETE | 2026-10-05 |
| Generate document-depth variants E1–E3 | COMPLETE | 2026-10-05 |
| Generate chunk-depth variants E4–E5 | COMPLETE | 2026-10-05 |
| Generate contiguous source-verbatim expansion variants E6–E7 | COMPLETE | 2026-10-05 |
| Generate Phase 6 hybrid and sparse variants E8–E9 | COMPLETE | 2026-10-05 |
| Strictly validate all nine JSON/ZIP submissions | COMPLETE | 2026-10-05 |
| Prove byte-identical output by full independent regeneration | COMPLETE | 2026-10-05 |
| Persist compact hashes, provenance, and experiment manifest | COMPLETE | 2026-10-05 |

**Phase 10A implementation status: COMPLETE — nine calibration submissions are ready for manual upload. No automatic upload, crawl, model inference, or local relevance evaluation was performed.**

---

## Phase 10B0 — Search-Based Discovery Feasibility

| Task | Status | Date |
|---|---|---|
| Predeclare provider, sampling, mapping, and viability gates | COMPLETE | 2026-10-05 |
| Build deterministic stratified 50/300-query samples | COMPLETE | 2026-10-05 |
| Implement cached sequential Bing RSS search collection | COMPLETE | 2026-10-05 |
| Implement exact/normalized/ambiguous official URL mapping | COMPLETE | 2026-10-05 |
| Complete 50-query Stage A with 100 successful requests | COMPLETE | 2026-10-05 |
| Complete 300-query Stage B with 600/600 successful cached requests | COMPLETE | 2026-10-05 |
| Measure query, variant, domain, overlap, random, and scale diagnostics | COMPLETE | 2026-10-05 |
| Apply predeclared viability gates | COMPLETE — NOT VIABLE | 2026-10-05 |
| Decline Stage C and candidate crawling | COMPLETE | 2026-10-05 |

**Phase 10B0 status: COMPLETE — public RSS search was technically reliable but NOT VIABLE for official-corpus coverage. Stage C, candidate crawling, submission generation, and Phase 10B1 were not started.**

---

## Phase 10B1 — Multilingual Site-Native Discovery

| Task | Status | Date |
|---|---|---|
| Record fatal assumptions and G0/G1/G2 gates before network work | COMPLETE | 2026-10-05 |
| Inventory all 97 domains across all 4,394,718 official rows | COMPLETE | 2026-10-05 |
| Select 19 share/language/accessibility-balanced sources | COMPLETE | 2026-10-05 |
| Audit local translation capability without downloading a model | COMPLETE | 2026-10-05 |
| Probe first-party search/sitemap/category mechanisms under hard request caps | COMPLETE | 2026-10-05 |
| Verify sampled official-URL mapping and query sensitivity | COMPLETE | 2026-10-05 |
| Apply predeclared G0 weighted-coverage gate | COMPLETE — WEAK (3.853%) | 2026-10-05 |
| Stop Stage 1/2 after failed G0; perform no candidate crawl | COMPLETE | 2026-10-05 |
| Publish reusable 97-source discovery/acquisition map | COMPLETE | 2026-10-05 |
| Add offline inventory, adapter, mapping, cache, routing, gate, and report tests | COMPLETE | 2026-10-05 |

**Phase 10B1 status: COMPLETE AT PREDECLARED STOP — site-native search is NOT VIABLE as a major discovery channel. Stage 1, Stage 2, candidate crawling, embeddings, reranking, and submissions were not run.**

---

## Phase 10B2 — Source Acquisition Benchmark

| Task | Status | Date |
|---|---|---|
| Freeze per-source sampling, early-stop, usable-text, and tier rules | COMPLETE | 2026-10-05 |
| Build deterministic 1,000-row manifests for five primary sources and one Vietnamese control | COMPLETE | 2026-10-05 |
| Reuse hash-verified Phase 8 results/bodies without duplicate requests | COMPLETE | 2026-10-05 |
| Evaluate every source at the 200-row checkpoint | COMPLETE | 2026-10-05 |
| Stop 120ask.com and ask.39.net after failing the usable-content gate | COMPLETE | 2026-10-05 |
| Continue four passing sources to at most 1,000 attempted rows | COMPLETE | 2026-10-05 |
| Run generic extraction and real BGE-M3 508/64 chunking | COMPLETE | 2026-10-05 |
| Measure per-source acquisition, content, storage, chunk, and time metrics | COMPLETE | 2026-10-05 |
| Produce 10k/50k/full-source engineering projections | COMPLETE | 2026-10-05 |
| Verify the full offline test suite and preserve large outputs under ignored data paths | COMPLETE | 2026-10-05 |

**Phase 10B2 status: COMPLETE — 4,400 deterministic rows benchmarked, including 2,121 new URL fetches. A bounded 20k Vietnamese-source milestone is recommended but was not started. No embeddings, index, submission, or leaderboard work was performed.**

---

## Phase 10B3 — Source Relevance Probe

| Task | Status | Date |
|---|---|---|
| Freeze source-only comparison and fixed submission policy | COMPLETE | 2026-10-06 |
| Reuse pilot embeddings and embed only 19,238 new source chunks | COMPLETE | 2026-10-06 |
| Build six independent 120,000-row candidate pools | COMPLETE | 2026-10-06 |
| Reuse Phase 7 scores and deduplicate novel query/chunk inference | COMPLETE | 2026-10-06 |
| Pass sustained 2,048-pair scorer gate and finish 18 isolated shards | COMPLETE | 2026-10-06 |
| Verify 90,349/90,349 global scores and six complete SQLite checkpoints | COMPLETE | 2026-10-06 |
| Finalize source-impact diagnostics for all 1,200 queries | COMPLETE | 2026-10-06 |
| Generate and strictly validate B0 plus all locally impactful variants | COMPLETE | 2026-10-06 |
| Prove source-verbatim provenance and byte-identical regeneration | COMPLETE | 2026-10-06 |
| Preserve large artifacts under ignored data/submission paths | COMPLETE | 2026-10-06 |

**Phase 10B3 status: COMPLETE — all six bounded source samples materially change local outputs and are ready for manual leaderboard probing. This is not relevance evaluation; no additional crawl, source scaling, or automatic upload was performed.**

---

## Phase 10C0 — S4 Targeting Tournament and Experiment Journal

| Task | Status | Date |
|---|---|---|
| Backfill evidence-grounded timeline and experiment journal | COMPLETE | 2026-10-06 |
| Record all supported organizer results in durable CSV history | COMPLETE | 2026-10-06 |
| Build S4 top10/top5/top3 and multi-query proxy sets | COMPLETE | 2026-10-06 |
| Evaluate hostname-free URL lexical ranking at common budgets | COMPLETE — WEAK | 2026-10-06 |
| Audit S4 path/category structure | COMPLETE — NOT VIABLE | 2026-10-06 |
| Evaluate seed expansion with five-fold held-out proxies | COMPLETE — WEAK | 2026-10-06 |
| Run bounded 50-query site-restricted external-search gate | COMPLETE — NOT VIABLE | 2026-10-06 |
| Reuse native-search preflight and stop the failed branch | COMPLETE — NOT VIABLE | 2026-10-06 |
| Apply the predeclared ensemble gate | COMPLETE — NOT RUN; no qualifying methods | 2026-10-06 |
| Recommend future broad deterministic S4 sampling without starting it | COMPLETE | 2026-10-06 |

**Phase 10C0 status: COMPLETE — cheap metadata/search targeting is too weak for an S4-5K acquisition cut. No crawl, adaptive acquisition, GPU work, model change, submission, or 300-query search stage was started.**
