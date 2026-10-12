# Phase 12 — Full PubMed Multilingual Retrieval

## Why Phase 12

Phase 11C reached organizer-confirmed FINAL `0.0492` with Docs@20 and
Chunks@40. Incremental source scaling is paused. The new official corpus
contains a high-quality block of 240,000 English PubMed records, while the
local retrieval system currently contains none of them. Phase 12 tests this
coverage gap as a single high-upside data and architecture change.

## Scientific contract

- PubMed/NCBI is used only for public data acquisition.
- Competition retrieval and ranking remain local with pinned open models.
- Official `doc_id` membership and URL-to-PMID mapping are authoritative.
- Original title and abstract-section text is preserved separately from
  metadata-enriched document retrieval text.
- Evidence chunks may use only original title/abstract wording; missing
  abstracts are never fabricated.
- The Phase 11C candidate universe remains frozen and PubMed is additive.
- Initial output remains Docs@20 / Chunks@40.

## Planned funnel

Vietnamese query → BGE-M3 dense top-100 PubMed documents → original PubMed
evidence chunks → pinned local reranker → merge with frozen Phase 11C → global
Docs@20 / Chunks@40 output.

The first arm uses no query translation. Shared acquisition and embeddings may
later support top-300 dense retrieval, dense-plus-English-lexical hybrid, or
Chunks@80, but those arms are not authorized before the first organizer result.

## Acquisition and provenance

The durable official manifest is
`data/pubmed_phase12/official_pubmed_manifest.parquet`. Acquisition uses NCBI
E-utilities EFetch in resumable 300-PMID batches at no more than 2.5 requests
per second without an API key. Every completed batch retains compressed raw XML,
an exact PMID-order hash, a parsed Parquet part, checksums, status counts, and a
transactional completion marker.

The acquisition completed all 800 batches and accounted for all 240,000
official PMIDs. The expanded parser recovered 197 valid `PubmedBookArticle`
records from already-retained XML. A bounded targeted retry confirmed that 104
PMIDs are unavailable from EFetch; they remain explicit `MISSING` rows and are
never replaced or fabricated. The resulting acquired population is 239,770
abstract records, 126 title-only records, and 104 confirmed unavailable rows.

Transient incomplete HTTP responses, disconnects, resets, timeouts, HTTP 429,
and retryable 5xx responses are retried with bounded exponential backoff,
jitter, `Retry-After`, and the fixed 2.5 requests/second ceiling. A response is
parsed only after a complete body is received, and a batch becomes durable only
after its raw XML, parsed Parquet, checksums, and completion marker agree.

## Durable processing and retrieval pipeline

`tools/phase12_worker.py` runs the experiment-scoped state machine:

`ACQUISITION_AUDIT -> PROCESS_PUBMED -> BENCHMARK_EMBED -> EMBED_AND_INDEX -> RETRIEVE_PUBMED -> RERANK_PUBMED -> MERGE_AND_PACKAGE -> FINAL_AUDIT`

Processing is checkpointed by original acquisition batch. Retrieval-enriched
document text and original evidence text are separate. Embeddings use a
transactional float32 memmap plus completion bitmap: vector pages are flushed
before completion bits. Dense search is blockwise over the full available
PubMed population, avoiding a corpus-wide in-memory matrix. Exact reranker
reuse remains keyed only by `(query_id, chunk_id)`.

The first post-processing launch stopped before model load because `psutil`,
used only by the benchmark/memory telemetry, was absent from the project
environment and dependency manifest. All 800 processed checkpoints and the
consolidated 239,896-document/744,861-chunk artifacts remained valid. The
recovery adds `psutil` explicitly and performs a remaining-stage import, CUDA,
local tokenizer, and pinned model-revision preflight before any long worker is
allowed to start.

The first full embedding continuation later became stale at 11,520/239,896:
both supervisor and worker disappeared without a new traceback, Windows Error
Reporting record, resource-exhaustion event, or supervisor terminal event.
Accordingly, the termination cause is recorded as unknown rather than inferred
from the older `psutil` traceback. The completion bitmap proved one contiguous
11,520-row prefix of finite float32 1,024-dimensional vectors, bound to the
document-order hash and pinned model revision; row 11,520 is the exact resume
boundary. Supervisor state now records supervisor/worker PIDs and a heartbeat,
refuses a competing restart if an orphan worker is alive, and retains the last
successful checkpoint/done counter for stale-run recovery.

## Results

Processing, embedding/index, candidate, reranker, attribution, validation, and
organizer-result fields remain pending. No Phase 12 leaderboard result exists.
