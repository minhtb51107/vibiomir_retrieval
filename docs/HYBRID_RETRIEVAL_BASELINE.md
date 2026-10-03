# Phase 6 Hybrid Dense and Sparse Retrieval Baseline

## Scope

Phase 6 adds deterministic BM25 retrieval and Reciprocal Rank Fusion (RRF) to
the completed Phase 4 dense baseline. It indexes the exact 5,643 validated
chunks from 1,097 pilot documents and retrieves all 1,200 official queries.
Phase 4 chunks, BGE-M3 embeddings, FAISS index, metadata, and ranking files are
reused unchanged.

There is no trustworthy local relevance ground truth. All comparisons below
are descriptive diagnostics or manually inspected sanity checks, not an
official evaluation. No F2 score or claim that one method is better is made.
Phase 5 remains deferred pending leaderboard or submission evidence, and no
Phase 7 reranker was implemented.

## Architecture

The pipeline has three independent layers:

1. `SparseRetriever` reads an immutable SQLite BM25 index and uses the Phase 4
   metadata store for source-derived chunk text and provenance.
2. The existing Phase 4 dense rankings provide BGE-M3/`IndexFlatIP` candidates
   without re-embedding or rebuilding the dense index.
3. `HybridRetriever` combines chunk rankings with deterministic RRF. Existing
   `best_chunk` and `top_n_mean` aggregation operate on dense, sparse, and
   fused chunk results. RRF can also combine rankings after document
   aggregation.

Standard chunk results retain `query_id`, normalized `query_text`, `rank`,
`score`, `chunk_id`, `doc_id`, exact source-derived `chunk_text`, `source_url`,
and available provenance. Different document IDs are never merged.

## Sparse tokenization

The validated chunks contain mixed scripts: 2,599 chunks have a strong Han
signal, 2,343 have a Vietnamese signal, and 975 contain both Han and Latin
characters. The environment contained no `jieba`, `underthesea`, `pyvi`, or
`rank_bm25` package. Adding a heavy language-specific framework was not
justified for this baseline.

The tokenizer `unicode_words_han_unigrams_bigrams_v1` is offline,
deterministic, and implemented with the Python standard library:

- text is NFC-normalized and case-folded for matching;
- contiguous Unicode letters/numbers form word tokens for Vietnamese,
  English, and other alphabetic text;
- every contiguous Han run emits character unigrams and adjacent character
  bigrams;
- token-family prefixes prevent Han tokens from colliding with word tokens;
- punctuation-only and empty queries safely produce no results.

Han unigrams tolerate short terms and unseen word boundaries; bigrams add
local order and specificity. The tradeoff is a larger index and weaker word
semantics than a dictionary segmenter. Token normalization affects matching
only: returned chunk text is never rewritten.

## BM25 and sparse index

The implementation uses the conventional positive-IDF BM25 form:

```text
idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

score(q, d) = sum_t idf(t) *
  tf(t,d) * (k1 + 1) /
  (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

Configured defaults are `k1=1.2` and `b=0.75`; they were not relevance-tuned.
Repeated query terms contribute repeatedly. Equal scores use source row order
as the deterministic tie-break.

The sparse index is an atomic SQLite inverted index under ignored `data/`.
It stores document row/chunk/document mappings, terms and document frequency,
postings and term frequency, and a manifest. The manifest records tokenizer,
BM25 settings, validated-chunk SHA-256, chunk-order SHA-256, and corpus stats.
The final index passed SQLite integrity checking and all 5,643 sparse rows
matched the Phase 4 row/chunk/document mapping with zero mismatches.

| Sparse index measurement | Result |
|---|---:|
| Chunks | 5,643 |
| Vocabulary | 108,690 terms |
| Indexed token occurrences | 1,974,098 |
| Mean document length | 349.83 tokens |
| Median / p95 / max length | 322 / 969 / 1,355 |
| Build time | 33.056 seconds |
| SQLite size | 52,080,640 bytes (49.67 MiB) |

The validated chunk input SHA-256 is
`9f875e0229b9c3f911ad502162efd84c574462d05b57648245d015177e921c3e`.
The chunk-order SHA-256 is
`91dbfe9c43d7bbc976acba86f6b87913ec4bb079b1f4d80ea4b47e780bfbdb78`.

## Candidate pools, fusion, and aggregation

The reproducible baseline uses:

- dense: existing top 50 Phase 4 chunks;
- sparse: top 100 BM25 chunks;
- RRF: `1 / (60 + rank)` per source, retaining 50 chunks;
- documents: top 10;
- default document aggregation: `best_chunk`;
- comparison aggregation: mean of the top three chunk scores.

RRF is the default because it depends on ranks instead of incomparable cosine,
BM25, and fused score scales. Optional normalized-score fusion was deliberately
not added: min-max normalization would be sensitive to each candidate pool's
score range without ground truth to justify it.

Duplicate chunk IDs are removed independently within each source ranking
before fusion contribution. RRF uses fused score, best contributing rank, and
chunk ID for deterministic ordering. Document-level RRF is supported after
both `best_chunk` and `top_n_mean` aggregation.

## Performance

BM25 returned candidates for every one of the 1,200 queries and produced
120,000 chunk rows at top 100.

| Component | Mean | p50 | p95 |
|---|---:|---:|---:|
| Existing Phase 4 dense retrieval | 16.207 ms | 11.064 ms | 16.309 ms |
| Sparse retrieval | 23.932 ms | 18.651 ms | 47.881 ms |
| In-memory RRF | 0.930 ms | 0.338 ms | 0.715 ms |

The component-mean sum is 41.069 ms/query. Component-wise p50 and p95 sums
are 30.054 and 64.905 ms/query. These are sums of separately measured stages,
not a jointly timed latency distribution. They exclude one-time model and
index loading. Occasional Python/OS scheduling outliers are retained in the
machine-readable maximum timings.

## Descriptive overlap diagnostics

These measurements describe ranking differences; they do not measure
relevance.

| Diagnostic | Dense vs sparse | Dense vs hybrid |
|---|---:|---:|
| Chunk rank-1 agreement | 34/1,200 (2.83%) | 221/1,200 (18.42%) |
| Document rank-1 agreement | 154/1,200 (12.83%) | 285/1,200 (23.75%) |
| Mean top-10 chunk Jaccard | 0.0535 | 0.2840 |
| Mean top-10 document Jaccard | 0.0738 | 0.4892 |
| Mean full candidate chunk Jaccard | 0.0795 | 0.4264 |

Candidate concentration also differs:

| Method / depth | Mean unique docs | Median max chunks from one doc | Maximum |
|---|---:|---:|---:|
| Dense / 50 | 27.67 | 11 | 50 |
| Sparse / 100 | 21.34 | 24 | 84 |
| Hybrid / 50 | 21.34 | 13 | 50 |

No result list contained duplicate chunk IDs. Across the source collection,
62 exact-text groups contained repeated chunk text, accounting for 929 rows
beyond each group's first occurrence (16.46% of chunks). Seven groups repeat
within at least one document and 55 span different documents. These often
reflect short repeated headings or boilerplate. They are diagnosed but not
merged, because one corpus row remains one canonical document ID.

## Multilingual sanity check — not evaluation

Across the top 10 results for all queries:

| Method | Vietnamese signal | Han-script signal | Other/unknown |
|---|---:|---:|---:|
| Dense | 5,455 | 5,452 | 1,093 |
| Sparse | 11,999 | 1 | 0 |
| Hybrid RRF | 10,023 | 1,605 | 372 |

Dense returned at least one Han-script top-10 result for 1,059 queries and at
least one Vietnamese-signal result for 1,047. Sparse returned Vietnamese-signal
results for all 1,200 and a Han-script result for one query. Hybrid returned
Vietnamese-signal results for all queries and Han-script results for 615.

This matches the expected mechanics: multilingual dense embeddings can
surface cross-script candidates, while lexical retrieval favors same-token and
same-script matches for Vietnamese queries. It does not establish which
results are relevant. Only 8 dense and 2 hybrid top-10 rows received the weak
`latin-other-signal`; that heuristic is not reliable enough to claim measured
English-language behavior.

Forty queries, stratified by query length, were inspected side by side. The
methods frequently selected different documents. Each method showed both
topic-related examples and non-specific, title-only, or boilerplate matches.
The pilot covers only 1,097 documents, so this inspection cannot support a
quality comparison.

## Reproduction and artifacts

```powershell
.venv\Scripts\python.exe scripts\build_sparse_index.py
.venv\Scripts\python.exe scripts\retrieve_sparse.py
.venv\Scripts\python.exe scripts\retrieve_hybrid.py
```

Operational settings are in `configs/hybrid_retrieval.yaml`. Compact JSON
summaries are under `artifacts/phase6_hybrid/`. The 49.67 MiB sparse index and
full sparse/hybrid Parquet rankings remain under ignored `data/indexes/` and
`data/retrieval/` directories.

## Limitations and Phase 7 handoff

- No ground-truth evaluator, leaderboard score, F2 score, or quality tuning is
  included.
- BM25 does not perform Vietnamese word segmentation and Han n-grams are only
  a lightweight deterministic approximation.
- Exact repeated text is retained across distinct document IDs.
- The baseline uses the bounded Phase 3/4 pilot rather than the full corpus.
- Hybrid latency is assembled from separately measured components.
- No score-normalized fusion, query expansion, learned fusion, reranking, or
  final submission is implemented.

Phase 7 can consume the persisted dense, sparse, and RRF candidate rankings,
their common provenance schema, both document aggregations, and the recorded
source hashes. Relevance decisions must wait for trustworthy external or
leaderboard evidence.
