# Phase 7 — Multilingual Reranking Baseline

## Scope and evidence policy

Phase 7 adds a local cross-encoder reranking and deterministic candidate
selection layer over the unchanged Phase 6 dense, sparse, and RRF rankings.
It does not implement an evaluator, report F2, generate a submission, or make
a relevance-quality claim. All comparisons below are operational or
descriptive sanity checks.

The experiment covers the Phase 3/4 pilot corpus: 5,643 validated chunks from
1,097 documents and all 1,200 official queries.

## Model and hardware

The reranker is `BAAI/bge-reranker-v2-m3`, pinned to revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. It is an
`XLMRobertaForSequenceClassification` model with 567,755,777 parameters. The
pipeline uses local inference only, FP16 on CUDA, float32 output scores,
longest-first truncation, and an operational maximum sequence length of 512.
No model weights or cache files are committed.

The benchmark used the 64 longest query/chunk pairs from the candidate pool;
all were truncated to 512 tokens.

| Batch | Result | Pairs/s | Mean ms/pair | Peak allocated | Peak reserved |
|---:|---|---:|---:|---:|---:|
| 1 | success | 17.48 | 57.19 | 1.076 GiB | 1.090 GiB |
| 2 | success | 20.83 | 48.01 | 1.087 GiB | 1.104 GiB |
| 4 | CUDA OOM | — | — | — | — |

Batch size 2 was therefore selected for the RTX 3050 Laptop GPU (4 GB). Batch
4 must not be retried on this hardware; it is listed as known-unsafe and
excluded from configured benchmark sizes. The batch-2 run retained about
2.90 GiB of VRAM beyond peak reserved memory.

## Candidate pool and provenance

For each query the unified pool reads dense top 50, sparse top 100, and hybrid
RRF top 50, then deduplicates by `chunk_id`. Source ranks and scores are kept
in separate fields together with source membership, `doc_id`, exact
`chunk_text`, URL, offsets, section type, heading path, and extraction method.
The union contained 167,731 candidates before the deterministic top-100 pool
cut and exactly 120,000 scored pairs afterward.

The persisted pool SHA-256 is
`9ffb934a0cfcfba7fcb11472179bab3aa5c7853ad147b6bceddfb4b521d849bc`.
Hashes of the unchanged Phase 6 dense, sparse, and hybrid inputs are recorded
in `artifacts/phase7_reranking/candidate_pool.json` and in the checkpoint
signature.

## Resumable scoring

GPU scoring and CPU finalization are separate operations. Scores are written
transactionally to SQLite under the primary key `(query_id, chunk_id)`. A
checkpoint can only be opened when its model name, exact revision, precision,
batch size, maximum length, and all candidate-input hashes match.

The completed checkpoint passed `PRAGMA integrity_check`, contains exactly
120,000 unique score rows, and has no missing or extra key relative to the
120,000-row pool. The final scoring run resumed from 8,704 existing rows and
scored the remaining 111,296 without replacing completed scores.

Summed per-pair inference timing stored across the resumed checkpoint is
3,813.47 seconds (63.56 minutes), or 31.47 pairs/s. This is the best available
complete-run measurement; the separately recorded finishing process took
3,520.32 seconds for its 111,296 new pairs, excluding model load. Model load
for that run took 60.32 seconds.

## Reranking depths and outputs

Depths 20, 50, and 100 were finalized for every query:

| Depth | Rows | Top-1 changed from pool order | Mean absolute movement |
|---:|---:|---:|---:|
| 20 | 24,000 | 86.50% | 5.60 |
| 50 | 60,000 | 89.00% | 13.73 |
| 100 | 120,000 | 89.75% | 26.81 |

These are rank-movement measurements, not evidence of improved relevance.
Every output has a complete per-query rank sequence, unique query/chunk keys,
and exact `chunk_id`/`doc_id`/text/URL correspondence with the source pool.

The default candidate depth is 100. Final chunk selection retains up to 50
chunks and document ranking retains up to 10 documents.

## Selection and duplicate controls

Three deterministic output variants make control effects observable:

| Variant | Exact-text suppression | Max chunks/doc | Short heading penalty | Selected rows | Mean unique docs |
|---|---|---:|---|---:|---:|
| `pure_rerank` | off | unlimited | off | 60,000 | 20.96 |
| `max_chunks_per_doc` | off | 3 | off | 56,441 | 31.48 |
| `controlled` | on | 3 | on | 55,500 | 29.92 |

The max-three variant suppressed 42,107 examined candidates. The controlled
variant suppressed 3,262 exact-text repeats and 44,422 candidates over the
per-document cap; 7,724 short title/heading candidates received the explicit
1.0 score penalty. Counts describe selection mechanics and may overlap because
candidates are examined until the output fills or the depth-100 pool ends.

Controls never merge document IDs globally and never rewrite text. The
conservative default remains pure rerank; the alternatives are persisted for
future externally grounded comparison.

## Document aggregation

Both existing aggregation rules are applied after each selection variant:

- `best_chunk`: a document receives its highest reranker score.
- `top_n_mean`: a document receives the mean of its best three reranked
  chunks.

Ties are deterministic and document outputs are deduplicated by `doc_id`.
Both methods cover all 1,200 queries, with zero duplicate document IDs within
a query. No relevance winner is selected without ground truth; `best_chunk`
remains the default handoff setting.

## Latency

Checkpoint timing gives a mean/p50/p95 reranker inference cost of
3,177.89/3,135.75/3,373.54 ms per query for 100 pairs. CPU post-processing was
1.94/1.00/2.93 ms, and document aggregation was 0.62/0.58/0.99 ms. Including
mean candidate preparation, total measured mean/p50/p95 was
3,181.83/3,139.38/3,381.21 ms per query. One-time model load is excluded.

The final CPU-only materialization exited successfully and its core work took
28.29 seconds. Large Parquet outputs and the SQLite checkpoint stay under the
ignored `data/retrieval/phase7_reranking/` directory.

## Source and multilingual diagnostics — not evaluation

At depth 100, reranking changed top-1 relative to pool order for 1,077 of
1,200 queries. In the top 10, candidates present only in dense, only in
sparse, and in overlapping source sets all moved upward. These movements show
that the cross-encoder materially changes composition; they do not establish
that the changes are beneficial.

The lightweight top-10 text signals after reranking were 6,608 Vietnamese,
4,748 Han-script, 643 unknown, and 1 Latin-other row. At least one Vietnamese
signal appeared for 1,076 queries and at least one Han-script signal for 979.
These are script/diacritic heuristics, not content-language identification.

A deterministic 40-query sample, evenly spaced by query length, compares
dense, sparse, hybrid, and reranked top results side by side. Reranking changed
the hybrid top result for 38 of 40 sampled queries. Inspection shows a mixture
of biomedical-term matches, cross-script Chinese Q&A promotions, useful
Vietnamese passages, generic headings, and residual boilerplate or unrelated
news passages. Short ambiguous queries and long detailed queries both exhibit
mixed behavior. This is a **SANITY CHECK, NOT EVALUATION**.

## Reproduction

Configuration is in `configs/reranking.yaml`. The normal staged commands are:

```powershell
.venv\Scripts\python.exe scripts\prepare_rerank_pool.py
.venv\Scripts\python.exe scripts\rerank_candidates.py --score-only
.venv\Scripts\python.exe scripts\rerank_candidates.py --finalize-only
```

The first two commands must not be run when resuming the completed Phase 7
state. Finalization is CPU-only and does not import or initialize the model.
Compact evidence is stored in `artifacts/phase7_reranking/`; checkpoints,
full rankings, model weights, and caches remain ignored.

## Limitations and Phase 8 handoff

- The reranker sees at most 512 tokens per query/chunk pair.
- Results cover the bounded pilot corpus, not all 4.39 million URLs.
- Language fields are heuristic signals rather than measured language labels.
- Candidate depth, control settings, and aggregation have no ground-truth or
  leaderboard calibration.
- No evaluator, F2 score, MMR selection, submission generation, or Phase 8
  implementation is included.

Phase 8 can consume the persisted source ranks/scores, reranker scores,
selection variants, document rankings, provenance, and source hashes. Any
claim about relevance must be grounded in trustworthy external evidence.
