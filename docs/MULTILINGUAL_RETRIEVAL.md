# ViBioMIR Multilingual Retrieval

## Current executable slice

`tools/multilingual_retrieval.py` is the reproducible M1 entrypoint. It reads
only frozen local artifacts and creates a 60-unit Vietnamese-query benchmark:
20 VI→VI, 20 VI→EN, and 20 VI→ZH units. Query-language units are assigned to
development or held-out before relevance review by a deterministic SHA-256
split.

```powershell
.\.venv\Scripts\python.exe tools\multilingual_retrieval.py prepare
.\.venv\Scripts\python.exe tools\multilingual_retrieval.py serve-review --split development
.\.venv\Scripts\python.exe tools\multilingual_retrieval.py evaluate
```

The current manifest is
`artifacts/multilingual_retrieval_m1/benchmark_manifest.json`. It pins the
official corpus, query set, rankings, canonical chunks, PubMed documents,
model revisions, and generated benchmark artifacts by SHA-256.

## Executed baseline (2026-10-22)

The cache-only preparation completed in about 11 seconds without loading a
model or using the GPU. It produced 2,260 document-ranking rows and 260 pooled
review rows. Every selected document is a member of the current 1,194,469-ID
official snapshot. Current canonical URL matching and exact Phase4 chunk
ownership/text checks passed with zero failures. Eighty-three historical
artifact URLs differed from the current official URL (mostly URL-scheme
changes); the current official URL is used and the discrepancy remains in the
manifest rather than being silently accepted. The preparation also excluded
375 historical Phase7 candidate-universe document IDs that are absent from
the current official snapshot.

| Track | Frozen retrieval arms | Queries | Observed depth/query |
|---|---|---:|---:|
| VI→VI | dense, lexical, hybrid, reranker, Phase29 | 20 | 1–20 depending on arm |
| VI→EN | translated dense, lexical, hybrid | 20 | 20 for every arm |
| VI→ZH | dense, hybrid, reranker, Phase29 | 20 | 1–9 for Phase7 arms; Phase29 is sparse |

Frozen Phase29 is also read directly from its validated packaging SQLite as a
baseline arm. Its ZIP SHA-256 is checked before use, and supporting chunk IDs
are recovered only by exact `(doc_id, expanded_text)` matches.

The English arm reuses the frozen Phase19 Helsinki translation and PubMed
retrieval. The Vietnamese and Chinese arms reuse Phase7 candidates from the
5,643-chunk mixed-language pilot. Consequently, this proves executable paths
and provenance closure, not full-corpus Chinese coverage.

## Relevance contract

The review pool takes the top document from every frozen arm and fills
deterministically to at most five candidates per query-language unit. The
editable file is
`artifacts/multilingual_retrieval_m1/human_labels.csv`; regeneration writes a
separate template and never overwrites rows once any row is marked
`HUMAN_VERIFIED`.

The development review service binds only to `127.0.0.1:8765`. It hides the
retrieval arm, rank, and score, stores drafts in browser local storage, and
writes explicit user judgments directly to `human_labels.csv` through an
atomic local POST. A backup CSV download remains available. The held-out
review has a separate file and must remain unopened until the development
policy is frozen.

Allowed judgments are `2` (directly relevant), `1` (partially relevant), `0`
(not relevant), and `UNSURE`.
Metrics only consume rows that also contain a reviewer and verification
timestamp. Until verified labels exist, evaluation deliberately returns
`INSUFFICIENT_VERIFIED_LABELS` and null Hit@20/MRR@20/Recall@20. Old Codex or
LLM judgments are not imported as gold.

Metrics are calculated separately for every language and retrieval arm using
only fully judged query-language units and label-2 documents. Unjudged
documents are not assumed to be non-relevant, so reported recall is explicitly
pooled Recall@20 rather than exhaustive corpus recall. Label-1 evidence is
reported as partial support but is not silently promoted to a direct positive.

## Known gaps before M2/M3 completion

- The Chinese branch is shallow and has no translated-query lexical arm.
- No human-verified row exists yet, so no relevance gain is claimed.
- No new model, index, reranking inference, competition output, or submission
  was created in this slice.
- Phase27 remained active during preparation; no competing GPU worker was
  started.

The next admissible engineering step is to complete the frozen human review,
then use development labels to choose one cross-language change while keeping
the held-out labels sealed. Full Chinese acquisition or new GPU inference is
not justified until the existing branch has measured relevance failures.
# M2 development vertical slice

The CPU/cache-only M2 vertical slice is implemented by `tools/multilingual_m2.py`. It validates imported AI labels as `AI_JUDGED` (never `HUMAN_VERIFIED`), applies Chinese extraction-quality filtering, builds a bounded Chinese character-bigram FTS index, and runs the frozen query-only medical English bridge against the existing PubMed FTS5 index. Results are written to `artifacts/multilingual_retrieval_m2/`; held-out review data is not used.
