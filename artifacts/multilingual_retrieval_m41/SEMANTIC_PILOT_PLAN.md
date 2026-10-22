# M4.1 bounded semantic pilot — deferred

Execution is deferred while any Phase27 Python worker is active. The
`semantic-preflight` command only inspects processes and resource telemetry;
it never loads a model or starts an inference worker. There is no auto-launch.

## Frozen comparison

Use the existing M4 corpus only: 368 current-official documents, 2,188 chunks.
Do not re-extract, crawl, change chunk text, use held-out, or move the frozen
M4/M4.1 lexical results. Freeze the corpus SHA-256, M4.1 term configuration
SHA-256, all 12 development query texts and the model configuration before
inference. Query ID is evaluation metadata, never required by retrieval.

Compare lexical M4.1, original-Vietnamese dense, automatically represented
Chinese dense, and a union with document deduplication at the same candidate
budget. Separate candidate inclusion from final evidence admission. Never
replace a missing exact score with a numeric sentinel labelled as inference.

## Initial resource/throughput trial

After Phase27 finishes and all GPU children exit, run only five deterministic
development units (ascending SHA-256 of query text, not silver labels). Embed
the five Vietnamese queries, five automatic Chinese representations and ten
deterministically selected source chunks: twenty texts total. Record load
time, elapsed time, tokens, throughput, peak process-tree RSS/private bytes,
minimum available RAM/commit headroom and GPU allocated/reserved memory.
Stop on a resource failure or nonfinite output. Use the existing repository
embedding and reranker resource/equivalence gates; do not lower their limits.

Use the locally pinned BGE-M3 contract in
`configs/source_relevance_probe.yaml`: revision
`5617a9f61b028005a4858fdac845db406aefb181`, dimension 1024, maximum length 512,
normalized vectors, seed 2026, CUDA half precision. Validate the exact contract
and permitted batch context with the existing equivalence control before any
production-sized work. These settings specify a future test; no model was
loaded in M4.1.

Only after the trial passes and measured ETA is acceptable, stream the 2,188
existing chunks into a separate resumable vector file/bitmap. Preserve the
canonical row manifest and model/config hashes. Use exact vector search over
this small corpus; do not rebuild large indexes. Stop rather than silently
changing precision, truncation, normalization, model revision or tolerances.

## Reranker trial and review

Reuse the input-compatible historical references recorded in
`cache_score_recovery.csv`, checking model signature and exact query/chunk text
before import. The pinned reranker is `BAAI/bge-reranker-v2-m3`, revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, tokenizer at the same revision,
512-token pair limit, FP16, batch size 2. Request scores only for missing keys
in a bounded pool of at most eight document candidates per development unit
(96 pairs), after a strict reference equivalence trial. Record truncation and
whether the answer span survives tokenization. Cache by full input/contract
hash, with finite-score validation and durable checkpoints.

Report where each of the 13 known M3 partial pairs appears, plus any newly
retrieved evidence. Have an independent reviewer read source excerpts with
retrieval arm/rank/score hidden. Keep relevance, completeness and provenance
as separate fields. Treat AI review as provisional silver. Higher dense or
reranker score is not a direct-answer label. No official Recall denominator
or leaderboard improvement can be inferred from this pilot.

GO requires acceptable measured resources, valid provenance, no known-positive
candidate loss and source-reviewed new answer-bearing evidence. Otherwise
record the failure and preserve the checkpoint. Do not start another crawl.
