# Phase 11A — Bottleneck Audit

## Purpose

Phase 11A pauses acquisition after the organizer-best `FULL_MEDLATEC_10000`
result (`FINAL=0.0426`). It isolates three possible ceilings while keeping the
organizer-valid corpus and model contracts fixed:

1. output truncation;
2. per-source candidate truncation;
3. missing corpus breadth.

No Phase 11A probe crawls a URL.

## Organizer output rule

The authoritative competition API/schema requires all 1,200 queries and the
`relevant_docs`/`relevant_chunks` arrays, but specifies no exact or maximum
array length. Empty arrays are legal. Returning more items therefore changes
precision through its ordinary denominator; no separate depth penalty or
10/20 maximum is documented. The project’s 10-document/20-chunk policy was an
internal Phase 10 default.

Evidence is recorded in
`artifacts/phase11a_bottleneck/output_rule_report.json`, with the repository
schema preserved at `artifacts/phase9_submission/organizer_schema.json`.

## Controlled probes

### Output depth: 20 documents / 40 chunks

The probe uses the frozen `FULL_MEDLATEC_10000` ranking. It performs no
retrieval or model inference. For all 1,200 queries, the first 10 documents and
first 20 chunks are identical to the organizer-best submission. The only
variable is emitted list depth.

- ZIP: `submissions/10_SUBMITTED_VALID/phase11a_OUTPUT20_CHUNKS40.zip`
- SHA-256: `9aa265f43650d20b439581213c3a74dfab9db4cc6a0ade0878344171654aea93`
- status: `SUBMITTED_VALID`; organizer FINAL `0.0470`

### Candidate depth: m=16

The candidate probe recomputes BGE-M3 dense and BM25/RRF source competition
over the exact nine-source Medlatec-10K corpus. It doubles only the per-source
cap from `m=8` to `m=16`; final output remains 10 documents/20 chunks.

Pre-inference accounting:

- m=8 candidate rows: 86,400;
- m=16 candidate rows and unique keys: 172,800;
- exact cached scores reused: 89,663;
- genuinely missing scores: 83,137.

The persistent FP16 scorer uses batch 2, max length 512, exact experiment
control keys/text, transactional 2,000-pair windows, and the previously
measured numerical-equivalence bound. At the established 20–40 active pairs/s,
missing-pair scoring is expected to take about 35–70 minutes plus packaging.

The completed package is archived at
`submissions/10_SUBMITTED_VALID/phase11a_CANDIDATE_M16.zip` with SHA-256
`67f62c688d37d6b66bf44e244d86bafa7e834dfb82cc09380e808bef6265706e`.
It reached organizer FINAL `0.0450` and is `SUBMITTED_VALID`.

An `m=32` run is deliberately not built. It would contain 345,600 candidate
rows, add at most 172,800 new pairs beyond m=16, require roughly 0.8–1.2 GiB of
candidate/score storage, and about 1.2–2.4 hours of active scoring at measured
rates. Organizer evidence from m=16 must justify that cost.

### Broad existing corpus

The widest provenance-valid local union reconciles:

- 62,766 unique official rows;
- 59,142 usable documents;
- 620,808 chunks;
- 81 domains.

The focused organizer-best core contributes 41,105 usable documents and
512,356 chunks. The additional broad pool contributes 18,037 usable documents
and 108,452 chunks after deterministic official-ID deduplication.

The breadth probe uses a single global BM25 index over only those additional
108,452 chunks and selects 32 lexical candidates per query. This is one global
competition, not equal source quotas. Those candidates are merged with the
unchanged focused m=8 candidates, reranked by exact cached/new scores, and then
compete under the normal 10-document/20-chunk policy. No broad dense-embedding
run is performed.

The organizer changed the official corpus before this package could become a
usable diagnostic. Its old-corpus candidate universe is stale; Phase 11B
reconciles membership first and does not treat this unfinished package as
leaderboard evidence.

## Organizer interpretation (predeclared)

| Result | Interpretation | Next action after organizer evidence |
|---|---|---|
| Output 20/40 improves strongly | output depth is a major ceiling | use reserved slots to tune legal output depth |
| m=16 improves strongly | candidate recall is a major ceiling | test m=32 or a stronger first stage |
| broad existing improves strongly | corpus coverage dominates | authorize a Phase 11B broad checkpoint |
| all probes are flat | deeper retrieval/ranking quality issue | investigate routing and lexical/dense fusion before crawling |

Two of today’s five leaderboard slots remain reserved until the three probe
scores are known.

## Phase 11B plan (not launched)

If breadth wins, the first checkpoint should contain approximately 100K–200K
official documents across three allocation tiers:

- Tier A: proven sources receive the largest deterministic budgets;
- Tier B: near-cutoff and previously eliminated healthy sources receive
  meaningful bounded budgets;
- Tier C: every remaining healthy domain retains a small exploration budget.

The retrieval architecture should build a disk-backed broad BM25/lexical
index first, shortlist a bounded top-N per query, apply dense/hybrid refinement
only where embeddings already exist or to the shortlist, and rerank only exact
missing pairs. It should not dense-embed every acquired document by default.

From the current 41K core, a 100K checkpoint adds about 59K official rows and
a 200K checkpoint about 159K. Recent multi-domain runs imply roughly 4–6 hours
of acquisition for the 100K point and 10–16 hours for 200K, before processing.
At the observed retained footprint of roughly 0.4–0.6 MiB per acquired row,
the incremental retained disk range is approximately 24–35 GiB for 100K and
64–95 GiB for 200K; transient safety headroom and the 20 GiB floor are extra.
Expected lexical index size is a few GiB at 100K and roughly twice that at
200K, depending on chunk density. A 200–400 candidate/query shortlist implies
240K–480K reranker pairs before exact cache reuse, or about 2–5 GPU hours at
the established scorer rate. These are planning ranges, not launch authority.
