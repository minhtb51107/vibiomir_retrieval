# Phase 10B2 — Source Acquisition Benchmark

## Scope and decision rules

This is an acquisition-engineering benchmark, not relevance evaluation. It
does not establish that any source is useful for leaderboard queries.

The gates were committed to
`artifacts/phase10b2_acquisition/decision_rules.json` before new Phase 10B2
requests:

- checkpoint every source at 200 URLs;
- stop if robots/access restriction exceeds 80%, usable extraction is below
  20%, or repeated anti-bot behavior becomes unsafe;
- continue passing sources to at most 1,000 URLs;
- Tier A requires at least 80% usable extraction, no major access blocker,
  acceptable throughput, and a safe projected 50k footprint;
- Tier B covers 50–80% usable extraction with moderate operational cost;
- Tier C is below 50% usable or operationally problematic;
- BLOCKED means robots/protection makes acquisition inappropriate.

`zysjonline.com` was not requested. Existing Phase 2/8 evidence already shows
it is robots-blocked.

## Sampling and reuse

All samples use the lowest SHA-256 ranks of `seed:doc_id` within each domain,
with seed `vibiomir-phase8-bounded-v1`. This spreads selection over the full
domain population rather than adjacent IDs. The exact 6,000-row manifest and
reuse flags are in `artifacts/phase10b2_acquisition/sample_manifest.json`.

The seed matches the verified Phase 8 5,000-URL sample. Results and bodies were
reused only when both `doc_id` and `original_url` matched. This supplied 2,279
of the 4,400 ultimately benchmarked rows; 2,121 new URLs were fetched. Reused
rows were copied into independent resumable SQLite/archive checkpoints before
processing. No source exceeded 1,000 attempted URLs and the total new-URL cap
was not approached.

The crawler, robots policy, retry logic, compressed archive, generic extractor,
and real cached BGE-M3 tokenizer were reused unchanged. Chunking used 508
content tokens, two special tokens, and overlap 64. No embeddings or indexes
were built.

## Usable-document definition

A document is counted as usable when it has:

- extraction status `SUCCESS`;
- at least 500 cleaned characters;
- at least two paragraphs;
- at least one deterministic Chinese, Vietnamese, or English medical term.

This is deliberately coarse content sanity, not relevance accuracy. Language
labels use Han/Latin/Vietnamese-character ratios and are heuristic.

## Results

Storage is retained crawl metadata + compressed bodies + cleaned-document
Parquet + chunk Parquet. Time combines observed source-specific wall time for
new requests with a conservative modeled cost for reused Phase 8 rows, plus
measured extraction/chunking time.

| Source | Corpus share | Attempted | Fetch success | Usable docs | Usable % | Language signal | URLs/min | Usable/min | Avg raw / compressed bytes | Chunks/usable doc | Projected 50k storage | Projected 50k time | Tier | Main blocker |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| cnkang.com | 21.923% | 1,000 | 100.0% | 713 | 71.3% | mostly Chinese | 30.0 | 21.4 | 34,433 / 9,022 | 2.68 | 831 MiB | 28.6 h | B | Slow configured source rate; 224 short/navigation-like extractions |
| 120ask.com | 20.900% | 200 | 100.0% | 29 | 14.5% | mostly Chinese | 30.0 | 4.4 | 67,477 / 17,497 | 3.83 | 987 MiB* | 29.6 h* | C | **Early stop:** usable extraction below 20%; 171/200 short |
| familydoctor.com.cn | 10.182% | 1,000 | 99.0% | 828 | 82.8% | mostly Chinese | 36.2 | 30.0 | 65,032 / 15,841 | 2.86 | 887 MiB | 24.2 h | A | Ten timeouts; 35 retries in new continuation |
| ask.39.net | 7.677% | 200 | 99.5% | 0 | 0.0% | mostly Chinese | 27.9 | 0.0 | 47,768 / 10,954 | 0.00 | 600 MiB* | 31.0 h* | C | **Early stop:** all successful extractions under usable-content rule |
| a-hospital.com | 3.853% | 1,000 | 100.0% | 848 | 84.8% | mostly Chinese | 59.6 | 50.5 | 42,969 / 12,040 | 9.88 | 1,179 MiB | 15.2 h | A | High text/chunk density; no access blocker observed |
| suckhoecongdongonline.vn | 3.516% | 1,000 | 100.0% | 998 | 99.8% | mostly Vietnamese | 59.2 | 59.1 | 42,864 / 11,172 | 4.58 | 1,022 MiB | 16.3 h | A | No blocker observed |

`*` Projections for stopped sources are diagnostic extrapolations only and are
not recommendations to scale.

Extraction itself succeeded for 1,000/1,000 CnKang, 200/200 120Ask,
990/1,000 FamilyDoctor, 199/200 Ask39, 1,000/1,000 A-Hospital, and
1,000/1,000 Vietnamese-control rows. The gap between extraction success and
usable rate is therefore primarily content length/structure, not parser crashes.

## Chunk projections

| Source | Mean / median / p95 chunks per usable doc | Projected chunks per 10k usable docs |
|---|---:|---:|
| cnkang.com | 2.68 / 2 / 5 | 26,802 |
| 120ask.com | 3.83 / 3 / 6 | 38,276* |
| familydoctor.com.cn | 2.86 / 3 / 3 | 28,647 |
| ask.39.net | 0 / 0 / 0 | 0* |
| a-hospital.com | 9.88 / 7 / 20 | 98,797 |
| suckhoecongdongonline.vn | 4.58 / 4 / 9 | 45,792 |

A-Hospital is inexpensive to acquire but produces roughly 3.5 times as many
chunks per usable document as FamilyDoctor, so downstream index cost is
materially higher.

## Source-efficiency score

For planning only:

`SOURCE_EFFICIENCY = corpus_share × usable_doc_rate / normalized_resource_cost`

where normalized resource cost equally weights seconds per usable document and
MiB per 1,000 usable documents, each normalized by the median of sources that
passed the early gate. The resulting order is CnKang, FamilyDoctor, Vietnamese
control, then A-Hospital. CnKang leads only because it represents 21.923% of
the official corpus; it remains Tier B rather than the safest first scale.

## Recommendation

1. **Safest source to scale first:** `suckhoecongdongonline.vn`. It delivered
   99.8% usable documents, 59.1 usable docs/minute, no observed access blocker,
   and moderate chunk density.
2. **Second safest:** `a-hospital.com`. It delivered 84.8% usable documents and
   50.5 usable docs/minute, but downstream chunk/storage density is higher.
3. **Largest weighted opportunity:** `cnkang.com`; consider only after the
   safer milestone because its usable rate is 71.3% and throughput is slower.
4. **Avoid for now:** stop `120ask.com` and `ask.39.net` under the declared
   content-yield gate; continue treating `zysjonline.com` as BLOCKED.

The exact proposed next milestone is **20,000 deterministic
`suckhoecongdongonline.vn` rows**, not 50,000. Projection:

- approximately 19,960 usable documents;
- approximately 91,400 chunks;
- about 409 MiB retained benchmark-equivalent storage;
- about 0.80 GiB raw network transfer;
- about 6.5 hours source-exclusive pipeline time.

With 37.31 GiB free on D: at finalization and a 20 GiB stop threshold, this
milestone fits the current laptop with substantial headroom. It must still use
the existing bounded confirmation, free-space guard, checkpoints, and archive;
this phase did not start it.

## Limitations

- Acquisition value is not relevance quality.
- Linear 10k/50k/full-source projections may miss temporal blocking or page
  distribution shifts.
- Historical reused rows use measured per-row latency plus configured pacing,
  while new rows use observed wall time.
- The medical vocabulary rule penalizes short Q&A answers, intentionally
  enforcing the predeclared usable-text threshold; a future relevance-backed
  study could revisit that definition, but this phase did not move the gate.
- No browser rendering, anti-bot bypass, embeddings, retrieval index, or
  leaderboard submission was used.
