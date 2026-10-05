# Phase 10B3 — Source Relevance Probe

## Scope

This phase tests whether each already-acquired Phase 10B2 source sample can
change retrieval outputs when added independently to the fixed Phase 7 pilot.
It made no network requests and did not crawl new URLs. The comparison is a
local signal probe, not relevance evaluation; leaderboard quality remains
unknown until the organizer scores a submission.

The independent variable is the added source. Every variant uses the same
1,200 queries, BGE-M3 tokenizer and embeddings, Phase 6 BM25/RRF candidate
construction, `BAAI/bge-reranker-v2-m3` revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, and submission policy.

## Fixed experiment

| Variant | Added source | Added docs | Added chunks |
|---|---|---:|---:|
| B0 | Phase 7 pilot only | 0 | 0 |
| S1 | cnkang.com | 1,000 | 2,254 |
| S2 | familydoctor.com.cn | 990 | 2,736 |
| S3 | a-hospital.com | 999 | 8,724 |
| S4 | suckhoecongdongonline.vn | 1,000 | 4,575 |
| S5 | 120ask.com short Q&A | 161 | 507 |
| S6 | ask.39.net short Q&A | 132 | 442 |

S5 and S6 use the predeclared relevance-probe rule: extraction status
`SUCCESS`, at least 100 cleaned characters, at least one paragraph, at least
one deterministic medical signal, and no shell, empty, failed,
access-restricted, or robots-blocked status. This intentionally differs from
the Phase 10B2 long-article acquisition gate; it preserves meaningful short
Q&A text without rewriting it.

The submission policy is pure reranker order, best-chunk document aggregation,
top 10 documents, top 20 chunks, and contiguous source-verbatim expansion to
approximately 1,024 BGE-M3 tokens. B0 and S1–S6 use exactly the same policy.

## Resource and scoring controls

The 19,238 new chunks required 78,798,848 bytes of float32 embeddings and took
303 seconds on the RTX 3050 Laptop GPU. Existing pilot embeddings were reused.
The largest exact combined index covered 14,367 chunks; the preflight estimate
of 217,481,216 bytes for combined exact indexes stayed well below the 10 GiB
stop gate.

Candidate preparation produced six 120,000-row pools. Phase 7 scores were
reused wherever `(query_id, chunk_id)` matched. The 95,255 missing variant
memberships deduplicated to 90,349 unique novel pairs, saving 4,906 duplicate
inferences.

An isolated 2,048-pair sustained benchmark passed at 36.42 pairs/s. Working-set
memory settled near 0.15–0.19 GiB after initialization, private bytes remained
stable near 7.8 GiB, peak CUDA allocation was 1.09 GiB, and peak CUDA reserved
memory was 1.11 GiB. There was no progressive paging collapse.

Production scoring used 18 independent fresh processes of at most 5,000 pairs,
FP16 CUDA, batch size 2, and maximum sequence length 512. It scored the 88,237
pairs remaining after the sustained benchmark at a weighted 35.11 pairs/s:
2,513.18 seconds inference plus 679.61 seconds of isolated model loads. Every
shard passed SQLite integrity checks. The final global checkpoint contains all
90,349 unique pairs with zero missing rows, and all six distributed variant
checkpoints contain exactly 120,000 rows.

At the final resume boundary, the global checkpoint contained 72,112 scores;
shard 15 was the first incomplete shard and 18,237 pairs remained. Shards
15–18 completed 18,237 pairs in 529.10 seconds inference plus 149.71 seconds
model loading (678.82 seconds measured components).

## Local impact

`Queries changed` compares the complete ordered top-10 document IDs and
top-20 chunk IDs with B0. A source submission was generated only when at least
12 of 1,200 queries changed. This threshold was fixed before finalization.

| Source | Added docs | Added chunks | Queries changed | Source doc in top 10 | Source chunk in top 20 | Local signal |
|---|---:|---:|---:|---:|---:|---|
| a-hospital.com (S3) | 999 | 8,724 | 1,188 | 1,123 | 1,016 | meaningful output change |
| cnkang.com (S1) | 1,000 | 2,254 | 1,167 | 1,047 | 963 | meaningful output change |
| suckhoecongdongonline.vn (S4) | 1,000 | 4,575 | 1,143 | 1,028 | 946 | meaningful output change |
| 120ask.com short Q&A (S5) | 161 | 507 | 1,102 | 960 | 888 | meaningful output change |
| familydoctor.com.cn (S2) | 990 | 2,736 | 1,007 | 619 | 546 | meaningful output change |
| ask.39.net short Q&A (S6) | 132 | 442 | 637 | 253 | 262 | meaningful output change |

These are output-movement measurements, not accuracy or F2. In particular,
S5's broad effect from only 161 documents is operationally interesting but
does not establish that the movement is beneficial.

## Submission validation

B0 and all six source variants contain exactly 1,200 queries. Every variant
passed the strict ZIP validator with zero invalid document IDs, duplicate
document IDs, duplicate chunk objects, and chunk provenance mismatches. Each
contains exactly 24,000 chunk objects (20/query); source-span verification
covered every expanded chunk requiring it. Independent full regeneration
produced byte-identical JSON and deterministic ZIP hashes.

An additional scan checked all 5,507 local pilot/source document IDs against
the official 4,394,718-row corpus and found zero invalid official IDs.

Recommended manual leaderboard order is based only on breadth of local output
change: S3, S1, S4, S5, S2, then S6. B0 should be uploaded or referenced as
the fixed control. Stop after each organizer result if submission budget is
limited. A strong signal is a material DOCS_RECALL or FINAL_SCORE improvement;
a positive signal is another reproducible organizer metric improvement; an
identical displayed result is no visible signal; lower precision/F2 without a
recall gain is negative. No visible signal from a bounded sample does not prove
the full source irrelevant.

## Reproducibility and retained outputs

Configuration is in `configs/source_relevance_probe.yaml`. Compact evidence is
under `artifacts/phase10b3_source_probe/`; large embeddings, candidate pools,
rankings, and SQLite checkpoints remain under ignored
`data/source_relevance_probe/`. Ready-to-upload JSON/ZIP files remain under the
ignored `submissions/` directory. No source scaling or additional crawl was
started.
