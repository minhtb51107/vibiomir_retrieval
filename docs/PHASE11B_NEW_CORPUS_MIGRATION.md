# Phase 11B — New Official Corpus Migration

## Organizer corpus change

On 2026-10-12 the authoritative corpus changed from
`links_corpus.parquet` to `links_corpus_vi_en_zh.parquet`. Document IDs remain
the join key, but membership—not the historical file—is now authoritative.
The old file is retained only as historical evidence.

| Measure | Old corpus | New corpus |
|---|---:|---:|
| Rows / unique IDs | 4,394,718 | 1,194,469 |
| Retained IDs | — | 954,469 |
| Removed IDs | — | 3,440,249 |
| Newly introduced IDs | — | 240,000 |
| Duplicate IDs | 0 | 0 |
| Duplicate URLs | not re-audited | 0 |

The new parquet SHA-256 is
`9e34cb4b7d6424b80ed2634ada581ea9e2d97da69c245d80bdbb31413a78e8ce`.
No retained ID changed URL. Organizer-declared composition is 714,469
Vietnamese, 240,000 Chinese, and 240,000 English rows. Exact set accounting
found 3,414,120 removed Chinese IDs and all 26,129 historical `vinmec.com`
IDs absent from the new file. The latter is recorded explicitly rather than
silently treated as retained Vietnamese content.

## Local retention impact

| Local universe | Old usable/local docs | Retained docs | Removed | Retained chunks | Domains |
|---|---:|---:|---:|---:|---:|
| Focused Medlatec-10K union | 41,210 | 26,639 | 14,571 | 444,411 | 7 |
| Phase 11A M16 corpus | 41,210 | 26,639 | 14,571 | 444,411 | 7 |
| Broad-existing inventory | 59,142 | 35,690 | 23,452 | 523,482 | 65 |

The largest focused removals are all Vinmec content plus reduced Chinese
memberships. Retained focused Chinese-heavy counts include 440
`cancer.39.net` documents and 14 `zydcd.com` documents. The wider retained
inventory includes 320 `familydoctor.com.cn` documents and smaller retained
samples from the 39.net family and other Chinese-associated domains.

The new corpus introduces exactly 240,000 PubMed IDs. None has local content,
so 240,000 remain unacquired. Phase 11B performed no crawling.

## Why the old broad package became stale

Phase 11A packages were valid under the historical 4.39M-row membership.
They remain diagnostic history, but they are not valid new-corpus controls:
their candidate and output universes can include IDs removed by the organizer.
A new baseline therefore required candidate reconciliation, not merely output
filtering.

## Clean control construction

The new control keeps the proven M16 retrieval/reranker contract but reconciles
every source candidate part to new membership. Five unaffected parts were
reused; `cancer.39.net` and `zydcd.com` were recomputed over their retained
local corpora. The frozen pilot pool was also filtered to the new corpus:
91,971 of 120,000 rows remained, while 1,715 unique chunks belonging to 669
removed documents were excluded.

The final score database contains exactly 134,400 unique query/chunk keys.
All 134,400 have finite scores, zero keys are missing or unexpected, SQLite
integrity is `ok`, and packaging required zero model inference.

## Packaging incidents and durable lessons

Four recovery incidents were corrected before readiness:

1. Nested filtered-parquet destinations did not create their parent directory.
   The generic writer now creates parents before opening its transactional file.
2. A chunk subset was incorrectly selected by `doc_id`; exact `chunk_id`
   selection is required when candidate identity is the contract.
3. The first canonical store omitted valid frozen-pilot chunks while rankings
   retained them, and the pilot pool still included removed official IDs.
   Candidate membership and canonical provenance are now reconciled together.
4. Phase 11B expected `validation.passed`, although strict submission
   validators historically expose `valid` and raise on errors. The shared
   normalization layer now requires the complete known schema and zero error
   counters; missing fields cannot default to success.

The final provenance closure contains 57,057 ranked and canonical unique chunk
IDs with zero missing, unexpected, non-official, conflicting, mismatched, or
non-reconstructable entries.

## Clean document Recall@K design

Every package uses one fixed ranking pool and exactly 20 identical source-
derived chunks per query. Only the number of emitted documents changes. Unique
ranked-document depth per query is:

| Minimum | p5 | Median | p95 | Maximum |
|---:|---:|---:|---:|---:|
| 54 | 80 | 96 | 111 | 120 |

The common scientifically valid depths are therefore 10, 20, and 50.
Docs@100 is not produced because some queries have only 54 unique ranked
documents; no padding, duplication, or candidate-pool change is allowed.

| Package | Documents/query | Chunks/query | ZIP SHA-256 | Status |
|---|---:|---:|---|---|
| `phase11b_NEW_CORPUS_CONTROL.zip` | 10 | 20 | `d094358409475d28a71dc8591a75d71c0412d8b9dc83d69bd93de1e4b494b35a` | `READY_FOR_LEADERBOARD` |
| `phase11b_DOCS20_CHUNKS20.zip` | 20 | 20 | `9a4a9a3adc9d0f1319b924b34ea7f11abaab29b8d8516e734e3176b8131d12c9` | `READY_FOR_LEADERBOARD` |
| `phase11b_DOCS50_CHUNKS20.zip` | 50 | 20 | `069480b6ec2b88ab1a00ee17ec22da4c0e9409debff35b958eddc2984453bfee` | `READY_FOR_LEADERBOARD` |

All three contain 1,200 queries, new-official IDs only, deterministic ZIPs,
zero duplicate outputs, zero provenance mismatches, and exactly 20 chunks per
query. Organizer results are pending.

## Decision after organizer results

- Recall rising strongly through K=50 indicates that useful documents already
  exist deeper in the fixed ranking; prioritize legal output-depth policy.
- An early plateau indicates that candidate or retained-corpus coverage is the
  tighter ceiling; use remaining quota to isolate those layers.
- PubMed acquisition, ICD routing, M32, new models, and deeper source crawling
  remain deferred until this clean curve is measured.

Evidence: `artifacts/phase11b_new_corpus/migration_audit.json`,
`provenance_closure.json`, `phase11b_report.json`, and
`phase11b_pre_submit_audit.json`.
