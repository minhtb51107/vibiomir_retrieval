# M4.2 fixed-document evidence-window ablation

## Decision

**NO-GO for automatically promoting ZH best-cosine windows or current RRF.**
The experiment is executed, reproducible and provenance-valid. It recovers some
useful partial evidence, but also loses useful context and increases navigation
proxies. No independently verified direct-answer improvement or competition gain
has been established. The 24-pair blind queue remains pending independent review;
HUMAN_VERIFIED = 0. No production retrieval policy was changed.

## Intervention and integrity

A uses the exact saved RRF ranking and representative chunks. B changes only the
representative to the same-owner frozen chunk with maximum **true cosine** against
the frozen automatic Chinese query vector. Norms are evaluated in float64;
ties use lexicographic chunk ID, then original corpus index. No query, document,
document score, rank, budget or admission threshold changes. All 1,200 document
positions and RRF scores are identical between A and B.

Before selection: original ZIP CRC PASS, all 10 extracted members match archive,
SQLite quick_check/integrity_check both `ok`, all 553 windows/2,212 vectors
checksummed, finite normalized FP32 vectors of dimension 1,024. Frozen manifest
text identities/order and current official IDs/URLs match 2,188 chunks and 361
owners. Trial20 explicitly passed **Colab-internal** consistency, not laptop
cross-platform equivalence. Only cached Colab embeddings are used here.

Colab ZIP SHA-256:
`54c6426c00d5766b61c6d961a1f7d526bf148cc58746fe60ac76e1b407ab7a9e`.
Phase27 ZIP before/after SHA-256:
`b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`.
No GPU, inference, reranker-score reuse, crawl, held-out access or submission.

## Measured results

| Measure | Top20 (240 positions) | Top100 (1,200 positions) |
|---|---:|---:|
| Changed representatives | 166 | 553 |
| Queries with changes | 12/12 | 12/12 |
| Navigation proxy A -> B | 54 -> 79 | 294 -> 366 |
| BODY -> NAV among changes | 52 | 143 |
| NAV -> BODY among changes | 27 | 71 |
| Old partial document pairs A -> B | 11 -> 11 | 12 -> 12 |
| Literal old evidence excerpt retained A -> B | 0 -> 2 | 0 -> 2 |
| NFC/whitespace-normalized excerpt retained A -> B | 1 -> 3 | 1 -> 3 |

The last three rows refer to 13 frozen AI-silver partial query-document pairs,
not official Recall. Retention means the old excerpt is contained verbatim, or
after normalization, in the selected current chunk; chunk IDs changed during
rechunking. Only 5/13 literal and 7/13 normalized old excerpts exist in any current
same-owner chunk, so this corpus imposes a ceiling. New chunks do **not** inherit
old labels. Two recovered literal excerpts have different official owners but
the same chunk-text suffix; they are not two independent clinical discoveries.

| Query | Changed@20 | Changed@100 | Navigation@20 A -> B |
|---|---:|---:|---:|
| 128 | 16 | 47 | 4 -> 7 |
| 257 | 12 | 46 | 5 -> 7 |
| 290 | 15 | 64 | 3 -> 11 |
| 409 | 18 | 40 | 6 -> 10 |
| 446 | 11 | 46 | 2 -> 3 |
| 491 | 15 | 42 | 5 -> 4 |
| 513 | 12 | 57 | 2 -> 5 |
| 571 | 16 | 50 | 3 -> 5 |
| 634 | 6 | 39 | 1 -> 3 |
| 715 | 13 | 47 | 6 -> 10 |
| 805 | 15 | 27 | 6 -> 9 |
| 1080 | 17 | 48 | 11 -> 5 |

Navigation is a generic heading/short-link-list diagnostic, not a relevance
judgment. BODY means only "not flagged". No directory owner is globally excluded.

## Source review: gains and regressions

The queue was frozen before source judgments and before old silver labels were
loaded. A deterministic minimum-case-ID subsample per eight depth/navigation
strata was fixed before reading those passages. This assistant examined those
eight and two additional contrasts from the frozen queue. These assessments are
**AI_DIAGNOSTIC_UNBLINDED_NOT_HUMAN**, not independent blind validation. The eight
preselected cases contain one partial gain and seven uninformative ties; this
stratified diagnostic sample does not estimate a population win rate. The two
additional contrasts illustrate contextual losses, not a measured clinical error
rate. Full assessment reasons are in `assistant_diagnostic_review.csv`.

| Query/doc, fixed rank | Observed change | Provisional interpretation |
|---|---|---|
| 571 / 3056474, 4 | A anemia -> B splenic enlargement | Partial gain: asks painful enlarged spleen after malaria. B says `主要原因是脾 充血 与 单核吞噬细胞 增生` under `脾肿大`. Explains enlargement, not what to do about pain. |
| 491 / 3055870, 2 | A detailed pain differential -> B generic summary | Contextual regression: A discusses localization, activity and neural mechanisms; B says to integrate pain features and ask questions. Neither establishes diagnosis or interprets the X-ray. |
| 290 / 3078883, 5 | A nasal-drug catalogue -> B pharmacy/navigation lists | Topical information lost, but A itself does not answer whether an evening obstruction is due to a cyst. Not proof of a direct-answer loss. |
| 715 / 3019879, 17 | A upper-airway differential -> B contents | Both fail HP infection symptoms/eradication; a BODY_TO_NAV flag alone is not a clinical regression. |
| 409 / 3078954, 38 | A pharmacy links -> B unrelated drugs | NAV_TO_BODY does not imply clinical gain. |
| 128 / 3027148 and 3058523, 2 and 3 | B recovers both old literal excerpts | Window-retention recovery only; these remain old AI partials, not new direct positives. |

Exact first three chunk transitions (A -> B):

- `3056474-ba9b186f545897bbc337` -> `3056474-24cfffa292aa06f97176`
- `3055870-fbf88c86c1e9c7b3dd93` -> `3055870-e6db31e6e44d44c527f4`
- `3078883-c21e4503601aab7857f4` -> `3078883-2a2e264ded5ae265ca48`

Canonical URLs and complete unchanged Chinese text are in `paired_comparison.csv`
and the reviewer queue. This report intentionally avoids interpreting catalogue
drug claims as treatment advice.

## Blinded independent-review protocol

Seed `m42-fixed-doc-window-review-v1`: select one hash-chosen changed Top20 pair
per query; cover depth x navigation-transition strata; fill to 24 balancing query
and stratum counts. The queue covers all 12 queries and all 8 strata, including
potential regressions. It is not weighted as a random sample of all 553 changes.
Side assignment is deterministic hash-randomized, and queue order uses opaque
case IDs. Reviewers see original Vietnamese query, official document/URL and full
LEFT/RIGHT Chinese excerpts. They do not see policy, arm, rank or cosine.

Keep `review_mapping.json`, `paired_comparison.csv`, this report and the assistant
review **away from the independent reviewer** until judgments are submitted.
`blind_review_queue.csv` alone remains unlabeled and pending. Source/URL cannot
hide document type; this is a paired evidence review, not an anonymous corpus.

Queue SHA-256:
`cd01f322743eb39bcf589cd8bd9bc94fe099b9536a2e4396e0f3fbb69b9977c2`.
Mapping SHA-256:
`519c15c4067ef8e19f728eb86ac43cae35a9e77a28d0684c7a301e0d4ccb9263`.
Paired CSV SHA-256:
`cfddcbc9ec4baf357c1cb7670e8d3ca7b99236122f61f30cf3535921d0f2d94c`.

## Runtime, memory and reproducibility

Final CPU run: **1.638 s**, including **1.009 s** integrity checks. Process peak
working set **344.43 MiB**, sampled private peak **782.49 MiB**; system available
RAM sampled as low as **306.56 MiB**. Private memory is not resident RAM, and
50-ms sampling can miss transients. This is small-corpus CPU work, not evidence
that model loading is safe. No longer or concurrent heavy workload was launched.

Two completed final executions reproduced identical paired CSV, queue, mapping
and selection-protocol bytes. Wall time and memory telemetry are expected to
vary. A pre-judgment round-trip check found Windows text-mode LF conversion;
UTF-8 byte writes fixed it before final queue freeze. Superseded sampling and
serialization drafts are preserved locally, never treated as trial results.

Reproduce (requires the existing private frozen corpus/Colab artifacts):

```powershell
.\.venv\Scripts\python.exe tools\m42_window_ablation.py
.\.venv\Scripts\python.exe -m pytest tests/test_m42_window_ablation.py tests/test_m42_colab_analysis.py tests/test_multilingual_m42.py tests/test_multilingual_m42_execution.py tests/test_multilingual_m42_portable.py -q
```

**31 tests PASS**, including actual local 1,200-pair and 24-queue source-text
roundtrip, fixed-owner/rank/score behavior, tie-breaking, true cosine, immutable
outputs, blinding fields and deterministic strata selection. The source-only
test suite skips the private-artifact test when its inputs are not installed.

## What this experiment resolves and does not resolve

Representative-window choice is demonstrably consequential even with fixed
documents. First-arm selection can lose body evidence, but maximum Chinese-query
cosine is not sufficient to identify an answer-bearing window. Next issue is
**query-relation/evidence-window ranking**, with upstream document ranking also
implicated by the many unrelated fixed owners. Source absence is **UNKNOWN**:
this experiment cannot prove that the full available corpus lacks an answer.

Recommend one next step only: complete independent paired source review of the
frozen 24 cases before designing a relation-aware window-ranking comparison.
Do not promote using cosine or the 13 silver partial pairs alone. No new crawl,
model trial, production policy or leaderboard submission is authorized by this
result.
