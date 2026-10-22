# M4.1 — query-text retrieval and evidence diagnostics

Completed the lightweight implementation and development comparison on
2026-10-23. Semantic inference is `DEFERRED_PHASE27_ACTIVE`. No acquisition,
model load, GPU inference, Phase27 modification, or organizer submission ran.

## What works

`src.multilingual_retrieval.m41.retrieve_by_text(query_vi, top_k=20)` executes
concept/relation understanding, Chinese lexical candidate generation,
document deduplication, ranking, compatibility admission, and canonical
evidence output. It requires neither query ID nor handwritten Chinese.
Inference reads the frozen M4 chunk corpus and terminology configurations;
it does not read benchmark judgments or the list of 12 query IDs.

This remains finite terminology transfer, not general neural translation.
Unknown residual tokens and unresolved negation scope are exposed explicitly.
Longest phrase matching with Unicode word boundaries prevents collisions such
as `ăn` inside `tăng` or `sốt` inside `sốt rét`. Question-ending `không` is
distinguished from negative exposure and clinical negation. Original text,
matched spans, population, timing, and unresolved clauses remain visible.
No dictionary-coverage flag claims that all clinical semantics were understood.

The lexical index is built once and reused across queries. It uses actual
Chinese bigrams without concatenating unrelated concepts into artificial
bigrams. Candidate generation retains entity matches before admission.
Page/list rejection reasons remain in the trace; scores and lexical matches
never assign answer-relevance labels.

## Evidence reassessment

See `M4_EVIDENCE_AUDIT.md`. Query 128/doc 1595280 is a useful differential
matching morning/on-waking sneezing; it does not fully answer awakening or
establish the patient's diagnosis. The separate unblinded provisional audit
downgrades it from label 2 to label 1. The original M4 labels are untouched.
There is **no new independently verified complete/direct Chinese answer**.

## Frozen development comparison, same corpus and Top-20 cap

| Measure | Frozen M4 | M4.1 |
|---|---:|---:|
| Available official documents / chunks | 368 / 2,188 | 368 / 2,188 |
| Development queries | 12 | 12 |
| Final query-document rows | 121 | 17 |
| Queries with final results | 10 | 4 |
| Known M3 partial pairs in full candidate diagnostics | 13/13 | 13/13 |
| Known M3 partial pairs at candidate Top-100 | 13/13 | 12/13 |
| Known M3 partial document pairs in final Top-20 | 11/13 | 6/13 |
| Query 128 / doc 1595280 final rank | 5 | 1 |
| Strict direct-answer judgment | Old AI-silver 2 claim | Revised provisional 1; no blind-verified direct |
| EN baseline / corrected-bridge positives | 24/24 + 2/2 | 24/24 + 2/2, unchanged control |

The M4.1 full trace has 3,440 query-document candidates; M4 stored only
Top-100 (1,200 rows). These trace depths differ and are **not** a candidate
recall gain. At equal candidate Top-100, the old HP drug-list pair
715/3078785 falls to rank 253. It remains in the full diagnostic trace.
Partial labels belong to their original chunks: document-level retention
does not transfer a label to a different evidence window automatically.

The final losses are exposed in `partial_retention.csv`: four catalog
documents fail article admission, two infertility documents lack the queried
age/risk relation, and the pregnancy document lacks all required entities in
its selected window. These are observable selection changes, **not proven
quality improvements**. The neonatal-bilirubin row still passes lexical
compatibility and is known to mismatch population/type; final compatibility
must not be mistaken for semantic answer admission.

## Candidate failure taxonomy

All 368 original extracted documents, including rejected catalog/navigation
content, were checked alongside the 2,188 frozen chunks. Full-document concept
hits and possible cross-chunk hits are diagnostic evidence, not answer labels.
The complete source examples and transitions are in the generated
`candidate_failure_taxonomy.json`.

| Query | Evidence and main unresolved failure |
|---|---|
| 128 | 1595280 retrieved at rank 1; differential supported, complete awakening answer unverified |
| 257 | 3078924 is a drug catalog; no established beef/intestine-spasm answer; source absence UNKNOWN |
| 290 | 1595304 describes another rhinitis condition; no reviewed cyst-causation answer; entity gap remains |
| 409 | 3016157 hypertrichosis and 3078850 drug indications answer different subparts; axillary clinical context is incompletely understood |
| 446 | 3062674 is neonatal hemolysis / unconjugated-bilirubin mismatch; lexical rank is not relevance |
| 491 | Long radiographic question loses lung-shadow/AST relationships in finite query understanding; UNKNOWN answer coverage |
| 513 | 3008933 and 3078711 have full-document multi-entity hits split across chunks; long lists do not establish a vitamin-C/thyroid interaction |
| 571 | 3003971 gives malaria–splenomegaly mechanism; management remains unanswered; 3056474 has split entity hits, not a verified cross-chunk answer |
| 634 | M4 selected 31 `/nkwk/ggths` pages about **股骨头坏死**, femoral-head necrosis, not GGT; the discovery abbreviation was misinterpreted |
| 715 | 1662742 gives HP transmission and general treatment advice, not symptoms plus an eradication regimen |
| 805 | 3094304 and 3820076 remain candidates at ranks 2/6; final admission rejects missing age/risk relation |
| 1080 | 3055183 and 3483799 cover pregnancy/labor or posture; no established answer about right chest pain at 38 weeks |

No query is classified as definitely lacking an answer solely because a
keyword gate fails. Possible cross-chunk entity matches are explicitly
unverified; there is insufficient evidence to assert extraction loss or a
hidden direct answer in those cases.

## Controlled ablation and cache recovery

The qualifier-ordering ablation holds corpus, vocabulary, candidate generation,
admission and K fixed. Only query 128 changes order: vasomotor-rhinitis doc
1595304 moves from rank 6 to rank 2. Doc 1595280 remains rank 1 with or without
qualifier ranking. This is a rank observation, without a new relevance gain.

The historical S1/S2/S3 exact cache audit examined 3,600 development rows.
Nineteen references across eight queries have identical original query text,
document ID and exact chunk text but different chunk IDs. All 19 match their
read-only SQLite score and pinned model signature. Hence M4's ID-only reuse
check understated input compatibility. `cache_score_recovery.csv` records
the aliases, original ranks and logits, text hashes and lexical positions.
Only one exact reference window enters M4.1 final output; that is the known
neonatal mismatch. These sparse scores neither establish direct relevance
nor constitute a full semantic reranking. No score was imported into the
lexical policy and no missing value was fabricated.

## Functional generalization and resources

Four new Vietnamese questions executed the full API without assigning IDs or
labels: hypertension/headache (6 returned rows), childhood asthma/morning
cough (2), kidney disease/ibuprofen (0), and Wilson disease/trientine (0).
The last case explicitly reports unsupported terms and abstains. These are
functional smoke outputs, not relevance or generalization metrics.

The final measured run took 27.78 seconds including loading, full-source
provenance audit and cache forensics. Twelve query runs, deterministic replay
and qualifier ablation together took 8.40 seconds. Four smoke queries took
0.061–0.249 seconds each after index construction. Process RSS was 86.58 MiB
before loading and 162.63 MiB after the lexical index; full diagnostic peak
working set was 358.86 MiB (earlier runs peaked at 486.39 MiB). System available
RAM was 655.94 → 622.87 MiB at the final run's sampled endpoints. These endpoint
samples do not claim continuous minimum-memory telemetry.

Phase27 was active in reranking, so no BGE-M3/reranker was loaded. The bounded
post-Phase27 trial is specified in `SEMANTIC_PILOT_PLAN.md`: twenty initial
texts, then only a measured and gated small-corpus dense/reranker experiment.
The preflight command performs no automatic launch.

## Validation and decision

36 regression tests pass. All 2,188 chunk windows pass current-official ID,
exact canonical URL, ownership, verbatim substring and offset checks.
Per-query results replay deterministically. Frozen M4 inputs/configurations
are hash-checked before/after evaluation. The EN protected file is untouched
(SHA-256 `d5e3b452a6c51e5af38b6ea607d26472686525b21fcf406277878bb78931753f`),
and its 24 baseline / 2 corrected-bridge positives were recounted from the
actual final ranking and silver files. Phase29 SHA remains
`e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04`.

**GO:** reusable query-text functionality and independent source review.
**DEFER:** bounded semantic pilot until Phase27 and its GPU children finish.
**NO-GO:** a claim of improved Chinese direct-answer retrieval, promotion of
the current final policy, further crawl, or multilingual leaderboard gain.
The observed partial-positive final loss and missing independent direct
labels prevent such claims.

## Reproduce

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_multilingual_m41.py tests\test_multilingual_m4.py tests\test_multilingual_retrieval.py -q
.\.venv\Scripts\python.exe tools\multilingual_m41.py query "Tăng huyết áp có gây đau đầu vào buổi sáng không?" --top-k 20
.\.venv\Scripts\python.exe tools\multilingual_m41.py query "Bệnh Wilson có thể dùng trientine không?" --include-candidates
.\.venv\Scripts\python.exe tools\multilingual_m41.py evaluate-development
.\.venv\Scripts\python.exe tools\multilingual_m41.py semantic-preflight
```

Generated Parquet and full diagnostic JSON remain local. The short review
queue contains six original excerpts with blank judgments, without retrieval
arm, score or rank. Original M4 silver labels and human labels are untouched.
