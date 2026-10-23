# Phase27 final pre-submission relevance check — 2026-10-23

## Recommendation: HOLD

The frozen package is technically ready, but this development-only review does
not support spending one submission on an expectation of improved relevance.
New evidence is mixed: one useful elderly-prolapse case is accompanied by
off-topic admission; a specific baseline hyperglycemia passage is displaced by
repeated hypoglycemia material. HOLD is a risk recommendation, not proof that
Phase27 would score lower. Only organizer evaluation could establish that.
No submission or artifact repair was performed.

## Frozen artifacts verified

Read `artifacts/phase27_suckhoedoisong_remaining_20749/PHASE25_FINAL_AUDIT.json`.
Despite the helper's filename, its package is Phase27, explicitly rebased onto
Phase29 by `tools/phase27_suckhoedoisong_submission.py`; the Phase27-named final
audit copy is identical. Final audit says READY_FOR_LEADERBOARD, structural,
official/provenance and deterministic checks passed. Its package readiness
subfield still distinguishes structural validation from scientific readiness.

- Phase27: `submissions/00_READY_TO_UPLOAD/p27_sdso20749_cap16.zip`
- Bytes: **48,080,657**
- SHA-256: `b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`
- Phase29 SHA-256: `e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04`
- Phase29 organizer control: 0.0874, supplied by user; no new score inferred.

Both ZIPs were read-only inputs and hashes rechecked after review.

## Full output differences — not relevance metrics

| Measurement | Count |
|---|---:|
| Queries compared | 1,200 |
| Changed Docs20 ordering/selection | 983 |
| Changed Chunks40 ordering/selection | 1,087 |
| Changed both | 976 |
| Newly inserted document slots / displaced document slots | 2,030 / 2,030 |
| Newly inserted exact owner+text chunk slots / displaced slots | 6,561 / 6,561 |
| Doc positions differing, including reordering | 13,230 |
| Chunk positions differing, including reordering | 30,625 |

All inserted document slots and chunk objects belong to the new frozen tranche.
These are query-slot counts, not unique corpus document counts or quality gains.

## Reproducible development-only sample

Used only M1 review metadata filtered to `split=development`: 39 distinct query
IDs, 38 changed. Two strata: new-tranche document inserted, other changed output.
Take up to six per stratum in ascending SHA256(`p27-final-review-v1:{query_id}`),
before inspecting content or labels. Only two other-output cases exist, giving
eight queries: **190, 386, 446, 491, 748, 855, 929, 1192**. Six introduce new
documents. All 40 final chunks in both arms were available for comparison;
review emphasized top evidence, inserted chunks and displaced evidence.
No sealed held-out labels or held-out review pages were opened.

These are difficult multilingual-development questions, not a representative
random sample of all 1,200. No population error rate or statistical gain is
estimated. All following judgments are provisional, unblinded AI review aids.

| Query / clinical intent | Observed comparison | Provisional interpretation |
|---|---|---|
| 190: interpret echogenic 7mm nodule, report says segment V despite urinary-ultrasound context | New 560463/535139/560249 add UTI/cystitis material; baseline also has size-matched gallbladder/renal material. | No demonstrated answer gain; unresolved anatomical ambiguity. Do not infer a diagnosis. |
| 386: nasogastric tube after appendectomy | Baseline 180961 indication evidence remains top; new 512817 cancer-surgery nutrition windows enter ranks 5/7 and displace an unrelated impalement-news document. | Mixed: removes weak document, but oncology context and overlapping windows do not directly resolve appendectomy concern. |
| 446: elevated direct bilirubin | Docs unchanged; new 502032 adds hepatobiliary context, but rank38 from 489704 promotes Gia An115 services/address/telephone. Some lost tail material was nonspecific lab information. | Mixed context gain plus promotional contamination; not an established improvement. |
| 491: left-sided pain, lung shadow and neuralgia concern | New 510986/559622/511689 broaden chest-pain discussion; unrelated migraine/stroke documents displaced, but some existing chest-pain evidence also removed. | Possible complementary differential, not a direct interpretation of the lung finding; mixed. |
| 748: frequent genital herpes; suppressive acyclovir plus lysine | New 535297 becomes top chunk and mentions recurrent herpes/acyclovir400mg; general baseline 190722 remains. Multiple overlapping windows and unrelated video text enter. | Partial treatment support, not evidence answering the proposed combination/safety. No demonstrated superiority over baseline. |
| 855: pregnancy timing after D12 spine trauma | Docs unchanged; new rank39 520176 discusses carpal tunnel, replacing a weak pregnancy story. Top evidence was already off-topic. | No demonstrated gain; wrong-topic retrieval persists. |
| 929: treatment of rectal/genital prolapse at age83 | New 526933 reports elderly woman's combined rectal/uterine/bladder prolapse and multidisciplinary surgical care; new 508414 rank12 discusses an elderly man's sexual function/STI anecdotes. Baseline 540625 surgical evidence retained. | Useful complementary case plus clear wrong-topic addition; age-specific feasibility is not individualized medical clearance. |
| 1192: danger of glucose300mg/dL | New 517377 repeats a Q&A about low glucose3.4mmol/L and includes supplement promotion; lost baseline rank34 665010 explicitly discusses high glucose250–300mg/dL. Top baseline glucose evidence remains. | Concrete evidence-selection regression risk: relevant threshold passage lost while opposite-condition material is inserted repeatedly. |

Source examples are traceable in the local `sample_evidence.json` to exact
frozen ZIP owner+text objects and official URLs. The full medical-text export
is deliberately not committed. Regenerate locally with the command below.

## Retention and redundancy

- Exact first three chunk objects unchanged in **7/8** sampled queries; q748
  changes them. This limits, but does not eliminate, relevance risk.
- Existing development silver-positive document identities present in baseline:
  **12/12 retained** in Phase27 Docs20; **7/7 label2** and **5/5 label1**.
  This is document presence only, not preservation of each labeled passage.
  Other known silver positives absent in baseline remain absent; do not count
  them as Phase27 losses. No HUMAN_VERIFIED claim is made.
- Sample overlap proxy: **95→101 of 320 chunk slots** overlap an earlier chunk
  by >=80% containment of whitespace/casefold 5-grams. It is an explicit text
  redundancy proxy, not a semantic-duplicate or relevance label. Herpes windows
  rise15→19; NG-tube windows17→20. Baseline also contains substantial repetition.
- No new drug-catalog page was established in this sample. Promotional sections
  and embedded video/navigation text are nevertheless present in final chunks;
  whole-page provenance correctness does not make these answer-bearing.

## Decision scope and reproduction

The sample does **not** establish that new source results generally improve or
generally harm all queries. It establishes mixed additions, no persuasive
sample-wide improvement, and concrete preventable relevance risks. Preserve
both frozen ZIPs; do not silently edit Phase27 to fix these examples. A user
may still choose the valid ZIP as a controlled organizer experiment, but this
review's recommendation for the next scarce submission is **HOLD**.

```powershell
.\.venv\Scripts\python.exe tools\phase27_final_relevance_check.py
.\.venv\Scripts\python.exe -m unittest tests.test_phase27_final_relevance_check -v
```

No acquisition, embeddings, BM25, reranking, paid API, production worker,
submission, or ranking change was run. Held-out labels were not used.
