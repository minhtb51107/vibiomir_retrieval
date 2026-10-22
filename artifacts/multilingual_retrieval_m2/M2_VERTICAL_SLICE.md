# ViBioMIR M2 development vertical slice

Status: **COMPLETE (development-only, AI-silver evaluation)**

## Contracts

- The 167 imported labels remain `AI_JUDGED`; `human_labels.csv` was not changed.
- The 10 new M2 reviews are isolated in `ai_judgments_m2.csv` as `AI_JUDGED_M2` (2 direct, 5 partial, 3 wrong).
- No held-out review page or held-out labels were opened or used.
- Frozen Phase29 remains byte-identical: `e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04`.
- These are development diagnostics, not official relevance metrics or organizer evidence.

## English translation/retrieval repair

The Helsinki query representation was replaced only for the 13 EN development queries by the previously frozen Phase18 query-only medical bridge. The PubMed FTS5 index was opened read-only with an 8 MiB page cache.

- Baseline Phase19 lexical Top-20: 260 query-document pairs for the 13 development queries.
- Corrected medical bridge returned 96 pairs for the 13 development queries.
- 88 were new relative to the M1 lexical arm; overlap was 8.
- Among the old pooled AI-silver positives, corrected FTS alone recovered 1/24 and a naïve RRF union retained 7/24 versus 9/24 for baseline lexical. Therefore **the union policy is not an improvement and must not be promoted**.

The translation repair nevertheless produced one clear new development success:

- Query 752 asks why long-term antihypertensive therapy can cause erectile dysfunction. The old five reviewed PubMed candidates were all label 0. The corrected query retrieved doc `5832210` at lexical rank 1. Its exact abstract states that ED can be “an adverse effect of antihypertensive medication” and discusses the endothelial mechanism. This is `AI_JUDGED_M2=2` and was absent from all cached M1 Top-20 arms.

Other useful but not uniformly novel evidence:

- Query 506 now preserves *cerebral infarction/stroke* rather than *heart attack* and retrieves doc `5612262`, which explicitly states that cerebral infarction is a major cause of oropharyngeal dysphagia (`AI_JUDGED_M2=1`). Baseline already had a stronger direct stroke-swallowing rehabilitation document, so this fixes translation drift but does not prove ranking gain.
- Query 928 retrieves doc `5945367` with a direct etiologic discussion of hyperferritinemia (`AI_JUDGED_M2=2`), but the baseline pool already contained direct ferritin evidence.
- Query 855 remains a failure: the returned testicular sperm-retrieval paper mentions spinal cord injury only incidentally and does not answer pregnancy timing after T12 injury (`AI_JUDGED_M2=0`).

## Chinese extraction repair (isolated from retrieval)

A deterministic quality gate was applied to cached Phase7 Top-100 rows without changing query or rank:

- 1,200 cached rows examined for the 12 ZH development queries.
- 21/32 old reviewed ZH candidates were rejected; all 21 were existing silver label 0.
- No silver-positive row was removed.
- Reasons: 17 too-short Chinese evidence rows, including 4 placeholder-only rows; 4 repeated FAQ/link-list templates.
- The filtered cached ranking can fill only 121/240 Top-20 slots. This is a quality improvement in extraction/index eligibility, not a coverage improvement.

Examples fixed:

- Four `我要咨询` placeholder chunks for query 446 are no longer index-eligible.
- Repeated `专家答疑` dated link lists are no longer treated as answer evidence.
- Doctor and hospital directories are rejected when they are navigation/profile content rather than medical answers.

## Chinese retrieval vertical slice

A separate query-only Chinese medical bridge and character-bigram FTS5 index were built over the existing 5,643 canonical Phase4 chunks:

- 1,974 chunks passed the extraction gate and were indexed; 3,669 were excluded.
- All 12 development queries produced 20 document candidates (240 pairs).
- New top results improved topic matching in a few cases but not direct answer coverage. Examples: query 128 retrieved allergic-rhinitis evidence (`AI_JUDGED_M2=1`); query 571 retrieved malaria/hepatosplenomegaly workup (`AI_JUDGED_M2=1`). Neither directly answers the full Vietnamese question.
- No new ZH candidate reviewed in this bounded slice earned label 2. The dominant remaining failure is weak/corrupted source content and lexical same-topic matching, not merely placeholder extraction.

## Controlled conclusion

- **VI→EN:** one demonstrable development retrieval improvement (query 752) plus a verified terminology repair for query 506. Across the whole pooled set, the naïve union loses known positives, so a calibrated fusion/reranking step is still required.
- **VI→ZH extraction:** measurable precision improvement—21 known wrong evidence rows removed with zero known positives removed.
- **VI→ZH retrieval:** not yet proven to improve direct answer relevance; the new lexical path still returns many same-topic/non-answer pages.

Artifacts: `m2_vertical_slice_report.json`, `en_medical_bridge_fts5_top20.parquet`, `en_baseline_plus_medical_bridge_top20.parquet`, `zh_quality_filtered_top20.parquet`, `zh_medical_bridge_fts_top20.parquet`, and the separate `ai_judgments_m2.csv`.
