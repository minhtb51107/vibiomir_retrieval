# M3.1 evidence-quality and retrieval repair

Status: `DEVELOPMENT_VERTICAL_SLICE_COMPLETE_NO_DIRECT_ZH_EVIDENCE`

This experiment uses only the 12 frozen development queries. The held-out review was not opened. All judgments are `AI_JUDGED_M3_1` silver diagnostics, not `HUMAN_VERIFIED` labels or official recall measurements.

## Failure decomposition from M3

The 240 imported M3 judgments contain 147 directory/navigation pages, 40 topic-only non-answer chunks, 39 wrong-topic chunks, 13 partial answers and one extraction-granularity failure. Page/extraction quality was the largest observed failure class, followed by answer relevance.

## Repair

- Page classification detects drug catalogs, symptom/disease indexes, navigation-heavy pages, and repetitive list fragments using structure and content rather than a length cutoff.
- Source text is split into deterministic verbatim windows of at most 1,200 characters. Every accepted/rejected page or chunk retains its reason trace.
- The Chinese representation freezes disease/entity groups separately from relation, population, timing and intent terms. Generic `治疗` or `症状` cannot retrieve a candidate by itself.
- Admission requires the primary disease/entity group plus an answer-bearing signal. All concept groups plus an answer signal form a stricter direct gate. Missing evidence produces abstention rather than irrelevant Top-20 padding.
- English uses the protected M3 fusion: its frozen baseline prefix remains intact and only bounded bridge additions are allowed.

## Development result at the same Top-20 cap

| Measure | M3 | M3.1 |
|---|---:|---:|
| Ranking rows | 240 | 22 |
| Query units with admitted candidates | 12 | 6 |
| Abstained query units | 0 | 6 |
| AI silver label 2 | 0 | 0 |
| AI silver label 1 | 13 | 5 |
| AI silver label 0 | 227 | 17 |
| Known M3 label-1/2 query-document pairs retained | 13 | 4 |

The six abstentions are queries 257, 491, 634, 715, 805 and 1080. Only one row passed the mechanical all-concept direct gate (query 446, doc 3062674), but silver review rejected it: it concerns neonatal hemolysis and mainly unconjugated bilirubin, not the general interpretation of elevated direct bilirubin. Consequently M3.1 found **no new direct Chinese evidence**. The large reduction in label-0 rows is not claimed as a relevance improvement.

Representative retained partial evidence:

- Query 128, doc 3004366: explains sneezing in a common cold, but not the early-morning relationship.
- Query 409, doc 3016157: explains drug-induced hypertrichosis, but not the requested axillary folliculitis treatment.
- Query 571, doc 3003971: explains malaria pathology, but its bounded evidence does not answer management of painful post-malaria splenomegaly.

## Extraction versus retrieval conclusion

Extraction quality improved as a contract: 94/318 documents were classified as directory/navigation, and the usable corpus is 800 bounded verbatim chunks from 205 documents. However, after excluding catalog pages, the local current-official corpus does not contain an admitted answer-bearing chunk for half the fixed queries. The remaining candidates usually mention the primary entity without the requested relation. The dominant residual blocker is source coverage/answer-content coverage, not a hidden rank loss: the required relation is absent before ranking for the six abstained queries. Any later acquisition should be targeted only at an identified current-official page for one of those exact disease–relation pairs; this run starts no crawl.

## English and integrity gates

The final English Top-K retains 24/24 frozen baseline positive pairs and 2/2 corrected-bridge direct pairs. Current-official membership, canonical URL, verbatim chunk provenance, development-only scope and deterministic ordering pass. Frozen Phase29 remains byte-identical at SHA-256 `e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04`. Phase27 was neither read for tuning nor modified.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_multilingual_retrieval.py -q
.\.venv\Scripts\python.exe tools\multilingual_m31.py
```

Machine-readable results are in `m31_report.json`; ranking and quality traces remain generated local artifacts and are not intended for Git.
