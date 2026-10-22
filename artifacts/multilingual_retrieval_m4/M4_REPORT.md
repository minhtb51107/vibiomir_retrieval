# M4 — prove Chinese answer retrieval

Status: `M4_VERTICAL_SLICE_COMPLETE_DIRECT_ZH_EVIDENCE`. This run used only the 12 frozen development units; held-out was not opened. Labels are `AI_JUDGED_M4` silver diagnostics, never `HUMAN_VERIFIED` or official recall.

## Result

M4 acquired 50/50 bounded current-official `cnkang.com` pages at 29.4 URL/min. The retrieval corpus contains 368 documents and 2188 exact source-derived chunks. M4 selected 121 rows for 10/12 development queries.

The newly acquired rows were independently recorded as 1 direct, 2 partial and 12 wrong/mismatched. M3/M3.1 had zero direct Chinese answers. This is one silver-positive query unit, not a relevance metric.

- query `128`, doc `1595280`, chunk `1595280-6e5706835814bc29a624`, candidate/final rank `5/5`
- URL: http://www.cnkang.com/dise/565/366382.html
- Verbatim evidence: > 鼻炎不一定是过敏，有可能是胃食管反流病  郑州大学一附院放射介入科崔强  很多鼻炎患者每天都被喷嚏、流涕和鼻塞困扰着，在医院无一例外都被诊断为过敏性鼻炎，开了很多抗过敏药物，吃着管用，但一停药仍是喷嚏、流涕不断。随着医学的发展。本人临床研究发现，很多鼻炎并不是真正的过敏，而是胃食管反流引起的，我们称之为胃食管反流源性鼻炎。  胃食管反流源性鼻炎虽然症状和过敏性鼻炎十分相似，但两者有本质区别： （1）前者没有季节性，而后者则有明显的季节性。 （2）前者在清晨和睡醒时最明显，另外，对环境的变化比较敏感，如冷热空气的交替；后者则在接触过敏原后发作。 （3）前者有时与进食有关。 （4）前者部分病人中有胃食管反流的症状，如反酸、烧心、胃胀、嗳气等。

The passage directly links early-morning/on-waking sneezing and rhinorrhea to a clinically meaningful rhinitis differential. Its page is current-official, and the exact evidence is a substring of the canonical extracted document.

## Candidate recall versus admission

All 13 previously known M3 partial/direct query-document pairs remain in the M4 candidate trace. Eight enter the final Top-20 and five remain visible but fail final relation admission; none silently disappear at candidate generation. This preserves failure evidence while keeping abstention separate from retrieval.

## Automatic VI→ZH path

The production function accepts raw Vietnamese text without `query_id` or a handwritten `query_zh`. It performs deterministic medical concept/relation/context transfer, explicitly detects ambiguity, then combines Chinese lexical retrieval with any compatible cached multilingual scores. The frozen 12 translations are not used as the production input.

This run found zero exact compatible cached semantic rows after source-verbatim rechunking. It did not approximate those scores or launch new BGE-M3/reranker inference while Phase27 was active; the demonstrated positive therefore comes from the automatic Chinese lexical/relation path.

The unrelated smoke query about an older diabetic patient and hypoglycemia also executes through this query-text-only path; it is a functional smoke, not benchmark evidence.

## English and integrity

The protected English control remains 24/24 baseline-positive pairs and 2/2 corrected-bridge direct pairs at final Top-K. Phase29 remains byte-identical at `e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04`. No GPU inference ran, Phase27 was not modified, and no held-out labels were read.

## Remaining failures and recommendation

Eleven of 12 fixed query units still lack AI-silver direct evidence. Term co-occurrence is recorded only as coverage diagnostics and remains `UNKNOWN`; it is not promoted to evidence. M4 therefore meets the primary vertical-slice objective but not the stretch target of three direct units.

Recommendation: **GO for the demonstrated architecture, NO-GO for further untargeted acquisition.** Any next crawl should require a current-official URL neighborhood tied to a specific unresolved disease–relation gap.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_multilingual_m4.py tests\test_multilingual_retrieval.py -q
.\.venv\Scripts\python.exe tools\multilingual_m4.py manifest
.\.venv\Scripts\python.exe tools\multilingual_m4.py run
.\.venv\Scripts\python.exe tools\multilingual_m4.py understand "Không uống rượu nhưng GGT tăng do đâu?"
```
