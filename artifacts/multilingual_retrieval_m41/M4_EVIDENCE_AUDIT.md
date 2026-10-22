# M4.1 source-based audit of the M4 claim

The original `AI_JUDGED_M4` data, configuration and reports are frozen and
unchanged. This is a new, **unblinded AI reassessment** after rereading the
Vietnamese question and canonical Chinese evidence. It is not an independent
blind experiment, human verification, or an organizer relevance label.

## Query 128 / document 1595280

Vietnamese question: “Tại sao gần sáng thường bị tỉnh giấc, liên tục bị hắt xì?”

Current-official URL: http://www.cnkang.com/dise/565/366382.html

Exact chunk: `1595280-6e5706835814bc29a624`. It is a 323-character canonical
source substring, including the document title and author. Source passage:

> 胃食管反流源性鼻炎虽然症状和过敏性鼻炎十分相似，但两者有本质区别： （1）前者没有季节性，而后者则有明显的季节性。 （2）前者在清晨和睡醒时最明显，另外，对环境的变化比较敏感，如冷热空气的交替；后者则在接触过敏原后发作。 （3）前者有时与进食有关。 （4）前者部分病人中有胃食管反流的症状，如反酸、烧心、胃胀、嗳气等。

The article proposes reflux-related rhinitis as a differential and explicitly
describes early-morning/on-waking symptoms. It does **not** establish that
sneezing causes awakening, explain the patient's sleep disturbance, or
determine whether this patient's condition is reflux, allergy, or something
else. The article's medical assertion itself is not independently corroborated
by this retrieval experiment.

| Dimension | Revised assessment |
|---|---|
| Evidence relevance | Useful differential; matches symptom and temporal context |
| Complete answer to the original question | Incomplete; awakening mechanism and patient cause remain unsupported |
| Strict provisional 0/1/2 label | **1**, revised from M4's 2 |
| Official ID / canonical URL / exact text | PASS, separately validated against the current snapshot and original document |
| M4 → M4.1 final rank | 5 → 1; ranking movement does not establish better relevance |

This narrows the earlier “direct Chinese answer” claim. No complete Chinese
answer is independently verified by this audit.

## Remaining M4 decisions

The frozen configuration contains 17 query-document decisions. Only ten are
present in the final frozen `m4_ai_judgments.csv`: one label 2, two label 1,
seven label 0. The M4 Markdown sentence reporting “1 direct, 2 partial and
12 wrong” and the “8 retained / 5 rejected” sentence are stale. The frozen
machine report has 11/13 known partial document pairs in final output.

All ten actual final decisions were rechecked separately in
`m4_evidence_reassessment.csv`. The other nine retain their earlier assessment:
the vasomotor-rhinitis and HP-transmission passages are partial; seven rows
have condition/population/intent or navigation problems. The revised ten-row
distribution is three partial and seven wrong/mismatched, zero strict direct.
The seven configuration decisions absent from the final output do not count
as final-result judgments and were not silently added to the denominator.

Static configuration labels are not independent evaluation. Retrieval code
does not load either the old decisions or this audit to select results. The
six-row `independent_review_queue.csv` leaves judgments blank and hides arm,
rank and score for a future independent source review. It includes the
uncertain morning-rhinitis result, neonatal-bilirubin mismatch, vitamin-C
interaction gap, post-malaria splenomegaly, HP treatment gap and infertility
age/risk gap. All retain original source text and identifiers.
