# M4.2: audit of the 24-case assistant paired review

## Status and scope

All 24 supplied case IDs match the frozen queue exactly, without duplicates,
missing cases or additions. All five source fields per case match exactly:
original Vietnamese query, document ID, canonical URL and complete LEFT/RIGHT
evidence. No newline/whitespace normalization was used to obtain a pass. All
48 evidence windows match frozen canonical chunks, same-owner mappings and
current-official URLs for 22 unique document owners. Frozen corpus and manifest
checksums pass. Queue, mapping, protocol, supplied review and Phase27 ZIP hashes
are unchanged before/after validation.

The review is **AI_JUDGED_M42_ASSISTANT_REPORT_EXPOSED_NOT_HUMAN_VERIFIED** for
24/24 rows. The assistant had seen the report. These are neither independent
blind judgments nor human verification, and are not organizer relevance labels.
The earlier eight-case diagnostic review, original silver labels and frozen
engineering report are preserved separately. No labels were copied into the
still-pending original reviewer queue.

Source hashes:

- Supplied review: `5b8db9f7e252b8a5860f7ec15632b9c6327b1f8c3970863e3ba7594222f12384`
- Frozen queue: `cd01f322743eb39bcf589cd8bd9bc94fe099b9536a2e4396e0f3fbb69b9977c2`
- Frozen mapping: `519c15c4067ef8e19f728eb86ac43cae35a9e77a28d0684c7a301e0d4ccb9263`
- Phase27 BEST: `b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`

## Policy-decoded results

A = saved RRF representative. B = maximum frozen Chinese-query cosine within
the same owner. Each LEFT/RIGHT preference and answer label was decoded through
`review_mapping.json`; side position was never assumed to identify a policy.
Relative preference is taken from `preference`, not reconstructed from labels.

| Review subset | B preferred | A preferred (B loses) | Tie | Total |
|---|---:|---:|---:|---:|
| All selected cases | 1 | 6 | 17 | 24 |
| Fixed rank 1-20 | 1 | 4 | 9 | 14 |
| Fixed rank 21-100 | 0 | 2 | 8 | 10 |

| Answer-support label in this review | A | B |
|---|---:|---:|
| 2: direct | 0 | 0 |
| 1: partial | 2 | 2 |
| 0: not answering | 22 | 22 |
| UNSURE | 0 | 0 |

A has partial support on query-document pairs 491/3055870 and 409/3078850;
B has partial support on 571/3056474 and 409/3078850. All four partial windows
are within Top20. These are counts in the selected review, not precision or
Recall estimates for all Top20 positions or the corpus.

Label transitions A -> B: **21 x 0->0, 1 x 0->1, 1 x 1->0, 1 x 1->1**.
Thus B's six relative losses do NOT mean six direct or partial answers lost:
one loses a partial answer, one is a preference between two partials, and four
are preferences between two label-0 passages. Relative usefulness is not the
same as satisfying the asked medical relationship.

## All seven non-tie cases

| case_id | Query/doc, rank | A->B label | Preferred | Explanation supplied by review |
|---|---|---|---|---|
| `2fcd99792c65c94c5dfb` | 571/3056474, 4 | 0->1 | B | B explains malaria-associated splenic enlargement rather than anemia. No management of painful splenomegaly. |
| `bc79536f02f7263e60b5` | 491/3055870, 2 | 1->0 | A | Detailed localization/provocation/neural pain differential becomes a generic summary. Neither explains the individual X-ray. |
| `c3c0aae25bbf537411b1` | 409/3078850, 7 | 1->1 | A | Both drug-catalogue windows partially address the conditional folliculitis-treatment subquestion; A is judged more focused. Neither explains female coarse hair. |
| `8a44fae4a975588fefbf` | 290/3078883, 5 | 0->0 | A | Nasal-treatment descriptions become pharmacy navigation; A still does not establish cyst causation or its treatment. |
| `9f35fa84924b765c3fc5` | 257/3078757, 42 | 0->0 | A | A mentions intestinal spasm as a drug indication; B is a menu. Neither answers whether beef can be eaten. |
| `e0525dc71a88192fdd31` | 805/3008933, 28 | 0->0 | A | A mentions medicines for low sperm/infertility; B is endocrine navigation. Neither answers age and risk factors. |
| `f9df17a06dcb4592f003` | 128/3101007, 7 | 0->0 | A | A includes sleep-apnea symptoms/insomnia; B is airway-obstruction etiology. Neither links early-morning awakening to sneezing. |

The main partial gain and loss are independently traceable to exact source
windows (not independently verified clinical relevance):

- Gain: `3056474-ba9b186f545897bbc337` -> `3056474-24cfffa292aa06f97176`.
  B states `脾肿大` and `主要原因是脾 充血 与 单核吞噬细胞 增生`.
- Partial loss: `3055870-fbf88c86c1e9c7b3dd93` -> `3055870-e6db31e6e44d44c527f4`.
  A discusses location and provoking/relieving factors; B reduces this to general
  advice to consider pain characteristics together.
- Partial preserved but less preferred:
  `3078850-b9f692daf574607294c2` -> `3078850-ccabd31cb9ca3f53e5fa`.
  Both contain folliculitis drug indications, not a diagnosis or individualized
  treatment recommendation.

The other 17 cases tie, all at 0/0. Examples include HP infection versus ear or
upper-airway material; direct bilirubin versus cold/diabetes material; GGT versus
menopause/hematemesis; hypothyroidism with oral vitamin C versus hyperthyroidism
and injected-vitamin indications. Those are entity/relationship failures, not
errors repaired simply by retrieving a longer chunk.

## Interpretation

**NO-GO for promoting B.** The supplied review reinforces that maximum cosine
cannot reliably select an answer window even after a document was retrieved.
It also demonstrates that a relative preference can favor an entirely
non-answering catalogue passage over navigation. There is no new direct-answer
label 2 and no net increase in partial count within this sample.

The sample was selected by depth/navigation strata, not uniformly sampled, and
contains repeated query units/owners. Report exposure further limits its
independence. Do not extrapolate a 6:1 failure ratio to all windows or use these
24 labels to choose rules/weights, hardcode IDs or claim benchmark improvement.
Candidate/document ranking also remains problematic: a same-owner selector
cannot turn an HP-unrelated ear page into an HP answer. Missing source coverage
is still UNKNOWN, not established by these reviews.

## Proposed next controlled ablation (NOT EXECUTED)

**Hypothesis:** a query-entity plus requested-relation window selector can retain
substantive answer-supporting statements better than either first-arm selection
or global cosine alone, without changing document retrieval.

Keep the same 12 development queries, frozen Chinese representations, all
2,188 chunks/vectors, document ranks/RRF scores and Top20/Top100 budgets.
No translation/lexicon tuning, candidate expansion, new inference or global
directory-document exclusion in this experiment.

Prepare one C policy with two predeclared components:

1. **Statement structure:** distinguish asserted prose from menus, repeated link
   lists and generic concluding summaries at paragraph level. Keep all source
   windows as candidates; a catalogue with a genuine medical indication remains
   eligible. Headings/length alone are insufficient. Do not rewrite/concatenate
   evidence or discard a document globally.
2. **Entity-relation support:** use only frozen automatic entity/relation groups
   and query-text-derived context. Score local asserted entity-relation spans,
   not words appearing independently across menus. Represent requested causal,
   symptom, management, risk and interaction relations separately; preserve
   polarity, temporality, population and conditional subquestions. Mark unknown
   or unsupported understanding explicitly. Hyperthyroidism is not evidence for
   hypothyroidism; treatment indications are not diet/age-risk evidence. These
   are generic semantic distinctions, not query-ID exceptions.

Use deterministic lexicographic criteria (no learned or hand-fit weights):
explicit incompatible-context warning, supported entity-relation/subquestion
count, asserted-prose support, then existing cosine and chunk-ID tie-breaking.
Do not call term/sentence co-occurrence a verified answer. When understanding is
ambiguous or no supported relation can be identified, retain A and emit a reason
trace rather than force the least-bad B window or silently remove the owner.
The selector must never load the review CSV or any silver-label file.

To attribute an effect specifically to relations, compare C against the same C
implementation with **only the relation component disabled**. Keep A and B as
frozen references. This is one component ablation, not multiple production
submissions. Freeze the implementation/config before viewing new judgments.

Evaluate all 1,200 fixed document positions and report exact window changes,
navigation diagnostics, known-silver literal/normalized retention without label
transfer, fallback counts and explicit query-understanding failures. Freeze a
new deterministic, balanced paired source-review queue before judgment, prefer
previously unreviewed windows where possible, and report overlap with the 24
exposed cases. No tuning on held-out. Source review of new windows must assess
direct/partial support and contradiction independently of rank/cosine; secure
an independent reviewer if available. Otherwise label results unblinded AI aids.

Predeclared decision: GO only for further validation if independently reviewed
new evidence shows actual answer-support gain over the relation-disabled
control without confirmed entity/polarity mismatches or partial-to-zero losses;
otherwise NO-GO/INCONCLUSIVE. Structural cleanup or improved silver retention
alone is insufficient. No automatic production promotion follows any result.

Expected compute is bounded CPU work on the existing small corpus; the prior
ablation measured 1.638 seconds but this selector has not been benchmarked, so
its runtime/memory remain UNKNOWN. Benchmark before execution and preserve all
resource gates. Nothing proposed here has been run.

## Reproduction and checks

```powershell
.\.venv\Scripts\python.exe tools\analyze_m42_paired_review.py
.\.venv\Scripts\python.exe -m pytest tests/test_m42_paired_review.py tests/test_m42_window_ablation.py tests/test_m42_colab_analysis.py -q
```

20 scoped tests PASS, including the real 24-row/48-window local audit, reverse
LEFT/RIGHT decoding, equal-label relative preferences, missing/duplicate cases,
verbatim/newline corruption, ownership/URL errors and rejection of an unexpected
HUMAN_VERIFIED status. The audit prints structured per-case results without
editing any source CSV, queue, mapping or frozen report. No inference, crawl,
retrieval rerun, submission or Phase27/Phase29 modification occurred.
