# Phase27 owner alignment: executed controlled replication

## Decision

**NO-GO for promoting the proposed substitutions. Stop this policy replication.**
The 1,200-query CPU experiment completed and is structurally/provenance valid,
but reuses Phase25's exact policy rather than establishing a new retrieval
method. The new control changes the proposals; it does not make agreement a
relevance label. A small source review finds useful complementary material and
real displacement risk, not enough evidence to approve B. No ZIP or submission
was created, and no historical approval was transferred.

## Historical control: what actually happened

The archived Phase25 experiment proposed52 substitutions on51 queries, then
kept21 on21 queries after Codex source review;31 proposals were not approved.
All chunks stayed fixed. Its pure policy is reusable:

- protect ranks1-15; at most2 substitutions per query;
- insert an owner outside Docs20 only when at least two selected windows contain
  >=24 words and a pair has five-gram overlap <=0.5 of the smaller set;
- candidates sorted by selected-window count, earliest chunk rank, document ID;
- only tail documents with <=1 selected window can be displaced.

Old selected sources: Medlatec8, Long Chau11, Suckhoedoisong2. Old rejected
proposals included entity, population, timing and requested-relation mismatches.
The21 approvals were AI aids, not independent labels or organizer qrels. No
separate organizer score or query-level outcome for the consensus ZIP was
recovered; neither success nor regression on organizer is established.

Archival checks: old output ZIP and JSON hashes, original Phase25 control hash,
1,200-row checkpoint/52 proposals,21 approvals contained in those proposals,
and42 verbatim owner-specific review quotes all pass. The old full pipeline was
not rerun. A seeded80-fixture test checks identical pure-selector behavior.
The old runner itself cannot be reused unchanged: it pins Phase25 paths, old
approvals, output names and an expired October21 deadline. Old outputs are not
valid substitutes for a Phase27-controlled experiment.

## Frozen A and proposed B

A is Phase27 BEST0.0923, USER_REPORTED_ORGANIZER_RESULT. Frozen ZIP:
`b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`.
The external control JSON was verified byte-for-byte against its ZIP member.
Input hashes bind canonical SQLite, current score cache, deep document ranking,
runtime configuration, current official corpus, queries and implementation.

B changes only Docs20 using the historical policy above. The full treatment is
represented by1,200 rows in `selection.sqlite` containing proposed Docs20,
changes and the unchanged Chunks40 digest. It is **B_PROPOSED_NOT_REVIEW_APPROVED**,
not a release candidate. All source evidence is retained separately on disk.

This is a new CONTROL/EVIDENCE comparison, not a new algorithm. Of40 Phase27
triplets,13 also occurred in the old52 proposals;8 of those were old approvals.
Thus27 triplets differ from the historical proposal set. None is automatically
accepted, including the8 formerly approved ones.
The other5 repeated triplets were historically not approved. This is a risk
diagnostic, not a hardcoded veto or a trusted ground-truth label.
Concrete baseline difference: query440 proposes the same candidate193995, but
the old victim was130455 and the Phase27 victim is182004. Approval against the
former cannot establish an advantage against the latter.

## Full 1,200-query results

| Measurement | Result |
|---|---:|
| Queries processed | 1,200 |
| Proposed substitutions / changed document slots | 40 |
| Queries with proposed changes | 38 |
| Queries with no substitution under the proxy | 1,162 |
| Distinct inserted / displaced IDs | 40 / 40 |
| Changed Chunks40 queries / slots / text | 0 / 0 / 0 |
| Protected ranks1-15 displaced | 0 |
| Victims with >1 selected window displaced | 0 |
| Victims with exactly1 selected window displaced | 40 |
| Changed document positions | rank20:32; rank19:6; rank18:1; rank17:1 |
| Candidate windows per inserted owner | 2:31; 3:7; 6:1; 7:1 |
| Candidate document ranks in the frozen deep pool | 21-30; missing:0 |
| Existing M1 development AI-positive document pairs present | 12 |
| Those pairs lost / newly added | 0 / 0 |
| Canonical final source spans verified | 48,000 |
| Current-official owners verified across outputs | 20,697 |

The12 development positives are only the existing M1 AI-positive rows present
in A; not a complete relevance set. They were consulted after selection and
queue freeze, never to protect IDs or tune rules. Chunks remain unchanged, so
all baseline evidence-window strings are retained regardless of label origin.
No human or held-out relevance labels were read. Split metadata alone excluded
sealed held-out units from source review; no proposed change happened to be in
those sealed units. Two proposals belong to frozen development units.

The7,557 outside-Docs20 chunk slots are not errors by definition. They narrow to
40 heuristic replacements here. Likewise1,162 nonchanging queries do not prove
that substitution is medically unjustified: only that this policy proposes none.

| Domain | Inserted slots | Displaced slots |
|---|---:|---:|
| tiemchunglongchau.com.vn | 16 | 12 |
| medlatec.vn | 10 | 17 |
| hellobacsi.com | 6 | 4 |
| suckhoedoisong.vn | 6 | 2 |
| bachmai.gov.vn | 1 | 1 |
| baoquangtri.vn | 1 | 0 |
| suckhoecongdongonline.vn | 0 | 2 |
| benhviennhitrunguong.gov.vn | 0 | 1 |
| 120ask.com | 0 | 1 |

Six inserted-source domains are aggregate diversity, not six-source corroboration
of each answer. Each proposal uses windows of ONE document. Five-gram independence
is a wording proxy and can count overlapping clinical subtopics or irrelevant
sections. No source quota, candidate depth, score, inference or corpus changed.

## Frozen source-review queue and provisional findings

`source_review_queue.json` freezes16 unique-query pairs before judgment, using
source/history strata followed by development-first deterministic hash filling.
It excludes sealed held-out units. It hides policy, ranks and scores; URLs and
original full source-derived passages are supplied. `review_mapping.json` is
separate. Queue SHA:
`269a05d25dbc404f0639255c6286214c2d7cb01c03b9d02471463ffd28f6d6eb`.

Three lexicographically smallest opaque case IDs were fixed before reading
their complete evidence arrays. Review is **AI_REVIEW_AID_UNBLINDED_HISTORY_EXPOSED**,
not independent blind or HUMAN_VERIFIED. The original16-case queue remains
UNJUDGED; findings are stored separately in `assistant_review.json`. Zero
substitutions were approved or used to change B. The other13 cases are unreviewed.

| Query / inserted -> displaced | Source-review observation | Replacement evidence |
|---|---|---|
| 765 / 124536 -> 125449 | Candidate explicitly covers neonatal phlegm management; baseline covers neonatal sneezing/cold management and warning signs. Both match parts of the question; candidate windows repeat some symptom material. | Candidate looks useful, but no proven whole-document superiority; UNCERTAIN. |
| 440 / 193995 -> 182004 | Candidate covers recurrent mouth-ulcer causes; displaced baseline explicitly addresses liver/blood testing, which the query asks about. | Complementary subquestions. One selected window does not make baseline weak; do not approve. |
| 561 / 663896 -> 666318 | Both discuss rubella during pregnancy. Neither reviewed excerpt interprets the requested chromosome-result relation or establishes causality from past infection. | No demonstrated gain; do not approve. |

This does not estimate a population win/loss rate. Protecting high ranks and
multi-window tails did not prevent risking a useful single-window baseline
subquestion in query440. The clinical strength of the other displaced tails
is UNKNOWN. Missing evidence from one excerpt is not proof it is absent from
the full document. No medical recommendations or source assertions were
independently verified by this retrieval review.

## Integrity, runtime and reproducibility

Canonical ownership/verbatim spans, official membership, exact query identities,
finite cached competitor references and input hashes pass. Cached scores only
document provenance/context; they were not used as a new relevance label or
cross-runtime comparison. Missing scores would remain missing, never synthetic.
The unchanged selected-window counts determine proposals, not old approvals.
SQLite quick_check/integrity_check return `ok`, with1,200 committed selections.

First attempt stopped at373.34MiB available during redundant historical parsing,
before Phase27 selection. Its error/logs remain preserved as a prior incident.
The correction reused archival checkpoints/hashes and source quotes instead of
reprocessing completed historical content; no resource floor was lowered.
One recovery then completed in39.52 seconds. Stage-process peak RSS125.71MiB;
private memory567.59MiB at report sampling is not resident RAM. Hard floor384MiB,
start gate512MiB, disk reserve20GiB. The completed worker exited0, and no worker
or model remains running. No crawl, GPU, model load, new scores or submission.

Reproduce on the existing local frozen artifacts:

```powershell
.\.venv\Scripts\python.exe tools\phase27_owner_alignment_ablation.py supervise
.\.venv\Scripts\python.exe -m unittest tests.test_phase27_owner_alignment_ablation -q
```

The executable uses the repository supervisor, one run lock, bounded streaming,
disk-backed proposal evidence and durable per-query checkpoints. Replays must
match the frozen signature, selections and chunk digests; no silent baseline
change is allowed. Raw evidence, queue and SQLite remain local, not committed.
Scoped tests also execute the complete1,200-query deterministic checkpoint replay,
the archival selector equivalence fixtures and all16 frozen review-pair source
checks. Tests are source-only where private inputs are absent and explicitly
skip local-artifact integration checks rather than fabricate cache data.
**10 tests PASS**, including the actual55 quoted windows in that16-pair queue.

## Next distinct bottleneck — proposed only

Stop owner-count tuning. Inspect **scored-window versus emitted-window fidelity**:
the cached packager expands source chunks after ranking (`SourceWindowExpander`
and its `expanded` cache in `tools/phase29_cached_package.py`). A candidate can
therefore be scored on a compact input but emitted with surrounding, duplicated
or promotional material. The mouth-ulcer sample visibly includes such a footer;
this is a diagnostic observation, not evidence that expansion always hurts.

One next controlled ablation: current expanded windows versus original cached
scored source windows, with Docs20, selected owners/chunk identities/order and
Chunks40 fixed where exact packaging lineage is unambiguous. Require exact
raw-to-expanded lineage, source containment and unchanged core evidence; stop
on ambiguous mappings or incompatible format rules. Review answer coverage and
context losses before any packaging. This changes the CHUNK evidence window,
not document admission or source mining. It is distinct from Phase25 consensus
and from naive duplicate removal. No new inference is needed in principle,
but feasibility and runtime must be measured first. Not executed here and no
organizer improvement is predicted.
