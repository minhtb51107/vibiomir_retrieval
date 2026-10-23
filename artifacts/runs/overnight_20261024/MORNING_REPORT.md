# Overnight research morning report

Updated: 2026-10-23T19:34:55.596657+00:00
Status: **COMPLETE**
Official BEST: Phase27 0.0923 (USER_REPORTED_ORGANIZER_RESULT). Target 0.1000 is not a forecast.
Frozen BEST hash verified: `b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`. No model, inference, score reuse, submission or power-setting change.

## Actually executed
- m43: COMPLETE; runtime 3.386s; peak process RSS 163.18359375 MiB.
- production: COMPLETE; runtime 52.104s; peak process RSS 150.8359375 MiB.
- sources: COMPLETE; runtime 4.170s; peak process RSS 215.9453125 MiB.

## M4.3: fixed documents; no exposed judgments in selector
A original RRF, B best-cosine, C0 statement structure, C1 structure + entity/relation co-occurrence.
C1 explicitly falls back for unsupported negation/understanding. Relation markers are not clinical entailment.
Implementation and input checksums frozen before any selection. No rule tuning after output.
```json
{
  "20": {
    "A": {
      "changed_vs_A": 0,
      "navigation": 54
    },
    "B": {
      "changed_vs_A": 166,
      "navigation": 79
    },
    "C0": {
      "changed_vs_A": 165,
      "navigation": 29
    },
    "C1": {
      "changed_vs_A": 17,
      "navigation": 54
    },
    "C1_vs_C0_changed": 157
  },
  "100": {
    "A": {
      "changed_vs_A": 0,
      "navigation": 294
    },
    "B": {
      "changed_vs_A": 553,
      "navigation": 366
    },
    "C0": {
      "changed_vs_A": 627,
      "navigation": 161
    },
    "C1": {
      "changed_vs_A": 26,
      "navigation": 294
    },
    "C1_vs_C0_changed": 618
  }
}
```
New paired review cases: 24; UNJUDGED, HUMAN_VERIFIED=0.
NO-GO automatic promotion without new source review. No claim of direct evidence gain.

## Phase27 production: all-query measured proxies
Queries audited: 1200/1200.
Final document source slots: {"120ask.com": 16, "ask.39.net": 2, "bachmai.gov.vn": 548, "baoangiang.com.vn": 9, "baocantho.com.vn": 13, "baochinhphu.vn": 4, "baogialai.com.vn": 2, "baoquangninh.vn": 15, "baoquangtri.vn": 13, "baotayninh.vn": 1, "baothanhhoa.vn": 6, "baovinhlong.com.vn": 10, "benhviennhitrunguong.gov.vn": 319, "benhvienvietduc.org": 21, "cancer.39.net": 32, "dantri.com.vn": 24, "giadinhonline.vn": 18, "hellobacsi.com": 2400, "khoahocphothong.vn": 18, "medlatec.vn": 6810, "nhandan.vn": 16, "pharmacity.vn": 16, "phunusuckhoe.giadinhonline.vn": 23, "phuyen.baodaklak.vn": 1, "sggp.org.vn": 5, "suckhoecongdongonline.vn": 740, "suckhoedoisong.vn": 6156, "suckhoeviet.org.vn": 10, "thanhnien.vn": 17, "thaythuocvietnam.vn": 7, "tiemchunglongchau.com.vn": 6641, "tienphong.vn": 13, "vietnamnet.vn": 21, "vietnamplus.vn": 9, "vnexpress.net": 18, "vov.vn": 18, "vov2.vov.vn": 8}
Final chunk source slots: {"120ask.com": 24, "ask.39.net": 12, "bachmai.gov.vn": 869, "baoangiang.com.vn": 14, "baocantho.com.vn": 25, "baochinhphu.vn": 8, "baogialai.com.vn": 5, "baoquangninh.vn": 21, "baoquangtri.vn": 29, "baotayninh.vn": 1, "baothanhhoa.vn": 8, "baovinhlong.com.vn": 19, "benhviennhitrunguong.gov.vn": 532, "benhvienvietduc.org": 33, "cancer.39.net": 59, "cnkang.com": 1, "dantri.com.vn": 33, "giadinhonline.vn": 21, "hellobacsi.com": 3879, "khoahocphothong.vn": 32, "medlatec.vn": 10057, "nhandan.vn": 20, "pharmacity.vn": 41, "phunusuckhoe.giadinhonline.vn": 34, "phuyen.baodaklak.vn": 2, "sggp.org.vn": 4, "suckhoecongdongonline.vn": 1128, "suckhoedoisong.vn": 20506, "suckhoeviet.org.vn": 13, "thanhnien.vn": 26, "thaythuocvietnam.vn": 14, "tiemchunglongchau.com.vn": 10395, "tienphong.vn": 17, "tuoitre.vn": 1, "vietnamnet.vn": 33, "vietnamplus.vn": 13, "vnexpress.net": 31, "vov.vn": 26, "vov2.vov.vn": 13, "zydcd.com": 1}
Source-language slot metadata: {"chunks": {"unknown": 101, "vi": 47839, "zh": 60}, "documents": {"unknown": 57, "vi": 23911, "zh": 32}}
```json
{
  "doc_unique": 24000,
  "chunk_owner_unique": 29982,
  "exact_duplicate_slots": 0,
  "same_text_cross_owner_slots": 313,
  "token_jaccard_085_pairs": 19962,
  "promotional_marker_slots": 3334,
  "chunk_owners_outside_docs": 6823,
  "chunk_slots_outside_docs": 7557,
  "deep_doc_candidates": 299133,
  "deep_docs_not_selected": 275133,
  "deep_chunk_rows": 487348,
  "deep_chunk_owners": 299133
}
```
Deep-vs-final attrition does not identify relevant documents. Token Jaccard is a near-duplicate indicator, not proof.
Promotional markers are diagnostic and may occur in useful medical passages. Domain-language is not text-language verification.
Separate English caches contain candidates but cannot be judged better or safely compared by cross-runtime scores here.
Per-domain slots/query coverage and source-language totals are in production_result.json; metadata UNKNOWN stays UNKNOWN.

## Ranked source feasibility (not gold relevance)
1. thaythuocvietnam.vn: current official 2927; cached sample 30; substantive relation-prose proxy 0.1; Phase27 doc/chunk slots 7/14; acquisition NO_GO.
2. benhvienvietduc.org: current official 425; cached sample 30; substantive relation-prose proxy 0.0; Phase27 doc/chunk slots 21/33; acquisition NO_GO.
3. youmed.vn: current official 7189; cached sample 30; substantive relation-prose proxy 0.0; Phase27 doc/chunk slots 0/0; acquisition NO_GO.
4. tamanhhospital.vn: current official 4261; cached sample 30; substantive relation-prose proxy 0.0; Phase27 doc/chunk slots 0/0; acquisition NO_GO.
5. familydoctor.com.cn: current official 26819; cached sample 0; substantive relation-prose proxy None; Phase27 doc/chunk slots 0/0; acquisition NO_GO.
6. cnkang.com: current official 40046; cached sample 0; substantive relation-prose proxy None; Phase27 doc/chunk slots 0/1; acquisition NO_GO.
7. a-hospital.com: current official 4701; cached sample 0; substantive relation-prose proxy None; Phase27 doc/chunk slots 0/0; acquisition NO_GO.
8. suckhoecongdongonline.vn: current official 154503; cached sample 0; substantive relation-prose proxy None; Phase27 doc/chunk slots 740/1128; acquisition NO_GO.
All sources lack fresh attempted-union and topic-specific source-answer verification; historical remaining counts are not live.
Acquisition pilot executed: False; attempted 0.
No acquisition was authorized on source size/title/keyword proxies alone.

## Exactly one recommended next experiment
Cache-only Phase27 document/chunk-owner alignment ablation, keeping Docs20/Chunks40 and source CAP16 fixed.
Measure whether already selected chunk owners outside Docs20 improve document evidence selection; no assumed relevance gain.
This is a falsifiable candidate-selection experiment, not a predicted organizer improvement. Requires separate source review and approval before packaging/submission.

## Reproducibility and limits
Run manifest and stage_results.json bind implementation/input/output hashes. research.sqlite stores committed per-query aggregates.
Full private per-query text is never published. New review queue/mapping remain separate under m43/.
Source/body screens use only caches; held-out labels and the exposed 24 judgments are not read.
Blocked stages are not retried indefinitely. Independent safe stages still run. Re-run the same worker to resume.
Total output bytes: 2503397 at report generation.
CPU-only entrypoint: `.\.venv\Scripts\python.exe tools\overnight_20261024.py worker`.
No official precision/recall/F2 is computed from proxies. No leaderboard gain is claimed.

## Post-completion verification and decision limits

All three stage subprocesses exited successfully. Detached WMI launch wrapper
PID15048 was parented by WmiPrvSE.exe PID7840, not the interactive shell.
Supervisor PID396, worker wrapper14808/real interpreter8092 and stage subprocesses
continued after the launch shell returned. The initial run finished normally in
about62 seconds; no process was left running merely to occupy the night.
Completed-checkpoint replay succeeded without rerunning any stage. SQLite
quick_check and integrity_check both return ok with1200 committed query rows.
No errors were recorded.25 scoped tests pass. Source code/tests remain local;
only this sanitized report and aggregate statistics are eligible for publishing.

The live official scan counted1194469 IDs across93 domains. `doc_unique=24000`
and `chunk_owner_unique=29982` in the totals are SUMS of per-query unique counts,
not global corpus-unique IDs. Likewise299133 deep candidates and275133 unselected
documents are query-document pairs. The6823 outside-Docs20 owners are per-query
owner pairs;7557/48000 chunk slots are outside Docs20. Cross-owner duplicate
text slots=313; token-Jaccard>=0.85 pairs=19962. These are not19962 bad answers.
Only exact same-owner normalized duplicates have count0. Promotional-marker
slots=3334, not3334 confirmed advertisements or irrelevant passages.

| Source | Queries with Docs20 evidence | Queries with Chunks40 evidence |
|---|---:|---:|
| medlatec.vn | 1190 | 1190 |
| tiemchunglongchau.com.vn | 1184 | 1178 |
| suckhoedoisong.vn | 1156 | 1192 |
| hellobacsi.com | 875 | 922 |
| suckhoecongdongonline.vn | 465 | 581 |
| bachmai.gov.vn | 378 | 471 |

These are source-presence counts, not queries answered correctly. Source-language
metadata: Docs20 slots VI23911, explicitly ZH32, UNKNOWN57, EN0. Chunks40 slots
VI47839, explicitly ZH60, UNKNOWN101, EN0. English remains a disconnected cached
candidate arm, not proven useful admission. Chinese source presence is not proof
of direct-answer evidence. Existing English candidate caches cover1200 dense
query sets and1197 comparison query sets without any score transfer tonight.

Post hoc, AFTER policy freeze and execution, the original13 M3 AI-partial pairs
were checked for literal/normalized old-excerpt containment. No exposed24-case
review was loaded into any selector. At Top20, all four policies retain11 known
document pairs; at Top100 all retain12. Window retention:

| Policy | Literal old excerpts @20/@100 | Normalized old excerpts @20/@100 |
|---|---:|---:|
| A | 0/0 | 1/1 |
| B | 2/2 | 3/3 |
| C0 | 2/2 | 3/3 |
| C1 | 4/4 | 5/5 |

This is a useful window-retention proxy, NOT four direct answers, new labels or
official Recall. C1 changes17/240 Top20 windows versus A and157 versus C0;
Top100 counts are26 versus A and618 versus C0. All48 windows in the new24-pair
C0/C1 queue exactly match canonical text/owner/URL. No selected query-document
pair overlaps the prior exposed24-pair queue. These cases remain UNJUDGED.
There was no source-based independent assessment of new direct-answer quality.
NO-GO automatic promotion; no further tuning of the12-query policy was run.
C1 is explicitly a paragraph-level terminology/relation heuristic. It does not
implement general clinical entailment or resolve all population/temporal
qualifiers. Query text was preserved, but preserving text does not prove correct
understanding. These limitations rule out silently treating C1 as a medical
answer judge or promoting it on the retention numbers.

Peak stage-process working set was215.95MiB. The worst sampled system available
RAM was550.84MiB; minimum sampled commit headroom13120.01MiB, disk36.759GiB.
These are periodic samples and per-process peaks, not a continuous whole-tree
RAM measurement. About2.5MB of local outputs were written. No model was loaded.
Selector SHA:3e7c121f6c65d54aff98f6092826d8b1c23709268feb0f915528a75d312f01cc.
Runner SHA:8c79aeab012ab7baf987e65727ac0ade43f7b7986ab3e922751f58986b9a38ff.

The small-source screen used30 unique current-official cache records per
available source, not necessarily30 usable bodies. Youmed records were access
restricted/empty. A zero body/relationship proxy therefore does not establish
medical irrelevance. Sources absent from the single round1 cache have UNKNOWN
quality, not no content in every local cache. This was a bounded screen, not a
fresh global crawl-ledger reconciliation. No source cleared all declared gates,
especially attempted-union and topic-specific answer verification, so pilot
acquisition correctly attempted0 URLs. No access controls were probed/bypassed.

The recommended document/chunk-owner alignment experiment is a Phase27-controlled
replication/refinement, NOT a claim of a wholly new hypothesis: the existing
Phase25 p25_doc_evidence_consensus report already made21 document substitutions
with fixed chunks. Its organizer outcome is UNKNOWN here. That historical
candidate must not be silently reused or called an improvement. Against the
current Phase27, test whether a frozen multi-window corroboration rule can
recover independently reviewed useful owners from the7557 outside-Docs20 chunk
slots while retaining Docs20/Chunks40 and protecting strong baseline documents.
Use the existing implementation where compatible; freeze before review. Stop if
the proposals are redundant, off-topic, already selected or lack exact current
provenance. No score forecasts, new source acquisition or automatic ZIP follows.
