# M4.2 resume — actual resource block and portable execution

2026-10-23; inspected main base4221810. No Phase27/Phase27.1 work was started.
Best remains user-reported Phase27 FINAL0.0923, not a new retrieval result.

## One strict local preflight: FAIL

| Resource | Actual | Required | Gate |
|---|---:|---:|---|
| Available RAM MiB |648.41|969.71 cold-start|FAIL|
| Available commit MiB |13,655.70|3,856.66|PASS|
| GPU free MiB |3,977|1,363.95|PASS|
| Disk free GiB |36.66|20|PASS|

Phase27 PIDs[], competitor-model PIDs[], model loaded=false. The nearby OS
counter sample had626MiB available, confirming fluctuating host pressure.
No unrelated process was killed. No second unchanged preflight/model-load
attempt was made. At the measured instant, >=321.30MiB additional available RAM
would be necessary merely to reach the unchanged gate, before fluctuation margin.
No model, precision, revision, batch size, max-length or tolerance changed.

Trial20 is **NOT_RUN_RESOURCE_GATE**, not a numeric-equivalence failure.
Model-load/inference time, peak model RAM/VRAM and empirical equivalence are
**NOT_MEASURED**. Dense and reranker are **NOT_RUN**. No semantic gain or
leaderboard improvement is claimed. Exact telemetry is preserved separately in
`resume_resource_20261023.json`; the pre-existing dirty telemetry file was not
overwritten.

## Working portable payload, not just an execution proposal

Private archive:
`artifacts/multilingual_retrieval_m42/portable/m42_rtx3060_v2.zip`

- Bytes:2,290,171
- SHA256: `278add12222bcc7e5907719f881a76d0b1a8059cb4a2d63a6c5784cc702acf19`
- The v2 filename supersedes the local draft archive; use only v2.
-36 checksummed source/config/development files plus manifest and exact runtime
  requirements. No weights, secrets, held-out files, large index or submission ZIP.
- Actual frozen20-text hash remains
  `2f99d5de39c0cd11c9dbbb3b1296d58430417d70ed3bd53c50bc49a0a30af625`.
- ZIP CRC and extracted file-hash verification PASS. Extracted trial imports
  and frozen manifest validation PASS without importing torch. Exact nineteen
  CSV reference query-text/chunk-text hashes and document ownership were checked
  against the exported frozen data, all PASS. Numerical replay is **not** thereby
  authorized or verified.
- Model/tokenizer snapshot checksums are included without the weights; matching
  snapshots can be copied or downloaded on the RTX3060 PC. Target package builds,
  revisions and all model/tokenizer file hashes must pass before execution.

`PORTABLE_EXECUTION.md` contains the Windows commands. The remote PC is not
connected to this session, so no target inference or detached target worker is
claimed. The archive captures the explicitly checksummed working-tree
implementation over base4221810, not a claim that its code was already in that
base commit. The ZIP is local/private and is not committed to Git.

## Dense execution implemented for when the target trial passes

The executable Dense stage requires a trial20 PASS with the same frozen input
and target GPU UUID/driver/runtime identity. It encodes2,188 chunks+12 Vietnamese
queries+12 automatic Chinese representations, using the existing pinned FP16
BGE-M3 contract. A vector memmap and small SQLite four-row window checksums
provide missing-only resume, flush-before-commit and corruption detection.

It produces exact, deterministic, document-deduplicated Top20/Top100 comparisons
for VI-Dense, ZH-Dense, RRF, M4 and M4.1 lexical candidates. M4/M4.1 final
admission results are separate arms. M4.1's full candidate trace was not saved;
the target CPU runner replays the unchanged policy and requires frozen final
identity agreement before treating that replay as a baseline. A saved Top20 is
not falsely reported as a measured Top100.

Reports distinguish document retention from exact source-window retention and
export up to12 unlabelled new evidence candidates. A known document's new chunk
does not inherit an old silver relevance label. No new labels have been made.
The old M4.1 failure taxonomy is preserved, not replaced by lexical/dense-score
claims about absent answers.

The19 historical reranker scores remain reference-only on the target host:
reuse requires exact input/tokenizer/contract verification and numerical replay
including historical batch-context and ordering checks. Missing context or
numerical drift means STOP. Bounded missing-only reranking remains conditional
on its own resource and equivalence gates; it was not run here.

## Existing baseline diagnostics, not new M4.2 measurements

All13 original M3 partial-positive document pairs remain in the frozen M4.1
candidate trace:12 are Top100 and6 are final Top20. Their prior ranks are:

|Query|Document|M4.1 candidate rank|M4.1 final rank|
|---|---|---:|---:|
|128|3004366|10|10|
|128|3019879|7|7|
|128|3019895|8|8|
|128|3027148|3|3|
|128|3058523|4|4|
|409|3078850|6|not admitted: page type|
|571|3003971|1|1|
|715|3078785|253|not admitted: entity/page type|
|715|3078801|9|not admitted: page type|
|715|3078924|8|not admitted: page type|
|805|3094304|2|not admitted: relation|
|805|3820076|6|not admitted: relation|
|1080|3055183|1|not admitted: entity|

Source: unchanged `artifacts/multilingual_retrieval_m41/partial_retention.csv`.
All are AI-silver partials. Their exact old windows were rechunked: exact-window
retention is UNVERIFIED, not13/13. Every M4.2 Dense/reranker rank is still UNKNOWN.

The12 frozen failure diagnoses remain:128 differential-only;257 unconfirmed
answer coverage;290 required entity mismatch;409 partial/understanding limit;
446 bilirubin type/population;491 query understanding;513 multi-entity answer
unconfirmed;571 mechanism versus management;634 wrong discovery neighborhood;
715 transmission versus eradication;805 risk versus age relation;1080 population/
pain location. These are inherited provisional diagnoses, not new independent
source judgments. No query is declared to lack an answer from a failed search.

## Decision / exactly one next experiment

**GO to execute this frozen Dense/RRF experiment on the RTX3060 PC (direction A
as the next test); NO-GO to claim multilingual improvement or make a submission.**
This is not a finding that A has already recovered useful evidence. The decisive
semantic experiment has not run. Existing partial evidence and admission losses
justify testing the same corpus first; they do not justify choosing B as a
proven direct-answer fix or C as a proven missing-source diagnosis. Do not crawl.

Accept A as a quality improvement only if source review confirms useful recovered
windows and measures losses under the same budgets. Otherwise return the actual
Dense outputs before choosing reranker/admission or targeted coverage changes.
No source expansion, new production variant, BEST mutation or automatic
submission occurred. Scoped unit tests cover runtime binding, checkpoint resume,
corruption rejection, document dedup and bundle tamper/path rejection.
