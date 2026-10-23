# Phase27.1 bounded refinement — 2026-10-23

## New BEST: Phase27 FINAL 0.0923

Status **USER_REPORTED_ORGANIZER_RESULT**, not direct API verification.
Previous Phase29 0.0874; absolute gain0.0049, relative gain5.61% rounded.
Component metrics were not supplied and remain unknown.
`configs/current_best.json` is the current pointer. The Phase26 registry is
explicitly marked historical/superseded, retaining its original Phase25 metrics.
No ZIP, final JSON, checkpoint or exact-score cache was altered.

- Phase27 SHA: `b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`
- Phase29 SHA: `e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04`

The previous eight-query AI HOLD recommendation is superseded **as a submission
decision** by the measured organizer result. It did not predict the overall
gain. Its concrete hypoglycemia/hyperglycemia, off-topic and promotional-content
examples remain provisional diagnostics, not query-level organizer labels.

## Controlled experiment actually executed

Control: frozen Phase27. Same 39 distinct development query IDs from M1,
including the previously sampled eight; no held-out labels or tuning.
Two predefined generic ablations; no query-ID branches or medical-answer labels
drive selection. Docs20 fixed, Chunks40 fixed depth. Candidate universe is the
union of frozen final Phase27/Phase29 chunks per query. This is **not** a new
retrieval or a full deep-pool rerank. It reuses frozen output rank priority,
whose upstream exact-score provenance is in the original audits; it never
assigns invented scores to expanded text. No new model inference was needed.

R: Greedy same-owner >=80% containment of whitespace/casefold five-grams.
Keep earlier Phase27 windows; backfill from Phase29 in frozen order. If fewer
than40 pass, pad from deferred candidates and report this limitation.

Q: Reject a chunk only when >=2 promotional/navigation segments occupy >=50%
of its segmented text. Backfill identically. A clinical article with a small
footer is not globally rejected. This conservative rule was not retuned after
seeing its zero-change result.

| Metric vs Phase27 | R: overlap | Q: dominant boilerplate |
|---|---:|---:|
| Development queries processed |39|39|
| Queries changed |39|0|
| Chunk slots replaced |194|0|
| Docs changed |0|0|
| Overlap deferrals / boilerplate rejections |570|0|
| Deferred padding slots required to retain Chunks40 |316|0|
| Original Top5 objects no longer in Top5 |36|0|
| Original Top5 objects absent from final40 |4|0|
| Baseline-present silver-positive document pairs retained |12/12|12/12|
| Baseline-supported silver exact excerpts retained |13/13|13/13|

Excerpt retention requires same owner and exact review excerpt substring in a
selected expanded chunk. It is not preservation of every complete clinical
answer, nor an independent blind relevance result. Labels remain AI silver.
Not all positive review cases had baseline support; none are called negatives.

R net chunk-slot source changes: suckhoedoisong.vn -135; Medlatec+45,
LongChau+37, HelloBacsi+26, S4+10, BachMai+9, children's hospital+3,
cancer.39.net+2; five other sources net+3. These are current-official URL
owners resolved after selection. No source quota or source preference was
introduced. This is exposure, **not** a source contribution to F2.

## Interpretation / GO-NO-GO

**NO-GO for a new organizer submission from these ablations.**

- R demonstrates that overlap policy has substantial selection/rank influence,
  but it cannot find40 non-overlapping windows in this small frozen pool and
  can remove high-ranked evidence. Retention of13 known excerpts does not prove
  the other changes beneficial. Backfilling from Phase29 may restore useful
  evidence or undo genuine Phase27 gains. Independent passage review is needed.
- Q is selection-identical to the new best: spending a submission cannot test
  a meaningful retrieval change. Embedded promotion remains an extraction/window
  boundary problem, not safely solved by whole-chunk rejection here.
- No observed relevance improvement is claimed. The official Phase27 gain
  supports retaining its output rather than promoting generic dedup blindly.

Both executions produced identical candidate SHA:
`f9c7978a799e197479d9944a7179c7d0cf25706894e1f1b857cc276d130a6601`.
Measured final run:10.33s. All candidate texts are unchanged frozen owner/text
objects with current-official owners. No new submission ZIP was generated.
Development candidate JSON and detailed `report.json` are local reproducible
outputs, not replacements of Phase27.

```powershell
.\.venv\Scripts\python.exe tools\phase271_refinement.py
.\.venv\Scripts\python.exe -m pytest tests/test_phase271_refinement.py tests/test_multilingual_m42_execution.py tests/test_multilingual_m42.py tests/test_phase27_final_relevance_check.py -q
```

## M4.2 actual execution/resource result

The missing `trial20` command now exists and was invoked against the frozen
20-text manifest. Result **BLOCKED_RESOURCES before model import/load**.
No Phase27 worker or competing model process was found. A development review
server remains running; no unrelated process was stopped.

| Resource | Measured | Required | Result |
|---|---:|---:|---|
| Available physical RAM MiB |678.48|969.71 cold-start|FAIL|
| Commit headroom MiB |10,390.26|3,856.66|PASS|
| RTX3050 VRAM free MiB |3,977|1,363.95|PASS|
| Free disk GiB |34.47|20|PASS|

RAM pressure is system-wide, not an active embedding allocation: GPU used0;
Windows memory compression working set about730MiB, ChatGPT main process374MiB,
Defender128MiB, Codex116MiB, plus browser/other processes. No process terminated.

Cold-start requirement uses the existing measured profile's process-start
45.62MiB to first-batch821.39MiB RSS delta, multiplied by the unchanged1.25
safety factor:969.71MiB. The old357.11MiB warm-load threshold starts after the
runtime already consumed roughly490MiB; applying it alone before import would
ignore that cost. VRAM and commit keep existing1.25/1.10 factors. No precision,
revision, normalization, dimension, max-length or equivalence tolerance changed.

Minimum resource change: recover at least291.23MiB **additional available RAM**
at this observation, with margin for system fluctuations (prefer >=1.2GiB
available before launch). Alternatively move the exact manifest/contract to a
machine with more host RAM, preferably >=16GiB installed and >=4GiB CUDA VRAM;
the latter is advice, not a newly imposed gate. More pagefile alone does not fix
physical-memory pressure; commit already passes.

Trial load time, inference time, peak runtime RAM/VRAM and empirical equivalence
are **NOT_MEASURED**, not zero. Dense2188 and reranker were **NOT_RUN** because
the mandatory trial could not safely load. The runner has pinned local FP16
loading, twenty-text batching, telemetry and batch-context equivalence against
the existing tolerance. A future PASS still does not certify answer relevance.

```powershell
.\.venv\Scripts\python.exe tools\multilingual_m42.py trial20
```

Execution preflight details are in
`artifacts/multilingual_retrieval_m42/execution_resource_preflight.json`.
No automatic retry, cloud migration, paid API or background job was launched.
