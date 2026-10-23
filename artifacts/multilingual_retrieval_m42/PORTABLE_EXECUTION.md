# M4.2 private RTX3060 execution bundle

This is an execution payload, **not** a submission ZIP. Do not upload it to the
leaderboard or publish its source evidence. It contains the existing 2,188
Chinese chunks, fixed 12 development query representations, frozen lexical
baselines, 13 partial-positive identity diagnostics, and 19 reference scores.
No held-out file, token, whole-corpus index, model weight or BEST ZIP is included.
No new query translation, crawl, extraction or preparation is required.

## Platform / setup

Extract to its own directory on the RTX3060 **Windows x64** PC. Do not extract
over a live repository. Use Python3.12 and CUDA-compatible PyTorch2.5.1+cu121.
`requirements.runtime.txt` records the actual source-machine package versions,
not stale pyproject bounds. Install those exact versions from compatible wheels;
the CUDA torch build needs a suitable official PyTorch wheel source. A CPU build
or newer unvalidated runtime is not an allowed silent substitution.

The exporter does not install packages or test remote wheel availability. If
the exact environment cannot be reproduced, stop and report that dependency
rather than changing the runtime and reusing scores. A Python virtual environment
must be created locally, not copied from the laptop.

## Commands in the extracted directory

```powershell
.\.venv\Scripts\python.exe tools\m42_portable.py verify-files
.\.venv\Scripts\python.exe tools\m42_portable.py download-models
.\.venv\Scripts\python.exe tools\m42_portable.py verify
.\.venv\Scripts\python.exe tools\m42_portable.py trial20
```

`download-models` is the sole optional network operation: download the two
existing pinned open-weight revisions, not inference or a paid provider call.
Alternatively copy snapshots, dereferencing their cache symlinks, into
`.cache/huggingface/models--ORG--NAME/snapshots/REVISION`. `verify` compares
every exported model/config/tokenizer file hash with the original snapshots.
The 20 unique texts are unchanged; a reversed-batch control re-embeds them for
equivalence. Existing resource factors and tolerances are not lowered.

Only after trial20 records `passed=true`, run the same-machine Dense stage:

```powershell
.\.venv\Scripts\python.exe tools\competition_supervisor.py --run-id m42_semantic_pc -- .\.venv\Scripts\python.exe tools\m42_portable.py dense --state artifacts\runs\m42_semantic_pc\state.json
```

To detach on Windows, launch that command using the repository's normal hidden
Start-Process supervisor mechanism. Do not launch two supervisors for this run.
The supervisor lock refuses duplicates. `semantic_execution/state.json` and
`artifacts/runs/m42_semantic_pc/state.json` expose durable progress; a failed
process stops with NEEDS_AGENT rather than retrying forever.

The Dense stage compares the trial's GPU UUID, driver/runtime/package identity;
a source-machine PASS cannot authorize target inference. It embeds2,188 chunks
plus12 Vietnamese and12 automatic Chinese query representations. Four-row
windows flush vectors before a SQLite commit containing each window checksum.
Resume verifies committed bytes and skips them. A changed signature, corrupt
file or missing checkpoint fails closed. Never copy partial vectors to a
different GPU/runtime and assume equivalence.

## Comparison and review

Outputs: `artifacts/multilingual_retrieval_m42/semantic_execution/`:

- `dense_report.json`: Top20/Top100 document retention, all13 known partial
  positions, resources and timing. No official Recall or relevance labels.
- `rankings.json`: M4 candidate/final, M4.1 candidate/final, VI-Dense,
  automatic-ZH-Dense and three-arm RRF with deterministic document dedup.
- `failure_taxonomy.json`: frozen development diagnoses plus observed positions;
  missing answer coverage stays UNKNOWN absent source review.
- `review_queue.json`: up to12 unseen evidence candidates with canonical text,
  no retrieval score/arm, PENDING_INDEPENDENT_REVIEW.

M4.1 did not save its full candidate trace. The runner replays its unchanged CPU
policy and requires exact selected-identity agreement with the frozen M4.1 final
Top20. It never labels a saved20-row list as a measured Top100 search.
Document-level recovery is not exact old evidence-window recovery: original M3
chunks were rechunked; old labels are not automatically transferred.

## Reranker / reuse stop

The19 reference scores are **not** an enabled target-host cache. Before reuse,
reconstruct exact Vietnamese query + current chunk from the frozen CSV hashes,
verify owner, revision,512-token longest-first pair tokenization, FP16,batch2,
raw-logit extraction and exact tokenizer file hashes. Replay reference pairs
with compatible historical batch context, preserving ordering and the existing
validated tolerance. Missing context, numeric drift or ordering drift means
STOP, not a widened tolerance or fabricated replacement score.

This bundle deliberately stops after Dense/review. Bounded reranking (at most96
missing pairs) needs a separate measured resource and target-reference
equivalence PASS. No remote reranker PASS or interoperability is claimed just
because source file hashes match. Source M4.2 has no completed Dense vectors to
compare across GPUs; target batch-context equivalence is explicitly different
from a proven source-vs-target vector equivalence test.

Return the execution reports, rankings and blinded queue for review. Do not
overwrite Phase27, Phase29, M4 or M4.1 outputs. No submission is generated.
