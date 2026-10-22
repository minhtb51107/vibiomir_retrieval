# M4.2 bounded semantic pilot — preparation handoff

Status: **PREPARATION_COMPLETE_INFERENCE_DEFERRED**.

Phase27 still has eight active Python processes (5924, 24168, 22864, 20424,
26620, 12936, 7944, 8872); its observed stage was `PACKAGE_PASS2`. Per the
execution contract, no BGE-M3, tokenizer runtime or reranker was loaded.
Resource/equivalence benchmark and all new Dense/reranker arms are **NOT_RUN**.
There is no evidence-based GO/NO-GO on semantic usefulness yet. Execution is
NO-GO while the Phase27 process tree remains active; the experiment is deferred.

## Completed preparation

- Frozen all 12 development query texts, their automatic Chinese forms, the
  exact 2,188-row corpus order/text hashes, lexical baselines and pinned model
  contracts in `frozen_manifest.json`; baseline integrity hashes are separate.
- Fixed 20-text trial: five units selected by ascending SHA-256 of original
  query text, their five automatic Chinese forms, and ten hash-selected source
  chunks. Selected query IDs are 634, 128, 257, 409, 491. No labels select them.
  Trial SHA-256 is
  `2f99d5de39c0cd11c9dbbb3b1296d58430417d70ed3bd53c50bc49a0a30af625`.
- Reverified 19 references across eight queries using exact Vietnamese query,
  document, chunk text, original SQLite score and pinned reranker metadata.
  Saved those 19 scores in a separate owned `exact_scores.sqlite`; quick_check
  is `ok`. Resume added zero duplicate rows and does not change earlier scores.
- Hashed local tokenizer artifacts at the pinned model revision. The native
  tokenizer limit is 8,192; the scoring contract remains **512**, longest-first
  pair truncation, FP16, batch 2, raw logits. Model and tokenizer use the same
  pinned revision in the loader. Historical SQLite has no tokenizer-file hash;
  this verifies the declared revision/local artifacts, not an unavailable
  historical byte-level attestation or new empirical equivalence.
- Added pure, tested Dense document ranking, deterministic RRF union, equal
  query/budget comparison, known-partial pair accounting, and a new-evidence
  queue exporter that removes arm/rank/score. These utilities do not load models
  or turn scores into answer labels. New model execution remains deferred.
- 50 scoped regression tests passed, including identity/provenance joins,
  status separation, duplicate scores, conflicting-score rejection, manifest
  resume/corruption, tokenizer mismatch and active-Phase27 refusal.

## Review import blocker

The specified file is absent locally:
`artifacts/multilingual_retrieval_m41/review/M41_assistant_silver_review_6.csv`.
Workspace filename search found no equivalent. **Zero rows were imported.**
The user's 3 partial / 3 wrong / 0 direct totals are recorded only as a reported
summary, never reconstructed into labels. The ready importer validates the
exact six frozen query/doc/chunk identities, optional supplied evidence text,
status `AI_JUDGED_M41_ASSISTANT_UNBLINDED`, label counts and original provenance.
It writes only M4.2 `assistant_review_diagnostics.json`; old silver and human
files remain unchanged. Until the CSV exists, row-level review integration is
blocked independently of the Phase27 inference restriction.

## Deferred comparison

| Arm | Current state |
|---|---|
| M4 lexical | Frozen, 121 rows / 10 queries; partial-pair final retention 11/13 |
| M4.1 lexical | Frozen, 17 rows / 4 queries; candidate trace 13/13, Top100 12/13, final 6/13 |
| Vietnamese Dense | NOT_RUN |
| Automatic Chinese Dense | NOT_RUN |
| Lexical + Dense union | NOT_RUN |
| New exact reranker | NOT_RUN; 19 historical references already recovered separately |

No new source evidence was discovered by inference, so no new-evidence review
queue was fabricated. Existing silver judgments remain provisional and unblinded;
unjudged items are not negatives and no official Recall or leaderboard gain is
claimed. Frozen M4/M4.1 hashes and Phase29 SHA remain unchanged:
`e8e68f52fe0557e810b91b1c6ce92b336a7915a4853cabe10d583034d9935d04`.

## Reproduce preparation

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_multilingual_m42.py tests\test_multilingual_m41.py tests\test_multilingual_m4.py tests\test_multilingual_retrieval.py -q
.\.venv\Scripts\python.exe tools\multilingual_m42.py prepare
.\.venv\Scripts\python.exe tools\multilingual_m42.py import-review
.\.venv\Scripts\python.exe tools\multilingual_m42.py inference-preflight
```

The last command deliberately fails while Phase27 is active. An idle result
does not assert that RAM/VRAM/equivalence gates passed and does not launch a
worker. After Phase27 exits, follow the existing
`artifacts/multilingual_retrieval_m41/SEMANTIC_PILOT_PLAN.md`: measured 20-text
trial before resumable small-corpus embeddings, then at most 96 missing
reranker pairs after the repository's resource and strict equivalence gates.
No auto-waiter or automatic inference restart has been installed.
