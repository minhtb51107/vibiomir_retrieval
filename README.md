# ViBioMIR Retrieval

**Archived / frozen on 2026-10-27.** No further acquisition, inference,
experiments or leaderboard submissions are authorized. See
[final status](docs/FINAL_STATUS.md) and the [archive plan](docs/ARCHIVE_PLAN.md).

A biomedical evidence-retrieval research project built for ViBioMIR:
Vietnamese questions, official competition document IDs, and source-verbatim
evidence windows. The production pipeline was predominantly Vietnamese.
English and Chinese retrieval were research branches, not a demonstrated
general-purpose multilingual medical search system.

## Recorded result and its provenance

Final best: **Phase27 FINAL_SCORE = 0.0923**, a
**USER_REPORTED_ORGANIZER_RESULT**, not a directly verified API response.
No final competition rank is claimed. The 0.1000 target was not reached.

Frozen package: `submissions/00_READY_TO_UPLOAD/p27_sdso20749_cap16.zip`

SHA-256:
`b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`

Its local bytes were checked during finalization. File integrity is distinct
from the provenance of the score. The folder name is historical: **nothing is
currently authorized for upload**. Older ready/pending instructions are archival.

| Recorded control | FINAL | Interpretation |
|---|---:|---|
| Phase17 Hello Bacsi | 0.0760 | Historical baseline |
| Phase21 Suckhoedoisong tranche | 0.0815 | Source expansion improved the recorded result |
| Phase23 M16 to M32 | 0.0815 | No displayed FINAL improvement |
| Phase25 Long Chau tranche | 0.0869 | Further source-expansion improvement |
| Phase29 Bach Mai | 0.0874 | Previous best |
| Phase27 remaining Suckhoedoisong | **0.0923** | Final best reported by the user |

See [score history](docs/leaderboard_history.csv) for recorded metrics and
provenance caveats. Historical corpus/evaluator versions must not be assumed
comparable. Source gains do not imply linear scaling.

## Implemented architecture

```text
Official ID/URL snapshot
  -> robots-aware acquisition + durable SQLite/body archives
  -> extraction/cleaning + canonical document/chunk ownership
  -> BGE-M3 Dense + BM25/FTS5 candidates
  -> RRF and controlled source competition
  -> BGE multilingual cross-encoder, exact-key score checkpoints
  -> document/chunk ranking + source-window packaging
  -> membership/verbatim/determinism/CRC/SHA validation
  -> historical Docs20 / Chunks40 competition output
```

The local competition snapshot contains 1,194,469 IDs across 93 domains and
1,200 Vietnamese queries. It is a **frozen local snapshot**, not a statement
about today's public Hugging Face population. Pin its checksum and membership.

The established models were `BAAI/bge-m3` and
`BAAI/bge-reranker-v2-m3`, with run-specific revisions, tokenization, precision,
batching and equivalence gates. Laptop execution used bounded memory,
streaming files and missing-only checkpoint recovery on a 4-GiB GPU.
Cached scores are not portable merely because model names match.

## What was and was not demonstrated

- **VI→VI:** source expansion and competition-compatible production outputs
  have recorded organizer evidence. This is not clinical validation.
- **VI→EN:** approximately 239,896 PubMed documents were processed in an
  auxiliary branch. Translation, lexical/Dense/hybrid retrieval and exact
  reranking were executed. English expansion/fusion did not establish a
  leaderboard gain; terminology drift and admission failures remained.
- **VI→ZH:** a small frozen corpus of 2,188 chunks / 361 official owners and
  12 development queries was evaluated with cached Colab vectors. Document
  retrieval and evidence-window selection could disagree. No independently
  confirmed general direct-answer capability was established.
- AI/silver review is **not HUMAN_VERIFIED**, organizer qrels or official Recall.
  No comprehensive trustworthy multilingual relevance benchmark was completed.
- More candidates, multiple windows, high scores, novelty and valid ZIPs did
  not by themselves demonstrate improved medical relevance.
- The final owner-alignment replication proposed 40 substitutions on 38 queries,
  kept chunks fixed and ended **NO-GO for promotion**.

Evidence: [Colab analysis](artifacts/multilingual_retrieval_m42/colab_analysis/M42_COLAB_ANALYSIS.md),
[window ablation](artifacts/multilingual_retrieval_m42/window_ablation/M42_WINDOW_ABLATION.md),
[owner-alignment report](artifacts/phase27_owner_alignment_ablation/OWNER_ALIGNMENT_REPORT.md).

## Reproduction and preservation

A Git clone alone cannot reproduce the best package. It also requires the
matching private official snapshot, canonical source content, checkpoints,
indexes/vectors, model contracts and exact-score caches. Some late-stage code
and local changes were never published; the private working-source snapshot
and worktree patch are part of the recovery plan.

`pyproject.toml` has dependency ranges, not a full historical lockfile. The
recorded environment and numerical-equivalence evidence must be retained.
No new downloads or inference are needed for archival integrity checks:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_final_freeze_archive -q
Get-FileHash submissions\00_READY_TO_UPLOAD\p27_sdso20749_cap16.zip -Algorithm SHA256
```

`PROJECT_FREEZE.json` blocks new supervised workers. It is not an OS sandbox;
direct scripts remain technically executable but are outside the frozen scope.
Only backup, verification and documentation maintenance may proceed.

Raw corpus/evidence, credentials, weights and large generated artifacts must
not be newly published. Legacy tracked reports remain unchanged; no exhaustive
historical privacy/license audit is claimed.

## Engineering lessons and documentation

The useful engineering outcomes are resumable acquisition, provenance-preserving
processing, bounded-memory retrieval/packaging, exact-key cache validation and
controlled negative results. The principal research limitation was insufficient
independent relevance supervision, especially across languages.

- [Final status and CV-ready claims](docs/FINAL_STATUS.md)
- [Backup layout, cleanup candidates and recovery](docs/ARCHIVE_PLAN.md)
- [Experiment journal](docs/EXPERIMENT_JOURNAL.md)
- [Timeline](docs/TIMELINE.md)
- [Submission schema and historical operations](docs/SUBMISSION_GENERATION.md)
- [Historical submission registry](submissions/README.md)

**No project files have been deleted as part of this freeze. External backup
verification and exact cleanup approval are still required.**
