# Final status — 2026-10-27

**FROZEN. Research and competition work ended by the user's decision.**
No further crawl, downloads, inference, multilingual experiments, optimizer
jobs or submissions are authorized. Historical unfinished tasks are not a
resume instruction. Only archival/integrity/documentation maintenance remains.

## Results: separate score provenance from file integrity

- Best recorded result: **Phase27 FINAL 0.0923**, supplied by the user as an
  organizer result. No direct organizer API verification or final rank was
  recovered. Component metrics for this final package remain UNKNOWN.
- Target 0.1000 was not reached. No success or ranking is inferred from proxies.
- Frozen ZIP: `submissions/00_READY_TO_UPLOAD/p27_sdso20749_cap16.zip`.
- Recomputed SHA-256:
  `b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`.
- Local official corpus and query hashes also match the preserved manifests.
  The corpus is the 1,194,469-ID / 93-domain local competition snapshot, not an
  assertion about a current public dataset revision.
- Phase25/29/27 packages, canonical data and scientific reports are unchanged.
  Older missing/purged files cannot be reconstructed from an archive plan;
  retain their registry records and distinguish them from present files.

Evidence: `artifacts/final_freeze_20261027/critical_hashes.json`, recorded
[score history](leaderboard_history.csv), and original phase audits. Older
"organizer-confirmed" notes must be interpreted with their source/recording-date
caveats, not as fresh API checks. Existing local historical rows are preserved
without changing their numerical values or inventing new submissions.

## Implemented and limited

The engineering stack includes robots-aware resumable acquisition, archive and
SQLite ledgers, source-derived extraction/chunking, BGE-M3 Dense and BM25/FTS5,
RRF, pinned cross-encoder reranking, exact-key score reuse, bounded-memory
checkpoint recovery and deterministic competition packaging/validation.
The production interface returns document IDs and evidence, not generated
medical advice. It is not clinically validated.

VI→EN experiments processed an auxiliary PubMed corpus and ran translation,
retrieval and reranking. Candidate availability did not establish relevance or
organizer gain; known translation/admission failures remain. VI→ZH experiments
executed on a small development corpus and verified Colab vectors, but did not
establish independently confirmed general direct-answer retrieval. Term-transfer
rules do not constitute robust general translation. AI/silver labels are not
HUMAN_VERIFIED or official qrels; no complete independently judged multilingual
benchmark was achieved. Development retention is not official Recall.

Final controlled results include NO-GO for best-cosine window promotion and
NO-GO for owner-count document substitution. The latter proposed 40 changes on 38
queries, held chunks fixed and exposed complementary-subquestion displacement
risk. No new candidate was submitted or promoted based on those reviews.

## Honest CV/interview description

Suitable wording:

> Built a resource-constrained biomedical retrieval research pipeline with
> resumable acquisition, provenance-preserving chunks, Dense/BM25/RRF retrieval,
> exact-key reranking caches and deterministic output validation for 1,200
> Vietnamese queries. Recorded a user-reported organizer best FINAL of 0.0923.
> Designed controlled source/ranking ablations and documented cross-lingual
> query drift and evidence-window failures.

Do not claim state-of-the-art ranking, top-N placement, clinical correctness,
general VI/EN/ZH relevance, verified human gold labels, or success reaching 0.10.
The useful research lesson is that reproducible valid outputs and candidate
novelty did not guarantee semantic relevance; negative results are retained.

## Freeze controls and remaining reproduction requirements

No ViBioMIR Python/model worker was active at inspection. The only generic
Node process observed was an unrelated app runtime and was not stopped.
`ViBioMIR_phase12_full_pubmed` was confirmed to execute inside this workspace,
exported privately and disabled from Ready; no running task/process was killed.
`PROJECT_FREEZE.json` now prevents new repository-supervised workers, including
from that legacy scheduled launcher. Direct scripts are not technically
sandboxed; their execution remains prohibited by the project decision.

Full reproduction requires the matching private corpus/body archives, canonical
text, chunk identities, manifests, vectors/indexes, exact scores, checkpoints
and model/runtime contracts. Some production scripts and local modifications
are uncommitted; a private source tar and worktree patch preserve them. Git alone
is not a complete reproducibility archive. Dependency ranges are not a lockfile;
preserve installed versions and numerical-equivalence evidence. Reopening or
new-machine inference requires separate approval and original equivalence gates.

No exhaustive historical privacy/license/secret audit or full-project test pass
is claimed. Legacy tracked review artifacts may contain original query/evidence
text. Do not newly publish raw data or private snapshots; consider repository
visibility and redistribution permissions before public portfolio reuse.

## Preservation outcome and blockers

The dated inventory contains 58,455 entries / 181,976,775,093 logical bytes,
zero scan errors and 8 untraversed reparse links. It precedes final documentation
changes and excludes its own audit outputs. Exact paths/sizes are in the private
inventory; finalization changes are recorded separately.

Local recovery snapshots are on the same SSD, not independent backups. C: and
D: are partitions of physical disk 0. External destination and full backup
verification are **NOT CONFIRMED**. No project data was deleted, moved, reset,
pruned or cleaned. Proposed cleanup requires the separate approval gates in
[ARCHIVE_PLAN.md](ARCHIVE_PLAN.md).
