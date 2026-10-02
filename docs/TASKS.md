# TASKS.md

## Phase 1 — Dataset Understanding

| Task | Status | Date |
|---|---|---|
| Inspect query.parquet (schema, counts, stats, Unicode) | COMPLETE | 2026-10-03 |
| Inspect links_corpus.parquet (schema, IDs, URLs, domains, extensions) | COMPLETE | 2026-10-03 |
| Analyze corpus crawling risks | COMPLETE | 2026-10-03 |
| Check ID integrity | COMPLETE | 2026-10-03 |
| Create scripts/inspect_dataset.py | COMPLETE | 2026-10-03 |
| Run script against real files | COMPLETE | 2026-10-03 |
| Create docs/DATASET_REPORT.md | COMPLETE | 2026-10-03 |

**Phase 1 status: COMPLETE**

---

## Phase 2A — Corpus Acquisition Core

| Task | Status | Date |
|---|---|---|
| Implement streaming corpus loader and fragment-safe URL handling | COMPLETE | 2026-10-03 |
| Implement pooled async HTTP fetcher with retry/backoff | COMPLETE | 2026-10-03 |
| Implement global/per-domain concurrency and delays | COMPLETE | 2026-10-03 |
| Implement cached robots policy | COMPLETE | 2026-10-03 |
| Implement transactional SQLite checkpoint/resume | COMPLETE | 2026-10-03 |
| Add safe pilot CLI and explicit `--full` gate | COMPLETE | 2026-10-03 |
| Add offline tests for URL, retry, resume, status, robots, and CLI safety | COMPLETE | 2026-10-03 |
| Run mixed-domain pilot (10 URLs) | COMPLETE | 2026-10-03 |
| Demonstrate second-run checkpoint skip (10/10 skipped) | COMPLETE | 2026-10-03 |
| Document crawler architecture and deferred work | COMPLETE | 2026-10-03 |

**Phase 2A status: COMPLETE**

---

## Phase 2B — Source Probing and Extraction Strategy

| Task | Status | Date |
|---|---|---|
| Build reproducible balanced source sampler | COMPLETE | 2026-10-03 |
| Probe 118 unique URLs across major and special categories | COMPLETE | 2026-10-03 |
| Measure response, encoding, language, DOM, and anti-bot signals | COMPLETE | 2026-10-03 |
| Compare semantic-container and density-scored extraction | COMPLETE | 2026-10-03 |
| Persist metadata-only JSON/CSV probe artifacts | COMPLETE | 2026-10-03 |
| Add fixture-based offline extraction and encoding tests | COMPLETE | 2026-10-03 |
| Document evidence and extraction policy | COMPLETE | 2026-10-03 |

**Phase 2B status: COMPLETE**

---

## Phase 2C

_Not started. Awaiting explicit approval._

---

## Phase 3 and beyond

_Not started._
