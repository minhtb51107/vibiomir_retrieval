# Phase 8 — Full-Corpus Scaling and Production Readiness

## Scope and safety decision

Phase 8 prepares a resumable, capacity-gated route from the 4,394,718-row
official URL corpus to production retrieval. It does not claim relevance,
generate a submission, or authorize an unrestricted crawl. The current
machine cannot safely retain the projected full pipeline, so
`unrestricted_enabled` remains `false` in
`configs/production_pipeline.yaml`.

The bounded proof uses three operational scales without pretending all three
are live crawls: the prior 1,225-row readiness run, a new deterministic
5,000-URL acquisition/extraction run, and a 50,000-vector dense-index
benchmark. This provides increasing evidence while avoiding a multi-week
network job. Results are operational measurements, not retrieval evaluation.

## Current pilot and measured foundations

The proven retrieval corpus remains 1,097 successfully extracted documents,
5,643 BGE-M3-validated chunks, and 1,200 queries. Phase 2C measured 1,225
selected URLs, 93.80% fetch success, 122.49 completed URLs/minute, 444.71
bytes of durable crawl metadata per row, and 113.4 MB of response data. Its
single-machine full-crawl extrapolation is 24.92 days; the documented planning
range remains 25–35 days.

Phase 3 measured gzip storage at 23.23% of original bytes. Phase 4 measured
49.008 BGE-M3 chunks/second on the RTX 3050 and fixed embeddings at normalized
float32, 1,024 dimensions. Phase 6 measured a 52,080,640-byte SQLite BM25
index for 5,643 chunks. Phase 7 measured 120,000 candidate pairs in 3,813.47
seconds at the safe reranker batch size of 2.

## Capacity model

The updated model intentionally gives low, expected, and high values. It uses
the new bounded measurement of 89.18% usable documents and 4.368 validated
chunks per usable document for the expected case, while retaining conservative
75%/95% and 3/8 bounds.

| Projection | Low | Expected | High |
|---|---:|---:|---:|
| Usable documents | 3,296,038 | 3,919,210 | 4,174,982 |
| BGE-M3 chunks | 9,888,114 | 17,120,066 | 33,399,856 |
| Network transfer | 220.0 GB | 296.9 GB | 371.1 GB |
| Retained pipeline storage | 267.8 GB | 428.0 GB | 832.8 GB |
| Peak with atomic sparse build | 352.9 GB | 575.2 GB | 1,120.1 GB |

The expected 17.12 million chunks require about 70.1 GB each for float32
embeddings and an exact `IndexFlatIP`. Dense embedding alone is projected at
349,330 seconds (4.04 days) at Phase 4 throughput. Linear exact-search
projections range from 2.82 to 10.02 seconds/query before query embedding, so
exact full-corpus search is not automatically authorized.

The 19,478-chunk SQLite BM25 density extrapolates to 147.3 GB expected
(85.1–287.3 GB range), with an optimistic linear build time of 41.4 hours.
SQLite remains
deterministic and appropriate for the bounded proof, but its projected size,
atomic-build duplicate space, and query behavior make it unapproved for the
full corpus. A sharded/on-disk alternative needs its own benchmark.

Reranking never scans the corpus. At a fixed 100 pairs per query, all 1,200
queries retain the measured Phase 7 cost of approximately 3,813 seconds. Only
the bounded dense/sparse/RRF candidate pool is passed to the reranker.

Machine-readable inputs, formulas, byte counts, and caveats are in
`artifacts/phase8_scaling/capacity_plan.json` and
`artifacts/phase8_scaling/storage_projection.json`.

## Disk audit and stop conditions

At the final planning audit, C: had 29.60 GiB free of 224.72 GiB and D: had
39.50 GiB free of 250.00 GiB. Expected retained and transient storage exceed both
free space and the total size of D:. Full production is therefore blocked.

Every stage evaluates projected growth with 1.5x headroom, a maximum 90%
projected disk utilization, and absolute reserves. Acquisition requires 25
GiB free before and after its bounded batch, indexing requires 30 GiB, and no
stage may cross the 20 GiB stop reserve. The 5,000-row 8A and bounded 8B gates
pass; projected 8C, 8D, 8E, and therefore 8F fail. A checked-in preflight CLI
requires an exact bounded confirmation token, rejects more than 5,000 crawl
rows, keeps later stages locked, and refuses full mode even with the full
confirmation token while the config gate is false.

## Production architecture and manifests

The stage flow is:

1. **8A:** bounded acquisition, compressed archive, extraction validation.
2. **8B:** real-tokenizer chunks plus bounded exact/ANN and sparse benchmarks.
3. **8C:** production acquisition in explicit batches after storage approval.
4. **8D:** streaming extraction and BGE-M3 chunk partitions.
5. **8E:** production dense and sparse indexes after backend selection.
6. **8F:** bounded retrieval candidates and candidate-only reranking.

Canonical `doc_id` is retained throughout. Acquisition uses the existing
SQLite WAL checkpoint and append-only independent gzip members. Processing
uses deterministic half-open row ranges and 1,000-document partitions.
Production chunking loads the pinned BGE-M3 tokenizer locally, uses a 508
content-token budget (512 with special tokens), 64-token overlap, and existing
paragraph/heading/Q&A boundaries. Each partition is written to a temporary
Parquet file, atomically renamed, hashed, then committed transactionally to a
signature-bound SQLite checkpoint.

Stage manifests bind source/config hashes, stage version, row bounds, model
and tokenizer revisions, timestamps, counts, resume count, and output hashes.
An incompatible source or configuration cannot reuse a checkpoint. IDs and
ranges are deterministic; completed partitions are not emitted twice.

## Bounded scaling results

The deterministic hash sample contained 5,000 URLs from 82 domains (3,481
HTTPS and 1,519 HTTP). Acquisition completed in 2,772.88 seconds:

| Acquisition metric | Measured |
|---|---:|
| Completed | 5,000 |
| `SUCCESS` | 4,505 (90.10%) |
| Robots blocked | 283 |
| Access restricted | 92 |
| HTTP errors | 97 |
| Timeouts | 23 |
| Retries beyond first attempt | 79 |
| Redirect hops | 1,411 |
| Downloaded bytes | 337,756,938 |
| Throughput | 108.19 URLs/minute |
| Durable metadata | 411.24 bytes/row |

The body archive stored 4,505 bodies: 337,645,308 original bytes became
79,582,144 gzip bytes (23.57%). Every member passed length and SHA-256
verification. Extraction took 608.05 seconds and produced 4,459 usable
documents (89.18% of selected rows); it also retained 283 robots-blocked, 157
access-restricted, 55 extraction-failed, 42 JS-shell, and 4 empty statuses.

Real BGE-M3 chunking took 42.97 seconds and produced 19,478 unique chunks.
Independent validation found median/p95/max lengths of 107/510/510 tokens
including special tokens, zero over 512, zero invalid offsets, and zero wrong
document references. A context-sensitive subword boundary discovered during
the first pass was corrected; the invalid checkpoint was preserved but is not
an input to any benchmark.

The real 19,478-chunk BM25 index built in 169.56 seconds, occupied 167,563,264
bytes, contained 248,627 terms and 6,205,019 postings tokens, and passed both
SQLite integrity and all 19,478 row/chunk/document mapping checks.

| 50k dense index | Build | Size | Mean search/query | Top-10 row overlap with exact |
|---|---:|---:|---:|---:|
| `IndexFlatIP` | 0.10 s | 204.8 MB | 8.24 ms | 100% |
| `IndexHNSWFlat` M=32 | 8.09 s | 218.4 MB | 0.33 ms | 95.9% |
| `IndexIVFPQ` 128×32×8 | 37.55 s | 3.57 MB | 0.28 ms | 90.8% |

The 50k test expands the already measured Phase 4 BGE vectors. A fresh dense
embedding pass for the 19,478 chunks was safely aborted before checkpoint
creation because only 632 MiB host RAM was available and native model loading
exited. The CUDA/PyTorch stack and model revision were not changed, and no
repeated memory-search loop was attempted. Full projections therefore retain
the measured Phase 4 throughput of 49.008 chunks/second.

The 50,000-vector expansion is strictly an operational index benchmark based
on real bounded BGE-M3 vectors with deterministic tiny perturbations when
replication is required. Its top-10 row overlap against exact search is an
approximation diagnostic, not relevance evidence and not a reason by itself
to replace the exact baseline.

## Resume guarantees

The crawl database commits every result transactionally. The exact same
5,000-ID manifest is rerun after completion; it must schedule zero requests
and report all 5,000 rows as skipped. Archive members are checksummed, the
archive SQLite index passes integrity checking, and all stored bodies are
read back for checksum verification.

Chunk checkpoints are signature-bound to source hash, tokenizer revision,
508/64 policy, and partition size. A second identical run writes zero
documents and zero partitions. Offline tests additionally simulate an
interrupted partition sequence, resume it, reject cross-output duplicates,
verify chunk offsets, and prove deterministic IDs.

## Retention policy

**Mandatory to keep until downstream validation:** crawl metadata SQLite,
body archive plus body index/checksums, stage manifests, current cleaned
document partitions, canonical chunk partitions, current production index
metadata, and final retrieval/reranking outputs.

**Safe to delete only after downstream hashes and manifests validate:** older
superseded body/archive copies, proxy-token chunks, temporary merged chunk
files, stale embedding generations, and superseded index generations.

**Reproducible and disposable:** temporary `.tmp` files, model-download cache
when weights can be reacquired, benchmark-only synthetic ANN indexes, logs,
and recomputable query candidate dumps. Nothing is deleted automatically;
operators must validate downstream hashes and follow the documented policy.

## Commands and gates

Capacity and disk audit:

```powershell
.venv\Scripts\python.exe scripts\plan_production_scaling.py
```

Bounded preflight (does not execute a stage):

```powershell
.venv\Scripts\python.exe scripts\production_pipeline.py `
  --stage 8A --max-records 5000 `
  --confirm-bounded confirm-bounded-phase8 `
  --projected-growth-bytes 2252227570
```

An unrestricted preflight is intentionally rejected:

```powershell
.venv\Scripts\python.exe scripts\production_pipeline.py `
  --stage 8A --full --confirm-full-production confirm-full-production `
  --projected-growth-bytes 94018375042
```

Before an unrestricted run can be considered, external storage must exceed
the high transient projection plus reserve; bounded 8A/8B manifests must be
complete; dense ANN and sparse backend decisions need bounded evidence; all
stage checksums/resume checks must pass; and enabling full mode requires an
explicit reviewed configuration change. This phase performs none of those
future production runs automatically.

## Known blockers

- Local D: capacity is insufficient for even the expected retained pipeline.
- Current host-memory pressure prevented an additional BGE-M3 load; production
  embedding requires a preflight RAM check or a clean machine session.
- Full `IndexFlatIP` search/storage is operationally impractical at the
  projected expected chunk count without an evidenced index choice.
- The projected monolithic SQLite BM25 index and atomic-build peak are unsafe
  on this disk; sharding or a different backend needs a dedicated benchmark.
- Acquisition remains network- and per-domain-rate-limited to 25–35 days;
  protections and robots policy will not be bypassed.
- No trustworthy relevance evaluator exists, so Phase 8 makes no quality
  claim and does not generate a submission.
