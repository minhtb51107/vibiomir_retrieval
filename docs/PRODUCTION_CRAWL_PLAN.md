# ViBioMIR Phase 2C Production Crawl Plan

> **Status:** readiness system complete; unrestricted full crawl not started
> **Benchmark date:** 2026-10-03
> **Benchmark corpus rows:** 1,225 unique URLs
> **Raw bodies stored:** none

## 1. Scope and safety

Phase 2C validates corpus acquisition, checkpointing, telemetry, and bounded
production batches. It does not implement article extraction, chunking,
retrieval, or Phase 3. The raw Parquet remains read-only.

An unrestricted crawl requires both `--full` and `--confirm-full-crawl`.
`--full --max-new-records N` is the normal production mode and is bounded to
at most `N` newly scheduled or explicitly retried rows. Existing terminal rows
are checked by indexed SQLite lookup rather than loading millions of IDs into
memory. Production `--full` mode rejects `--store-raw-body`; that pilot-only
flag cannot accidentally put millions of BLOBs in the metadata database.

## 2. Benchmark methodology

`scripts/sample_crawl_benchmark.py` performs a seeded, deterministic Parquet
scan. The 1,225-row union is domain-interleaved and includes:

- Cnkang, 120ask, FamilyDoctor, Ask39, A-Hospital, and Zhongyibaodian;
- Sức Khỏe Cộng Đồng, Sức Khỏe & Đời Sống, Thanh Niên, and Lao Động;
- 50 Zysjonline robots-policy samples;
- 25 Long Châu Cloudflare/access-restriction samples;
- 30 `test.pmphai.com` samples;
- 150 additional HTTP-stratum samples, producing 299 HTTP URLs overall.

The benchmark used global concurrency 8, default per-domain concurrency 2,
a one-second default delay, and conservative two-second/one-worker overrides
for Cnkang and 120ask. Bodies were downloaded only to measure size and cheap
quality signals, capped at 2 MiB, and discarded after each response.

## 3. Measured results

The first run was deliberately interrupted after 66 committed rows. SQLite
reported `integrity_check = ok`. Resuming selected all 1,225 rows, skipped the
66 checkpoints, and completed the remaining 1,159 in 567.707 seconds. A second
identical run skipped 1,225/1,225, scheduled zero, and made zero HTTP requests.

| Measurement | Result |
|---|---:|
| Unique selected/completed rows | 1,225 |
| Success | 1,149 (93.80%) |
| HTTP 403 | 25 |
| Robots blocked | 50 |
| Timeout | 1 |
| Retry attempts beyond first attempt | 16 |
| Redirect hops | 278 |
| Downloaded response bytes | 113,430,674 (108.2 MiB) |
| Content type `text/html` | 1,174 |
| Missing content type because no response | 51 |
| Tiny HTML / JS-shell candidates | 50 / 50 |
| Responses truncated at 2 MiB | 0 |
| Timed resume throughput | 122.49 URLs/minute |
| Fetch-attempt rate | 1.9817 attempts/second |
| HTTP-operation rate including robots/redirects | 2.5101/second |

The 50 tiny pages are exactly the Lao Động shell pattern observed in Phase 2B.
All 25 Long Châu requests returned 403, and all 50 selected Zysjonline rows
were robots-blocked. HTTP 200 is therefore retained separately from the tiny
HTML/JS-shell signal.

Response-size distribution:

| Downloaded bytes | Rows |
|---|---:|
| 0 | 51 |
| 1–511 | 50 |
| 4,096–65,535 | 794 |
| 65,536–262,143 | 190 |
| 262,144–1,048,575 | 140 |

The timed resumed run's domain contributions show actual concurrent benchmark
throughput (count divided by the 567.707-second wall clock):

| Domain | Completed in timed run | URLs/minute contribution |
|---|---:|---:|
| `cnkang.com` | 179 | 18.92 |
| `120ask.com` | 175 | 18.49 |
| `familydoctor.com.cn` | 120 | 12.68 |
| `ask.39.net` | 120 | 12.68 |
| `a-hospital.com` | 100 | 10.57 |
| `suckhoecongdongonline.vn` | 100 | 10.57 |
| `zhongyibaodian.net` | 100 | 10.57 |
| `suckhoedoisong.vn` | 80 | 8.46 |
| `thanhnien.vn` | 60 | 6.34 |
| `zysjonline.com` | 50 blocked | 5.28 metadata rows/minute |

The durable SQLite file is 544,768 bytes, or 444.71 bytes per result row.
Transient WAL allocation is excluded from this durable per-row estimate. An
offline 10,000-row writer test completed in 1.464 seconds (6,832.6 rows/s), far
above measured network throughput; the serialized writer is not the current
bottleneck.

## 4. Production tuning decisions

- Keep global concurrency at 8 and connection pooling at 16 connections.
- Keep the one-second default domain delay and concurrency 2. The delay limits
  request starts to at most one per second per default domain.
- Keep Cnkang and 120ask at concurrency 1 with a two-second delay. Their scale
  makes these limits the dominant crawl-duration constraint, but the benchmark
  does not justify more aggressive traffic.
- Keep transient-only retries at three attempts with jittered exponential
  backoff. Permanent 4xx responses are not blindly retried.
- Apply cached robots decisions before acquiring a page-fetch delay slot, so
  a disallowed corpus row remains visible without simulating a page request.
- Classify future Long Châu rows as `ACCESS_RESTRICTED` without a page request.
  This is supported by 8/8 Phase 2B and 25/25 Phase 2C Cloudflare 403 results.
- Preserve Lao Động responses as `SUCCESS` plus `tiny_html` and
  `js_shell_candidate`; do not invoke a browser or bypass mechanism.

## 5. Capacity planning

All values below are **ESTIMATED**, not promises.

At 122.49 URLs/minute, a simple aggregate extrapolation for 4,394,718 rows is
24.92 days. Per-domain rate limits independently imply about 22.30 days for
Cnkang and 21.26 days for 120ask; measured Ask39 latency implies about 14.54
days for that domain. Because domains run concurrently, these durations are
not additive. A planning range of **25–35 days** on one process is appropriate,
subject to changing network behavior, robots policy, and server responses.

The measured SQLite density extrapolates to approximately **1.95 GB** for the
main metadata database; plan for **2–2.5 GB** including indexes, WAL headroom,
and operational variance.

Domain-weighting measured response averages by Phase 1's top-30 counts and
using the measured response average for other domains estimates approximately
**287.7 GB (268.0 GiB)** of downloaded/decompressed response data. The sample
was deliberately balanced, and source content can change, so a practical
planning range is roughly **220–360 GB**.

Storage planning assumptions—not benchmark measurements—are:

- compressed HTML at 25–40% of measured bytes: about 72–115 GB;
- later extracted text at 10–25%: about 29–72 GB;
- allow 0.5–1 TB free space for archives, extracted text, indexes, temporary
  compaction space, and at least one recoverable copy.

Machine-readable calculations and caveats are in
`artifacts/crawl_benchmark/results.json`.

## 6. Body-storage decision

Phase 2C does not store production bodies. Millions of BLOBs in the metadata
SQLite database would mix checkpoint durability with large sequential data,
and one file per document would create filesystem and backup pressure.

Before Phase 3, use append-only compressed batch archives (for example,
roughly 10,000 responses per shard) plus a small index mapping `doc_id` to
archive, member offset/key, byte length, checksum, and fetch timestamp. Write a
temporary shard, fsync/close it, atomically publish it, then transactionally
publish index rows. This preserves resumability without millions of files.
The archive implementation is intentionally deferred until storage capacity
and Phase 3 requirements are approved.

## 7. Interruption and resume behavior

Each result is committed in its own transaction under SQLite WAL mode. On
Ctrl+C, workers are cancelled, the result-writer queue is flushed, and the
database is closed. A restart checks each canonical `doc_id` through the
primary-key index. Successful and permanent terminal rows are skipped.
`--retry-failures` is required to reconsider only `TIMEOUT`, `NETWORK_ERROR`,
and `RETRY_EXHAUSTED` rows. A successful row cannot be downgraded.

## 8. Commands

Create the reproducible benchmark manifest:

```powershell
.venv\Scripts\python.exe scripts\sample_crawl_benchmark.py
```

Run or resume the bounded benchmark:

```powershell
.venv\Scripts\python.exe scripts\crawl_corpus.py `
  --full `
  --max-new-records 1225 `
  --ids-file artifacts\crawl_benchmark\selection.json `
  --output-db data\crawled\phase2c_benchmark.sqlite `
  --summary-out artifacts\crawl_benchmark\benchmark_summary.json
```

Run a bounded production batch from the canonical corpus:

```powershell
.venv\Scripts\python.exe scripts\crawl_corpus.py `
  --full `
  --max-new-records 10000 `
  --output-db data\crawled\crawl_results.sqlite `
  --summary-out artifacts\crawl_runs\batch_summary.json
```

Optional selectors include `--start-after-id`, repeatable `--domain`,
`--ids`, and `--ids-file`. Repeating the command advances past existing rows;
it does not restart from zero.

An unrestricted command is deliberately harder to invoke:

```powershell
.venv\Scripts\python.exe scripts\crawl_corpus.py `
  --full `
  --confirm-full-crawl
```

Before network work, it prints remaining rows, concurrency, default domain
delay, and output location. This unrestricted command was **not** run in
Phase 2C.

## 9. Remaining risks and deferred work

- Robots fetch failures currently follow `allow_on_fetch_error: true`; every
  such event remains visible in the robots summary and should be monitored.
- Ask39 was the slowest measured major source and generated all retry pressure.
- Encoding telemetry records HTTP declarations cheaply; HTML-meta decoding
  (including Cnkang GB2312) remains an extraction-stage concern.
- Raw-body archives, final extraction, browser rendering decisions, chunking,
  embeddings, indexes, retrieval, reranking, and submissions remain Phase 3
  or later work.
