# ViBioMIR Phase 2A Crawler Architecture

> **Status:** COMPLETE
> **Scope:** corpus acquisition foundation and a 10-URL pilot only
> **Raw corpus:** read-only; `id` remains the canonical `doc_id`

Phase 2C production controls and measured capacity decisions are documented in
`docs/PRODUCTION_CRAWL_PLAN.md`. The Phase 2A architecture remains the base.

## Data flow

1. `corpus_loader.py` streams `id` and `url` columns from the raw Parquet in
   bounded Arrow batches. Optional ID and domain selectors are applied before
   records enter the scheduler.
2. `url_utils.py` preserves the original URL and query string. It removes only
   the fragment from the HTTP request URL. Rows are never merged.
3. `scheduler.py` feeds a bounded worker queue. Global and per-domain
   semaphores limit concurrency, while a per-domain start lock enforces the
   configured minimum delay between request starts.
4. `robots.py` obtains at most one `robots.txt` per normalized hostname during
   a process run, caches the parsed policy, and emits `ROBOTS_BLOCKED` results
   rather than silently dropping disallowed rows.
5. `fetcher.py` uses one pooled `httpx.AsyncClient`, follows redirects, applies
   independent connect/read/write/pool timeouts, and retries only transient
   failures with exponential backoff and jitter.
6. `writer.py` serializes result writes through a bounded async queue.
7. `checkpoint.py` commits every result incrementally to SQLite. A restart
   reads completed `doc_id` values and skips them before any HTTP work.

The default CLI is pilot-safe: without `--full`, it applies the configured
default limit (20) and rejects limits above 100. `--full` is the only way to
request an unrestricted run.

## URL handling

For each corpus row:

- `doc_id`: unchanged `links_corpus.parquet.id`;
- `original_url`: exact corpus URL, including query and fragment;
- `fetch_url`: exact URL with only the fragment removed;
- `final_url`: URL after HTTP redirects.

The conventional `www.` hostname prefix is removed only from the scheduler and
robots cache key so `www.cnkang.com` receives the `cnkang.com` policy. It is
not removed from `fetch_url`, and other subdomains are not collapsed.

## Configuration and concurrency

Operational settings live in `configs/crawler.yaml`:

- global concurrency and default per-domain concurrency;
- minimum delay between starts for each domain;
- conservative overrides for concentrated domains;
- connection-pool sizes and request timeouts;
- user agent, retry/backoff parameters, and robots behavior;
- SQLite output path, body-storage switch, and body-size cap;
- pilot default and maximum limits.

The controller acquires both a global semaphore and the selected domain's
semaphore. Domain overrides are matched case-insensitively after the limited
`www.` normalization described above. There is no IP rotation or anti-bot
bypass behavior.

## Retry logic

Retries are limited to:

- HTTP `408`, `425`, `429`, `500`, `502`, `503`, and `504`;
- `httpx` timeout exceptions;
- `httpx` network/request exceptions.

Backoff is `min(base * 2^(attempt-1), maximum) + random jitter`. Permanent
client errors such as `400`, `401`, `403`, and `404` are recorded immediately
as `HTTP_ERROR`. Exhausted transient HTTP responses become
`RETRY_EXHAUSTED`; exhausted timeout and network exceptions remain `TIMEOUT`
and `NETWORK_ERROR`, preserving failure type.

## Robots policy

Robots handling is deliberately small and modular:

- each normalized hostname is fetched at most once per process run;
- successful files are parsed with Python's `RobotFileParser`;
- `401` and `403` robots responses conservatively disallow crawling;
- other `4xx` responses mean no robots file was found;
- network and `5xx` behavior follows `allow_on_fetch_error` in configuration;
- explicit disallows are persisted as `ROBOTS_BLOCKED` rows.

The pilot configuration allows on robots fetch errors so transient robots
availability does not silently discard documents. This choice is explicit and
can be changed without modifying crawler code.

## SQLite persistence and resume

The database uses WAL mode, transactions, and `doc_id INTEGER PRIMARY KEY`.
The `crawl_results` table contains:

| Column | Purpose |
|---|---|
| `doc_id` | Canonical unique corpus ID |
| `original_url` | Unmodified corpus URL |
| `fetch_url` | Fragment-free request URL |
| `final_url` | Redirect-resolved URL |
| `status` | Explicit crawler status |
| `http_status` | Final HTTP status when available |
| `content_type` | Media type without parameters |
| `encoding` | HTTP client's selected encoding |
| `content_length` | Response header value, or stored body length when available |
| `fetched_at` | UTC attempt timestamp |
| `attempt_count` | Attempts made during this run |
| `elapsed_ms` | Total fetch/retry elapsed time |
| `error_type`, `error_message` | Structured failure details |
| `raw_body` | Optional capped response bytes; disabled by default |
| `updated_at` | SQLite update timestamp |

Upserts preserve one row per `doc_id`. A successful row cannot be downgraded
to a failure. By default every existing terminal row is skipped. With
`--retry-failures`, only `TIMEOUT`, `NETWORK_ERROR`, and `RETRY_EXHAUSTED` rows
are eligible for another attempt; successful and permanent outcomes remain
skipped.

## Status meanings

| Status | Meaning |
|---|---|
| `PENDING` | In-memory initial state; not a completed fetch |
| `SUCCESS` | Supported response with a `2xx` status |
| `HTTP_ERROR` | Non-retryable HTTP response |
| `TIMEOUT` | Timeout attempts exhausted |
| `NETWORK_ERROR` | Network/request attempts exhausted |
| `ROBOTS_BLOCKED` | Cached robots policy disallowed the request |
| `ACCESS_RESTRICTED` | Evidence-backed domain policy prevents known-futile access attempts |
| `UNSUPPORTED_CONTENT` | `2xx` response with a non-allowed media type |
| `RETRY_EXHAUSTED` | Retryable HTTP status persisted after final attempt |
| `INVALID_URL` | URL is malformed or not HTTP(S) |

## Running safely

Default safe pilot (20 URLs):

```powershell
.venv\Scripts\python.exe scripts\crawl_corpus.py
```

Bounded mixed-domain pilot:

```powershell
.venv\Scripts\python.exe scripts\crawl_corpus.py `
  --limit 10 `
  --domain bachmai.gov.vn `
  --domain suckhoecongdongonline.vn `
  --domain ask.familydoctor.com.cn `
  --domain 120ask.com `
  --domain cnkang.com `
  --output-db data\crawled\phase2a_pilot.sqlite
```

Specific records can be selected with `--ids 584 585` or comma-separated
values. Raw pilot bodies require the explicit `--store-raw-body` flag and are
capped by `storage.max_body_bytes`.

## Phase 2A pilot evidence

On 2026-10-03, the bounded command above selected 10 URLs across five domains:

- first run: 10 scheduled, 10 `SUCCESS`, 10 attempts, 0 raw bodies;
- robots: one fetch for each of the five domains;
- content types: 10 `text/html` responses;
- redirects: 6 HTTP-to-HTTPS final URLs;
- second identical run: 10 selected, 0 scheduled, 10 skipped, 0 attempts,
  and 0 robots fetches;
- persisted database total after both runs: 10 unique rows.

The live sample did not encounter an error. Offline mocked tests cover
transient retry, permanent `404` classification, robots disallow, retryable
resume, duplicate `doc_id` protection, and successful-row transition safety.

## Intentionally deferred

Phase 2A does not include:

- unrestricted full-corpus execution (production readiness is complete, but the full crawl was not run);
- article-text extraction or HTML cleaning;
- raw-body archives beyond explicitly capped pilot storage;
- content chunking, embeddings, FAISS, BM25, reranking, or evaluation;
- submission generation;
- Phase 2B/2C processing decisions based on fetched content.
