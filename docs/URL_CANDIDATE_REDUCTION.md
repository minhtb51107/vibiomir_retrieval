# Query-Conditioned URL Candidate Reduction

## Scope and verdict

This Phase 8C experiment asks whether the 4,394,718-row corpus can be reduced
using only the 1,200 official queries and already-local URL metadata. It made
zero HTTP requests, fetched no page content, generated no submission, and did
not start Phase 9.

**Verdict: NOT VIABLE.** The combined method produces a tractable 415,767-URL
set at depth 1,000 per query (90.54% corpus reduction), but it retains only
0.2534% of the existing reranked pilot candidates for the same queries. It
also sharply under-represents Han-script and opaque URLs. These failures trip
the predeclared promising and borderline gates. A 10,000-URL candidate crawl
is therefore **not justified** by this experiment.

Candidate retention is a proxy comparison with prior pilot retrieval output,
not ground-truth recall. No relevance or F2 claim is made.

## Reproducible method

`scripts/evaluate_url_reduction.py` reads `query.parquet` and
`links_corpus.parquet` in bounded Arrow batches. A hard socket guard rejects
network connections for the duration of the command. The source Parquet
SHA-256 values, settings, row counts, and index hash are recorded in the
artifacts.

Each URL is parsed without changing its `doc_id` or original value. The
representation:

- strips the fragment from retrieval text while retaining query-string keys
  and values;
- safely percent-decodes path/query text, normalizes it to NFC, case-folds it,
  and treats URL separators as token boundaries;
- emits Unicode word tokens plus accent-folded variants and Han unigrams and
  bigrams;
- records domain, scheme, path-token count, numeric ratio, Han,
  Vietnamese-like, and opaque signals;
- hashes the normalized host/path/query representation for duplicate-text
  diagnostics.

Hostname text is deliberately excluded from lexical matching. Domain evidence
is handled separately by the domain prior. This prevents accent-folded
Vietnamese terms such as `cơm -> com` from becoming false matches against the
`.com` top-level domain.

The lexical index is contentless SQLite FTS5 over URL path/query tokens that
occur in the query vocabulary. Its deterministic BM25 uses FTS5's fixed
conventional `k1=1.2`, `b=0.75`. Per query, at most the 12 nonzero-DF terms
with highest inverse-document-frequency are used, preventing long natural
language questions from turning every generic URL word into a posting scan.
Zero-DF terms are explicitly excluded from that budget.

Three outputs are materialized at depths 50, 100, 200, 500, and 1,000:

1. lexical BM25;
2. deterministic domain-prior samples (up to 32 stable rows per domain);
3. a combined selection with a 20% domain cap, at least 10 domains where
   possible, and a 50% fallback reserve only for weak queries.

Weak means fewer than 25 lexical candidates or best BM25 below 2.0. The
configuration and viability gates were fixed in
`configs/url_candidate_reduction.yaml` before interpreting results.

## Corpus and index measurements

All 4,394,718 corpus rows and all 1,200 queries were processed. The scan found
904,065 URLs with at least one matching query-vocabulary path/query token;
905,359 rows are stored after adding the compact fallback reservoir. There
are 2,784 query-vocabulary terms and 8,349,922 term-document incidences.

| Measurement | Result |
|---|---:|
| URL preprocessing scan | 526.27 s (8,350.63 URLs/s) |
| Index finalization | 13.72 s |
| Index build total | 539.99 s |
| Retrieval, 1,200 queries | 562.21 s (468.51 ms/query) |
| Candidate/global diagnostics command | 729.58 s |
| Combined measured pipeline | 1,269.57 s (21.16 min) |
| FTS index | 283,308,032 bytes (270.18 MiB) |
| URL-text hash array | 35,157,744 bytes (33.53 MiB) |
| Full candidate database | 662,343,680 bytes (631.66 MiB) |

Peak process RSS was not available from the installed environment and is not
claimed. Both SQLite databases passed `PRAGMA integrity_check`. Large index,
hash, and candidate files remain ignored under `data/url_reduction/`.

## Reduction and diversity

The combined method returns exactly the configured number of candidates for
every query at every depth.

| Depth/query | Global unique URLs | Reduction | Domains | Top-1 share | Top-5 share |
|---:|---:|---:|---:|---:|---:|
| 50 | 45,594 | 98.96% | 89 | 10.27% | 43.91% |
| 100 | 83,561 | 98.10% | 97 | 10.31% | 44.95% |
| 200 | 145,824 | 96.68% | 97 | 10.52% | 45.97% |
| 500 | 279,772 | 93.63% | 97 | 11.64% | 47.03% |
| 1,000 | 415,767 | 90.54% | 97 | 13.03% | 48.19% |

For comparison, the full corpus top-1 and top-5 domain shares are 21.92% and
66.22%. Domain balancing therefore preserves broad hostname coverage and does
not collapse onto the largest sources. It does not, however, preserve the
distribution within those domains.

| Signal | Full corpus | Combined depth 50 | Combined depth 1,000 |
|---|---:|---:|---:|
| Han-script URL share | 3.8433% | 0.0461% | 0.0685% |
| Vietnamese-like URL share | 3.7120% | 23.7860% | 22.5254% |
| Opaque/numeric URL share | 6.1052% | 0.3093% | 0.2093% |
| HTTPS share | 68.95% | 98.97% | 96.32% |
| URLs with path signal | ~100% | 100% | 100% |

“Vietnamese-like” is a URL-character/slug heuristic, not measured page
language. The strong Vietnamese-like enrichment is expected from Vietnamese
queries, while the severe Han, opaque, and HTTP loss demonstrates destructive
metadata bias. At depth 1,000, 1,253 normalized URL-text duplicate groups are
present; distinct `doc_id` values are never merged.

## Query behavior

Only 2/1,200 queries (0.17%) meet the configured weak-query definition after
zero-DF terms are excluded. Lexical pool size has min/median/max 10/3,000/3,000.
Best lexical BM25 has min/median/p95/max
12.4023/21.0357/30.0987/45.2025. At depth 1,000, candidates marked as having
lexical evidence average 98.43% per query. These scores show that URL words
can usually be found; they do not show that the URLs lead to relevant pages.

The fallback remains important for script and domain coverage, but its small
deterministic reservoir cannot recover query-specific opaque URLs. Expanding
it enough to cover those URLs would erode the intended crawl reduction.

## Pilot candidate-retention diagnostic

For each query, prior content-level candidate `doc_id` values were checked
against the combined URL selection for that same query. The totals are unique
query/document pairs from the persisted Phase 4/6/7 outputs.

| URL depth | Dense (33,208) | Sparse (25,606) | Hybrid (25,610) | Reranked (46,561) |
|---:|---:|---:|---:|---:|
| 50 | 0.0120% | 0.0195% | 0.0195% | 0.0107% |
| 100 | 0.0271% | 0.0508% | 0.0469% | 0.0301% |
| 200 | 0.0512% | 0.0937% | 0.0781% | 0.0494% |
| 500 | 0.1174% | 0.2499% | 0.2030% | 0.1332% |
| 1,000 | 0.2319% | 0.4882% | 0.3944% | 0.2534% |

Even for prior top-10 content candidates, depth-1,000 retention is only
0.7164% dense, 1.3496% sparse, 1.1591% hybrid, and 0.7406% reranked. This is
the decisive feasibility failure. It is descriptive retention against a
small pilot index, not relevance recall.

## Predeclared gates and decision

The promising gate required at most 300,000 global URLs at depth 1,000, at
least 80% reranked retention, no more than 30% weak queries, at least 80% of
domains, and at least 50% preservation of both Han and opaque shares. The
borderline gate allowed 800,000 URLs, 60% retention, 50% weak queries, 60% of
domains, and 30% script/opaque share preservation.

The result passes weak-query and domain-count gates. It fails the promising
size gate, both retention gates by orders of magnitude, and both Han/opaque
preservation gates. The verdict is therefore **NOT VIABLE**, despite the
apparently attractive corpus-size reduction.

No 10,000-URL crawl should be run from this selector. A future experiment
would require a fundamentally different source of evidence (for example,
content or source-specific metadata), not looser interpretation of these
gates.
