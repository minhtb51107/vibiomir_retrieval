# Phase 10B0 — Search-Based Discovery Feasibility

## Hypothesis and scope

Phase 8C showed that Vietnamese-query matching against URL path text is not a
safe corpus reducer: at depth 1,000 it kept only 0.2534% of prior reranked
pilot candidates and systematically excluded Han-script and opaque URLs.
Phase 10B0 tests a different signal: whether a public search engine's existing
page-content index can discover official URLs for a Vietnamese medical query.

The experiment is discovery-only. It fetched search-result metadata but did
not request any candidate page, crawl any mapped URL, build a retrieval index,
embed or rerank content, generate a submission, or run Stage C.

**Final verdict: NOT VIABLE.** Search access was technically reliable, but the
300-query study mapped only 21 unique official documents for 39 queries, and
results collapsed onto five Vietnamese-associated domains. No meaningful
category subgroup crossed the predeclared gates.

## Provider and access policy

The mechanism was Bing's public machine-readable RSS response from
`https://www.bing.com/search?format=rss`. It required no API key, account,
billing, CAPTCHA, HTML scraping, proxy, or bypass. Requests were sequential,
spaced by at least 1.25 seconds, retried only for transient network/429/5xx
failures, and cached transactionally.

The access probe found that RSS returned at most 10 results and ignored
`count`/`first` pagination. The study therefore measured depth 10 only; it did
not scrape HTML to force depths 20 or 50. This endpoint is a public RSS
mechanism rather than a paid supported API, so long-term availability is not
guaranteed.

All 600 Stage A/B responses completed successfully: no rate limits, provider
errors, bot challenges, or retries exhausted. The Stage A completion window
was 130.64 seconds for 100 requests; the 500 new Stage B responses completed
over 830.19 seconds. Raw titles, snippets, URLs, statuses, mappings, and request
fingerprints are cached in ignored
`data/search_discovery/search_cache.sqlite` (3.45 MiB, SQLite integrity `ok`).

## Sample construction

The official 1,200 queries were characterized deterministically using:

- five character-length quantiles;
- keyword proxies for drug, symptom, diagnosis/testing, treatment,
  anatomy/physiology, disease, international terms, and general questions;
- simple versus multipart/long-question signals;
- a descriptive pilot-evidence proxy based on dense/sparse top-10 document
  agreement (`strong`, `mixed`, or `weak`).

No semantic category or relevance label was fabricated. A seeded round-robin
over length quantile × category proxy × evidence proxy selected 300 Stage B
queries; a second deterministic round-robin selected a 50-query Stage A
subset. The IDs, original text, features, and rationale are recorded in
`artifacts/phase10b0_search/query_sample.json`.

Stage A covered all five length quantiles, seven represented category proxies,
12 multipart queries, and 14/14/22 strong/mixed/weak evidence queries. Stage B
contained 300 queries, including 59 multipart and 76/102/122
strong/mixed/weak evidence queries.

## Query variants

- **Q0:** original Vietnamese query with NFC and whitespace normalization.
- **Q1:** deterministic concise Vietnamese form that removes common question
  scaffolding while retaining medical terms, numbers, and up to 24 tokens.
- **EN/ZH:** **NOT TESTED**. The local cache contained BGE-M3 but no translation
  model. No large model was downloaded and no closed LLM/API was used.

## URL mapping

Every distinct result URL was compared with all 4,394,718 official URLs. The
mapper applies increasingly permissive levels and assigns a `doc_id` only when
the relevant level has exactly one official candidate:

1. exact original URL;
2. safe normalized URL: NFC, lower-case scheme/host, default-port removal,
   fragment removal, unreserved percent decoding, trailing-slash normalization,
   and removal of configured common tracking parameters;
3. canonical-like key that additionally ignores HTTP versus HTTPS.

`www`/non-`www` equivalence is disabled because the corpus does not provide
document-equivalence evidence. Ambiguous matches are retained as diagnostics
and never assigned. In the measured data all 21 mapped distinct result URLs
were exact; normalized and ambiguous matches were both zero.

## Stage A — technical pilot

| Metric | Result |
|---|---:|
| Queries | 50 |
| Query variants / requests | 2 / 100 |
| Search result rows | 951 |
| Unique result URLs | 579 |
| Exact mapped result rows | 8 |
| Unique mapped official documents | 5 |
| Queries with at least one mapped document | 4 (8.0%) |
| Official domains represented | 2 |
| Provider failures/rate limits | 0 / 0 |

The predeclared hard-failure condition required both a map rate at or below 2%
and at most two mapped documents, or an access completion rate below 90%.
Stage A therefore passed technically and triggered Stage B. It was not treated
as evidence that the method was useful.

## Stage B — 300-query feasibility study

### Core results

| Metric | Combined Q0 + Q1 |
|---|---:|
| Queries | 300 |
| Successful cached requests | 600/600 |
| Search result rows | 5,535 |
| Unique result URLs | 2,414 |
| Exact mapped result rows | 78 |
| Normalized mapped result rows | 0 |
| Ambiguous result rows | 0 |
| Result-row mapping rate | 1.4092% |
| Queries with ≥1 mapped official document | 39 (13.0%) |
| Mapped docs/query min / median / p90 / max | 0 / 0 / 1 / 4 |
| Unique query/document pairs | 48 |
| Global unique official document IDs | 21 |
| Official domains represented | 5/97 |

### Variant contribution

| Variant | Result rows | Queries mapped | Map rate | Unique mapped docs |
|---|---:|---:|---:|---:|
| Q0 original VI | 2,766 | 22 | 7.33% | 18 |
| Q1 concise VI | 2,769 | 39 | 13.00% | 21 |

Every Q0-success query also succeeded with Q1; Q1 added 17 query successes, 18
unique query/document pairs, and three global document IDs not found by Q0.
Those 18 pairs are 37.5% of the combined 48 pairs. Concision helped, but the
absolute candidate coverage remained too small. EN/ZH contribution is unknown,
not zero, because those variants were not tested.

### Zero-map failures

There were 261 zero-map queries. For 257, the provider returned results but
none belonged to the official corpus. For four, both variants returned no
results. There were no ambiguous-only failures, provider errors, rate limits,
or canonicalization failures. At query-variant level, 522 cases had results
outside the official corpus and 17 returned nothing.

The highest category-proxy map rates were treatment 19.30%, symptom 17.39%,
general 15.79%, and drug 13.33%. Diagnosis/testing reached 6%; the small
disease, international-term, and anatomy/physiology groups yielded none. No
subgroup with at least 30 queries reached the predeclared 65% strong-subgroup
gate.

## Domain behavior

Only five of 97 official domains appeared:

| Official hostname | Mapped result rows | Search share | Corpus share |
|---|---:|---:|---:|
| `nhathuoclongchau.com.vn` | 57 | 73.08% | 1.81% |
| `www.vinmec.com` | 11 | 14.10% | 0.59% |
| `medlatec.vn` | 4 | 5.13% | 0.56% |
| `hellobacsi.com` | 4 | 5.13% | 0.27% |
| `tiemchunglongchau.com.vn` | 2 | 2.56% | 0.30% |

The top domain alone supplied 73.08% of mapped rows and the top five supplied
100%. Large Chinese-associated domains—including the two domains holding 42.8%
of the corpus—were absent. Search discovery therefore collapses onto a small,
popular Vietnamese web subset rather than covering the official corpus.

## Pilot overlap and random baseline

For the same 300 queries, the 48 discovered query/document pairs had zero
overlap with dense, sparse, hybrid, or reranked pilot candidates at top 10,
top 50, and full available candidate depth. A deterministic random official-ID
sample with exactly the same per-query candidate counts also produced zero
overlap at every level.

This is **candidate overlap with independently content-retrieved pilot docs,
not relevance recall or evaluation**. Search did not outperform random on this
proxy, but its 21 documents are outside the tiny processed pilot and cannot be
judged without crawling—which this phase intentionally does not do.

## Gates and verdict

The gates were committed to configuration before Stage B interpretation.
Promising required either at least 50% query coverage with ≥10 domains and
top-domain share ≤70%, or a ≥30-query subgroup with ≥65% coverage. Borderline
required at least 20% query coverage, ≥5 domains, and top-domain share ≤85%.

Measured coverage was 13.0%, only five domains appeared, top-domain share was
73.08%, and no subgroup passed. The verdict is therefore **NOT VIABLE**. The
small candidate pool is not a success: it reflects missing coverage rather
than a useful reduction.

## Stage C and cost projection

Stage C was not run. A simple linear projection from 300 to 1,200 queries is:

| Projection | Value |
|---|---:|
| Search requests (Q0 + Q1) | 2,400 |
| Unique search-result URLs | ~9,656 |
| Unique mapped official IDs / crawl candidates | ~84 |
| Crawl time at Phase 8's 122 URLs/minute | ~0.69 minutes |
| Retained storage low / expected / high | 5.25 / 8.75 / 13.13 MiB |

These costs are trivial, but approximately 84 candidates from a 4.39-million
URL corpus cannot address the demonstrated coverage blocker. Extrapolation is
uncertain because domains and URLs repeat; it is an engineering scale estimate,
not a relevance claim.

## Reproduction and resume

```powershell
.venv\Scripts\python.exe scripts\run_search_discovery.py `
  --config configs\search_discovery.yaml --stage A

# Run only after Stage A passes its hard technical gate:
.venv\Scripts\python.exe scripts\run_search_discovery.py `
  --config configs\search_discovery.yaml --stage B
```

Response fingerprints hash provider, normalized request text, and depth.
Completed responses and mappings are reused; rerunning either command makes no
unnecessary request. SQLite transactions preserve every completed response and
result list across interruption. Tests use a mock provider and make no network
calls.

## Recommended action

Do not run the 1,200-query Stage C, do not crawl these 21/84 projected
candidates as a corpus-coverage solution, and do not loosen Phase 8C URL
selection. If discovery work continues, it needs a fundamentally different
source with broader official-domain coverage—particularly indexed Chinese
medical sources—not more depth from this RSS mechanism. Phase 10B1 is not
started automatically.
