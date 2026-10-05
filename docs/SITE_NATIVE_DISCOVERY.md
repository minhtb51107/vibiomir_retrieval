# Phase 10B1 — Multilingual Site-Native Discovery

## Outcome

**Final verdict: NOT VIABLE AS A MAJOR DISCOVERY CHANNEL.** The predeclared G0
gate stopped the experiment before official-query testing. Of 19 deliberately
selected domains representing 88.728% of all official rows, only
`a-hospital.com` exposed a query-sensitive first-party search whose sampled
content URLs mapped back to the official corpus. That domain represents 3.853%
of corpus rows, below both the 10% PARTIAL threshold and the 5% specialist
exception.

No Stage 1 or Stage 2 query experiment ran. No candidate page was crawled, no
translation model was downloaded, and no embedding, reranking, or submission
work was performed.

## Predeclared expected-outcome matrix

These branches were written to
`artifacts/phase10b1_site_discovery/premortem.json` before the first request.

| Result | Predeclared meaning | Required action |
|---|---|---|
| G0 STRONG | Usable, mappable native-search domains represent at least 30% of official rows | Permit the 120-query Stage 1 signal test |
| G0 PARTIAL | Usable domains represent 10–30% | Permit Stage 1, but interpret as a partial channel |
| G0 PARTIAL_SPECIALIST | Under 10% overall, but a strategically valuable source/language class represents at least 5% | Permit a justified specialist Stage 1 only |
| G0 WEAK | Under 10% and no qualifying specialist | Stop before Stage 1 and publish the source map |
| G1 PROMISING | At least 30% query coverage, or 40% in an important subgroup, or 5× Bing candidates/query, with reliable mapping, low duplication, and cheap operation | Permit the 300-query Stage 2 confirmation |
| G1 BORDERLINE | 15–30% overall coverage or strong value confined to a subgroup | Stage 2 only with explicit justification |
| G1 NOT VIABLE | Under 15%, no valuable subgroup, and poor novelty | Stop before Stage 2 |
| G2 valuable | Stage 2 has useful mapped candidates, novelty, efficiency, stable source contribution, and a locally affordable 1,200-query projection | Recommend—but do not start—1,200-query scaling |

The actual G0 result was WEAK, so later branches were not eligible to run.

## Why this differed from Phase 10B0

Phase 10B0 let Bing RSS choose exposed domains and found only 21 unique official
documents across five domains. Phase 10B1 instead inspected first-party
mechanisms domain by domain. This directly tested whether domain bias could be
removed without repeating Phase 8C URL-text BM25. It could not: native search
was absent, nonfunctional, blocked, or unmappable for nearly all weighted
corpus share.

## Stage 0 domain inventory

All 4,394,718 rows and all 97 domains were scanned locally in 23.9 seconds.
The source/domain association heuristic classified 47 domains (3,654,120 rows,
83.15%) as Chinese-associated, 47 (732,785 rows, 16.67%) as
Vietnamese-associated, and three small domains (7,813 rows, 0.18%) as unknown.
These labels are metadata inference, not measured full-page language.

The selected 19 domains represented 88.728% of corpus rows. Selection balanced
weighted share, source language, medium-sized sources, and known accessibility
outcomes. It deliberately included robots-blocked `zysjonline.com`,
Cloudflare-restricted `nhathuoclongchau.com.vn`, and JS-shell-producing
`laodong.vn` so that optimistic accessibility assumptions could fail cheaply.

| Domain | Corpus share | Association | Key Stage 0 observation |
|---|---:|---|---|
| cnkang.com | 21.923% | zh | Public GET form, but two distinct terms returned the identical mapped popular-page set; query-insensitive |
| 120ask.com | 20.900% | zh | Category links; no usable native query search detected |
| familydoctor.com.cn | 10.182% | zh | Sitemap declared; no usable native search detected |
| ask.39.net | 7.677% | zh | Category links; no usable native query search detected |
| zysjonline.com | 5.537% | zh | 403/robots evidence; blocked |
| a-hospital.com | 3.853% | zh | Query-sensitive MediaWiki-style search; mapped sampled pages |
| suckhoecongdongonline.vn | 3.516% | vi | Search form responded, but sampled official mapping did not pass |
| zhongyibaodian.net | 3.338% | zh | Search action was cross-host/unavailable; no usable mapped mechanism |
| suckhoedoisong.vn | 1.953% | vi | Sitemap declared; not relevance search |
| nhathuoclongchau.com.vn | 1.811% | vi | Sitemap declared, but homepage/search access remained restricted |
| zydcd.com | 1.790% | zh | No usable discovery mechanism detected |
| thanhnien.vn | 1.641% | vi | Native search page found; five sampled result URLs did not map |
| wujue.com | 1.214% | zh | Probe unavailable; no mechanism established |
| youlai.cn | 1.049% | zh | No usable mechanism detected |
| laodong.vn | 0.712% | vi | No usable search; prior Phase 2 pages were JS shells |
| vinmec.com | 0.595% | vi | No usable first-party mechanism established in bounded probe |
| medlatec.vn | 0.563% | vi | Sitemap/category mechanism; not native relevance search |
| vietnamnet.vn | 0.354% | vi | Search-looking form returned a non-query home page; sitemap declared |
| pmc-ecm-healthblog.beta.pharmacity.io | 0.121% | unknown | 403 responses; no usable mechanism |

## Stage 0A mechanism probe

The probe used 49 new sequential HTTP requests total, a two-second inter-request
delay, a 20-second timeout, no concurrency, and at most four requests per
domain. Responses were transactionally cached in ignored
`data/site_discovery/stage0.sqlite`. It found:

- six public GET search forms or search links;
- seven robots-declared sitemaps;
- three observable category-index patterns;
- eight selected domains with no usable mechanism detected;
- no paid endpoint, login, CAPTCHA solution, private API, proxy, or bypass.

Sitemaps and categories are recorded as acquisition mechanisms, not relevance
search, and therefore did not inflate G0.

## Stage 0B mapping and query-sensitivity probe

At most five URLs per mechanism were checked against every official corpus URL
using exact and conservative normalized mapping. No ambiguous mapping was
accepted. An initial generic term made `cnkang.com` look promising, but the
pre-mortem required a duplicate/popular-page control. A second Chinese medical
term returned the exact same five official URLs, proving the form was not
query-sensitive for this access pattern. It was disqualified.

`a-hospital.com` returned different page sets for `健康` and `糖尿病`; four of
five first-probe URLs and two of five second-probe URLs mapped to official IDs.
It was the sole usable, query-sensitive mechanism.

Across the six mechanisms that yielded mapping samples, 30 URLs were inspected:
17 (56.7%) were content-like by explicit path rules and 13 (43.3%) were
category/sitemap-index-like. Content-like rates were 100% for the sampled
`cnkang.com`, `a-hospital.com`, and `thanhnien.vn` sets, 40% for
`suckhoecongdongonline.vn`, and 0% for the sampled `suckhoedoisong.vn` and
`medlatec.vn` sitemap entries. Content-like shape alone was not treated as a
mapping or relevance signal.

## G0 result and stop

- Usable native-search domains: **1/97** (`a-hospital.com`).
- Corpus-weighted usable search coverage: **3.853%**.
- G0: **WEAK**.
- Stage 1: **not run** (0 queries, 0 requests).
- G1: **not reached**.
- Stage 2: **not run** (0 queries, 0 requests).
- G2: **not reached**.

Consequently, query coverage, mapped documents/query, candidate novelty versus
Bing/pilot, discovery efficiency, and a 1,200-query candidate projection are
correctly marked **NOT MEASURED**, rather than fabricated from a failed
precondition.

## Translation decision

The cache audit found BGE-M3 and unrelated vision models, but no suitable open
translation model; `sentencepiece` and `sacremoses` were also absent. Per the
pre-mortem, translation could not rescue a failed weighted mechanism gate, so
no model was downloaded. Q2 English and Q3 Chinese official-query translation
remain **UNTESTED**. The two fixed Chinese terms used only to validate mechanism
query sensitivity were not translations of competition queries.

## Reusable source map

The complete [97-domain source map](OFFICIAL_SOURCE_DISCOVERY_MAP.md) combines
official row counts, inferred source association, Phase 2 accessibility and
pilot evidence, Stage 0 mechanism observations, mapping behavior, and a
conservative acquisition recommendation. Counts are:

- SEARCH: 1 domain;
- SITEMAP: 5 domains;
- CATEGORY: 2 domains;
- DIRECT_CRAWL: 20 domains with prior accessibility/pilot evidence;
- BLOCKED: 3 domains;
- UNKNOWN: 66 domains not safely characterized.

`UNKNOWN` is intentional. The experiment did not manufacture 97 adapters or
infer private endpoints.

## Cost and scale implications

A 120-query run against the sole usable source would require at least 120
search requests after obtaining Chinese translations, yet could address only
3.853% of corpus rows. A 1,200-query run would require at least 1,200 requests,
but Stage 0 provides no defensible candidate-count, crawl-storage, or indexing
projection because official-query behavior was gated off. Spending those
requests and downloading a translation model is not justified.

The exact recommended next action is: **do not scale site-native query search**.
Use the source map to plan bounded, source-specific acquisition. Sitemaps and
category feeds may be assessed separately as acquisition enumerators, while
known accessible domains can use the existing resumable direct crawler. That
would be a new approved phase; it was not started here.

## Reproducibility and limitations

Configuration is in `configs/site_discovery.yaml`; gate definitions and fatal
assumptions are in `premortem.json`; inventory/probe/repeat/summary artifacts
are under `artifacts/phase10b1_site_discovery/`. Raw HTTP cache is ignored.
First-party sites can change, and a bounded probe can miss undocumented public
features. This does not justify bypassing the gate: the observed usable weighted
share is too small for query-scale work under current resource constraints.
