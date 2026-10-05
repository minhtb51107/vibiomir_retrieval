# Phase 10C0 — S4 Targeting Tournament

## Decision

**No tested targeting method is viable for selecting an S4-5K crawl.** The
next acquisition experiment should add 4,000 broadly and deterministically
sampled, previously uncrawled S4 URLs. No candidate pages were crawled and no
candidate manifest was created in this phase.

This answers the strategic question—“We know the neighborhood is good, but
can we identify promising houses before knocking on more doors?”—with **not
reliably from the cheap signals tested here**.

## Fixed evidence and proxy sets

S4 (`suckhoecongdongonline.vn`) contains 154,503 official URLs (3.516% of the
full corpus). Phase 10B2 measured 99.8% usable extraction and about 59 usable
documents/minute. The fixed Phase 10B3 organizer result improved from B0
`FINAL_SCORE=0.0001` to S4 `0.0004`; document and chunk recall each moved from
`0.0001` to `0.0003`.

The tournament uses retrieval-selected proxies, not gold labels:

| Proxy | Distinct S4 documents |
|---|---:|
| Entered final top 10 for at least one query | 315 |
| Entered final top 5 | 216 |
| Entered final top 3 | 171 |
| Selected for at least two queries | 210 |

The organizer evidence warns that proxy movement is not relevance: S3 and S5
changed many local rankings without improving organizer metrics. Retention
below is therefore a diagnostic, not accuracy or recall against ground truth.

The common candidate budgets cover 5,000 (3.24%), 10,000 (6.47%), 20,000
(12.94%), and 40,000 (25.89%) of the 154,503-URL source. The full-source
reference is 154,503 (100%).

## Predeclared gates

At no more than 20K candidates, STRONG requires at least 70% primary, top-5,
and top-3 retention and clear lift over random. PROMISING requires 40–70%
primary retention and material lift. Under 40% at 20K is WEAK. A method that
is not query-sensitive, cannot map official IDs, or has no meaningful lift is
NOT VIABLE. Only PROMISING/STRONG methods may enter the unweighted RRF ensemble.

## Methods and common-budget results

| Method | 5K retention | 10K retention | 20K retention | 40K retention | Top5 @20K | Top3 @20K | Random @20K | Lift @20K | Verdict | Cost/network |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| A — URL lexical BM25/RRF | 7.30% | 15.87% | 24.13% | 45.71% | 31.02% | 36.26% | 12.94% | 1.86× | WEAK | 170.96 s CPU; 0 requests |
| B — path/category | — | — | — | — | — | — | 12.94% | — | NOT VIABLE | local inspection; 0 requests |
| C — seed-and-expand | 3.81% | 7.62% | 15.24% | 28.57% | 15.74% | 15.79% | 12.94% | 1.18× | WEAK | 7.64 s CPU; 0 requests |
| D — site-restricted Bing RSS | 0% | 0% | 0% | 0% | 0% | 0% | 12.94% | 0× | NOT VIABLE | 50 cached sequential requests |
| E — native search | — | — | — | — | — | — | 12.94% | — | NOT VIABLE | prior preflight reused; 0 requests |

URL preprocessing took 37.15 seconds. Method A excludes hostname/TLD tokens,
preserves Vietnamese Unicode and accent-folded variants, separates the 90
opaque URLs from 154,413 informative slugs, scores each query with BM25, then
aggregates per-query top-1,000 rankings using RRF.

Method B failed before ranking: 154,502/154,503 URLs have exactly one root path
segment and the remaining URL is the root. There are no repeated category
prefixes such as `/benh/...` or `/topic/...`; inventing categories from words
inside unique slugs would merely duplicate Method A.

Method C used stable-hash five-fold cross-validation over the 315 primary
proxies. Each fold used only the other four folds as token-family seeds and
measured retention solely on held-out IDs. The all-seed ordering is exploratory
and was not used as evaluation evidence.

Method D stopped at its declared 50-query gate. It returned 481 rows and 390
unique URLs, but zero mapped to an official S4 ID and 0/50 queries obtained a
mapped candidate. Stage B (300 queries) did not run. Method E reused the Phase
10B1 preflight: the public search form produced 0/5 official mappings and only
40% content-like sampled pages, so no new first-party request was justified.

## Ensemble

No ensemble was run. A and C were both WEAK, while B, D, and E were NOT
VIABLE. Letting failed arms vote would violate the predeclared rule and could
make a poor ranking look more sophisticated without adding evidence. There is
therefore no “ensemble versus best method” winner; Method A is the least weak
single method but still misses 75.87% of primary proxies at 20K.

## Acquisition recommendation

There is no supported way to shrink roughly 154K S4 URLs substantially with
these signals. A 20K lexical pool is too destructive; even 40K retains only
45.71% of primary proxies. The exact next milestone, if separately approved,
is:

- exclude the 1,000 already acquired S4 document IDs;
- select **4,000 new official S4 URLs by the existing broad deterministic
  sampling rule** across the remaining source;
- add them to the existing S4-1K to form S4-5K;
- do not privilege URL lexical, seed, external-search, or native-search ranks.

This phase recommends broad sampling rather than targeted or mixed selection.
The general explore/exploit policy still stands: if a future content-backed
selector becomes STRONG, use at most 80% targeted and retain 20% broad
exploration so new regions remain discoverable and targeting drift is visible.

## What failed and what was learned

- Informative-looking Vietnamese slugs were not sufficient: lexical lift was
  real but far below the retention gate.
- Seed expansion looked intuitive but, under leakage-safe held-out evaluation,
  was barely above random.
- S4 exposes no useful category hierarchy in official URL paths.
- General search indexed S4 pages that did not correspond to official corpus
  URLs; canonicalization was not the blocker.
- Native search had already failed query-sensitive official mapping.
- Negative methods were stopped at their gates; no 300-query search, S4-5K
  crawl, adaptive crawl, GPU work, or leaderboard submission occurred.

## Reproduction

Configuration is `configs/s4_targeting.yaml`. Local rankings and the search
cache remain under ignored `data/s4_targeting/`. Compact evidence is under
`artifacts/phase10c0_s4_targeting/`.

```powershell
.venv\Scripts\python.exe scripts\run_s4_targeting.py local
.venv\Scripts\python.exe scripts\run_s4_targeting.py search-a
.venv\Scripts\python.exe scripts\run_s4_targeting.py finalize
```

`search-b` is gate-protected and was not run.
