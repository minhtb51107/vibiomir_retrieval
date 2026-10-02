# SOURCE_PROBE_REPORT.md — Phase 2B

> **Status:** COMPLETE
> **Executed:** 2026-10-03
> **Scope:** source probing and generic extraction experiments only
> **Unique final sample:** 118 corpus URLs
> **Cumulative live corpus URL operations:** 173, including targeted corrective re-probes
> **Raw response bodies committed:** NONE

## 1. Method and safety boundary

The sampler reads only `id` and `url` from the raw Parquet and keeps the lowest
seeded BLAKE2 ranks for each configured rule. This makes selection repeatable,
independent of corpus file order, and balanced by rule rather than corpus
frequency. The seed is `vibiomir-phase2b-v1`.

The 23 rules selected:

- 8 URLs from each of the Phase 1 top-ten domains (80 slots);
- 3 each from Thanhnien, Laodong, Vinmec, and Medlatec (12 slots);
- 4 from `test.pmphai.com`;
- 4 each for query strings, HTTP URLs, and no-extension paths;
- 2 each for `.ldo`, `.vov`, `.vov2`, `.vnp`, and `.tpo`.

The union contained 118 unique `doc_id` values across 21 actual hostnames.
Fetch URLs retained query strings and contained no fragments. The initial run
probed all 118. Two corrective runs repeated 35 and 20 affected rows after
improving analysis logic; cumulative corpus URL operations were therefore 173,
below the absolute limit of 200. There were no request retries. Robots and
redirect requests were ancillary protocol requests and are not additional
corpus rows.

Artifacts:

- `artifacts/source_probe/sample_manifest.json` and `.csv`;
- `artifacts/source_probe/probe_results.json` and `.csv`;
- `artifacts/source_probe/summary.json`;
- `artifacts/source_probe/run_log.json`.

No HTML, response body, extracted article text, cookies, or credentials are
stored in these artifacts.

## 2. Response outcomes

| Outcome | Count | Share of 118 |
|---|---:|---:|
| Successful HTTP HTML | 99 | 83.9% |
| HTTP error (`403`) | 8 | 6.8% |
| Robots blocked, not fetched | 11 | 9.3% |

All 107 fetched responses declared `text/html`; no PDF or other content type
was observed. The 11 robots-blocked rows have no response content type. There
were 13 redirects: six on `120ask.com` and seven on `cnkang.com`, principally
HTTP-to-HTTPS. No redirect loop, timeout, network error, login wall, or empty
HTTP body occurred. Successful response sizes ranged from 177 to 604,891 bytes
(median 47,669); none reached the 2 MiB probe cap.

Category checks were intentionally small but useful:

- all 4 query-string samples succeeded as HTML;
- all 4 HTTP-feature samples succeeded as HTML;
- all 10 unusual-suffix samples (`.ldo`, `.vov`, `.vov2`, `.vnp`, `.tpo`)
  succeeded as HTML, confirming that suffix is not a content-type signal;
- one of four no-extension feature samples succeeded; three belonged to a
  domain whose robots policy disallowed probing.

## 3. Domain and template observations

The content-language column below is measured from decoded page text using
Han-script counts and Vietnamese-specific characters/stopwords. It is not
derived from the hostname. Template labels are probe classifications, not
hardcoded extraction selectors.

| Domain | n | Outcome | Measured content signal | Evidence-based template / handling |
|---|---:|---|---|---|
| `cnkang.com` | 10 | 10 success | 10 Han | Mostly generic static; 3 structurally unknown; density fallback needed |
| `120ask.com` | 8 | 8 success | 8 Han | 8 Q&A; preserve question/answer boundaries |
| `familydoctor.com.cn` | 8 | 8 success | 8 Han | 4 Q&A, 4 generic static |
| `ask.39.net` | 8 | 8 success | 8 Han | 8 Q&A; semantic container missed most text |
| `zysjonline.com` | 11 | 11 robots blocked | Not measured | Respect disallow; no page request made |
| `a-hospital.com` | 8 | 8 success | 8 Han | 8 generic static |
| `suckhoecongdongonline.vn` | 8 | 8 success | 8 Vietnamese | 8 news; density fallback needed |
| `zhongyibaodian.net` | 8 | 8 success | 8 Han | 7 generic static, 1 unknown |
| `suckhoedoisong.vn` | 8 | 8 success | 8 Vietnamese | 8 news; large extraction and contamination review needed |
| `nhathuoclongchau.com.vn` | 8 | 8 HTTP 403 | Block page, not article language | Cloudflare access restriction; do not bypass |
| `test.pmphai.com` | 8 | 8 success | 8 Han | 8 generic static; retain staging/provenance flag |
| `laodong.vn` | 5 | 5 success | No visible text | 5 JavaScript shells: 177 bytes, one script, zero text/title |
| `medlatec.vn` | 3 | 3 success | 3 Vietnamese | Generic static; article text present |
| `thanhnien.vn` | 3 | 3 success | 3 Vietnamese | 3 news with structured data |
| `vinmec.com` | 3 | 3 success | 3 Vietnamese | Generic article pages with structured data |
| `vov.vn` | 2 | 2 success | 2 Vietnamese | 2 news; CAPTCHA token present but article text remained available |
| `vov2.vov.vn` | 2 | 2 success | 2 Vietnamese | Mixed news/generic structure |
| `tienphong.vn` | 2 | 2 success | 2 Vietnamese | 2 news; density fallback needed |
| `vietnamplus.vn` | 2 | 2 success | 2 Vietnamese | 2 news; density fallback needed |
| `zydcd.com` | 2 | 2 success | 2 Han | 1 generic static, 1 unknown |
| `phuyen.baodaklak.vn` | 1 | 1 success | 1 Vietnamese | Article content present; automated Q&A marker was ambiguous |

Across the 99 successful HTML pages, the automatic taxonomy produced 21 Q&A,
42 generic static, 26 news, 5 JavaScript-heavy, and 5 unknown pages. The eight
403 pages are additionally classified as access-restricted in per-row results.

## 4. DOM structure findings

The probe measured, rather than stored, DOM features:

- 307 `<article>` elements and 28 `<main>` elements across the sample;
- 1,301 headings and 2,581 paragraphs;
- 484 Q&A class/ID markers;
- 140 `<time>` elements and 54 author markers;
- 103 related-content markers and 316 ad/banner markers;
- JSON-LD on multiple Vietnamese news/health sources, including 19
  `NewsArticle` objects; no sampled JSON-LD exposed `articleBody`.

Q&A sources often use repeated question/answer blocks without an `<article>`
element. News sources more often combine `<article>`, timestamps, authors,
related-content blocks, ads, and JSON-LD. Navigation/sidebar/footer removal is
therefore useful, but selecting the first semantically named container is not
reliable across all domains.

## 5. Encoding and Unicode findings

For the 99 successful HTML responses:

| Signal | Count |
|---|---:|
| HTTP declared UTF-8 | 79 |
| HTTP charset missing | 20 |
| Meta declared UTF-8 | 87 |
| Meta declared GB2312 | 7 |
| Meta charset missing | 5 tiny JS shells |
| HTTP client selected UTF-8 | 99 |
| Probe decoder selected UTF-8 | 87 |
| Probe decoder selected GB2312 | 7 |
| Probe decoder selected ASCII | 5 tiny JS shells |

The seven GB2312 pages were all from `cnkang.com`; their HTTP header omitted a
charset while their HTML meta tag declared GB2312. The HTTP client defaulted to
UTF-8, but the probe correctly selected GB2312 from the meta declaration. This
is evidence for meta/detection fallback, not for an unconditional GB18030
fallback.

Decoding produced zero U+FFFD replacement characters and zero corrected
multi-character mojibake signatures. Seven decoded pages changed under NFC
normalization, so downstream text should be normalized to NFC. Page titles and
sampled visible text remained readable after the selected decoding.

## 6. Measured language signals

Among the 99 successful HTML pages:

| Lightweight content signal | Count |
|---|---:|
| Han-script dominant | 60 |
| Vietnamese character/stopword signal | 34 |
| Unknown (empty JS shells) | 5 |

This is a transparent script/character signal, not a general-purpose language
model. The eight Cloudflare block pages and eleven robots-blocked rows are not
included as measured article language. Domain-based language associations
remain separately labelled inference in the manifest.

## 7. Anti-bot and dynamic behavior

- `zysjonline.com`: 11/11 selected rows were explicitly disallowed by the
  cached robots policy and were persisted as blocked without page fetching.
- `nhathuoclongchau.com.vn`: 8/8 returned HTTP 403 with an “Attention
  Required” Cloudflare page. No bypass was attempted.
- `laodong.vn`: 5/5 returned a 177-byte, one-script shell with no title or
  visible text. Static HTTP extraction is insufficient for these sampled URLs.
- Medlatec (3 pages) and VOV (2 pages) contained a visible CAPTCHA token, but
  returned substantial readable article text. The token alone is therefore a
  warning signal, not grounds to discard the page.

No evidence of redirect loops or login walls was observed.

## 8. Generic extraction experiment

Two offline, generic BeautifulSoup approaches were compared on all 99
successful HTML pages:

1. **Semantic container**: removes script/style/nav/footer/aside/form noise,
   then selects `article`, `main`, or a content-like container.
2. **Density scored**: performs the same cleanup, scores generic containers by
   visible text and paragraph count while penalizing link text, and selects the
   best-scoring container.

| Metric | Semantic container | Density scored |
|---|---:|---:|
| Median extracted characters | 452 | 1,961 |
| Mean extracted characters | 5,024.4 | 6,347.4 |
| Empty results | 5 (5.05%) | 5 (5.05%) |
| Title preserved | 94/99 (94.95%) | 94/99 (94.95%) |
| Simple boilerplate markers | 47 | 63 |

The five empty results are the Lao Động JS shells, not extraction failures on
static content. The semantic method was cleaner by the simple marker count,
but severely under-extracted Ask39, Cnkang, Sức Khỏe Cộng Đồng, the test
PMPhai source, Tiền Phong, and VietnamPlus. The density method recovered much
more paragraph text at the cost of modestly higher boilerplate. On 120ask and
A-Hospital the two methods were effectively equivalent. Both methods produced
very large output on Sức Khỏe & Đời Sống, indicating a contamination risk that
simple character counts cannot resolve.

Decision: Phase 3 should use a hybrid candidate strategy. Prefer the semantic
candidate when it is substantial and not materially shorter; otherwise use
the density candidate. Preserve titles explicitly, retain extraction metrics,
and evaluate a mature generic extractor before introducing site-specific CSS
selectors.

## 9. Phase 2C scale-crawl risks

1. Robots exclusions can remove an entire high-volume source and must remain
   visible in status reporting.
2. Cloudflare 403 responses must not be retried aggressively or bypassed.
3. Static requests can return successful but empty JS shells; HTTP 200 is not
   sufficient evidence of extractable content.
4. Domain concentration makes conservative per-domain pacing essential.
5. Missing HTTP charset is material: meta GB2312 changed decoding for seven
   Cnkang pages.
6. Q&A sources need question/answer boundary preservation rather than one flat
   article blob.
7. News pages contain related links, ads, navigation, and large repeated DOM
   regions; extraction quality monitoring is required.
8. The unusual newspaper suffixes are HTML and must not be filtered as files.

## 10. Recommendation and deferred work

`configs/extraction_policy.yaml` records only decisions supported by this
probe. Phase 2C may consume these results when explicitly authorized, but this
phase did not start a scale crawl, browser rendering, raw-body archive,
site-specific selector implementation, chunking, indexing, retrieval, or
submission work.
