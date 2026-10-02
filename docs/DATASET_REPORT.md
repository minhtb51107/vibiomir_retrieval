# DATASET_REPORT.md — Phase 1: Dataset Understanding

> **Status**: COMPLETE
> **Phase**: 1 — Dataset Understanding
> **Script**: `scripts/inspect_dataset.py`
> **Executed**: 2026-10-03
> **Raw data**: NOT modified (read-only inspection only)
> **Network requests**: NONE made

---

## Legend

| Tag | Meaning |
|---|---|
| **[MEASURED]** | Directly computed from real data via script |
| **[OBSERVED]** | Interpretation of measured facts |
| **[HYPOTHESIS]** | Inference or recommendation — not yet verified |

---

## 1. query.parquet

### 1.1 Schema

**[MEASURED]**

| # | Column | Parquet dtype |
|---|---|---|
| 0 | `id` | `int64` |
| 1 | `query` | `string` (UTF-8) |

### 1.2 Row and ID counts

**[MEASURED]**

| Metric | Value |
|---|---|
| Total rows | **1,200** |
| Unique `id` count | **1,200** |
| Duplicate `id` rows | **0** |
| Missing `id` | 0 |
| Missing `query` | 0 |

### 1.3 Empty / whitespace queries

**[MEASURED]**

| Metric | Value |
|---|---|
| NULL query rows | 0 |
| Whitespace-only query rows | 0 |

**[OBSERVED]** All 1,200 queries contain non-empty content.

### 1.4 Query character-length statistics

**[MEASURED]**

| Statistic | Characters |
|---|---|
| min | 18 |
| p25 | 56.0 |
| median | 81.0 |
| mean | 129.5 |
| p75 | 169.2 |
| p95 | 364.0 |
| p99 | 494.1 |
| max | 1,243 |

**[OBSERVED]** Distribution is right-skewed: median 81 chars vs mean 129.5, indicating a tail of long multi-question queries. Shortest: "Vi sao rang bi nut" (18 chars, id=911). Longest: 1,243 chars (id=219), a compound multi-question about COVID vaccination.

### 1.5 Representative examples

**Shortest 3 queries:**

| id | chars | query |
|---|---|---|
| 911 | 18 | "Vi sao rang bi nut" |
| 124 | 21 | "Be dau dau la do dau?" |
| 217 | 27 | "Nguyen nhan lech cam la gi?" |

**Longest 3 queries:**

| id | chars | preview |
|---|---|---|
| 219 | 1,243 | Compound COVID vaccination multi-question |
| 271 | 726 | Pediatric fever with complex symptom list |
| 71 | 646 | Bowel disorder with daily variation description |

### 1.6 Unicode / encoding anomalies

**[MEASURED]**

| Metric | Value |
|---|---|
| Rows with control characters | 0 |
| Rows with U+FFFD replacement char | 0 |
| Rows with non-ASCII characters | **1,200 (100%)** |
| Rows NOT in NFC normalization | **4** |

**[MEASURED]** All 1,200 queries contain non-ASCII characters, none contain U+FFFD, and four are not NFC-normalized. The text appears Vietnamese on inspection; language was not classified programmatically.

**[HYPOTHESIS]** The 4 non-NFC queries likely use NFD (decomposed) Vietnamese diacritics. This can cause retrieval mismatches. Pre-processing must apply `unicodedata.normalize("NFC", text)` to all queries and indexed text.

---

## 2. links_corpus.parquet

### 2.1 Schema

**[MEASURED]**

| # | Column | Parquet dtype |
|---|---|---|
| 0 | `id` | `int64` |
| 1 | `url` | `string` (UTF-8) |

### 2.2 Row counts and ID integrity

**[MEASURED]**

| Metric | Value |
|---|---|
| Total rows | **4,394,718** |
| Unique `id` count | **4,394,718** |
| Duplicate `id` rows | **0** |
| Missing `id` | 0 |
| Missing `url` | 0 |
| ID range min | **583** |
| ID range max | **4,420,561** |
| Expected sequential IDs (max-min+1) | 4,419,979 |
| IDs match sequential range? | **No** |
| Gap positions in sequence | **7** |
| Total missing IDs in range | **25,261** |
| Max missing IDs in one gap | **14,096** |
| Missing IDs per gap (all 7) | 118, 4,098, 109, 14,096, 368, 3,083, 3,389 |

**[OBSERVED]**
- README states 4,420,561 total links, ID range 1-4,420,561. Actual parquet: 4,394,718 rows, starting at ID 583. This is a discrepancy of ~25,843 rows.
- IDs are globally unique with no duplicates. `id` is safe as `doc_id`.
- 7 gap clusters in the ID sequence total 25,261 missing IDs. The largest gap contains 14,096 missing IDs. The reason for the gaps is not established by URL metadata.

### 2.3 URL uniqueness

**[MEASURED]**

| Metric | Value |
|---|---|
| Unique URL count | **4,394,718** |
| Duplicate URL rows | **0** |

**[OBSERVED]** Every row has a distinct URL. No duplicate documents at the URL level.

### 2.4 HTTP vs HTTPS

**[MEASURED]**

| Scheme | Count | % |
|---|---|---|
| HTTPS | 3,030,078 | **68.9%** |
| HTTP | 1,364,640 | **31.1%** |
| Other / malformed | 0 | 0.0% |
| Missing (null) | 0 | 0.0% |

**[HYPOTHESIS]** Many HTTP URLs are Chinese medical Q&A sites (cnkang.com, 120ask.com) that may not support HTTPS. Crawlers should follow HTTP-to-HTTPS redirects when available.

### 2.5 Domain distribution

**[MEASURED]** — Top 30 of **97 total distinct domains**. The final
column is a hostname/TLD-based inference only; page-content language was not
measured in Phase 1.

| Domain | Count | % | Domain-based inference |
|---|---|---|---|
| cnkang.com | 963,438 | 21.9% | Chinese |
| 120ask.com | 918,479 | 20.9% | Chinese |
| familydoctor.com.cn | 447,453 | 10.2% | Chinese |
| ask.39.net | 337,375 | 7.7% | Chinese |
| zysjonline.com | 243,347 | 5.5% | Chinese |
| a-hospital.com | 169,339 | 3.9% | Chinese |
| suckhoecongdongonline.vn | 154,503 | 3.5% | Vietnamese |
| zhongyibaodian.net | 146,700 | 3.3% | Chinese (TCM) |
| suckhoedoisong.vn | 85,823 | 2.0% | Vietnamese |
| nhathuoclongchau.com.vn | 79,598 | 1.8% | Vietnamese (pharmacy) |
| zydcd.com | 78,680 | 1.8% | Chinese |
| thanhnien.vn | 72,111 | 1.6% | Vietnamese (newspaper) |
| wujue.com | 53,341 | 1.2% | Chinese |
| youlai.cn | 46,119 | 1.0% | Chinese |
| laodong.vn | 31,282 | 0.7% | Vietnamese (newspaper) |
| baby.39.net | 27,621 | 0.6% | Chinese |
| phunusuckhoe.giadinhonline.vn | 27,577 | 0.6% | Vietnamese |
| vinmec.com | 26,129 | 0.6% | Vietnamese (hospital) |
| bingli.iiyi.com | 24,990 | 0.6% | Chinese |
| medlatec.vn | 24,762 | 0.6% | Vietnamese (medical lab) |
| qihuangzhishu.com | 23,637 | 0.5% | Chinese (TCM) |
| jbk.39.net | 19,142 | 0.4% | Chinese |
| giadinhonline.vn | 16,729 | 0.4% | Vietnamese |
| baohaiphong.vn | 16,286 | 0.4% | Vietnamese (newspaper) |
| baonghean.vn | 15,941 | 0.4% | Vietnamese (newspaper) |
| vietnamnet.vn | 15,543 | 0.4% | Vietnamese (newspaper) |
| care.39.net | 15,134 | 0.3% | Chinese |
| test.pmphai.com | 15,120 | 0.3% | Unknown (test subdomain) |
| baodanang.vn | 14,500 | 0.3% | Vietnamese (newspaper) |
| tiemchunglongchau.com.vn | 13,281 | 0.3% | Vietnamese (pharmacy) |

**[OBSERVED]**
- The top 2 domains (cnkang.com + 120ask.com) alone = **42.8%** of all URLs (1,881,917 rows).
- The top 6 domains account for approximately **70%** of all corpus URLs. Their hostnames are commonly associated with Chinese-language medical sites, but this is a domain-based inference, not a measurement of page-content language.
- Only 97 distinct domains — extremely concentrated corpus.
- A number of `.vn` and Vietnam-associated hostnames appear in the distribution; no percentage of Vietnamese-language content was measured.
- `test.pmphai.com` (rank 28, 15,120 URLs) is suspicious — a "test." subdomain in a production corpus.

### 2.6 URL extension distribution

**[MEASURED]**

| Extension | Count | % |
|---|---|---|
| `.html` | 2,422,992 | 55.1% |
| `.htm` | 1,117,753 | 25.4% |
| (no extension) | 799,288 | 18.2% |
| `.ldo` | 31,282 | 0.7% |
| `.vov` | 9,784 | 0.2% |
| `.vov2` | 6,899 | 0.2% |
| `.vnp` | 3,695 | 0.1% |
| `.tpo` | 3,021 | 0.1% |
| Other (`.vn`, `.von`, `.0`, `.5`) | 4 | 0.0% |
| **URLs with `.pdf` path extension** | **0** | **0%** |
| **URLs with a known binary-file path extension** | **0** | **0%** |

**[OBSERVED]**
- No PDF-like URL patterns were detected from URL metadata. Real response `Content-Type` is unknown until Phase 2 crawling, so this does not prove that the corpus contains no PDFs or other binary responses.
- `.ldo`, `.vov`, `.vov2`, `.vnp`, and `.tpo` are unusual path suffixes associated with news-site URLs. Their actual response type has not been measured.
- 18.2% of URLs have no file extension — typical for CMS/REST-style article paths.

### 2.7 Query strings, tracking, and fragments

**[MEASURED]**

| Metric | Value |
|---|---|
| URLs with query string `?` | **20,534** |
| URLs with likely tracking parameters (`utm_*`, `fbclid`, `gclid`, `msclkid`) | **0** |
| URLs with fragments `#` | **1,698** |
| URLs containing `.pdf` (case-insensitive substring) | **0** |

**[OBSERVED]**
- Query strings appear in 0.47% of URLs — typically internal article IDs embedded as query params (e.g. `?id=bda58e58-62e5-a99c-...` at bachmai.gov.vn).
- None of the configured common tracking-parameter patterns were found. This does not establish that every URL is canonical or free of site-specific tracking parameters.
- 1,698 fragment URLs (#) should have the fragment stripped before crawling to avoid duplicate fetches.

### 2.8 Sample URLs (actual data)

**[MEASURED]**

```
id=583    http://special.vietnamplus.vn/benh_vien_nhan_ai
id=584    https://bachmai.gov.vn/bai-viet/-nguoi-dan-se-duoc-thu-huong-chat-luong-kham-chua-benh-quoc-te-ngay-
id=585    https://bachmai.gov.vn/bai-viet/-roi-loan-su-thich-ung-va-nhung-he-luy--9826?id=bda58e58-62e5-a99c-f...
id=766442 http://ask.familydoctor.com.cn/q/10420069.html
id=766443 http://ask.familydoctor.com.cn/q/3909357.html
```

---

## 3. ID Integrity Assessment

**[MEASURED]**

| Check | Result |
|---|---|
| IDs are unique in corpus | YES |
| Duplicate ID rows | 0 |
| Missing URLs for any ID | 0 |
| URL-to-ID ratio | 1:1 (bijection) |

**VERDICT**: The `id` column is a safe, unique canonical `doc_id`.

**[HYPOTHESIS]** The 25,261 missing IDs in the 583-4,420,561 range represent URLs removed during corpus curation (broken links, duplicates, out-of-scope pages). No action required.

---

## 4. Corpus Crawling Risk Analysis

### 4.1 Domain concentration

**[OBSERVED]** There are 97 total domains. The top 2 domains account for 42.8% combined, and the top 6 account for approximately 70%.

**[HYPOTHESIS]** HIGH rate-limiting risk for cnkang.com and 120ask.com (~1M URLs each). Requires polite crawling with per-domain delays and robots.txt compliance.

### 4.2 Language and extraction complexity

**[OBSERVED]** Hostname and TLD metadata suggest a mixture of Chinese- and Vietnamese-associated sources, but content language was not measured.

**[HYPOTHESIS]**
- Content extraction and indexing may need both Chinese and Vietnamese text handling; verify this from fetched content in Phase 2.
- Vietnamese query text needs NFC normalization throughout.
- Different content structures per site type (Q&A forums vs hospital articles vs newspapers) may need site-specific or adaptive extractors.

### 4.3 Non-standard URL extensions

**[OBSERVED]** `.ldo`, `.vov`, `.vov2`, `.vnp`, and `.tpo` occur as unusual URL path suffixes on news-associated domains.

**[HYPOTHESIS]** Crawlers must use the HTTP `Content-Type` header, not URL extension, for content-type detection.

### 4.4 PDF / binary file risk

**[MEASURED]** No `.pdf` path extensions or case-insensitive `.pdf` URL substrings were detected.

**[OBSERVED]** This is URL-metadata evidence only. Real response `Content-Type` is unknown until Phase 2 crawling, so Phase 1 cannot conclude that the corpus contains no PDFs or that binary handling is unnecessary.

### 4.5 Dynamic URL patterns

**[MEASURED]** 20,534 URLs contain a query string; examples include UUID-style article IDs. URL metadata alone does not establish whether every such URL is a stable content page.

**[HYPOTHESIS]** Preserve full URL including query string when crawling.

### 4.6 `test.pmphai.com` anomaly

**[OBSERVED]** 15,120 URLs from a "test." subdomain included in the official corpus.

**[HYPOTHESIS]** Check for duplicate/staging content during crawl. May overlap with a production subdomain.

### 4.7 Fragment URLs

**[MEASURED]** 1,698 URLs contain `#` fragments.

**[HYPOTHESIS]** Strip `#fragment` before requesting to avoid duplicate fetches.

---

## 5. README vs Actual Data Discrepancy

**[MEASURED]**

| Metric | README claim | Actual measurement | Delta |
|---|---|---|---|
| Total links | 4,420,561 | **4,394,718** | -25,843 |
| ID range | 1-4,420,561 | **583-4,420,561** | Start differs by 582 |

**[HYPOTHESIS]** README was written before final corpus curation. 25,843 URLs were removed. The parquet file is authoritative.

---

## 6. Ten Most Important Findings

1. **[MEASURED]** query.parquet: 1,200 rows, 0 nulls, 0 duplicates — fully clean.
2. **[MEASURED]** 4 queries are not NFC-normalized — must normalize before indexing.
3. **[MEASURED]** links_corpus.parquet: **4,394,718 rows** (README claims 4,420,561 — 25,843 discrepancy).
4. **[MEASURED]** All IDs unique, zero duplicate URLs — `id` is a safe `doc_id`.
5. **[MEASURED]** ID range starts at 583 (not 1); 7 gaps, 25,261 missing IDs total.
6. **[MEASURED]** Only **97 distinct domains**. The top 2 domains = **42.8%** of corpus; any language association is inferred from domains, not measured from content.
7. **[MEASURED]** 68.9% HTTPS, 31.1% HTTP, zero malformed URLs.
8. **[MEASURED]** Zero PDF-like URL patterns detected; actual response content types remain unknown until crawling.
9. **[MEASURED]** Extensions: 55.1% .html, 25.4% .htm, 18.2% no ext; newspaper-specific extensions present.
10. **[MEASURED]** 20,534 URLs with query strings (0.47%); 1,698 with fragments; 0 matches for the configured likely tracking-parameter patterns.

---

## 7. Phase 2 Risks and Blockers

| Risk | Severity | Notes |
|---|---|---|
| Rate-limiting on cnkang.com / 120ask.com (~1.88M URLs) | HIGH | Polite delays, robots.txt compliance, possible IP rotation |
| Multilingual content handling | HIGH | Domain metadata suggests Chinese- and Vietnamese-associated sources; verify actual language from fetched content |
| `test.pmphai.com` staging content (15,120 URLs) | MEDIUM | Verify not duplicate of production content |
| Fragment URLs (1,698) | LOW | Strip #fragment before requesting |
| Query-string URLs (20,534) | LOW | Preserve the full URL initially; validate canonicalization during crawling |
| NFC normalization mismatch (4 queries) | LOW | Normalize both queries and indexed text consistently |
| README count discrepancy (-25,843 rows) | INFO | Use actual parquet row count, not README statistics |

---

*Report generated from `scripts/inspect_dataset.py` on 2026-10-03.
No network requests were made. Raw data files were not modified.*
