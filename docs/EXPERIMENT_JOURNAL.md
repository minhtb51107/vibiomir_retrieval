# ViBioMIR Experiment Journal

This is an evidence ledger, not a retrospective narrative reconstructed from
memory. Dates, numbers, and conclusions come from tracked reports, artifacts,
Git metadata, or organizer values supplied by the project owner. `UNKNOWN`
means the repository does not support a more precise claim.

## Phase 0 — Project harness

Date: UNKNOWN
Phase / commit: Phase 0 / UNKNOWN
Question: Could the project be organized into a reproducible phased workflow?
Why we tried it: Later dataset, crawl, and retrieval experiments required a stable repository structure.
Hypothesis: A documented harness would make phased work and Git checkpoints auditable.
What we changed: The current repository contains source, scripts, configs, tests, docs, artifacts, and ignore rules.
What stayed fixed: No reliable Phase 0 change record is visible in current Git history.
Result: The harness exists, but its exact creation date and commit cannot be recovered from tracked evidence.
What failed / surprised us: The visible history begins with Phase 1 rather than a harness commit.
What we learned: Missing history must be marked unknown, not reconstructed.
Decision: Treat Phase 1 as the first evidence-backed checkpoint.
Next question: What is the actual dataset shape?
Cost/resources: UNKNOWN.
Evidence: Current repository layout; `AGENTS.md`; first visible commit `24ff790`.
Commit: UNKNOWN.

## Phase 1 — Dataset understanding

Date: 2026-10-03
Phase / commit: Phase 1 / `24ff790`
Question: What are the true query and corpus dimensions and risks?
Why we tried it: Architecture and capacity decisions required measured data rather than README assumptions.
Hypothesis: A complete local scan could reproduce all important statistics quickly.
What we changed: Added reproducible Parquet inspection and a machine-readable dataset report.
What stayed fixed: Raw Parquet files remained unchanged; no network access occurred.
Result: 1,200 queries, 4,394,718 URLs, 97 domains; the top two domains hold 42.8%.
What failed / surprised us: Initial row-wise parsing was too slow; URL metadata could not reveal real content type or language.
What we learned: Vectorized/simple string operations were sufficient, and metadata inferences must be labelled.
Decision: Build a bounded, domain-aware crawler rather than assume homogeneous sources.
Next question: Can acquisition be resumable and polite at this scale?
Cost/resources: One full local corpus scan; no network.
Evidence: `docs/DATASET_REPORT.md`, `artifacts/dataset_stats.json`.
Commit: `24ff790`.

## Phase 2 — Crawler, source probing, and readiness

Date: 2026-10-03
Phase / commit: Phases 2A–2C / `b0354fc`, `ccb79bf`, `aba8455`
Question: Can heterogeneous sources be acquired safely, resumed, and measured?
Why we tried it: Millions of URLs and domain concentration made naive crawling unsafe.
Hypothesis: Async fetching with domain limits, robots policy, SQLite checkpoints, and bounded pilots would expose production constraints.
What we changed: Added crawler core, source probes, retry/status taxonomy, robots caching, telemetry, and full-crawl gates.
What stayed fixed: No unrestricted crawl and no anti-bot bypass.
Result: The 1,225-URL readiness benchmark was resumable; Phase 2B identified robots blocks, Cloudflare responses, JS shells, and mixed encodings.
What failed / surprised us: HTTP 200 did not imply useful content; some sources were blocked or shell-only.
What we learned: Acquisition status and extraction usability must be tracked separately.
Decision: Keep bodies outside metadata SQLite and proceed with bounded extraction.
Next question: Can bytes be decoded, cleaned, and chunked with provenance?
Cost/resources: 118-source-probe URLs and a 1,225-URL readiness benchmark.
Evidence: `docs/CRAWLER_ARCHITECTURE.md`, `docs/SOURCE_PROBE_REPORT.md`, `docs/PRODUCTION_CRAWL_PLAN.md`.
Commit: `b0354fc`, `ccb79bf`, `aba8455`.

## Phase 3 — Cleaning and chunking

Date: 2026-10-03
Phase / commit: Phase 3 / `d3756d2`
Question: Can fetched bytes become traceable multilingual retrieval chunks?
Why we tried it: Retrieval requires clean Unicode text while preserving source evidence.
Hypothesis: Hybrid semantic/density extraction plus paragraph-aware chunking would handle the bounded corpus.
What we changed: Added compressed shards, robust decoding, structured Q&A, explicit unusable states, and deterministic offsets.
What stayed fixed: One corpus row remained one `doc_id`; chunks were never generated or rewritten.
Result: Processed the bounded 1,225-row/1,148-body pilot and compared 256/32, 512/64, and 768/96 chunking.
What failed / surprised us: Proxy token counts were insufficient for BGE-M3 limits; legacy charset evidence had to be respected.
What we learned: Real-tokenizer validation was mandatory before indexing.
Decision: Carry 512/64 forward, subject to BGE-M3 revalidation.
Next question: Can the actual dense model index the pilot locally?
Cost/resources: Bounded local CPU processing; no new crawl.
Evidence: `docs/CLEANING_AND_CHUNKING.md`.
Commit: `d3756d2`.

## Phase 4 — Dense retrieval baseline

Date: 2026-10-03
Phase / commit: Phase 4 / `c596b3d`
Question: Can BGE-M3 provide an exact multilingual dense baseline on local hardware?
Why we tried it: Vietnamese queries may need cross-lingual access to Vietnamese and Chinese documents.
Hypothesis: Normalized BGE-M3 vectors with `IndexFlatIP` would provide a correctness-first baseline.
What we changed: Validated/regenerated chunks, embedded them on the RTX 3050, and persisted exact FAISS mappings.
What stayed fixed: Model/revision and source-derived text; no sparse retrieval or evaluation.
Result: 5,643 validated chunks, zero over-limit chunks, 1,200-query retrieval, deterministic chunk/document mappings.
What failed / surprised us: 1,152 original chunks exceeded 512 tokens; initial CPU embedding was impractical.
What we learned: GPU batching and resumable artifacts were necessary even for the pilot.
Decision: Use best-chunk document aggregation as the next baseline input.
Next question: What complementary lexical signal exists?
Cost/resources: Local BGE-M3 inference; model already cached.
Evidence: `docs/DENSE_RETRIEVAL_BASELINE.md`, `artifacts/phase4_dense/`.
Commit: `c596b3d`.

## Phase 6 — Hybrid retrieval

Date: 2026-10-03
Phase / commit: Phase 6 / `d5d06aa`
Question: Can deterministic lexical retrieval complement dense candidates?
Why we tried it: Dense and exact-token matches expose different candidate regions.
Hypothesis: Unicode words plus Han unigrams/bigrams and BM25, fused by RRF, would provide a modular candidate pool.
What we changed: Added SQLite BM25, multilingual tokenization, RRF, and overlap diagnostics.
What stayed fixed: Dense artifacts and the 5,643-chunk pilot; no relevance claims.
Result: Dense, sparse, and hybrid rankings were produced for all 1,200 queries.
What failed / surprised us: Exact-text duplicates affected 16.46% of chunks, and method overlap was limited.
What we learned: Candidate diversity and duplicate controls matter before reranking.
Decision: Pass all three sources into a bounded cross-encoder pool.
Next question: Can a local multilingual reranker reorder the union safely?
Cost/resources: Offline CPU BM25 and reuse of dense results.
Evidence: `docs/HYBRID_RETRIEVAL_BASELINE.md`.
Commit: `d5d06aa`.

## Phase 7 — Multilingual reranking

Date: 2026-10-04
Phase / commit: Phase 7 / `d363376`
Question: Can a multilingual cross-encoder rerank the candidate union on 4 GB VRAM?
Why we tried it: Candidate retrieval alone lacked query–chunk joint scoring.
Hypothesis: `BAAI/bge-reranker-v2-m3` could score bounded top-100 pools at batch size 2.
What we changed: Added unified pools, isolated GPU scoring, SQLite checkpoints, selection controls, and CPU finalization.
What stayed fixed: Candidate sources, exact chunk text, and model revision.
Result: All 120,000 pairs were scored and rankings at depths 20/50/100 were produced.
What failed / surprised us: Batch 4 OOMed; batch 2 was the safe production setting.
What we learned: GPU scoring must be isolated from large host-side dataframes.
Decision: Preserve batch 2 and checkpoint signatures for later experiments.
Next question: Can the pipeline scale beyond the pilot?
Cost/resources: RTX 3050 FP16 inference; 120,000 pairs.
Evidence: `docs/RERANKING_BASELINE.md`.
Commit: `d363376`.

## Phase 8 — Full-corpus scaling audit

Date: 2026-10-04
Phase / commit: Phase 8 / `a075202`
Question: Can the 4.39M-URL pipeline fit this laptop?
Why we tried it: A production plan needed explicit time, disk, and resume gates before any large run.
Hypothesis: Bounded streaming might make exhaustive acquisition feasible locally.
What we changed: Added production manifests, disk guards, bounded stages, and capacity projections.
What stayed fixed: No unrestricted crawl was started.
Result: Expected retained storage was about 428 GB, transient peak about 575 GB, with a roughly 28.2-day crawl estimate.
What failed / surprised us: Local free space and exact-index scale were insufficient despite resumable architecture.
What we learned: Engineering readiness does not make a resource plan feasible.
Decision: Seek a defensible candidate-reduction strategy.
Next question: Can URL metadata reduce the corpus without destroying candidate coverage?
Cost/resources: Bounded 5K-scale validation and local capacity analysis.
Evidence: `docs/FULL_CORPUS_SCALING.md`, `artifacts/phase8_scaling/`.
Commit: `a075202`.

## Phase 8C — Query-conditioned URL reduction

Date: 2026-10-05
Phase / commit: Phase 8C / `34d7f2b`
Question: Can local URL/domain/path text reduce 4.39M URLs cheaply?
Why we tried it: Exhaustive acquisition exceeded local resources.
Hypothesis: Multilingual URL BM25 plus domain fallback could retain useful candidates.
What we changed: Indexed all URL metadata and evaluated depths through 1,000/query.
What stayed fixed: Zero network requests; no candidate crawl.
Result: A 415,767-URL set was tractable but retained only 0.2534% of reranked pilot candidates and excluded Han/opaque sources.
What failed / surprised us: Good numerical reduction concealed destructive selection bias.
What we learned: URL metadata alone is too weak for cross-lingual biomedical discovery.
Decision: Reject the proposed 10K crawl and do not loosen the selector.
Next question: Can a content-indexed external search engine discover official URLs?
Cost/resources: 4.39M-row local indexing; zero network.
Evidence: `docs/URL_CANDIDATE_REDUCTION.md`.
Commit: `34d7f2b`.

## Phase 9 — First valid organizer submission

Date: 2026-10-05 (artifact commit); organizer result date UNKNOWN
Phase / commit: Phase 9 / `abf2bf7`
Question: What real organizer signal does the pilot retrieval system receive?
Why we tried it: No trustworthy local ground truth existed.
Hypothesis: Strict provenance and schema validation could at least obtain a valid external score.
What we changed: Added deterministic submission generation, validation, manifests, and hashes.
What stayed fixed: Existing pilot rankings; no fabricated labels or new crawl.
Result: First organizer result was `FINAL_SCORE=0.0001`, with document and chunk recall both `0.0001`.
What failed / surprised us: A technically valid retrieval pipeline had essentially no coverage.
What we learned: Corpus coverage, not file format or ranking depth, dominated performance.
Decision: Calibrate cheap output choices before acquiring more data.
Next question: Do depth, expansion, or retrieval source alter the displayed metrics?
Cost/resources: Offline generation; manual organizer submission.
Evidence: `docs/SUBMISSION_GENERATION.md`, `docs/LEADERBOARD_CALIBRATION.md`.
Commit: `abf2bf7`.

## Phase 10A — Leaderboard calibration

Date: 2026-10-05
Phase / commit: Phase 10A / `3c228ff`
Question: Can output depth, source-verbatim expansion, or candidate strategy repair the baseline?
Why we tried it: Controlled organizer experiments were the only trustworthy relevance evidence.
Hypothesis: One-variable variants could identify a cheap submission-side gain.
What we changed: Generated nine deterministic variants E1–E9.
What stayed fixed: Persisted retrieval artifacts and exact source provenance.
Result: Project-owner evidence records that more docs/chunks did not improve displayed recall; expanded chunks improved chunk precision; reranking remained slightly stronger.
What failed / surprised us: Increasing output volume did not address missing relevant documents.
What we learned: Submission formatting cannot compensate for corpus coverage.
Decision: Investigate discovery rather than deeper output.
Next question: Can public search map query results to official corpus IDs?
Cost/resources: Offline generation and organizer submission budget; exact submission dates/costs unavailable.
Evidence: `docs/LEADERBOARD_CALIBRATION.md`; project-owner results recorded in later phase prompts.
Commit: `3c228ff`.

## Phase 10B0 — Public-search discovery

Date: 2026-10-05
Phase / commit: Phase 10B0 / `86c1462`
Question: Can Bing RSS act as a cheap content-indexed discovery layer?
Why we tried it: Search engines might know page content even when URL slugs are weak.
Hypothesis: Original/concise Vietnamese queries would map a manageable official candidate set.
What we changed: Ran cached sequential search for 300 stratified queries with exact/normalized mapping.
What stayed fixed: No candidate pages were crawled.
Result: 5,535 rows mapped to only 21 unique official docs; 39/300 queries mapped anything; five domains appeared.
What failed / surprised us: 73.08% of mapped rows came from one domain and pilot overlap was zero.
What we learned: General search visibility is strongly biased and canonicalization was not the main problem.
Decision: Reject 1,200-query scaling; inspect first-party source mechanisms.
Next question: Can site-native discovery recover major source coverage?
Cost/resources: 600 cached Stage B requests; no content crawl.
Evidence: `docs/SEARCH_DISCOVERY_FEASIBILITY.md`, `artifacts/phase10b0_search/`.
Commit: `86c1462`.

## Phase 10B1 — Site-native discovery

Date: 2026-10-05
Phase / commit: Phase 10B1 / `b66c5b4`
Question: Can official sites expose query-sensitive, mappable discovery?
Why we tried it: Domain-targeted search could avoid general-engine domain bias.
Hypothesis: A small set of source adapters might cover a meaningful corpus share.
What we changed: Inventoried 97 domains and preflighted representative native search/sitemap/category mechanisms.
What stayed fixed: G0 was declared before query-scale work; no bypass or candidate crawl.
Result: Only a-hospital search passed; usable weighted coverage was 3.853%, below the 10% gate.
What failed / surprised us: Several visible search forms were query-insensitive or returned URLs absent from the official corpus.
What we learned: Source discovery maps are useful even when query search is not.
Decision: Stop before 120/300 queries and use direct bounded acquisition benchmarks.
Next question: Which major sources yield usable text per unit cost?
Cost/resources: A few bounded probes per selected domain.
Evidence: `docs/SITE_NATIVE_DISCOVERY.md`, `docs/OFFICIAL_SOURCE_DISCOVERY_MAP.md`.
Commit: `b66c5b4`.

## Phase 10B2 — Source acquisition benchmark

Date: 2026-10-05
Phase / commit: Phase 10B2 / `603c22a`
Question: Which source is operationally safest to scale?
Why we tried it: Discovery failed, so crawl budget needed source-level evidence.
Hypothesis: Bounded 200/1,000-row checkpoints would separate accessible, usable sources.
What we changed: Benchmarked six sources using the existing crawler/extractor/chunker.
What stayed fixed: Maximum 1,000/source and early-stop gates; no embeddings.
Result: S4 delivered 99.8% usable documents at 59.1 usable docs/minute; 120ask and ask.39 stopped early under the long-content gate.
What failed / surprised us: Some extraction-success pages were meaningful short Q&A but failed the acquisition length gate.
What we learned: Operational usability and relevance value require separate tests.
Decision: Run a source-only relevance probe before scaling.
Next question: Does any bounded source sample change organizer metrics?
Cost/resources: 4,400 sampled rows, including 2,121 new fetches.
Evidence: `docs/SOURCE_ACQUISITION_BENCHMARK.md`, `artifacts/phase10b2_acquisition/`.
Commit: `603c22a`.

## Phase 10B3 — Source relevance probe and S4 discovery

Date: 2026-10-06 (artifact commit); organizer result date UNKNOWN
Phase / commit: Phase 10B3 / `fa9128b`
Question: Which already-acquired source sample has organizer value?
Why we tried it: Acquisition rate alone says nothing about relevance.
Hypothesis: Holding retrieval/reranking/submission policy fixed would isolate source signal.
What we changed: Added each source independently, reused embeddings/scores, reranked, and generated fixed-policy submissions.
What stayed fixed: Queries, models, chunking, candidate depths, reranker, and output policy.
Result: S4 improved final score from `0.0001` to `0.0004`, document recall to `0.0003`, and chunk recall to `0.0003`; S2 reached `0.0002`.
What failed / surprised us: S3 and S5 changed 1,188 and 1,102 local query rankings respectively but produced no organizer gain.
What we learned: Local output movement is not relevance; S4 is the first clear multi-metric source signal.
Decision: Investigate whether S4 can be targeted before spending on S4-5K.
Next question: We know the neighborhood is good, but can we identify promising houses before knocking on more doors?
Cost/resources: 19,238 new chunk embeddings; 90,349 deduplicated novel reranker pairs; no new crawl.
Evidence: `docs/SOURCE_RELEVANCE_PROBE.md`, `docs/leaderboard_history.csv`.
Commit: `fa9128b`.

## Phase 10C0 — S4 targeting tournament

Date: 2026-10-06
Phase / commit: Phase 10C0 / this phase checkpoint
Question: We know the neighborhood is good, but can we identify promising houses before knocking on more doors?
Why we tried it: S4-1K produced the first clear source gain, but crawling another 4K without a cheap targeting check would spend time blindly.
Hypothesis: URL lexical evidence, path structure, held-out seed similarity, or bounded search could retain most S4 proxy positives inside 20K URLs.
What we changed: Compared five predeclared arms at common 5K/10K/20K/40K budgets; Method C used five-fold held-out evaluation.
What stayed fixed: Existing 154,503 URLs, 1,200 queries, S4-1K proxy outputs; no page crawl, GPU, model download, or submission.
Result: URL lexical retained 24.13% at 20K (1.86× random); held-out seed expansion retained 15.24% (1.18×); path categories did not exist; external search mapped 0/481 rows; native search was already unmappable.
What failed / surprised us: Nearly every S4 URL had an informative Vietnamese slug, yet lexical aggregation still missed over three quarters of proxies at 20K. Seed similarity was barely better than random.
What we learned: Within this source, attractive metadata signals are not reliable enough for aggressive acquisition cuts. Proxy discipline also prevented all-seed leakage from creating a false success.
Decision: Do not create targeted manifests or an ensemble. Recommend the next 4,000 new S4 URLs as a broad deterministic sample excluding the existing 1,000.
Next question: Does a broad S4-5K corpus reproduce or extend the organizer gain?
Cost/resources: About 216 seconds local preprocessing/ranking plus 50 cached sequential Bing RSS requests; zero candidate-page fetches.
Evidence: `docs/S4_TARGETING_TOURNAMENT.md`, `artifacts/phase10c0_s4_targeting/`.
Commit: this phase checkpoint (`feat: evaluate s4 targeting strategies`).
