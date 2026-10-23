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

## Phase 10C1 — Broad S4 scaling from 1K to 5K

Date: 2026-10-06 (organizer-confirmed result supplied by the project owner)
Phase / commit: Phase 10C1 / uncommitted current worktree
Question: Does broad deterministic acquisition within S4 continue the organizer signal observed at S4-1K?
Why we tried it: The S4-1K source probe produced the first clear multi-metric gain, while the Phase 10C0 targeting tournament found no trustworthy metadata selector.
Hypothesis: Expanding the same source from 1,000 to 5,000 sampled official document IDs, while holding retrieval, reranking, and submission policy fixed, could improve corpus coverage.
What we changed: Added 4,000 disjoint official `suckhoecongdongonline.vn` IDs to the existing S4 sample and rebuilt only the required downstream source-scale artifacts.
What stayed fixed: The 1,200 queries, BGE-M3 embeddings, BM25/RRF candidate settings, pinned multilingual reranker, top-10 document policy, top-20 chunks, and approximately 1,024-token source-verbatim expansion.
Result: The organizer reported `FINAL_SCORE=0.0014`, `DOCS_F2MACRO=0.0014`, `CHUNKS_F2MACRO=0.0014`, `DOCS_PRECISION=0.0071`, `DOCS_RECALL=0.0012`, `CHUNKS_PRECISION=0.0071`, and `CHUNKS_RECALL=0.0012`. The prior S4-1K result was final `0.0004`, document F2 `0.0003`, chunk F2 `0.0004`, document precision/recall `0.0019/0.0003`, and chunk precision/recall `0.0031/0.0003`.
What failed / surprised us: Local reranking was operationally fragile on the 6 GiB RAM / 4 GiB VRAM laptop: 5,000-pair workers suffered paging collapse, while smaller isolated workers were stable. The organizer gain was materially larger than the earlier bounded S4 signal, but its scaling curve remains unknown.
What we learned: This is organizer-confirmed evidence that scaling S4 from 1K to 5K materially improved both precision and recall. It is not evidence that S4 contains most gold documents, and it does not establish that improvement will continue linearly to 20K.
Decision: Treat deeper S4 acquisition as justified for a future bounded experiment, but retain explicit resource gates and do not start S4-20K automatically.
Next question: Does another bounded S4 increment preserve the organizer gain under the same fixed retrieval policy, and what execution architecture can run it unattended and safely?
Cost/resources: 4,000 new S4 fetch/extract/chunk records, 18,594 newly embedded chunks, and 52,900 newly scored C1 query–chunk pairs; exact electricity cost was not measured.
Evidence: `artifacts/phase10c1_three_way/`, `artifacts/phase10c1_early_release/C1_early_release.json`, and `submissions/phase10c1_C1_S4_5k.zip`; organizer metrics supplied by the project owner.
Commit: pending review; no commit requested yet.

## Phase 10E — G1A focused-scaling preflight

Date: 2026-10-09
Phase / commit: Phase 10E G1A ~10K / pending checkpoint
Question: Can the corrected depth-1000 G1A corpus be expanded to 5,000 Medlatec IDs, all 10 v.familydoctor IDs, and 5,000 Vinmec IDs while preserving the validated retrieval contract and local disk safety?
Why we tried it: Organizer-confirmed G1A depth1000 reached `FINAL_SCORE=0.0062` from only 2,005 usable documents, the highest observed organizer yield per document among the tested priority trios.
Hypothesis: Increasing only G1A source depth may preserve its high relevant-document density and produce the next useful scaling point.
What we changed: Added a parameterized focused-scaling configuration and fail-fast preflight for G1A. The preflight validates the persisted `m=8` calibration, model and scorer contract, prior corpus/cache inputs, output writability, control replay input, and the 20 GiB disk floor before acquisition.
What stayed fixed: Queries, baseline corpus, dense/BM25/RRF policy, candidate cap `m=8`, pinned models, reranker FP16/batch-2/max-length-512 policy, top-10 documents, top-20 chunks, provenance, and deterministic tie-breaking.
Result: Durable counts were 996 successful/usable Medlatec documents, 10 v.familydoctor documents, and 999 Vinmec documents. Reaching official target depths `5,000/10/5,000` requires exactly `4,004/0/4,001` new successful official IDs, or 8,005 total. Preflight dependencies and scientific contract passed, but the disk gate failed: 20.37 GiB free minus a conservative 3.91 GiB retained-growth projection would leave 16.46 GiB, below the active 20 GiB floor. No crawl was launched.
What failed / surprised us: The repository had only about 0.37 GiB headroom above the safety floor before acquisition, despite the experiment itself being bounded and operationally modest.
What we learned: Disk capacity must be a launch gate rather than a runtime incident. A correct scientific plan is not authorization to consume the safety reserve.
Decision: Stop at `NEEDS_AGENT`; preserve the exact manifest and implementation checkpoint, and do not launch G1A until at least the projected 3.91 GiB plus safety margin is safely reclaimed or another approved storage plan is provided.
Resolution: Later on 2026-10-09, free space increased to 30.11 GiB. The unchanged 8,005-ID manifest projected 3.91 GiB retained growth and 26.20 GiB remaining, so the same preflight passed without changing the experiment contract or rebuilding prior artifacts. This authorizes the prepared unattended G1A run; it is not an experiment result.
Next question: Which regenerable local artifacts can be removed safely without violating preservation policy and while keeping at least 20 GiB free after the G1A run?
Cost/resources: No network requests and no model inference. Manifest construction scanned only local metadata. Projected concurrent crawl time is about 68 minutes; conservative end-to-end projection is about 2.88 hours once the disk gate is satisfied.
Evidence: `artifacts/source_census/phase10e_g1a_10010_manifest.json` and `artifacts/source_census/phase10e_g1a_10010/preflight.json`.
Commit: pending.

## Phase 10E — G1A embedding native-crash recovery

Date: 2026-10-09
Phase / commit: Phase 10E G1A ~10K embedding recovery / pending checkpoint
Question: Could the completed G1A acquisition/extraction/chunking work be preserved while making BGE-M3 embedding safe on the 5.69 GiB host?
Why we tried it: The original embedding process terminated with Windows access violation `0xC0000005` after all 8,005 IDs, 7,996 successful extractions, and 105,050 new chunks were already durable.
Hypothesis: The model contract was valid, but co-locating all 131,166 text-heavy chunk rows, the 93,433-ID prior lookup, output memmap, and CUDA model in one process amplified host-memory commitment enough to trigger a native PyTorch failure.
What we changed: Replaced whole-corpus materialization with isolated CPU preparation and a fresh GPU streaming process. CPU preparation copies exact old vectors to a new recovery memmap, records per-row completion in a durable byte map, and exits. GPU embedding reads 128-row Parquet batches, retains no corpus-wide text list, uses no DataLoader workers or prefetching, flushes vectors before completion bits, and checkpoints after each batch. At the EMBEDDINGS boundary, the PyArrow/NumPy-heavy orchestrator now replaces itself with a standard-library-only continuation parent before model load; recognized native startup failures receive bounded fresh-process retries without repeating completed rows. The original `.f32.tmp` remains untouched as forensic evidence.
What stayed fixed: BAAI/bge-m3 revision `5617a9f61b028005a4858fdac845db406aefb181`, tokenizer, 512-token limit, FP16 CUDA, inference batch size 4, normalization, output dimension/dtype/order, chunk corpus, and all downstream retrieval policy.
Result: Windows Event 1000 identified `python.exe` faulting in PyTorch `c10.dll` at offset `0x6ce14`. The old temporary file had the exact expected 537,255,936-byte shape, 26,116 bit-exact reused rows, and zero nonzero new rows; without a row checkpoint it was not append-safe. A 2,048-new-row streaming profile completed with 16 checkpoints, peak working set 614.62 MiB, peak private commitment 3,925.52 MiB, only 2.18 MiB second-half private growth, and projected steady peak 3,927.70 MiB. A separate 1,024-row replay against valid stored embeddings had 920 bit-identical rows, maximum absolute difference `0.000732421875`, mean difference `4.45e-06`, and minimum cosine `0.999512`, passing the measured numerical-equivalence gate. The first production resume then reproduced the native crash before its first row because it still mapped the full 537 MiB recovery file before loading `c10.dll`; moving that mapping after model load was the final necessary isolation step. A real 1,024-row transactional production window subsequently completed, preserved those rows, and stabilized at 3,961.63 MiB private commitment with 1,186 MiB GPU reservation.
What failed / surprised us: The original output file was fully preallocated and therefore looked complete by size even though no new embeddings had been written. File size alone was not a valid checkpoint. A first streaming resume also failed because the otherwise isolated GPU child still had both a 537 MiB mapping and a heavy imported Python parent alive during model initialization; bounded direct runs exposed that difference.
What we learned: Large embedding jobs require explicit row-completion state and process isolation. Model loading must never share a process with corpus-wide Python text objects, and native crash recovery must use completion keys rather than apparent file size.
Decision: Preserve acquisition/extraction/chunking and all 26,116 exact prior embeddings. Resume only the 105,050 missing rows through the streaming checkpoint path, then continue the unchanged experiment if the live memory gate remains healthy.
Next question: Does the focused G1A corpus pass the complete retrieval, reranking, replay, provenance, and pre-submission scientific gates?
Cost/resources: Two bounded profiles totaling 3,072 rows; no acquisition or chunking repeated. Measured new-row embedding throughput was roughly 93 rows/s after model load.
Evidence: `artifacts/incidents/20261009_165945_phase10e_g1a_10010`, `artifacts/source_census/phase10e_g1a_10010/embedding_memory_profile/profile.json`, `artifacts/source_census/phase10e_g1a_10010/embedding_equivalence.json`, and the streaming checkpoint under `data/source_census/phase10e_g1a_10010/round1/`.
Commit: pending.

## Phase 10E organizer checkpoint — prioritize G1A focused scaling

Date: 2026-10-09
Phase / commit: Phase 10E organizer checkpoint / pending checkpoint
Question: After corrected depth-1000 scoring and the first focused G5A expansion, which source trio should receive the next bounded acquisition budget?
Why we tried it: Source census isolated G1A, G5A, and G6B, but their organizer-valid scaling results were not yet all available. The next acquisition decision required organizer evidence rather than local ranking movement.
Hypothesis: Source depth matters, but relevant-document density differs enough across source families that the highest-yield trio may deserve the next checkpoint even when another family has the highest absolute score.
What we changed: No retrieval experiment changed in this checkpoint. We recorded three organizer-confirmed results and selected the next predeclared corpus-depth experiment.
What stayed fixed: Organizer metrics are recorded verbatim; corrupted depth-1000 submissions remain invalid and excluded. No source-level attribution within a trio is claimed.
Result: Corrected G1A depth1000 reached FINAL `0.0062` at exactly 2,005 usable documents and 26,116 chunks (Medlatec 996/19,559; v.familydoctor 10/22; Vinmec 999/6,535). Corrected G6B depth1000 reached FINAL `0.0045` at 2,910 usable documents. G5A at 11,200 searchable documents and 92,768 chunks reached FINAL `0.0085`, up from corrected G5A depth1000 FINAL `0.0037` at 2,997 usable documents. Full organizer metrics are recorded in `docs/leaderboard_history.csv`.
What failed / surprised us: Absolute corpus size alone did not explain organizer yield. G1A approached the much larger G5A score with fewer than one fifth as many documents, while G6B also exceeded corrected G5A at roughly the same depth.
What we learned: Source depth is important, but source quality or gold density differs materially. G1A is the highest-yield corpus per document among the tested priority trios; this does not prove that it is globally best or that its yield will persist at greater depth. G5A remains a proven scalable family, and G6B remains a justified reserve track.
Decision: Run one G1A focused checkpoint with Medlatec and Vinmec targeted to 5,000 official IDs each and all 10 v.familydoctor IDs. Do not start G5A 21.2K or G6B 15K.
Next question: Does G1A retain its high organizer yield when Medlatec and Vinmec scale from about 1K to 5K official IDs each?
Cost/resources: Documentation and registry update only at this boundary; the G1A delta and operational projections must be computed from durable artifacts before acquisition.
Evidence: Organizer metrics supplied by the project owner; `artifacts/source_census/depth1000_fixed_report.json`; `artifacts/validation/phase10d_depth1000_fixed_pre_submit_audit.json`; `artifacts/source_census/phase10e_g5a_11200_report.json`; and `artifacts/validation/phase10e_g5a_11200_pre_submit_audit.json`.
Commit: pending Git checkpoint.

## Phase 10D Round 3 — Shallow trio baselines and depth-scaling decision

Date: 2026-10-08 (organizer-confirmed results supplied by the project owner)
Phase / commit: Phase 10D Round 3 and depth-1000 preparation / uncommitted current worktree
Question: Do the surviving source trios warrant further splitting, or must source acquisition depth be measured before eliminating any of their nine members?
Why we tried it: Round 2 isolated three six-source survivor regions, and Round 3 tested their predeclared sibling trios at the same shallow source depth. The resulting scores provide a controlled shallow baseline, but not individual-source evidence.
Hypothesis: A roughly one-order-of-magnitude source-depth increase, from about 100 to up to 1,000 official IDs per source, is more likely than a 100-to-300 step to produce a measurable first saturation point at the organizer's displayed resolution.
What we changed: Recorded the organizer-confirmed G1A, G5A, and G6B shallow results and replaced the planned source-splitting/depth-300 path with three fixed-composition depth-1000 experiments. The acquisition manifest is nested, deterministic, uses only official IDs, and requests only IDs not already attempted.
What stayed fixed: The 1,200 queries, pilot/control corpus, retrieval architecture, per-source candidate cap, BGE-M3 embedder, pinned multilingual reranker, top-10 documents, top-20 chunks, provenance policy, and deterministic tie-breaking.
Result: Shallow organizer FINAL scores were G1A `0.0008`, G5A `0.0009`, and G6B `0.0008`; full F2, precision, and recall values are recorded in `docs/leaderboard_history.csv`. Preflight found 6,300 incremental official IDs: 900 for each of seven sources, none for `v.familydoctor.com.cn` (all 10 official IDs already available), and none for `suckhoedoisong.vn` because its prior C3 acquisition already exceeds the 1,000-ID target. No depth-1000 leaderboard result exists yet.
What failed / surprised us: The shallow trio results do not justify attributing quality to individual sources, and `v.familydoctor.com.cn` has only 10 official corpus rows. A depth-300 step could consume acquisition and leaderboard budget without separating true saturation from display rounding.
What we learned: All nine sources must remain active for the first depth-scaling point. Existing acquisitions can materially reduce new work when reconciled by official ID, but the experiment corpus must still be capped deterministically to the declared per-source target.
Decision: Skip depth 300. Acquire only the 6,300 missing official IDs needed to reach 1,000 attempted IDs for seven sources, retain all 10 IDs for `v.familydoctor.com.cn`, and use a deterministic 1,000-ID subset of the already acquired `suckhoedoisong.vn` population. Stop at three validated ZIPs and wait for organizer evidence before considering depth 5,000.
Next question: Relative to their shallow baselines, do G1A, G5A, or G6B materially improve when source depth rises to up to 1,000 official IDs?
Cost/resources: Preflight projects about 1.59 GiB conservative retained growth and about 5.19 hours end-to-end; these are operational estimates, not completed-run measurements.
Evidence: `artifacts/source_census/depth1000_manifest.json`, `configs/source_census_depth1000.yaml`, existing Round-1/C3 crawl and processed artifacts, and organizer metrics supplied by the project owner.
Commit: pending review; this run was explicitly launched without commit or push.

### Depth-1000 recovery incident — cached-score validation

Date: 2026-10-08
Phase / commit: Phase 10D depth-1000 execution / uncommitted current worktree
Question: Why did the completed depth-1000 run stop on entry to RANKINGS?
Why we tried it: The incident reported a complete 86,400-row score cache but the cached-subset validator still raised an incomplete-cache exception.
Hypothesis: The validator encoded the old Round-1 cache size rather than validating general completeness invariants.
What we changed: Replaced exact whole-dictionary equality against 582,000 rows with explicit integrity, remaining, done/total, and durable candidate-key-count checks. Added a RANKINGS-only recovery entry point.
What stayed fixed: All acquisition, extraction, chunking, embeddings, candidate generation, reranker scores, retrieval semantics, group membership, and submission policy.
Result: Independent SQLite and candidate-part reconciliation found 86,400 expected unique query/chunk keys, 86,400 stored and scored keys, zero missing or unexpected keys, zero duplicate keys, finite scores, valid inference timings, and `PRAGMA integrity_check=ok`.
What failed / surprised us: A phase-specific literal (`582000`) survived inside generic cached-subset validation and falsely rejected a smaller but complete cache.
What we learned: Cache validity must be schema-tolerant and derived from current durable candidates, never from a historical experiment's absolute row count.
Decision: Resume only at RANKINGS; do not repeat any model inference or earlier pipeline stage.
Next question: Do the three organizer depth-1000 results show material scaling relative to their shallow baselines?
Cost/resources: No repeated model inference; only streaming ranking, packaging, and strict validation are repeated.
Evidence: `artifacts/incidents/20261008_094257_phase10d_depth1000`, `data/source_census/depth1000/round1/source_scores.sqlite`, and `artifacts/source_census/depth1000_report.json`.
Commit: none; commit/push explicitly deferred.

## Phase 10E — G5A 11.2K completion and recovery incidents

Date: 2026-10-09
Phase / commit: Phase 10E focused corpus scaling / uncommitted current worktree
Question: Could the corrected G5A source trio be expanded from about 1,000 documents per source to a meaningful next checkpoint without changing the retrieval experiment contract?
Why we tried it: Corrected G5A depth1000 produced organizer-confirmed positive scaling: FINAL `0.0009` at shallow depth to `0.0037`, with document F2 `0.0035`, chunk F2 `0.0038`, document precision/recall `0.0204/0.0030`, and chunk precision/recall `0.0211/0.0033`.
Hypothesis: Expanding `benhviennhitrunguong.gov.vn` to all 1,200 official rows and `zydcd.com`/`hellobacsi.com` to 5,000 each would provide a useful first Phase 10E scaling point.
What we changed: Source depth only. The completed searchable union contains 11,200 documents and 92,768 chunks: 1,200/18,726 from `benhviennhitrunguong.gov.vn`, 5,000/14,894 from `zydcd.com`, and 5,000/59,148 from `hellobacsi.com`.
What stayed fixed: Queries, pilot control, candidate cap `m=8`, dense/BM25/RRF policy, pinned reranker revision, top-10 documents, top-20 chunks, source-verbatim expansion, provenance, and tie-breaking.
Result: Acquisition processed 8,203 incremental official IDs with 8,203 successful extractions and 62,885 new chunks. Candidate competition produced 28,800 pairs; 12,310 exact `(query_id, chunk_id)` scores were reused and 16,490 were newly inferred. The package `phase10e_G5A_11200.zip` passed the mandatory audit and reached `READY_FOR_LEADERBOARD` with SHA-256 `6881d21914aa44e307254f352354439077426f043c0e6f8c4d4f2afea12fb635`. No organizer result exists yet.
What failed / surprised us: First, the Phase 10E config omitted inherited `inputs.c1_candidate_pool`; the failure happened only after expensive data processing. Recovery reused the persisted calibration and preserved `m=8`. Second, an exact-score startup check rejected a deterministic maximum difference of `0.005859375`. Five same-process repeats and three fresh-process repeats were bit-identical, the scorer contract and sample keys matched, ordering was unchanged, and reconstructing original batch-of-two contexts reproduced stored scores exactly. The cause was FP16 numerical variation from dynamic-padding batch shape, not model drift.
What we learned: Expensive stages need preflight validation of inherited configuration dependencies. Numerical reranker equivalence must require an exact semantic contract and stable ordering while allowing only a measured FP16 envelope; exact floating-point equality is not a scientific contract.
Decision: Mark the single 11.2K package ready only after exact key reconciliation, non-degenerate score checks, cache-reuse audit, corrected-depth1000 replay, strict provenance validation, and deterministic regeneration all pass. Do not start 21.2K until organizer evidence arrives.
Next question: Does the organizer confirm continued G5A scaling at 11.2K, or has ranking dilution begun?
Cost/resources: 8,203 network acquisitions, 62,885 new embeddings, and 16,490 new reranker inferences; exact stage runtimes are preserved in the Phase 10E report and run events.
Evidence: `artifacts/source_census/phase10e_g5a_11200_report.json`, `artifacts/validation/phase10e_g5a_11200_pre_submit_audit.json`, `artifacts/source_census/phase10e_g5a_11200/reranker_equivalence_investigation.json`, and `artifacts/incidents/20261009_015800_phase10e_g5a_11200` / `20261009_030646_phase10e_g5a_11200`.
Commit: none; commit/push explicitly deferred.

## Phase 10E — Focused G5A corpus scaling to up to 11,200 official documents

Date: 2026-10-09
Phase / commit: Phase 10E launch / uncommitted current worktree
Question: Does G5A continue to produce organizer-relevant signal when expanded substantially beyond about 1,000 official documents per source?
Why we tried it: The corrected, scientifically validated depth-1000 G5A submission improved organizer FINAL from `0.0009` at shallow depth to `0.0037`, with gains in every reported precision and recall metric.
Hypothesis: A focused increase to all 1,200 official `benhviennhitrunguong.gov.vn` rows and 5,000 official rows each for `zydcd.com` and `hellobacsi.com` may provide a meaningful next scaling point without jumping directly to 10,000 rows per large source.
What we changed: Only G5A source depth. The exact durable-ID delta is 200, 4,003, and 4,000 URLs respectively; the three prior failed `zydcd.com` IDs are not retried and are replaced by deterministic unattempted official IDs.
What stayed fixed: The 1,200-query set, pilot control corpus, BGE-M3 embeddings, BM25/RRF candidate policy, candidate cap `m=8`, pinned BGE reranker, top-10 document and top-20 chunk policy, 1,024-token source-verbatim expansion, provenance rules, and tie-breaking.
Result: Organizer result pending. The preflight manifest contains 8,203 incremental official IDs, projects about 4.01 GiB conservative retained growth, and passes the 20 GiB free-space gate with about 24.21 GiB expected free after completion.
What failed / surprised us: The earlier ambiguous-SQL depth cache incident demonstrated that structural ZIP validity alone is insufficient. Phase 10E therefore cannot become ready without exact-key reuse audit, non-degenerate scores, and a corrected-depth1000 G5A control replay.
What we learned: Corrected G5A depth scaling is strong enough to move from source discovery to focused acquisition, but one positive scaling interval does not establish linear continuation or source-level attribution.
Decision: Launch one bounded, supervised G5A checkpoint at up to 11,200 official IDs. Do not proceed to the prepared 21,200-ID checkpoint without organizer evidence.
Next question: Does the up-to-11,200 G5A corpus improve organizer metrics over corrected depth1000 without unacceptable precision dilution?
Cost/resources: Projected concurrent crawl time is about 101.6 minutes and conservative end-to-end time about 3.44 hours; projected retained growth is about 4.01 GiB. Actual values will be recorded by the unattended worker.
Evidence: `artifacts/source_census/phase10e_g5a_11200_manifest.json`, `configs/phase10e_g5a_11200.yaml`, corrected depth1000 audit/submission artifacts, and organizer metrics supplied by the project owner.
Commit: none; commit/push explicitly deferred.

### Depth-1000 invalidation — ambiguous SQLite score seeding

Date: 2026-10-08
Phase / commit: Phase 10D depth-1000 repair / uncommitted current worktree
Question: Did the apparent organizer collapse represent negative source-depth scaling?
Why we tried it: G1A depth-1000 fell from shallow `0.0008` to `0.0002`, requiring a strict replay control before interpreting depth.
Hypothesis: Replaying the shallow population through the current finalizer would distinguish ranking dilution from a pipeline defect.
What we changed: No experimental policy changed. We replayed the exact shallow population and audited candidate keys and scores.
What stayed fixed: Corpus populations, queries, candidates, ranking, packaging, provenance, and all model configurations.
Result: The shallow replay was byte-identical at JSON level for all 1,200 queries. The depth cache had 86,400 scored rows but only one distinct score and one timing value. Ambiguous correlated SQL had copied the first old-cache row into every destination row. Organizer results for corrupted G1A (`0.0002`), G5A (`0.0002`), and G6B (`0.0001`) are marked `INVALID_EXPERIMENT_CACHE_CORRUPTION`.
What failed / surprised us: Row-count completeness and SQLite integrity both passed even though value-level semantics were catastrophically wrong. All new candidates were falsely treated as cached.
What we learned: Reuse must be verified by exact composite keys and sampled/exhaustive value agreement; complete row counts cannot establish score correctness.
Decision: Preserve the invalid results historically but exclude them from saturation curves, source prioritization, and model claims. Reset only score columns, seed through an explicitly aliased exact-key `UPDATE ... FROM`, infer only missing keys, and emit separately named `FIXED` packages.
Next question: What organizer depth-scaling signal appears after correct reranker scoring?
Cost/resources: Acquisition, extraction, chunking, embeddings, and candidates are reused; only true missing query/chunk pairs require GPU inference.
Evidence: `artifacts/source_census/g1a_depth1000_control_audit.json`, `data/source_census/depth1000/round1/source_scores.corrupted_20261008.sqlite`, and the repaired score/report artifacts.
Commit: none; commit/push explicitly deferred.

### Depth-1000 cache repair and mandatory scientific validation gate

Date: 2026-10-09
Phase / commit: Phase 10D depth-1000 repair / uncommitted current worktree
Question: Could the depth-1000 experiment be reconstructed without repeating acquisition and protected against structurally complete but scientifically corrupt score caches?
Why we tried it: The corrupted cache had valid row counts and SQLite integrity despite assigning one repeated score to all 86,400 candidate pairs.
Hypothesis: Explicitly aliased exact-key reuse plus value-level and replay controls would recover the intended experiment and prevent this bug class from reaching the leaderboard again.
What we changed: Reset the corrupt score values while preserving candidate identity, reused scores only on exact `(query_id, chunk_id)` joins, inferred only unmatched pairs, and introduced the mandatory pre-submission scientific gate.
What stayed fixed: Acquired documents, extracted text, chunks, embeddings, candidate keys, source groups, models, ranking policy, top-k policy, and provenance rules.
Result: Of 86,400 candidate pairs, 16,707 exact prior-cache matches were reused and 69,693 pairs were newly inferred. The repaired cache had 10,523 distinct finite scores, zero missing/unexpected/duplicate keys, valid timings, and `PRAGMA integrity_check=ok`. Control replay matched all 1,200 queries exactly for top-10 documents and top-20 chunks. The audit status was `READY_FOR_LEADERBOARD`.
What failed / surprised us: Database integrity and complete key coverage alone were insufficient; scientific validity required score-distribution sanity, exact reuse-value agreement, change accounting, and control replay.
What we learned: `READY_FOR_LEADERBOARD` is a scientific status, not merely a ZIP-schema status. Every future package must identify its exact canonical path/hash and pass cache integrity, reuse audit, experiment-contract, replay, provenance, and determinism gates.
Decision: Preserve corrupted submissions under `INVALID_DO_NOT_SUBMIT`, use only separately named `FIXED` packages, and block any future submission whose mandatory audit is not fully passing.
Next question: Which corrected depth-1000 groups show organizer-confirmed scaling?
Cost/resources: Reused 16,707 exact scores and performed 69,693 necessary reranker inferences; acquisition through candidate generation was not repeated.
Evidence: `artifacts/source_census/depth1000_fixed_report.json` and `artifacts/validation/phase10d_depth1000_fixed_pre_submit_audit.json`.
Commit: none; commit/push explicitly deferred.

## Phase 10D Round 1 — Adaptive source census group screen

Date: 2026-10-07 (organizer-confirmed results supplied by the project owner)
Phase / commit: Phase 10D Round 1 / uncommitted current worktree
Question: Can balanced group testing identify useful regions among the previously untested official sources without spending one leaderboard submission per source?
Why we tried it: S4 was valuable but FamilyDoctor scaling was low-yield, and local ranking movement had repeatedly failed to predict organizer relevance.
Hypothesis: Three balanced 21-source groups, using the same 100-document/source depth and fixed retrieval policy, would expose enough organizer separation to eliminate a lower-yield region while preserving ambiguous positive regions.
What we changed: Only source membership changed among Groups A, B, and C. The census reused a fixed pilot control, per-source candidate cap `m=8`, and 582,000 cached reranker scores.
What stayed fixed: Queries, sampled source depth, extraction/chunking, candidate construction, reranker, top-10 documents, top-20 source-verbatim chunks, tie-breaking, and validation policy.
Result: Organizer results were A `FINAL=0.0016`, document/chunk F2 `0.0017/0.0015`; B `FINAL=0.0008`, document/chunk F2 `0.0009/0.0007`; and C `FINAL=0.0017`, document/chunk F2 `0.0017/0.0018`. Full precision/recall metrics are recorded in `docs/leaderboard_history.csv`.
What failed / surprised us: B separated clearly downward, but A and C remained effectively adjacent. C's `0.0001` final-score edge is too small to support declaring it the sole winner.
What we learned: Group testing can safely deprioritize a broad lower-yield source region, but numerical winner-takes-all selection would discard plausible signal. Organizer evidence supports retaining all 42 sources from A+C.
Decision: Deprioritize all Round-1 B sources. Split the 42 A+C survivors into seven disjoint, stratified six-source groups, with exactly three former-A and three former-C sources in each, and reuse the completed score cache without new inference.
Next question: Which of the seven finer source groups show distinguishable organizer signal, and should multiple groups survive to Round 3?
Cost/resources: 8,173 official IDs were sampled in the census; Round-1 packaging reused 582,000 reranker scores. Exact electricity cost was not measured.
Evidence: `artifacts/source_census/round1_groups.json`, `artifacts/source_census/round1_report.json`, `submissions/phase10d_R1_group_A.zip`, `submissions/phase10d_R1_group_B.zip`, `submissions/phase10d_R1_group_C.zip`, and organizer metrics supplied by the project owner.
Commit: pending review; no commit requested.

## Phase 10D Round 2 — Seven-way source group test

Date: 2026-10-07 (organizer-confirmed results supplied by the project owner)
Phase / commit: Phase 10D Round 2 / uncommitted current worktree
Question: Which smaller regions within the 42 surviving Round-1 A+C sources warrant finer source-resolution tests?
Why we tried it: Round 1 clearly rejected B but could not distinguish A from C; seven available submissions allowed a six-source resolution while retaining balanced A/C ancestry.
Hypothesis: Seven disjoint groups of six sources, with only source membership changing, would reveal multiple survivor regions without prematurely attributing signal to individual domains.
What we changed: Split all 42 A+C sources into seven deterministic groups containing exactly three former-A and three former-C sources. Reused the fixed pilot control, Round-1 source depth, candidate cap `m=8`, and all 582,000 reranker scores.
What stayed fixed: Queries, source samples, extraction/chunking, candidates, scores, ranking, top-10 documents, top-20 source-verbatim chunks, tie-breaking, and strict validation.
Result: Organizer FINAL scores were G1 `0.0008`, G2 `0.0004`, G3 `0.0003`, G4 `0.0002`, G5 `0.0011`, G6 `0.0009`, and G7 `0.0002`. Full F2, precision, and recall values are recorded in `docs/leaderboard_history.csv`.
What failed / surprised us: The strongest signal was distributed across three groups rather than one. Group-level gains still cannot identify which member source caused the result.
What we learned: G5, G6, and G1 merit sibling splits; G2 is a reserve, while G3/G4/G7 are lower priority under the fixed policy. Selecting individual sources now would exceed the evidence.
Decision: Preserve all 18 sources from G1/G5/G6 and split each group into two predeclared three-source siblings for Round 3. Do not crawl deeper until source-level evidence strengthens.
Next question: Within each surviving six-source parent, which sibling trio carries the organizer signal?
Cost/resources: Zero crawl, extraction, chunking, embedding, or reranking; seven submissions were packaged from cached candidates and 582,000 scores.
Evidence: `artifacts/source_census/round2_groups.json`, `artifacts/source_census/round2_report.json`, Round-2 submission ZIPs, and organizer metrics supplied by the project owner.
Commit: pending review; no commit requested.

## Phase 10C1 — FamilyDoctor scaling from 1K to 5K

Date: 2026-10-06 (organizer-confirmed result supplied by the project owner)
Phase / commit: Phase 10C1 / uncommitted current worktree
Question: Does scaling the weakly positive S2/FamilyDoctor source from approximately 1,000 to 5,000 sampled official IDs produce enough additional organizer value to justify deeper acquisition?
Why we tried it: S2-1K had produced a small document-side signal (`FINAL_SCORE=0.0002`) in the fixed Phase 10B3 source probe.
Hypothesis: Increasing coverage within the same source, with retrieval, reranking, and submission policy fixed, might improve both document and chunk recall.
What we changed: Added a disjoint deterministic FamilyDoctor sample to reach 5,000 sampled official IDs and recomputed only the missing downstream work.
What stayed fixed: The 1,200 queries, tokenizer/chunking policy, BGE-M3 dense model, BM25/RRF retrieval, pinned multilingual reranker, top-10 document policy, top-20 chunks, and source-verbatim expansion.
Result: The organizer reported `FINAL_SCORE=0.0002`, `DOCS_F2MACRO=0.0003`, `CHUNKS_F2MACRO=0.0002`, `DOCS_PRECISION=0.0017`, `DOCS_RECALL=0.0003`, `CHUNKS_PRECISION=0.0022`, and `CHUNKS_RECALL=0.0001`. The prior S2-1K result was final `0.0002`, document F2 `0.0002`, chunk F2 `0.0001`, document precision/recall `0.0008/0.0002`, and chunk precision/recall `0.0019/0.0001`.
What failed / surprised us: Document-side precision, recall, and F2 rose slightly, but the displayed final score and chunk recall remained flat after a five-fold sampled-source expansion.
What we learned: S2 is not proven irrelevant, but deeper acquisition has low marginal value under the current pipeline. The result is consistent with retaining S2 as a low-yield control or routing source rather than a near-term scaling priority.
Decision: Deprioritize further S2 scaling. Shift bounded acquisition effort toward a source census and adaptive group testing while keeping S4 out of unknown-source pools.
Next question: Which remaining untested sources merit deeper acquisition when screened in balanced groups rather than one leaderboard submission per source?
Cost/resources: 5,000 sampled official S2 IDs; 18,838 novel reranker scores were checkpointed for the completed C2 run. Exact electricity cost was not measured.
Evidence: `artifacts/runs/phase10c1_C2_S2_5k/final_report.json`, `artifacts/phase10c1_early_release/C2_early_release.json`, and `submissions/phase10c1_C2_S2_5k.zip`; organizer metrics supplied by the project owner.
Commit: pending review; no commit requested yet.

## Phase 10E G1A post-reboot reranker-gate recovery

Date: 2026-10-09
Phase / commit: Phase 10E G1A focused scaling / this checkpoint
Question: Can the interrupted G1A rerank resume without rescoring durable pairs while retaining a scientifically meaningful startup-equivalence gate?
Why we tried it: An unexpected reboot left 13,432 of 28,800 exact score rows durable and 15,368 missing. Startup replay was numerically exact, finite, contract-identical, and order-identical, yet the gate rejected it because G1A keys differed from a G5A investigation sample.
Hypothesis: Numerical tolerance evidence may be shared across experiments, but exact control keys and source texts must be deterministic and scoped to the current experiment's valid reused-score population.
What we changed: Separated the shared FP16 investigation contract/tolerance from a deterministic per-experiment control fixture. The G1A fixture selects the first eight exact reused `(query_id, chunk_id)` keys in stable order, records query/chunk text hashes and reference scores, and verifies their provenance against the corrected depth-1000 cache. Added stale-run recovery that marks a dead lock owner `STALE_RUNNING` while preserving checkpoints.
What stayed fixed: Corpus, candidates, 28,800 score keys, the 13,432 durable reused scores, BGE reranker model/revision, FP16, batch size 2, maximum length 512, ranking policy, and submission policy.
Result: Embeddings were independently verified at 131,166 × 1,024 float32 values with a complete 131,166-row bitmap and zero non-finite values. The score DB passed SQLite integrity with 28,800 unique expected keys, 13,432 finite completed rows, 15,368 missing rows, and no duplicate/missing/unexpected keys. The repaired startup gate passed all eight current controls with zero numerical difference, exact text/key identity, unchanged ordering, and zero production pairs scored during the test.
What failed / surprised us: `sample_keys_match_investigation` incorrectly coupled every future experiment to the historical G5A keys used only to measure FP16 batch-shape variation. This converted a valid cross-experiment scorer check into an impossible global candidate-identity requirement.
What we learned: Shared scorer invariants are model contract, finite bounded numerical behavior, and stable ordering. Candidate identity and text provenance belong to an experiment-scoped deterministic control, not to a prior experiment's investigation fixture. Reboot recovery must distinguish a live lock from stale `RUNNING` state without deleting durable work.
Decision: Resume only the 15,368 NULL score rows under the persistent scorer after the experiment-scoped gate passes. Do not repeat embeddings, candidates, or the 13,432 reused scores.
Next question: Does the completed G1A ~10K package pass control replay and the mandatory scientific audit?
Cost/resources: No acquisition, extraction, chunking, embedding, candidate generation, or completed reranking was repeated. The gate-only verification loaded the pinned GPU scorer once and scored eight controls.
Evidence: `data/source_census/phase10e_g1a_10010/round1/chunk_embeddings.f32`, its streaming bitmap/checkpoint, `data/source_census/phase10e_g1a_10010/round1/source_scores.sqlite`, `artifacts/source_census/phase10e_g1a_10010/reranker_equivalence_control.json`, and the Phase 10E G1A run log/state.
Commit: this checkpoint.

## Phase 12 — PubMed acquisition and benchmark-dependency recovery

Date: 2026-10-13
Phase / commit: Phase 12 full PubMed multilingual retrieval / this checkpoint
Question: Can the complete new-official PubMed block be acquired and converted into a crash-resumable multilingual retrieval arm without corrupting the Phase 11C control?
Why we tried it: Phase 11C reached organizer FINAL `0.0492`, while all 240,000 official English PubMed IDs remained absent locally; this was the largest high-quality coverage gap.
Hypothesis: Full PubMed coverage plus pinned BGE-M3 Vietnamese-to-English dense retrieval can add complementary evidence, but acquisition, embeddings, reranking, and packaging must remain transactional on the laptop.
What we changed: Acquired all 800 exact-PMID EFetch batches with atomic raw/parsed/checkpoint triples, added bounded transport recovery, repaired historical `PubmedBookArticle` parsing, created deterministic source-grounded evidence processing, and added a resumable processing-to-package state machine. A missing `psutil` dependency was then caught at `BENCHMARK_EMBED`; recovery added it to `pyproject.toml` and introduced a fail-fast remaining-stage dependency/CUDA/model-contract preflight.
What stayed fixed: The new official ID universe, pinned BGE-M3 and reranker revisions, Phase 11C control candidates and scores, exact-key reranker reuse, Docs@20/Chunks@40 output, and source-verbatim provenance.
Result: Acquisition accounts for 240,000/240,000 PMIDs: 239,770 abstracts, 126 title-only records, and 104 confirmed unavailable records. The corrected parser recovered 197 book records from retained XML. Processing completed 800/800 checkpoints into 239,896 valid documents and 744,861 deterministic evidence chunks; their consolidated hashes and all per-part hashes passed recovery verification. No Phase 12 organizer result exists.
What failed / surprised us: Memory telemetry was imported only when benchmarking began, so the absent dependency was discovered after processing rather than at launch. After that repair, both the supervisor and embedding worker disappeared at 11,520 rows without a new traceback, WER record, resource-exhaustion event, or terminal supervisor event; the cause of process termination is unknown. No acquisition, processing, or committed embedding data was lost.
What we learned: Long state machines need a complete remaining-stage dependency and pinned-model cache preflight before expensive work. They also need explicit supervisor/worker PIDs, heartbeats, orphan-worker checks, and a committed-prefix audit because a supervisor can itself disappear before producing an incident.
Decision: Resume exactly at embedding row 11,520 after proving a contiguous finite bitmap/memmap prefix; preserve all acquisition and processing checkpoints and use an operating-system-detached supervisor.
Next question: What full-index throughput and PubMed attribution does BGE-M3 achieve, and do PubMed candidates enter the final Phase 11C union meaningfully?
Cost/resources: Public NCBI acquisition at 2.5 requests/second; processing was CPU-only. Embedding and reranker costs remain pending measured execution.
Evidence: `artifacts/phase12_pubmed/acquisition_summary.json`, `artifacts/phase12_pubmed/processing_quality_audit.json`, and `artifacts/incidents/20261013_035705_phase12_full_pubmed`.
Commit: this checkpoint.

## Phase 11B — New official corpus control and document Recall@K curve

Date: 2026-10-12
Phase / commit: Phase 11B-0/11B-1 / this checkpoint
Question: Under the organizer's new 1,194,469-row corpus membership, are useful documents already present deeper in one fixed ranking, or are they missing before ranking?
Why we tried it: The organizer replaced the 4,394,718-row official corpus with a Vietnamese/Chinese/English corpus while preserving document IDs as keys. Historical outputs could contain removed IDs. Phase 11A had also shown organizer gains from legal deeper output (`0.0470`) and M16 candidates (`0.0450`) over the `0.0426` baseline, so a clean new-corpus depth curve was required before further acquisition.
Hypothesis: If relevant documents exist deeper in the reconciled ranking, organizer recall should continue improving as document output increases while chunks remain fixed at 20.
What we changed: Replaced official membership with `links_corpus_vi_en_zh.parquet`; reconciled source and frozen-pilot candidates to exact new IDs; rebuilt rankings from the completed score cache; and emitted document depths 10, 20, and 50 from one ranking with identical 20-chunk output.
What stayed fixed: All 1,200 queries, retained local content, M16 retrieval/fusion policy, BGE-M3 and reranker revisions, reranker scores, deterministic ordering, source-verbatim chunks, and chunk depth 20. There was no crawling or new model inference.
Result: The official diff retained 954,469 old IDs, removed 3,440,249, and introduced 240,000 PubMed IDs; no retained ID changed URL. The focused corpus retained 26,639 documents/444,411 chunks, and the broad inventory retained 35,690/523,482. The final ranking's unique-document depth distribution was min/p5/median/p95/max `54/80/96/111/120`, making K=10/20/50 scientifically common and K=100 invalid. All three packages passed new-official membership, score-cache, deterministic, fixed-prefix/chunk, structural, and provenance gates and reached `READY_FOR_LEADERBOARD`. Organizer results are pending.
What failed / surprised us: Recovery exposed four implementation boundaries rather than scientific failures: a missing nested output directory, doc-ID selection where exact chunk IDs were required, a canonical store that omitted valid pilot provenance while stale pilot candidates retained removed IDs, and a Phase 11B consumer that expected `validation.passed` although strict validators expose `valid`. The shared validator adapter now rejects incomplete schemas rather than defaulting to success.
What we learned: Corpus migration requires reconciling candidate membership and provenance together. A ZIP can be structurally present yet scientifically blocked by a contract mismatch. Document-depth variants can safely reuse frozen chunks only when prefix, chunk identity, official membership, and deterministic regeneration are verified explicitly.
Decision: Submit Docs@10, then Docs@20, then Docs@50. Do not fabricate Docs@100 or start PubMed/Broad/ICD/model experiments until organizer results characterize the curve.
Next question: Does organizer document recall continue rising strongly from K=10 to K=50, or plateau early enough to implicate candidate/corpus coverage?
Cost/resources: Zero network requests and zero new embedding/reranker inference during recovery. All 134,400 scores were preserved; package-only streaming produced three deterministic ZIPs.
Evidence: `docs/PHASE11B_NEW_CORPUS_MIGRATION.md`, `artifacts/phase11b_new_corpus/migration_audit.json`, `provenance_closure.json`, `phase11b_report.json`, and `phase11b_pre_submit_audit.json`.
Commit: this checkpoint.

## Phase 11A — output and candidate bottleneck organizer probes

Date: 2026-10-12 (organizer-confirmed results supplied by the project owner)
Phase / commit: Phase 11A bottleneck diagnosis / this checkpoint
Question: Is the Medlatec-10K baseline primarily limited by final output depth or by the M8 per-source candidate cap?
Why we tried it: FULL_MEDLATEC_10000 reached FINAL `0.0426`, but the system could discard useful ranked results at either the candidate or final-output boundary.
Hypothesis: Changing one boundary at a time would reveal whether either truncation suppresses organizer recall.
What we changed: One experiment changed only final output from 10/20 to 20 documents/40 chunks; a second changed only candidate depth from M8 to M16 while retaining 10/20 output.
What stayed fixed: The historical official corpus, queries, focused corpus, retrieval/reranker contracts apart from the declared variable, provenance, and deterministic ranking.
Result: OUTPUT20/40 reached FINAL `0.0470` (`DOCS_F2=0.0487`, `CHUNKS_F2=0.0454`). M16 reached FINAL `0.0450` (`DOCS_F2=0.0431`, `CHUNKS_F2=0.0469`). Both improved over `0.0426`; output depth produced the larger FINAL gain. These remain historical diagnostics after the official corpus migration.
What failed / surprised us: The old Broad-existing package became stale before organizer use because the official corpus changed. It cannot answer the new-corpus breadth question.
What we learned: Both output and candidate truncation matter, and a clean document-only depth curve is more informative than the historical OUTPUT20/40 experiment because the latter changed document and chunk depths together.
Decision: Migrate first, then hold chunks at 20 while varying only document K. Preserve the Phase 11A organizer results as historical evidence; do not reuse the stale Broad package.
Next question: On the new corpus, how far does organizer document recall rise with K alone?
Cost/resources: Organizer evaluation of two no-crawl diagnostic packages.
Evidence: `artifacts/phase11a_bottleneck/`, `docs/leaderboard_history.csv`, and organizer metrics supplied by the project owner.
Commit: this checkpoint.

## Phase 10F — Medlatec depth response and Phase 11A pivot

Date: 2026-10-11 (organizer-confirmed results supplied by the project owner)
Phase / commit: Phase 10F to Phase 11A transition / this checkpoint
Question: Does Medlatec continue improving the organizer score through 10,000 official IDs, and should the next budget deepen it again?
Why we tried it: Medlatec had the largest measured marginal contribution inside the nine-source union, but one more bounded depth curve was needed before allocating another crawl tranche.
Hypothesis: If Medlatec retained useful uncovered documents, organizer FINAL would continue rising from the 5K checkpoint through 7.5K and 10K under the fixed retrieval contract.
What we changed: Increased only Medlatec official-ID depth while holding the other eight sources, queries, retrieval, reranker, and output policy fixed.
What stayed fixed: The full nine-source composition apart from Medlatec depth, 1,200 queries, `m=8`, BGE-M3 and reranker revisions, dense/BM25/RRF policy, FP16 batch 2/max length 512, top-10 documents, top-20 chunks, provenance, and deterministic tie-breaking.
Result: Organizer FINAL increased from `0.0383` at Medlatec 5K to `0.0398` at 7.5K and `0.0426` at 10K. At 10K, document F2/precision/recall were `0.0418/0.1971/0.0370`; chunk F2/precision/recall were `0.0435/0.1576/0.0399`. This is the current organizer best.
What failed / surprised us: Scaling still helped through 10K, but the absolute recall remains low relative to the 4,394,718-row official corpus. Another single-source crawl would not reveal whether the larger ceiling is final output depth, per-source candidate truncation, or corpus breadth.
What we learned: Medlatec remains valuable at 10K, but depth response alone cannot identify the dominant end-to-end bottleneck.
Decision: Pause further Medlatec acquisition. Start Phase 11A with three controlled no-crawl probes: legal deeper output, `m=16` candidate depth, and the widest provenance-valid existing local corpus.
Next question: Is current performance limited primarily by final output depth, candidate recall, or missing corpus coverage?
Cost/resources: Organizer evaluation of already prepared packages; no additional local model or network work for this checkpoint.
Evidence: `docs/leaderboard_history.csv`, `artifacts/source_census/phase10f_full_medlatec_7500_report.json`, `artifacts/source_census/phase10f_full_medlatec_10000_report.json`, and organizer metrics supplied by the project owner.
Commit: this checkpoint.

## Phase 10E — Full nine-source union organizer result

Date: 2026-10-10 (organizer-confirmed result supplied by the project owner)
Phase / commit: Phase 10E focused corpus scaling / pending this checkpoint
Question: Does organizer-valid G5A add complementary signal beyond G1A+G6B?
Why we tried it: G1A+G6B reached FINAL `0.0332`, while G5A independently reached `0.0085`.
Hypothesis: Some G5A documents remain complementary after full competition with the stronger union.
What we changed: Added the three G5A sources to the six-source union and recomputed rankings from cached source candidates and exact scores.
What stayed fixed: Queries, pilot corpus, `m=8`, retrieval/fusion, model revisions, reranker policy, top-10/top-20 output, verbatim provenance, and deterministic tie-breaking.
Result: Organizer FINAL reached `0.0383`; document F2/precision/recall were `0.0379/0.1796/0.0338`, and chunk F2/precision/recall were `0.0387/0.1465/0.0354`.
What failed / surprised us: G5A added measurable value but less than its independent score, again showing overlap and competition between source families.
What we learned: The full nine-source union is the strongest organizer-valid baseline. Marginal-source evidence now has greater decision value than another blind depth increase.
Decision: Use the final three pre-reset slots for leave-out Medlatec, leave-out Sức Khỏe Đời Sống, and shallow Round-3 resurrection probes. Do not acquire deeper data yet.
Next question: Which source contributes the most marginal value, and do eliminated shallow groups contain complementary gold?
Cost/resources: No new acquisition or model inference was required for the scored union.
Evidence: `artifacts/source_census/phase10e_g1a10k_g6b15k_g5a11200_union_report.json`, its pre-submit audit, and organizer metrics supplied by the project owner.
Commit: pending this checkpoint.

## Phase 10E — G1A focused scaling to 10,010 official documents

Date: 2026-10-09 (organizer-confirmed result supplied by the project owner)
Phase / commit: Phase 10E G1A focused scaling / pending this checkpoint
Question: Does G1A retain its high organizer yield when Medlatec and Vinmec scale from about 1,000 to 5,000 official IDs each?
Why we tried it: Corrected G1A depth1000 reached organizer FINAL `0.0062` with only 2,005 usable documents, the highest observed yield per document among the tested priority trios.
Hypothesis: Increasing only G1A corpus depth under the fixed retrieval and submission contract would expose additional organizer-relevant documents and chunks.
What we changed: Increased Medlatec and Vinmec to 5,000 official IDs each while retaining all 10 available `v.familydoctor.com.cn` IDs. The completed union contained 10,010 official documents and 131,166 chunks.
What stayed fixed: All 1,200 queries, pilot control corpus, `m=8` source candidate cap, BGE-M3 dense retrieval, BM25/RRF policy, pinned BGE reranker revision, FP16 batch 2 and maximum length 512, top-10 documents, top-20 source-verbatim chunks, deterministic tie-breaking, provenance, and scientific validation gates.
Result: Organizer FINAL reached `0.0234`; document F2/precision/recall were `0.0245/0.1204/0.0219`, and chunk F2/precision/recall were `0.0224/0.0830/0.0207`. This compares with corrected G1A depth1000 FINAL `0.0062`.
What failed / surprised us: The improvement was substantially larger than the earlier G1A depth1000 checkpoint suggested, despite the same retrieval contract. This does not establish that the gain will remain linear at greater depth.
What we learned: G1A has organizer-confirmed positive depth scaling and remains a high-density source family. Source-family behavior must still be measured independently; G1A success cannot be transferred to G6B by assumption.
Decision: Mark `phase10e_G1A_10010.zip` `SUBMITTED_VALID`. Launch only the already-prepared G6B 15K reserve checkpoint next; do not run deeper G1A, G5A, or any other source-family acquisition concurrently.
Next question: Does G6B also scale positively when its three sources reach 5,000 official IDs each?
Cost/resources: 10,010 official documents, 131,166 chunks, 28,800 candidate pairs, exact prior-score reuse plus only required new reranker inference; detailed stage costs remain in the Phase 10E report.
Evidence: `artifacts/source_census/phase10e_g1a_10010_report.json`, `artifacts/validation/phase10e_g1a_10010_pre_submit_audit.json`, `submissions/10_SUBMITTED_VALID/phase10e_G1A_10010.zip`, and organizer metrics supplied by the project owner.
Commit: pending this checkpoint.

## Phase 10E G6B post-reboot assembly recovery

Date: 2026-10-09
Phase / commit: Phase 10E G6B focused scaling / this checkpoint
Question: Can the interrupted G6B 15K run resume from its first incomplete atomic boundary without repeating durable acquisition, extraction, or chunking?
Why we tried it: An external machine shutdown left stale `ASSEMBLE/RUNNING` state after all 11,900 incremental official IDs, 11,900 new document rows, and 150,798 new chunks had been durably written.
Hypothesis: The union outputs were absent rather than corrupt, so rebuilding only ASSEMBLE from verified old and new inputs would preserve the depth-only experiment contract.
What we changed: Added an explicit `assemble` resume boundary, required the durable new document/chunk inputs before using it, strengthened preflight to require the validated streaming-embedding memory profile and equivalence evidence, and made the lightweight embedding continuation wait for measured physical-memory and commit headroom before loading the model. G6B reuses the validated G1A streaming profile because the embedding model, revision, precision, batch, dimensions, and streaming implementation are identical.
What stayed fixed: G6B source membership and depth, all acquired/extracted/chunked data, `m=8`, dense and sparse retrieval, fusion, embedding and reranker contracts, top-10/top-20 policy, tie-breaking, provenance, and the 20 GiB disk floor.
Result: The audit found no durable or partial ASSEMBLE outputs. Verified inputs contain 3,100 old plus 11,900 new official document rows and 39,304 old plus 150,798 new chunks, with zero old/new ID overlap. The expected union is 15,000 document rows, 14,908 usable documents, and 190,102 chunks. Recovery therefore starts at ASSEMBLE only and preserves all prior stages.
What failed / surprised us: The run state remained `RUNNING` after reboot, and the worker previously had no ASSEMBLE-only resume choice. Disk remained safe, but current host-memory headroom was below the peak measured by the validated streaming embedding profile.
What we learned: Every expensive atomic boundary needs an explicit resume entry point. A memory-safe implementation also needs a measured startup gate; after reboot, the supervisor should wait with checkpoints intact rather than loading a native model into insufficient host/commit headroom.
Decision: Rebuild only the union atomically, then allow the unattended run to wait at `EMBEDDINGS_MEMORY_WAIT` until the evidence-based memory gate passes before continuing automatically.
Next question: Does the fixed-policy G6B 15K depth checkpoint pass its control replay and mandatory pre-submission scientific audit?
Cost/resources: No network acquisition, extraction, or chunking is repeated. At audit time D: had 30.73 GiB free; the validated embedding profile measured about 833.41 MiB peak RSS and 3,931.96 MiB projected peak private memory.
Evidence: `data/source_census/phase10e_g6b_15000/new_documents.parquet`, `data/source_census/phase10e_g6b_15000/new_chunks.parquet`, `artifacts/source_census/phase10e_g6b_15000/new_chunk_merge.json`, and `artifacts/runs/phase10e_g6b_15000/state.json`.
Commit: this checkpoint.

## Phase 10E G6B stale-embedding recovery

Date: 2026-10-09
Phase / commit: Phase 10E G6B focused scaling / this checkpoint
Question: Can the post-reboot embedding stage resume from its transactional bitmap without accepting unsafe near-zero startup-memory headroom?
Why we tried it: ASSEMBLE completed, but a reboot left stale `EMBEDDINGS/RUNNING` state with no live owner. Preparation had durably reused 37,434 of 190,102 rows and left 152,668 rows for inference.
Hypothesis: The exact partial-vector shape, binary completion bitmap, and finite completed rows permit a lossless EMBEDDINGS-only resume; startup should wait until physical memory covers both the measured peak and a measured warmup excursion.
What we changed: Reconciled lockless `RUNNING` state as `STALE_RUNNING`, preserving checkpoints. Revised the startup gate from bare peak RSS to peak RSS plus one additional observed model-loaded-to-peak excursion: `833.41 + (833.41 - 717.41) = 949.41 MiB`. Commit headroom remains tied to the measured projected peak private memory of 3,931.96 MiB.
What stayed fixed: The 190,102-row union, 1,024-dimensional float32 layout, 37,434 reused vectors, completion bitmap, BGE-M3 model/revision, batch size 4, maximum length 512, 128-row Parquet windows, zero workers/prefetch, retrieval contract, and every stage before EMBEDDINGS.
Result: The partial vector file is exactly 778,657,792 bytes, its bitmap contains 37,434 completed and 152,668 missing rows with no invalid status bytes, and all completed vectors are finite. Current free physical memory was below 949.41 MiB, so the unattended supervisor was configured to remain at `EMBEDDINGS_MEMORY_WAIT` without loading the model until the evidence-based gate passes.
What failed / surprised us: The earlier gate technically passed with only 5.33 MiB above the measured peak, which is not enough to absorb the warmup excursion already observed in the same validated profile. Also, a missing lock previously prevented explicit stale-state annotation.
What we learned: A measured peak is not itself an operational safety reserve. Reusing one additional observed warmup excursion gives a profile-derived margin without imposing a broad arbitrary RAM threshold, and lockless `RUNNING` state must be recorded as interrupted before resumption.
Decision: Preserve all 37,434 completed rows and resume only the 152,668 missing embeddings once available physical memory is at least 949.41 MiB and available commit is at least 3,931.96 MiB.
Next question: After embeddings complete, does G6B 15K pass fixed-policy candidate generation, reranking, control replay, and the mandatory scientific audit?
Cost/resources: No acquisition, extraction, chunking, assembly, or completed embedding row was repeated. Only read-only checkpoint and finite-vector verification was performed before relaunch.
Evidence: `data/source_census/phase10e_g6b_15000/round1/chunk_embeddings.streaming.checkpoint.json`, `.streaming.status.u8`, `.streaming.partial`, and `artifacts/runs/phase10e_g6b_15000/state.json`.
Commit: this checkpoint.

## Phase 10E — G6B focused scaling to 15K official IDs

Date: 2026-10-10 (organizer-confirmed result supplied by the project owner)
Phase / commit: Phase 10E G6B focused scaling / pending this checkpoint
Question: Does G6B retain organizer-relevant signal when all three sources scale from about 1,000 to 5,000 official IDs each?
Why we tried it: Corrected G6B depth1000 reached organizer FINAL `0.0045`, exceeding corrected G5A at comparable depth and justifying one independent focused-scaling checkpoint.
Hypothesis: Increasing only G6B corpus depth under the fixed retrieval and submission contract would expose additional organizer-relevant documents and chunks.
What we changed: Increased `tiemchunglongchau.com.vn`, `cancer.39.net`, and `suckhoedoisong.vn` to 5,000 official IDs each. The final searchable corpus contained 15,000 official rows, 14,908 usable documents, and 190,102 chunks.
What stayed fixed: All 1,200 queries, pilot control corpus, `m=8` source candidate cap, BGE-M3 dense retrieval, BM25/RRF policy, pinned BGE reranker revision, FP16 batch 2 and maximum length 512, top-10 documents, top-20 source-verbatim chunks, deterministic tie-breaking, provenance, and scientific validation gates.
Result: Organizer FINAL reached `0.0140`; document F2/precision/recall were `0.0145/0.0762/0.0128`, and chunk F2/precision/recall were `0.0134/0.0697/0.0119`. This compares with corrected G6B depth1000 FINAL `0.0045`, an approximately 3.11x increase.
What failed / surprised us: G6B scaled strongly but remained below independently validated G1A 10K FINAL `0.0234`. This does not show whether the groups retrieve overlapping or complementary gold documents.
What we learned: G6B has organizer-confirmed positive depth scaling, independently of G1A. Scores cannot be assumed additive, so complementarity must be tested by full cross-source ranking competition rather than concatenating final lists.
Decision: Mark `phase10e_G6B_15000.zip` `SUBMITTED_VALID`. Run one no-acquisition G1A 10K + G6B 15K union experiment, reusing exact embeddings and reranker scores but rebuilding final cross-source rankings. Do not start deeper acquisition yet.
Next question: Does the union improve organizer performance beyond G1A alone?
Cost/resources: 15,000 official rows, 14,908 usable documents, 190,102 chunks, 28,800 candidate pairs, 5,836 exact reused reranker scores, and 22,964 necessary new inferences; detailed stage costs are in the Phase 10E report.
Evidence: `artifacts/source_census/phase10e_g6b_15000_report.json`, `artifacts/validation/phase10e_g6b_15000_pre_submit_audit.json`, `submissions/10_SUBMITTED_VALID/phase10e_G6B_15000.zip`, and organizer metrics supplied by the project owner.
Commit: pending this checkpoint.

## Phase 10E — G1A 10K + G6B 15K union

Date: 2026-10-10
Phase / commit: Phase 10E focused corpus scaling / this checkpoint
Question: Do the two strongest independently organizer-validated source groups provide complementary retrieval signal beyond G1A alone?
Why we tried it: G1A 10K reached organizer FINAL `0.0234`, while G6B 15K independently reached `0.0140`. Scores are not additive, and only full cross-source competition can test complementarity.
Hypothesis: Re-ranking the fixed per-source candidates from all six sources together may expose useful G6B documents without sacrificing G1A's stronger signal.
What we changed: Source membership became the union of G1A (`medlatec.vn`, `v.familydoctor.com.cn`, `vinmec.com`) and G6B (`tiemchunglongchau.com.vn`, `cancer.39.net`, `suckhoedoisong.vn`). Final ranking was rebuilt over the combined candidates; parent final lists were not concatenated.
What stayed fixed: All 1,200 queries, pilot control corpus, `m=8`, BGE-M3 embedding contract, dense/BM25/RRF policy, exact reranker model/revision and scores, FP16 batch 2/max length 512 contract, top-10 documents, top-20 source-verbatim chunks, deterministic tie-breaking, and provenance rules.
Result: The union contains 25,010 official document rows (24,909 usable) and 321,268 chunks with zero duplicate document or chunk IDs. All 321,268 embeddings and all 57,600 reranker scores were reused exactly; no model inference ran. Both parent controls reproduced their organizer-valid canonical JSON byte-for-byte. The deterministic union package passed the mandatory scientific audit and is `READY_FOR_LEADERBOARD`; no organizer result exists yet.
What failed / surprised us: The legacy package helper exhausted host memory by materializing a 118 MB compressed text parquet as Python strings and dictionaries. Candidate/ranking work was unaffected. A disk-backed canonical SQLite store and per-query streaming finalizer completed at bounded resident memory. The union changes top-10 documents and top-20 chunks for 1,199 of 1,200 queries, showing that it is a genuine competition test rather than a near-copy of G1A.
What we learned: Cache reuse can eliminate all model inference for a source-union experiment, but final cross-source ranking must still be recomputed. Memory-bounded packaging is part of scientific reliability because a valid score cache is not sufficient if finalization cannot reproduce parent controls.
Decision: Make `phase10e_G1A_10010_G6B_15000_UNION.zip` the sole current upload candidate. Do not deepen G1A or G6B until its organizer result is observed.
Next question: Does the organizer score show complementary G6B signal beyond G1A 10K alone?
Cost/resources: Zero network acquisition, zero extraction/chunking, zero new embeddings, and zero reranker inference. CPU work was limited to combined ranking, source-window expansion, deterministic packaging, and validation.
Evidence: `artifacts/source_census/phase10e_g1a10k_g6b15k_union_report.json`, `artifacts/validation/phase10e_g1a10k_g6b15k_union_pre_submit_audit.json`, and `submissions/00_READY_TO_UPLOAD/phase10e_G1A_10010_G6B_15000_UNION.zip`.
Commit: this checkpoint.

## Phase 10E — G1A 10K + G6B 15K organizer result

Date: 2026-10-10 (organizer-confirmed result supplied by the project owner)
Phase / commit: Phase 10E focused corpus scaling / pending this checkpoint
Question: Do organizer-valid G1A and G6B provide complementary signal when ranked together over the full union?
Why we tried it: G1A 10K reached FINAL `0.0234` and G6B 15K reached `0.0140`, but separate scores cannot establish overlap or complementarity.
Hypothesis: Full cross-source competition would improve over G1A alone if G6B contributes distinct relevant documents and chunks.
What we changed: Added G6B source membership to G1A while reusing exact embeddings and reranker scores and recomputing combined rankings.
What stayed fixed: Queries, pilot control corpus, `m=8`, retrieval/fusion, BGE-M3 and reranker revisions, FP16 batch 2/max length 512, top-10/top-20 policy, verbatim provenance, and deterministic tie-breaking.
Result: Organizer FINAL reached `0.0332`; document F2/precision/recall were `0.0334/0.1597/0.0299`, and chunk F2/precision/recall were `0.0331/0.1323/0.0303`. This exceeds G1A alone at `0.0234` and confirms complementary signal.
What failed / surprised us: The gain is substantial but not additive, reinforcing that source groups must compete under one ranking rather than having their independent scores summed or final lists concatenated.
What we learned: Independently useful source families can contribute complementary organizer signal under a fixed retrieval contract. Exact cached source-level scores make further controlled union tests inexpensive.
Decision: Mark the two-group union `SUBMITTED_VALID`. Add organizer-valid G5A 11.2K in one no-acquisition nine-source union before authorizing deeper crawling.
Next question: Does G5A add complementary relevant documents beyond the G1A+G6B union?
Cost/resources: No acquisition or new model inference was required for the scored two-group union.
Evidence: `artifacts/source_census/phase10e_g1a10k_g6b15k_union_report.json`, `artifacts/validation/phase10e_g1a10k_g6b15k_union_pre_submit_audit.json`, and organizer metrics supplied by the project owner.
Commit: pending this checkpoint.

## Phase 10E — G1A 10K + G6B 15K + G5A 11.2K union

Date: 2026-10-10
Phase / commit: Phase 10E focused corpus scaling / this checkpoint
Question: Does organizer-valid G5A add complementary relevant documents beyond the already strong G1A+G6B union?
Why we tried it: The G1A+G6B union reached organizer FINAL `0.0332`, while G5A 11.2K independently reached `0.0085`. Only a full nine-source ranking can measure additional complementarity.
Hypothesis: G5A may contribute distinct relevant documents and chunks after competing directly with the stronger two-group union.
What we changed: Added `benhviennhitrunguong.gov.vn`, `zydcd.com`, and `hellobacsi.com` to the six G1A/G6B sources and recomputed combined candidate rankings.
What stayed fixed: All 1,200 queries, pilot control corpus, `m=8`, BGE-M3 and reranker revisions, retrieval/fusion policy, FP16 batch 2/max length 512, top-10/top-20 policy, source-verbatim expansion, provenance, and deterministic tie-breaking.
Result: The union contains 36,210 unique official document rows (36,109 usable) and 414,036 unique chunks. All embeddings and 86,400 reranker scores were reused exactly; no model inference ran. G1A, G6B, G5A, and G1A+G6B controls reproduced byte-for-byte. The final package passed structural, provenance, score-cache, official-ID, and determinism gates and is `READY_FOR_LEADERBOARD`; no organizer result exists yet.
What failed / surprised us: The first packaging attempt from the previous union had already shown corpus-wide Python text materialization was unsafe, so this experiment used the disk-backed streaming path from the start. Adding G5A changed top-10 documents for 994 queries and top-20 chunks for 1,120 queries, but this movement is not itself relevance evidence.
What we learned: The source-level cache architecture supports scientifically controlled multi-group unions with zero new inference, while byte-identical parent replays protect against silent semantic drift.
Decision: Submit only the validated nine-source union manually. Do not start deeper acquisition until its organizer result is known.
Next question: Does G5A improve organizer metrics beyond the two-group FINAL `0.0332`?
Cost/resources: Zero acquisition, extraction, chunking, embedding, or reranker inference; CPU-only ranking, source-window expansion, packaging, and validation.
Evidence: `artifacts/source_census/phase10e_g1a10k_g6b15k_g5a11200_union_report.json`, `artifacts/validation/phase10e_g1a10k_g6b15k_g5a11200_union_pre_submit_audit.json`, and `submissions/00_READY_TO_UPLOAD/phase10e_G1A_10010_G6B_15000_G5A_11200_UNION.zip`.
Commit: this checkpoint.

## Phase 10E — final three pre-reset information probes

Date: 2026-10-10
Phase / commit: Phase 10E focused corpus scaling / this checkpoint
Question: How much marginal organizer-visible value do `medlatec.vn` and `suckhoedoisong.vn` contribute, and do the already-acquired shallow samples from the eliminated Round-3 groups add complementary signal beyond the validated nine-source union?
Why we tried it: Only three organizer submissions remained before quota reset. Leave-one-source-out controls provide direct marginal evidence for two large sources, while a no-acquisition resurrection union tests whether near-cutoff shallow groups were eliminated too early.
Hypothesis: Removing a valuable source should measurably reduce organizer performance, while adding complementary shallow sources may improve it despite their weaker earlier group scores.
What we changed: Built three independently ranked corpora: the validated full union minus Medlatec; the full union minus Suckhoedoisong; and the full union plus all durable shallow sources from Round-3 G1B, G5B, and G6A. Candidate competition and final ranking were recomputed for each source membership; prior final lists were not filtered or concatenated.
What stayed fixed: All 1,200 queries, pilot control corpus, calibrated `m=8`, BGE-M3 embedding contract, dense/BM25/RRF policy, exact reranker model/revision and cached scores, FP16 batch 2/max length 512 contract, top-10 documents, top-20 source-verbatim chunks, deterministic tie-breaking, and provenance rules.
Result: All three packages reached `READY_FOR_LEADERBOARD`. Minus-Medlatec contains 31,210 official rows (31,117 usable), 315,703 chunks, and 76,800 exactly reused scores; it changes top-10 documents for 1,136 queries and top-20 chunks for 1,181. Minus-Suckhoedoisong contains 31,210 official rows (31,110 usable), 299,555 chunks, and 76,800 reused scores; it changes top-10 documents for 824 queries and top-20 chunks for 1,055. The resurrection union contains 36,994 official rows (36,890 usable), 423,903 chunks, and 172,800 reused scores; it changes top-10 documents for 178 queries and top-20 chunks for 344. No acquisition, embedding, or reranker inference ran.
What failed / surprised us: Integrating legacy Round-1 caches exposed two implementation assumptions: cache artifacts were not nested under an additional `round1/` directory, and legacy document error columns used `string` where newer successful-only corpora used Arrow `null`. Explicit cache roots and schema unification repaired these representation differences without changing retrieval semantics. Local ranking movement remains diagnostic only and is not organizer relevance evidence.
What we learned: Scientifically valid source-membership probes can be assembled entirely from exact per-source candidate and score caches, provided cache selection is scoped to the declared parent sources and the validated full-union control is replayed first. The shallow resurrection sources affect a minority of final rankings, but organizer results are required to determine whether those changes contain complementary gold.
Decision: Reserve the final three pre-reset submissions for the probes in this order: minus Medlatec, minus Suckhoedoisong, then Round-3 resurrection. Do not authorize deeper acquisition from local movement alone.
Next question: Which probe deltas, after organizer scoring, identify the best source family or neglected source region for the next depth-scaling budget?
Cost/resources: CPU-only cached ranking, disk-backed canonical text expansion, deterministic packaging, and validation. Reused 1,039,161 embedding rows across the three evaluated corpora and 326,400 exact reranker scores in total; new model inference and network requests were zero.
Evidence: `artifacts/source_census/phase10e_final_pre_reset_probes_report.json`, the three `artifacts/validation/phase10e_full_*_pre_submit_audit.json` files, and the registered packages under `submissions/00_READY_TO_UPLOAD/`.
Commit: this checkpoint.

## Phase 10F — pre-reset source attribution results

Date: 2026-10-10 (organizer-confirmed results supplied by the project owner)
Phase / commit: Phase 10F source attribution and focused scaling / this checkpoint
Question: Which current source contributes the greatest marginal organizer-visible value, and do the eliminated shallow Round-3 sources add complementary signal?
Why we tried it: The nine-source union reached FINAL `0.0383`, but a scaling decision requires marginal evidence rather than independent source-group scores or local ranking movement.
Hypothesis: Removing a high-value source would reduce organizer score materially, while adding complementary shallow sources could improve the full union without new acquisition.
What we changed: Submitted three fixed-contract source-membership probes: full union minus `medlatec.vn`, full union minus `suckhoedoisong.vn`, and full union plus the durable shallow G1B/G5B/G6A sources.
What stayed fixed: Queries, pilot corpus, calibrated `m=8`, retrieval/fusion, model revisions, exact reranker policy, top-10/top-20 output, verbatim provenance, and deterministic tie-breaking.
Result: FULL_MINUS_MEDLATEC reached FINAL `0.0288`, a `0.0095` decline from the `0.0383` baseline. FULL_MINUS_SUCKHOEDOISONG reached `0.0351`, a `0.0032` decline. FULL_PLUS_R3_RESURRECTION remained at `0.0383`, with no measurable displayed FINAL gain.
What failed / surprised us: The shallow resurrection corpus changed hundreds of local rankings but produced no displayed organizer gain, again demonstrating that output movement is not relevance evidence. Its flat score does not prove those sources cannot scale at greater depth.
What we learned: Medlatec currently provides very strong marginal organizer value; Suckhoedoisong provides smaller but positive marginal value. The shallow resurrection set does not presently justify priority over already validated sources.
Decision: Scale only Medlatec from 5K toward 10K while retaining the other eight union sources unchanged. In parallel, prepare zero-acquisition leave-one-out probes for Vinmec, Tiemchunglongchau, and Hellobacsi to identify the next scaling target.
Next question: What is Medlatec's depth response at 7.5K and 10K, and which of Vinmec, Tiemchunglongchau, or Hellobacsi has the largest current marginal value?
Cost/resources: Organizer evaluation only; probe construction previously required no acquisition or new model inference.
Evidence: `docs/leaderboard_history.csv`, the three Phase 10E probe audits, and organizer metrics supplied by the project owner.
Commit: this checkpoint.

## Phase 10F — Medlatec depth-scaling preflight and ablation packaging safety stop

Date: 2026-10-10
Phase / commit: Phase 10F source attribution and focused scaling / this checkpoint
Question: Can three cached source ablations and nested Medlatec 7.5K/10K checkpoints execute safely on the current laptop without changing the validated retrieval contract?
Why we tried it: Organizer attribution made Medlatec the strongest current scaling priority, while Vinmec, Tiemchunglongchau, and Hellobacsi still require cheap marginal attribution before receiving future acquisition budget.
Hypothesis: The ablations require zero inference, and acquiring one additional 5K Medlatec tranche should remain bounded under the streaming pipeline if the 20 GiB disk floor holds.
What we changed: Added fixed-contract ablation and Medlatec configurations, deterministic nested 7.5K/10K ID checkpoints, measured checkpoint-output allowances, and a streaming query-by-query submission comparator. No retrieval, model, or output policy changed.
What stayed fixed: The organizer-valid nine-source control, all source populations except proposed Medlatec depth, `m=8`, BGE-M3 and reranker revisions, dense/BM25/RRF policy, FP16 batch 2/max length 512, top-10/top-20 policy, verbatim provenance, and deterministic tie-breaking.
Result: Medlatec has 24,762 official IDs, 5,004 attempted IDs, 4,992 successful/usable documents, and 98,333 current chunks. Reaching 10,000 requires exactly 5,008 deterministic unattempted IDs; the nested 7,500 checkpoint requires the first 2,508 of those IDs. The projection estimates 5,000 new usable documents, 98,491 new chunks/embedding rows, 3.015 GiB retained growth including both union checkpoints, a 4.266 GiB transient peak, and about 4.52 hours end to end. With 22.24 GiB free, projected minimum free space is 17.97 GiB, so preflight correctly failed. The ablation packager also crossed the disk floor through transient pagefile commitment; its incomplete Vinmec ZIP was quarantined and no ablation was marked ready.
What failed / surprised us: Even a zero-inference package can create several GiB of Windows private/pagefile commitment. The earlier ranking-change comparator amplified this by materializing two text-heavy JSON submissions; it is now streaming, but current disk headroom is still insufficient for a scientifically complete run.
What we learned: Disk preflight must include model-independent packaging, pagefile/transient commitment, and both requested depth checkpoints—not just retained crawl data. A ZIP from a first pass is not ready without deterministic regeneration and the mandatory audit.
Decision: Do not launch Medlatec acquisition or resume ablation packaging until D: has at least 26 GiB free. Preserve the deterministic manifest and quarantine the incomplete ZIP under `99_REVIEW_REQUIRED`.
Next question: After storage is reclaimed, do the three leave-one-source-out scores identify the next source, and does Medlatec continue scaling from 5K to 7.5K and 10K?
Cost/resources: No network acquisition or model inference. CPU-only packaging was stopped when the disk floor was crossed; all organizer-valid parent artifacts remain unchanged.
Evidence: `artifacts/source_census/phase10f_medlatec_10000_manifest.json`, `artifacts/source_census/phase10f_medlatec_10000/preflight.json`, and `artifacts/source_census/phase10f_source_ablations_status.json`.
Commit: this checkpoint.

## Phase 10F — next-source ablations completed after storage recovery

Date: 2026-10-11
Phase / commit: Phase 10F source attribution and focused scaling / this checkpoint
Question: Which of `vinmec.com`, `tiemchunglongchau.com.vn`, or `hellobacsi.com` contributes the largest current marginal organizer value within the validated nine-source union?
Why we tried it: Medlatec already has strong organizer-visible marginal value. Three zero-acquisition leave-one-source-out controls can identify the next scaling candidate without spending crawl or model budget.
Hypothesis: Removing a valuable source will produce an organizer-visible loss even though local ranking movement alone is not relevance evidence.
What we changed: Recomputed full candidate competition and rankings three times, excluding exactly one declared source in each experiment.
What stayed fixed: All 1,200 queries, the other eight source corpora, pilot control, calibrated `m=8`, retrieval/fusion policy, exact embedding and reranker caches, top-10/top-20 policy, source-verbatim provenance, and deterministic tie-breaking.
Result: All three packages passed structural validation, full-union replay, exact cache reconciliation, provenance, and deterministic regeneration and reached `READY_FOR_LEADERBOARD`. Minus Vinmec has 31,110 usable documents/381,225 chunks and changes top-10 documents for 1,053 queries; minus Tiemchunglongchau has 31,111/360,675 and changes 1,146; minus Hellobacsi has 31,109/354,888 and changes 889. No organizer results exist yet.
What failed / surprised us: The original attempt was stopped by transient pagefile pressure even though it required no model inference. After disk recovery, the fixed per-query streaming comparator completed all three packages at bounded memory. The quarantined first-pass Vinmec ZIP was byte-identical to the newly audited package, but it remains separately labeled as a historical duplicate rather than an upload candidate.
What we learned: Deterministic validation and comparison can consume meaningful transient resources independently of acquisition or inference. Local ranking-change counts remain diagnostics only; organizer scores are required for attribution.
Decision: Register exactly the three audited canonical ZIPs as upload candidates. Proceed with the separately authorized Medlatec 7.5K/10K acquisition only after its current disk preflight passes.
Next question: Which ablation loses the most organizer score, and does Medlatec continue scaling from 5K to 7.5K and 10K?
Cost/resources: Zero network requests and zero embedding/reranker inference; CPU-only cached ranking, streaming packaging, and validation.
Evidence: `artifacts/source_census/phase10f_next_source_ablations_report.json`, `artifacts/source_census/phase10f_source_ablations_status.json`, and the three Phase 10F pre-submit audits under `artifacts/validation/`.
Commit: this checkpoint.

## Phase 10F — Medlatec 7.5K/10K scientific-gate recovery

Date: 2026-10-11
Phase / commit: Phase 10F Medlatec focused scaling / this checkpoint
Question: Can the completed Medlatec depth run be finalized as two scientifically controlled full-union checkpoints without repeating its expensive stages?
Why we tried it: Acquisition, extraction, chunking, embeddings, source candidates, and the Medlatec-10K reranker cache had completed, but the mandatory control replay compared unlike corpora and the intermediate Medlatec-only source package returned only nine documents for one query.
Hypothesis: The score cache was valid; freezing the organizer-valid nine-source/Medlatec-5K control and rebuilding final competition over the actual nine-source 7.5K and 10K unions would repair experiment scoping without changing retrieval policy.
What we changed: Replaced the cross-wired Medlatec-only replay with an immutable full-nine-source control fixture; added nested deterministic 7,500-ID corpus construction; reused embeddings and scores by exact keys; inferred only the 823 genuinely missing 7.5K pairs; separated experiment-wide inference from packaging-only inference; and published readiness only after all scientific gates complete.
What stayed fixed: The 1,200 queries, other eight source corpora, calibrated `m=8`, BGE-M3 model/revision, BM25/RRF policy, pinned reranker, FP16 batch 2/max length 512, top-10 documents, top-20 source-verbatim chunks, tie-breaking, and provenance rules.
Result: The historical control replay is byte-identical (`fead3f75...459`). `FULL_MEDLATEC_7500` contains 38,710 official rows and 462,946 chunks; its package hash is `60b01c5982829b9b8c073b3003ca7ee41700e45005bad112d13e7a4c95ce221a`. `FULL_MEDLATEC_10000` contains 41,210 official rows and 512,356 chunks; its package hash is `bd83efe0cf909981cfa034be053dbd2dddf0223b0da00bf75ed608fbafcfa72d`. Both contain exactly 10 documents and 20 chunks for all 1,200 queries, pass provenance/determinism/cache audits, and are `READY_FOR_LEADERBOARD`. No organizer result exists yet.
What failed / surprised us: The old control replay used the new Medlatec source cache and compared a Medlatec-only ranking against the historical full-nine-source submission, causing 0/1,200 matches by construction. Query `1195` had only nine unique documents in the intermediate package because its pilot candidates represented four documents and fixed `m=8` Medlatec candidates represented five; no result was lost by an output deduplication bug. The intermediate source-cache package was therefore never a valid final leaderboard artifact.
What we learned: A control must freeze corpus membership and ranking inputs, not merely reuse the same code. Source-level caches are reusable building blocks, not final submissions. Inference accounting must distinguish model work performed for the experiment (823 new 7.5K pairs; 4,738 new 10K pairs) from zero inference during packaging.
Decision: Register both full-union depth checkpoints as current upload candidates and quarantine the Medlatec source-cache intermediate from the upload directory. Do not deepen Medlatec until organizer results establish the depth response.
Next question: How does organizer performance change from the known Medlatec-5K full-union FINAL `0.0383` to 7.5K and 10K?
Cost/resources: Heavy 10K stages were reused unchanged. The 7.5K checkpoint reused 147,243 embeddings and 8,777 exact reranker scores, requiring 823 new reranker inferences; final union packaging reused all 86,400 source scores at each depth.
Evidence: `artifacts/source_census/phase10f_full9_control_fixture.json`, `artifacts/source_census/phase10f_full_medlatec_7500_report.json`, `artifacts/source_census/phase10f_full_medlatec_10000_report.json`, and corresponding pre-submit audits under `artifacts/validation/`.
Commit: this checkpoint.

## Phase 12 — full PubMed release checkpoint

Date: 2026-10-13
Phase / commit: Phase 12 full PubMed multilingual retrieval / this checkpoint
Question: Can the completed full-PubMed arm be released without changing or rerunning any expensive scientific stage?
Why we tried it: Acquisition, processing, embeddings, dense retrieval, and all 603,587 exact reranker pairs were already durable; the final task was to reconcile provenance and package lifecycle semantics before spending a leaderboard submission.
Hypothesis: The existing ZIP is scientifically ready if its bytes match the audited package, both provenance closures are complete, the Phase 11C frozen control remains intact, and the preliminary package-stage status is correctly distinguished from final scientific readiness.
What we changed: Fixed Phase 12 document/provenance boundaries and public-chunk attribution in the completed recovery work, then normalized the package lifecycle so the provisional structural status is preserved as `package_stage_status` and the terminal status is published only after all scientific gates pass. No package bytes or model outputs were regenerated in this release checkpoint.
What stayed fixed: The NEW official corpus, 1,200 queries, full Phase 11C frozen control, 239,896 PubMed documents, 744,861 PubMed chunks, embeddings, 120,000 PubMed retrieval rows, 603,587 reranker scores, Docs@20/Chunks@40 output, and the existing submission ZIP.
Result: The existing 46,774,263-byte ZIP has SHA-256 `777316f54e563586c19eec59fa1598467f557b7db323d1c87eaff5c13f9e361`. It contains 1,200 queries, exactly 20 documents and 40 chunks per query, zero invalid IDs or duplicate outputs, complete document/chunk provenance and reconstruction, deterministic bytes, and a passing frozen Phase 11C control. PubMed contributes 5,913 document slots and 11,592 chunk slots across 932 and 1,039 queries respectively. The final state is `READY_FOR_LEADERBOARD`; no organizer result exists yet.
What failed / surprised us: The nested package object retained the preliminary `STRUCTURALLY_VALIDATED_NOT_SCIENTIFICALLY_READY` label even after the top-level scientific audit passed. This was lifecycle-reporting ambiguity, not a missing gate. Also, official-ID and structural unique-document counts differed because one counts only document outputs while the other includes chunk-owner documents.
What we learned: Provisional packaging state and terminal scientific readiness must be separate explicit fields, and every audit count must document its population scope.
Decision: This exact ZIP is the sole new Phase 12 upload candidate. Do not regenerate it, start another experiment, or record an organizer result until the human submits it and returns metrics.
Next question: What organizer score does the full-PubMed additive arm achieve relative to Phase 11C FINAL `0.0492`?
Cost/resources: Release verification only; zero new crawling, embedding, retrieval, reranking, or packaging inference.
Evidence: `artifacts/phase12_pubmed/phase12_pre_submit_audit.json`, `artifacts/phase12_pubmed/phase12_pubmed_attribution.json`, `artifacts/phase12_pubmed/document_provenance_closure.json`, and `artifacts/phase12_pubmed/provenance_closure.json`.
Commit: this checkpoint.

## Phase 12 organizer failure diagnosis and Phase 13 strategy

Date: 2026-10-13
Phase / commit: Phase 12 result → Phase 13 diagnosis / this checkpoint
Question: Why did adding the complete locally available PubMed block reduce every organizer metric, and what should change before more coverage is added?
Why we tried it: Phase 12 added 239,896 usable official English documents yet FINAL fell from `0.0492` to `0.0451`; document and chunk precision, recall, and F2 all declined.
Hypothesis: The failure is not explained by corpus size alone. The additive PubMed arm may have poor VI→EN candidates, cross-language reranker errors, or volume-sensitive max-chunk competition that displaces stronger Vietnamese evidence.
What we changed: Added a read-only deterministic diagnostic over frozen Phase 11C/12 submissions, rankings, candidate parts, SQLite scores, and official/local manifests. It produces per-query displacement, per-arm score/margin/rank analysis, exact coverage funnels, and a 60-query topic-stratified human review sheet. No retrieval or model artifacts changed.
What stayed fixed: Both submissions and ZIPs, 603,587 Phase 12 source scores, pilot scores, embeddings, candidate pools, canonical text, and official corpus membership.
Result: PubMed displaced exactly 5,913 Phase 11C document slots across 932 queries and 11,592 chunk slots across 1,039 queries. Every inserted slot was PubMed; retained Phase 11C items had zero relative-order inversions. PubMed supplied 349,187 reranked chunk pairs (median score `-5.0977`, per-query maximum median `0.6812`) versus 19,200 per focused source. Because documents are ranked by their highest-scoring chunk, candidate multiplicity creates an extreme-value opportunity advantage. This is observed volume-sensitive fusion, not proof that the reranker score scale itself is broken.
What failed / surprised us: Near-complete English coverage did not improve recall. Local coverage is now 239,896/240,000 identified English documents, but only 29,396 identified Vietnamese and 592 identified Chinese documents plus 5,702 unknown-language local documents. Coverage breadth and ranking quality are separate bottlenecks.
What we learned: More corpus can reduce quality when a large arm is allowed to compete through many more chunks per query. Hidden organizer gold prevents attributing error to dense retrieval versus reranking without human review.
Decision: Phase 13A will first replay document-level aggregation/fusion on existing scores and annotate the 60-query review set. No new acquisition, embedding, inference, or leaderboard submission is authorized.
Next question: Can source-neutral document aggregation, bounded representative chunks, and docs/chunks consistency recover Phase 11C quality while retaining genuinely relevant PubMed evidence?
Cost/resources: Read-only CPU analysis of existing Parquet/SQLite artifacts; zero network or model inference.
Evidence: `artifacts/phase13_diagnosis/ranking_displacement.json`, `source_score_analysis.json`, `coverage_funnel.json`, `human_review.csv`, `diagnosis_summary.md`, and `phase13_prioritized_plan.md`.
Commit: this checkpoint.

## Phase 13A — source-neutral aggregation/fusion replay

Date: 2026-10-13
Phase / commit: Phase 13A frozen-score diagnostics / this checkpoint
Question: Can document aggregation or source-fusion policy reduce harmful candidate-volume competition without new model computation?
Why we tried it: Phase 12 added 239,896 PubMed documents but reduced organizer FINAL from `0.0492` to `0.0451`; diagnostics showed 5,913 displaced document slots and a much larger PubMed scored-candidate arm.
What we changed: Replayed seven CPU-only aggregation/fusion variants over the frozen exact score universe. Canonical document ownership is now distinct from candidate-arm origin, multi-arm discoveries are deduplicated, and evidence chunks are selected independently from the valid ranked chunk pool as permitted by the organizer schema. The supervisor now supports an explicit `READY_FOR_HUMAN_REVIEW` terminal state and preserves fast-worker terminal updates.
What stayed fixed: Phase 11C/12 submissions, candidates, 695,539 unique exact query/chunk scores, embeddings, canonical provenance, and original evidence text. There was zero crawling, embedding, reranker inference, or leaderboard submission.
Result: Phase 11C and Phase 12 replayed exactly for 1,200/1,200 queries. All seven variants produced 24,000 document and 48,000 chunk slots with zero provenance mismatch. Top-2/top-3 document aggregation reduced PubMed document slots from 5,913 to 4,046/3,514, while RRF reduced them to 2,404 but displaced 12,567 Phase 11C document slots. These are diagnostics, not relevance results. No winner was selected.
What failed / surprised us: The completed worker exited `0`, but it did not publish a supported terminal state; the supervisor therefore labeled it `NEEDS_AGENT`. A second short-worker race overwrote its terminal update with a stale `STARTING` snapshot. Both lifecycle bugs are now regression-tested. The current reports preserve aggregate displacement and PubMed-query counts, but not the exact total number of changed queries for every non-control aggregation; recomputing that statistic alone is not scientifically necessary before review.
What we learned: Reducing PubMed presence is not evidence of improved relevance. Top-2/top-3 aggregation changes competition substantially, while source-arm RRF also disrupts many non-PubMed results. Human judgments are the remaining scientific gate.
Decision: The computation is complete and the run is `READY_FOR_HUMAN_REVIEW`. Use the blinded 60-query interface; treat unjudged items as UNKNOWN and do not select a fusion winner until labels exist.
Next question: Which aggregation/fusion policy improves judged precision and ranking quality on the pooled review set without degrading evidence quality?
Cost/resources: CPU-only replay, 4,224.60 seconds measured wall time, peak RSS 851,337,216 bytes, zero new model inference.
Evidence: `artifacts/phase13a_fusion/baseline_replay_audit.json`, `aggregation_comparison.json`, `fusion_comparison.json`, `document_chunk_consistency.json`, `candidate_ownership_audit.json`, `post_run_audit.json`, and `review_interface.html`.
Commit: this checkpoint.

Post-run UI recovery: Chrome `file://` testing exposed a generator escaping bug: Python emitted a literal newline inside the JavaScript `lines.join()` string, producing `Uncaught SyntaxError` before query rendering. After that fix, browser execution exposed a second JavaScript bug where a loop variable named `document` shadowed the DOM object. The repaired interface embeds JSON in a non-executable application/json block, paginates one query at a time, exports all 4,200 judgments with unanswered rows explicitly marked `UNKNOWN`, and displays a visible initialization error instead of failing blank. A real headless-Chrome `file://` regression selects `RELEVANT`, exercises CSV generation, and verifies all 60 queries remain addressable. No experiment/model artifact changed.

Organizer checkpoint: the single controlled Top-3-mean document candidate with Phase 11C chunks reached FINAL `0.0472`. This partially recovered Phase 12 (`0.0451`) but did not exceed Phase 11C (`0.0492`). Its document evaluation uses the frozen Phase 12 FULL candidate/score universe, while its chunks are exactly Phase 11C; comparison against Phase 11C therefore includes corpus expansion plus aggregation and is not a pure aggregation ablation. No additional ranking variant is authorized.

## System architecture upgrade — Milestone 1 audit and offline evaluation foundation

Date: 2026-10-14
Phase / commit: architecture audit and evaluation foundation / this checkpoint
Question: What is actually searchable/indexed/retrieved today, where are measurable bottlenecks, and can architecture changes be selected without spending leaderboard submissions?
Why we tried it: Phase 12 added near-complete PubMed coverage but reduced every organizer metric, and Phase 13A recovered only part of the loss. Repeated isolated leaderboard tuning cannot identify corpus, candidate, and ranking failures reliably.
What we changed: Added a streaming CPU-only system audit, deterministic data-quality sampling, a provenance-valid silver benchmark builder, and a reusable known-positive evaluator for Recall@20/@100, MRR, candidate coverage, source diversity, and runtime. Produced implementation/target-architecture specifications grounded in the frozen Phase 11C/12 artifacts.
What stayed fixed: Official and local corpora, embeddings, BM25/dense indexes, reranker caches, Phase 11C/12/13A outputs, canonical evidence, and all model configurations. No crawling, download, embedding, inference, index rebuild, package, or leaderboard action occurred.
Result: The NEW official corpus has 1,194,469 IDs across 93 domains. The deduplicated local usable union has 275,586 documents and 1,268,343 chunks: 35,690 broad-existing plus 239,896 PubMed, leaving 918,883 official IDs without local usable content. There is no unified index: focused content uses per-source dense+BM25, broad-extra uses global BM25, and PubMed uses a separate 239,896-document BGE-M3 dense matrix. Lightweight quality flags found navigation/boilerplate in 1,201/35,690 broad documents, short-review candidates in 797 broad and 233 PubMed documents, and 284 exact duplicate-content groups; these are review flags, not automatic rejection. The silver builder found only 21 retained source-title/answer pairs (4 VI, 17 ZH); all queries are verbatim source titles, with no trustworthy EN or VI→EN track.
What failed / surprised us: Authentic Q&A extraction does not provide a broad clean benchmark after NEW-corpus filtering. Many historical Q&A pages were removed, and surviving structured question sections can contain navigation/related-question contamination. The generated 21-pair set is valid for source-lookup regression but explicitly `INSUFFICIENT_FOR_ARCHITECTURE_SELECTION`.
What we learned: The strongest confirmed engineering bottlenecks are fragmented indexes and volume-sensitive candidate competition; the strongest coverage fact is the 918,883-document local gap. Which missing/retrieved/ranked documents are organizer-relevant remains UNKNOWN without labels. Official Recall@K cannot be computed locally.
Decision: **NO-GO for Milestone 2 retrieval-architecture implementation.** First expand a defensible held-out relevance foundation (especially Vietnamese and VI→EN), version the unified document registry, and run the evaluator in CI. Reassess GO only when the benchmark has sufficient source/language/topic coverage or adjudicated human judgments.
Cost/resources: CPU-only streaming audit; approximately 275K documents scanned, bounded samples, no GPU/network. The first implementation item is low-cost (hours of engineering, under 1 GiB additional working storage); unified BM25/dense builds require separate approval and resource measurement.
Evidence: `artifacts/system_upgrade_m1/`, `tools/system_upgrade_m1.py`, `tools/offline_evaluator.py`, and `src/evaluation/silver.py`.
Commit: this checkpoint.
## 2026-10-14 — System Upgrade Milestone 1.5

Milestone 1.5 added diagnostic infrastructure without crawling, model inference, index rebuilding, or leaderboard submission. The canonical registry contains 275,877 unique NEW-official documents with zero official-URL mismatches and valid document references. This is 291 higher than the M1 total because valid `ORIGINAL_PILOT` documents in the frozen Phase 12 candidate universe were not members of either the broad-existing or PubMed inventory.

The document candidate ledger contains 332,407 unique query-document pairs for all 1,200 queries. Per query, the frozen Phase 12 universe contains 229–316 unique candidate documents (median 278); 308,407 pairs do not reach Docs@20. This is structural truncation, not measured relevance loss. Phase 12 displaced 5,913 Phase 11C document slots and inserted exactly 5,913 alternatives.

The broader benchmark audit explained why M1 found only 21 title-derived pairs. After NEW-official filtering, four non-title structured question/answer candidates were found, all from a non-medical highway interview. No defensible primary VI→VI, VI→EN, EN→EN, or ZH→ZH held-out pairs were discovered. Title-derived candidates remain audit-only and the offline architecture-selection gate remains `INSUFFICIENT`.

## 2026-10-14 — Milestone 2A PubMed hybrid first-stage retrieval

Question: Can an independent disk-backed lexical arm add a materially different PubMed candidate signal to the existing BGE-M3 dense top-100 without new model inference?

Method: Built a deterministic SQLite FTS5 document index over 239,896 processed PubMed documents. For each Vietnamese query, the frozen dense top-100 supplied a bounded seed set; discriminative English terms were selected using corpus document frequency and used for an independent lexical top-100. Dense and lexical candidates were deduplicated by official document ID, while retaining arm ranks. At most two original evidence chunks per document were selected for cost accounting.

Result: All 1,200 queries produced lexical candidates. The lexical arm contributed 97,811 query-document pairs not present in dense top-100; 22,189 pairs overlapped. The bounded evidence pool contains 435,558 exact query/chunk pairs: 239,954 already have exact cached reranker scores and 195,604 require new inference. Provenance and ownership checks passed. The index occupies 678,948,864 bytes; measured incremental peak RSS during construction was 271 MiB.

Limitations: Candidate novelty is not relevance. The Vietnamese-to-English bridge is pseudo-relevance feedback derived from dense seeds and may drift; it requires trace review before GPU scoring. The 21-pair title-derived silver set is insufficient evidence of general retrieval quality. No reranker inference or leaderboard package was produced in M2A.

Decision: M2A is operationally complete. M2B may proceed only after query-bridge trace review, an exact dense-only Phase 12 replay, exact missing-score reconciliation, and resource preflight.

## 2026-10-14 — Milestone 2B controlled hybrid reranking gate

The mandatory pre-inference trace review blocked M2B. Across the deliberately low-overlap M2A trace sample, pseudo-relevance feedback repeatedly replaced the Vietnamese medical intent with rare terminology from imperfect dense seeds: tonsillectomy became hymen/endometriosis, meningitis-related paralysis became hemiarthroplasty/Alzheimer topics, a liver ultrasound nodule became thyroid nodules, and chronic colitis treatment became biliary chemotherapy/H. pylori. Some traces (vitamin E; gout/COVID vaccination) preserved topic, so lexical novelty is not declared useless; relevance remains UNKNOWN. Because drift recurred across symptom, disease, treatment, and short-query groups, no GPU gate, reranker inference, ranking merge, package, or submission was run. Exact M2B new inference is zero.

### M2A bridge repair decision

A CPU-only repair replaced seed-derived TF-IDF terms with versioned anchors whose English form is either copied literally from the Vietnamese query or documented by an explicit bilingual phrase in a retained local medical source. It retains deterministic FTS5 retrieval and falls back exactly to Dense Top-100 when no anchor exists. On 30 frozen high-risk traces plus an independently SHA256-selected 30-query stratified sample, only 15/60 queries had a traceable anchor (12/30 high-risk, 3/30 independent); 45/60 therefore fell back. The repaired bridge introduced zero untraceable expansion terms and fixed the known topic-replacement examples mechanically, but its local lexicon is too sparse to form a meaningful independent lexical arm. M2B remains blocked; no GPU/model inference, index rebuild, package, or submission occurred.

### Bounded VI→EN translation feasibility gate

The only candidate, `Helsinki-NLP/opus-mt-vi-en`, was pinned to official revision `c8d2853e77f5fae31124d993e0b35176b1c8914e` under Apache-2.0. No local OPUS/Marian model was present. The frozen 30 high-risk plus 30 independent queries and the existing PubMed FTS5 index were preserved unchanged. Disk preflight passed with 130.539 GiB free, but physical RAM fell from 1,036.7 to 772.2 MiB during the audit. The existing Python runtime measured 376.4 MiB RSS after importing Torch and Transformers before adding the 289 MB FP32 weights, load-time duplication, generation buffers, or an evidence-based reserve. The gate therefore stopped before download/load; no translation, retrieval, model inference, or candidate artifact was produced. Medical concept preservation and contradiction rates remain UNKNOWN. Resume requires at least 1.25 GiB free physical RAM, with 1.5 GiB recommended from the observed 264.5 MiB system fluctuation. Until then, direct-translation rollout is BLOCKED and no further TF-IDF bridge patch is authorized.

### Colab A100 reranker portability proof of concept

A bounded Windows-to-Colab harness now exports exactly 200 existing scored query/chunk pairs from the immutable Phase 12 cache. The deterministic sample contains 120 language/available-length-stratified rows plus 80 query-local close-score rows; observed coverage is 84 English, 71 Vietnamese, and 45 Chinese pairs. Chinese candidates contain no passage of at least 1,200 characters, so the sample uses only the real Chinese short/medium strata instead of fabricating symmetry. The portable ZIP validates CRC, exact keys, finite reference scores, text SHA256 values, and the pinned `BAAI/bge-reranker-v2-m3` FP16 batch-2 contract. Local import is dry-run and production-cache read-only by design. Status remains `AWAITING_COLAB_RESULT`: no A100 score or performance measurement exists yet, so equivalence is explicitly not passed. No crawl, embedding, retrieval, reranking, submission generation, or frozen artifact modification occurred.

### Phase 14 acquisition recovery

The first Phase 14 launch failed before acquisition because it reused the Phase 10E worker while its configuration did not yet satisfy the closed Phase 10D/E loader. A corrected second launch acquired 856 durable rows, then Windows Event 1074 recorded an unplanned system shutdown at the exact final checkpoint timestamp; the old loader traceback was therefore historical, not the reason the live worker disappeared. Recovery introduced a dedicated `phase14_medlatec_coverage` adapter without weakening the shared validator. It verifies the immutable 3,000-ID selection and SHA256, audits SQLite read-only, exposes both `selected/completed` and `total/done`, and resumes through the existing crawler checkpoint. The retained database passed integrity with 855 successes and one HTTP error; a supervised resume advanced safely from 856 to 858 with no repeated completed IDs. Embedding and reranking remained stopped.

### Phase 14 extraction/chunking and remote embedding gate

The bounded 3,000-ID acquisition closed with 2,996 successful archived bodies and four HTTP 404s. The dedicated resumable adapter produced 2,996 usable documents and 58,839 unique, source-reconstructable chunks without touching Phase 11C, PubMed, embeddings, or reranker caches. A structural audit found 32,893 exact-duplicate chunk slots: 32,762 are site boilerplate or related-article navigation and therefore retrieval-noise risks, while 131 are potentially legitimate repeated source content. No rows were removed because corpus membership remains the experiment variable.

A 200-row BGE-M3 Colab equivalence package now uses only old chunks with exact reference vectors and preserves the historical aligned batch-4 context. Its verifier retains the measured Phase 10E gates (`max_abs <= 0.000732421875`, `min_cosine >= 0.9995`) and is read-only. Status is `AWAITING_COLAB_EMBEDDING_EQUIVALENCE`; no remote embedding result exists yet, and the separate reranker A100 equivalence remains failed.

Follow-up audit found that the first package treated consecutive pending rows as the historical batch, while SentenceTransformers 5.7.0 actually length-sorted each Phase 10E pending list before slicing batch 4. The corrected package reconstructs `np.argsort(-len(normalized_text))` within every 128-row historical window after excluding reused rows, selects 50 complete real batches, and makes 50 isolated four-row Colab encode calls. Transformers 4.48.3 could not activate the optional flatten/interleave path. The measured numerical thresholds were not changed, no GPU inference ran locally, and readiness remains `AWAITING_COLAB_EMBEDDING_EQUIVALENCE`.

### Phase 14 final Medlatec coverage result

Phase 14 added a bounded 2,996 usable official Medlatec documents and 58,839 chunks to the frozen Phase 11C architecture. Local BGE-M3 embedding and the pinned reranker completed transactionally; 250,114/254,400 exact scores were reused and 4,286 were newly inferred. The reranker control initially failed because the eight unchanged target pairs were regrouped into different FP16 padding batches; replaying their exact historical batch-size-two contexts restored 8/8 equivalence with maximum difference `0.0` without changing tolerance or historical scores. Final canonical closure recovered 15 valid historical Medlatec document/chunk owners from the Phase 10F source artifacts; no ranked ID was removed or fabricated.

The final package passed 1,200-query Docs@20/Chunks@40 validation, official-ID/provenance closure, frozen Phase 11C hashes, score-cache audit, deterministic regeneration, and ZIP integrity. It inserted 1,884 new Medlatec document slots and 2,767 new chunk slots; zero flagged boilerplate/navigation chunks reached the final output. Organizer-confirmed results are FINAL `0.0533`, Docs F2 `0.0536`, Chunks F2 `0.0529`, Docs precision/recall `0.1458/0.0503`, and Chunks precision/recall `0.1297/0.0507`. This is the new frozen BEST; it is organizer evidence, not an offline estimate.

### Phase 15A — controlled ranking replay closed without a winner

Phase 15A reproduced frozen Phase 14 exactly for all 1,200 queries and tested
two CPU-only variants with zero inference. `DOC_TOP2_MEAN` changed 1,196 queries
and 1,614 document slots while all 48,000 Phase 14 chunk slots stayed fixed.
`CHUNK_DIVERSITY` changed 1,156 queries and 6,952 chunk slots while all 24,000
Phase 14 document slots stayed fixed. Both variants passed deterministic,
official-ID and provenance gates. No trustworthy human labels or adequate
held-out silver benchmark exists, so structural changes cannot select a
relevance winner. Status is `NO_WINNER_PENDING_RELEVANCE_EVIDENCE`; no ZIP was
created and Phase 14 FINAL `0.0533` remains frozen BEST.

The single Phase 15B plan is a 120-query, same-language Vietnamese unified
document-BM25 pilot over existing local content. It does not repeat failed
PubMed PRF, sparse bilingual anchors or blocked translation. It must integrate
the shared overall/stage progress contract before launch and performs no crawl,
embedding or reranker inference in the pilot.

### Phase 15B — unified Vietnamese document BM25 pilot

The CPU-only pilot indexed 37,786 provenance-valid NEW-official Vietnamese
documents across 38 domains in a 186,785,792-byte contentless SQLite FTS5
index. All 120 deterministic stratified queries returned Top-100: 12,000 rows,
9,329 distinct documents, and 10,254 query-document pairs absent from the
Phase 14 candidate universe. Every query had lexical-only candidates (52–100,
median 86). Mean Phase 14 overlap was 14.55/100 and mean domain diversity was
8.98/query. The bounded evidence estimate contains 23,973 exact keys; 1,607
are reusable and 22,366 are unscored. No inference ran.

All official-ID, provenance, ownership, determinism, index-integrity, resource,
and frozen Phase 14 gates passed. Exact duplicate-content exposure was
82/12,000 slots; a Phase15B-specific boilerplate mapping was not measured and
must not be inferred from the Phase 14 35-row diagnostic. Novelty and diversity
are structural evidence only. Existing relevance assets remain insufficient:
the 21 silver pairs are title-derived (only four VI), and there is no verified
external benchmark mapped to NEW-official IDs. Decision: GO only for a bounded
automatic evidence gate; NO-GO for reranking, 1,200-query scale-up, packaging,
or any F2 claim.

## 2026-10-22 — Multilingual M2 development vertical slice

Ran a CPU/cache-only development experiment using 167 imported ChatGPT labels
strictly as `AI_JUDGED` silver evidence; `human_labels.csv` and held-out review
data were not used. A Chinese extraction gate removed 21/32 reviewed ZH rows,
all silver label 0, while preserving every known positive. A quality-filtered
character-bigram Chinese FTS branch still produced mostly same-topic rather
than answer-bearing evidence, so no direct ZH relevance gain is claimed.

For English, the frozen query-only medical bridge corrected errors such as
`nhồi máu não` being translated as `heart attack`. It found 88 lexical pairs
not in the M1 lexical Top-20. Query 752 is the clearest new result: PubMed doc
5832210 directly links antihypertensive medication to erectile dysfunction,
where all five old reviewed candidates were silver label 0. A naïve RRF union
retained only 7/24 old pooled positives versus 9/24 for baseline lexical, so it
is rejected as a production fusion policy. Evidence:
`artifacts/multilingual_retrieval_m2/M2_VERTICAL_SLICE.md` and
`m2_vertical_slice_report.json`. Phase29 SHA remained unchanged and Phase27
was not modified.

## 2026-10-23 — Multilingual M3.1 evidence-quality repair

On the 12 frozen VI→ZH development queries, the 240 M3 silver judgments were
decomposed into 147 directory/navigation pages, 40 topic-only chunks, 39
wrong-topic chunks, 13 partial answers and one extraction-granularity failure.
M3.1 added structural page typing, bounded source-verbatim chunks, traceable
quality decisions and entity-first Chinese admission. It returned 22 candidates
for six queries and abstained on six rather than padding Top-20 with context-only
hits. `AI_JUDGED_M3_1` review found 5 partial and 17 wrong results, but zero
direct answers; the sole mechanical direct-gate row was a population and
bilirubin-type mismatch. No ZH relevance gain is claimed. The residual blocker
is answer-content coverage for the requested disease–relation pairs. English
protected fusion retained 24/24 baseline positives and 2/2 corrected-bridge
direct positives. Phase29 remained hash-identical and Phase27 was untouched.
Evidence: `artifacts/multilingual_retrieval_m31/M31_REPORT.md`.

## 2026-10-23 — M4 proves one current-official VI→ZH answer retrieval

M4 froze a 50-URL targeted `cnkang.com` tranche from the current official
snapshot and acquired/extracted 50/50 pages (29.4 URL/min) without GPU work.
Its query-text-only medical concept transfer, causal synonym expansion,
navigation-tail-aware extraction and separate candidate/admission traces found
one direct Chinese answer on the frozen development set: query 128 retrieved
official doc 1595280 at final rank 5. The verbatim passage explains sneezing
and rhinorrhea that are strongest in the early morning or on waking and gives
a meaningful rhinitis differential. This is `AI_JUDGED_M4` silver evidence,
not `HUMAN_VERIFIED` or an official recall result.

Among newly acquired Top-20 rows, silver review recorded 1 direct, 2 partial
and 7 wrong/mismatched. All 13 earlier M3 partial/direct query-document pairs
remain in the candidate trace; 11 enter final Top-20 and two are transparently
rejected at answer admission. The protected English control remains 24/24
baseline positives plus 2/2 corrected-bridge direct positives. Twenty-one
tests, current-official membership, canonical URLs, source-verbatim evidence
and deterministic replay passed. Phase29 remained hash-identical and Phase27
was not modified. Evidence: `artifacts/multilingual_retrieval_m4/M4_REPORT.md`.

## 2026-10-23 — M4.1 query-text retrieval and evidence reassessment

Implemented a reusable VI→ZH `retrieve_by_text()` path on the frozen 368-doc,
2,188-chunk corpus. Longest boundary-aware terminology matching, explicit
unmapped spans, scoped negation and candidate/admission diagnostics replace
the benchmark-ID-dependent runner. Four new raw-text functional questions ran
without relevance labels. The twelve frozen development units and replay /
qualifier ablation ran in 8.40 seconds; full CPU forensics took 27.78 seconds,
with 358.86 MiB peak working set in the final run. Thirty-six tests passed.

The unblinded source reassessment narrows M4's claim: query 128/doc 1595280 is
a useful morning-rhinitis differential, not a complete explanation of the
patient's awakening. Its original AI-silver 2 remains frozen; a separate
provisional audit assigns 1. No independent direct-answer gain is established.
All 13 known M3 partial document pairs remain in candidate diagnostics, 12 at
Top-100, but only six in final Top-20 versus M4's eleven. This is not promoted
as a relevance improvement. Cache forensics recovered 19 exact-input score
references across eight queries that ID-only matching had missed. M4's 31
`/nkwk/ggths` pages concern femoral-head necrosis, not GGT. Source absence
remains UNKNOWN where only lexical failure is observed.

Phase27 was active, so semantic inference is deferred with a bounded trial
plan; no model was loaded. EN final positives were recounted at 24/24 plus
2/2, and Phase29 / frozen M4 hashes remained unchanged. Decision: GO for
query-text functionality and independent source review; NO-GO for current
policy promotion or direct-answer/generalization claims. Evidence:
`artifacts/multilingual_retrieval_m41/M41_REPORT.md` and
`M4_EVIDENCE_AUDIT.md`.

## 2026-10-23 — M4.2 bounded semantic pilot preparation, inference deferred

Phase27 remained active at PACKAGE_PASS2, so M4.2 performed CPU-only preparation
and exited without loading BGE-M3/reranker. Froze the 12 development query
representations, 2,188-chunk order and 20-text resource-trial manifest. Added
tested Dense ranking / RRF / equal-budget comparison and blinded-queue utilities.
Reverified and separately checkpointed 19 exact cached scores, with conflict
checks and no duplicate inserts on resume. Pinned local tokenizer artifacts
were hashed; historical tokenizer byte hashes and new empirical equivalence
are not available and are not claimed.

The specified six-row assistant review CSV was absent from the workspace;
zero labels were imported and the reported 3/3/0 totals were not fabricated
into row-level judgments. A strict separate importer is ready. Fifty tests
passed. Semantic trial, new embeddings, reranking and relevance comparison
remain NOT_RUN. M4/M4.1, EN and Phase29 are frozen. Decision: defer inference
until Phase27 exits and import the review when its actual file is available.
Evidence: `artifacts/multilingual_retrieval_m42/M42_STATUS.md`.

## 2026-10-23 — Phase27 frozen final relevance review

Cached-only comparison verified the Phase27 final ZIP SHA b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e
against immutable Phase29. Docs changed on 983 queries, chunks on 1,087;
2,030 document slots and 6,561 chunk slots are new-tranche additions.
Hash-stratified review of eight changed development queries found mixed
complementary evidence and concrete wrong-topic/promotional/redundancy risks,
including loss of a glucose250–300mg/dL passage to repeated hypoglycemia Q&A.
All 12 baseline-present development silver-positive document pairs remain;
this does not verify passage retention or official relevance. Recommendation
HOLD for one scarce submission, not a claim of lower organizer score.
Four diagnostic tests pass. No held-out labels, new inference, pipeline
execution or submission. Both ZIP hashes unchanged.
Evidence: `artifacts/phase27_final_relevance_check/RELEVANCE_REVIEW.md`.

## 2026-10-23 — Phase27 organizer gain and bounded Phase27.1 refinement

User reports organizer FINAL0.0923 for Phase27 (USER_REPORTED_ORGANIZER_RESULT,
not directly API verified), versus Phase29 0.0874: +0.0049 (+5.61%). New pointer
is configs/current_best.json; both frozen ZIP hashes verified unchanged.
The earlier eight-query HOLD is superseded as a submission decision, while
its provisional failure examples remain diagnostics.

Two cache-only ablations on39 frozen development queries completed in10.33s.
Same-owner overlap/backfill changes194 chunk slots, retains12/12 known
baseline-present silver-positive documents and13/13 supported exact excerpts,
but still needs316 deferred padding slots and loses4 original Top5 objects
from final40. Conservative dominant-boilerplate rejection changes nothing.
NO-GO for a new submission: no independent evidence of relevance gain.
No ZIP, inference, crawl or held-out-label use.

M4.2 executable trial20 was invoked but correctly blocked before model load:
678.48MiB RAM available vs969.71MiB measured cold-start requirement, preserving
existing safety multipliers; VRAM3977MiB, commit10390.26MiB and disk34.47GiB pass.
The cold gate includes runtime-import cost absent from the357.11MiB warm gate.
No claimed semantic/equivalence result; Dense and reranker remain NOT_RUN.
Twenty-four scoped tests pass. Evidence:
artifacts/phase271_cached_refinement/PHASE271_REPORT.md.

## 2026-10-23 — M4.2 resume blocked locally; RTX3060 execution payload verified

Ran one existing strict inference preflight: RAM648.41MiB <969.71MiB, while
VRAM3977MiB, commit13655.70MiB and disk36.66GiB pass. No Phase27/model worker
was active. No unchanged retry, model load, Dense/reranker inference or crawl.
The dirty earlier resource report was preserved; new telemetry is separate.

Implemented target-runtime-bound Dense execution for the frozen2188 chunks and
24 query representations, checksummed missing-only windows, separate lexical
candidate/final arms, unchanged M4.1 replay validation and blinded review export.
Portable Windows payload contains only fixed-development data/code and exact
model/tokenizer/runtime hashes, not weights, secrets or held-out files. The19
historical scores remain reference-only until target-host numerical replay.
Tests and extracted archive checks cover frozen input identity, nineteen exact
reference text/owner hashes, resume/corruption, final-rank preservation and
path/tamper rejection. Twenty-two scoped tests pass.

Decision: GO to execute the same bounded Dense/RRF test on RTX3060; scientific
usefulness remains INCONCLUSIVE and no submission is justified. No remote
session is available, so target inference is NOT_RUN, not a claimed handoff.
Evidence: artifacts/multilingual_retrieval_m42/M42_RESUME_REPORT.md.

## 2026-10-24 — M4.2 Colab results verified and reviewed

Returned Colab ZIP CRC and10 extracted members match. Trial20 explicitly PASS
for Colab internal FP16 consistency only; A100/Linux/torch2.11+cu130 differs from
source laptop, so source-target equivalence remains NOT_VERIFIED and historical
reranker scores were not reused. Dense completed2212 rows across553 checksummed
SQLite windows, integrity/finite/normalization PASS. CPU arithmetic replay of
both Dense Top100 and RRF reproduces saved IDs/order/scores without new inference.
Actual candidate universe is361 official document owners/2188 chunks, versus
the historical368-document report; the seven-ID discrepancy remains unresolved.

Known13 AI-partial pairs, counts at Top20/Top100: VI-Dense3/12, ZH-Dense11/13,
RRF11/12, versus M4 lexical13/13 and M4.1 lexical12/12. These are pair
retention counts, not gold Recall. RRF changes165/239 common ZH document windows;
it can retain a document while choosing navigation rather than medical body.
Literal old-window Top20 retention is2 for ZH-Dense versus0 for RRF; rechunking
and whitespace differences are explicitly separated. Original13 pairs contain
11 distinct old evidence strings and only6 query units.

Reviewed all12 new queue excerpts as unblinded AI aids:1 partial/11 wrong/0
direct; no human verification or corpus-wide error-rate claim. Reranker NOT_RUN.
Recommend one cached, fixed-document window-selection ablation before more
source acquisition; NO-GO for promoting current RRF or claiming score gain.
Phase27 BEST0.0923 SHA verified unchanged; no inference rerun, crawl or submission.
Twenty-five scoped tests pass. Evidence:
artifacts/multilingual_retrieval_m42/colab_analysis/M42_COLAB_ANALYSIS.md.

### 2026-10-24 — M4.2 fixed-document evidence-window ablation

Executed cached CPU-only A/B over all12 development queries: A saved RRF versus
B same-owner maximum true cosine to frozen Chinese-query vectors. All1200
document positions/scores unchanged;166/240 Top20 and553/1200 Top100 windows
change. Old AI-partial document retention stays11@20/12@100; literal window
retention0->2, normalized1->3. Navigation proxies rise54->79@20 and294->366@100,
not clinical error labels. Eight preselected unblinded AI diagnostics show one
partial splenic-mechanism gain and seven ties; two additional fixed-queue
contrasts expose loss of pain-differential/nasal-treatment context. No independent
direct-answer improvement or HUMAN_VERIFIED result is claimed.

Frozen24-pair blind queue spans all12 queries/eight depth-navigation strata and
remains pending independent review. Inputs/official ownership/553 SQLite windows
PASS; byte-identical output replay and exact source-text CSV roundtrip PASS.
Final runtime1.638s, process peak RSS344.43MiB, sampled private782.49MiB,
minimum available306.56MiB.31 scoped tests PASS. NO-GO automatic promotion;
next step independent paired source review, not acquisition or more inference.
Phase27 BEST0.0923 ZIP SHA unchanged; no held-out/model/reranker/crawl/submission.
Report: artifacts/multilingual_retrieval_m42/window_ablation/M42_WINDOW_ABLATION.md.
