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
