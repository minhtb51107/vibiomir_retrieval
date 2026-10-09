# ViBioMIR Project Timeline

This timeline records direction-changing milestones only. Dates and outcomes
come from Git metadata, tracked reports, artifacts, or organizer values supplied
by the project owner. The current Git history contains no Phase 0 checkpoint,
so its date and commit are explicitly unknown.

| Date | Phase / commit | Direction-changing outcome |
|---|---|---|
| UNKNOWN | Phase 0 / commit UNKNOWN | Project harness existed before the currently visible history; no reliable reconstruction is possible. |
| 2026-10-03 | Phase 1 / `24ff790` | Established the real scale: 1,200 queries, 4,394,718 URLs, 97 domains, and a highly concentrated corpus. |
| 2026-10-03 | Phase 2A–2C / `b0354fc`, `ccb79bf`, `aba8455` | Built resumable, robots-aware acquisition and measured production crawl behavior without starting an unrestricted crawl. |
| 2026-10-03 | Phase 3 / `d3756d2` | Converted the bounded crawl into provenance-preserving cleaned documents and deterministic chunks. |
| 2026-10-03 | Phase 4 / `c596b3d` | Created the BGE-M3 + exact FAISS dense baseline over 5,643 validated pilot chunks. |
| 2026-10-03 | Phase 6 / `d5d06aa` | Added multilingual lexical BM25 and deterministic RRF; Phase 5 remained deferred because no trustworthy labels existed. |
| 2026-10-04 | Phase 7 / `d363376` | Added the multilingual cross-encoder reranker with crash-safe SQLite scoring. |
| 2026-10-04 | Phase 8 / `a075202` | Capacity evidence blocked exhaustive local production: expected storage and crawl duration exceeded laptop resources. |
| 2026-10-05 | Phase 8C / `34d7f2b` | URL-only query-conditioned reduction was rejected after destructive pilot-candidate retention. |
| 2026-10-05 | Phase 9 / `abf2bf7` | Produced the first valid organizer submissions; the first score (`0.0001`) exposed corpus coverage as the dominant problem. |
| 2026-10-05 | Phase 10A / `3c228ff` | Calibration showed depth alone did not repair recall; source-verbatim expansion helped chunk precision, not coverage. |
| 2026-10-05 | Phase 10B0 / `86c1462` | General Bing RSS discovery was technically reliable but mapped only 21 unique official documents and was rejected. |
| 2026-10-05 | Phase 10B1 / `b66c5b4` | Site-native discovery failed its predeclared weighted-coverage gate; the reusable 97-source map redirected work toward source acquisition. |
| 2026-10-05 | Phase 10B2 / `603c22a` | Bounded acquisition identified S4 as the safest source: 99.8% usable and about 59 usable documents/minute. |
| 2026-10-06 | Phase 10B3 / `fa9128b` | Organizer results separated local movement from relevance: S3/S5 moved many rankings without gain, while S4 reached `0.0004` and S2 reached `0.0002`. |
| 2026-10-06 | Phase 10C0 / this phase checkpoint | Cheap within-S4 targeting failed; the next 4,000-document S4 batch should remain broad and deterministic rather than metadata-targeted. |
| 2026-10-06 | Phase 10C1 / uncommitted | Organizer-confirmed S4-5K reached `0.0014` final score with precision and recall gains over S4-1K, supporting deeper bounded S4 testing without implying linear scaling or majority-gold coverage. |
| 2026-10-06 | Phase 10C1 / uncommitted | Organizer-confirmed FamilyDoctor S2-5K kept final score at `0.0002`; small document-side gains but flat chunk recall deprioritized further S2 scaling and redirected effort to adaptive source census. |
| 2026-10-07 | Phase 10D Round 1 / uncommitted | Organizer group tests separated the census: A reached `0.0016`, B `0.0008`, and C `0.0017`. B was deprioritized; all 42 A+C sources survived because C's `0.0001` edge was not sufficient evidence of superiority. |
| 2026-10-07 | Phase 10D Round 2 / uncommitted | Seven six-source tests narrowed the 42 A+C survivors to 18 sources in G1, G5, and G6; G2 remains reserve-only, while G3/G4/G7 were deprioritized. No individual source was credited from group-level evidence. |
| 2026-10-08 | Phase 10D Round 3 / uncommitted | Organizer-confirmed shallow trio scores did not justify further source elimination. The plan skipped depth 300 and moved all nine G1A/G5A/G6B sources to a controlled up-to-1,000-ID depth-scaling test. |
| 2026-10-08 | Phase 10D depth1000 repair / uncommitted | A byte-identical shallow replay exposed ambiguous SQLite cache seeding: all 86,400 depth candidate rows inherited one score. Three submitted depth results were marked invalid and isolated for exact-key repair. |
| 2026-10-09 | Phase 10E launch / uncommitted | Corrected G5A depth1000 reached organizer FINAL `0.0037` versus shallow `0.0009` (about 4.1x), moving the project from source census to one bounded focused-scaling checkpoint at up to 11,200 G5A official IDs. |
| 2026-10-09 | Phase 10E completion / uncommitted | Built the 11,200-document/92,768-chunk G5A union, recovered an inherited calibration-contract omission and a deterministic FP16 batch-shape equivalence false alarm, then passed the mandatory scientific gate. The package is ready; no organizer score exists yet. |
| 2026-10-09 | Phase 10E organizer checkpoint / pending commit | Organizer results confirmed continued G5A scaling (`0.0037` to `0.0085`) and strong corrected G1A/G6B depth-1000 results (`0.0062`/`0.0045`). G1A had the highest observed yield per document among the tested priority trios, so the next authorized checkpoint became G1A at up to 5K Medlatec + all 10 v.familydoctor + 5K Vinmec; G5A 21.2K and G6B 15K remain deferred. |
| 2026-10-09 | Phase 10E G1A preflight / pending commit | The exact G1A delta was 8,005 official IDs (`4,004/0/4,001` for Medlatec/v.familydoctor/Vinmec), but the preflight correctly blocked launch: 20.37 GiB free would fall to a projected 16.46 GiB, below the 20 GiB safety floor. No crawl or inference started. |
| 2026-10-09 | Phase 10E G1A preflight recovery / pending commit | After free space increased to 30.11 GiB, the unchanged experiment projected 26.20 GiB remaining and passed every fixed-contract and dependency gate. The prepared 8,005-ID unattended acquisition was authorized without rebuilding prior artifacts. |
| 2026-10-09 | Phase 10E G1A embedding recovery / pending commit | A PyTorch `c10.dll` access violation exposed whole-corpus host-memory amplification and non-transactional embedding output. Acquisition/extraction/chunking were preserved; a row-checkpointed streaming implementation passed bounded memory/equivalence gates, deferred the 537 MiB mapping until after model load, and replaced the heavy orchestrator with a standard-library-only continuation parent at the GPU boundary. |
| 2026-10-09 | Phase 10E G1A rerank recovery / this checkpoint | Post-reboot audit preserved complete embeddings and 13,432 exact reused scores. A falsely global G5A sample-key requirement was replaced by deterministic experiment-scoped key/text controls while retaining the shared pinned scorer contract and measured FP16 tolerance. |
| 2026-10-09 | Phase 10E G1A organizer checkpoint / `65f9469` successor | G1A at 10,010 official documents and 131,166 chunks reached organizer FINAL `0.0234` (`DOCS_F2=0.0245`, `CHUNKS_F2=0.0224`), up from corrected depth1000 FINAL `0.0062`. This confirms strong G1A depth scaling and authorizes the already-prepared G6B 15K reserve checkpoint; no deeper G1A run will execute concurrently. |
| 2026-10-10 | Phase 10E G6B organizer checkpoint / pending commit | G6B at 15,000 official IDs (14,908 usable documents, 190,102 chunks) reached organizer FINAL `0.0140`, up from corrected depth1000 FINAL `0.0045` (about 3.11x). Both G1A and G6B now have independent positive scaling evidence, so the next experiment is a zero-acquisition union test of complementarity rather than deeper crawling. |
