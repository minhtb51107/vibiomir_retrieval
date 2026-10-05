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
