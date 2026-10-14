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
| 2026-10-10 | Phase 10E G1A+G6B union / this checkpoint | Combined the two organizer-valid source families without acquisition or model inference. Full cross-source ranking over 25,010 official rows and 321,268 chunks reused 57,600 exact reranker scores, reproduced both parent controls byte-for-byte, and reached `READY_FOR_LEADERBOARD`; organizer result remains pending. |
| 2026-10-10 | Phase 10E G1A+G6B organizer checkpoint / pending commit | The no-acquisition union reached organizer FINAL `0.0332` versus G1A alone at `0.0234`, confirming complementarity. The next experiment adds organizer-valid G5A 11.2K through the same full-ranking union path, without new acquisition or inference. |
| 2026-10-10 | Phase 10E nine-source union / this checkpoint | Added organizer-valid G5A to G1A+G6B without acquisition or inference. Full ranking over 36,210 official rows and 414,036 chunks reused 86,400 exact scores, reproduced all four prior controls byte-for-byte, and reached `READY_FOR_LEADERBOARD`; organizer result remains pending. |
| 2026-10-10 | Phase 10E nine-source organizer checkpoint / pending commit | The full union reached organizer FINAL `0.0383`, up from G1A+G6B at `0.0332`. The last three pre-reset slots were reserved for two leave-one-source-out probes and one shallow Round-3 resurrection union to maximize information for the next scaling decision. |
| 2026-10-10 | Phase 10E final pre-reset probes / this checkpoint | Prepared three no-acquisition, zero-inference source-membership probes: full minus Medlatec, full minus Suckhoedoisong, and full plus nine shallow Round-3 resurrection sources. All replayed the validated full-union control exactly and passed scientific, provenance, official-ID, and deterministic packaging gates; organizer results remain pending. |
| 2026-10-10 | Phase 10F attribution checkpoint / this checkpoint | Organizer ablations measured Medlatec's current marginal FINAL contribution at `0.0095` and Suckhoedoisong's at `0.0032`; adding the shallow Round-3 resurrection set left FINAL at `0.0383`. Medlatec therefore receives the next bounded depth-scaling budget, while three cheap source ablations will identify the next source priority. |
| 2026-10-10 | Phase 10F preflight / this checkpoint | Prepared deterministic nested Medlatec 7.5K/10K checkpoints requiring 5,008 new IDs. The measured projection reached only 17.97 GiB minimum free space, so the 20 GiB gate blocked acquisition. A source-ablation packaging attempt was quarantined after transient private/pagefile commitment crossed the same floor; no package was marked ready. |
| 2026-10-11 | Phase 10F storage recovery / this checkpoint | With 163.61 GiB free, the unchanged 20 GiB gate passed. Three cached leave-one-source-out experiments (Vinmec, Tiemchunglongchau, Hellobacsi) completed with zero acquisition or model inference and reached `READY_FOR_LEADERBOARD`; organizer results remain pending. |
| 2026-10-11 | Phase 10F Medlatec gate recovery / this checkpoint | Corrected a catastrophically mis-scoped control that compared a Medlatec-only source cache with the historical full-nine-source experiment. The immutable Medlatec-5K/full-union control now replays byte-for-byte; full-union Medlatec 7.5K and 10K packages contain exactly 10 documents and 20 chunks per query and reached `READY_FOR_LEADERBOARD`. Organizer results remain pending. |
| 2026-10-11 | Phase 10F organizer checkpoint / this checkpoint | Medlatec depth continued improving organizer FINAL from `0.0383` at 5K to `0.0398` at 7.5K and `0.0426` at 10K. Further acquisition paused so Phase 11A can distinguish output, candidate, and corpus bottlenecks with no-crawl controls. |
| 2026-10-12 | Phase 11A organizer diagnostics / this checkpoint | Legal 20-document/40-chunk output improved FINAL from `0.0426` to `0.0470`; increasing source candidates from M8 to M16 reached `0.0450`. Both output and candidate truncation matter, with the larger observed gain from output depth. |
| 2026-10-12 | Phase 11B new-corpus migration / this checkpoint | Migrated validation to the 1,194,469-row Vietnamese/Chinese/English official corpus, reconciled retained local candidates and provenance, and produced a clean fixed-ranking document curve at K=10/20/50 with chunks fixed at 20. K=100 was unsupported because minimum ranked depth was 54. Organizer results remain pending. |
| 2026-10-13 | Phase 12 PubMed acquisition / this checkpoint | Accounted for all 240,000 official PubMed PMIDs, recovered 197 book records, confirmed 104 unavailable records, and produced 239,896 valid documents plus 744,861 source-grounded chunks. A missing benchmark telemetry dependency stopped before model load; fail-fast dependency/CUDA/model-contract preflight now protects the resumable continuation. |
| 2026-10-13 | Phase 12 full PubMed release / this checkpoint | Completed full PubMed embedding, retrieval, exact-key reranking, merge, provenance reconciliation, attribution, and final scientific validation. The immutable Docs@20/Chunks@40 ZIP passed every release gate and reached `READY_FOR_LEADERBOARD`; its organizer result remains pending. |
| 2026-10-13 | Phase 12 organizer result and Phase 13 diagnosis / this checkpoint | Full PubMed fell from Phase 11C FINAL `0.0492` to `0.0451`; all metrics declined while PubMed displaced exactly 5,913 document and 11,592 chunk slots. New data/inference paused. Phase 13 prioritizes frozen-score document aggregation/fusion replay plus a stratified 60-query human review before any new crawl or leaderboard package. |
| 2026-10-13 | Phase 13A frozen-score replay / this checkpoint | Exact Phase 11C/12 controls and seven CPU-only aggregation/fusion variants completed with zero new inference and zero provenance mismatch. Aggregation and source fusion materially changed PubMed competition, but no relevance winner is scientifically supportable without the prepared blinded 60-query human review; run state is `READY_FOR_HUMAN_REVIEW`. |
| 2026-10-14 | Phase 13A organizer result / this checkpoint | The controlled Top-3-mean document package with frozen Phase 11C chunks reached FINAL `0.0472`: better than Phase 12 (`0.0451`) but below Phase 11C (`0.0492`). Open-ended ranking tweaks stop; the project moves to a measured architecture/evaluation foundation. |
| 2026-10-14 | System upgrade Milestone 1 / this checkpoint | Audited the real 1.194M official corpus and 275,586-document local union, measured data-quality and retrieval funnels, added a provenance-validated offline evaluator, and produced a 21-pair source-title silver regression set. The set is explicitly insufficient for architecture selection because only four Vietnamese pairs and no VI→EN held-out positives exist; Milestone 2 is not authorized yet. |
### 2026-10-14 — Milestone 1.5 registry and candidate ledger

- Built a canonical CPU-only registry for 275,877 NEW-official local documents; this corrected the M1 union by adding 291 valid frozen-pilot documents that participated in retrieval but were absent from the broad/PubMed inventory.
- Built a 332,407-row document candidate ledger covering all 1,200 queries and the complete frozen Phase 12 candidate universe.
- Expanded benchmark discovery found no defensible non-title biomedical or VI→EN positive pairs; architecture selection remains evaluation-gated.

### 2026-10-14 — Milestone 2A PubMed lexical candidate arm

- Built a 239,896-document SQLite FTS5 PubMed index without crawling, embedding, or reranker inference.
- A bounded dense-seed Vietnamese-to-English bridge produced lexical top-100 for 1,200/1,200 queries: 97,811 lexical-only and 22,189 dense/lexical-overlap query-document pairs.
- The two-chunk-per-document evidence pool contains 435,558 pairs; 239,954 exact scores are reusable and 195,604 remain for a gated M2B experiment.
- Novel candidates remain relevance-UNKNOWN. M2B is authorized only as one controlled reranking experiment after drift review and exact Phase 12 isolation replay.

### 2026-10-14 — Milestone 2B stopped at query-bridge gate

- The mandatory Vietnamese-to-English trace review found repeated topic drift caused by rare terms inherited from imperfect dense seeds.
- M2B stopped before Phase 12 replay, resource preflight, GPU loading, or any of the 195,604 missing exact scores.
- No candidate ZIP or leaderboard submission was produced; lexical candidate relevance remains UNKNOWN.

### 2026-10-14 — M2A query bridge repair remains GPU-blocked

- Replaced uncontrolled dense-seed terms with provenance-bearing local bilingual anchors and exact Dense fallback.
- The 60-query diagnostic found traceable anchors for only 15 queries; the independent half had 3/30 and 45/60 total queries required fallback.
- The repaired bridge is mechanically topic-safe but too sparse for a useful M2B hybrid arm. No GPU inference or leaderboard artifact was produced.

### 2026-10-14 — Direct VI→EN translation feasibility stopped at RAM gate

- Pinned `Helsinki-NLP/opus-mt-vi-en` to official revision `c8d2853e77f5fae31124d993e0b35176b1c8914e` (Apache-2.0); no cached local copy existed.
- Disk passed, but free physical RAM fell to 772.2 MiB versus a measured, evidence-based minimum of 1.25 GiB (1.5 GiB recommended), so the model was neither downloaded nor loaded.
- The 60-query frozen diagnostic remains untouched; translation accuracy, PubMed Top-5 behavior, and the 1,200-query rollout decision remain UNKNOWN. No TF-IDF patch or new retrieval phase started.

### 2026-10-14 — Bounded Colab reranker portability harness

- Exported a deterministic 200-pair proof-of-concept package from already cached Phase 12 scores, covering EN/VI/ZH, available passage-length strata, and query-local close scores.
- Added a pinned FP16 batch-2 Colab runner and a strict local dry-run verifier that cannot overwrite production scores.
- Local structure, CRC, hashes, and tests pass; A100 numerical equivalence and performance remain `AWAITING_COLAB_RESULT` until the manual Colab run is returned.

### 2026-10-14 — Phase 14 extraction/chunking and embedding PoC

- Completed the bounded Medlatec acquisition boundary as 2,996 usable documents and 58,839 unique chunks; extraction and source-span reconstruction gates passed.
- Exact-duplicate analysis identified 32,762 chunk slots (55.68% of all chunks) as site boilerplate or related-article navigation risk; no dataset rows were altered.
- Exported a deterministic 200-row, batch-context-preserving BGE-M3 reference package. Full embedding remains stopped at `AWAITING_COLAB_EMBEDDING_EQUIVALENCE`; reranker A100 score import remains prohibited after its failed equivalence test.
