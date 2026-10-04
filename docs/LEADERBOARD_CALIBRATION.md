# Phase 10A — Public Leaderboard Calibration Submissions

## Scope

Phase 10A creates nine controlled submission variants from the already
persisted Phase 4, 6, and 7 outputs. It performs no crawling, network access,
embedding, reranking, retraining, URL selection, or local relevance scoring.
Each experiment changes one retrieval-output choice relative to the Phase 9 A
baseline so that future organizer scores can provide external evidence.

The Phase 9 A organizer result supplied by the project owner is:

| Metric | Value |
|---|---:|
| `FINAL_SCORE` | 0.0001 |
| `DOCS_F2MACRO` | 0.0001 |
| `CHUNKS_F2MACRO` | 0.0001 |
| `DOCS_PRECISION` | 0.0006 |
| `DOCS_RECALL` | 0.0001 |
| `CHUNKS_PRECISION` | 0.0008 |
| `CHUNKS_RECALL` | 0.0001 |

These values are external leaderboard observations, not a locally reproduced
evaluation. No result is yet available for E1–E9, and no variant is claimed to
be better.

## Experiment design

The reproduced baseline is byte-identical to the Phase 9 A JSON
(`7672ef2be6f975b113a48c4c15b3db2da4d7c83ac04232bbd0e23726aa716acd`).
The configuration is `configs/phase10a_calibration.yaml`.

| ID | Single change from Phase 9 A | Docs/query min–median–max | Chunks/query min–median–max |
|---|---|---:|---:|
| E1 | reranked document depth cap 10 → 20 | 4–20–20 | 48–50–50 |
| E2 | reranked document depth cap 10 → 50 | 4–39–50 | 48–50–50 |
| E3 | all unique documents available in reranked depth-100 pool | 4–39–75 | 48–50–50 |
| E4 | baseline chunk depth 50 → 20 | 1–10–10 | 20–20–20 |
| E5 | reranked chunk depth cap 50 → 100 | 1–10–10 | 98–100–100 |
| E6 | source-verbatim windows targeting 768 BGE-M3 tokens | 1–10–10 | 43–49–50 |
| E7 | source-verbatim windows targeting 1,024 BGE-M3 tokens | 1–10–10 | 42–48.5–50 |
| E8 | Phase 6 hybrid/RRF document and chunk ordering | 1–10–10 | 48–50–50 |
| E9 | Phase 6 sparse/BM25 document and chunk ordering | 3–10–10 | 50–50–50 |

Depths are caps, not fabricated quotas. Queries with fewer locally available
candidates remain shorter.

## Source-verbatim expansion

E6 and E7 widen each baseline canonical chunk using its recorded offsets in
the processed document. The actual BGE-M3 tokenizer is loaded from the local
cache with `local_files_only=true`. The algorithm tokenizes each source
document once per experiment, centers a bounded window on the original chunk,
clamps to document limits, and emits exactly
`normalized_text[expanded_start:expanded_end]`. It neither rewrites nor
synthesizes text.

| Evidence | E6 (768) | E7 (1,024) |
|---|---:|---:|
| Emitted chunks after duplicate-window collapse | 58,462 | 57,867 |
| Expanded beyond original | 59,597 | 59,597 |
| Whole-document windows before collapse | 12,254 | 14,117 |
| Duplicate expanded windows collapsed | 1,435 | 2,030 |
| Median emitted tokens | 769 | 1,025 |
| Maximum emitted tokens | 819 | 1,066 |
| Verbatim failures | 0 | 0 |
| Original-containment failures | 0 | 0 |
| Expanded source spans independently validated | 58,162 | 57,567 |

The small amount above the nominal token target comes from bounded outward
snapping to avoid cutting words. Every emitted string remains one contiguous
substring of its source document.

## Validation and determinism

Every JSON and ZIP passed the strict organizer validator:

- exactly 1,200 unique official query IDs in original order;
- organizer field names and types only;
- zero invalid local document IDs;
- zero duplicate `relevant_docs` within a query;
- zero duplicate emitted chunk objects;
- zero chunk provenance mismatches;
- exactly one root-level JSON member in each ZIP;
- ZIP member hash equal to its generated JSON hash.

Canonical variants require exact `(doc_id, chunk_text)` membership in the
validated Phase 4 chunks. E6/E7 additionally accept only non-empty text proven
to be a contiguous substring of the corresponding processed source document.

A complete second generation used fresh source loads and fresh local tokenizer
instances. JSON and ZIP files were byte-identical for all nine experiments.

## Upload files

| ID | ZIP | SHA-256 |
|---|---|---|
| E1 | `submissions/phase10a_E1_docs20.zip` | `c9216f1bb40b09cd1dfb98c6b0bc67c16de014aa35a79633d99dc88921173318` |
| E2 | `submissions/phase10a_E2_docs50.zip` | `7615681a43ac0226de60e421867132d03cf209142157c88fb097df1a01ad5a3e` |
| E3 | `submissions/phase10a_E3_docs_max.zip` | `591412e484c98e21047c5f7e4b0285245bbdc90be32038443f957f6d6292537c` |
| E4 | `submissions/phase10a_E4_chunks20.zip` | `37b19da04a8a0cb4ddd6e90e6ba49a5039eb3893347e5b496ca201d271085142` |
| E5 | `submissions/phase10a_E5_chunks_more.zip` | `9e17c5be726d8fa32091e9305716f09fc06b7e5a9d8b96c408a888fbaa04dcad` |
| E6 | `submissions/phase10a_E6_chunk_expand_768.zip` | `49e9b029e00fe47c7364f0d782f454c18c9b7157fa3fc619bf60e77558ebb500` |
| E7 | `submissions/phase10a_E7_chunk_expand_1024.zip` | `61fd1c0489a29540b136fae16e4349be49a30e13c6e9229c7cb767aa16c03291` |
| E8 | `submissions/phase10a_E8_hybrid.zip` | `4453d71fdc980d90b361c3bfcf2589d4571afcc71b262886944f1a7a7ce9c372` |
| E9 | `submissions/phase10a_E9_sparse.zip` | `e235dde15062f4b7ee9668b56f60ac95fcab94d10f8434c5b63bca848bf116ef` |

Large JSON/ZIP outputs and `SHA256SUMS_phase10a.txt` remain under the ignored
`submissions/` directory. The committed compact manifest at
`artifacts/phase10a_calibration/manifest.json` records JSON and ZIP hashes,
source hashes, counts, hypotheses, validation results, and determinism proof.

## Reproduction

From the repository root, with the existing environment and local model cache:

```powershell
.venv\Scripts\python.exe scripts\generate_calibration_submissions.py `
  --config configs\phase10a_calibration.yaml `
  --phase9-manifest artifacts\phase9_submission\submission_manifest.json `
  --verify-determinism
```

The command is offline by construction: Hugging Face and Transformers offline
environment variables are set before tokenizer loading, and the tokenizer uses
`local_files_only=true`.

## Leaderboard recording policy

Submit variants manually and record the organizer submission ID, phase,
timestamp, ZIP SHA-256, and every returned metric. Interpret each comparison
only against its declared one-change hypothesis. Do not infer general relevance
quality from this pilot corpus, and do not call internal overlap or provenance
checks competition evaluation.
