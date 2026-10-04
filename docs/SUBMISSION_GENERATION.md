# Phase 9 — Valid Competition Submission Generation

## Scope

Phase 9 converts the already-persisted Phase 7 reranking outputs into strict,
deterministic organizer submissions. It performs no crawling, embedding,
retrieval, GPU reranking, label creation, or local relevance evaluation. The
submission corpus remains the bounded pilot: 5,643 chunks from 1,097 locally
processed documents, not the full 4,394,718-URL corpus.

**QUALITY UNKNOWN UNTIL ORGANIZER SCORE.**

## Organizer contract

The authoritative source is the public organizer competition API for
competition 15, specifically Submission Instructions page 100 and Evaluation
page 108:

`https://leaderboard.aiguru.com.vn/api/competitions/15/`

The required upload is a ZIP containing exactly one root-level UTF-8 `.json`
file. The JSON root is an array of exactly 1,200 query objects in this form:

```json
[
  {
    "id": 1,
    "relevant_docs": [123],
    "relevant_chunks": [
      {"doc_id": 123, "chunk_text": "verbatim source-derived text"}
    ]
  }
]
```

- `id` is the integer ID from `query.parquet`.
- `relevant_docs` is an array of integer IDs from `links_corpus.parquet`.
- `relevant_chunks` contains integer `doc_id` plus string `chunk_text`.
- Optional non-negative integer `chunk_order` is permitted by the organizer;
  Phase 9 omits it, so array position supplies order.
- Empty result arrays are permitted.
- The public instructions do not state a maximum document or chunk count.
- The organizer does not state a duplicate policy. This implementation
  conservatively rejects duplicate document IDs and duplicate
  `(doc_id, chunk_text)` objects within a query.

The compact, machine-readable transcription is
`artifacts/phase9_submission/organizer_schema.json`.

## Inputs and variants

All variants preserve the official query order and read source-derived text
from the unchanged real-tokenizer chunk file. Document and chunk rankings come
from Phase 7 Parquet outputs under the ignored `data/retrieval/` tree.

| Variant | Chunk selection | Document aggregation | Documents/query | Chunks/query |
|---|---|---|---:|---:|
| A `pure_best_chunk` | pure reranker order | best chunk | 1–10 | 48–50 |
| B `pure_top3_mean` | pure reranker order | top-three mean | 1–10 | 48–50 |
| C `max3_best_chunk` | maximum 3 chunks/document | best chunk | 4–10 | 10–50 |
| D `controlled_best_chunk` | exact-text suppression, max 3/document, measured short-heading penalty | best chunk | 4–10 | 10–50 |

The intended depths are 10 documents and 50 chunks. Some Phase 7 query pools
contain fewer distinct documents or fewer candidates after controls, so the
generator never fabricates padding. Formatting-time duplicate-object removal
suppressed 103 objects in each pure variant and 25 in max-three; controlled
already contained none. It does not alter the Phase 7 artifacts.

Variant A is the recommended first upload because it is the Phase 7 default,
uses the direct reranker ordering and best-chunk aggregation, and adds the
fewest uncalibrated selection heuristics. This is a conservative baseline
choice, not a quality claim.

## Provenance and validation

For every emitted chunk, generation checks `chunk_id`, `doc_id`, and exact
`chunk_text` against
`data/chunks/phase4_bge_m3/tokens_512_overlap_64.parquet`. The submission
format omits `chunk_id`, so the standalone validator proves the emitted
`(doc_id, chunk_text)` pair exists in that canonical source. Text is never
summarized, translated, normalized, reconstructed, or rewritten.

The strict validator checks:

- JSON/ZIP parseability and the single-root-file ZIP rule;
- exact 1,200-query coverage, uniqueness, integer IDs, and official order;
- exact top-level fields and organizer field types;
- unique document IDs and unique chunk objects per query;
- local document membership and exact chunk/document provenance;
- optional `chunk_order` type/range when present;
- absence of NaN and infinity anywhere in the payload.

All four JSON files and all four ZIP files passed. Across all variants there
were zero invalid document IDs, zero duplicate emitted document IDs, zero
duplicate emitted chunk objects, and zero chunk-provenance mismatches.

## Determinism and hashes

Serialization uses compact UTF-8 JSON, fixed field/query/result ordering, no
timestamps, and a final LF. ZIPs use a fixed member timestamp, permissions,
compression level, and root filename. Two isolated complete generations
produced byte-identical JSON and ZIP hashes.

| Variant | JSON SHA-256 | Upload ZIP SHA-256 |
|---|---|---|
| A | `7672ef2be6f975b113a48c4c15b3db2da4d7c83ac04232bbd0e23726aa716acd` | `341b968c42969ef3a2f0f7bb4164420a6e75ad55c2f55c5b386d76a340f805a3` |
| B | `886cb901f2a74c6ebc067b14b32c084ae22ccd123d41adccf592b3eda2164d97` | `5cd65f11ee9dc665ad44514e47b68e934c93a419e1d4f4ce38c82292cee650f3` |
| C | `86daccc570f05b55f27710fc2fb50d732cd3bb1f5c4e727f81de2041cd3256b0` | `304f691787a4772f198eb28dcd1fb366874bd2b550ef1fb8fff04e2d77e05bde` |
| D | `9f473cd879cc4049019564d5e07ee4f55b35abc65c60bdc936212bc027c5f0ad` | `4ecfa3c365a8c4a24465917a3f5bc53eac49408ea73e234341ac66b1b47ba5d3` |

The JSON files are 87–107 MB and the ZIPs are 18–24 MB, so all generated
submissions remain local and ignored by Git. Compact hashes, input hashes,
counts, and validation evidence are committed under
`artifacts/phase9_submission/`.

## Reproduction

From the repository root:

```powershell
.venv\Scripts\python.exe scripts\generate_submission.py `
  --config configs\submission.yaml `
  --verify-determinism

$paths = Get-ChildItem submissions -File |
  Where-Object { $_.Extension -in '.json', '.zip' } |
  Sort-Object Name |
  ForEach-Object { $_.FullName }
.venv\Scripts\python.exe scripts\validate_submission.py `
  --config configs\submission.yaml `
  --report artifacts\phase9_submission\validation_summary.json `
  @paths
```

## Manual upload

1. Confirm the intended competition phase on the organizer site.
2. Verify `submissions/SHA256SUMS.txt`; for the first run use
   `submissions/phase9_A_pure_best_chunk.zip`.
3. Inspect the ZIP and confirm it contains exactly
   `phase9_A_pure_best_chunk.json` at archive root and no other member.
4. Upload that ZIP through the competition submission form. Do not upload all
   variants together because each archive must contain exactly one JSON.
5. After organizer processing, record the submission filename, SHA-256,
   competition phase, organizer submission ID/time, document score, chunk
   score, aggregate score, and any rejection/error message before choosing a
   second variant.

No upload is performed automatically by this repository.

## Limitation and next evidence

Every ID is a real corpus ID, but retrieval covers only 1,097 processed pilot
documents. Uncrawled documents are not invented or represented. Consequently,
coverage against organizer ground truth may be low even though the files are
schema-valid. The first organizer score is external calibration evidence; it
must not be predicted from pilot overlap, URL-retention diagnostics, or manual
inspection.
