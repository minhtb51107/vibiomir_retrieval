# Phase 3 Cleaning, Extraction, and Chunking

## Scope and reproducibility

Phase 3 converts fetched response bytes into traceable clean documents and
deterministic chunks. It does not create embeddings, retrieval indexes,
rerankers, evaluations, or submissions. The pilot reused the balanced Phase
2C manifest and remained bounded to 1,225 metadata rows. Of these, 1,148 had
successful response bodies and 1,097 produced usable extracted documents.

The committed machine-readable results are in
`artifacts/phase3_pilot/summary.json`. Crawl bodies, the archive index, cleaned
Parquet, and chunk Parquet files remain under ignored `data/` directories.

Run the bounded processing pipeline with:

```powershell
.venv\Scripts\python.exe scripts\process_corpus.py `
  --crawl-db data\crawled\phase3_pilot.sqlite `
  --archive-dir data\crawled\body_archives\phase3_pilot `
  --documents-out data\processed\phase3_pilot_documents.parquet `
  --chunks-dir data\chunks\phase3_pilot `
  --summary-out artifacts\phase3_pilot\summary.json
```

`--limit` can constrain local development further. The core extraction API
also accepts fixture bytes directly, so tests do not need crawl state or live
network access.

## Body archive and reader

`src/storage/body_archive.py` implements append-only gzip shards plus a small
SQLite index. Each response is an independent gzip member, allowing
`get_body(doc_id)` to seek directly to the indexed byte range rather than
inflating an entire shard. The default shard target is 5,000 documents.

The index maps canonical `doc_id` to:

- shard name, member offset, and compressed length;
- original byte length and SHA-256 checksum;
- original/final URL and fetch timestamp;
- content type, client encoding, and declared HTTP encoding.

Writes fsync the compressed member before committing its index row. A crash
may therefore leave only an unindexed tail, which startup truncates. If a
crash happens after atomic shard publication but before its index-name update,
startup reconciles the published `.gz` file. Duplicate IDs with the same
checksum are idempotent; a different checksum raises an error. Reads verify
length and checksum by default, and `verify_integrity()` streams through all
members without loading the full archive into memory.

The pilot stored 113,278,186 original bytes as 26,312,942 compressed bytes,
a measured ratio of 0.2323. All 1,148 bodies passed checksum verification. The
crawl SQLite database contains zero non-null raw-body BLOBs.

## Decoding strategy

The decoder considers the HTTP charset first, then BOM and HTML meta charset,
then `charset-normalizer`, with strict UTF-8 and finally replacement decoding
as safe fallbacks. An invalid or incompatible declaration does not prevent a
later source of evidence from succeeding. Encodings are resolved through
Python's codec registry rather than assigned by domain. This handles measured
GB2312 pages without assuming every Chinese page uses GB2312.

Decoded Unicode is normalized to NFC. The selected encoding, replacement
character count, and a severe-decoding flag are recorded. A response whose
replacement ratio exceeds the configured threshold is not silently accepted
as successful extraction. Original bytes remain in the archive unchanged.

Measured pilot encoding selections were 991 UTF-8, 95 GB2312, 12 GB18030,
and 50 ASCII. The 50 ASCII responses were the known tiny JavaScript shells.
No replacement characters were introduced across the pilot.

## Hybrid extraction pipeline

The generic cleaner removes scripts, styles, templates, navigation, footers,
asides, forms, hidden elements, and confidently named ad/banner regions. It
retains titles, headings, paragraphs, lists, block quotes, and time elements.
Exact repeated blocks within a page are collapsed, but different `doc_id`
values are never merged.

Two candidates are built for each HTML response:

1. A semantic candidate from `article`, `main`, an explicit main role, or a
   content-like container.
2. A density candidate scored using visible text, paragraphs, and link-text
   penalty.

The semantic candidate wins only when it has at least 500 characters and at
least 55% of the density candidate's text. Otherwise the density candidate is
used. These thresholds live in `configs/extraction.yaml`.

Q&A evidence comes from repeated question/answer-like containers or JSON-LD
types. When present, question and answer blocks become separate structured
sections instead of one flattened string. The pilot preserved structured Q&A
sections in 375 successful documents.

## Status taxonomy

- `SUCCESS`: usable normalized text was extracted.
- `EMPTY_CONTENT`: decoding succeeded but meaningful text was absent.
- `JS_SHELL`: tiny visible content plus script evidence, without browser
  rendering.
- `EXTRACTION_FAILED`: missing body, severe decoding damage, checksum/read
  failure, or another extraction error.
- `ACCESS_RESTRICTED`: Phase 2 recorded a permanent access restriction.
- `ROBOTS_BLOCKED`: Phase 2 robots policy denied the request.

The pipeline does not fabricate content for any non-success status. Browser
rendering remains deferred.

## Clean document schema

Each document records `doc_id`, original/final URL, title, lightweight
language signal, selected encoding, canonical NFC `normalized_text`, structured
sections, extraction method/status, byte and character counts, paragraph and
candidate metrics, replacement/boilerplate signals, raw SHA-256, archive
location/checksum, and optional error details. Section records include type,
text, canonical offsets, and heading path.

Language values are lightweight measured signals (`zh-Han-script`,
`vi-signal`, `latin-undetermined`, or `unknown`), not a claim of full language
identification.

## Chunking and provenance

Chunking is deterministic, heading-aware, paragraph-aware, and Q&A-boundary
aware. It operates only on successful documents. Full chunk windows prefer a
paragraph boundary when doing so still makes forward progress; long paragraphs
may be split on token boundaries. Question and answer sections are always
separate logical ranges.

Offsets are Python Unicode character offsets into the document's canonical NFC
`normalized_text`. For every chunk:

```text
raw_text == document.normalized_text[start_offset:end_offset]
```

`normalized_text` is the NFC form of that exact slice and is never generated
or rewritten. Chunk IDs hash the document ID, configuration name, chunk index,
offsets, and exact text. Each row also records token count, section type,
heading path, source URL, and extraction method.

The tokenizer is pluggable. Phase 3 uses a deterministic offline Unicode
tokenizer that treats Han characters, words, and punctuation as tokens. This
avoids a model download in tests. `BAAI/bge-m3` is recorded as the future
production tokenizer; Phase 4 must remeasure lengths with that tokenizer before
index construction.

## Pilot results

The 1,225-row pilot included Chinese Q&A and article sources, Vietnamese news
and health sources, HTTP redirects, unusual suffixes, robots-blocked rows,
access-restricted rows, and known JS shells. Acquisition was bounded and its
second run skipped 1,225/1,225 rows with zero network requests.

Extraction results:

| Metric | Measured result |
|---|---:|
| Archived successful responses | 1,148 |
| Extracted `SUCCESS` | 1,097 (95.56% of archived bodies) |
| Empty content | 1 |
| JS shells | 50 |
| Extraction failures without bodies | 2 |
| Robots blocked | 50 |
| Access restricted | 25 |
| Semantic candidate selected | 500 |
| Density candidate selected | 597 |
| Structured Q&A documents | 375 |
| Median paragraphs | 9 |
| Mean / median clean-to-decoded-character ratio | 0.0367 / 0.0164 |

Measured language signals among successful documents were 854 Han-script,
240 Vietnamese-signal, and 3 unknown. These are content-derived signals, not
domain inference.

## Chunk experiment

All counts use the offline proxy tokenizer and all 1,097 successful documents.
`Very short` means fewer than 64 proxy tokens.

| Configuration | Chunks | Chunks/doc | Mean / median tokens | Very short | Over target | Q&A crossings |
|---|---:|---:|---:|---:|---:|---:|
| 256 / 32 | 8,780 | 8.00 | 178.56 / 221 | 18.29% | 0 | 0 |
| 512 / 64 | 5,668 | 5.17 | 272.73 / 296 | 26.96% | 0 | 0 |
| 768 / 96 | 4,660 | 4.25 | 327.61 / 215 | 32.79% | 0 | 0 |

The recommended Phase 4 starting point is **512 target / 64 overlap**. It cuts
the chunk count by 35.4% relative to 256/32 while retaining substantially finer
units than 768/96. Its higher short-chunk rate reflects deliberately preserved
small structural sections, especially Q&A content, rather than overflow or
boundary failure. This is a starting configuration, not a retrieval-quality
winner; Phase 4 evaluation and the real BGE-M3 tokenizer must validate it.

## Known limitations and deferred work

- Generic extraction can still retain boilerplate or omit content on unfamiliar
  templates; no large set of domain selectors was introduced.
- JavaScript shells remain explicit and unrendered.
- Lightweight language and token signals are diagnostic approximations.
- Full-corpus acquisition is not required for this pipeline and was not run.
- Embeddings, dense/sparse indexes, retrieval, reranking, evaluation, and
  submission generation are deferred to later phases.
