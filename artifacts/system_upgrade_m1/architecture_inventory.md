# Kiểm kê kiến trúc hiện tại

## Sự thật triển khai

- Official: **1,194,469** URL/ID, không có text trong manifest.
- Local usable union: **275.586 documents / 1.268.343 chunks**; gồm broad-existing 35.690 và PubMed 239.896, không trùng ID.
- Không có một unified index. Focused dùng dense+BM25 theo source; broad-extra dùng global BM25; PubMed dùng dense BGE-M3 document-level.
- Phase 12 rerank cache có 603.587 key trong experiment DB; frozen FULL universe có 695.539 exact query/chunk scores sau reuse pilot.
- Đầu ra luôn chỉ 24.000 doc slots và 48.000 chunk slots (1.200 × Docs20/Chunks40).

## Luồng thật

`official URL -> acquire -> extract/clean -> chunk -> arm-specific index -> candidate -> exact-key reranker -> global max-chunk document ranking -> independent docs/chunks packaging`

Các count ở mỗi mũi tên không đồng nghĩa. `combined_documents.parquet` Phase 12 (111.812 rows) là provenance closure của candidate/ranking, không phải toàn bộ 239.896 PubMed document index.

## Thành phần đã có / còn thiếu

| Thành phần | Trạng thái |
|---|---|
| Resumable acquisition/extraction/chunking | Có, theo experiment/source |
| BGE-M3 dense | Có cho focused chunks và toàn bộ PubMed documents |
| BM25 | Có theo focused source và broad-extra |
| Unified disk-backed multilingual document index | **Chưa có** |
| Exact-key reranker cache | Có |
| Provenance/official/package gates | Có |
| Offline relevance evaluator | Có từ M1, nhưng silver benchmark bị giới hạn |
| Trustworthy VI→EN held-out relevance labels | **Chưa có** |
