# Target architecture — thiết kế sau Milestone 1

1. **Registry chất lượng/corpus:** một bảng doc-level authoritative ghi official membership, acquisition, extraction quality, language, hashes và provenance.
2. **Disk-backed document retrieval:** BM25 sharded + BGE-M3 dense/memmap/ANN có version manifest; không materialize corpus text vào RAM.
3. **Candidate union theo canonical doc ID:** dense, sparse và terminology expansion tạo ranks riêng; deduplicate trước khi lấy bounded representative chunks.
4. **Source-neutral fusion:** RRF/rank features ở doc level; source/arm là feature/audit, không phải magic boost. Candidate volume không được tự tạo lợi thế.
5. **Candidate-only rerank:** exact `(query_id, chunk_id)` cache; giới hạn chunks/document trước inference.
6. **Hai output độc lập hợp schema:** Docs và chunks có ranking riêng, bounded evidence/document, exact source text và provenance closure.
7. **Evaluation-first:** silver regression + human pooled judgments + frozen controls trước model/crawl/leaderboard.

Reuse bắt buộc: Phase 11C control, PubMed documents/embeddings, exact reranker scores, canonical provenance.
