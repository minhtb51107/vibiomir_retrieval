# Phase 13 prioritized plan

## Mốc triển khai kế tiếp

**Phase 13A: xây dựng replay fusion có kiểm soát trên score hiện có, trước khi chạy model mới.** Giữ nguyên candidate và reranker score; thay cách chuyển chunk-score thành document competition bằng quota/cap được khai báo, document-level aggregation ổn định, và consistency giữa docs/chunks. Đánh giá bằng 60 nhãn người thật, tỷ lệ giữ lại control, source displacement, và replay determinism. Không dùng leaderboard để debug.

## A. Candidate recall: hybrid có kiểm soát

1. Tạo union lexical BM25 + multilingual dense ở cấp **document**, rồi giới hạn số chunk đại diện/tài liệu trước rerank. Failure addressed: 349k PubMed chunks làm candidate volume mất cân bằng. Input: embedding/BM25/index hiện có. Cost: CPU replay thấp; model inference bằng 0 cho phase đầu. Diagnostic: candidate diversity, số tài liệu/query, overlap dense–lexical. Acceptance: không nguồn nào có lợi chỉ vì nhiều chunk; control replay giữ nguyên khi tắt arm. Fallback: cap theo document với top-N chunk bằng dense/lexical score.
2. Thêm medical synonym expansion dạng từ điển cục bộ, versioned (VI–EN thuật ngữ/viết tắt), không dùng API đóng. Failure addressed: lexical cross-language gap. Cost: thấp cho query expansion/BM25; dense không đổi. Diagnostic: candidate changes trên review set. Acceptance: tăng relevant/partial candidates theo human labels mà không tăng irrelevant displacement. Fallback: chỉ dùng synonym cho lexical arm, không sửa query gốc.

## B. Ranking và source fusion

1. So sánh offline các aggregation đã khai báo: max chunk hiện tại, top-k mean, và document score + best evidence score. Failure addressed: extreme-value bias của max trên nhiều chunk. Input: 603.587 score hiện có. Cost: phút CPU. Diagnostic: displacement, margin, review precision@20. Acceptance: cải thiện human-judged precision và giữ/khôi phục phần lớn Phase 11C relevant results. Fallback: gated additive PubMed arm chỉ thay thế khi margin vượt ngưỡng đo từ review set.
2. Không dùng source boost tùy tiện. Nếu cần calibration, fit/choose trên review labels và báo riêng theo language/source, với holdout. Acceptance: calibration ổn định trên bootstrap/holdout; fallback: source-neutral document aggregation.

## C. Chunk selection và consistency

1. Chọn docs trước, sau đó chọn evidence chunks thuộc docs đã chọn; cho phép một quota nhỏ chunk-only nếu contract cần. Failure addressed: nhiều chunk owner không nằm trong top20 docs và output không nhất quán. Cost: CPU only. Diagnostic: owner-consistency, unique docs represented, redundancy. Acceptance: 100% ownership/provenance, giảm trùng nội dung, không giảm human evidence relevance. Fallback: tối đa 2 chunk/tài liệu rồi fill theo score.
2. Deduplicate near-identical chunks trong cùng article trước competition. Input: text/hash hiện có. Acceptance: không mất exact provenance; fallback: exact-normalized duplicates only.

## D. Mở rộng nội dung thật

1. Ưu tiên domain Việt Nam có official gap lớn trong `coverage_funnel.json`; crawl lịch sự theo robots/rate limit, resumable và disk gate 20 GiB. Failure addressed: local VI coverage remains a small fraction of 714.469 official IDs. Cost: network + extraction; embedding chỉ sau khi retrieval design passes. Diagnostic: success/extraction rate and lexical candidate yield on review queries. Acceptance: source quality gate plus measurable new candidate coverage. Fallback: smaller per-domain exploration allocation.
2. Chinese remains a later multilingual track; a negative Vietnamese-query lexical result must not be interpreted as no Chinese value.

## Stage gates

1. Annotate 60-query review set with two passes or adjudication.
2. Implement CPU-only document aggregation/fusion replay using frozen scores.
3. Accept only variants that improve human-judged precision/coverage and pass frozen control/provenance/determinism.
4. Only then authorize missing-pair inference or new acquisition.
5. Use at most one leaderboard submission for the selected Phase 13 architecture; keep remaining quota reserved.
