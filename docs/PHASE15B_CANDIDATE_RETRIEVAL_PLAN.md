# Phase 15B — Vietnamese unified document BM25 pilot

## Quyết định duy nhất

Thử một **chỉ mục BM25 cấp tài liệu, thống nhất cho toàn bộ nội dung tiếng Việt
đã có tại chỗ**, rồi chạy pilot trên 120 query được chọn xác định. Truy vấn dùng
nguyên văn tiếng Việt; không dịch, không pseudo-relevance expansion và không
dùng PubMed. Đây là phép thử candidate discovery, không phải biến thể ranking
hay submission leaderboard.

## Cơ sở lựa chọn

1. Phase 11C đã có organizer evidence dương khi arm BM25 global trên
   BROAD_EXTRA được cộng vào control: FINAL `0.0480 -> 0.0492`.
2. Phase 14 bổ sung nội dung Medlatec tiếng Việt và tăng tiếp lên `0.0533`.
3. PubMed pseudo-relevance bridge đã lệch chủ đề có hệ thống; bridge sửa chỉ có
   anchor ở `3/30` query độc lập; direct VI→EN translation chưa vượt resource
   gate. Phase 15B không lặp lại các hướng này.
4. Phase 14 vẫn tìm candidate qua các arm/source phân mảnh. BM25 document-level
   toàn cục cho phép tài liệu Việt cạnh tranh không quota nguồn và không cần
   embedding/reranker mới ở pilot.

## Input đóng băng

- NEW-official IDs, canonical registry, Phase 11C/14 documents và chunks.
- Phase 14 candidate ledger, scores và submission chỉ đọc.
- Chỉ tài liệu `language_signal=vi`, content/provenance hợp lệ tham gia arm mới.
- PubMed/EN, ZH, dịch máy và query expansion bị loại khỏi pilot.

## Pilot 120 query

1. Tạo manifest exact-ID cho toàn bộ tài liệu Việt local; deduplicate `doc_id`,
   chỉ báo content-hash duplicates và không sửa dữ liệu.
2. Xây SQLite FTS5/BM25 disk-backed từ `title` và `normalized_text` tách field,
   streaming và resumable; không materialize toàn corpus trong RAM.
3. Chọn 120/1.200 query xác định theo hash, phân tầng theo độ dài/topic proxy;
   truy vấn nguyên văn và lấy Top-100 document toàn cục.
4. So sánh read-only với Phase 14 ledger: overlap, lexical-only documents,
   source diversity, coverage và duplicate/boilerplate exposure.
5. Chọn tối đa hai chunk nguyên bản/document bằng lexical match chỉ để tính
   exact missing reranker keys. **Không chạy reranker trong pilot.**
6. Silver source-title benchmark chỉ là smoke test có leakage; không gộp với
   competition queries và không gọi là official recall.

## Instrumentation bắt buộc trước worker

Worker phải dùng `src/runtime/progress_contract.py` và ghi
`overall_percent`, `stage_percent`, `done`, `total`, `current_operation`, elapsed,
ETA (hoặc `INDETERMINATE`), heartbeat và worker state. Các stage:
`MANIFEST`, `INDEX_BUILD`, `PILOT_RETRIEVAL`, `AUDIT`, `DECISION`, `PACKAGING`.
Trong pilot, PACKAGING là `NOT_APPLICABLE`; overall không được đạt 100% trước
terminal audit. `tools/watch_run_progress.ps1` hiển thị hai thanh tiến độ.

## Chi phí dự kiến

- Khoảng 38–40 nghìn tài liệu Việt; con số chạy phải lấy từ manifest thực tế.
- CPU index build: 5–15 phút; retrieval 120 query: dưới 2 phút sau build.
- Peak incremental RAM mục tiêu: `<=800 MiB`.
- Disk dự kiến: `0.2–1.0 GiB`; hard stop nếu vượt `1 GiB` hoặc disk floor 20 GiB.
- GPU, embeddings, reranker inference, crawling: `0`.

## Điều kiện dừng

Dừng trước scoring nếu frozen Phase 14 thay đổi; provenance/official-ID mismatch
khác 0; index manifest không đóng đúng IDs/content hashes; kết quả không
deterministic; RAM/disk vượt budget; lỗi tokenizer/query parsing có hệ thống;
hoặc exact missing reranker work vượt budget phải khai báo trước inference.

## Tiêu chí đánh giá

- 120/120 query hoàn chỉnh, deterministic và provenance pass.
- Báo coverage, overlap@100, lexical-only count, source diversity,
  duplicate/boilerplate rate, latency và exact missing-score count.
- Silver Recall@20/100, MRR phải gắn nhãn `TITLE-LEAKAGE REGRESSION`; không chọn
  winner và không suy ra F2.
- Kết quả cấu trúc tốt chỉ cho phép đề xuất một reranking budget nhỏ sau phê
  duyệt; không cho phép tạo ZIP hay tuyên bố relevance improvement.

Trạng thái: **PLAN_ONLY — NOT_LAUNCHED**.
