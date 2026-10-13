# Chẩn đoán bottleneck retrieval

## Đã xác nhận

1. **Coverage local còn thiếu 918.883 official documents.** English PubMed gần đủ; VI/ZH thiếu nhiều.
2. **Candidate-volume không cân bằng:** PubMed đưa 349.187 query–chunk pairs, mỗi focused source khoảng 19.200. Phase 12 giảm FINAL 0.0492 → 0.0451.
3. **Document rank dùng max chunk:** nguồn/document có nhiều candidate chunks có nhiều cơ hội tạo extreme score. Đây là volume sensitivity, chưa phải bằng chứng calibration sai.
4. PubMed chiếm 5.913 doc slots và 11.592 chunk slots; mọi leaderboard metric giảm. Displacement có hại tổng thể, nhưng relevance từng item vẫn UNKNOWN.
5. Docs và chunks được chọn độc lập hợp lệ theo schema; ownership consistency không đồng nghĩa relevance.

## Chưa thể xác nhận

- Bao nhiêu gold documents vắng khỏi local corpus: **UNKNOWN**.
- Bao nhiêu gold có local nhưng candidate retrieval bỏ sót: **UNKNOWN**.
- Official Recall@20/100 offline: **UNKNOWN** vì không có hidden gold.
- Dense VI→EN hay reranker VI→EN là lỗi chính: **UNKNOWN**; cần held-out bilingual judgments.
