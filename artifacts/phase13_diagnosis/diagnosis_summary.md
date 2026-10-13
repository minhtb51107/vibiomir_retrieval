# Phase 12 failure diagnosis

## Kết luận ngắn gọn

**Nút thắt mạnh nhất quan sát được là cạnh tranh/xếp hạng ứng viên (candidate competition and source fusion), không phải thiếu dữ liệu đơn thuần.** Phase 12 thêm 239.896 tài liệu PubMed nhưng FINAL giảm từ `0.0492` xuống `0.0451`. PubMed thay thế 5,913 vị trí tài liệu và 11,592 vị trí đoạn văn; thay đổi xảy ra ở 932/1.200 truy vấn tài liệu và 1,039/1.200 truy vấn đoạn văn.

“Candidate competition” nghĩa là nhiều kết quả cùng tranh một số vị trí đầu ra hữu hạn. PubMed đưa 349,187 cặp query–chunk vào reranker, trong khi mỗi nguồn tập trung chỉ có 19.200. Hệ thống xếp toàn bộ chunk theo một điểm chung rồi lấy lần xuất hiện đầu tiên của mỗi tài liệu. Vì vậy nguồn có rất nhiều chunk có nhiều cơ hội tạo một điểm cực đại và đẩy tài liệu Việt Nam đã được kiểm chứng ra ngoài.

## Bằng chứng về điểm số

- PubMed median reranker score: `-5.0977`; p95: `-0.3230`.
- BROAD_EXTRA median: `-6.8945`; p95: `-1.5020`.
- Đây **không** chứng minh thang điểm bị hỏng. Điểm PubMed không cao hơn mọi nguồn mạnh; vấn đề quan sát được là khối lượng ứng viên lớn cộng với phép lấy max-chunk ở cấp tài liệu.
- Kết quả leaderboard cho biết các thay thế này có hại về tổng thể, nhưng không cho biết từng PubMed result nào sai. File `human_review.csv` được tạo để con người gắn nhãn mà không giả mạo gold của organizer.

## Phạm vi bao phủ

Corpus chính thức có 1,194,469 ID. Local union hiện có 275,586 tài liệu dùng được: 35.690 broad-existing và 239.896 PubMed. English PubMed gần đầy đủ (thiếu 104 record không khả dụng), nhưng độ phủ Việt Nam/Trung Quốc vẫn rất nhỏ và 5.702 tài liệu local chưa có nhãn ngôn ngữ chắc chắn.

## Bất định lớn nhất

Không có hidden gold theo truy vấn. Ta chưa biết thất bại chủ yếu do (1) dense VI→EN lấy sai bài, (2) reranker VI→EN đánh giá sai, hay (3) cơ chế max-chunk/fusion ưu tiên PubMed quá mức. Review 60 truy vấn và các replay offline có kiểm soát là mốc tiếp theo để tách ba nguyên nhân này.

## Tệp bằng chứng

- `ranking_displacement.json`: thay thế theo từng query và 60 trường hợp nổi bật.
- `source_score_analysis.json`: phân phối điểm, margin và thành phần theo rank.
- `coverage_funnel.json`: số lượng qua từng tầng corpus → candidate → reranker → output.
- `human_review.csv`: phiếu đánh giá RELEVANT / PARTIALLY_RELEVANT / IRRELEVANT / UNCLEAR.
