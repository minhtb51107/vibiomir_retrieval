# Phase 13A — replay aggregation và fusion

## Trạng thái

- Replay Phase 11C: **PASS**, 1200/1.200 query giống hệt.
- Replay Phase 12: **PASS**, 1200/1.200 query giống hệt.
- CPU only; không crawl, embedding, reranker inference hoặc leaderboard submission.

## Kết quả chẩn đoán (không phải relevance)

| Variant | PubMed docs | PubMed chunks | Phase11C docs retained / displaced | Query có PubMed docs | Chunk owner trong Top20 docs | Rank inversions |
|---|---:|---:|---:|---:|---:|---:|
| `max_chunk_global_raw` | 5913 | 12787 | 18087 / 5913 | 932 | 35658 / 48000 (74.29%) | 0 |
| `top2_mean_global_raw` | 4046 | 12787 | 19136 / 4864 | 750 | 35475 / 48000 (73.91%) | 13645 |
| `top3_mean_global_raw` | 3514 | 12787 | 19320 / 4680 | 685 | 34668 / 48000 (72.22%) | 17580 |
| `robust_top3_mean_global_raw` | 3515 | 12787 | 19309 / 4691 | 685 | 34632 / 48000 (72.15%) | 17625 |
| `max_chunk_rrf_by_candidate_arm` | 2404 | 12787 | 11433 / 12567 | 1200 | 22157 / 48000 (46.16%) | 8694 |
| `max_chunk_balanced_premerge_top40_per_arm` | 5913 | 12787 | 18087 / 5913 | 932 | 35658 / 48000 (74.29%) | 0 |
| `max_chunk_pubmed_cap5_diagnostic` | 3547 | 12787 | 20453 / 3547 | 932 | 33827 / 48000 (70.47%) | 0 |

Toàn bộ bảy variant dùng chung chunk ranking toàn cục nên số PubMed chunk không đổi; thí nghiệm này chủ yếu cô lập document aggregation/fusion. Tổng runtime đo được là 4.224,60 giây, peak RSS 851.337.216 byte. Báo cáo hiện có lưu số query có PubMed document, nhưng không lưu tổng số query thay đổi với mọi phương pháp; không chạy lại bảy variant chỉ để bổ sung chỉ số phụ này.

Không được chọn winner từ bảng này: giảm PubMed không đồng nghĩa tăng relevance. `review_interface.html` xáo thứ tự phương pháp theo query và giữ nhãn trống.

## Còn thiếu

Cần con người chấm pooled candidates bằng RELEVANT / PARTIALLY_RELEVANT / IRRELEVANT / UNCLEAR. Unjudged = UNKNOWN. Sau đó mới tính judged precision, nDCG/ranking quality trên tập đã chấm và báo coverage của judgment; tuyệt đối không gọi đó là official F2/recall.

## Trạng thái vòng đời

Post-run audit đã kiểm tra đủ 7 variant, 1.200 query/variant, 24.000 document slots và 48.000 chunk slots/variant, ownership/provenance, hash của hai frozen control, và giao diện review 60 query. Trạng thái hợp lệ là `READY_FOR_HUMAN_REVIEW`, stage `AWAITING_HUMAN_REVIEW`; đây không phải `READY_FOR_LEADERBOARD`.
