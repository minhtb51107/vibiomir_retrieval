# Implementation backlog (xếp theo impact/cost)

| Ưu tiên | Thay đổi | Evidence/benefit | Cost | Acceptance | Rollback |
|---:|---|---|---|---|---|
| 1 | Unified document registry + offline evaluator CI | Count/provenance hiện phân tán; ngăn đánh giá sai | CPU thấp, <1 GiB | silver validation + frozen controls pass; không leakage | giữ artifacts cũ, feature flag |
| 2 | Document-first candidate ledger, cap representative chunks/doc | PubMed 349k pairs vs 19.2k/source; max-chunk volume bias | CPU, reuse scores | arm duplication không tăng score; measured candidate coverage; provenance 100% | quay lại frozen candidate pool |
| 3 | Unified disk-backed BM25 over all 275.586 local docs | broad BM25 và source BM25 đang tách rời | vài GiB disk, giờ CPU; cần approval | deterministic Recall@K trên silver + resource gate | giữ index version cũ |
| 4 | Unified multilingual dense document index reusing vectors | Không có unified dense index; broad-extra thiếu dense | index build CPU/disk; embedding mới chỉ cho thiếu vectors sau approval | silver held-out improves; memory/disk bounds | disable dense arm |
| 5 | Versioned VI/EN/ZH medical terminology expansion | lexical cross-language gap | CPU thấp; dictionary provenance required | improves judged/silver coverage without leakage | disable expansion |
| 6 | Vietnamese acquisition quality-gated expansion | VI local identified chỉ 29.396/714.469 | network/disk/GPU cao; separate milestone | extraction success + offline candidate gain | stop per-domain worker |

Không triển khai source boost/quota tùy ý. Threshold chỉ được chọn trên development và kiểm tra held-out/human labels.
