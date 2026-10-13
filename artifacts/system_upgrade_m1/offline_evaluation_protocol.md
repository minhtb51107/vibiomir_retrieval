# Protocol đánh giá offline

Silver benchmark có **21** known-source pairs, status **INSUFFICIENT_FOR_ARCHITECTURE_SELECTION**.

## Metric hợp lệ

- Known-positive Recall@20/@100, MRR, candidate coverage, source diversity.
- Chỉ chạy bằng `tools/offline_evaluator.py` trên query IDs `silver_*`.

## Không được diễn giải

- Không gọi metric này là official recall/F2.
- Query được lấy nguyên title nguồn (`verbatim_source_query=true`), nên chỉ đo source lookup/index regression, không chứng minh generalization.
- EN và VI→EN tracks hiện **INSUFFICIENT**.
- Unjudged documents là UNKNOWN, không phải irrelevant.

Split theo cluster exact/near-duplicate; một cluster không xuất hiện ở cả development và held-out. Positive phải là NEW-official doc ID và giữ URL/provenance.
