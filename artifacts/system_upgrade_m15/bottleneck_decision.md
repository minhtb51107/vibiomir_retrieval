# Quyết định bottleneck M1.5

## Hạ tầng đã xác nhận

- Registry: 275,877 canonical documents, URL mismatch=0, missing paths=0.
- Candidate ledger: 332,407 query-document pairs cho 1.200 queries.
- Phase 12 loại khỏi Docs@20 308,407 candidate pairs (92.78%). Đây là mất mát cấu trúc, relevance vẫn UNKNOWN.
- Multi-arm query/document pairs: 19; duplicate arm/chunk rows: 19.

## Quyết định

Offline architecture comparison vẫn **NO-GO** vì primary A/D benchmark có 0 pair. Thử nghiệm kiến trúc có giá trị cao nhất tiếp theo, sau khi có judgment set, là **document-first unified candidate fusion với bounded representative chunks/document**, dùng ledger hiện tại và exact scores; không cần model inference cho replay đầu tiên.

Ước lượng replay CPU: 5–15 phút, RAM <1 GiB, disk <250 MiB; phải chạy supervisor nếu được phê duyệt. Không được diễn giải candidate discard như official recall.
