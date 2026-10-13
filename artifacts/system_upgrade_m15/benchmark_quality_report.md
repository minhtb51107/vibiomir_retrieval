# Chất lượng benchmark discovery

- M1 chỉ thấy 21 pair vì lọc title có dấu hỏi, `section_type=answer`, và yêu cầu chính answer block đó dài ≥80 ký tự.
- Scan rộng tìm 4 structural non-title Q&A candidates nhưng **0** pair y-sinh đủ tin cậy: bốn câu thuộc một bài phỏng vấn về cao tốc, một câu là tên/chức danh bị extractor gắn nhầm.
- Title-derived audit pool lớn nhưng không được nâng thành gold: 10,594 broad titles và 8,714 PubMed titles.
- VI→EN: **INSUFFICIENT**. PMID chỉ xác nhận document identity, không cung cấp câu hỏi tiếng Việt.

Kết luận: benchmark chính vẫn không đủ để chọn kiến trúc. Không tạo relevance labels giả và không thay thế silver benchmark M1.
