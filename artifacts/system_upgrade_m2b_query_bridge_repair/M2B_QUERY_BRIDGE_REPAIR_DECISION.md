# Quyết định kỹ thuật — sửa M2A query bridge

**Khuyến nghị: BLOCKED_FOR_M2B_GPU**

## Nguồn thuật ngữ

Không tìm thấy model dịch hoặc từ điển y khoa VI–EN chuẩn hóa trong cache/repository. Bản sửa chỉ dùng các cặp song ngữ xuất hiện trực tiếp trong tài liệu Việt đã thu thập và các literal y khoa vốn có trong câu hỏi. Mọi anchor đều mang doc_id/URL/trích đoạn nguồn; không có thuật ngữ TF–IDF từ dense seed trong truy vấn mới.

## Kết quả 60 câu

- Trace rủi ro cũ: 30; mẫu độc lập đóng băng theo SHA256 và 5 strata: 30.
- Có ít nhất một anchor truy vết được: 15/60.
  - high-risk: 12/30; độc lập: 3/30.
- Fallback an toàn về Dense Top-100: 45/60.
- Thuật ngữ seed cũ không truy vết về câu hỏi: 716 term trên 60 câu.
- Thuật ngữ mở rộng không truy vết trong bridge mới: 0.
- Candidate lexical mới: 1483 hàng; overlap Dense: 101.
- Median latency/câu: 0.0004s; ngoại suy 1.200 câu: 16.0s CPU.

Đây là kiểm tra bảo toàn chủ đề cơ học, không phải đánh giá liên quan và không suy ra F2.

## Ví dụ trước/sau

| Query | Anchor mới có provenance | Từ cũ nổi bật | Top title mới đầu tiên | Fallback |
|---|---|---|---|---|
| 190: Em có làm các siêu âm hệ tiết niệu. Kết quả ghi nhận hạ ph | — | splenic, nascent, cysts, parathyroid, myoma | Dense fallback | True |
| 277: Tôi 44 tuổi, có tiền sử hen suyễn, hở van tim đã được chữa | asthma, covid-19 | gbs, perimyocarditis, hepatopathy, keratitis, vaccination | [COVID-19 Vaccination in Asthma Patients Treated with Biologicals - Statement of the Austr | False |
| 384: Có nên uống vitamin E hằng ngày không? | vitamin e | atbc, vitamin, supplements, intake, supplementation | The effect of carrier for vitamin E on liver concentrations of vitamin E and vitamin E exc | False |
| 727: Điều trị liệt toàn thân do viêm màng não như thế nào? | meningitis | nir, ane, letm, lymphovenous, botulinum | Calculated decisions: Bacterial Meningitis Score for Children. | False |
| 783: Viêm đại tràng và hậu môn kéo dài nên dùng thuốc gì? | colitis | gemcitabine, mpfs, vcm, capecitabine, diarrhea | Noninfectious colitides: collagenous colitis, lymphocytic colitis, diversion colitis, and  | False |
| 1024: Em bị a-mi-đan, có nên đi cắt không? | tonsillitis | hymen, seacs, endometriosis, imperforate, superficial | [Treatment of tonsillitis]. | False |
| 1199: Viêm lợi trùm có điều trị dứt điểm được không? | — | turbinectomy, ripe, tuberculoma, sinusitis, tbm | Dense fallback | True |

## Gate

Bridge mới loại bỏ cơ chế seed-specific term thay thế ý định gốc. Tuy nhiên lexicon cục bộ chỉ có phạm vi hẹp; các câu không có anchor phải quay về Dense và không tạo tín hiệu lexical độc lập. Vì vậy quyết định chỉ là GO nếu không còn drift lặp lại và tỷ lệ anchor đủ cho một M2B có ý nghĩa; nếu không thì BLOCKED.

Chi tiết đủ 60 câu: `artifacts\system_upgrade_m2b_query_bridge_repair\query_bridge_60_diagnostics.json`.
