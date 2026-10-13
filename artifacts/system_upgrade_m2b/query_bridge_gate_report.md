# M2B Gate 1 — kiểm tra cầu truy vấn Việt–Anh

Ngày kiểm tra: 2026-10-14  
Nguồn: `artifacts/system_upgrade_m2a/sample_query_traces.json` và chỉ mục FTS5 đã đóng băng `data/system_upgrade_m2a/pubmed_lexical.sqlite`  
Kết luận: **BLOCKED — không cho phép GPU inference**

## Phạm vi và cách hiểu

Kiểm tra thủ công một mẫu đa dạng từ 30 trace M2A: triệu chứng, bệnh, điều trị,
thuốc/vắc-xin và câu hỏi ngắn/mơ hồ. Tiêu chí chỉ là thuật ngữ và tiêu đề tiếng
Anh có giữ chủ đề y khoa được nêu trong câu hỏi tiếng Việt hay không. Đây không
phải nhãn liên quan, không phải gold của organizer và không đo Recall/F2.

Tập 30 trace được M2A chọn ở biên overlap dense/lexical thấp, nên không đại diện
ngẫu nhiên cho toàn bộ 1.200 câu hỏi. Tuy vậy, lỗi lặp lại ở nhiều loại câu hỏi
đủ rõ để kích hoạt gate an toàn trước khi chi 195.604 lượt rerank.

## Trace đại diện

| Nhóm | Query | Thuật ngữ cầu nối nổi bật | Ba tiêu đề lexical đầu | Quan sát bảo toàn chủ đề |
|---|---|---|---|---|
| Vắc-xin/bệnh nền | 277 — “44 tuổi, hen suyễn, hở van tim… có nên tiêm vắc xin Covid-19?” | `gbs`, `perimyocarditis`, `vaccination`, `guillain`, `barre` | *Vaccine hesitancy with a history of Guillain Barre Syndrome*; *Sensory Ataxic Guillain-Barré Syndrome… after mRNA COVID-19 Vaccination*; *Post COVID-19 vaccination Guillain-Barré syndrome* | Giữ chủ đề vắc-xin nhưng thu hẹp quá mức sang GBS, bỏ phần bệnh nền được hỏi. |
| Triệu chứng nhi khoa | 100 — “bé 10 tháng hết sốt rồi phát ban khắp người” | `pustulosis`, `scalded`, `ssss`, `staphylococcal`, `fever` | Ba tiêu đề đều tập trung vào staphylococcal scalded skin syndrome | Còn chủ đề sốt/phát ban nhưng seed đã khóa vào một chẩn đoán rất hẹp. |
| Điều trị thần kinh/nhiễm trùng | 727 — “Điều trị liệt toàn thân do viêm màng não” | `botulinum`, `femoral`, `alzheimer`, `hemiarthroplasty` | *Botulinum toxin… hip dislocation after hemiarthroplasty*; *Lymphovenous Bypass… Alzheimer's*; *reoperation for… hemiarthroplasty* | Sai lệch chủ đề rõ ràng; không còn viêm màng não hoặc liệt toàn thân theo ngữ cảnh câu hỏi. |
| Điều trị tiêu hóa | 783 — “Viêm đại tràng và hậu môn kéo dài nên dùng thuốc gì?” | `gemcitabine`, `capecitabine`, `oxaliplatin`, `cisplatin`, `helicobacter` | Hóa trị ung thư đường mật; tác dụng phụ điều trị H. pylori; viêm dạ dày/tiêu chảy do H. pylori | Sai lệch rõ ràng sang hóa trị và dạ dày; không bảo toàn bệnh đại tràng/hậu môn. |
| Thuốc/phơi nhiễm thai kỳ | 909 — “mang thai… dùng Listerine và xịt thơm miệng Oralan” | `tretinoin`, `botulinum`, `ondansetron`, `spray`, `lactation` | Botulinum khi cho con bú; dị tật liên quan tretinoin bôi; botulinum thẩm mỹ môi | Chỉ giữ ngữ cảnh thai/sản rất rộng, mất tác nhân được hỏi. |
| Chẩn đoán hình ảnh | 190 — “siêu âm… phân thuỳ V có nốt tăng âm 7mm” | `parathyroid`, `nodules`, `subcentimeter`, `suspicious` | Ba tiêu đề về nốt tuyến giáp dưới 1 cm | Sai lệch cơ quan rõ ràng: nốt gan bị chuyển thành nốt tuyến giáp. |
| Thuốc/thực phẩm bổ sung | 384 — “Có nên uống vitamin E hằng ngày không?” | `vitamin`, `supplements`, `intake`, `supplementation` | Vitamin E/mortality; vitamin E supplementation; safety of vitamin supplementation | Chủ đề được giữ tương đối tốt; đây là bằng chứng vận hành tích cực, không phải nhãn liên quan. |
| Điều trị sản khoa/vắc-xin | 711 — “bầu 30 tuần, tiêm uốn ván… sưng và đau bắp tay” | `ustekinumab`, `psoriasis`, `uae`, `iud`, `cesarean` | Ustekinumab cho psoriasis thai kỳ; đau sau uterine artery embolization; UAE cho sẹo mổ lấy thai | Sai lệch rõ ràng; mất phản ứng tại chỗ sau tiêm uốn ván. |
| Câu hỏi ngắn | 1024 — “Em bị a-mi-đan, có nên đi cắt không?” | `hymen`, `endometriosis`, `imperforate`, `peritoneal` | Phẫu thuật endometriosis; imperforate hymen; hymenectomy | Sai lệch hoàn toàn do từ “cắt”/phẫu thuật chung, không còn a-mi-đan. |
| Câu hỏi ngắn/răng miệng | 1199 — “Viêm lợi trùm có điều trị dứt điểm được không?” | `tuberculoma`, `sinusitis`, `rifampin`, `isoniazid`, `cured` | Ba tiêu đề về lao/hypersensitivity/QuantiFERON | Sai lệch hoàn toàn; không còn bệnh lý lợi/răng miệng. |
| Triệu chứng nhiễm trùng | 571 — “Lách to và đau sau sốt rét” | `tonsillectomy`, `laryngopharyngeal`, `swallowing`, `phlegmon` | Khó nuốt sau cắt amidan; laryngopharyngitis; xuất huyết sau cắt amidan | Sai lệch rõ ràng khỏi lách/sốt rét. |
| Bệnh/vắc-xin | 753 — “huyết áp 150/90 và gout nhẹ… tiêm vaccine Covid-19” | `gout`, `vaccination`, `vaccine`, `flare`, `booster` | Ba tiêu đề về gout flare sau tiêm COVID-19 | Bảo toàn tốt phần gout/vắc-xin; không chứng minh tính liên quan của từng tài liệu. |

## Chẩn đoán cơ chế

Cầu truy vấn lấy thuật ngữ TF–IDF trực tiếp từ tám tài liệu dense seed. Khi seed
chệch hoặc chỉ khớp một từ/ngữ cảnh rộng, các thuật ngữ hiếm của chính seed
(`hemiarthroplasty`, `hymen`, thuốc hóa trị, v.v.) được ưu tiên hơn ý định gốc.
FTS5 sau đó hoạt động đúng với truy vấn tiếng Anh đã bị lệch. Đây là sai lệch ở
bước pseudo-relevance feedback, không phải lỗi thực thi của FTS5.

Lỗi xuất hiện ở nhiều nhóm độc lập (triệu chứng, bệnh, điều trị và câu hỏi ngắn),
trong khi một số câu vẫn giữ chủ đề. Vì vậy không thể coi toàn bộ bridge là vô
dụng, nhưng cấu hình hiện tại không vượt qua gate an toàn cho một thí nghiệm GPU
duy nhất có chi phí lớn.

## Trạng thái thí nghiệm

- Gate 1: **BLOCKED** do topic drift có hệ thống trong mẫu rủi ro.
- Gate 2 (replay Phase 12 khi tắt lexical): **NOT RUN**, vì Gate 1 yêu cầu dừng.
- Gate 3 (GPU/resource preflight): **NOT RUN**.
- Điểm mới thực sự cần rerank theo M2A: 195.604.
- Điểm mới đã tính trong M2B: **0**.
- GPU/model inference: **0**.
- ZIP leaderboard: **không tạo**.

Candidate mới vẫn có mức liên quan **UNKNOWN**; báo cáo này không gán nhãn
RELEVANT/IRRELEVANT và không suy diễn điểm leaderboard.
