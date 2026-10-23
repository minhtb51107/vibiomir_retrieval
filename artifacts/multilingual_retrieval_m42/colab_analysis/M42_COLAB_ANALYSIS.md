# M4.2 — Tổng hợp kết quả Colab thực tế

Ngày phân tích: **24/10/2026**. Chỉ đọc dữ liệu tải về và tính lại phép toán
trên vectors có sẵn bằng CPU; không chạy lại embedding/reranker hoặc inference API.

## 1. Kết luận

**Trial20 PASS trong phạm vi nội bộ Colab; Dense đã hoàn tất thật.**
ZH-query Dense tốt hơn Vietnamese-query Dense trên các partial-positive đã biết,
nhưng chưa vượt lexical M4 về retention Top20. RRF không tạo gain trên bộ partial
này, còn làm mất document và thường đổi sang representative chunk kém phù hợp.
Chưa có direct-answer Trung Quốc được xác nhận độc lập; chưa có cơ sở tạo
submission hoặc tuyên bố cải thiện FINAL.

**Bước tiếp theo duy nhất đề xuất: sửa/chạy ablation lựa chọn evidence window
trong các document đã retrieve (hướng B), giữ nguyên document pool/order.**
Không mở rộng crawl trước khi giải quyết lỗi chọn đúng tài liệu nhưng sai đoạn.

## 2. Đã kiểm tra những gì

Input gốc: `colab_results/M42_Colab_Dense_Results.zip`, **5,406,072 bytes**.
SHA256: `54c6426c00d5766b61c6d961a1f7d526bf148cc58746fe60ac76e1b407ab7a9e`.

- CRC PASS; cả10 files extracted khớp byte-hash với ZIP; không có file thừa.
- `trial20_execution.json`: status PASS, passed=true, đúng frozen trial hash
  `2f99d5de39c0cd11c9dbbb3b1296d58430417d70ed3bd53c50bc49a0a30af625`.
- `dense_report.json`: DENSE_COMPLETE_PENDING_SOURCE_REVIEW,12 queries,
  2,188 chunks,2,212 checkpoint rows; progress READY_FOR_HUMAN_REVIEW,2212/2212.
- SQLite mở read-only: quick_check=ok, integrity_check=ok; **553 windows** liên
  tục, mỗi window4 rows. Kiểm lại từng checksum vector; đủ2212 rows, không gap.
- Vectors thực tế2212x1024 FP32, finite100%, norms0.9997559–1.0004801.
  Layout:2188 chunks +12 Vietnamese queries +12 automatic Chinese queries.
- Checkpoint signature khớp progress và công thức manifest/runtime của runner.
- Tính lại exact VI-Dense, ZH-Dense Top100 và RRF từ vectors/ranks cached:
  **0 khác biệt identity/order**, dense score difference0.0 trong lần CPU replay.
- M4/M4.1 final identities khớp frozen baselines trên12/12 queries.
- `failure_taxonomy.json` có đủ12 units, phần inherited khớp nguyên bản M4.1;
  mọi partial position trong taxonomy/dense report khớp rankings thực tế.
- Current-official membership/canonical URL PASS cho mọi owner trong corpus;
  cả12 review excerpts khớp nguyên văn chunk, doc_id và canonical URL.

**Sửa số lượng corpus:** candidate Parquet và row manifest thực tế có **361
unique doc IDs**,2,188 chunks. M4 report cũ ghi368 documents. Chưa xác định từ
gói results vì sao chênh7; không tự coi7 IDs đó đã được embedded. Tất cả arms ở
đây dùng cùng frozen361-owner/2188-chunk universe, nên phép so sánh vẫn đồng nhất.

## 3. Runtime, trial và giới hạn equivalence

Colab dùng NVIDIA A100-SXM4-80GB, Linux, Python3.13.16, torch2.11.0+cu130,
CUDA13.0; sentence-transformers5.7.0, transformers4.48.3, NumPy2.1.3.
Không phải Windows/torch2.5.1+cu121/NumPy2.5.3 của laptop.

| Số đo | Kết quả |
|---|---:|
| Trial model revision khai báo |5617a9f61b028005a4858fdac845db406aefb181|
| Model load time, trial |2.1691s|
| Trial inference20 texts |1.3888s|
| Internal batch-consistency check |0.1349s|
| Reverse-context max difference, theo report |0.0|
| Reverse-context minimum cosine, theo report |0.99999988|
| Trial norm error, tính lại từ NPY |0.00044501|
| Trial peak GPU allocated, theo report |1,138.19MiB|
| RAM available sau trial, theo report |166,047.84MiB|
| Dense elapsed, theo report |23.1295s|

Trial NPY shape20x1024/finite được kiểm lại. So20 trial rows với đúng rows trong
full Dense: max difference0.00045776, min cosine0.99999219, trong tolerance
0.000732421875 /0.9995 đã có. Đây là continuity/batch-context check **cùng run**,
không phải source-laptop→Colab equivalence. Reverse-control tensor không được
trả về, nên difference0.0 của riêng reverse check chỉ được xác nhận từ report.

Peak host RSS/private, minimum commit headroom, GPU reserved peak và phân rã
thời gian Dense load/inference không có trong gói: **UNKNOWN**, không suy ra từ
RAM available-after hoặc thời gian23.13s. Mười một model files được report là
đã checked, nhưng không có hash inventory/Colab notebook trong results để kiểm
độc lập cách load/tokenization. Model revision là khai báo khớp manifest, không
phải bằng chứng kiểm định toàn bộ execution contract xuyên runtime.

`source_target_equivalence=NOT_VERIFIED`; `source_score_reuse=PROHIBITED`.
Reranker **NOT_RUN_AND_NO_HISTORICAL_SCORE_REUSE**. Không nhập19 scores cũ vào
cache Colab hay giả lập scores mới.

## 4. So sánh retention đúng phạm vi

Cả VI_DENSE và ZH_DENSE đều retrieve **tài liệu tiếng Trung**; chỉ khác ngôn ngữ
query. Đây không phải kết quả VI→VI hay VI→EN.

Các số dưới là retention của13 **AI-silver partial query-document pairs**,
không phải official Recall/Precision. Chúng nằm trên6/12 query units, gồm11
exact evidence strings khác nhau: hai cặp document có nội dung trùng. Không
coi các document chưa judged là negative.

| Arm |Partial pairs Top20|Partial pairs Top100|Units có partial Top20|Directory-tagged Top20|
|---|---:|---:|---:|---:|
|M4 lexical candidates|13/13|13/13|6|101/240|
|M4.1 lexical candidates|12/13|12/13|6|48/240|
|VI-query Dense|3/13|12/13|2|90/240|
|Automatic-ZH-query Dense|11/13|13/13|6|76/240|
|Three-arm RRF|11/13|12/13|6|66/240|
|M4 final admission|11/13|N/A|5|55/121|
|M4.1 final admission|6/13|N/A|2|0/17|

M4/M4.1 final chỉ có tổng121/17 rows sau admission; không phải các search arms
có đầy đủ100 candidates. Vì vậy không gọi11/6 ở hai dòng final là Top100 recall.
Page-type là tag cached, chưa phải review chất lượng từng chunk: một số đoạn
navigation nằm trong document được tag MEDICAL_ARTICLE. Các candidate arms
được phép chứa rejected content để chẩn đoán; số directory không phải số lỗi
production admission, vì final semantic admission/reranker chưa chạy.

So với M4.1 candidate Top20, VI Dense có187, ZH Dense163, RRF120 query-document
slots mới. Đây chỉ là novelty. Review bên dưới không cho phép đổi novelty thành
relevance gain.

## 5. Tất cả13 known partial pairs

`—` = không có trong saved Top100, không có nghĩa không nằm trong corpus.

|Query|Doc ID|M4 lex|M4.1 lex|VI Dense|ZH Dense|RRF|
|---|---:|---:|---:|---:|---:|---:|
|128|3004366|4|10|37|13|8|
|128|3019879|6|7|40|5|5|
|128|3019895|7|8|39|6|6|
|128|3027148|1|3|29|1|2|
|128|3058523|2|4|30|2|3|
|409|3078850|2|6|20|16|7|
|571|3003971|1|1|45|2|2|
|715|3078785|11|—|—|83|—|
|715|3078801|3|9|66|10|10|
|715|3078924|1|8|60|94|21|
|805|3094304|4|2|1|1|1|
|805|3820076|9|6|6|15|6|
|1080|3055183|1|1|31|8|2|

ZH Dense khôi phục3078785 vào Top100 nhưng RRF cắt mất. RRF không giữ được
3078924 trong Top20 dù M4/M4.1 lexical đặt nó rank1/8. Không tự coi các catalog
partials này là đầy đủ bằng chứng điều trị HP.

Ngoài13 pairs, diagnostic doc1595280 của query128 còn trong ZH Dense rank19,
nhưng RRF rank25, VI Dense ngoài Top100. M4 rank5, M4.1 rank1. Đây là case từng
được M4 gọi direct nhưng M4.1 hạ đánh giá về differential/partial; không dùng
static M4 label làm ground truth độc lập.

### Document retention không phải evidence-window retention

Trong13 old silver excerpts, chỉ5 còn nguyên văn trong một frozen chunk;7 nếu
chỉ chuẩn hóa NFC/whitespace để đối chiếu serialization. Ghép lại các chunks
theo offsets không tìm thêm old excerpt nào ngoài7 trên. Không kết luận phần
source còn lại thiếu câu trả lời: có thể rechunk, extraction hoặc khác window.

|Arm|Giữ nguyên old excerpt trong representative Top20|Sau chuẩn hóa whitespace|
|---|---:|---:|
|M4 lexical|2|3|
|M4.1 lexical|0|1|
|VI Dense|0|1|
|ZH Dense|2|3|
|RRF|0|1|

Đây là kiểm tra containment của whole old excerpt, không phải semantic score.
Đoạn khác trong cùng doc vẫn có thể hữu ích. Nó chứng minh không được báo
“11/13 evidence giữ được” chỉ từ11 document IDs.

## 6. Evidence mới và regression cụ thể

Đọc đủ12 rows trong `review_queue.json`, đối chiếu source và lưu riêng
`assistant_review_12.csv`: **1 PARTIAL,11 WRONG,0 DIRECT**, trạng thái
AI_JUDGED_M42_ASSISTANT_UNBLINDED. Không blind, không human-verified; không
ngoại suy tỷ lệ lỗi này sang240 candidates hay toàn corpus. Queue chỉ phủ8/12
units và được chọn bằng hash, không phải12 cửa sổ có score cao nhất.

- **409 /3037030 /3037030-2f1d575023096230e08b**, VI Dense rank7:
  `女性不育`, có “毛发 分布…观察有无多毛、男性化”. Hỗ trợ ngữ cảnh khám nội tiết/
  rậm lông, nhưng chưa trả lời nguyên nhân và cách xử trí nang lông/nách cụ thể.
  URL: https://www.a-hospital.com/w/%E5%A5%B3%E6%80%A7%E4%B8%8D%E8%82%B2
- **715 /1662742 /1662742-416d4bece59d74322fe1**, ZH Dense/RRF rank1:
  nói HP và đường lây, kết thúc bằng “采取必要的正规抗HP治疗”. Đúng entity nhưng
  không cung cấp đầy đủ dấu hiệu và cách diệt HP như query yêu cầu.
  URL: http://www.cnkang.com/dise/9/391001.html
- **805 /3094304 /3094304-360cbcd7564344a5969f**, cả Dense và RRF rank1:
  representative là định nghĩa nam giới vô sinh. Trong cùng owner đã có các
  chunks `...-85965c5cdc91527467d2` về rối loạn sinh tinh và
  `...-8578856f43385ef6c987` về nhiễm trùng sinh dục. Câu trả lời từng phần về
  nguyên nhân nằm trong source nhưng representative không chọn chúng; phần tuổi
  dễ mắc vẫn chưa được chứng minh. Không cần crawl để kiểm tra lỗi window này.
- **409 /3016157**, ZH Dense rank1 là đoạn điều trị đa毛症; RRF vẫn doc rank1
  nhưng dùng đoạn “参看…药品列表…医院列表”. Đây là regression ở chunk selection,
  không phải document retrieval. Tính chung,165/239 RRF Top20 docs cũng có trong
  ZH Top100 dùng chunk khác. Khác chunk không tự động là sai, nhưng ví dụ này sai rõ.
- **446 /3062674**, ZH Dense rank1 là newborn hemolysis, có
  “以未结合胆红素为主”; query hỏi direct/conjugated bilirubin hơi cao. Population
  newborn không được thiết lập trong query và bilirubin subtype không phù hợp.
- **634**: bridge đã giữ `不饮酒 不喝酒 无饮酒史` cùng GGT; ZH Dense/RRF vẫn đưa
  alcohol-induced mental disorder3138129 lên rank1. Không được đổ lỗi mặc định
  cho việc bản dịch làm mất phủ định: đây là contrast/intent failure sau representation.
- Queue còn có catalog/navigation3095750,3654892,3008961,3037031,3095737,
  3008959. Sáu rows này không phải answer text; ba trong số chúng vẫn mang tag
  MEDICAL_ARTICLE. Các rows sai khác gồm ICP3149391, menopause3037051,
  pharyngitis3111825, sexual-exercise3865998 và cystic-fibrosis3027550.

## 7. Failure taxonomy trên12 development units

|Query|Chẩn đoán có bằng chứng/giới hạn hiện tại|
|---|---|
|128|ZH giữ sneezing partials; VI ưu tiên sleep/somnolence; RRF hạ1595280 khỏi20 và đổi representative. Chưa giải thích đầy đủ quan hệ thức giấc–hắt hơi.|
|257|Intestinal spasm + beef/diet chưa có answer được xác nhận; Dense trả pain/catalog. Missing coverage vẫn UNKNOWN, không phải chứng minh source không có.|
|290|Nasopharyngeal cyst/night obstruction bị trôi sang pharyngitis/sleep-apnea/nasopharyngeal-cancer remedy. Entity/clinical-relation discrimination thất bại.|
|409|Bridge rút còn female/folliculitis, bỏ dominant coarse axillary hair concern; Dense tìm endocrine/hirsutism liên quan, RRF có thể lấy navigation. Understanding + window selection.|
|446|Direct bilirubin bị trộn neonatal/unconjugated và generic lab navigation; candidate relevance chưa bảo toàn subtype/context.|
|491|Bridge bỏ nhiều chi tiết X-ray/AST và còn UNRESOLVED_NEGATION_SCOPE; menopause/chest topic overlap không đủ trả lời. Query-understanding và source answer coverage chưa xác định.|
|513|VitaminC + hypothyroid bridge có đủ core terms; returned endocrine/vitamin catalog không giải thích tương tác. Không lấy co-occurrence làm answer; coverage UNKNOWN.|
|571|Malaria/spleen partial doc rank45VI→2ZH; new GI drug catalog sai bệnh. Cơ chế chưa đủ management cho lách to/đau.|
|634|Phủ định có trong bridge nhưng bị ranking/fusion lấn bởi alcohol topics; GGT-specific non-alcohol explanation chưa được xác nhận.|
|715|ZH tìm lại partial ở83 nhưng RRF làm mất; top1 mới giải thích transmission không eradication. Candidate truncation + intent/window gap.|
|805|Đúng male-infertility doc rank1 nhưng chọn definition thay risk/cause sections; M4.1 admission từng loại relation. Age component chưa có bằng chứng.|
|1080|ZH top1 là symptom directory; RRF chọn pregnancy lifestyle/general pregnancy; labor-vs-right-chest-pain relation chưa được trả lời.|

Không có hidden gold, không gán mới HUMAN_VERIFIED. Đây là phân tích development,
không phải generalization trên held-out. Không đủ dữ liệu để kết luận phải mở
rộng corpus, cũng không có bằng chứng mọi query đều giải được từ source hiện có.

## 8. Một thí nghiệm tiếp theo — giữ docs, sửa representative

**GO cho ablation cache-only về evidence-window selection; NO-GO cho promotion
RRF hiện tại hoặc submission multilingual.**

Freeze cùng RRF document IDs/order và cùng Top20/100. A giữ first-arm lexical
representative hiện tại. B chọn window có ZH-query similarity cao nhất trong
các canonical chunks cùng owner, sử dụng vectors hiện có; không đổi RRF doc
scores, query representation, corpus hay admission thresholds. Không hardcode
query IDs. So exact-window retention, và blind review A/B trên12 units để xác
định answer completeness/contradiction, không lấy cosine làm verdict.

Hypothesis có thể bác bỏ: representative first-arm đang làm mất body evidence
đã tìm thấy, nên sửa riêng window choice giúp evidence tốt hơn mà không cần
thêm dữ liệu. Tính toán là CPU dot products trên9MB vectors; không cần GPU/API,
crawl, reranker mới hay source-score interoperability. Chưa benchmark policy
mới nên không hứa runtime/gain cụ thể. Đây chỉ là đề xuất, **chưa chạy**.

Nó không tự sửa document3078785 bị cắt khỏi RRF100, q634 phủ định hay q491 hiểu
query thiếu. Nếu source review không xác nhận gain, không promote và quay lại
phân tích relation/coverage cụ thể, không mass crawl. Model-score reuse xuyên
runtime vẫn bị chặn; bounded reranker chỉ xét sau resource/equivalence riêng.

## Reproduction / phạm vi bất biến

```powershell
.\.venv\Scripts\python.exe tools\analyze_m42_colab.py
.\.venv\Scripts\python.exe -m pytest tests/test_m42_colab_analysis.py tests/test_multilingual_m42.py tests/test_multilingual_m42_portable.py tests/test_multilingual_m42_execution.py -q
```

25 tests PASS. File số liệu đầy đủ: `verified_analysis.json`.
Raw ZIP/extracted/vectors/SQLite và frozen M4/M4.1 không thay đổi; không rerun
inference, không tạo competition ZIP, không auto-submit. Phase27 BEST0.0923
vẫn giữ SHA `b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`.
