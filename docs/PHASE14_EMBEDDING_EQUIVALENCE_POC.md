# Phase 14 — kiểm toán chunk trùng và PoC embedding Colab

Phase 14 đã dừng an toàn sau `EXTRACT + CHUNK`: 2.996 tài liệu usable và 58.839 chunk duy nhất. Không có crawl, extraction, chunking, embedding hay reranking nào được chạy lại trong checkpoint này.

## Chunk trùng hệt nhau

Trong 58.839 chunk, 32.893 vị trí (55,90%) thuộc 353 nhóm văn bản trùng hệt nhau:

- 14.856 vị trí trong 5 nhóm là boilerplate toàn site như đặt lịch, lấy mẫu tại nhà, đăng ký khám và bình luận.
- 17.906 vị trí trong 327 nhóm là điều hướng “Tin cùng chuyên mục” hoặc tiêu đề bài liên quan. Chúng có vẻ y khoa nhưng không phải nội dung chính của tài liệu sở hữu chunk.
- 131 vị trí trong 21 nhóm là nội dung lặp có thể hợp lệ, chủ yếu tiểu sử bác sĩ/khối nội dung nguồn; cần đánh giá riêng nếu sau này có chính sách lọc.

Ước lượng thận trọng: 32.762/58.839 = 55,68% vị trí chunk có nguy cơ gây nhiễu retrieval do boilerplate hoặc điều hướng bài liên quan. Đây là phân loại cấu trúc dựa trên văn bản và `heading_path`, không phải nhãn relevance. Dataset gốc không bị sửa và không chunk nào bị xóa. Chi tiết 20 nhóm lớn nhất nằm trong `artifacts/phase14_medlatec_coverage_3000/duplicate_chunk_analysis.json`.

## PoC BGE-M3

Gói PoC lấy 200 chunk từ Phase 10E G1A, nơi Parquet 131.166 hàng khớp chính xác vector `float32` 131.166 × 1.024. Bộ chọn loại 26.116 hàng vector tái sử dụng từ depth1000 và chỉ lấy trong 105.050 hàng thực sự được encode bởi streaming Phase 10E. Mẫu mới có 95 Medlatec và 105 Vinmec; 58/84/58 hàng ngắn-vừa-dài.

PoC cũ đã ghép bốn hàng theo thứ tự `pending`, nhưng đó chưa phải batch thực tế. Môi trường Phase 10E còn giữ `sentence-transformers==5.7.0` và `transformers==4.48.3`; metadata của package trong `.venv` có thời điểm 2026-10-03, trước artifact embedding Phase 10E ngày 2026-10-09. `SentenceTransformer.encode()` áp dụng `np.argsort(-len(normalized_text))` riêng cho danh sách pending của từng cửa sổ 128 rồi mới cắt batch 4; sau inference nó mới hoàn nguyên thứ tự vector. Nhánh flatten/interleave không khả dụng với Transformers 4.48.3 và Phase 10E không yêu cầu Flash Attention.

Gói sửa đổi khôi phục đúng 50 batch hoàn chỉnh từ 39 cửa sổ lịch sử, sau khi loại reused rows. Mỗi hàng lưu window, vị trí pending, vị trí sau sort, batch ID và offset. Runner Colab gọi `model.encode()` độc lập đúng 50 lần, mỗi lần chỉ bốn thành viên của một batch lịch sử; tuyệt đối không encode chung 200 văn bản.

Contract cố định:

- `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`
- FP16 CUDA, batch 4, max length 512
- chuẩn hóa text: NFC rồi `split/join`
- `SentenceTransformer.encode(... normalize_embeddings=True)`
- output `float32`, dimension 1.024
- ngưỡng đã đo từ Phase 10E: max sai lệch tuyệt đối `0.000732421875`, cosine tối thiểu `0.9995`

Trạng thái hiện tại là `AWAITING_COLAB_EMBEDDING_EQUIVALENCE`; chưa có kết quả L4/A100 nên chưa được tuyên bố PASS và chưa vector nào được phép nhập production. PoC reranker A100 riêng biệt vẫn FAIL equivalence và không được dùng để nhập score.

## Chạy trên Colab

Tải lên:

- `artifacts/phase14_embedding_poc/phase14_bge_m3_embedding_poc_200.zip`
- `notebooks/colab_embedding_poc.py`

Chọn GPU L4 hoặc A100 rồi chạy:

```bash
!pip -q install sentence-transformers==5.7.0 transformers==4.48.3 psutil
!python colab_embedding_poc.py \
  --package phase14_bge_m3_embedding_poc_200.zip \
  --output colab_embedding_results.zip
```

Tải `colab_embedding_results.zip` về `artifacts/phase14_embedding_poc/`, sau đó xác minh read-only trên Windows:

```powershell
.\.venv\Scripts\python.exe tools\phase14_embedding_poc.py verify `
  --results artifacts\phase14_embedding_poc\colab_embedding_results.zip
```

Runner ghi lại batch ID và bốn `chunk_id` của từng lần gọi. Verifier chỉ so vector batch 4, kiểm tra CRC, package/hash/row identity, dimension, finite, norm, sai lệch từng chiều và cosine. Không có lệnh import production trong PoC này; ngưỡng equivalence lịch sử không thay đổi.
