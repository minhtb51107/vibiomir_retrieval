# Phase 14 — kiểm toán chunk trùng và PoC embedding Colab

Phase 14 đã dừng an toàn sau `EXTRACT + CHUNK`: 2.996 tài liệu usable và 58.839 chunk duy nhất. Không có crawl, extraction, chunking, embedding hay reranking nào được chạy lại trong checkpoint này.

## Chunk trùng hệt nhau

Trong 58.839 chunk, 32.893 vị trí (55,90%) thuộc 353 nhóm văn bản trùng hệt nhau:

- 14.856 vị trí trong 5 nhóm là boilerplate toàn site như đặt lịch, lấy mẫu tại nhà, đăng ký khám và bình luận.
- 17.906 vị trí trong 327 nhóm là điều hướng “Tin cùng chuyên mục” hoặc tiêu đề bài liên quan. Chúng có vẻ y khoa nhưng không phải nội dung chính của tài liệu sở hữu chunk.
- 131 vị trí trong 21 nhóm là nội dung lặp có thể hợp lệ, chủ yếu tiểu sử bác sĩ/khối nội dung nguồn; cần đánh giá riêng nếu sau này có chính sách lọc.

Ước lượng thận trọng: 32.762/58.839 = 55,68% vị trí chunk có nguy cơ gây nhiễu retrieval do boilerplate hoặc điều hướng bài liên quan. Đây là phân loại cấu trúc dựa trên văn bản và `heading_path`, không phải nhãn relevance. Dataset gốc không bị sửa và không chunk nào bị xóa. Chi tiết 20 nhóm lớn nhất nằm trong `artifacts/phase14_medlatec_coverage_3000/duplicate_chunk_analysis.json`.

## PoC BGE-M3

Gói PoC lấy 200 chunk từ Phase 10E G1A, nơi Parquet 131.166 hàng khớp chính xác vector `float32` 131.166 × 1.024. Bộ chọn loại 26.116 hàng vector tái sử dụng từ depth1000 và chỉ lấy trong 105.050 hàng thực sự được encode bởi streaming Phase 10E. Mẫu có 105 Medlatec và 95 Vinmec; 46/83/71 hàng ngắn-vừa-dài.

Mẫu giữ nguyên 50 batch bốn hàng liên tiếp, căn theo chỉ số Parquet gốc. Pipeline lịch sử dùng batch 4 trong cửa sổ 128 hàng, nên cách này giữ đúng padding/batch context thay vì ghép các hàng rời rạc.

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
  --output colab_embedding_results.zip \
  --benchmark-larger-batches
```

Tải `colab_embedding_results.zip` về `artifacts/phase14_embedding_poc/`, sau đó xác minh read-only trên Windows:

```powershell
.\.venv\Scripts\python.exe tools\phase14_embedding_poc.py verify `
  --results artifacts\phase14_embedding_poc\colab_embedding_results.zip
```

Benchmark batch lớn chỉ là telemetry. Verifier chỉ so vector batch 4, kiểm tra CRC, package/hash/row identity, dimension, finite, norm, sai lệch từng chiều và cosine. Không có lệnh import production trong PoC này.
