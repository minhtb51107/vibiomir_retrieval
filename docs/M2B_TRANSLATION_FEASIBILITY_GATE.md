# M2B — Cổng khả thi dịch trực tiếp VI→EN

Ngày kiểm tra: 2026-10-14
Trạng thái: **BLOCKED_BEFORE_MODEL_DOWNLOAD**

## Phạm vi đã đóng băng

- Model duy nhất được phép thử: `Helsinki-NLP/opus-mt-vi-en`.
- Revision chính thức đã ghim: `c8d2853e77f5fae31124d993e0b35176b1c8914e`.
- Giấy phép trên model card: Apache-2.0.
- Bộ chẩn đoán giữ nguyên 60 câu từ lần sửa bridge trước: 30 câu high-risk và
  30 câu độc lập được chọn trước bằng SHA-256 theo năm strata.
- Chỉ số PubMed FTS5 hiện hữu không thay đổi; SHA-256:
  `c658fc92f6ad272db80976f09861d55a53ac23e594379a1f55fa383b986fa73b`.

Không có model OPUS/Marian VI→EN trong repository hoặc các vị trí Hugging Face/
Torch cache cục bộ đã kiểm tra. Môi trường retrieval hiện có `torch 2.5.1+cu121`
và `transformers 4.48.3`, nhưng không có `sentencepiece`, `sacremoses` hoặc
`accelerate`. Không dependency nào đã được cài và không file model nào đã được
tải trong gate này.

## Preflight tài nguyên

| Hạng mục | Số đo / bằng chứng |
|---|---:|
| D: trống | 130.539 GiB |
| RAM vật lý tổng | 5,829.0 MiB |
| RAM trống đầu audit | 1,036.7 MiB |
| RAM trống lần đo cuối | 772.2 MiB |
| RSS Python cơ sở | 16.8 MiB |
| RSS sau import `torch` | 359.9 MiB |
| RSS sau import `transformers` | 376.4 MiB |
| Trọng số PyTorch chính thức | 289 MB |
| Kích thước toàn repository model | 581 MB (gồm cả PyTorch và TensorFlow) |

Ổ đĩa đạt ngưỡng an toàn 20 GiB. RAM thì không đạt. Với runtime đã đo
`376.4 MiB`, trọng số FP32 `289 MB`, activation/buffer sinh chuỗi và biên dự
phòng, trạng thái ổn định đã cần xấp xỉ 0.75–0.85 GiB. Checkpoint là
`pytorch_model.bin`; nếu nạp thông thường không có low-memory loader, bản sao
tạm của trọng số có thể đưa đỉnh gần 1.0 GiB trước cả biên an toàn.

Biên dao động RAM quan sát ngay trong audit là 264.5 MiB. Do đó yêu cầu vận
hành có căn cứ là **ít nhất 1.25 GiB RAM vật lý trống**, và **khuyến nghị 1.5
GiB** trước khi khởi động model. Mức hiện tại 772.2 MiB thiếu khoảng 0.48 GiB
so với mức tối thiểu và 0.75 GiB so với mức khuyến nghị. Nạp model lúc này có
nguy cơ paging nặng hoặc làm lặp lại kiểu native crash đã từng xảy ra trên máy.

## Kết quả gate

Theo điều kiện dừng của thí nghiệm, model **không được tải hoặc nạp**. Vì vậy:

- chưa có bản dịch y khoa thực tế để trình bày;
- chưa có PubMed Top-5 từ bản dịch;
- tỷ lệ bảo toàn khái niệm độc lập và số mâu thuẫn nghiêm trọng đều là
  **UNKNOWN**, không được gán bằng 0;
- chưa thể kiểm tra tiêu chí `>=24/30` bảo toàn khái niệm và `<=2/30` mâu thuẫn;
- chi phí GPU là 0; độ trễ CPU cho 60/1.200 câu chưa được đo và không được suy
  đoán từ benchmark chất lượng của model card.

## Quyết định

**BLOCKED.** Chưa có bằng chứng cho phép rollout 1.200 câu. Để chạy lại đúng
gate, cần giải phóng RAM đến tối thiểu 1.25 GiB (khuyến nghị 1.5 GiB), sau đó
dùng môi trường phụ không thay đổi dependency retrieval, tải riêng các file
PyTorch/tokenizer ở revision đã ghim, dịch tuần tự CPU và dừng nếu preflight
RAM giảm dưới ngưỡng.

Cho đến khi điều kiện RAM được đáp ứng, không vá thêm TF-IDF bridge. Theo chỉ
dẫn của milestone, effort nên quay về corpus retrieval tiếng Việt và unified
document retrieval; đây là định hướng, không phải một experiment mới được
khởi chạy.
