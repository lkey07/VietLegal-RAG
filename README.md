# Vietnamese Legal QA · Fixed RAG (UIT DSC 2026)

Hệ thống hỏi–đáp pháp luật tiếng Việt cho UIT Data Science Challenge 2026, Task 2. Đây là pipeline **fixed, deterministic RAG**: truy xuất bằng BM25 và dense embedding, hợp nhất kết quả, bổ sung ngữ cảnh cùng Điều, rồi sinh câu trả lời bằng Qwen3.5-2B đã fine-tune LoRA trên dữ liệu chính thức.

> **Kết quả tốt nhất đã nộp trên public-1000:** **METEOR 0.5584 · ROUGE-L 0.5727** (E23, 1.000 câu). Đây chỉ là điểm **public**; chưa có kết quả private để công bố.

## Bài toán và đóng góp

Một chunk pháp luật tìm đúng từ khóa vẫn có thể thiếu phần quy định đối tượng áp dụng, điều kiện hoặc ngoại lệ ở cùng Điều. Pipeline giải quyết hai nút thắt: chọn đúng nguồn bằng hybrid retrieval và đưa đủ phần văn bản pháp luật liên quan vào prompt mà vẫn giữ giới hạn token. Generator được huấn luyện với cùng định dạng metadata mà nó nhận khi suy luận.

| Thành phần | Cách triển khai ở bản E23 |
| --- | --- |
| Dữ liệu | Corpus và 5.636 câu train chính thức; tách train/dev theo nhóm câu hỏi đã chuẩn hóa để tránh trùng câu. Đáp án train chỉ giám sát phần trả lời, không dùng làm nhãn retrieval. |
| Index | Văn bản được audit, chuẩn hóa thận trọng và chia theo cấu trúc Điều/Khoản. BM25 trên SQLite FTS5; dense index dùng `AITeamVN/Vietnamese_Embedding`. |
| Retrieval | Mỗi nhánh lấy ứng viên, hợp nhất bằng weighted RRF với trọng số BM25/dense 0,5/0,5; lấy 12 chunk xếp hạng cao nhất. **Không dùng reranker** ở bản E23. |
| Context | Bổ sung tên nguồn, số văn bản khi xác định được và tên Điều. Gộp các đoạn trùng/chồng lấn, giữ nguyên 12 seed; mở rộng bằng các span nguyên văn, có giới hạn, từ **cùng Điều**. Metadata cùng nguồn chỉ ghi một lần. |
| Generator | `Qwen/Qwen3.5-2B` + LoRA rank 8, alpha 16, huấn luyện mới từ base trong 1 epoch trên 5.636 QA chính thức với ngữ cảnh có metadata; giới hạn chuỗi train 3.072 token. |
| Suy luận | Prompt tối đa 8.192 token, greedy decoding (`do_sample=false`, `num_beams=1`), `max_new_tokens=704`. Hai T4 xử lý 500 câu mỗi GPU; checkpoint sau từng question ID và từ chối resume khi identity thay đổi. |
| Đầu ra | `submission.zip` chỉ chứa UTF-8 `submission.json` theo mapping `question_id → {"answer": string}`. |

Không có Agent, model API, web search hay dữ liệu/synthetic QA ngoài phạm vi được phép. Hệ thống E23 dùng embedding và generator; các thao tác BM25/RRF/context là thuật toán không có tham số. Giới hạn tham số được kiểm tra theo **model gốc**, không trừ đi nhờ quantization/LoRA.

## Các mốc tăng điểm public

Tất cả số dưới đây là **METEOR/ROUGE-L public trên cùng 1.000 câu**, không phải điểm dev hay dự đoán private. Mỗi mốc chỉ nêu thay đổi chính so với phiên bản ngay trước nó; E08B là mốc đầu tiên của pipeline fixed-RAG mới, độc lập với code M45–M48 cũ.

| Bản | Submission ID | METEOR public | ROUGE-L public | Thay đổi dẫn tới bản nộp |
| --- | ---: | ---: | ---: | --- |
| **E08B** | 911313 | **0.5209** | 0.5533 | Hybrid BM25 + AITeam dense, RRF 0,5/0,5, top-12 context; Qwen3.5-2B fine-tune LoRA rank 8 có context, output tối đa 704 token. |
| **E18** | 915723 | **0.5251** (**+0.0042**) | 0.5623 | Giữ retrieval, thứ tự chunk, adapter và decoding; thêm metadata nguồn chính xác (tên nguồn, số văn bản không mơ hồ, tên Điều) vào context khi suy luận. |
| **E19** | 916686 | **0.5309** (**+0.0058**) | 0.5632 | Train lại LoRA rank 8 **từ base** với cùng metadata E18 trên 5.636 câu chính thức; định dạng train và inference đồng nhất. |
| **E23** | 917914 | **0.5584** (**+0.0275**) | 0.5727 | Giữ adapter E19, top-12 retrieval và greedy max704; gộp span chồng lấn và mở rộng có giới hạn bằng đoạn nguyên văn **cùng Điều** quanh seed. |

Từ E08B đến E23, METEOR public tăng **0.0375 điểm tuyệt đối**. Các số trên được đối chiếu từ ảnh leaderboard do nhóm vận hành cung cấp; repo không phát hành prediction hoặc đáp án. Đây là chuỗi kết quả quan sát được, **không chứng minh** từng cải tiến sẽ tiếp tục hiệu quả trên private. Các thử nghiệm khác chưa vượt điểm public E23 không được trình bày như bản đã chọn.

## Luồng xử lý

```text
Official legal corpus
  → audit / cleaning bảo toàn Điều–Khoản và số liệu pháp lý
  → legal chunking → SQLite FTS5 BM25 + AITeam dense index

Câu hỏi
  → BM25 top-k + dense top-k → weighted RRF → ranked top-12 seeds
  → metadata nguồn → gộp span trùng → bounded same-article expansion
  → Qwen3.5-2B + E19 metadata-aware LoRA → greedy max704
  → kiểm tra đủ ID / câu trả lời / định dạng → submission.zip
```

Mở rộng ngữ cảnh diễn ra **sau retrieval**: không đổi thứ tự top-12, không tìm kiếm lại và không lấy đáp án chuẩn để chọn văn bản. Bản E23 dùng đúng chính sách E21 GPU1; mỗi parent bị chặn ở 1.200 generator token hoặc 6.000 ký tự, đồng thời không được đẩy 12 seed gốc ra khỏi prompt.

## Mã nguồn trong repository

- [`src/uit_dsc_fixed_rag/`](src/uit_dsc_fixed_rag/) — snapshot mã nguồn đúng gói E23 `code-v57`; các module trọng tâm: `e08_context_retrieval.py`, `e18_source_metadata.py`, `e19_metadata_lora.py`, `e21_parent_context.py`, `final_public_e23.py`.
- [`configs/`](configs/) — contract và checksum được pin cho E23 cùng các bước upstream cần thiết.
- [`scripts/run_final_public_e23_kaggle.py`](scripts/run_final_public_e23_kaggle.py) và [notebook public E23](notebooks/FINAL-public1000-e23-parent-expanded-max704-dual-gpu-kaggle.ipynb) — runner và workflow Kaggle hai GPU.
- [`docs/`](docs/) — mô tả chính sách E21/E23 và [đoạn trình bày dự án cho CV](docs/CV-PROJECT.md).

Repo **chỉ phát hành source/config/notebook/tài liệu**. Không có official corpus/QA/scorer archive, index, adapter, checkpoint, public prediction, `submission.zip`, Kaggle output hoặc secret. Chạy lại đầy đủ đòi hỏi người có quyền truy cập cung cấp các input chính thức và tự xây dựng/tải artifact E00-v2, E08A, E19, E21 theo checksum trong config. Đây không phải bộ chạy một lệnh từ clone trống.

## Tính đúng đắn và giới hạn

- Kiểm tra input/artifact bằng SHA-256 và identity của code/config/model; resume fail-closed.
- Chia dev theo normalized question để hạn chế leakage; không dùng đáp án public/private khi retrieval hoặc generation.
- Chỉ nộp khi đủ 1.000 ID, đáp án không rỗng và ZIP đúng contract scorer.
- Điểm public có thể không phản ánh private; không công bố hay suy đoán điểm private.

Chi tiết triển khai: [E21 context policy](docs/27-E21-PARENT-CONTEXT.md) · [E23 public trial](docs/29-FINAL-PUBLIC-E23-PARENT-EXPANDED.md).
