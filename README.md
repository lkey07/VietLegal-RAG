# VietLegal-RAG

## Inference-Aligned Hierarchical RAG for Vietnamese Legal QA

VietLegal-RAG là hệ thống hỏi–đáp pháp luật tiếng Việt được phát triển cho
UIT Data Science Challenge 2026, Task 2. Hệ thống kết hợp lexical retrieval,
dense retrieval, ngữ cảnh phân cấp theo Điều luật và LoRA fine-tuning trong một
pipeline deterministic, có kiểm tra identity bằng SHA-256 và checkpoint theo
từng question ID.

> **Thành tích:** **Top 13 public leaderboard**, public-best METEOR **0.5817**;
> private-best METEOR **0.597327402**, ROUGE-L **0.586192885** trên 1.918 câu và đạt thứ hạng top 11 private.

## Đóng góp chính

- Xây dựng hybrid retrieval trên corpus pháp luật chính thức bằng SQLite.
  BM25 và `AITeamVN/Vietnamese_Embedding`, sau đó hợp nhất thứ hạng bằng RRF.
- Giữ 12 seed context, bổ sung metadata nguồn và mở rộng có giới hạn sang toàn
  Điều hoặc các chunk lân cận mà không dùng đáp án làm retrieval label.
- Fine-tune generator bằng đúng định dạng parent-expanded evidence và prompt
  được dùng khi inference, giảm train–inference context mismatch.
- Generation greedy 1.024 token, chỉ sinh lại câu chạm trần từ prompt gốc với
  giới hạn 1.536 token; hậu xử lý chỉ xóa phần lặp ở suffix.
- Batch inference fail-closed: pin model/config/artifact hash, checkpoint sau
  từng question ID và chỉ resume khi toàn bộ identity khớp.

## Pipeline private tốt nhất

```text
Official legal corpus
  → conservative cleaning và Điều/Khoản-aware chunking
  → SQLite FTS5 BM25 top-40
  + AITeamVN/Vietnamese_Embedding top-40
  → weighted RRF 0.5/0.5, k=60
  → fused top-20 → lấy 12 seed đầu
  → bounded parent expansion theo cùng Điều
  → metadata: tên nguồn, số văn bản, Điều và tiêu đề Điều
  → Vi-Qwen2-3B-RAG + E45 rank-8 LoRA
  → greedy max1024
  → fresh max1536 restart cho finish_reason=length
  → E43+E44 Unified Clean, suffix-only
  → validated submission.zip
```

Không có reranker trong bản đạt điểm private cao nhất. Parent expansion thử lần
lượt `whole_article → both_neighbors → previous_neighbor → next_neighbor`, với
giới hạn 1.200 token/6.000 ký tự cho parent và 8.192 input token. Các span
chồng lấn hoặc tiếp giáp của cùng văn bản/Điều được merge; 12 seed gốc không bị
loại để nhường chỗ cho expansion.



## Tiến trình public

| Bản | METEOR public | ROUGE-L public | Thay đổi chính |
| --- | ---: | ---: | --- |
| E08B | 0.5209 | 0.5533 | Hybrid BM25+dense, RRF top-12 và context-aware LoRA. |
| E18 | 0.5251 | 0.5623 | Thêm source/document/article metadata chính xác. |
| E19 | 0.5309 | 0.5632 | Train adapter mới với metadata giống inference. |
| E23 | 0.5584 | 0.5727 | Bounded same-article parent expansion. |
| E43/E44 | **0.5817** | E44 cải thiện nhẹ so với E43 | Vi-Qwen E38, FP16 max1024, length-only max1536 và hai tầng suffix cleanup. |



## Tính đúng đắn và giới hạn

- Train/dev được group theo normalized question để hạn chế leakage.
- Reference answer không được đọc trong retrieval hoặc private inference.
- Model stack được kiểm tra dưới giới hạn 4 tỷ tham số theo tham số công bố;
  quantization và LoRA không được dùng để lách cách tính này.
- Mọi output phải đủ ID, answer không rỗng và ZIP chỉ chứa UTF-8
  `submission.json` đúng schema của scorer.
- Điểm leaderboard là kết quả quan sát của submission đã lưu, không chứng minh
  riêng từng thay đổi sẽ luôn tạo cùng mức tăng trên dữ liệu khác.
