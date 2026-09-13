# CV-ready project entry — Vietnamese Legal QA Fixed RAG

## Bản ngắn (đưa vào CV)

**Vietnamese Legal QA · Fixed RAG | UIT DSC 2026 Task 2**

- Xây dựng pipeline RAG pháp luật Việt Nam với BM25 + dense retrieval (`AITeamVN/Vietnamese_Embedding`), weighted RRF và mở rộng ngữ cảnh nguyên văn theo cùng Điều; giữ nguyên các top-12 evidence seed và giới hạn token để giảm mất điều kiện/ngoại lệ pháp lý.
- Fine-tune `Qwen3.5-2B` bằng LoRA rank 8 trên **5.636 câu hỏi–đáp án chính thức**, đồng bộ metadata nguồn giữa train và inference; dùng greedy decoding, checkpoint từng câu và kiểm tra artifact bằng SHA-256.
- Đạt **METEOR public 0.5584, ROUGE-L public 0.5727 trên 1.000 câu**. Các mốc METEOR public: **0.5209 → 0.5251 → 0.5309 → 0.5584** nhờ lần lượt bổ sung metadata suy luận, train metadata-aware và bounded same-article context expansion.

**Kỹ thuật:** Python, PyTorch, Transformers/PEFT, QLoRA, SQLite FTS5 BM25, dense retrieval, reciprocal rank fusion, legal document chunking, Kaggle dual-T4, checksum-based validation.

## Bản chi tiết (portfolio / phỏng vấn)

**Bối cảnh.** Bài toán là trả lời câu hỏi pháp luật tiếng Việt dựa trên corpus chính thức của UIT Data Science Challenge 2026 Task 2. Một chunk có thể chứa mức phạt nhưng thiếu phạm vi áp dụng hoặc ngoại lệ nằm ở đoạn khác của cùng Điều. Vì thế, chỉ tăng số chunk hoặc output token không đủ để bảo đảm câu trả lời tốt hơn.

**Thiết kế.** Audit và chia văn bản theo Điều/Khoản; kết hợp tìm kiếm từ khóa BM25 với embedding dense bằng weighted RRF 0,5/0,5. Từ 12 chunk truy xuất, đưa metadata nguồn chính xác, gộp span trùng/chồng lấn và thêm phần nguyên văn cùng Điều theo ngân sách chặt chẽ, không loại seed gốc. Generator Qwen3.5-2B được fine-tune mới bằng LoRA rank 8 trên 5.636 QA chính thức với cùng định dạng metadata, rồi chạy greedy ở giới hạn 704 output token. Không có reranker/agent/model API trong bản nộp E23.

**Đo lường public.** E08B 0.5209; E18 +metadata suy luận 0.5251; E19 +metadata-aware LoRA 0.5309; E23 +same-article expansion 0.5584 METEOR. Điểm ROUGE-L public của E23 là 0.5727. Mỗi lần nộp đều dùng 1.000 câu public; không có con số private để nêu.

**Độ tin cậy.** Tách train/dev theo normalized question; đáp án train chỉ supervise generation, không làm retrieval labels. Kiểm tra checksum và identity của artifacts, checkpoint từng question ID, resume fail-closed, xác thực đầy đủ câu trả lời trước khi tạo ZIP theo đúng mapping scorer.

**Giới hạn diễn đạt trong CV.** Có thể nói “đạt 0.5584 METEOR public” và “tăng 0.0375 so với mốc E08B public”. Không nói “0.5584 private”, “state-of-the-art”, “tăng chắc chắn trên dữ liệu chưa thấy”, hoặc nhận định mỗi thay đổi là ablation hoàn toàn độc lập.

## English version (optional)

**Vietnamese Legal QA · Fixed RAG | UIT DSC 2026 Task 2**

- Built a deterministic legal QA pipeline combining SQLite FTS5 BM25, AITeam Vietnamese dense embeddings, weighted RRF, and bounded same-article evidence expansion while preserving the original top-12 retrieval seeds.
- Fine-tuned Qwen3.5-2B with rank-8 LoRA on **5,636 official QA pairs**, aligning source metadata between training and inference; implemented per-question checkpoints, SHA-256 artifact validation, and fail-closed resume.
- Achieved **0.5584 public METEOR and 0.5727 public ROUGE-L on 1,000 questions**; public METEOR progressed from **0.5209 → 0.5251 → 0.5309 → 0.5584** through inference metadata, metadata-aware fine-tuning, and bounded same-article context expansion.

No private-test result is claimed.
