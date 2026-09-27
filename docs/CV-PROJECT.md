# CV Project Entry — VietLegal-RAG

## Ready to paste into a CV

**VietLegal-RAG — Inference-Aligned Hierarchical RAG for Vietnamese Legal QA**

*UIT Data Science Challenge 2026, Task 2 · Top 10 Public Leaderboard*

*Python · PyTorch · Hugging Face Transformers/PEFT · QLoRA · SQLite FTS5
BM25 · FAISS · AITeamVN/Vietnamese_Embedding · Vi-Qwen2-3B-RAG · Kaggle T4*

- Designed a deterministic Vietnamese legal QA system combining SQLite FTS5
  BM25 and dense retrieval, 0.5/0.5 weighted reciprocal rank fusion, top-12
  evidence selection, exact legal-source metadata, and bounded same-article
  parent expansion under an 8,192-token input budget.
- Developed an inference-aligned rank-8 QLoRA adapter for
  `AITeamVN/Vi-Qwen2-3B-RAG` on 5,636 authorized official QA pairs, using the
  same parent-expanded evidence renderer for training and inference while
  restricting supervised loss to the answer and EOS tokens.
- Implemented resumable multi-GPU inference with per-question atomic
  checkpoints, SHA-256 model/config/artifact identities, greedy 1,024-token
  generation, length-only fresh 1,536-token restarts, and conservative
  suffix-only repetition cleanup.
- Reached **Top 10 on the public leaderboard** with a team-reported best public
  METEOR of **0.5817**; achieved private METEOR **0.597327402** and private
  ROUGE-L **0.586192885** over 1,918 questions.

## One-line Vietnamese version

**VietLegal-RAG** — xây dựng hệ thống hỏi–đáp pháp luật tiếng Việt kết hợp
BM25+dense RRF, parent context theo Điều luật và inference-aligned Vi-Qwen
QLoRA; đạt **Top 10 public leaderboard**, METEOR public **0,5817** và METEOR
private **0,597327402**.

## Result scope

Public and private metrics were produced on different question sets and should
not be treated as a paired comparison. The public placement is team-reported;
the private metric is bound to the stored scoring artifact and submission hash
documented in [`PRIVATE-BEST-E45.md`](PRIVATE-BEST-E45.md).
