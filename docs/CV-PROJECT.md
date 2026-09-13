# CV Project Entry — Vietnamese Legal QA with Fixed RAG

## Ready to paste into a CV

**Vietnamese Legal QA with Fixed RAG** | UIT Data Science Challenge 2026, Task 2

*Python · PyTorch · Hugging Face Transformers/PEFT · QLoRA · SQLite FTS5 · Dense Retrieval · RRF*

- Built a deterministic legal question-answering pipeline over **8,532 official legal documents**, combining structure-aware Điều/Khoản chunking, SQLite FTS5 BM25, AITeam Vietnamese dense embeddings, and weighted reciprocal rank fusion to retrieve the top 12 evidence chunks.
- Fine-tuned **Qwen3.5-2B with rank-8 QLoRA on 5,636 official QA pairs**; aligned document/article metadata between training and inference, then added bounded same-article evidence expansion without discarding the original retrieval seeds.
- Improved **public METEOR from 0.5209 to 0.5584 (+0.0375 absolute)** on 1,000 questions and achieved **0.5727 public ROUGE-L**; implemented deterministic decoding, per-question checkpoints, SHA-256 artifact validation, and fail-closed resume.

## Public score progression and engineering decisions

| Version | Public METEOR | Main change |
| --- | ---: | --- |
| E08B | 0.5209 | Context-aware rank-8 LoRA with BM25 + dense retrieval, weighted RRF, and top-12 legal evidence. |
| E18 | 0.5251 | Added exact source title, unambiguous document number, and article title to inference contexts. |
| E19 | 0.5309 | Retrained rank-8 LoRA from the base model using the same source-metadata format seen at inference. |
| E23 | **0.5584** | Merged overlapping evidence spans and added bounded verbatim context from the same legal article, preserving all top-12 seeds. |

These are public-leaderboard results, not private-test results. The score progression describes observed submissions; it is not a claim that each change alone guarantees the same gain on unseen data.
