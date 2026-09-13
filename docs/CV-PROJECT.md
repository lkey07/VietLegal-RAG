# CV Project Entry — Vietnamese Legal QA with Fixed RAG

## Ready to paste into a CV

**Vietnamese Legal QA with Fixed RAG** | UIT Data Science Challenge 2026, Task 2

*Python · PyTorch · Hugging Face Transformers/PEFT · QLoRA · SQLite FTS5 BM25 · AITeamVN/Vietnamese_Embedding · Qwen3.5-2B · Kaggle T4 × 2*

- Processed **8,532 official legal documents** with conservative cleaning and Điều/Khoản-aware chunking; combined **SQLite FTS5 BM25** with dense retrieval using **AITeamVN/Vietnamese_Embedding**, fused candidates through **0.5/0.5 weighted reciprocal rank fusion**, and selected the top 12 evidence chunks.
- Enriched retrieved chunks with exact source titles, unambiguous document numbers, and article titles; merged overlapping source spans and added **bounded, verbatim same-article context** while preserving all original top-12 evidence seeds within an **8,192-token input budget**.
- Fine-tuned **Qwen/Qwen3.5-2B from its base checkpoint** using **PEFT rank-8/alpha-16 QLoRA** on **5,636 official QA pairs** for one epoch; used greedy decoding with **up to 704 generated tokens**, dual Kaggle T4 GPUs, per-question checkpoints, SHA-256 artifact checks, and fail-closed resume.
- Raised **public METEOR from 0.5209 to 0.5584 (+0.0375 absolute)** on **1,000 questions**, with **0.5727 public ROUGE-L**. The public gains came from source metadata at inference (E18), metadata-aligned fine-tuning (E19), and same-article evidence expansion (E23).

## Public score progression and engineering decisions

| Version | Public METEOR | Main change |
| --- | ---: | --- |
| E08B | 0.5209 | SQLite FTS5 BM25 + AITeam dense retrieval, 0.5/0.5 weighted RRF, top-12 chunks, and context-aware Qwen3.5-2B LoRA. |
| E18 | 0.5251 | Kept retrieval/model fixed; prepended exact E00 source title, unambiguous document number, and article title at inference. |
| E19 | 0.5309 | Trained a fresh rank-8 QLoRA adapter on 5,636 official QA pairs with the same metadata format used at inference. |
| E23 | **0.5584** | Kept the E19 model and greedy max-704 decoding; merged overlapping spans and added bounded verbatim context from the same legal article without evicting top-12 seeds. |

These are public-leaderboard results, not private-test results. The score progression describes observed submissions; it is not a claim that each change alone guarantees the same gain on unseen data.
