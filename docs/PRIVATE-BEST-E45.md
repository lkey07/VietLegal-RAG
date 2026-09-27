# Private-best E45 — Inference-Aligned Parent LoRA

## Result

The stored Codabench scoring artifact reports:

- METEOR: `0.597327402`
- ROUGE-L: `0.586192885`
- Questions: `1,918`
- Submission ZIP SHA-256:
  `d697c48e4ce48410ad4695291c98891f9f872a88cf0f6d1dfc576d8dbf86496b`
- `submission.json` SHA-256:
  `5232019db849feac81182aa7f870941886a9812b689aa8015dc56c3461d6bbdf`

Private reference answers were not available to retrieval, context selection,
generation, cleanup or merging.

## Retrieval and context construction

1. Retrieve 40 BM25 and 40 dense candidates from the official corpus.
2. Dense retrieval uses `AITeamVN/Vietnamese_Embedding`, normalized vectors
   and exact inner-product search.
3. Fuse rankings with RRF (`k=60`, BM25/dense weights `0.5/0.5`).
4. Keep the fused top 20 and use the first 12 as evidence seeds. No reranker or
   legal-priority top12-v2 selector is used.
5. For each seed, try `whole_article`, `both_neighbors`, `previous_neighbor`
   and `next_neighbor` expansion in that order.
6. Bound each parent to 1,200 tokenizer tokens and 6,000 characters; preserve
   all compact seeds and merge overlapping/touching spans from the same legal
   document/article.
7. Render one source header per document followed by the document number,
   article label/title and verbatim source content. Maximum input length is
   8,192 tokens.

## Fine-tuning alignment

E45 trains a fresh adapter on the same parent-expanded evidence representation
used during inference. The old E38 adapter is a frozen control identity and is
not stacked into the E45 candidate.

| Setting | Value |
| --- | --- |
| Base model | `AITeamVN/Vi-Qwen2-3B-RAG` |
| Training records | 5,636 official QA pairs |
| LoRA | rank 8, alpha 16, dropout 0.05 |
| Target modules | q/k/v/o and gate/up/down projections |
| Training | 1 epoch, 705 optimizer steps |
| Optimizer schedule | LR `1e-4`, cosine, warmup 3%, weight decay 0 |
| Effective batch | 8 across two workers |
| Quantization | NF4 double quantization, FP16 compute |
| Sequence cap | 8,192 tokens |
| Loss | assistant answer and exactly one EOS only |

The official answer is generation supervision only. It is never interpreted
as a retrieval label and does not affect evidence selection.

## Private generation

- Full-FP16 greedy decoding: `do_sample=False`, `num_beams=1`,
  `repetition_penalty=1.0`, `no_repeat_ngram_size=0`.
- Prompts up to 6,000 tokens are alternated across two independent T4 model
  replicas. Longer prompts use one model sharded across both T4 GPUs.
- First pass uses `max_new_tokens=1024`.
- Only rows classified as `finish_reason=length` restart from the exact
  original prompt with `max_new_tokens=1536`; this is fresh generation, not
  continuation.
- Unified E43+E44 cleanup runs to a fixed point for at most 16 passes and may
  remove only a proven repeated suffix. It cannot modify interior bytes.

The submitted run had 326 length-restart candidates. A deadline merge retained
183 completed second-pass answers and used the already-checkpointed 1,024-token
first-pass answer for the remaining 143. All selected answers received the
same suffix-only cleanup.

## Released source

[`../private-e45-source/`](../private-e45-source/) is the code/config/notebook/
test snapshot used to audit this lineage. Large or restricted artifacts are
intentionally absent: official data, retrieval indices, candidate adapter,
private questions, checkpoints, predictions and submission payload.
