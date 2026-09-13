# E23 — E21 parent-expanded policy on public-1000

## Decision

E23 is an operator-selected public trial, not a formal promotion. It runs the
E21 GPU1 `parent_expanded_max704` policy on all 1,000 public questions. E19
METEOR 0.5309 / ROUGE-L 0.5632 remains the rollback.

The E21 comparison used the already-reused E19 dev-200 subset. Its expanded
variant scored METEOR 0.563389 versus E19 0.523317, a paired delta of +0.040073
with 95% bootstrap interval [0.012370, 0.068056]. This is evidence for a public
trial, not an estimate or guarantee of the public/private score.

## Exact change from E19

- Reuse the byte-verified public top-12 retrieval; BM25 and dense retrieval do
  not run again.
- Group seed chunks by document and merge overlapping/touching exact spans.
- Add source metadata once per document rather than once per chunk.
- For every seed, try a bounded exact span from the same contiguous article:
  whole article, both adjacent chunks, previous, then next.
- Keep all 12 original seeds. Expansion may not evict them. Fail closed if the
  compact seeds exceed the 8,192-token input budget.
- Keep the E19 rank-8 metadata-trained QLoRA, prompt, greedy decoding and
  `max_new_tokens=704` unchanged.

No E20 output is loaded. E20 is only an internal config-chain dependency of the
already-pinned E21 policy. No new retrieval, model, training, external data,
API, URL crawl, synthetic QA or public answer is used.

## Kaggle inputs

Upload `kaggle-upload-v57-final-public-e23-parent-expanded-kaggle-fixed.zip` as
one dataset (for example `code-v57`) and import the included notebook. Add:

1. code-v57 only; remove older code packages from this notebook.
2. `full_1000cau_dataset`, the saved public top-12 retrieval artifact.
3. E00-v2 dataset containing `manifest.json`, `documents.jsonl` and
   `chunks.jsonl`.
4. E19 Notebook 1 full output containing `training/adapter-final`.
5. E21 full output containing `report.json`, both worker states and results.

Notebook outputs or datasets created from those exact outputs both work. E19
Notebook 2, E20 and E22 are not inputs.

Select T4 x2 and enable Internet for the pinned model/dependency download. Run
all. The CPU preparation reconstructs exact same-article candidate spans, then
both GPUs run the E21 GPU1 policy on disjoint public IDs: GPU0 gets 500 and GPU1
gets 500. Every answer is checkpointed atomically. A resume is accepted only
when code, config, runtime, adapter, prepared contexts and assigned IDs match.

The last cell creates `/kaggle/working/submission.zip`; the archive contains
only UTF-8 `submission.json`. Submit it to learn the leaderboard score, then
save the full notebook output. Send `report.json` and the leaderboard METEOR /
ROUGE-L. The notebook cannot know the public score locally because all official
public answers are null.
