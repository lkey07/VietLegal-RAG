# E21 — Compact seeds and bounded same-article parents

## Purpose and limits

Current observed public leader: E19 metadata-trained rank8, max704, METEOR
0.5309 / ROUGE-L 0.5632. E21 tests a possible missing-context mechanism; it does
not establish that retrieval is the dominant error source or promise 0.6.
The local workspace lacks the complete E00-v2 / E19 dev output artifacts, so
real-data GPU inference and error categorization must occur on Kaggle.

Existing E00 chunking is already article/clause-aware; search_text already
contains source metadata. E21 does not claim to introduce either for the first
time. It adds a deterministic parent-context assembly layer over saved seeds.

## Comparison

| Variant | Generation | Changes |
|---|---|---|
| E19 control | Exact saved 200 answers | None |
| Compact seeds | GPU0, all same 200 | Merge overlapping/touching exact spans; group by document; source header once; retain article labels |
| Parent expanded | GPU1, all same 200 | Same compact assembly plus bounded text from the seed's original article |

Compact versus E19 includes metadata formatting, grouping/order and overlap
removal, not only deduplication. Expanded versus compact isolates the expansion
policy. Neither candidate uses E20's top20 source resolver. E20 config is loaded
only as an existing pinned E19 source/control validation contract; no E20 output
or E20 generation is required.

Same E19 adapter, prompt instruction, greedy generation and max_new_tokens=704.
No new models, training, embedding database, retrieval search, API, URL crawl,
synthetic QA or reference-derived retrieval labels. Existing model parameter
budget validation remains active (<4 billion including adapter).

## Assembly policy

1. Verify E08A and E00 hashes. Take the original ranked top12 seeds.
2. Identify the containing article from original cleaned-document offsets using
   the existing E00 parser. Also require contiguous chunk indices and matching
   article metadata. Repeated article numbers are separated by actual boundaries;
   unknown article seeds remain unexpanded.
3. Merge overlapping/touching spans only within that document/article run.
   Every body remains an exact substring; conflicting overlap fails closed.
4. Pack all original seeds first. If they do not fit 8192 generator prompt tokens,
   stop rather than silently drop evidence. Documents follow their first seed
   rank; spans within each document follow seed rank.
5. For each seed in rank order, try the bounded whole article-run span, then the
   immediate preceding+following chunks, preceding only, following only. A parent
   must be at most 6000 characters and 1200 generator tokens, and the complete
   rendered prompt must remain at most 8192 tokens. Keep original seeds. Reject
   expansion with no increase in unique body coverage. No open-ended expansion.

These caps are a predeclared experiment choice, not an optimized policy.
More context can add irrelevant evidence or reduce attention to the right clause.

## Data and validation

Reuse E19 dev order indices 400:600 (dev521 indices 200:400), exactly 200 IDs.
This subset has already been used for tuning/model selection: it is NOT a fresh
test or a guarantee of private-test performance. Full dev metadata is loaded for
hash/group checks; final121 are not selected/scored. Public and holdout are not
loaded. Normalized question groups remain disjoint from the 5636 training records.

Reference answers are used only after generation for official scoring and human
review. No answer text is passed to context assembly or model input. The human
review output is not a retrieval-label training set.

Preflight requires the existing model inventory, approved plan, split/scorer
checks and explicit runtime approval flags. CPU tests use tiny software fixtures,
not new task training data. No model has been downloaded or trained locally.

## Kaggle usage

Upload `kaggle-upload-v54-e21-parent-context-kaggle-fixed.zip` as code-v54.
Import `notebooks/E21-parent-context-dev200-dual-gpu-kaggle.ipynb`.
Add exactly these inputs (notebook outputs or equivalent datasets):

- code-v54 (remove other code versions from this notebook).
- E00-v2: original documents.jsonl, chunks.jsonl, manifest.json.
- E08A: frozen train5636/dev521 retrieval, including reports.
- E19 Notebook 1: training/adapter-final and evidence.
- E19 Notebook 2: exact E19 dev results and report.

Select T4 x2 and Internet for pinned model/dependencies. The notebook uses existing
E19 dependency ranges and records runtime versions; different versions are rejected
on resume, but cross-runtime equivalence to old E19 control is not guaranteed.
Run all: preflight, CPU prepare, two workers, final scoring. Each GPU handles one
variant of 200, not 100; total 400 newly generated answers. Existing E19 control
is never regenerated. Estimated runtime should be based on observed new worker
throughput, not the older 100/100 runs.

Checkpoint each ID atomically. An interrupted run resumes only with matching
code/config/model/prepared contexts/runtime. Restore saved output only into an
absent OUTPUT; never merge two output directories. For rerun in the same session
leave RESUME_FROM=None. Always preserve/download full output before ending session.

## Outputs and decision

Send report.json, review_low50.jsonl and both evaluation/*/results.jsonl files.
The report contains three metrics tables, paired METEOR differences/bootstrap
intervals, expansion/input-size diagnostics and answer repetition/finish rates.
Review_low50 selects the 50 lowest E19 METEOR answers and exports references,
all answers, source spans and empty manual error categories. They have not been
automatically classified as retrieval failures.

No submission archive or automatic promotion is created. Keep E19 public 0.5309
as rollback. If expanded beats compact consistently and review confirms useful
evidence recovery, consider a separate metadata+parent-context training experiment
on the existing 5636 answers. If not, do not spend a training run on this format.
Freeze the chosen design before an independently reserved validation check;
inspect stability across document/question groups, not public score alone.
