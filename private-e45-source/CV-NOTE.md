# E45 source snapshot included for technical review

This directory is an exact source/config/notebook/test snapshot for the
inference-aligned parent-LoRA lineage described in the repository root README.
Repository metadata, Python caches and runtime artifacts were excluded during
export.

Source archive SHA-256 before those exclusions:
`7796632cc9dd1a6baeedd42c6b2f78804ab2dcca49003bf0fc2e14351543ee39`.
The frozen E45 config SHA-256 is
`210c2450db00525905247aa6ec47479a4ec9ce63406e5688010cf2b84f7e665c`.

Not included: official corpus/QA files, BM25 or dense indexes, model/adapter
weights, private questions, generated answers, checkpoints, submissions and
scoring archives. The code remains fail-closed and expects separately
authorized artifacts whose hashes are pinned by its configuration.

Start with:

- `configs/e45-inference-aligned-parent-lora-v1.json`
- `src/uit_dsc_fixed_rag/e45_parent_training.py`
- `src/uit_dsc_fixed_rag/e45_holdout_contexts.py`
- `src/uit_dsc_fixed_rag/e45_paired_generation.py`
- `src/e45_private_resumable_generation.py`
- `scripts/run_e45_train_kaggle.py`
- `scripts/run_e45_private_shard.py`
- `scripts/merge_e45_emergency_checkpoints.py`
- `docs/architecture.md`
- `docs/submitted-run.md`
