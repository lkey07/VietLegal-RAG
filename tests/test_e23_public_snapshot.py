"""Data-free checks for the published E23 source snapshot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uit_dsc_fixed_rag import final_public_e23 as e23  # noqa: E402


def test_pinned_public_contract() -> None:
    config = e23.load_config(ROOT, ROOT / "configs/final-public-e23-parent-expanded-max704-v1.json")
    assert config.raw["context_policy"]["variant"] == "parent_expanded_max704"
    assert config.raw["context_policy"]["seed_contexts"] == 12
    assert config.public.section("retrieval")["reranker"] is None
    assert config.public.section("inference")["max_new_tokens"] == 704
    assert e23.code_sha(ROOT) == "7242a88491f3fcc01ecb474a80d7482d824bb8f71a369a188e8030f0ffdcada6"


def test_dual_gpu_partitions_are_complete_and_disjoint() -> None:
    left, right = e23.assigned(0), e23.assigned(1)
    assert len(left) == len(right) == 500
    assert set(left).isdisjoint(right)
    assert sorted(left + right) == list(range(1000))


def test_published_notebook_has_no_saved_outputs_and_compiles() -> None:
    path = ROOT / "notebooks/FINAL-public1000-e23-parent-expanded-max704-dual-gpu-kaggle.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"E23-cell-{index}", "exec")
