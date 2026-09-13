#!/usr/bin/env python3
"""Run the operator-selected E21 parent-expanded policy on public-1000."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag.e02_answer import _atomic_json  # noqa: E402
from uit_dsc_fixed_rag.final_public import FinalPublicError  # noqa: E402
from uit_dsc_fixed_rag.final_public_e23 import (  # noqa: E402
    EXPERIMENT,
    check_preflight,
    finalize,
    load_config,
    prepare,
    run_worker,
    validate_preflight,
)

APPROVALS = (
    "inventory-reviewed", "experiment-plan-reviewed", "weights-download-authorized",
    "operator-selection-authorized", "public-inference-authorized",
    "submission-creation-authorized", "e23-public-trial-authorized",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "prepare-contexts", "generate-worker", "finalize"))
    for key in ("project-root", "retrieval-artifact", "e00-artifact", "e19-artifact",
                "selection-artifact", "public", "output", "config"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--worker-rank", type=int)
    parser.add_argument("--device")
    for key in APPROVALS:
        parser.add_argument("--" + key, action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.project_root, args.config)
    approvals = {key.replace("-", "_"): getattr(args, key.replace("-", "_")) for key in APPROVALS}
    if not all(approvals.values()):
        raise FinalPublicError("Read the E21/E23 plan and provide every explicit approval.")
    args.output.mkdir(parents=True, exist_ok=True)
    common = dict(root=args.project_root, output=args.output, config=config)

    if args.phase == "preflight":
        import torch
        if torch.cuda.device_count() != 2:
            raise FinalPublicError(f"Choose GPU T4 x2; found {torch.cuda.device_count()} GPU(s).")
        evidence = validate_preflight(
            root=args.project_root, retrieval=args.retrieval_artifact,
            e00=args.e00_artifact, e19=args.e19_artifact,
            selection=args.selection_artifact, public=args.public, config=config,
        )
        payload = {
            "schema_version": "1.0", "experiment_id": EXPERIMENT,
            "approvals": approvals, "evidence": evidence,
            "cuda_devices": [{"index": i, "name": torch.cuda.get_device_name(i)} for i in range(2)],
        }
        _atomic_json(args.output / "preflight.json", payload)
        result = payload
    else:
        checked = check_preflight(**common)
        saved = json.loads((args.output / "preflight.json").read_text(encoding="utf-8"))
        if saved.get("approvals") != approvals or checked.get("adapter_sha256") is None:
            raise FinalPublicError("E23 approvals/preflight changed.")
        if args.phase == "prepare-contexts":
            result = prepare(retrieval=args.retrieval_artifact, e00=args.e00_artifact,
                             public=args.public, output=args.output, config=config)
        elif args.phase == "generate-worker":
            if args.worker_rank is None or args.device is None:
                raise FinalPublicError("generate-worker requires --worker-rank and --device.")
            result = run_worker(**common, e19=args.e19_artifact, public=args.public,
                                rank=args.worker_rank, device=args.device)
            _atomic_json(args.output / "generation" / f"worker-{args.worker_rank}-summary.json", result)
        else:
            result = finalize(**common, public=args.public)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
