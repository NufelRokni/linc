#!/usr/bin/env python3
"""
Recompute postprocessed generations (PRC) from a saved RAW JSON using the
task's postprocessing logic, without loading the model.

Usage:
  python utils/reprocess_from_raw.py \
    --task folio-neurosymbolic-1shot \
    --raw outputs/Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_raw.json \
    --out outputs/Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_prc_reproc.json

Notes:
  - RAW entries should be the "prompt + completion" strings stored by this repo.
  - This script uses the dataset to reconstruct prompts and stopwords, matching
    the evaluation pipeline.
"""
import argparse
import json
import os
from typing import List

from eval import tasks as tasks_mod


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Task name, e.g., folio-neurosymbolic-1shot")
    ap.add_argument("--raw", required=True, help="Path to RAW generations JSON (list[list[str]])")
    ap.add_argument("--out", required=True, help="Path to write PRC JSON (list[list[str]])")
    ap.add_argument("--limit", type=int, default=None, help="Optional: limit number of tasks")
    ap.add_argument("--prefix", default="", help="Optional prefix used during generation (if any)")
    return ap.parse_args()


def main():
    args = parse_args()

    task = tasks_mod.get_task(args.task)

    with open(args.raw, "r") as fp:
        gens_raw: List[List[str]] = json.load(fp)

    ds = task.get_dataset()
    n_tasks = min(len(ds), len(gens_raw))
    if args.limit is not None:
        n_tasks = min(n_tasks, args.limit)

    # Ensure lengths match
    gens_raw = gens_raw[:n_tasks]

    print(f"Re-postprocessing {n_tasks} tasks for {args.task} from RAW JSON: {os.path.basename(args.raw)}")

    gens_prc: List[List[str]] = []
    for i, cand_list in enumerate(gens_raw):
        # If a global prefix was used during generation, strip it only (full prompt remains)
        def strip_pref(s: str) -> str:
            return s[len(args.prefix):] if args.prefix and s.startswith(args.prefix) else s

        cand_list = [strip_pref(c) for c in cand_list]
        prc_row = [task.postprocess_generation(c, i) for c in cand_list]
        gens_prc.append(prc_row)

    with open(args.out, "w") as fp:
        json.dump(gens_prc, fp)
    print(f"Wrote postprocessed generations to: {args.out}")


if __name__ == "__main__":
    main()
