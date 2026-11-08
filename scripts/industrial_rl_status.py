#!/usr/bin/env python3
"""Read live RL progress without loading models or interrupting training."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ["rl_rank_lr1e5", "rl_dense_lr3e6"]:
        directory = ROOT/"outputs"/name
        records = []
        for line in (directory/"console.log").read_text().splitlines():
            if line.startswith("{'loss'"):
                records.append(ast.literal_eval(line))
        validations = [json.loads(line) for line in (directory/"validation_metrics.jsonl").read_text().splitlines()]
        best = max(validations, key=lambda row: row["eval_NDCG@10"])
        last = records[-1]
        workers = []
        for proc in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                args = proc.read_bytes().split(b"\0")
                if args and b"python" in args[0] and any(name.encode() == x.split(b"/")[-1] for x in args):
                    workers.append(int(proc.parent.name))
            except (OSError, ProcessLookupError):
                pass
        print(json.dumps({"trial": name, "worker_pids": workers,
              "step_approx": round(last["epoch"]*4533/4), "epoch": round(last["epoch"], 3),
              "best_step": best["global_step"], "best_valid_ndcg10": round(best["eval_NDCG@10"], 6),
              "last_valid_step": validations[-1]["global_step"],
              "last_valid_ndcg10": round(validations[-1]["eval_NDCG@10"], 6),
              "training_complete": (directory/"train_summary.json").exists()}))
    output = ROOT/"outputs/rl_comparison"
    for filename in ["selection.json", "comparison.json"]:
        print(filename, (output/filename).exists())


if __name__ == "__main__":
    main()
