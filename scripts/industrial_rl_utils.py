"""Shared data and deterministic SID evaluation for the Industrial RL runs."""

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LogitProcessor import ConstrainedLogitsProcessor

CATEGORY = "Industrial_and_Scientific"
DATA = ROOT / "data/Amazon"
SFT = ROOT / "outputs/qwen2.5-0.5b-industrial-sft/final_checkpoint"
INFO = DATA / f"info/{CATEGORY}_5_2016-10-2018-11.txt"
KS = (1, 3, 5, 10, 20, 50)
INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Can you predict the next possible item that the user may expect?

"""


def load_rows(split, limit=0, seed=42):
    path = DATA / f"{split}/{CATEGORY}_5_2016-10-2018-11.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    examples = []
    for index, row in enumerate(rows):
        history = ", ".join(ast.literal_eval(row["history_item_sid"]))
        user_input = (
            "Can you predict the next possible item the user may expect, given "
            f"the following chronological interaction history: {history}"
        )
        prompt = INSTRUCTION + f"### User Input: \n{user_input}\n\n### Response:\n"
        examples.append({"prompt": prompt, "target_sid": row["item_sid"], "row_id": index})
    if limit and limit < len(examples):
        indices = sorted(random.Random(seed).sample(range(len(examples)), limit))
        examples = [examples[index] for index in indices]
    return examples


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def constraint_table(tokenizer):
    catalog = {line.split("\t")[0].strip() for line in INFO.read_text().splitlines()}
    table = {}
    for sid in sorted(catalog):
        tokens = tokenizer.encode(f"### Response:\n{sid}\n", add_special_tokens=False)
        tokens.append(tokenizer.eos_token_id)
        assert tokens[:3] == tokenizer.encode("### Response:\n", add_special_tokens=False)
        for index in range(3, len(tokens)):
            key = tuple(tokens[:index]) if index == 3 else tuple(tokens[3:index])
            table.setdefault(key, set()).add(tokens[index])
    return catalog, {key: sorted(values) for key, values in table.items()}


@torch.no_grad()
def evaluate_model(model, tokenizer, rows, batch_size=16, num_beams=50, output=None):
    """Match evaluate.py's prompts, full-vocabulary probabilities and SID constraints."""
    started = time.monotonic()
    catalog, table = constraint_table(tokenizer)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    hits = {k: 0 for k in KS}
    ndcg = {k: 0.0 for k in KS}
    invalid = duplicates = 0
    predictions = []
    try:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            encoded = tokenizer([row["prompt"] for row in batch], padding=True,
                                add_special_tokens=False, return_tensors="pt").to(device)
            processor = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=lambda _, ids: table.get(tuple(ids), []),
                num_beams=num_beams, base_model="Qwen2.5-0.5B",
                eos_token_id=tokenizer.eos_token_id,
            )
            sequences = model.generate(
                **encoded, num_beams=num_beams, num_return_sequences=num_beams,
                max_new_tokens=8, length_penalty=0.0, do_sample=False,
                top_k=None, top_p=None, use_cache=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([processor]),
            )
            decoded = tokenizer.batch_decode(sequences[:, encoded.input_ids.shape[1]:],
                                             skip_special_tokens=True)
            for index, row in enumerate(batch):
                candidates = [x.strip() for x in decoded[index*num_beams:(index+1)*num_beams]]
                invalid += sum(candidate not in catalog for candidate in candidates)
                duplicates += len(candidates) - len(set(candidates))
                rank = candidates.index(row["target_sid"]) + 1 if row["target_sid"] in candidates else None
                for k in KS:
                    if rank is not None and rank <= k:
                        hits[k] += 1
                        ndcg[k] += 1 / math.log2(rank + 1)
                predictions.append({"row_id": row["row_id"], "output": row["target_sid"],
                                    "predict": candidates})
            if offset % (batch_size * 32) == 0:
                print(f"Evaluation {min(offset+batch_size, len(rows))}/{len(rows)}", flush=True)
    finally:
        model.train(was_training)
    metrics = {f"{name}@{k}": values[k]/len(rows)
               for name, values in [("HR", hits), ("NDCG", ndcg)] for k in KS}
    metrics.update(samples=len(rows), num_beams=num_beams, invalid_sid_count=invalid,
                   duplicate_sid_count=duplicates, runtime_seconds=time.monotonic()-started)
    if invalid or duplicates:
        raise RuntimeError(f"Invalid or repeated evaluation SIDs: {invalid}, {duplicates}")
    if output:
        output = Path(output)
        write_json(output / "metrics.json", metrics)
        write_json(output / "predictions.json", predictions)
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(SFT))
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    metrics = evaluate_model(model, tokenizer, load_rows(args.split, args.limit),
                             batch_size=args.batch_size, output=args.output_dir)
    manifest = {**vars(args), "model_sha256": hashlib.sha256(
        (Path(args.model)/"model.safetensors").read_bytes()).hexdigest()}
    write_json(Path(args.output_dir) / "evaluation_config.json", manifest)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
