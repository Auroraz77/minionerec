#!/usr/bin/env python3
"""Continue the verified SFT checkpoint with MiniOneRec's ReRe/GRPO objective.

Rewards use row-local training targets, never a shared history-to-target map.
Only validation NDCG@10 selects checkpoints. Test data is not loaded here.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from datasets import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback, set_seed
from trl import GRPOConfig

from industrial_rl_utils import ROOT, SFT, INFO, load_rows, evaluate_model, write_json
from minionerec_trainer import ReReTrainer


def ranking_reward(completions, target_sid, group_size, prefix_weight=0.0, **kwargs):
    """MiniOneRec's exact-hit reward plus normalized rank-weighted penalties."""
    weights = [1/math.log2(index+2) for index in range(group_size)]
    total = sum(weights)
    rewards = []
    for offset in range(0, len(completions), group_size):
        group = [x.strip() for x in completions[offset:offset+group_size]]
        targets = target_sid[offset:offset+group_size]
        if len(set(targets)) != 1:
            raise ValueError("A GRPO group must contain repeats of exactly one training row")
        target = targets[0]
        found = target in group
        for rank, completion in enumerate(group):
            value = float(completion == target)
            if found and completion != target:
                value -= weights[rank]/total
            if prefix_weight:
                # Optional dense reward: matching semantic prefixes of the row's target.
                import re
                predicted_tokens = re.findall(r"<[abc]_\d+>", completion)
                target_tokens = re.findall(r"<[abc]_\d+>", target)
                depth = 0
                for p, t in zip(predicted_tokens, target_tokens):
                    if p != t:
                        break
                    depth += 1
                value += prefix_weight * depth/3
            rewards.append(value)
    return rewards


class IndustrialTrainer(ReReTrainer):
    def __init__(self, *args, validation_rows, **kwargs):
        self.validation_rows = validation_rows
        super().__init__(*args, **kwargs)
        # Match final deterministic ranking; avoid beam sampling with duplicate SIDs.
        self.generation_config.do_sample = False
        self.generation_config.use_cache = True
        self.model.config.use_cache = False

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # Training needs gradients only for the short generated SID, not prompt logits.
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       logits_to_keep=logits_to_keep+1, use_cache=False).logits[:, :-1]
        # Five completion positions fit in GPU memory as one tensor. Batching
        # this operation avoids hundreds of per-row CUDA launches in TRL's
        # memory-saving fallback while preserving full-vocabulary probabilities.
        return logits.log_softmax(-1).gather(
            -1, input_ids[:, -logits_to_keep:].unsqueeze(-1)
        ).squeeze(-1)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.accelerator.num_processes != 1:
            raise ValueError("Run independent single-GPU trials; validation is not sharded")
        metrics = evaluate_model(self.accelerator.unwrap_model(self.model),
                                 self.processing_class, self.validation_rows, batch_size=16)
        logs = {f"{metric_key_prefix}_{key}": value for key, value in metrics.items()}
        logs["global_step"] = self.state.global_step
        with (Path(self.args.output_dir) / "validation_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(logs) + "\n")
        self.log(logs)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, logs)
        return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=150)
    parser.add_argument("--validation-limit", type=int, default=1024)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--prefix-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(4)
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows("train", args.train_limit, args.seed)
    validation_rows = load_rows("valid", args.validation_limit)
    tokenizer = AutoTokenizer.from_pretrained(SFT, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    def reward(completions, target_sid, **kwargs):
        return ranking_reward(completions, target_sid, args.generations,
                              prefix_weight=args.prefix_weight, **kwargs)

    config = GRPOConfig(
        output_dir=str(out), model_init_kwargs={"torch_dtype": "bfloat16", "attn_implementation": "sdpa"},
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.generations,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.generations, max_prompt_length=256, max_completion_length=5,
        temperature=1.0, beta=args.beta, learning_rate=args.learning_rate,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        warmup_ratio=0.03, max_grad_norm=0.3, bf16=True,
        optim="adamw_torch", lr_scheduler_type="cosine", report_to="none",
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="NDCG@10", greater_is_better=True,
        logging_steps=10, seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
        dataloader_num_workers=0, disable_tqdm=True,
    )
    manifest = {**vars(args), "sft_model": str(SFT), "train_rows": len(train_rows),
                "validation_rows": len(validation_rows), "selection_metric": "validation NDCG@10",
                "training_algorithm": "MiniOneRec ReRe/GRPO; exact-hit and rank penalty reward",
                "sft_weights_sha256": hashlib.sha256((SFT/"model.safetensors").read_bytes()).hexdigest(),
                "validation_row_ids": [row["row_id"] for row in validation_rows]}
    write_json(out / "run_config.json", manifest)
    trainer = IndustrialTrainer(
        model=str(SFT), base_model=str(SFT), reward_funcs=reward, args=config,
        beam_search=True, test_during_training=False, info_file=str(INFO),
        train_dataset=Dataset.from_list(train_rows), eval_dataset=Dataset.from_list(validation_rows),
        processing_class=tokenizer, validation_rows=validation_rows,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(str(out / "final_checkpoint"))
    tokenizer.save_pretrained(out / "final_checkpoint")
    metrics = trainer.evaluate()
    write_json(out / "train_summary.json", {"runtime_seconds": time.monotonic()-started,
               "train_metrics": result.metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_validation_ndcg10": trainer.state.best_metric, "final_validation_metrics": metrics})
    print("TRAINING_COMPLETE", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
