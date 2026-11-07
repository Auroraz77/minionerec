#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${SFT_PYTHON_BIN:-/root/miniconda3/envs/MiniOneRec/bin/python}"
MODEL_DIR="$PROJECT_DIR/models/Qwen2.5-0.5B"
OUTPUT_DIR="$PROJECT_DIR/outputs/qwen2.5_0.5b_industrial_sft"
mkdir -p "$OUTPUT_DIR"

if [[ -f "$OUTPUT_DIR/final_checkpoint/model.safetensors" ]]; then
    echo "Training output already exists: $OUTPUT_DIR/final_checkpoint" >&2
    exit 1
fi
if compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null && [[ -z "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    echo "Checkpoints exist. Set RESUME_FROM_CHECKPOINT to resume this run." >&2
    exit 1
fi

trap 'run_exit_code=$?; printf "%s\n" "$run_exit_code" > "$OUTPUT_DIR/exit_code"' EXIT
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export PYTHONUNBUFFERED=1

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    resume_args+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"$PYTHON_BIN" -u sft.py \
    --base_model "$MODEL_DIR" \
    --train_file "$PROJECT_DIR/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
    --eval_file "$PROJECT_DIR/data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 128 \
    --micro_batch_size 4 \
    --cutoff_len 512 \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --category Industrial_and_Scientific \
    --sid_index_path "$PROJECT_DIR/data/Amazon/index/Industrial_and_Scientific.index.json" \
    --item_meta_path "$PROJECT_DIR/data/Amazon/index/Industrial_and_Scientific.item.json" \
    --train_from_scratch False \
    --freeze_LLM False \
    --seed 42 \
    --wandb_run_name qwen2.5_0.5b_industrial_sft \
    "${resume_args[@]}"
