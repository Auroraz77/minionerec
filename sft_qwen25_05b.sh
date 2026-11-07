#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CATEGORY="Industrial_and_Scientific"
MODEL="${MODEL:-./models/modelscope_cache/Qwen/Qwen2___5-0___5B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen2.5-0.5b-industrial-sft}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_FILE="./data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
EVAL_FILE="./data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv"
SID_INDEX="./data/Amazon/index/${CATEGORY}.index.json"
ITEM_META="./data/Amazon/index/${CATEGORY}.item.json"

for file in "$MODEL/config.json" "$TRAIN_FILE" "$EVAL_FILE" "$SID_INDEX" "$ITEM_META"; do
    if [[ ! -f "$file" ]]; then
        echo "Required file is missing: $file" >&2
        exit 1
    fi
done

export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" sft.py \
    --base_model "$MODEL" \
    --batch_size 256 \
    --micro_batch_size 8 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "" \
    --wandb_run_name "qwen2.5-0.5b-industrial-sft" \
    --category "$CATEGORY" \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path "$SID_INDEX" \
    --item_meta_path "$ITEM_META" \
    --freeze_LLM False
