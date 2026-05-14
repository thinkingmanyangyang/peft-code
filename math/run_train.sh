#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
CODE_DIR="$BASE_DIR/math"
RUN_ID=${RUN_ID:-"semilora_llama2_7b_metamath_rank128_$(date -u +%Y%m%dT%H%M%SZ)"}

MODEL_PATH=${MODEL_PATH:-"/path/to/Llama-2-7b-hf"}
DATA_PATH=${DATA_PATH:-"meta-math/MetaMathQA"}
OUTPUT_PATH=${OUTPUT_PATH:-"$BASE_DIR/runs/$RUN_ID"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$OUTPUT_PATH" "$BASE_DIR/logs" "$BASE_DIR/hf_cache/datasets" "$BASE_DIR/hf_cache/hub"

cd "$CODE_DIR"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$BASE_DIR/hf_cache"
export HF_DATASETS_CACHE="$BASE_DIR/hf_cache/datasets"
export TRANSFORMERS_CACHE="$BASE_DIR/hf_cache/hub"

python -u run_instruct_tuning.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --dataset_split 'train[:100000]' \
  --dataset_field instruction output \
  --output_dir "$OUTPUT_PATH" \
  --use_lora true \
  --target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
  --lora_rank 128 \
  --lora_alpha 128 \
  --lora_dropout 0.0 \
  --num_train_epochs 1 \
  --model_max_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --logging_steps 1 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --gradient_checkpointing true \
  --report_to none \
  --no_remove_unused_columns \
  2>&1 | tee "$BASE_DIR/logs/$RUN_ID.log"
