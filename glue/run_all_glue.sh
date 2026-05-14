#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
CODE_DIR="$BASE_DIR/glue"
MODEL_PATH=${MODEL_PATH:-"checkpoint/roberta-large"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TASKS=${TASKS:-"rte mrpc cola stsb sst2 qnli"}
LEARNING_RATES=${LEARNING_RATES:-"1e-4 2e-4 3e-4 4e-4 5e-4 6e-4"}

LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
GROUP_SIZE=${GROUP_SIZE:-8}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-32}

mkdir -p "$BASE_DIR/logs/glue"
cd "$CODE_DIR"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

metric_for_task() {
  case "$1" in
    cola) echo "eval_matthews_correlation" ;;
    stsb) echo "eval_pearson" ;;
    *) echo "eval_accuracy" ;;
  esac
}

for TASK_NAME in $TASKS; do
  METRIC=$(metric_for_task "$TASK_NAME")

  for LR in $LEARNING_RATES; do
    RUN_NAME="semilora_${TASK_NAME}_lr${LR}"
    OUTPUT_DIR="checkpoint/$RUN_NAME"

    echo "Running task=$TASK_NAME lr=$LR metric=$METRIC"

    python run_glue.py \
      --model_name_or_path "$MODEL_PATH" \
      --task_name "$TASK_NAME" \
      --do_train \
      --do_eval \
      --do_predict \
      --bf16 true \
      --max_seq_length 256 \
      --per_device_train_batch_size "$BATCH_SIZE" \
      --learning_rate "$LR" \
      --lr_scheduler_type constant_with_warmup \
      --apply_lora true \
      --use_glora true \
      --use_mix true \
      --group_size "$GROUP_SIZE" \
      --lora_r "$LORA_R" \
      --lora_alpha "$LORA_ALPHA" \
      --lora_weight_dropout 0.00 \
      --num_train_epochs "$EPOCHS" \
      --output_dir "$OUTPUT_DIR" \
      --logging_steps 100 \
      --warmup_ratio 0.06 \
      --weight_decay 0.1 \
      --overwrite_output_dir \
      --evaluation_strategy epoch \
      --save_strategy epoch \
      --save_total_limit 5 \
      --load_best_model_at_end \
      --metric_for_best_model "$METRIC" \
      --seed 0 \
      --report_to none \
      --run_name "$RUN_NAME" \
      --max_grad_norm 1.0 \
      2>&1 | tee "$BASE_DIR/logs/glue/${RUN_NAME}.log"
  done
done
