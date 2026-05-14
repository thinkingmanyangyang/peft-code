export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_PATH=${MODEL_PATH:-/path/to/Llama-2-7b-hf}

# Precision options: auto (default, GPU auto-detect), bf16, fp16, or fp32
# --precision=auto   # Default: auto-detect based on GPU capability
# --precision=bf16   # Force BF16 (recommended for A100/H100)
# --precision=fp16   # Force FP16 (compatible with more GPUs)
# --precision=fp32   # Full precision (slower but most accurate)

python train_hira.py \
--peft_type=hira \
--model="$MODEL_PATH" \
--r_ab=32 \
--enable_grad_ckpt --epoch=3 --lr=1e-3 --batch=16 \
--dataset=common_170k --seed=36 \
--warmup=100 --eval_strategy=steps --eval_steps=80 \
--output_folder=results_hira --target_modules=q_proj,k_proj,v_proj,up_proj,down_proj \
--precision=auto
