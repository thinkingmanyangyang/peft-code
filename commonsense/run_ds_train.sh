# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Precision options: auto (default, GPU auto-detect), bf16, fp16, or fp32
# --precision=auto   # Default: auto-detect based on GPU capability
# --precision=bf16   # Force BF16 (recommended for A100/H100)
# --precision=fp16   # Force FP16 (compatible with more GPUs)
# --precision=fp32   # Full precision (slower but most accurate)

deepspeed --master_port=29500 --num_gpus=4 --num_nodes=1 train_hira.py \
--peft_type=hira \
--model=Llama-2-7b-hf \
--r_ab=32 \
--enable_grad_ckpt --epoch=3 --lr=1e-3 --batch=16 \
--dataset=common_170k --ds_config=ds_configs/ds_config_auto_z3.json --seed=36 \
--warmup=100 --eval_strategy=steps --eval_steps=80 \
--output_folder=results_hira --target_modules=q_proj,k_proj,v_proj,up_proj,down_proj \
--precision=auto