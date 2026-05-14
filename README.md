# SeMi-LoRA

Code for **SeMi-LoRA: Enhancing Low-Rank Adaptation via Separation and Mixing**.

SeMi-LoRA is a parameter-efficient fine-tuning method that extends LoRA by separating features into channels and mixing them in the latent space. It keeps the update linear, so the learned adapter can be merged into the base model weights for inference.

## Repository Structure

```text
commonsense/   Commonsense reasoning experiments and evaluation scripts.
glue/          GLUE experiments based on RoBERTa-large.
math/          Math reasoning fine-tuning and evaluation scripts.
```

Main entry files:

```text
commonsense/train_hira.py
commonsense/run_train.sh
commonsense/run_eval_decoding.sh
glue/run_all_glue.sh
glue/run_glue.py
math/run_train.sh
math/run_instruct_tuning.py
math/instruction_tuning_eval/
docs/appendix_notes.md
```

## Environment

For commonsense reasoning, an environment file is provided:

```bash
cd commonsense
conda env create -f env.yml
conda activate hira
```

If you use a different environment name, install the required packages according to `env.yml` and the imported packages in each script.

## Commonsense Reasoning

Edit the model path and GPU setting in `commonsense/run_train.sh`, then run:

```bash
cd commonsense
bash run_train.sh
```

Evaluate a trained checkpoint with:

```bash
bash run_eval_decoding.sh /path/to/checkpoint
```

The evaluation script runs BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA.

## GLUE

Run all configured GLUE tasks with:

```bash
cd glue
MODEL_PATH=/path/to/roberta-large bash run_all_glue.sh
```

Useful environment variables:

```bash
TASKS="rte mrpc cola stsb sst2 qnli"
LEARNING_RATES="1e-4 2e-4 3e-4 4e-4 5e-4 6e-4"
CUDA_VISIBLE_DEVICES=0
```

The script calls `run_glue.py` with `--apply_lora`, `--use_glora`, `--use_mix`, `--group_size`, `--lora_r`, and `--lora_alpha`.

## Math Reasoning

Run MetaMathQA instruction tuning with:

```bash
cd math
MODEL_PATH=/path/to/Llama-2-7b-hf bash run_train.sh
```

By default, the script trains on `meta-math/MetaMathQA` with `train[:100000]`, rank `128`, and one epoch.

Evaluation utilities are under:

```text
math/instruction_tuning_eval/
```

## Method Notes

SeMi-LoRA writes the update matrix as:

```text
Delta W = B^m (C^m + I) (R^m + I) A^m
```

where `A^m` and `B^m` are block-diagonal compression/decompression matrices, `R^m` performs rank mixing, and `C^m` performs channel mixing.

The rank capacity is bounded by:

```text
rank(Delta W) <= min(d_in, d_out, c r)
```

Thus increasing the channel number `c` expands the update rank capacity while keeping the base rank `r` fixed.

For a single adapted linear layer, SeMi-LoRA adds the following trainable parameters over the LoRA-style compression/decompression path:

```text
c r^2 + r c^2
```

The corresponding extra FLOPs are:

```text
2 c r^2 + 2 r c^2
```

More supplementary notes are available in `docs/appendix_notes.md`.

## Citation

```bibtex
@inproceedings{yang2026semilora,
  title={SeMi-LoRA: Enhancing Low-Rank Adaptation via Separation and Mixing},
  author={Yang, Zhenfei and Yu, Beiming and Lin, Peiqin and Liu, Yongkang and Xiong, Deyi},
  booktitle={Proceedings of IJCAI-ECAI},
  year={2026}
}
```
