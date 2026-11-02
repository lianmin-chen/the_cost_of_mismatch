# GSM8K: BOLT and Matched ZO-BOLT

This directory reproduces the first-order BOLT baseline and **matched ZO-BOLT** on GSM8K. Both train Qwen2.5-1.5B LoRA adapters on the *same fixed, pre-generated responses*; the difference is how they optimize the weighted loss. Checkpoints are evaluated with the same boxed-answer scorer and the same prompt format used in data generation.

## Reproduction workflow

Run the commands below from `the_cost_of_mismatch/ZO-BOLT/` (run `cd ZO-BOLT` first if you are at the top-level repository root). Use Python 3.12, compatible CUDA dependencies, and one local Qwen2.5-1.5B snapshot. See the [ZO-BOLT README](../README.md#environment) for environment setup. Replace `/path/to/Qwen2.5-1.5B` with your model path in **every** command.

1. **Generate the shared training data.** This loads GSM8K train, builds the fixed boxed few-shot prompt, generates eight answers per question, filters all-wrong questions, and truncates the retained answers.

   ```bash
   CUDA_VISIBLE_DEVICES=0 python gsm8k/data.py \
     --model /path/to/Qwen2.5-1.5B \
     --output-dir outputs/gsm8k-data
   ```

   The three files are `inference_results.json` (raw), `inference_results_filtered.json`, and `inference_results_filtered_truncated.json` (the shared training input).

2. **Train the two methods on that same input.** BOLT uses the included DeepSpeed ZeRO-3 configuration; single-GPU ZO-BOLT runs directly with Python.

   ```bash
   CUDA_VISIBLE_DEVICES=0 deepspeed --master_port=29500 gsm8k/train.py \
     --method bolt --model /path/to/Qwen2.5-1.5B \
     --data-file outputs/gsm8k-data/inference_results_filtered_truncated.json \
     --output-dir outputs/bolt --no-wandb

   CUDA_VISIBLE_DEVICES=0 python gsm8k/train.py \
     --method zo-bolt --model /path/to/Qwen2.5-1.5B \
     --data-file outputs/gsm8k-data/inference_results_filtered_truncated.json \
     --output-dir outputs/zo-bolt --no-wandb
   ```

   Each output directory contains `run_config.json` and training checkpoints. Omit `--no-wandb` to enable W&B logging. For multi-GPU ZO-BOLT, replace `python` with `torchrun --nproc_per_node=<GPU_COUNT> --master_port=29500` and set `CUDA_VISIBLE_DEVICES` accordingly.

3. **Evaluate checkpoints.** The bundled `gsm8k_eval` set is scored with one greedy completion per question. The evaluator temporarily merges each LoRA checkpoint with the base model.

   ```bash
   CUDA_VISIBLE_DEVICES=0 python gsm8k/eval.py \
     --checkpoint-dir outputs/bolt --output-dir outputs/bolt-eval \
     --model /path/to/Qwen2.5-1.5B

   CUDA_VISIBLE_DEVICES=0 python gsm8k/eval.py \
     --checkpoint-dir outputs/zo-bolt --output-dir outputs/zo-bolt-eval \
     --model /path/to/Qwen2.5-1.5B
   ```

   Each evaluation directory contains `gsm8k_eval.csv` (scores by checkpoint step) and `<checkpoint>/summary.json` plus `samples.json`. Use `--checkpoint checkpoint-<STEP>` to evaluate only one checkpoint.

## Default configuration and notes

| Setting | BOLT | Matched ZO-BOLT |
| --- | --- | --- |
| Model precision; LoRA | BF16; rank 8 on `q_proj`/`k_proj` | Same |
| Learning rate; schedule | `5e-6`; cosine | `1e-4`; constant |
| Per-device batch; accumulation | 8; 8 steps | 128; 1 step |
| LoRA dropout | 0.05 | 0 |
| ZO perturbation; history | — | $\varepsilon=10^{-3}$; 40 histories |

Both presets train for two epochs with seed 42 and save every 50 steps; exact values are in [`configs/bolt.json`](configs/bolt.json) and [`configs/zo-bolt.json`](configs/zo-bolt.json). Data generation uses temperature 0.8, top-p 0.95, and at most 1,024 new tokens. Evaluation uses greedy decoding, at most 1,024 new tokens, and the **training prompt format**. Model weights and generated training data are not bundled. Every `--output-dir` must be new or empty; interrupted runs do not resume automatically through these scripts.
