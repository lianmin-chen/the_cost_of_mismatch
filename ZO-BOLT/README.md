<div align="center">

# The Cost of Mismatch: Noise Amplification in Zeroth-Order Reinforcement Learning

Code repository

</div>

## Contents

This repository contains two experiment groups: end-to-end first-order BOLT and **matched ZO-BOLT** training on GSM8K, and separate oracle-noise variance measurements. The former follows `data → train → eval`: both methods use the same fixed response data, and evaluation uses the same prompt format as data generation. The latter measures estimator noise at the initial model without training it.

| Directory | Purpose |
| --- | --- |
| [`gsm8k/`](gsm8k/README.md) | Direct data, training, and checkpoint-evaluation scripts and two training presets |
| [`zobolt/data/`](zobolt/data/gsm8k.py) | GSM8K answer extraction and fixed Qwen boxed few-shot prompts |
| `zobolt/train/` | First-order BOLT and matched ZO-BOLT trainers |
| `zobolt/rewards/` | Boxed-answer math grading |
| `zobolt/eval/` | GSM8K vLLM evaluation and bundled evaluation set |
| [`../oracle_noise_variance/`](../oracle_noise_variance/README.md) | Independent BOLT and on-policy RL oracle-noise variance experiments |

## Environment

Use Linux, Python 3.12, and an NVIDIA CUDA GPU. In a Python environment compatible with your CUDA setup, install the main versions used in the original experiments:

```bash
pip install \
  torch==2.9.1 transformers==4.57.6 trl==1.8.0 peft==0.19.1 \
  datasets==5.0.0 vllm==0.16.0 deepspeed==0.19.2 accelerate==1.14.0 \
  numpy==2.2.6 sympy==1.14.0 wandb==0.28.1 liger-kernel==0.8.0 \
  math-verify==0.9.0 latex2sympy2-extended==1.11.0 pylatexenc==2.10 \
  antlr4-python3-runtime==4.9.3 tqdm==4.69.0
```

PyTorch, vLLM, and DeepSpeed must be built for your CUDA environment. If this installation command does not fit your system, install compatible builds first, then the remaining packages. The variance experiments have a separate [dependency file](../oracle_noise_variance/requirements.txt).

Provide the Qwen2.5-1.5B model weights locally or via Hugging Face. Large generated datasets, training checkpoints, and experimental outputs are not bundled. The fixed evaluation set is included at `zobolt/eval/processed_datasets/gsm8k_eval/`.

## Reproduce GSM8K

From `the_cost_of_mismatch/ZO-BOLT/`, run data generation, the two training methods, and checkpoint evaluation in that order. Filtering and `\boxed{...}` truncation are part of the data entry; no extra script is needed. Complete commands, outputs, and defaults are in the [GSM8K reproduction guide](gsm8k/README.md). Hydra is not needed.

## Oracle-noise variance

The fixed weighted-GSM8K minibatch match/mismatch comparison and the fresh on-policy rollout variance measurement have separate code and entry points. Fixed inputs are included; model weights and results are not. See the [variance guide](../oracle_noise_variance/README.md).
