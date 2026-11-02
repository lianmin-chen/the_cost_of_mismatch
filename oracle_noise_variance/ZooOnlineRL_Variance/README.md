# Direct On-Policy Oracle-Noise Variance

This directory reproduces the on-policy RL curve in Table 1 and Figure 1. It evaluates forward zeroth-order (ZO) differences at the **initial** LoRA parameters: each endpoint generates fresh completions under its own policy, so the two function values use independent rollout data. The model is not trained.

## Estimator and measurement

A fixed set of $B$ GSM8K prompts is shared across queries. At parameters $\theta$, one oracle query generates a new completion for each prompt and returns $\widetilde J(\theta,\xi)$, the **negative mean verifier reward**. For Gaussian direction $u_k$, scale $\mu$, and repetition $r$, the perturbed and base queries use distinct rollout seeds:

$$
\hat\delta_{k,r}(\mu)=
\frac{\widetilde J(\theta^*+\mu u_k,\xi_{k,r,+})
      -\widetilde J(\theta^*,\xi_{k,r,0})}{\mu},
\qquad
\bar\delta_k(\mu)=\frac{1}{R}\sum_{r=1}^{R}\hat\delta_{k,r}(\mu).
$$

The full on-policy objective cannot be evaluated exactly from finite rollouts. The code therefore estimates **conditional oracle-noise variance** from $R$ independent query pairs at each fixed $(u_k,\mu)$, then averages $K$ directions:

$$
\hat V_{\mathrm{on}}(\mu)=\frac{1}{K}\sum_{k=1}^{K}
\frac{\|u_k\|^2}{d(R-1)}
\sum_{r=1}^{R}\bigl(\hat\delta_{k,r}(\mu)-\bar\delta_k(\mu)\bigr)^2,
$$

where $d$ is the number of trainable LoRA parameters. The repetition mean only centers the sample variance; the experiment does **not** average the ZO gradient vectors into one update.

## Reproduction workflow

Run all commands from `the_cost_of_mismatch/oracle_noise_variance/`. Use Python 3.12, a CUDA GPU with compatible PyTorch, and a local Qwen2.5-1.5B model snapshot. The 16 fixed GSM8K prompts and answers are included at `ZooOnlineRL_Variance/data/fixed_gsm8k_prompts_16.json`; only the rollout completions are sampled during evaluation.

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set `assets.base_model` and `assets.results_root` in `ZooOnlineRL_Variance/config.json`. The included prompt set is already selected by `assets.prompt_set`; relative asset paths resolve from the config file's directory. Run fresh paired rollouts and analyze their variance:

   ```bash
   CUDA_VISIBLE_DEVICES=0 \
   python run_experiment.py RL \
     --config ZooOnlineRL_Variance/config.json
   ```

   Each invocation creates and prints a new `<results_root>/<run_id>-<UTC-timestamp>-<suffix>/`. Its `directions/direction_XXXX.csv` files store the variance for each direction and scale; `analysis/main_table.csv` averages directions, and `analysis/online_rlvr_oracle_noise.{png,pdf}` plots the sweep. Individual completions and rewards are not saved. There is no stage or resume mode; an interrupted run must be restarted in a new directory.

## Default configuration

The config uses FP32, LoRA rank 8 on `q_proj`/`k_proj`, $B=16$ fixed prompts, $K=10$ directions, $R=32$ fresh rollout pairs per direction and scale, and $\mu\in\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2}\}$. Each query samples up to 512 new tokens. For comparison with fixed BOLT, use the same model snapshot, `experiment.direction_seed`, and `num_directions`; each experiment generates its directions from those settings independently.
