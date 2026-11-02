# Fixed-BOLT Oracle-Noise Variance

This directory reproduces the two BOLT oracle-noise curves in Table 1 and Figure 1: at the **initial** Qwen2.5-1.5B LoRA parameters, it compares zeroth-order (ZO) estimates whose paired function evaluations use the same minibatch (`match`) or independently drawn minibatches (`mismatch`). The model is never trained.

## Estimator and measurement

The input is a fixed weighted Dataset $D$ of $N$ prompt–response pairs. For any subset $B$, $F_B(\theta)$ is the mean of its per-example losses, each defined as `weight × mean response-token NLL`; $F_D$ uses all $N$ examples. At $\theta^*$, draw $K$ Gaussian directions $u_k$. For each direction and repetition $r$, independently draw two minibatches $B^{(1)}_{k,r}$ and $B^{(2)}_{k,r}$ of size $b$ (they may overlap). Reuse the same direction and minibatch pair across all perturbation scales $\mu$.

For each direction, scale, and repetition, the full-data reference and two controlled ZO estimates are

$$
\begin{aligned}
g_{D,k}(\mu)
  &= \frac{F_D(\theta^*+\mu u_k)-F_D(\theta^*)}{\mu}\,u_k,\\
\hat g_{\mathrm{match},k,r}(\mu)
  &= \frac{F_{B^{(1)}_{k,r}}(\theta^*+\mu u_k)-F_{B^{(1)}_{k,r}}(\theta^*)}{\mu}\,u_k,\\
\hat g_{\mathrm{mismatch},k,r}(\mu)
  &= \frac{F_{B^{(1)}_{k,r}}(\theta^*+\mu u_k)-F_{B^{(2)}_{k,r}}(\theta^*)}{\mu}\,u_k.
\end{aligned}
$$

With $d$ trainable LoRA parameters, the reported value for either condition $q$ is

$$
\hat V_q(\mu)=\frac{1}{KRd}\sum_{k=1}^{K}\sum_{r=1}^{R}
\left\|\hat g_{q,k,r}(\mu)-g_{D,k}(\mu)\right\|^2.
$$

Thus each estimate is compared with the **same-direction full-data reference before averaging squared errors**; the gradient vectors themselves are not averaged. The program computes per-example losses at $\theta^*$ once and at each $\theta^*+\mu u_k$ once, then evaluates all repetitions from those in-memory values.

## Reproduction workflow

Run these commands from `the_cost_of_mismatch/oracle_noise_variance/`. Use Python 3.12, a CUDA GPU with compatible PyTorch, and a local Qwen2.5-1.5B snapshot. The fixed weighted Dataset is included at `ZooBOLT_Variance/data/weighted_gsm8k_2048/`; its 2,048 rows already contain `prompt`, `response`, and `weight`.

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set `assets.base_model` and `assets.results_root` in `ZooBOLT_Variance/config.json`. The included Dataset is already selected by `assets.weighted_dataset`; relative asset paths resolve from the config file's directory. Run the complete experiment:

   ```bash
   CUDA_VISIBLE_DEVICES=0 \
   python run_experiment.py BOLT \
     --config ZooBOLT_Variance/config.json
   ```

   Each run creates and prints a new `<results_root>/<run_id>-<UTC-timestamp>-<suffix>/`. Its `directions/direction_XXXX.csv` files hold per-repetition errors; `analysis/main_table.csv` holds one row per $\mu$ with `match_mse_per_dim`, `mismatch_mse_per_dim`, and `mismatch_to_match_ratio`. There is no stage or resume mode; an interrupted run must be restarted in a new directory.

## Default configuration

The included Dataset has $N=2048$ fixed examples. The config uses FP32, LoRA rank 8 on `q_proj`/`k_proj`, $K=10$ directions, $R=32$ repetitions per direction, $b=16$, and $\mu\in\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2}\}$. The model forward microbatch size is 8; it does **not** change the statistical minibatch size. Exact reproduction requires the included Dataset and its row order, the same model snapshot, and the configured seeds. To compare with the RL experiment, keep `experiment.direction_seed` and `num_directions` equal across their configs.
