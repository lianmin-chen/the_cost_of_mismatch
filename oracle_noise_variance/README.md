# Oracle-Noise Variance Experiments

This directory reproduces the three oracle-noise curves in Table 1 and Figure 1 of *The Cost of Mismatch: Noise Amplification in Zeroth-Order Reinforcement Learning*. Both experiments evaluate the **initial** Qwen2.5-1.5B LoRA model; neither trains it.

| Experiment | Comparison | Reported quantity | Details |
| --- | --- | --- | --- |
| BOLT | Same versus independent minibatches at the two finite-difference endpoints | Error relative to a full-dataset reference | [BOLT experiment](ZooBOLT_Variance/README.md) |
| RL | Independent fresh rollouts at the two endpoints, with fixed prompts | Conditional oracle-noise sample variance | [RL experiment](ZooOnlineRL_Variance/README.md) |

The BOLT and RL curves use different estimators; their definitions, sampling schemes, and default settings are documented in the linked experiment guides.

## Run the experiments

Run all commands from `the_cost_of_mismatch/oracle_noise_variance/` in a Python 3.12 environment with a compatible CUDA GPU:

```bash
pip install -r requirements.txt
```

The fixed BOLT Dataset and RL prompt set are included. No data-preparation step is required. In both [BOLT config](ZooBOLT_Variance/config.json) and [RL config](ZooOnlineRL_Variance/config.json), set `assets.base_model` to the **same local model snapshot** and `assets.results_root` to the desired output directory. Replace the path placeholders before running; relative asset paths are resolved from each config file's directory.

Run BOLT and RL separately:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py BOLT --config ZooBOLT_Variance/config.json
CUDA_VISIBLE_DEVICES=0 python run_experiment.py RL --config ZooOnlineRL_Variance/config.json
```

Each command prints a newly created run directory. Use its `analysis/main_table.csv` in the comparison step:

```bash
python make_paper_comparison.py --bolt-table /path/to/bolt_run/analysis/main_table.csv --rl-table /path/to/rl_run/analysis/main_table.csv --output-dir results/paper_comparison
```

The comparison writes `table1_comparison.csv` and `noise_variance_comparison.{png,pdf}`. Individual runs also retain per-direction CSV files; the RL run generates its own variance plot. Runs do not support resume, and model weights or experimental results are not included in this repository.

## Code

[`run_experiment.py`](run_experiment.py) loads only the selected BOLT or RL experiment. Shared model setup and perturbation utilities live in [`prepare_model.py`](prepare_model.py) and [`common.py`](common.py); experiment-specific code is described in the [BOLT](ZooBOLT_Variance/README.md) and [RL](ZooOnlineRL_Variance/README.md) guides.
