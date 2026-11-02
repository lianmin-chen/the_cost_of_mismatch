"""Evaluate and summarize direct on-policy oracle-noise variance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tqdm.auto import tqdm

from common import (
    derived_seed,
    make_direction,
    restore_parameters,
    set_perturbed_parameters,
    write_csv,
)
from experiment_setup import create_run_dir, resolve_config
from prepare_model import prepare_model
from ZooOnlineRL_Variance.analyze_results import analyze
from ZooOnlineRL_Variance.online_oracle import OnPolicyOracle
from ZooOnlineRL_Variance.online_rl_core import (
    direction_variance,
    rollout_seeds,
    validate_config,
)


DIRECTION_FIELDS = ("direction_id", "mu", "oracle_noise_variance_per_dim")


def evaluate(config: Mapping[str, Any], run_dir: Path, prompt_rows: list[dict]) -> None:
    """Measure sample variance over fresh rollout pairs at each direction and scale."""
    model, tokenizer, named_parameters = prepare_model(config)
    oracle = OnPolicyOracle(model, tokenizer, prompt_rows, config)
    baseline = {
        name: parameter.detach().clone() for name, parameter in named_parameters
    }
    shapes = [(name, list(parameter.shape)) for name, parameter in named_parameters]
    dimension = sum(parameter.numel() for _, parameter in named_parameters)
    experiment = config["experiment"]
    num_directions = int(experiment["num_directions"])
    num_repeats = int(experiment["num_repeats"])
    mus = [float(mu) for mu in experiment["mus"]]
    rollout_seed = int(experiment["rollout_seed"])
    seen_seeds: set[int] = set()
    progress = tqdm(
        total=2 * num_directions * len(mus) * num_repeats,
        desc="Fresh on-policy oracle queries",
        unit="query",
        dynamic_ncols=True,
    )

    try:
        for direction_id in range(num_directions):
            # One Gaussian direction is shared by all scales and repetitions here.
            direction_seed = derived_seed(
                int(experiment["direction_seed"]), direction_id
            )
            direction, norm_sq = make_direction(shapes, direction_seed)
            rows = []
            for mu_index, mu in enumerate(mus):
                deltas = []
                for repeat_id in range(num_repeats):
                    plus_seed, center_seed = rollout_seeds(
                        rollout_seed, direction_id, mu_index, repeat_id
                    )
                    if (
                        plus_seed == center_seed
                        or plus_seed in seen_seeds
                        or center_seed in seen_seeds
                    ):
                        raise RuntimeError("rollout seed collision")
                    seen_seeds.update((plus_seed, center_seed))

                    # Each endpoint samples new completions under its own policy.
                    try:
                        set_perturbed_parameters(
                            named_parameters, baseline, direction, mu
                        )
                        plus = oracle.query(plus_seed)
                        progress.update(1)
                    finally:
                        restore_parameters(named_parameters, baseline)
                    center = oracle.query(center_seed)
                    progress.update(1)
                    deltas.append((plus - center) / mu)

                # Estimate variance across repetitions before averaging directions.
                rows.append(
                    {
                        "direction_id": direction_id,
                        "mu": format(mu, ".17g"),
                        "oracle_noise_variance_per_dim": format(
                            direction_variance(deltas, norm_sq, dimension), ".17g"
                        ),
                    }
                )
            write_csv(
                run_dir / "directions" / f"direction_{direction_id:04d}.csv",
                rows,
                DIRECTION_FIELDS,
            )
    finally:
        progress.close()
        restore_parameters(named_parameters, baseline)


def run(config_path: Path) -> None:
    """Run the complete online experiment with fresh rollout pairs."""
    config = resolve_config(config_path, validate_config)
    prompt_path = Path(config["assets"]["prompt_set"])
    if not prompt_path.is_file():
        raise FileNotFoundError(f"fixed prompt set not found: {prompt_path}")
    prompt_rows = json.loads(prompt_path.read_text(encoding="utf-8"))["rows"]
    if len(prompt_rows) != int(config["experiment"]["batch_size"]):
        raise ValueError("fixed prompt set size disagrees with experiment.batch_size")

    run_dir = create_run_dir(config)
    print(f"[run] new experiment directory: {run_dir}", flush=True)
    evaluate(config, run_dir, prompt_rows)
    analyze(config, run_dir)
