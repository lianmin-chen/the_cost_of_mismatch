"""Evaluate and summarize the fixed BOLT oracle-noise experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from datasets import Dataset
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
from ZooBOLT_Variance.bolt_objective import FixedBoltObjective
from ZooBOLT_Variance.oracle_noise_core import (
    observation_statistics,
    validate_config,
)


RAW_FIELDS = [
    "direction_id",
    "repeat_id",
    "mu",
    "match_mse_per_dim",
    "mismatch_mse_per_dim",
]


def evaluate(
    config: Mapping[str, Any],
    run_dir: Path,
    dataset: Dataset,
) -> None:
    """Measure match and mismatch errors around the initial LoRA parameters."""
    # Load the model and save its initial parameters as the common reference point.
    model, tokenizer, named_parameters = prepare_model(config)
    objective = FixedBoltObjective(model, tokenizer, dataset, config["model"])
    baseline = {
        name: parameter.detach().clone() for name, parameter in named_parameters
    }
    shapes = [(name, list(parameter.shape)) for name, parameter in named_parameters]
    dimension = sum(parameter.numel() for _, parameter in named_parameters)
    experiment = config["experiment"]
    population_size = len(dataset)
    microbatch_size = int(experiment["forward_microbatch_size"])
    batch_size = int(experiment["batch_size"])
    num_directions = int(experiment["num_directions"])
    num_repeats = int(experiment["num_repeats"])
    direction_seed_base = int(experiment["direction_seed"])
    batch_seed_base = int(experiment["batch_seed"])
    # Compute per-example losses at theta* once for all directions and scales.
    base_values = objective.evaluate_all(microbatch_size, "baseline")

    try:
        # Use one fixed Gaussian vector per direction across all scales.
        for direction_id in tqdm(
            range(num_directions),
            desc="ZOBOLT directions",
            unit="direction",
            dynamic_ncols=True,
        ):
            raw_path = run_dir / "directions" / f"direction_{direction_id:04d}.csv"
            direction_seed = derived_seed(direction_seed_base, direction_id)
            direction, norm_sq = make_direction(shapes, direction_seed)

            # Fix one minibatch pair per repetition and reuse it at every scale.
            batches = []
            for repeat_id in range(num_repeats):
                seed = derived_seed(batch_seed_base, direction_id, repeat_id)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                batch_1 = torch.randperm(population_size, generator=generator)[
                    :batch_size
                ].tolist()
                batch_2 = torch.randperm(population_size, generator=generator)[
                    :batch_size
                ].tolist()
                while batch_2 == batch_1:
                    batch_2 = torch.randperm(population_size, generator=generator)[
                        :batch_size
                    ].tolist()
                batches.append((batch_1, batch_2))

            rows = []
            for mu_value in tqdm(
                experiment["mus"],
                desc=f"direction {direction_id}: scales",
                unit="scale",
                dynamic_ncols=True,
                leave=False,
            ):
                mu = float(mu_value)
                # Evaluate theta* + mu*u once; reuse these losses for all repetitions.
                set_perturbed_parameters(named_parameters, baseline, direction, mu)
                perturbed_values = objective.evaluate_all(
                    microbatch_size, f"direction {direction_id} mu={mu:g}"
                )
                restore_parameters(named_parameters, baseline)

                # Compute each repetition's error from the cached in-memory losses.
                for repeat_id, (batch_1, batch_2) in enumerate(batches):
                    # Reuse batch_1 at the base point for match; use batch_2 for mismatch.
                    statistics = observation_statistics(
                        perturbed_values,
                        base_values,
                        batch_1,
                        batch_2,
                        mu,
                        norm_sq,
                        dimension,
                    )
                    rows.append(
                        {
                            "direction_id": direction_id,
                            "repeat_id": repeat_id,
                            "mu": format(mu, ".17g"),
                            "match_mse_per_dim": format(statistics["match"], ".17g"),
                            "mismatch_mse_per_dim": format(
                                statistics["mismatch"], ".17g"
                            ),
                        }
                    )
                print(
                    f"[direction {direction_id + 1}/{num_directions}] "
                    f"mu={mu:g} complete",
                    flush=True,
                )
            # Save every scale and repetition for this direction.
            write_csv(raw_path, rows, RAW_FIELDS)
            print(f"[direction {direction_id}] committed {len(rows)} observations")
    finally:
        restore_parameters(named_parameters, baseline)


def run(config_path: Path) -> None:
    """Run the complete offline experiment with its own objective and statistics."""
    config = resolve_config(config_path, validate_config)
    # Load the fixed evaluation dataset and check its expected size.
    dataset = Dataset.load_from_disk(config["assets"]["weighted_dataset"])
    expected_size = int(config["experiment"]["dataset_size"])
    if len(dataset) != expected_size:
        raise ValueError(f"expected {expected_size} dataset rows, found {len(dataset)}")

    # Create a fresh directory, then evaluate and summarize the complete run.
    run_dir = create_run_dir(config)
    print(f"[run] new experiment directory: {run_dir}")
    evaluate(config, run_dir, dataset)

    from ZooBOLT_Variance.analyze_results import analyze

    analyze(config, run_dir)
