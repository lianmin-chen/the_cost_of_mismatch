"""Configuration, rollout seeds, and variance for the on-policy experiment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from common import derived_seed


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    for key in ("batch_size", "num_directions"):
        if int(experiment[key]) <= 0:
            raise ValueError(f"experiment.{key} must be positive")
    if int(experiment["num_repeats"]) < 2:
        raise ValueError("experiment.num_repeats must be at least 2")
    mus = [float(mu) for mu in experiment["mus"]]
    if not mus or any(mu <= 0 for mu in mus) or len(mus) != len(set(mus)):
        raise ValueError("experiment.mus must contain distinct positive values")
    generation = config["generation"]
    if float(generation["temperature"]) <= 0:
        raise ValueError("stochastic on-policy evaluation requires temperature > 0")
    if not 0 < float(generation["top_p"]) <= 1:
        raise ValueError("generation.top_p must lie in (0, 1]")
    if int(generation["max_new_tokens"]) <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if float(config["model"]["lora_dropout"]) != 0:
        raise ValueError("controlled evaluation requires lora_dropout=0")


def rollout_seeds(
    base_seed: int, direction_id: int, mu_index: int, repeat_id: int
) -> tuple[int, int]:
    """Derive separate seeds for the perturbed and base-policy queries."""
    coordinates = (direction_id, mu_index, repeat_id)
    return (
        derived_seed(base_seed, *coordinates, 1),
        derived_seed(base_seed, *coordinates, 0),
    )


def direction_variance(
    deltas: Sequence[float], direction_norm_sq: float, dimension: int
) -> float:
    """Return the dimension-normalized sample variance at fixed direction and mu."""
    values = np.asarray(deltas, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("at least two rollout pairs are required")
    return float(np.var(values, ddof=1) * direction_norm_sq / dimension)
