"""Configuration checks and noise metrics for fixed BOLT."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    for key in (
        "dataset_size",
        "batch_size",
        "forward_microbatch_size",
        "num_directions",
        "num_repeats",
    ):
        if int(experiment[key]) <= 0:
            raise ValueError(f"experiment.{key} must be positive")
    if int(experiment["batch_size"]) >= int(experiment["dataset_size"]):
        raise ValueError("batch_size must be smaller than dataset_size for mismatch")
    if not experiment["mus"]:
        raise ValueError("experiment.mus must not be empty")
    if len(experiment["mus"]) != len(set(float(x) for x in experiment["mus"])):
        raise ValueError("experiment.mus contains duplicates")
    if any(float(mu) <= 0 for mu in experiment["mus"]):
        raise ValueError("all perturbation scales must be positive")
    if config["model"]["loss_impl"] not in {"liger", "original"}:
        raise ValueError("model.loss_impl must be 'liger' or 'original'")


def observation_statistics(
    f_perturbed: np.ndarray,
    f_base: np.ndarray,
    batch_1: Sequence[int],
    batch_2: Sequence[int],
    mu: float,
    direction_norm_sq: float,
    dimension: int,
) -> dict[str, float]:
    """Compare minibatch differences with the full-data finite difference."""
    first = np.asarray(batch_1, dtype=np.int64)
    second = np.asarray(batch_2, dtype=np.int64)
    # The full-data difference is the noise-free reference for this direction and scale.
    f_full_plus = float(np.mean(f_perturbed, dtype=np.float64))
    f_full_base = float(np.mean(f_base, dtype=np.float64))
    f_batch_plus = float(np.mean(f_perturbed[first], dtype=np.float64))
    f_batch_base_1 = float(np.mean(f_base[first], dtype=np.float64))
    f_batch_base_2 = float(np.mean(f_base[second], dtype=np.float64))
    delta_f = (f_full_plus - f_full_base) / mu

    output: dict[str, float] = {}
    # The two conditions differ only in the base-point minibatch.
    for condition, f_batch_base in (
        ("match", f_batch_base_1),
        ("mismatch", f_batch_base_2),
    ):
        delta_hat = (f_batch_plus - f_batch_base) / mu
        scalar_error = delta_hat - delta_f
        # The gradient error is scalar_error*u; normalize its squared norm by d.
        output[condition] = scalar_error**2 * direction_norm_sq / dimension
    return output
