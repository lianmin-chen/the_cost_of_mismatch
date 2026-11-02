"""Random seeds, parameter perturbations, and file I/O shared by both noise experiments."""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    # NumPy's legacy global RNG accepts only unsigned 32-bit seeds, whereas
    # our collision-resistant derived rollout seeds occupy the positive
    # 63-bit range. Preserve the full seed for Python/PyTorch and map it
    # deterministically into NumPy's supported range.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derived_seed(base_seed: int, *coordinates: int) -> int:
    payload = ":".join(str(value) for value in (base_seed, *coordinates)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (2**63 - 1)


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None
) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_direction(
    parameter_shapes: Sequence[tuple[str, Sequence[int]]], seed: int
) -> tuple[dict[str, torch.Tensor], float]:
    """Generate a Gaussian direction for the parameter shapes and its squared norm."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction: dict[str, torch.Tensor] = {}
    norm_sq = 0.0
    for name, shape in parameter_shapes:
        value = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
        direction[name] = value
        norm_sq += float(torch.sum(value.double().square()))
    return direction, norm_sq


@torch.no_grad()
def set_perturbed_parameters(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    baseline: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    mu: float,
) -> None:
    for name, parameter in named_parameters:
        parameter.copy_(
            baseline[name]
            + mu * direction[name].to(parameter.device, dtype=parameter.dtype)
        )


@torch.no_grad()
def restore_parameters(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    baseline: Mapping[str, torch.Tensor],
) -> None:
    for name, parameter in named_parameters:
        parameter.copy_(baseline[name])
