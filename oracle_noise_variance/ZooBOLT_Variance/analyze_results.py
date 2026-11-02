#!/usr/bin/env python
"""Aggregate per-direction observations into the offline main table."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common import write_csv
OBSERVATION_FIELDS = {
    "direction_id",
    "repeat_id",
    "mu",
    "match_mse_per_dim",
    "mismatch_mse_per_dim",
}


def analyze(config: Mapping[str, Any], run_dir: Path) -> None:
    """Average repetitions, then directions, for each perturbation scale."""
    experiment = config["experiment"]
    num_directions = int(experiment["num_directions"])
    num_repeats = int(experiment["num_repeats"])
    mus = sorted(float(mu) for mu in experiment["mus"])
    grouped: dict[tuple[float, int], list[dict[str, str]]] = defaultdict(list)

    # Group all observations by perturbation scale and direction.
    for direction_id in range(num_directions):
        path = run_dir / "directions" / f"direction_{direction_id:04d}.csv"
        if not path.is_file():
            raise RuntimeError(f"direction {direction_id} is incomplete: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not OBSERVATION_FIELDS.issubset(reader.fieldnames or []):
                raise RuntimeError(f"direction file has the wrong columns: {path}")
            for row in reader:
                if int(row["direction_id"]) != direction_id:
                    raise RuntimeError(f"direction ID disagrees with file name: {path}")
                grouped[(float(row["mu"]), direction_id)].append(row)

    table = []
    if len(grouped) != len(mus) * num_directions:
        raise RuntimeError("direction files contain missing or unexpected mu values")
    # Average repetitions within each direction before averaging directions.
    for mu in mus:
        per_direction = []
        for direction_id in range(num_directions):
            rows = grouped[(mu, direction_id)]
            repeats = [int(row["repeat_id"]) for row in rows]
            if sorted(repeats) != list(range(num_repeats)):
                raise RuntimeError(
                    f"expected repeat IDs 0-{num_repeats - 1} "
                    f"for direction {direction_id}, mu={mu}"
                )
            per_direction.append(
                (
                    float(np.mean([float(row["match_mse_per_dim"]) for row in rows])),
                    float(np.mean([float(row["mismatch_mse_per_dim"]) for row in rows])),
                )
            )

        # Give each direction equal weight rather than treating all rows as new directions.
        match = float(np.mean([item[0] for item in per_direction]))
        mismatch = float(np.mean([item[1] for item in per_direction]))
        table.append(
            {
                "mu": mu,
                "match_mse_per_dim": match,
                "mismatch_mse_per_dim": mismatch,
                "mismatch_to_match_ratio": mismatch / match,
            }
        )

    # Write one row per scale for the three-curve comparison script.
    output = run_dir / "analysis" / "main_table.csv"
    write_csv(output, table)
    print(f"[analyze] wrote match/mismatch MSE to {output}")


