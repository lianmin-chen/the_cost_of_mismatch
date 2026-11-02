"""Average on-policy variance across directions and plot the scale sweep."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common import write_csv


def analyze(config: Mapping[str, Any], run_dir: Path) -> None:
    """Read each direction's variance and write the final main table and plot."""
    experiment = config["experiment"]
    mus = sorted(float(mu) for mu in experiment["mus"])
    num_directions = int(experiment["num_directions"])
    by_mu: dict[float, list[float]] = {mu: [] for mu in mus}

    # Each direction contributes one variance estimate per perturbation scale.
    for direction_id in range(num_directions):
        path = run_dir / "directions" / f"direction_{direction_id:04d}.csv"
        if not path.is_file():
            raise RuntimeError(f"missing direction result: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(mus):
            raise RuntimeError(f"wrong number of perturbation scales: {path}")
        seen_mus = set()
        for row in rows:
            mu = float(row["mu"])
            if int(row["direction_id"]) != direction_id or mu not in by_mu or mu in seen_mus:
                raise RuntimeError(f"invalid direction result: {path}")
            seen_mus.add(mu)
            by_mu[mu].append(float(row["oracle_noise_variance_per_dim"]))

    table = [
        {
            "mu": mu,
            "oracle_noise_variance_per_dim": float(np.mean(by_mu[mu])),
        }
        for mu in mus
    ]
    output_dir = run_dir / "analysis"
    write_csv(output_dir / "main_table.csv", table)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    axis.plot(
        [row["mu"] for row in table],
        [row["oracle_noise_variance_per_dim"] for row in table],
        marker="o",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Perturbation scale $\mu$")
    axis.set_ylabel("Oracle-noise variance per dimension")
    axis.grid(True, which="both", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "online_rlvr_oracle_noise.png", dpi=240)
    figure.savefig(output_dir / "online_rlvr_oracle_noise.pdf")
    plt.close(figure)
    print(f"[analyze] wrote on-policy variance to {output_dir}")
