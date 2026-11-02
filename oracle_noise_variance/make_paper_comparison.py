#!/usr/bin/env python
"""Combine the two variance experiments into a Table 1 CSV and trend figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from common import write_csv


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def combine(bolt_path: Path, rl_path: Path) -> list[dict[str, float]]:
    bolt = read_rows(bolt_path)
    rl = read_rows(rl_path)
    bolt_by_mu = {float(row["mu"]): row for row in bolt}
    rl_by_mu = {float(row["mu"]): row for row in rl}
    if not bolt_by_mu or bolt_by_mu.keys() != rl_by_mu.keys():
        raise ValueError("BOLT and RL tables must contain the same nonempty mu sweep")
    if len(bolt_by_mu) != len(bolt) or len(rl_by_mu) != len(rl):
        raise ValueError("each input table must contain one row per mu")

    combined = []
    for mu in sorted(bolt_by_mu):
        if mu <= 0:
            raise ValueError("perturbation scales must be positive")
        fixed = bolt_by_mu[mu]
        sampled = rl_by_mu[mu]
        combined.append(
            {
                "mu": mu,
                "bolt_match_variance_per_dim": float(fixed["match_mse_per_dim"]),
                "bolt_mismatch_variance_per_dim": float(
                    fixed["mismatch_mse_per_dim"]
                ),
                "on_policy_rl_variance_per_dim": float(
                    sampled["oracle_noise_variance_per_dim"]
                ),
            }
        )
    return combined


def plot(rows: list[dict[str, float]], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    series = (
        ("bolt_match_variance_per_dim", "BOLT: match", "#2563eb"),
        ("bolt_mismatch_variance_per_dim", "BOLT: mismatch", "#dc2626"),
        ("on_policy_rl_variance_per_dim", "RL: on-policy", "#059669"),
    )
    for field, label, color in series:
        axis.plot(
            [row["mu"] for row in rows],
            [row[field] for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Perturbation scale $\mu$")
    axis.set_ylabel("Oracle-noise variance per dimension")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "noise_variance_comparison.png", dpi=240)
    figure.savefig(output_dir / "noise_variance_comparison.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bolt-table", type=Path, required=True)
    parser.add_argument("--rl-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = combine(args.bolt_table, args.rl_table)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "table1_comparison.csv", rows)
    plot(rows, output_dir)
    print(f"Wrote Table 1 comparison and figure to {output_dir}")


if __name__ == "__main__":
    main()
