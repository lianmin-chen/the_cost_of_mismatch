#!/usr/bin/env python
"""Run the BOLT or RL oracle-noise experiment through one entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("BOLT", "RL"))
    parser.add_argument("--config", type=Path, help="Config JSON for the selected experiment")
    args = parser.parse_args()

    # Load only the selected experiment's objective and evaluation dependencies.
    if args.experiment == "BOLT":
        from ZooBOLT_Variance.experiment import run

        config_path = args.config or HERE / "ZooBOLT_Variance" / "config.json"
    else:
        from ZooOnlineRL_Variance.experiment import run

        config_path = args.config or HERE / "ZooOnlineRL_Variance" / "config.json"
    run(config_path)


if __name__ == "__main__":
    main()
