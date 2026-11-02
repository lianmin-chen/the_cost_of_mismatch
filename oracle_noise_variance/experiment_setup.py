"""Shared configuration and output setup for the two oracle-noise experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4


def resolve_config(
    config_path: Path, validate: Callable[[Mapping[str, Any]], None]
) -> dict[str, Any]:
    """Load one experiment's config and resolve asset paths relative to it."""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in config["assets"].items():
        if str(value).startswith("<") and str(value).endswith(">"):
            raise ValueError(
                f"replace the path placeholder assets.{key}={value!r} in {config_path}"
            )
        config["assets"][key] = str((config_path.parent / value).resolve())
    validate(config)
    return config


def create_run_dir(config: Mapping[str, Any]) -> Path:
    """Create a distinct output directory for one complete experiment."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        Path(config["assets"]["results_root"])
        / f"{config['run_id']}-{stamp}-{uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
