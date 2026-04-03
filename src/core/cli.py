from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import RunConfig, load_run_config


def _serialize_config_summary(config: RunConfig) -> dict[str, object]:
    return {
        "setting_id": config.setting_id,
        "variant": config.variant,
        "description": config.description,
        "method": config.method.to_dict(),
    }


def run_experiment_placeholder(
    *,
    expected_setting: str,
    entrypoint: str,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog=entrypoint)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    if config.setting_id != expected_setting:
        raise ValueError(
            f"{entrypoint} expected setting_id={expected_setting!r}, "
            f"got {config.setting_id!r}"
        )

    payload = {
        "status": "scaffold_only",
        "entrypoint": entrypoint,
        "config_path": str(Path(args.config)),
        "config": _serialize_config_summary(config),
        "message": "M0 placeholder CLI reached successfully; scientific execution is not implemented yet.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_analysis_placeholder(
    *,
    entrypoint: str,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog=entrypoint)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    payload = {
        "status": "scaffold_only",
        "entrypoint": entrypoint,
        "run_dir": str(run_dir),
        "run_dir_exists": run_dir.exists(),
        "message": "M0 placeholder analysis CLI reached successfully; run artifact analysis is not implemented yet.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
