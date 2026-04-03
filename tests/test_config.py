from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.config import LOCKED_METHOD_CONSTANTS, RunConfig, load_run_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_run_config_round_trip() -> None:
    original = load_run_config(REPO_ROOT / "configs" / "planted" / "base.yaml")
    restored = RunConfig.from_dict(original.to_dict())

    assert restored == original
    assert restored.method.to_dict() == LOCKED_METHOD_CONSTANTS


def test_locked_method_constants_are_enforced() -> None:
    config_dict = load_run_config(REPO_ROOT / "configs" / "planted" / "base.yaml").to_dict()
    config_dict["method"]["bootstrap_n_reps"] = 999

    with pytest.raises(ValueError, match="bootstrap_n_reps"):
        RunConfig.from_dict(config_dict)


def test_gpt2_ioi_cli_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.gpt2_ioi", "--config", "configs/gpt2_ioi/base.yaml"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["fixed_prompt_length"] == 11
    assert Path(REPO_ROOT / payload["run_dir"]).exists() or Path(payload["run_dir"]).exists()


def test_mini_ioi_cli_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.mini_ioi", "--config", "configs/mini_ioi/base.yaml"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["training_final_accuracy"] >= 0.5
    assert Path(REPO_ROOT / payload["run_dir"]).exists() or Path(payload["run_dir"]).exists()


def test_planted_cli_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.planted", "--config", "configs/planted/smoke.yaml"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert Path(REPO_ROOT / payload["run_dir"]).exists() or Path(payload["run_dir"]).exists()
