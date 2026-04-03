from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

LOCKED_METHOD_CONSTANTS = {
    "bootstrap_n_reps": 1000,
    "null_bin_width_bits": 2,
    "null_min_family_count": 5,
    "frontier_balance_seed": 0,
}

ALLOWED_SETTINGS = {"planted", "mini_ioi", "gpt2_ioi"}


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")
    return value


@dataclass(frozen=True, slots=True)
class MethodConstants:
    bootstrap_n_reps: int
    null_bin_width_bits: int
    null_min_family_count: int
    frontier_balance_seed: int

    def __post_init__(self) -> None:
        for field_name, expected in LOCKED_METHOD_CONSTANTS.items():
            actual = getattr(self, field_name)
            if actual != expected:
                raise ValueError(
                    f"{field_name} must remain locked at {expected}, got {actual}"
                )

    @classmethod
    def default(cls) -> "MethodConstants":
        return cls(**LOCKED_METHOD_CONSTANTS)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MethodConstants":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SeedBundle:
    global_seed: int
    dataset_seed: int
    model_init_seed: int
    candidate_search_seed: int
    bootstrap_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "global": self.global_seed,
            "dataset": self.dataset_seed,
            "model_init": self.model_init_seed,
            "candidate_search": self.candidate_search_seed,
            "bootstrap": self.bootstrap_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedBundle":
        return cls(
            global_seed=int(data["global"]),
            dataset_seed=int(data["dataset"]),
            model_init_seed=int(data["model_init"]),
            candidate_search_seed=int(data["candidate_search"]),
            bootstrap_seed=int(data["bootstrap"]),
        )


@dataclass(frozen=True, slots=True)
class PathConfig:
    results_dir: str
    artifacts_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathConfig":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str
    dtype: str
    num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RunConfig:
    setting_id: str
    variant: str
    description: str
    paths: PathConfig
    method: MethodConstants
    seeds: SeedBundle
    runtime: RuntimeConfig
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.setting_id not in ALLOWED_SETTINGS:
            raise ValueError(
                f"setting_id must be one of {sorted(ALLOWED_SETTINGS)}, got {self.setting_id!r}"
            )
        if not self.variant:
            raise ValueError("variant must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "setting": {
                "id": self.setting_id,
                "variant": self.variant,
                "description": self.description,
            },
            "paths": self.paths.to_dict(),
            "method": self.method.to_dict(),
            "seeds": self.seeds.to_dict(),
            "runtime": self.runtime.to_dict(),
        }
        if self.extras:
            payload["extras"] = self.extras
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        setting = _ensure_mapping(data["setting"], context="setting")
        return cls(
            setting_id=str(setting["id"]),
            variant=str(setting["variant"]),
            description=str(setting["description"]),
            paths=PathConfig.from_dict(_ensure_mapping(data["paths"], context="paths")),
            method=MethodConstants.from_dict(
                _ensure_mapping(data["method"], context="method")
            ),
            seeds=SeedBundle.from_dict(_ensure_mapping(data["seeds"], context="seeds")),
            runtime=RuntimeConfig.from_dict(
                _ensure_mapping(data["runtime"], context="runtime")
            ),
            extras=dict(data.get("extras", {})),
        )


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    mapping = _ensure_mapping(data, context=f"config {config_path}")
    return RunConfig.from_dict(mapping)


def save_run_config(config: RunConfig, path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
