from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

MAP_FAMILY_HYPERGRIDS: dict[str, tuple[object, ...]] = {
    "linear_dense": ("default_dense",),
    "linear_sparse_l1": (1e-4, 1e-3, 1e-2),
    "linear_lowrank": (4, 8, 16),
    "mlp1_relu": (32, 64, 128),
}
MAP_FAMILY_ORDER = tuple(MAP_FAMILY_HYPERGRIDS)
MLP_RESTARTS = 3
NONZERO_THRESHOLD = 1e-8


@dataclass(frozen=True, slots=True)
class StandardizationStats:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor) -> "StandardizationStats":
        mean = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False)
        safe_scale = torch.where(
            scale.abs() < 1e-8,
            torch.ones_like(scale),
            scale,
        )
        return cls(mean=mean, scale=safe_scale)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.scale


@dataclass(slots=True)
class ReadoutFitResult:
    variable_name: str
    map_family_id: str
    hyperparameter_value: object
    input_dim: int
    output_dim: int
    standardization: StandardizationStats
    module: torch.nn.Module
    val_nll: float
    restart_count: int
    seed: int

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        standardized = self.standardization.transform(features)
        self.module.eval()
        with torch.no_grad():
            return self.module(standardized)

    def predict_classes(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.predict_logits(features)
        return torch.argmax(logits, dim=1)

    @property
    def parameter_count(self) -> int:
        if self.map_family_id == "linear_dense":
            return self.input_dim * self.output_dim + self.output_dim
        if self.map_family_id == "linear_sparse_l1":
            linear_layer = self.module
            if not isinstance(linear_layer, torch.nn.Linear):
                raise TypeError("linear_sparse_l1 expects a Linear module")
            nnz = int((linear_layer.weight.abs() > NONZERO_THRESHOLD).sum().item())
            return nnz + self.output_dim
        if self.map_family_id == "linear_lowrank":
            rank = int(self.hyperparameter_value)
            return rank * (self.input_dim + self.output_dim) + self.output_dim
        if self.map_family_id == "mlp1_relu":
            width = int(self.hyperparameter_value)
            return (
                self.input_dim * width
                + width
                + width * self.output_dim
                + self.output_dim
            )
        raise ValueError(f"Unknown map_family_id: {self.map_family_id}")


class LowRankReadout(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        self.left = torch.nn.Linear(input_dim, rank, bias=False)
        self.right = torch.nn.Linear(rank, output_dim, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(inputs))


class MLP1Readout(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(input_dim, width)
        self.output = torch.nn.Linear(width, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.hidden(inputs))
        return self.output(hidden)


def hyperparameter_id(map_family_id: str, hyperparameter_value: object) -> str:
    if map_family_id == "linear_dense":
        return "default_dense"
    if map_family_id == "linear_sparse_l1":
        return f"lambda={float(hyperparameter_value):g}"
    if map_family_id == "linear_lowrank":
        return f"rank={int(hyperparameter_value)}"
    if map_family_id == "mlp1_relu":
        return f"width={int(hyperparameter_value)}"
    raise ValueError(f"Unknown map family: {map_family_id}")


def build_readout_module(
    *,
    map_family_id: str,
    hyperparameter_value: object,
    input_dim: int,
    output_dim: int,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if map_family_id in {"linear_dense", "linear_sparse_l1"}:
        return torch.nn.Linear(input_dim, output_dim).to(dtype=dtype)
    if map_family_id == "linear_lowrank":
        return LowRankReadout(
            input_dim=input_dim,
            output_dim=output_dim,
            rank=int(hyperparameter_value),
        ).to(dtype=dtype)
    if map_family_id == "mlp1_relu":
        return MLP1Readout(
            input_dim=input_dim,
            output_dim=output_dim,
            width=int(hyperparameter_value),
        ).to(dtype=dtype)
    raise ValueError(f"Unknown map family: {map_family_id}")


def _l1_penalty(module: torch.nn.Module) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        return module.weight.abs().sum()
    raise TypeError("L1 penalty is only defined for plain Linear modules")


def fit_readout(
    *,
    variable_name: str,
    map_family_id: str,
    hyperparameter_value: object,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    output_dim: int,
    seed: int,
    linear_epochs: int = 40,
    mlp_epochs: int = 60,
    learning_rate: float = 0.05,
) -> ReadoutFitResult:
    input_dim = int(train_features.shape[1])
    standardization = StandardizationStats.fit(train_features)
    train_inputs = standardization.transform(train_features)
    val_inputs = standardization.transform(val_features)
    train_targets = train_labels.long()
    val_targets = val_labels.long()

    restart_count = MLP_RESTARTS if map_family_id == "mlp1_relu" else 1
    epochs = mlp_epochs if map_family_id == "mlp1_relu" else linear_epochs

    best_module: torch.nn.Module | None = None
    best_val_nll = float("inf")
    best_seed = seed

    for restart_index in range(restart_count):
        restart_seed = seed + restart_index
        torch.manual_seed(restart_seed)
        module = build_readout_module(
            map_family_id=map_family_id,
            hyperparameter_value=hyperparameter_value,
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=train_features.dtype,
        )
        optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)

        for _ in range(epochs):
            module.train()
            optimizer.zero_grad()
            logits = module(train_inputs)
            loss = F.cross_entropy(logits, train_targets)
            if map_family_id == "linear_sparse_l1":
                loss = loss + float(hyperparameter_value) * _l1_penalty(module)
            loss.backward()
            optimizer.step()

        module.eval()
        with torch.no_grad():
            val_logits = module(val_inputs)
            val_nll = float(F.cross_entropy(val_logits, val_targets).item())
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_module = module
            best_seed = restart_seed

    if best_module is None:
        raise RuntimeError("fit_readout failed to produce a module")

    return ReadoutFitResult(
        variable_name=variable_name,
        map_family_id=map_family_id,
        hyperparameter_value=hyperparameter_value,
        input_dim=input_dim,
        output_dim=output_dim,
        standardization=standardization,
        module=best_module,
        val_nll=best_val_nll,
        restart_count=restart_count,
        seed=best_seed,
    )


def parse_map_family_grid(config_value: Any) -> dict[str, tuple[object, ...]]:
    if config_value is None:
        return MAP_FAMILY_HYPERGRIDS
    if not isinstance(config_value, dict):
        raise TypeError("map_families config must be a mapping")
    parsed: dict[str, tuple[object, ...]] = {}
    for map_family_id, values in config_value.items():
        if map_family_id not in MAP_FAMILY_HYPERGRIDS:
            raise ValueError(f"Unknown map family in config: {map_family_id}")
        if not isinstance(values, list):
            raise TypeError(f"map_families[{map_family_id!r}] must be a list")
        parsed_values: list[object] = []
        for raw_value in values:
            if map_family_id == "linear_dense":
                parsed_values.append("default_dense")
            elif map_family_id == "linear_sparse_l1":
                parsed_values.append(float(raw_value))
            else:
                parsed_values.append(int(raw_value))
        parsed[map_family_id] = tuple(parsed_values)
    return parsed
