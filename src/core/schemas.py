from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_INTERVENTIONS = {"N1", "N2", "R"}
VALID_SPLITS = {"train", "val", "test", "shift"}


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class Site:
    layer_index: int
    token_index: int

    def __post_init__(self) -> None:
        if self.layer_index < 0 or self.token_index < 0:
            raise ValueError("site indices must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "layer_index": self.layer_index,
            "token_index": self.token_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Site":
        return cls(
            layer_index=int(data["layer_index"]),
            token_index=int(data["token_index"]),
        )


@dataclass(frozen=True, slots=True)
class TupleMetadata:
    group_id: str
    template_family: str
    latent_base: dict[str, Any]
    latent_source: dict[str, Any]
    prompt_id_base: str | None = None
    prompt_id_source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.group_id, field_name="group_id")
        _require_non_empty(self.template_family, field_name="template_family")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "template_family": self.template_family,
            "latent_base": self.latent_base,
            "latent_source": self.latent_source,
            "prompt_id_base": self.prompt_id_base,
            "prompt_id_source": self.prompt_id_source,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TupleMetadata":
        return cls(
            group_id=str(data["group_id"]),
            template_family=str(data["template_family"]),
            latent_base=dict(data["latent_base"]),
            latent_source=dict(data["latent_source"]),
            prompt_id_base=data.get("prompt_id_base"),
            prompt_id_source=data.get("prompt_id_source"),
            extra=dict(data.get("extra", {})),
        )


@dataclass(frozen=True, slots=True)
class BaseSourceTuple:
    tuple_id: str
    base_input: dict[str, Any]
    source_input: dict[str, Any]
    intervention_type: str
    metadata: TupleMetadata

    def __post_init__(self) -> None:
        _require_non_empty(self.tuple_id, field_name="tuple_id")
        if self.intervention_type not in VALID_INTERVENTIONS:
            raise ValueError(
                f"intervention_type must be one of {sorted(VALID_INTERVENTIONS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tuple_id": self.tuple_id,
            "base_input": self.base_input,
            "source_input": self.source_input,
            "intervention_type": self.intervention_type,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseSourceTuple":
        return cls(
            tuple_id=str(data["tuple_id"]),
            base_input=dict(data["base_input"]),
            source_input=dict(data["source_input"]),
            intervention_type=str(data["intervention_type"]),
            metadata=TupleMetadata.from_dict(dict(data["metadata"])),
        )


@dataclass(frozen=True, slots=True)
class CandidateScoredInterventionRecord:
    candidate_id: str
    split: str
    tuple_record: BaseSourceTuple
    observed_output_token: str
    predicted_output_token: str

    def __post_init__(self) -> None:
        _require_non_empty(self.candidate_id, field_name="candidate_id")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
        _require_non_empty(
            self.observed_output_token, field_name="observed_output_token"
        )
        _require_non_empty(
            self.predicted_output_token, field_name="predicted_output_token"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "split": self.split,
            "tuple_record": self.tuple_record.to_dict(),
            "observed_output_token": self.observed_output_token,
            "predicted_output_token": self.predicted_output_token,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateScoredInterventionRecord":
        return cls(
            candidate_id=str(data["candidate_id"]),
            split=str(data["split"]),
            tuple_record=BaseSourceTuple.from_dict(dict(data["tuple_record"])),
            observed_output_token=str(data["observed_output_token"]),
            predicted_output_token=str(data["predicted_output_token"]),
        )


@dataclass(frozen=True, slots=True)
class ResidualContribution:
    candidate_id: str
    split: str
    group_id: str
    residual_bits: float
    n_examples: int

    def __post_init__(self) -> None:
        _require_non_empty(self.candidate_id, field_name="candidate_id")
        _require_non_empty(self.group_id, field_name="group_id")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
        if self.n_examples <= 0:
            raise ValueError("n_examples must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "split": self.split,
            "group_id": self.group_id,
            "residual_bits": self.residual_bits,
            "n_examples": self.n_examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResidualContribution":
        return cls(
            candidate_id=str(data["candidate_id"]),
            split=str(data["split"]),
            group_id=str(data["group_id"]),
            residual_bits=float(data["residual_bits"]),
            n_examples=int(data["n_examples"]),
        )


@dataclass(frozen=True, slots=True)
class CodeLengthBreakdown:
    high_level_bits: float
    budget_bits: float
    site_bits: float
    family_bits: float
    hyperparameter_bits: float
    parameter_bits: float

    @property
    def total_structural_bits(self) -> float:
        return (
            self.high_level_bits
            + self.budget_bits
            + self.site_bits
            + self.family_bits
            + self.hyperparameter_bits
            + self.parameter_bits
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "high_level_bits": self.high_level_bits,
            "budget_bits": self.budget_bits,
            "site_bits": self.site_bits,
            "family_bits": self.family_bits,
            "hyperparameter_bits": self.hyperparameter_bits,
            "parameter_bits": self.parameter_bits,
            "total_structural_bits": self.total_structural_bits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeLengthBreakdown":
        return cls(
            high_level_bits=float(data["high_level_bits"]),
            budget_bits=float(data["budget_bits"]),
            site_bits=float(data["site_bits"]),
            family_bits=float(data["family_bits"]),
            hyperparameter_bits=float(data["hyperparameter_bits"]),
            parameter_bits=float(data["parameter_bits"]),
        )
