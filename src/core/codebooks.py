from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .schemas import CodeLengthBreakdown

PRIMARY_CODEBOOK_ID = "primary"
QUANTIZED_CODEBOOK_ID = "quantized"
VALID_CODEBOOK_IDS = (PRIMARY_CODEBOOK_ID, QUANTIZED_CODEBOOK_ID)


def validate_codebook_id(codebook_id: str) -> str:
    normalized = str(codebook_id).strip().lower()
    if normalized not in VALID_CODEBOOK_IDS:
        raise ValueError(
            f"codebook_id must be one of {list(VALID_CODEBOOK_IDS)}, got {codebook_id!r}"
        )
    return normalized


def codebook_id_from_extras(extras: dict[str, Any]) -> str:
    return validate_codebook_id(str(extras.get("codebook", PRIMARY_CODEBOOK_ID)))


def parameter_bits(
    *,
    parameter_count_eff: int,
    n_train_tuples: int,
    codebook_id: str,
) -> float:
    validated_codebook = validate_codebook_id(codebook_id)
    if validated_codebook == PRIMARY_CODEBOOK_ID:
        return 0.5 * float(parameter_count_eff) * math.log2(max(int(n_train_tuples), 2))
    return 16.0 * float(parameter_count_eff)


def build_code_length_breakdown(
    *,
    high_level_bits: float,
    budget_bits: float,
    site_bits: float,
    family_bits: float,
    hyperparameter_bits: float,
    parameter_count_eff: int,
    n_train_tuples: int,
    codebook_id: str,
) -> CodeLengthBreakdown:
    return CodeLengthBreakdown(
        high_level_bits=float(high_level_bits),
        budget_bits=float(budget_bits),
        site_bits=float(site_bits),
        family_bits=float(family_bits),
        hyperparameter_bits=float(hyperparameter_bits),
        parameter_bits=parameter_bits(
            parameter_count_eff=parameter_count_eff,
            n_train_tuples=n_train_tuples,
            codebook_id=codebook_id,
        ),
    )


def recode_code_length_breakdown(
    *,
    code_lengths: CodeLengthBreakdown | dict[str, Any],
    parameter_count_eff: int,
    n_train_tuples: int,
    codebook_id: str,
) -> CodeLengthBreakdown:
    breakdown = (
        code_lengths
        if isinstance(code_lengths, CodeLengthBreakdown)
        else CodeLengthBreakdown.from_dict(dict(code_lengths))
    )
    return build_code_length_breakdown(
        high_level_bits=breakdown.high_level_bits,
        budget_bits=breakdown.budget_bits,
        site_bits=breakdown.site_bits,
        family_bits=breakdown.family_bits,
        hyperparameter_bits=breakdown.hyperparameter_bits,
        parameter_count_eff=int(parameter_count_eff),
        n_train_tuples=int(n_train_tuples),
        codebook_id=codebook_id,
    )


def recode_scored_record(
    *,
    record: dict[str, Any],
    n_train_tuples: int,
    codebook_id: str,
) -> dict[str, Any]:
    updated = deepcopy(record)
    code_lengths = recode_code_length_breakdown(
        code_lengths=updated["code_lengths"],
        parameter_count_eff=int(updated["parameter_count_eff"]),
        n_train_tuples=n_train_tuples,
        codebook_id=codebook_id,
    )
    updated["code_lengths"] = code_lengths.to_dict()
    updated["test_total_bits"] = (
        float(updated["residual_bits"]["test"]) + code_lengths.total_structural_bits
    )
    split_sizes = dict(updated.get("split_sizes", {}))
    test_size = int(split_sizes.get("test", 0))
    updated["test_total_bits_per_example"] = updated["test_total_bits"] / max(test_size, 1)
    return updated
