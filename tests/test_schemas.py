from __future__ import annotations

from src.core.schemas import (
    BaseSourceTuple,
    CandidateScoredInterventionRecord,
    CodeLengthBreakdown,
    ResidualContribution,
    TupleMetadata,
)


def test_tuple_and_candidate_record_round_trip() -> None:
    metadata = TupleMetadata(
        group_id="group-001",
        template_family="canonical",
        latent_base={"N1": "Alice", "N2": "Bob", "R": 1},
        latent_source={"N1": "Alice", "N2": "Bob", "R": 2},
        prompt_id_base="base-001",
        prompt_id_source="source-001",
        extra={"setting_id": "planted"},
    )
    tuple_record = BaseSourceTuple(
        tuple_id="tuple-001",
        base_input={"kind": "state", "payload": [1, 2, 3]},
        source_input={"kind": "state", "payload": [1, 2, 4]},
        intervention_type="R",
        metadata=metadata,
    )
    candidate_record = CandidateScoredInterventionRecord(
        candidate_id="candidate-001",
        split="val",
        tuple_record=tuple_record,
        observed_output_token="Bob",
        predicted_output_token="Alice",
    )

    restored = CandidateScoredInterventionRecord.from_dict(candidate_record.to_dict())
    assert restored == candidate_record


def test_residual_and_code_length_round_trip() -> None:
    residual = ResidualContribution(
        candidate_id="candidate-001",
        split="test",
        group_id="group-001",
        residual_bits=12.5,
        n_examples=4,
    )
    code_lengths = CodeLengthBreakdown(
        high_level_bits=2.0,
        budget_bits=1.5849625007,
        site_bits=10.0,
        family_bits=2.0,
        hyperparameter_bits=1.5849625007,
        parameter_bits=8.0,
    )

    restored_residual = ResidualContribution.from_dict(residual.to_dict())
    restored_code_lengths = CodeLengthBreakdown.from_dict(code_lengths.to_dict())

    assert restored_residual == residual
    assert restored_code_lengths == code_lengths
    assert restored_code_lengths.total_structural_bits == code_lengths.total_structural_bits
