"""Core config and schema utilities for the MDCA scaffold."""

from .codebooks import (
    PRIMARY_CODEBOOK_ID,
    QUANTIZED_CODEBOOK_ID,
    VALID_CODEBOOK_IDS,
    build_code_length_breakdown,
    codebook_id_from_extras,
    parameter_bits,
    recode_code_length_breakdown,
    recode_scored_record,
    validate_codebook_id,
)
from .config import (
    LOCKED_METHOD_CONSTANTS,
    MethodConstants,
    PathConfig,
    RunConfig,
    RuntimeConfig,
    SeedBundle,
    load_run_config,
    save_run_config,
)
from .schemas import (
    BaseSourceTuple,
    CandidateScoredInterventionRecord,
    CodeLengthBreakdown,
    ResidualContribution,
    Site,
    TupleMetadata,
)

__all__ = [
    "BaseSourceTuple",
    "PRIMARY_CODEBOOK_ID",
    "QUANTIZED_CODEBOOK_ID",
    "VALID_CODEBOOK_IDS",
    "CandidateScoredInterventionRecord",
    "CodeLengthBreakdown",
    "LOCKED_METHOD_CONSTANTS",
    "MethodConstants",
    "PathConfig",
    "ResidualContribution",
    "RunConfig",
    "RuntimeConfig",
    "SeedBundle",
    "Site",
    "TupleMetadata",
    "build_code_length_breakdown",
    "codebook_id_from_extras",
    "load_run_config",
    "parameter_bits",
    "recode_code_length_breakdown",
    "recode_scored_record",
    "save_run_config",
    "validate_codebook_id",
]
