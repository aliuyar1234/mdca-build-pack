from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from src.core.codebooks import QUANTIZED_CODEBOOK_ID
from src.planted import S1_TRUE_SITE_GROUPS, S1PlantedModel, build_s1_dataset_bundle
from src.planted.high_level import HIGH_LEVEL_MODELS
from src.planted.scoring import CandidateSearchEngine


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_structural_code_lengths_match_locked_formula() -> None:
    model = S1PlantedModel()
    engine = CandidateSearchEngine(model=model, dataset_bundle=build_s1_dataset_bundle())

    parameter_count = 50
    breakdown = engine._structural_code_lengths(
        site_budget=2,
        map_family_id="linear_dense",
        site_groups=S1_TRUE_SITE_GROUPS,
        parameter_count=parameter_count,
    )

    expected_site_bits = (
        math.log2(math.comb(len(model.site_universe), 2))
        + math.log2(math.comb(len(model.site_universe) - 2, 2))
        + math.log2(math.comb(len(model.site_universe) - 4, 2))
    )
    expected_total = (
        math.log2(4)
        + math.log2(3)
        + expected_site_bits
        + math.log2(4)
        + math.log2(1)
        + 0.5 * parameter_count * math.log2(engine.n_train_tuples)
    )

    assert math.isclose(breakdown.site_bits, expected_site_bits, rel_tol=1e-9)
    assert math.isclose(breakdown.total_structural_bits, expected_total, rel_tol=1e-9)


def test_structural_code_lengths_quantized_formula() -> None:
    model = S1PlantedModel()
    engine = CandidateSearchEngine(
        model=model,
        dataset_bundle=build_s1_dataset_bundle(),
        codebook_id=QUANTIZED_CODEBOOK_ID,
    )

    parameter_count = 7
    breakdown = engine._structural_code_lengths(
        site_budget=2,
        map_family_id="linear_dense",
        site_groups=S1_TRUE_SITE_GROUPS,
        parameter_count=parameter_count,
    )

    expected_site_bits = (
        math.log2(math.comb(len(model.site_universe), 2))
        + math.log2(math.comb(len(model.site_universe) - 2, 2))
        + math.log2(math.comb(len(model.site_universe) - 4, 2))
    )
    expected_total = (
        math.log2(4)
        + math.log2(3)
        + expected_site_bits
        + math.log2(4)
        + math.log2(1)
        + 16.0 * parameter_count
    )

    assert math.isclose(breakdown.parameter_bits, 16.0 * parameter_count, rel_tol=1e-9)
    assert math.isclose(breakdown.total_structural_bits, expected_total, rel_tol=1e-9)


def test_residual_code_matches_locked_symmetric_noise_model() -> None:
    model = S1PlantedModel()
    engine = CandidateSearchEngine(model=model, dataset_bundle=build_s1_dataset_bundle())
    behavior = engine.build_candidate_behavior(
        map_family_id="linear_dense",
        hyperparameter_value="default_dense",
        site_groups=S1_TRUE_SITE_GROUPS,
    )
    record = engine.score_candidate_behavior(
        high_level_model_id="H_true_other",
        site_budget=2,
        behavior=behavior,
    )

    matches = 0
    mismatches = 0
    high_level_model = HIGH_LEVEL_MODELS["H_true_other"]
    for tuple_behavior in behavior.tuple_behaviors_by_split["val"]:
        effective_states = dict(tuple_behavior.base_states)
        intervention_type = tuple_behavior.tuple_record.intervention_type
        effective_states[intervention_type] = tuple_behavior.source_states[intervention_type]
        predicted_token = model.name_vocab[
            high_level_model.predict_index(
                effective_states["N1"],
                effective_states["N2"],
                effective_states["R"],
            )
        ]
        if predicted_token == tuple_behavior.observed_output_token:
            matches += 1
        else:
            mismatches += 1

    epsilon = record["epsilon"]
    expected_bits = (
        matches * (-math.log2(1 - epsilon))
        + mismatches * (-math.log2(epsilon / (len(model.name_vocab) - 1)))
    )
    assert math.isclose(record["residual_bits"]["val"], expected_bits, rel_tol=1e-9)


def test_candidate_search_end_to_end_returns_candidate_and_logs() -> None:
    model = S1PlantedModel()
    engine = CandidateSearchEngine(model=model, dataset_bundle=build_s1_dataset_bundle())

    candidate_records, proposal_logs = engine.run_search(
        high_level_model_ids=("H_true_other",),
        site_budgets=(2,),
        map_family_grid={"linear_dense": ("default_dense",)},
    )

    assert candidate_records
    assert proposal_logs
    assert candidate_records[0]["site_budget"] == 2
    assert candidate_records[0]["high_level_model_id"] == "H_true_other"
    assert proposal_logs[0]["n_raw_full_proposals"] <= 64
    assert proposal_logs[0]["n_valid_full_proposals"] <= proposal_logs[0]["n_raw_full_proposals"]
    assert len(candidate_records[0]["residual_contributions"]["test"]) == candidate_records[0]["split_sizes"]["test"]


def test_score_fixed_candidate_returns_planted_record() -> None:
    model = S1PlantedModel()
    engine = CandidateSearchEngine(model=model, dataset_bundle=build_s1_dataset_bundle())

    record = engine.score_fixed_candidate(
        high_level_model_id="H_true_other",
        map_family_id="linear_dense",
        hyperparameter_value="default_dense",
        site_groups=S1_TRUE_SITE_GROUPS,
    )

    assert record["site_budget"] == 2
    assert record["high_level_model_id"] == "H_true_other"
    assert record["site_groups"]["N1"] == [site.to_dict() for site in S1_TRUE_SITE_GROUPS["N1"]]
    assert "test" in record["residual_bits"]
    assert "shift" in record["residual_bits"]


def test_planted_score_cli_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments.planted",
            "--config",
            "configs/planted/test_score.yaml",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["n_candidate_records"] > 0
    assert Path(REPO_ROOT / payload["candidate_table_path"]).exists() or Path(payload["candidate_table_path"]).exists()
