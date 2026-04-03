from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from src.analysis.acceptance_core import compute_support
from src.analysis.frontier_core import build_balanced_frontier
from src.planted import (
    CandidateSearchEngine,
    S1PlantedModel,
    build_s1_dataset_bundle,
    build_shuffled_pair_dataset_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shuffled_pair_bundle_only_changes_canonical_train_val() -> None:
    original = build_s1_dataset_bundle()
    shuffled = build_shuffled_pair_dataset_bundle(original, seed=0)

    original_by_split = original.tuples_by_split
    shuffled_by_split = shuffled.tuples_by_split

    for split_name in ("test", "shift"):
        original_records = sorted(
            (record.to_dict() for record in original_by_split[split_name]),
            key=lambda record: record["tuple_id"],
        )
        shuffled_records = sorted(
            (record.to_dict() for record in shuffled_by_split[split_name]),
            key=lambda record: record["tuple_id"],
        )
        assert shuffled_records == original_records

    changed_train_val = False
    for split_name in ("train", "val"):
        original_sources = [record.metadata.latent_source for record in original_by_split[split_name]]
        shuffled_sources = [record.metadata.latent_source for record in shuffled_by_split[split_name]]
        if original_sources != shuffled_sources:
            changed_train_val = True
            break
    assert changed_train_val


def test_shuffled_pair_changes_tuple_conditioned_search_inputs() -> None:
    model = S1PlantedModel()
    original = build_s1_dataset_bundle()
    shuffled = build_shuffled_pair_dataset_bundle(original, seed=0)
    site = model.site_universe[0]

    original_engine = CandidateSearchEngine(model=model, dataset_bundle=original)
    shuffled_engine = CandidateSearchEngine(model=model, dataset_bundle=shuffled)

    original_features, original_labels = original_engine._dataset_for_variable(
        split_name="train",
        variable_name="N1",
        sites=(site,),
    )
    shuffled_features, shuffled_labels = shuffled_engine._dataset_for_variable(
        split_name="train",
        variable_name="N1",
        sites=(site,),
    )

    assert not torch.equal(original_features, shuffled_features)
    assert not torch.equal(original_labels, shuffled_labels)

    original_rankings = original_engine._single_site_rankings(
        variable_name="N1",
        map_family_id="linear_dense",
        hyperparameter_value="default_dense",
    )
    shuffled_rankings = shuffled_engine._single_site_rankings(
        variable_name="N1",
        map_family_id="linear_dense",
        hyperparameter_value="default_dense",
    )
    assert original_rankings != shuffled_rankings


def test_frontier_builder_balances_families_and_sets_domain() -> None:
    null_records = []
    families = ("random_site", "shuffled_pair", "untrained_model")
    for family in families:
        for index in range(6):
            null_records.append(
                {
                    "candidate_id": f"{family}_{index}",
                    "null_family": family,
                    "code_lengths": {"total_structural_bits": 10.2 + 0.01 * index},
                    "residual_bits": {"test": 100.0 + index, "shift": 110.0 + index},
                }
            )
        for index in range(4):
            null_records.append(
                {
                    "candidate_id": f"{family}_invalid_{index}",
                    "null_family": family,
                    "code_lengths": {"total_structural_bits": 20.2 + 0.01 * index},
                    "residual_bits": {"test": 120.0 + index, "shift": 130.0 + index},
                }
            )

    frontier = build_balanced_frontier(
        null_records=null_records,
        split="test",
        bin_width_bits=2,
        min_family_count=5,
        balance_seed=0,
    )

    valid_bins = [summary for summary in frontier.bin_summaries if summary["valid"]]
    assert len(valid_bins) == 1
    assert frontier.domain == (11.0, 11.0)
    assert len(valid_bins[0]["selected_null_ids_by_family"]["random_site"]) == 6
    assert frontier.evaluate(11.0) is not None
    assert frontier.evaluate(21.0) is None


def test_frontier_single_valid_bin_uses_bin_membership_not_center_only() -> None:
    null_records = []
    for family in ("random_site", "shuffled_pair", "untrained_model"):
        for index in range(6):
            null_records.append(
                {
                    "candidate_id": f"{family}_{index}",
                    "null_family": family,
                    "code_lengths": {"total_structural_bits": 10.2 + 0.01 * index},
                    "residual_bits": {"test": 100.0 + index, "shift": 110.0 + index},
                }
            )

    frontier = build_balanced_frontier(
        null_records=null_records,
        split="test",
        bin_width_bits=2,
        min_family_count=5,
        balance_seed=0,
    )

    assert frontier.domain == (11.0, 11.0)
    assert frontier.defined_at(10.25)
    assert frontier.defined_at(11.95)
    assert frontier.evaluate(10.25) == frontier.evaluate(11.95)
    assert not frontier.defined_at(12.001)
    assert frontier.evaluate(12.001) is None


def test_support_candidate_in_valid_bin_is_not_frontier_undefined() -> None:
    null_records = []
    for family_index, family in enumerate(("random_site", "shuffled_pair", "untrained_model")):
        for index in range(6):
            null_records.append(
                {
                    "candidate_id": f"{family}_{index}",
                    "null_family": family,
                    "code_lengths": {"total_structural_bits": 10.2 + 0.01 * index},
                    "residual_bits": {
                        "test": 10.0 + family_index + 0.1 * index,
                        "shift": 10.0 + family_index + 0.1 * index,
                    },
                    "residual_contributions": {
                        "test": [
                            {"group_id": "g1", "residual_bits": 5.0 + family_index},
                            {"group_id": "g2", "residual_bits": 5.0 + 0.1 * index},
                        ]
                    },
                }
            )

    candidate_records = [
        {
            "candidate_id": "candidate_1",
            "high_level_model_id": "H_true_other",
            "map_family_id": "linear_dense",
            "hyperparameter_id": "default_dense",
            "site_budget": 2,
            "test_total_bits_per_example": 1.0,
            "code_lengths": {"total_structural_bits": 10.25},
            "residual_bits": {"test": 0.5, "shift": 0.5},
            "residual_contributions": {
                "test": [
                    {"group_id": "g1", "residual_bits": 0.2},
                    {"group_id": "g2", "residual_bits": 0.3},
                ]
            },
        }
    ]

    frontier_test = build_balanced_frontier(
        null_records=null_records,
        split="test",
        bin_width_bits=2,
        min_family_count=5,
        balance_seed=0,
    )
    frontier_shift = build_balanced_frontier(
        null_records=null_records,
        split="shift",
        bin_width_bits=2,
        min_family_count=5,
        balance_seed=0,
    )

    support = compute_support(
        candidate_records=candidate_records,
        null_records=null_records,
        frontier_test=frontier_test,
        frontier_shift=frontier_shift,
        bootstrap_reps=32,
        bootstrap_seed=0,
        test_group_ids=["g1", "g2"],
    )

    row = support.support_table[0]
    assert row["frontier_defined_test"] is True
    assert row["frontier_defined_shift"] is True
    assert row["support_reason"] != "frontier_undefined"


def test_support_summary_reports_frontier_ineligible_global_best() -> None:
    null_records = []
    for family_index, family in enumerate(("random_site", "shuffled_pair", "untrained_model")):
        for index in range(6):
            null_records.append(
                {
                    "candidate_id": f"{family}_{index}",
                    "null_family": family,
                    "code_lengths": {"total_structural_bits": 10.2 + 0.01 * index},
                    "residual_bits": {
                        "test": 10.0 + family_index + 0.1 * index,
                        "shift": 10.0 + family_index + 0.1 * index,
                    },
                    "residual_contributions": {
                        "test": [
                            {"group_id": "g1", "residual_bits": 5.0 + family_index},
                            {"group_id": "g2", "residual_bits": 5.0 + 0.1 * index},
                        ]
                    },
                }
            )

    candidate_records = [
        {
            "candidate_id": "candidate_best",
            "high_level_model_id": "H_rep",
            "map_family_id": "linear_dense",
            "hyperparameter_id": "default_dense",
            "site_budget": 1,
            "test_total_bits_per_example": 1.0,
            "code_lengths": {"total_structural_bits": 12.25},
            "residual_bits": {"test": 0.5, "shift": 0.5},
            "residual_contributions": {
                "test": [
                    {"group_id": "g1", "residual_bits": 0.2},
                    {"group_id": "g2", "residual_bits": 0.3},
                ]
            },
        },
        {
            "candidate_id": "candidate_frontier",
            "high_level_model_id": "H_true_other",
            "map_family_id": "linear_dense",
            "hyperparameter_id": "default_dense",
            "site_budget": 1,
            "test_total_bits_per_example": 1.05,
            "code_lengths": {"total_structural_bits": 10.25},
            "residual_bits": {"test": 0.5, "shift": 0.5},
            "residual_contributions": {
                "test": [
                    {"group_id": "g1", "residual_bits": 0.2},
                    {"group_id": "g2", "residual_bits": 0.3},
                ]
            },
        },
    ]

    frontier_test = build_balanced_frontier(
        null_records=null_records,
        split="test",
        bin_width_bits=2,
        min_family_count=5,
        balance_seed=0,
    )
    frontier_shift = build_balanced_frontier(
        null_records=null_records,
        split="shift",
        bin_width_bits=2,
        min_family_count=5,
        balance_seed=0,
    )

    support = compute_support(
        candidate_records=candidate_records,
        null_records=null_records,
        frontier_test=frontier_test,
        frontier_shift=frontier_shift,
        bootstrap_reps=32,
        bootstrap_seed=0,
        test_group_ids=["g1", "g2"],
    )

    assert support.summary["best_candidate_id"] == "candidate_best"
    assert support.summary["best_candidate_frontier_eligible"] is False
    assert support.summary["best_candidate_support_reason"] == "frontier_undefined"
    assert support.summary["frontier_defined_candidate_count"] == 1
    assert support.summary["best_frontier_defined_candidate_id"] == "candidate_frontier"
    assert support.summary["best_frontier_defined_candidate_within_best_bits"] is False


def test_full_null_frontier_and_acceptance_clis_smoke() -> None:
    experiment = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments.planted",
            "--config",
            "configs/planted/test_full.yaml",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert experiment.returncode == 0, experiment.stderr
    experiment_payload = json.loads(experiment.stdout)
    run_dir = experiment_payload["run_dir"]

    frontier = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.null_frontier",
            "--run-dir",
            run_dir,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert frontier.returncode == 0, frontier.stderr
    frontier_payload = json.loads(frontier.stdout)
    assert frontier_payload["status"] == "ok"

    acceptance = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.acceptance",
            "--run-dir",
            run_dir,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert acceptance.returncode == 0, acceptance.stderr
    acceptance_payload = json.loads(acceptance.stdout)
    assert acceptance_payload["status"] == "ok"
    assert Path(REPO_ROOT / acceptance_payload["support_table_path"]).exists() or Path(
        acceptance_payload["support_table_path"]
    ).exists()

    oracle = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.planted_oracle",
            "--run-dir",
            run_dir,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert oracle.returncode == 0, oracle.stderr
    oracle_payload = json.loads(oracle.stdout)
    assert oracle_payload["status"] == "ok"
    assert Path(REPO_ROOT / oracle_payload["oracle_support_path"]).exists() or Path(
        oracle_payload["oracle_support_path"]
    ).exists()
