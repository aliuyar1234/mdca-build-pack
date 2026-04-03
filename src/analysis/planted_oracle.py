from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.core.config import load_run_config
from src.planted import (
    CandidateSearchEngine,
    S1PlantedModel,
    S1_TRUE_SITE_GROUPS,
    SearchSpec,
    build_s1_dataset_bundle,
)

from .acceptance_core import compute_support
from .frontier_core import FrontierResult, load_json, save_json

VARIABLE_ORDER = ("N1", "N2", "R")


def _site_key(site_record: dict[str, Any]) -> tuple[int, int]:
    return (int(site_record["layer_index"]), int(site_record["token_index"]))


def _site_keys(site_records: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((_site_key(site_record) for site_record in site_records)))


def _site_overlap_report(
    candidate_site_groups: dict[str, list[dict[str, int]]],
    true_site_groups: dict[str, list[dict[str, int]]],
) -> dict[str, Any]:
    by_variable: dict[str, Any] = {}
    total_overlap_count = 0
    exact_site_match = True
    for variable_name in VARIABLE_ORDER:
        candidate_sites = set(_site_keys(candidate_site_groups[variable_name]))
        true_sites = set(_site_keys(true_site_groups[variable_name]))
        overlap = sorted(candidate_sites & true_sites)
        total_overlap_count += len(overlap)
        variable_exact = candidate_sites == true_sites
        exact_site_match = exact_site_match and variable_exact
        by_variable[variable_name] = {
            "candidate_sites": [
                {"layer_index": layer_index, "token_index": token_index}
                for layer_index, token_index in sorted(candidate_sites)
            ],
            "true_sites": [
                {"layer_index": layer_index, "token_index": token_index}
                for layer_index, token_index in sorted(true_sites)
            ],
            "overlap_sites": [
                {"layer_index": layer_index, "token_index": token_index}
                for layer_index, token_index in overlap
            ],
            "overlap_count": len(overlap),
            "exact_site_match": variable_exact,
        }
    return {
        "exact_site_match": exact_site_match,
        "total_overlap_count": total_overlap_count,
        "by_variable": by_variable,
    }


def _true_site_groups_payload() -> dict[str, list[dict[str, int]]]:
    return {
        variable_name: [site.to_dict() for site in S1_TRUE_SITE_GROUPS[variable_name]]
        for variable_name in VARIABLE_ORDER
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.planted_oracle")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = load_run_config(run_dir / "config_snapshot.yaml")
    if config.setting_id != "planted":
        raise ValueError("python -m src.analysis.planted_oracle requires a planted run")

    search_spec = SearchSpec.from_config_extras(config.extras)
    dataset_bundle = build_s1_dataset_bundle()
    model = S1PlantedModel()
    search_engine = CandidateSearchEngine(
        model=model,
        dataset_bundle=dataset_bundle,
        search_seed=config.seeds.candidate_search_seed,
        linear_epochs=search_spec.linear_epochs,
        mlp_epochs=search_spec.mlp_epochs,
        learning_rate=search_spec.learning_rate,
    )

    oracle_record = search_engine.score_fixed_candidate(
        high_level_model_id="H_true_other",
        map_family_id="linear_dense",
        hyperparameter_value="default_dense",
        site_groups=S1_TRUE_SITE_GROUPS,
    )
    oracle_record["record_kind"] = "oracle"
    oracle_record["oracle_id"] = "A_star"

    frontier_test = FrontierResult.from_dict(load_json(run_dir / "frontier_test.json"))
    frontier_shift = FrontierResult.from_dict(load_json(run_dir / "frontier_shift.json"))
    null_records = load_json(run_dir / "null_records.json")
    split_manifest = load_json(run_dir / "split_manifest.json")
    test_group_ids = sorted(
        {
            str(record["group_id"])
            for record in split_manifest
            if str(record["split"]) == "test"
        }
    )
    oracle_support = compute_support(
        candidate_records=[oracle_record],
        null_records=null_records,
        frontier_test=frontier_test,
        frontier_shift=frontier_shift,
        bootstrap_reps=config.method.bootstrap_n_reps,
        bootstrap_seed=config.seeds.bootstrap_seed,
        test_group_ids=test_group_ids,
    )
    oracle_support_row = oracle_support.support_table[0]
    structural_bits = float(oracle_record["code_lengths"]["total_structural_bits"])
    oracle_support_payload = {
        "oracle_id": "A_star",
        "frontier_test_value": frontier_test.evaluate(structural_bits),
        "frontier_shift_value": frontier_shift.evaluate(structural_bits),
        "test_bin_start": frontier_test.structural_bin_start(structural_bits),
        "shift_bin_start": frontier_shift.structural_bin_start(structural_bits),
        "support_summary": oracle_support.summary,
        "support_row": oracle_support_row,
    }

    true_site_groups = _true_site_groups_payload()
    candidate_records = load_json(run_dir / "candidate_records.json")
    overlap_records = []
    for candidate_record in sorted(
        candidate_records,
        key=lambda record: (float(record["test_total_bits_per_example"]), str(record["candidate_id"])),
    ):
        overlap_records.append(
            {
                "candidate_id": candidate_record["candidate_id"],
                "high_level_model_id": candidate_record["high_level_model_id"],
                "map_family_id": candidate_record["map_family_id"],
                "hyperparameter_id": candidate_record["hyperparameter_id"],
                "site_budget": candidate_record["site_budget"],
                "test_total_bits_per_example": candidate_record["test_total_bits_per_example"],
                "site_overlap": _site_overlap_report(
                    candidate_record["site_groups"],
                    true_site_groups,
                ),
            }
        )
    oracle_overlap_payload = {
        "oracle_id": "A_star",
        "oracle_site_overlap": _site_overlap_report(oracle_record["site_groups"], true_site_groups),
        "top_candidates_by_test_total_bits_per_example": overlap_records[:8],
        "all_h_true_other_candidates": [
            record for record in overlap_records if record["high_level_model_id"] == "H_true_other"
        ],
    }

    save_json(oracle_record, run_dir / "oracle_candidate_record.json")
    save_json(oracle_support_payload, run_dir / "oracle_support.json")
    save_json(oracle_overlap_payload, run_dir / "oracle_site_overlap.json")

    payload = {
        "status": "ok",
        "entrypoint": "python -m src.analysis.planted_oracle",
        "run_dir": str(run_dir),
        "oracle_candidate_record_path": str(run_dir / "oracle_candidate_record.json"),
        "oracle_support_path": str(run_dir / "oracle_support.json"),
        "oracle_site_overlap_path": str(run_dir / "oracle_site_overlap.json"),
        "oracle_candidate_id": oracle_record["candidate_id"],
        "oracle_support_reason": oracle_support_row["support_reason"],
        "oracle_frontier_defined_test": oracle_support_row["frontier_defined_test"],
        "oracle_frontier_defined_shift": oracle_support_row["frontier_defined_shift"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
