from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.core.config import load_run_config, save_run_config
from src.core.codebooks import codebook_id_from_extras
from src.core.search_scope import (
    attach_recorded_candidate_counts,
    candidate_pool_scope_from_extras,
    save_candidate_pool_scope,
)
from src.core.schemas import Site
from src.planted import (
    CandidateSearchEngine,
    NullSpec,
    S1_TRUE_SITE_GROUPS,
    S1PlantedModel,
    PlantedLatents,
    SearchSpec,
    build_s1_dataset_bundle,
    build_shuffled_pair_dataset_bundle,
    candidate_table_rows,
    run_candidate_like_null_search,
    run_random_site_null_search,
    shuffled_pair_is_available,
    split_manifest_records,
)


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _site_records(sites: Sequence[Site]) -> list[dict[str, int]]:
    return [site.to_dict() for site in sites]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.experiments.planted")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    if config.setting_id != "planted":
        raise ValueError(
            "python -m src.experiments.planted requires a planted setting config"
        )

    model = S1PlantedModel()
    dataset_bundle = build_s1_dataset_bundle()
    split_manifest = split_manifest_records(list(dataset_bundle.tuples))
    phase = str(config.extras.get("phase", "smoke"))
    codebook_id = codebook_id_from_extras(config.extras)

    canonical_example = next(
        tuple_record
        for tuple_record in dataset_bundle.tuples
        if tuple_record.metadata.template_family == "canonical"
        and tuple_record.intervention_type == "R"
    )
    base_latents = PlantedLatents.from_dict(canonical_example.metadata.latent_base)
    source_latents = PlantedLatents.from_dict(canonical_example.metadata.latent_source)

    true_sites = S1_TRUE_SITE_GROUPS[canonical_example.intervention_type]
    base_clean_run = model.run_clean(base_latents)
    source_clean_run = model.run_clean(source_latents)
    observed_run = model.patch_and_run(
        base_latents=base_latents,
        source_latents=source_latents,
        intervention_type=canonical_example.intervention_type,
        patch_sites=true_sites,
    )
    worked_example = {
        "split": dataset_bundle.split_by_group[canonical_example.metadata.group_id],
        "tuple_record": canonical_example.to_dict(),
        "candidate_patch_sites": _site_records(true_sites),
        "base_clean_output_token": base_clean_run.output_token,
        "source_clean_output_token": source_clean_run.output_token,
        "patched_output_token": observed_run.output_token,
    }

    run_slug = f"{_timestamp_slug()}_{config.setting_id}_{config.variant}"
    run_dir = Path(config.paths.results_dir) / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(config, run_dir / "config_snapshot.yaml")
    _json_dump(dataset_bundle.dataset_stats(), run_dir / "dataset_stats.json")
    _json_dump(split_manifest, run_dir / "split_manifest.json")
    _json_dump(model.site_table_records(), run_dir / "site_table.json")
    candidate_pool_scope = candidate_pool_scope_from_extras(config.extras)
    save_candidate_pool_scope(candidate_pool_scope, run_dir / "candidate_pool_scope.json")

    summary = {
        "status": "ok",
        "entrypoint": "python -m src.experiments.planted",
        "config_path": str(Path(args.config)),
        "run_dir": str(run_dir),
        "dataset_stats_path": str(run_dir / "dataset_stats.json"),
        "split_manifest_path": str(run_dir / "split_manifest.json"),
        "site_table_path": str(run_dir / "site_table.json"),
        "candidate_pool_scope_path": str(run_dir / "candidate_pool_scope.json"),
        "site_universe_size": len(model.site_universe),
        "true_site_groups": {
            name: _site_records(sites) for name, sites in S1_TRUE_SITE_GROUPS.items()
        },
        "candidate_pool_scope": {
            "scope_label": candidate_pool_scope["scope_label"],
            "covers_full_locked_candidate_pool": candidate_pool_scope[
                "covers_full_locked_candidate_pool"
            ],
            "configured_candidate_cells": candidate_pool_scope["configured_candidate_cells"],
            "locked_candidate_cells": candidate_pool_scope["locked_candidate_cells"],
        },
    }
    if phase == "smoke":
        _json_dump(worked_example, run_dir / "worked_example.json")
        summary["worked_example_path"] = str(run_dir / "worked_example.json")
    elif phase in {"score_only", "full"}:
        _json_dump(worked_example, run_dir / "worked_example.json")
        search_spec = SearchSpec.from_config_extras(config.extras)
        search_engine = CandidateSearchEngine(
            model=model,
            dataset_bundle=dataset_bundle,
            codebook_id=codebook_id,
            search_seed=config.seeds.candidate_search_seed,
            linear_epochs=search_spec.linear_epochs,
            mlp_epochs=search_spec.mlp_epochs,
            learning_rate=search_spec.learning_rate,
        )
        candidate_records, proposal_logs = search_engine.run_search(
            high_level_model_ids=search_spec.high_level_model_ids,
            site_budgets=search_spec.site_budgets,
            map_family_grid=search_spec.map_family_grid,
        )
        for candidate_record in candidate_records:
            candidate_record["record_kind"] = "candidate"
            candidate_record["null_family"] = None
        for proposal_log in proposal_logs:
            proposal_log["record_kind"] = "candidate"
            proposal_log["null_family"] = None
        candidate_table = candidate_table_rows(candidate_records)
        _json_dump(candidate_table, run_dir / "candidate_table.json")
        _json_dump(candidate_records, run_dir / "candidate_records.json")
        _json_dump(proposal_logs, run_dir / "proposal_logs.json")
        candidate_pool_scope = attach_recorded_candidate_counts(
            candidate_pool_scope,
            recorded_candidate_records=len(candidate_records),
            recorded_unevaluable_candidate_cells=sum(
                1 for proposal_log in proposal_logs if proposal_log["status"] != "evaluable"
            ),
        )
        save_candidate_pool_scope(candidate_pool_scope, run_dir / "candidate_pool_scope.json")
        summary.update(
            {
                "worked_example_path": str(run_dir / "worked_example.json"),
                "candidate_table_path": str(run_dir / "candidate_table.json"),
                "candidate_records_path": str(run_dir / "candidate_records.json"),
                "proposal_logs_path": str(run_dir / "proposal_logs.json"),
                "n_candidate_records": len(candidate_records),
                "n_unevaluable_cells": sum(
                    1 for proposal_log in proposal_logs if proposal_log["status"] != "evaluable"
                ),
                "candidate_pool_scope": {
                    "scope_label": candidate_pool_scope["scope_label"],
                    "covers_full_locked_candidate_pool": candidate_pool_scope[
                        "covers_full_locked_candidate_pool"
                    ],
                    "configured_candidate_cells": candidate_pool_scope[
                        "configured_candidate_cells"
                    ],
                    "recorded_total_candidate_cells": candidate_pool_scope[
                        "recorded_total_candidate_cells"
                    ],
                },
            }
        )
        if candidate_table:
            summary["best_candidate_id"] = candidate_table[0]["candidate_id"]
            summary["best_candidate_test_total_bits"] = candidate_table[0]["test_total_bits"]
        if phase == "full":
            null_spec = NullSpec.from_config_extras(
                config.extras,
                default_untrained_seed=config.seeds.model_init_seed,
            )
            null_records: list[dict[str, object]] = []
            null_logs: list[dict[str, object]] = []
            executed_null_families: list[str] = []
            skipped_null_families: dict[str, str] = {}
            if "random_site" in null_spec.null_families:
                records, logs = run_random_site_null_search(
                    model=model,
                    dataset_bundle=dataset_bundle,
                    search_spec=search_spec,
                    codebook_id=codebook_id,
                    search_seed=config.seeds.candidate_search_seed,
                )
                null_records.extend(records)
                null_logs.extend(logs)
                executed_null_families.append("random_site")
            if "shuffled_pair" in null_spec.null_families:
                if shuffled_pair_is_available(dataset_bundle):
                    shuffled_bundle = build_shuffled_pair_dataset_bundle(
                        dataset_bundle,
                        seed=null_spec.shuffled_pair_seed,
                    )
                    records, logs = run_candidate_like_null_search(
                        null_family="shuffled_pair",
                        model=model,
                        dataset_bundle=shuffled_bundle,
                        search_spec=search_spec,
                        codebook_id=codebook_id,
                        search_seed=config.seeds.candidate_search_seed,
                    )
                    null_records.extend(records)
                    null_logs.extend(logs)
                    executed_null_families.append("shuffled_pair")
                else:
                    skipped_null_families["shuffled_pair"] = "degenerate_or_identity_bundle"
            if "untrained_model" in null_spec.null_families:
                untrained_model = S1PlantedModel(plant_seed=null_spec.untrained_model_seed)
                records, logs = run_candidate_like_null_search(
                    null_family="untrained_model",
                    model=untrained_model,
                    dataset_bundle=dataset_bundle,
                    search_spec=search_spec,
                    codebook_id=codebook_id,
                    search_seed=config.seeds.candidate_search_seed,
                )
                null_records.extend(records)
                null_logs.extend(logs)
                executed_null_families.append("untrained_model")
            null_records.sort(
                key=lambda record: (
                    str(record["null_family"]),
                    float(record["test_total_bits"]),
                    str(record["candidate_id"]),
                )
            )
            null_logs.sort(
                key=lambda log: (
                    str(log["null_family"]),
                    str(log["cell_id"]),
                )
            )
            _json_dump(null_records, run_dir / "null_records.json")
            _json_dump(null_logs, run_dir / "null_search_logs.json")
            summary.update(
                {
                    "null_records_path": str(run_dir / "null_records.json"),
                    "null_search_logs_path": str(run_dir / "null_search_logs.json"),
                    "n_null_records": len(null_records),
                    "requested_primary_null_families": list(null_spec.null_families),
                    "available_primary_null_families": executed_null_families,
                    "skipped_null_families": skipped_null_families,
                }
            )
    else:
        raise ValueError(f"Unsupported planted phase: {phase!r}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
