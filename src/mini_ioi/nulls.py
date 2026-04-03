from __future__ import annotations

import random
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from src.core.codebooks import PRIMARY_CODEBOOK_ID
from src.core.schemas import BaseSourceTuple, TupleMetadata

from .data import S2DatasetBundle
from .model import FAMILY_CANONICAL, MiniIOITransformer
from .scoring import (
    MiniIOICandidateSearchEngine,
    SearchSpec,
    _is_disjoint,
    _normalize_sites,
    _serialize_site_groups,
    _site_groups_key,
)
from src.planted.readouts import hyperparameter_id

SHUFFLED_PAIR_SEED = 0
PRIMARY_NULL_FAMILIES = ("random_site", "shuffled_pair", "untrained_model")


def _stable_digest(*parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()


def _copy_tuple_with_source(
    tuple_record: BaseSourceTuple,
    *,
    source_tuple: BaseSourceTuple,
    split_name: str,
) -> BaseSourceTuple:
    base_latents = dict(tuple_record.metadata.latent_base)
    source_latents = dict(source_tuple.metadata.latent_source)
    group_id = _stable_digest(
        "shuffled_pair",
        split_name,
        tuple_record.metadata.template_family,
        tuple_record.intervention_type,
        tuple_record.metadata.group_id,
        source_tuple.metadata.group_id,
    )
    metadata = TupleMetadata(
        group_id=group_id,
        template_family=tuple_record.metadata.template_family,
        latent_base=base_latents,
        latent_source=source_latents,
        prompt_id_base=tuple_record.metadata.prompt_id_base,
        prompt_id_source=source_tuple.metadata.prompt_id_source,
        extra={
            **dict(tuple_record.metadata.extra),
            "shuffled_pair_original_group_id": tuple_record.metadata.group_id,
            "shuffled_pair_source_group_id": source_tuple.metadata.group_id,
        },
    )
    return BaseSourceTuple(
        tuple_id=f"shuffled_{group_id}",
        base_input=dict(tuple_record.base_input),
        source_input=dict(source_tuple.source_input),
        intervention_type=tuple_record.intervention_type,
        metadata=metadata,
    )


def _shuffled_permutation(
    records: list[BaseSourceTuple],
    *,
    rng: random.Random,
) -> list[int]:
    permutation = list(range(len(records)))
    rng.shuffle(permutation)
    if len(records) > 1 and all(index == value for index, value in enumerate(permutation)):
        permutation = permutation[1:] + permutation[:1]
    return permutation


def shuffled_pair_is_available(dataset_bundle: S2DatasetBundle) -> bool:
    for split_name in ("train", "val"):
        strata_sizes: dict[tuple[str, str], int] = {}
        for tuple_record in dataset_bundle.tuples_by_split[split_name]:
            if tuple_record.metadata.template_family != FAMILY_CANONICAL:
                continue
            key = (
                tuple_record.metadata.template_family,
                tuple_record.intervention_type,
            )
            strata_sizes[key] = strata_sizes.get(key, 0) + 1
        if any(size > 1 for size in strata_sizes.values()):
            return True
    return False


def build_shuffled_pair_dataset_bundle(
    dataset_bundle: S2DatasetBundle,
    *,
    seed: int = SHUFFLED_PAIR_SEED,
) -> S2DatasetBundle:
    tuples_by_split = dataset_bundle.tuples_by_split
    shuffled_records: list[BaseSourceTuple] = []
    split_by_group: dict[str, str] = {}

    for split_name in ("train", "val"):
        strata: dict[tuple[str, str], list[BaseSourceTuple]] = {}
        for tuple_record in tuples_by_split[split_name]:
            if tuple_record.metadata.template_family != FAMILY_CANONICAL:
                shuffled_records.append(tuple_record)
                split_by_group[tuple_record.metadata.group_id] = split_name
                continue
            key = (
                tuple_record.metadata.template_family,
                tuple_record.intervention_type,
            )
            strata.setdefault(key, []).append(tuple_record)

        for key, records in sorted(strata.items()):
            rng = random.Random(
                int(_stable_digest("shuffled_pair", seed, split_name, *key)[:12], 16)
            )
            permutation = _shuffled_permutation(records, rng=rng)
            for index, tuple_record in enumerate(records):
                source_tuple = records[permutation[index]]
                shuffled_record = _copy_tuple_with_source(
                    tuple_record,
                    source_tuple=source_tuple,
                    split_name=split_name,
                )
                shuffled_records.append(shuffled_record)
                split_by_group[shuffled_record.metadata.group_id] = split_name

    for split_name in ("test", "shift"):
        for tuple_record in tuples_by_split[split_name]:
            shuffled_records.append(tuple_record)
            split_by_group[tuple_record.metadata.group_id] = split_name

    shuffled_records.sort(
        key=lambda record: (split_by_group[record.metadata.group_id], record.tuple_id)
    )
    return S2DatasetBundle(tuples=tuple(shuffled_records), split_by_group=split_by_group)


@dataclass(frozen=True, slots=True)
class NullSpec:
    null_families: tuple[str, ...]
    shuffled_pair_seed: int
    untrained_model_seed: int

    @classmethod
    def from_config_extras(
        cls,
        extras: dict[str, Any],
        *,
        default_untrained_seed: int,
    ) -> "NullSpec":
        raw_nulls = extras.get("null_families", list(PRIMARY_NULL_FAMILIES))
        if not isinstance(raw_nulls, list):
            raise TypeError("extras.null_families must be a list when present")
        null_families = tuple(str(value) for value in raw_nulls)
        for null_family in null_families:
            if null_family not in PRIMARY_NULL_FAMILIES:
                raise ValueError(f"Unknown null family: {null_family}")
        return cls(
            null_families=null_families,
            shuffled_pair_seed=int(extras.get("shuffled_pair_seed", SHUFFLED_PAIR_SEED)),
            untrained_model_seed=int(extras.get("untrained_model_seed", default_untrained_seed)),
        )


def _annotate_records(
    records: list[dict[str, Any]],
    *,
    record_kind: str,
    null_family: str | None,
) -> list[dict[str, Any]]:
    for record in records:
        if null_family is not None:
            original_id = str(record["candidate_id"])
            record["candidate_id"] = f"{null_family}__{original_id}"
            for split_name in ("val", "test", "shift"):
                for contribution in record["residual_contributions"][split_name]:
                    contribution["candidate_id"] = record["candidate_id"]
            if "cell_id" in record:
                record["cell_id"] = f"{null_family}__{record['cell_id']}"
        record["record_kind"] = record_kind
        record["null_family"] = null_family
    return records


def _annotate_logs(
    logs: list[dict[str, Any]],
    *,
    record_kind: str,
    null_family: str | None,
) -> list[dict[str, Any]]:
    for log in logs:
        if null_family is not None and "cell_id" in log:
            log["cell_id"] = f"{null_family}__{log['cell_id']}"
        log["record_kind"] = record_kind
        log["null_family"] = null_family
    return logs


def run_random_site_null_search(
    *,
    model: MiniIOITransformer,
    dataset_bundle: S2DatasetBundle,
    search_spec: SearchSpec,
    codebook_id: str = PRIMARY_CODEBOOK_ID,
    search_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(model.site_universe) < 3 * max(search_spec.site_budgets):
        raise ValueError("random_site null invalid because |U_s| < 3b")

    engine = MiniIOICandidateSearchEngine(
        model=model,
        dataset_bundle=dataset_bundle,
        codebook_id=codebook_id,
        search_seed=search_seed,
        linear_epochs=search_spec.linear_epochs,
        mlp_epochs=search_spec.mlp_epochs,
        learning_rate=search_spec.learning_rate,
    )
    null_records: list[dict[str, Any]] = []
    null_logs: list[dict[str, Any]] = []
    site_universe = tuple(model.site_universe)

    for map_family_id, hyper_values in search_spec.map_family_grid.items():
        for hyperparameter_value in hyper_values:
            hyper_id = hyperparameter_id(map_family_id, hyperparameter_value)
            for site_budget in search_spec.site_budgets:
                rng = random.Random(
                    int(
                        _stable_digest(
                            "random_site",
                            search_seed,
                            map_family_id,
                            hyper_id,
                            site_budget,
                        )[:12],
                        16,
                    )
                )
                sampled_behaviors: list[tuple[dict[str, tuple[Any, ...]], Any]] = []
                for _ in range(64):
                    sampled_sites = rng.sample(site_universe, 3 * site_budget)
                    site_groups = {
                        "N1": _normalize_sites(sampled_sites[:site_budget]),
                        "N2": _normalize_sites(sampled_sites[site_budget : 2 * site_budget]),
                        "R": _normalize_sites(sampled_sites[2 * site_budget :]),
                    }
                    if not _is_disjoint(site_groups):
                        raise RuntimeError("random_site sampled non-disjoint groups")
                    behavior = engine.build_candidate_behavior(
                        map_family_id=map_family_id,
                        hyperparameter_value=hyperparameter_value,
                        site_groups=site_groups,
                    )
                    sampled_behaviors.append((site_groups, behavior))

                for high_level_model_id in search_spec.high_level_model_ids:
                    sampled_proposals: list[tuple[dict[str, tuple[Any, ...]], dict[str, Any]]] = []
                    for site_groups, behavior in sampled_behaviors:
                        record = engine.score_candidate_behavior(
                            high_level_model_id=high_level_model_id,
                            site_budget=site_budget,
                            behavior=behavior,
                        )
                        sampled_proposals.append((site_groups, record))
                    sampled_proposals.sort(
                        key=lambda item: (item[1]["residual_bits"]["val"], _site_groups_key(item[0]))
                    )
                    best_site_groups, best_record = sampled_proposals[0]
                    cell_id = _stable_digest(
                        "null_cell",
                        "random_site",
                        high_level_model_id,
                        map_family_id,
                        hyper_id,
                        site_budget,
                    )[:16]
                    best_record["cell_id"] = cell_id
                    best_record["search_status"] = "evaluable"
                    best_record["n_raw_full_proposals"] = 64
                    best_record["n_valid_full_proposals"] = 64
                    best_record = _annotate_records(
                        [best_record],
                        record_kind="null",
                        null_family="random_site",
                    )[0]
                    null_records.append(best_record)
                    null_logs.append(
                        {
                            "cell_id": cell_id,
                            "status": "evaluable",
                            "high_level_model_id": high_level_model_id,
                            "map_family_id": map_family_id,
                            "hyperparameter_id": hyper_id,
                            "hyperparameter_value": hyperparameter_value,
                            "site_budget": site_budget,
                            "search_budget": {
                                "sampling_mode": "64_full_disjoint_random_proposals",
                                "n_sampled_proposals": 64,
                            },
                            "n_raw_full_proposals": 64,
                            "n_valid_full_proposals": 64,
                            "full_proposals": [
                                {
                                    "site_groups": _serialize_site_groups(site_groups),
                                    "val_residual_bits": record["residual_bits"]["val"],
                                }
                                for site_groups, record in sampled_proposals
                            ],
                            "best_full_proposal": {
                                "site_groups": _serialize_site_groups(best_site_groups),
                                "val_residual_bits": best_record["residual_bits"]["val"],
                            },
                            "local_refinement": {
                                "applied": False,
                                "reason": "random_site_null_has_no_refinement",
                            },
                        }
                    )

    null_records.sort(key=lambda record: (record["test_total_bits"], record["candidate_id"]))
    null_logs = _annotate_logs(null_logs, record_kind="null", null_family="random_site")
    null_logs.sort(key=lambda log: log["cell_id"])
    return null_records, null_logs


def run_candidate_like_null_search(
    *,
    null_family: str,
    model: MiniIOITransformer,
    dataset_bundle: S2DatasetBundle,
    search_spec: SearchSpec,
    codebook_id: str = PRIMARY_CODEBOOK_ID,
    search_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    engine = MiniIOICandidateSearchEngine(
        model=model,
        dataset_bundle=dataset_bundle,
        codebook_id=codebook_id,
        search_seed=search_seed,
        linear_epochs=search_spec.linear_epochs,
        mlp_epochs=search_spec.mlp_epochs,
        learning_rate=search_spec.learning_rate,
    )
    records, logs = engine.run_search(
        high_level_model_ids=search_spec.high_level_model_ids,
        site_budgets=search_spec.site_budgets,
        map_family_grid=search_spec.map_family_grid,
    )
    return (
        _annotate_records(records, record_kind="null", null_family=null_family),
        _annotate_logs(logs, record_kind="null", null_family=null_family),
    )
