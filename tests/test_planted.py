from __future__ import annotations

from src.core.schemas import Site
from src.planted import (
    FAMILY_SHIFT,
    S1_TRUE_SITE_GROUPS,
    S1PlantedModel,
    build_s1_dataset_bundle,
)
from src.planted.model import PlantedLatents


def test_s1_site_universe_is_fixed_and_reversible() -> None:
    model = S1PlantedModel()

    assert model.sequence_length == 9
    assert len(model.site_universe) == 36
    assert model.site_id(S1_TRUE_SITE_GROUPS["N1"][0]) == model.site_index[(1, 4)]


def test_generated_tuples_differ_in_exactly_one_variable() -> None:
    dataset_bundle = build_s1_dataset_bundle()

    for tuple_record in dataset_bundle.tuples:
        base = PlantedLatents.from_dict(tuple_record.metadata.latent_base)
        source = PlantedLatents.from_dict(tuple_record.metadata.latent_source)
        diffs = base.abstract_difference(source)
        assert diffs == [tuple_record.intervention_type]
        assert base.family == source.family


def test_grouped_splits_are_leak_free_and_shift_only() -> None:
    dataset_bundle = build_s1_dataset_bundle()

    split_to_groups = {split: set() for split in ("train", "val", "test", "shift")}
    for tuple_record in dataset_bundle.tuples:
        split = dataset_bundle.split_by_group[tuple_record.metadata.group_id]
        split_to_groups[split].add(tuple_record.metadata.group_id)
        if tuple_record.metadata.template_family == FAMILY_SHIFT:
            assert split == "shift"
        else:
            assert split in {"train", "val", "test"}

    assert split_to_groups["train"].isdisjoint(split_to_groups["val"])
    assert split_to_groups["train"].isdisjoint(split_to_groups["test"])
    assert split_to_groups["val"].isdisjoint(split_to_groups["test"])


def test_true_site_patching_is_reproducible() -> None:
    model = S1PlantedModel()
    base = PlantedLatents(n1_index=0, n2_index=1, r_value=1, family="canonical")
    source = PlantedLatents(n1_index=0, n2_index=1, r_value=2, family="canonical")

    first = model.patch_and_run(
        base_latents=base,
        source_latents=source,
        intervention_type="R",
        patch_sites=S1_TRUE_SITE_GROUPS["R"],
    )
    second = model.patch_and_run(
        base_latents=base,
        source_latents=source,
        intervention_type="R",
        patch_sites=S1_TRUE_SITE_GROUPS["R"],
    )

    assert first.output_token == second.output_token
    assert first.output_index == second.output_index


def test_candidate_specific_sites_can_change_observed_output() -> None:
    model = S1PlantedModel()
    nuisance_sites = (Site(layer_index=1, token_index=7), Site(layer_index=2, token_index=7))

    dataset_bundle = build_s1_dataset_bundle()
    found_difference = False
    for tuple_record in dataset_bundle.tuples:
        if tuple_record.intervention_type != "R":
            continue
        base = PlantedLatents.from_dict(tuple_record.metadata.latent_base)
        source = PlantedLatents.from_dict(tuple_record.metadata.latent_source)
        true_run = model.patch_and_run(
            base_latents=base,
            source_latents=source,
            intervention_type="R",
            patch_sites=S1_TRUE_SITE_GROUPS["R"],
        )
        nuisance_run = model.patch_and_run(
            base_latents=base,
            source_latents=source,
            intervention_type="R",
            patch_sites=nuisance_sites,
        )
        if true_run.output_token != nuisance_run.output_token:
            found_difference = True
            break

    assert found_difference
