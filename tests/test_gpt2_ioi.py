from __future__ import annotations

from transformers import AutoTokenizer

from src.gpt2_ioi import (
    DEFAULT_NAME_VOCAB,
    FAMILY_SHIFT,
    GPT2IOILatents,
    GPT2IOIModel,
    build_s3_dataset_bundle,
    build_shuffled_pair_dataset_bundle,
    select_name_vocab,
    shuffled_pair_is_available,
    validate_fixed_length_prompts,
)


def test_gpt2_name_vocab_and_prompt_lengths_validate() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    name_vocab = select_name_vocab(tokenizer, target_size=3)
    summary = validate_fixed_length_prompts(tokenizer, name_vocab=name_vocab)

    assert tuple(name_vocab) == ("Alice", "Bob", "Carol")
    assert summary["fixed_length"] == 11
    assert summary["canonical_prompt_lengths"] == [11]
    assert summary["shift_prompt_lengths"] == [11]


def test_gpt2_site_universe_is_fixed_and_reversible() -> None:
    model = GPT2IOIModel(device="cpu")

    assert model.sequence_length == 11
    assert model.n_layers == 12
    assert len(model.site_universe) == 143
    assert model.site_id(model.site_universe[0]) == 0


def test_gpt2_generated_tuples_are_fixed_length_and_shift_only() -> None:
    model = GPT2IOIModel(device="cpu")
    dataset_bundle = build_s3_dataset_bundle(model)

    split_to_groups = {split: set() for split in ("train", "val", "test", "shift")}
    for tuple_record in dataset_bundle.tuples:
        split = dataset_bundle.split_by_group[tuple_record.metadata.group_id]
        split_to_groups[split].add(tuple_record.metadata.group_id)
        assert len(tuple_record.base_input["token_ids"]) == 11
        assert len(tuple_record.source_input["token_ids"]) == 11
        if tuple_record.metadata.template_family == FAMILY_SHIFT:
            assert split == "shift"

    assert split_to_groups["train"].isdisjoint(split_to_groups["val"])
    assert split_to_groups["train"].isdisjoint(split_to_groups["test"])
    assert split_to_groups["val"].isdisjoint(split_to_groups["test"])


def test_gpt2_patch_and_run_is_deterministic() -> None:
    model = GPT2IOIModel(device="cpu")
    dataset_bundle = build_s3_dataset_bundle(model)
    tuple_record = next(
        record
        for record in dataset_bundle.tuples
        if record.intervention_type == "R" and record.metadata.template_family != FAMILY_SHIFT
    )
    base = GPT2IOILatents.from_dict(tuple_record.metadata.latent_base)
    source = GPT2IOILatents.from_dict(tuple_record.metadata.latent_source)
    site = model.site_universe[-1]
    first = model.patch_and_run(
        base_latents=base,
        source_latents=source,
        intervention_type="R",
        patch_sites=(site,),
    )
    second = model.patch_and_run(
        base_latents=base,
        source_latents=source,
        intervention_type="R",
        patch_sites=(site,),
    )

    assert first.output_index == second.output_index
    assert first.output_token == second.output_token


def test_gpt2_shuffled_pair_is_available_and_changes_sources() -> None:
    model = GPT2IOIModel(device="cpu")
    dataset_bundle = build_s3_dataset_bundle(model)
    shuffled_bundle = build_shuffled_pair_dataset_bundle(dataset_bundle, seed=0)

    assert shuffled_pair_is_available(dataset_bundle)
    changed = False
    for split_name in ("train", "val"):
        for tuple_record in shuffled_bundle.tuples_by_split[split_name]:
            source_group_id = tuple_record.metadata.extra.get("shuffled_pair_source_group_id")
            original_group_id = tuple_record.metadata.extra.get("shuffled_pair_original_group_id")
            if source_group_id is not None and source_group_id != original_group_id:
                changed = True
                break
        if changed:
            break
    assert changed


def test_gpt2_latents_use_runtime_name_vocab() -> None:
    runtime_vocab = ("Alice", "Bob", "Carol", "Dave")
    GPT2IOILatents.set_name_vocab(runtime_vocab)
    try:
        latents = GPT2IOILatents(
            n1_index=0,
            n2_index=3,
            r_value=1,
            family=FAMILY_SHIFT,
        )
        assert latents.n2_token == "Dave"
        assert latents.to_dict()["N2_token"] == "Dave"
    finally:
        GPT2IOILatents.set_name_vocab(DEFAULT_NAME_VOCAB)
