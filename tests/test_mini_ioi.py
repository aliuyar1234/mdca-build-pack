from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.analysis.frontier_core import load_json
from src.mini_ioi import (
    FAMILY_SHIFT,
    NAME_VOCAB,
    MiniIOICandidateSearchEngine,
    MiniIOILatents,
    MiniIOITransformer,
    build_s2_dataset_bundle,
    build_shuffled_pair_dataset_bundle,
    generate_training_latents,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_s2_tokenizer_and_site_universe_are_fixed() -> None:
    model = MiniIOITransformer(model_seed=0)

    assert model.sequence_length == 10
    assert len(model.site_universe) == 30
    assert tuple(model.name_vocab) == NAME_VOCAB
    assert model.tokenizer.encode(model.render_prompt_tokens(generate_training_latents()[0]))


def test_s2_generated_tuples_are_fixed_length_and_shift_only() -> None:
    model = MiniIOITransformer(model_seed=1)
    dataset_bundle = build_s2_dataset_bundle(model)

    seen_lengths = set()
    split_to_groups = {split: set() for split in ("train", "val", "test", "shift")}
    for tuple_record in dataset_bundle.tuples:
        base_tokens = tuple_record.base_input["tokens"]
        source_tokens = tuple_record.source_input["tokens"]
        seen_lengths.add(len(base_tokens))
        seen_lengths.add(len(source_tokens))
        split = dataset_bundle.split_by_group[tuple_record.metadata.group_id]
        split_to_groups[split].add(tuple_record.metadata.group_id)
        if tuple_record.metadata.template_family == FAMILY_SHIFT:
            assert split == "shift"

    assert seen_lengths == {10}
    assert split_to_groups["train"].isdisjoint(split_to_groups["val"])
    assert split_to_groups["train"].isdisjoint(split_to_groups["test"])
    assert split_to_groups["val"].isdisjoint(split_to_groups["test"])


def test_s2_training_learns_canonical_prompt_task() -> None:
    model = MiniIOITransformer(model_seed=2)
    summary = model.train_on_canonical_prompts(
        canonical_latents=generate_training_latents(),
        epochs=120,
        batch_size=16,
        learning_rate=3.0e-3,
        weight_decay=0.0,
    )

    assert summary["final_accuracy"] >= 0.95


def test_s2_candidate_specific_patch_can_change_observed_output() -> None:
    model = MiniIOITransformer(model_seed=3)
    model.train_on_canonical_prompts(
        canonical_latents=generate_training_latents(),
        epochs=120,
        batch_size=16,
        learning_rate=3.0e-3,
        weight_decay=0.0,
    )
    dataset_bundle = build_s2_dataset_bundle(model)
    changed = False
    for tuple_record in dataset_bundle.tuples:
        if tuple_record.intervention_type != "R":
            continue
        base_latents = MiniIOILatents.from_dict(tuple_record.metadata.latent_base)
        source_latents = MiniIOILatents.from_dict(tuple_record.metadata.latent_source)
        base_clean = model.run_clean(base_latents)
        patched = model.patch_and_run(
            base_latents=base_latents,
            source_latents=source_latents,
            intervention_type="R",
            patch_sites=(model.site_universe[-1],),
        )
        if patched.output_token != base_clean.output_token:
            changed = True
            break
    assert changed


def test_s2_shuffled_pair_changes_tuple_conditioned_search_inputs() -> None:
    model = MiniIOITransformer(model_seed=4)
    model.train_on_canonical_prompts(
        canonical_latents=generate_training_latents(),
        epochs=60,
        batch_size=16,
        learning_rate=3.0e-3,
        weight_decay=0.0,
    )
    dataset_bundle = build_s2_dataset_bundle(model)
    shuffled_bundle = build_shuffled_pair_dataset_bundle(dataset_bundle, seed=0)

    original_engine = MiniIOICandidateSearchEngine(model=model, dataset_bundle=dataset_bundle)
    shuffled_engine = MiniIOICandidateSearchEngine(model=model, dataset_bundle=shuffled_bundle)

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


def test_mini_ioi_full_pipeline_smoke() -> None:
    experiment = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments.mini_ioi",
            "--config",
            "configs/mini_ioi/test_full.yaml",
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

    null_records = load_json(Path(run_dir) / "null_records.json")
    shuffled_bundle = build_shuffled_pair_dataset_bundle(
        build_s2_dataset_bundle(MiniIOITransformer(model_seed=12)),
        seed=0,
    )
    assert len(null_records) > 0
    assert len(shuffled_bundle.tuples) > 0
