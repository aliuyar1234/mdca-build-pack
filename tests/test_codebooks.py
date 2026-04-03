from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

from src.analysis import robustness_codebook
from src.analysis.acceptance_core import SupportResult
from src.analysis.frontier_core import FrontierResult
from src.core.codebooks import QUANTIZED_CODEBOOK_ID, build_code_length_breakdown
from src.core.config import MethodConstants, PathConfig, RunConfig, RuntimeConfig, SeedBundle, save_run_config


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_TEMP_ROOT = REPO_ROOT / "artifacts" / "test_tmp"


def _candidate_like_record(
    *,
    candidate_id: str,
    null_family: str | None,
    test_group_ids: list[str],
    residual_bits_test: float,
) -> dict[str, object]:
    code_lengths = build_code_length_breakdown(
        high_level_bits=2.0,
        budget_bits=1.0,
        site_bits=3.0,
        family_bits=2.0,
        hyperparameter_bits=0.0,
        parameter_count_eff=1,
        n_train_tuples=6,
        codebook_id="primary",
    )
    residual_contributions = [
        {
            "candidate_id": candidate_id,
            "split": "test",
            "group_id": group_id,
            "residual_bits": residual_bits_test / len(test_group_ids),
            "n_examples": 1,
        }
        for group_id in test_group_ids
    ]
    return {
        "candidate_id": candidate_id,
        "record_kind": "null" if null_family else "candidate",
        "null_family": null_family,
        "high_level_model_id": "H_true_other",
        "map_family_id": "linear_dense",
        "hyperparameter_id": "default_dense",
        "hyperparameter_value": "default_dense",
        "site_budget": 1,
        "site_groups": {
            "N1": [{"layer_index": 0, "token_index": 0}],
            "N2": [{"layer_index": 0, "token_index": 1}],
            "R": [{"layer_index": 0, "token_index": 2}],
        },
        "parameter_count_eff": 1,
        "cell_id": f"cell_{candidate_id}",
        "n_raw_full_proposals": 1,
        "n_valid_full_proposals": 1,
        "val_error_rate": 0.25,
        "epsilon": 0.25,
        "code_lengths": code_lengths.to_dict(),
        "residual_bits": {
            "val": 2.0,
            "test": residual_bits_test,
            "shift": residual_bits_test + 1.0,
        },
        "residual_bits_per_example": {
            "val": 1.0,
            "test": residual_bits_test / len(test_group_ids),
            "shift": (residual_bits_test + 1.0) / len(test_group_ids),
        },
        "test_total_bits": code_lengths.total_structural_bits + residual_bits_test,
        "test_total_bits_per_example": (
            code_lengths.total_structural_bits + residual_bits_test
        )
        / len(test_group_ids),
        "residual_contributions": {
            "val": [
                {
                    "candidate_id": candidate_id,
                    "split": "val",
                    "group_id": "g_val_0",
                    "residual_bits": 2.0,
                    "n_examples": 1,
                }
            ],
            "test": residual_contributions,
            "shift": [
                {
                    "candidate_id": candidate_id,
                    "split": "shift",
                    "group_id": f"g_shift_{index}",
                    "residual_bits": (residual_bits_test + 1.0) / len(test_group_ids),
                    "n_examples": 1,
                }
                for index, _ in enumerate(test_group_ids)
            ],
        },
        "split_sizes": {
            "val": 1,
            "test": len(test_group_ids),
            "shift": len(test_group_ids),
        },
    }


def test_robustness_codebook_cli_rewrites_records(monkeypatch) -> None:
    WORKSPACE_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_root = WORKSPACE_TEMP_ROOT / f"mdca_m6_{uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=False)
    run_dir = temp_root / "toy_run"
    run_dir.mkdir(parents=True)
    save_run_config(
        RunConfig(
            setting_id="planted",
            variant="toy_primary",
            description="Toy run for robustness codebook smoke test.",
            paths=PathConfig(results_dir="results", artifacts_dir="artifacts"),
            method=MethodConstants.default(),
            seeds=SeedBundle(
                global_seed=0,
                dataset_seed=0,
                model_init_seed=0,
                candidate_search_seed=0,
                bootstrap_seed=0,
            ),
            runtime=RuntimeConfig(device="cpu", dtype="float32", num_workers=0),
            extras={},
        ),
        run_dir / "config_snapshot.yaml",
    )
    split_manifest = [
        {"group_id": f"g_train_{index}", "split": "train"}
        for index in range(6)
    ] + [
        {"group_id": "g_val_0", "split": "val"},
        {"group_id": "g_test_0", "split": "test"},
        {"group_id": "g_test_1", "split": "test"},
    ]
    (run_dir / "split_manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")

    candidate_record = _candidate_like_record(
        candidate_id="cand_a",
        null_family=None,
        test_group_ids=["g_test_0", "g_test_1"],
        residual_bits_test=3.0,
    )
    null_records = []
    for family in ("random_site", "shuffled_pair"):
        for index in range(5):
            null_records.append(
                _candidate_like_record(
                    candidate_id=f"{family}_{index}",
                    null_family=family,
                    test_group_ids=["g_test_0", "g_test_1"],
                    residual_bits_test=4.0 + index,
                )
            )
    (run_dir / "candidate_records.json").write_text(
        json.dumps([candidate_record]),
        encoding="utf-8",
    )
    (run_dir / "null_records.json").write_text(json.dumps(null_records), encoding="utf-8")

    monkeypatch.setattr(
        robustness_codebook,
        "maybe_render_frontier_plot",
        lambda **_: None,
    )
    monkeypatch.setattr(
        robustness_codebook,
        "build_balanced_frontier",
        lambda **kwargs: FrontierResult(
            split=str(kwargs["split"]),
            bin_width_bits=2,
            min_family_count=5,
            available_families=(),
            valid_bin_centers=(),
            valid_bin_quantiles=(),
            isotonic_values=(),
            domain=None,
            bin_summaries=(),
        ),
    )
    monkeypatch.setattr(
        robustness_codebook,
        "compute_support",
        lambda **_: SupportResult(
            summary={
                "n_supported": 0,
                "supported_class_ids": [],
                "control_calibration_changed_decision": True,
            },
            support_table=[],
        ),
    )
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = robustness_codebook.main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    rewritten_dir = Path(REPO_ROOT / payload["run_dir"])
    assert rewritten_dir.exists()
    rewritten_candidates = json.loads((rewritten_dir / "candidate_records.json").read_text(encoding="utf-8"))
    assert rewritten_candidates[0]["code_lengths"]["parameter_bits"] == 16.0
    assert Path(rewritten_dir / "robustness_comparison.json").exists()
