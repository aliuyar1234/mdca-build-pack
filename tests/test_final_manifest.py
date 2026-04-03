from __future__ import annotations

from pathlib import Path

from src.analysis.final_manifest import _interpretive_caveats, _recommended_claim_statuses
from src.core.config import load_run_config
from src.core.search_scope import candidate_pool_scope_from_extras


REPO_ROOT = Path(__file__).resolve().parents[1]


def _setting_stub(
    setting_id: str,
    *,
    full_scope: bool,
    oracle: dict[str, object] | None = None,
    best_candidate_frontier_eligible: bool = False,
    best_frontier_defined_candidate_within_best_bits: bool = False,
    unevaluable_cells: int = 0,
    test_valid_bins: int = 1,
    shift_valid_bins: int = 1,
) -> dict[str, object]:
    return {
        "setting_id": setting_id,
        "candidate_pool_scope": {
            "covers_full_locked_candidate_pool": full_scope,
        },
        "oracle": oracle,
        "proposal_coverage": {
            "n_unevaluable_cells": unevaluable_cells,
        },
        "primary": {
            "n_supported": 0,
            "control_calibration_changed_decision": True,
            "best_candidate_frontier_eligible": best_candidate_frontier_eligible,
            "best_frontier_defined_candidate_id": "cand_frontier",
            "best_frontier_defined_candidate_within_best_bits": (
                best_frontier_defined_candidate_within_best_bits
            ),
            "test_valid_bins": test_valid_bins,
            "shift_valid_bins": shift_valid_bins,
        },
        "robustness": {
            "n_supported": 0,
            "control_calibration_changed_decision": True,
        },
        "support_changed": False,
        "best_candidate_changed": False,
    }


def test_candidate_pool_scope_defaults_to_full_locked_grid() -> None:
    scope = candidate_pool_scope_from_extras({})

    assert scope["covers_full_locked_candidate_pool"] is True
    assert scope["scope_label"] == "full_locked_candidate_pool"
    assert scope["configured_candidate_cells"] == 120
    assert scope["locked_candidate_cells"] == 120


def test_candidate_pool_scope_detects_reduced_slice_from_config() -> None:
    config = load_run_config(REPO_ROOT / "configs" / "mini_ioi" / "full.yaml")
    scope = candidate_pool_scope_from_extras(config.extras)

    assert scope["covers_full_locked_candidate_pool"] is False
    assert scope["scope_label"] == "reduced_locked_slice"
    assert scope["configured_candidate_cells"] == 40
    assert scope["missing_site_budgets"] == [2, 4]


def test_final_manifest_scope_guard_blocks_negative_result_recommendation() -> None:
    statuses = _recommended_claim_statuses(
        [
            _setting_stub("planted", full_scope=False),
            _setting_stub("mini_ioi", full_scope=False),
            _setting_stub("gpt2_ioi", full_scope=False),
        ]
    )

    assert statuses["C2"] == "weakened"
    assert statuses["C4"] == "weakened"
    assert statuses["C5"] == "weakened"
    assert statuses["C8"] == "partially supported"
    assert statuses["paper_shape"] == "reduced_scope_no_support"


def test_final_manifest_recommends_negative_result_only_for_full_locked_scope() -> None:
    statuses = _recommended_claim_statuses(
        [
            _setting_stub("planted", full_scope=True),
            _setting_stub("mini_ioi", full_scope=True),
            _setting_stub("gpt2_ioi", full_scope=True),
        ]
    )

    assert statuses["C2"] == "weakened"
    assert statuses["C4"] == "unsupported"
    assert statuses["C5"] == "unsupported"
    assert statuses["C8"] == "supported"
    assert statuses["paper_shape"] == "negative_result"


def test_final_manifest_requires_oracle_backing_for_c2() -> None:
    statuses = _recommended_claim_statuses(
        [
            _setting_stub(
                "planted",
                full_scope=True,
                oracle={
                    "exact_site_match": True,
                    "supported": False,
                },
            ),
            _setting_stub("mini_ioi", full_scope=True),
            _setting_stub("gpt2_ioi", full_scope=True),
        ]
    )

    assert statuses["C2"] == "unsupported"


def test_interpretive_caveats_surface_frontier_and_unevaluable_risks() -> None:
    caveats = _interpretive_caveats(
        [
            _setting_stub(
                "planted",
                full_scope=True,
                oracle={
                    "exact_site_match": True,
                    "supported": False,
                },
                best_candidate_frontier_eligible=False,
                best_frontier_defined_candidate_within_best_bits=False,
                unevaluable_cells=0,
                test_valid_bins=3,
                shift_valid_bins=3,
            ),
            _setting_stub(
                "mini_ioi",
                full_scope=True,
                best_candidate_frontier_eligible=False,
                best_frontier_defined_candidate_within_best_bits=False,
                unevaluable_cells=4,
                test_valid_bins=3,
                shift_valid_bins=3,
            ),
            _setting_stub(
                "gpt2_ioi",
                full_scope=True,
                best_candidate_frontier_eligible=False,
                best_frontier_defined_candidate_within_best_bits=False,
                unevaluable_cells=28,
                test_valid_bins=2,
                shift_valid_bins=2,
            ),
        ]
    )

    assert caveats["oracle_backed_planted_recovery_failure"] is True
    assert caveats["settings_with_frontier_ineligible_global_best"] == [
        "planted",
        "mini_ioi",
        "gpt2_ioi",
    ]
    assert caveats["settings_where_best_frontier_defined_candidate_fails_best_bits_gate"] == [
        "planted",
        "mini_ioi",
        "gpt2_ioi",
    ]
    assert caveats["settings_with_sparse_valid_frontier"] == ["gpt2_ioi"]
    assert caveats["unevaluable_cells_by_setting"]["gpt2_ioi"] == 28
