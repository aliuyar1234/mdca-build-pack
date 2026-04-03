from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.core.config import load_run_config
from src.core.search_scope import (
    attach_recorded_candidate_counts,
    candidate_pool_scope_from_extras,
)

from .frontier_core import load_json, save_json


def _load_optional_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def _candidate_pool_scope(run_dir: Path, *, config_extras: dict[str, Any]) -> dict[str, Any]:
    scope_summary = candidate_pool_scope_from_extras(config_extras)
    candidate_records_path = run_dir / "candidate_records.json"
    proposal_logs_path = run_dir / "proposal_logs.json"
    candidate_records = (
        load_json(candidate_records_path)
        if candidate_records_path.exists()
        else None
    )
    proposal_logs = load_json(proposal_logs_path) if proposal_logs_path.exists() else None
    recorded_unevaluable = (
        sum(1 for item in proposal_logs if str(item.get("status")) != "evaluable")
        if proposal_logs is not None
        else None
    )
    return attach_recorded_candidate_counts(
        scope_summary,
        recorded_candidate_records=(
            len(candidate_records) if candidate_records is not None else None
        ),
        recorded_unevaluable_candidate_cells=recorded_unevaluable,
    )


def _proposal_coverage(run_dir: Path) -> dict[str, Any] | None:
    proposal_logs = _load_optional_json(run_dir / "proposal_logs.json")
    if proposal_logs is None:
        return None
    if not isinstance(proposal_logs, list):
        raise TypeError("proposal_logs.json must contain a list")

    by_site_budget: dict[str, dict[str, int]] = {}
    by_map_family: dict[str, dict[str, int]] = {}
    by_site_budget_and_map_family: dict[str, int] = {}
    for item in proposal_logs:
        status = str(item["status"])
        site_budget = str(item["site_budget"])
        map_family_id = str(item["map_family_id"])
        by_site_budget.setdefault(
            site_budget,
            {"total": 0, "unevaluable": 0},
        )
        by_map_family.setdefault(
            map_family_id,
            {"total": 0, "unevaluable": 0},
        )
        by_site_budget[site_budget]["total"] += 1
        by_map_family[map_family_id]["total"] += 1
        by_site_budget_and_map_family.setdefault(f"b={site_budget}:{map_family_id}", 0)
        if status != "evaluable":
            by_site_budget[site_budget]["unevaluable"] += 1
            by_map_family[map_family_id]["unevaluable"] += 1
            by_site_budget_and_map_family[f"b={site_budget}:{map_family_id}"] += 1

    return {
        "n_total_cells": len(proposal_logs),
        "n_unevaluable_cells": sum(
            1 for item in proposal_logs if str(item["status"]) != "evaluable"
        ),
        "by_site_budget": by_site_budget,
        "by_map_family": by_map_family,
        "unevaluable_cells_by_site_budget_and_map_family": by_site_budget_and_map_family,
    }


def _oracle_summary(run_dir: Path) -> dict[str, Any] | None:
    oracle_support = _load_optional_json(run_dir / "oracle_support.json")
    oracle_site_overlap = _load_optional_json(run_dir / "oracle_site_overlap.json")
    if oracle_support is None or oracle_site_overlap is None:
        return None
    if not isinstance(oracle_support, dict) or not isinstance(oracle_site_overlap, dict):
        raise TypeError("oracle artifacts must contain mappings")

    support_row = dict(oracle_support["support_row"])
    support_summary = dict(oracle_support["support_summary"])
    oracle_overlap = dict(oracle_site_overlap["oracle_site_overlap"])
    return {
        "oracle_id": oracle_support["oracle_id"],
        "exact_site_match": bool(oracle_overlap["exact_site_match"]),
        "total_overlap_count": int(oracle_overlap["total_overlap_count"]),
        "supported": bool(support_row["supported"]),
        "support_reason": support_row["support_reason"],
        "g_test": support_row["g_test"],
        "g_shift": support_row["g_shift"],
        "g_test_lcb95": support_row["g_test_lcb95"],
        "frontier_defined_test": bool(support_row["frontier_defined_test"]),
        "frontier_defined_shift": bool(support_row["frontier_defined_shift"]),
        "best_candidate_frontier_eligible": support_summary.get(
            "best_candidate_frontier_eligible"
        ),
    }


def _setting_payload(run_dir: Path) -> dict[str, Any]:
    config = load_run_config(run_dir / "config_snapshot.yaml")
    primary_acceptance = load_json(run_dir / "acceptance_summary.json")
    primary_frontier_test = load_json(run_dir / "frontier_test.json")
    primary_frontier_shift = load_json(run_dir / "frontier_shift.json")
    robustness_dir = run_dir / "robustness_quantized"
    robustness_acceptance = load_json(robustness_dir / "acceptance_summary.json")
    robustness_comparison = load_json(robustness_dir / "robustness_comparison.json")
    candidate_pool_scope = _candidate_pool_scope(run_dir, config_extras=config.extras)
    proposal_coverage = _proposal_coverage(run_dir)
    oracle_summary = _oracle_summary(run_dir)
    return {
        "setting_id": config.setting_id,
        "primary_run_dir": str(run_dir),
        "robustness_run_dir": str(robustness_dir),
        "candidate_pool_scope": candidate_pool_scope,
        "proposal_coverage": proposal_coverage,
        "oracle": oracle_summary,
        "primary": {
            "n_supported": primary_acceptance["n_supported"],
            "supported_class_ids": primary_acceptance["supported_class_ids"],
            "control_calibration_changed_decision": primary_acceptance[
                "control_calibration_changed_decision"
            ],
            "test_domain": primary_frontier_test["domain"],
            "shift_domain": primary_frontier_shift["domain"],
            "test_valid_bins": sum(
                1 for item in primary_frontier_test["bin_summaries"] if bool(item["valid"])
            ),
            "shift_valid_bins": sum(
                1 for item in primary_frontier_shift["bin_summaries"] if bool(item["valid"])
            ),
            "best_candidate_id": primary_acceptance.get("best_candidate_id"),
            "best_candidate_frontier_eligible": primary_acceptance.get(
                "best_candidate_frontier_eligible"
            ),
            "best_candidate_support_reason": primary_acceptance.get(
                "best_candidate_support_reason"
            ),
            "best_frontier_defined_candidate_id": primary_acceptance.get(
                "best_frontier_defined_candidate_id"
            ),
            "best_frontier_defined_candidate_within_best_bits": primary_acceptance.get(
                "best_frontier_defined_candidate_within_best_bits"
            ),
            "best_frontier_defined_candidate_g_test": primary_acceptance.get(
                "best_frontier_defined_candidate_g_test"
            ),
            "best_frontier_defined_candidate_g_shift": primary_acceptance.get(
                "best_frontier_defined_candidate_g_shift"
            ),
        },
        "robustness": {
            "n_supported": robustness_acceptance["n_supported"],
            "supported_class_ids": robustness_acceptance["supported_class_ids"],
            "control_calibration_changed_decision": robustness_acceptance[
                "control_calibration_changed_decision"
            ],
            "best_candidate_frontier_eligible": robustness_acceptance.get(
                "best_candidate_frontier_eligible"
            ),
        },
        "support_changed": robustness_comparison["support_changed"],
        "best_candidate_changed": robustness_comparison["best_candidate_changed"],
    }


def _evidence_scope_summary(settings: list[dict[str, Any]]) -> dict[str, Any]:
    settings_missing_full_scope = [
        item["setting_id"]
        for item in settings
        if not bool(item["candidate_pool_scope"]["covers_full_locked_candidate_pool"])
    ]
    return {
        "all_settings_cover_full_locked_candidate_pool": not settings_missing_full_scope,
        "settings_missing_full_locked_candidate_pool": settings_missing_full_scope,
    }


def _interpretive_caveats(settings: list[dict[str, Any]]) -> dict[str, Any]:
    planted_oracle = next(
        (item.get("oracle") for item in settings if item["setting_id"] == "planted"),
        None,
    )
    return {
        "settings_with_frontier_ineligible_global_best": [
            item["setting_id"]
            for item in settings
            if item["primary"].get("best_candidate_frontier_eligible") is False
        ],
        "settings_where_best_frontier_defined_candidate_fails_best_bits_gate": [
            item["setting_id"]
            for item in settings
            if item["primary"].get("best_frontier_defined_candidate_id") is not None
            and item["primary"].get("best_frontier_defined_candidate_within_best_bits") is False
        ],
        "settings_with_sparse_valid_frontier": [
            item["setting_id"]
            for item in settings
            if int(item["primary"]["test_valid_bins"]) <= 2
            or int(item["primary"]["shift_valid_bins"]) <= 2
        ],
        "unevaluable_cells_by_setting": {
            item["setting_id"]: (
                item["proposal_coverage"]["n_unevaluable_cells"]
                if item["proposal_coverage"] is not None
                else None
            )
            for item in settings
        },
        "oracle_backed_planted_recovery_failure": bool(
            planted_oracle
            and planted_oracle["exact_site_match"]
            and not planted_oracle["supported"]
        ),
    }


def _recommended_claim_statuses(settings: list[dict[str, Any]]) -> dict[str, str]:
    by_setting = {item["setting_id"]: item for item in settings}
    all_no_support = all(
        item["primary"]["n_supported"] == 0 and item["robustness"]["n_supported"] == 0
        for item in settings
    )
    all_full_locked_candidate_pool = all(
        bool(item["candidate_pool_scope"]["covers_full_locked_candidate_pool"])
        for item in settings
    )
    stability_holds = not any(
        item["support_changed"] or item["best_candidate_changed"] for item in settings
    )
    calibration_signal_holds = all(
        bool(item["primary"]["control_calibration_changed_decision"])
        and bool(item["robustness"]["control_calibration_changed_decision"])
        for item in settings
    )
    planted_scope_complete = bool(
        by_setting.get("planted", {})
        .get("candidate_pool_scope", {})
        .get("covers_full_locked_candidate_pool", False)
    )
    planted_oracle = by_setting.get("planted", {}).get("oracle")
    planted_oracle_failure = bool(
        planted_oracle
        and planted_oracle["exact_site_match"]
        and not planted_oracle["supported"]
    )
    return {
        "C1": "weakened",
        "C2": (
            "unsupported"
            if planted_scope_complete and planted_oracle_failure
            else "weakened"
        ),
        "C3": "supported" if calibration_signal_holds else "weakened",
        "C4": (
            "unsupported"
            if all_no_support and all_full_locked_candidate_pool
            else "weakened"
        ),
        "C5": (
            "unsupported"
            if all_no_support and all_full_locked_candidate_pool
            else "weakened"
        ),
        "C6": "supported" if stability_holds else "weakened",
        "C7": "partially supported",
        "C8": (
            "supported"
            if all_no_support and all_full_locked_candidate_pool
            else "partially supported"
        ),
        "paper_shape": (
            "negative_result"
            if by_setting.get("gpt2_ioi", {}).get("robustness", {}).get("n_supported", 0) == 0
            and all_no_support
            and all_full_locked_candidate_pool
            else "reduced_scope_no_support"
            if all_no_support
            else "mixed_or_weakened_positive"
        ),
    }


def _figure_table_manifest(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    figures: list[dict[str, str]] = []
    tables: list[dict[str, str]] = []
    for run_dir in run_dirs:
        config = load_run_config(run_dir / "config_snapshot.yaml")
        setting_id = config.setting_id
        robustness_dir = run_dir / "robustness_quantized"
        figures.extend(
            [
                {
                    "id": f"{setting_id}_frontier_test_primary",
                    "path": str(run_dir / "frontier_test.png"),
                    "caption": f"{setting_id} primary-codebook test frontier",
                },
                {
                    "id": f"{setting_id}_frontier_shift_primary",
                    "path": str(run_dir / "frontier_shift.png"),
                    "caption": f"{setting_id} primary-codebook shift frontier",
                },
                {
                    "id": f"{setting_id}_frontier_test_quantized",
                    "path": str(robustness_dir / "frontier_test.png"),
                    "caption": f"{setting_id} quantized-codebook test frontier",
                },
                {
                    "id": f"{setting_id}_frontier_shift_quantized",
                    "path": str(robustness_dir / "frontier_shift.png"),
                    "caption": f"{setting_id} quantized-codebook shift frontier",
                },
            ]
        )
        tables.extend(
            [
                {
                    "id": f"{setting_id}_support_primary",
                    "path": str(run_dir / "support_table.json"),
                    "caption": f"{setting_id} primary-codebook support table",
                },
                {
                    "id": f"{setting_id}_support_quantized",
                    "path": str(robustness_dir / "support_table.json"),
                    "caption": f"{setting_id} quantized-codebook support table",
                },
                {
                    "id": f"{setting_id}_comparison_quantized",
                    "path": str(robustness_dir / "robustness_comparison.json"),
                    "caption": f"{setting_id} primary-vs-quantized comparison summary",
                },
            ]
        )
    tables.append(
        {
            "id": "final_claim_support_summary",
            "path": str(output_dir / "final_claim_support_summary.json"),
            "caption": "Final cross-setting claim support summary",
        }
    )
    return {"figures": figures, "tables": tables}


def _audit_artifact_manifest(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    settings: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config = load_run_config(run_dir / "config_snapshot.yaml")
        robustness_dir = run_dir / "robustness_quantized"
        setting_payload: dict[str, Any] = {
            "setting_id": config.setting_id,
            "primary_run_dir": str(run_dir),
            "artifacts": {
                "config_snapshot": str(run_dir / "config_snapshot.yaml"),
                "candidate_pool_scope": str(run_dir / "candidate_pool_scope.json"),
                "candidate_records": str(run_dir / "candidate_records.json"),
                "candidate_table": str(run_dir / "candidate_table.json"),
                "proposal_logs": str(run_dir / "proposal_logs.json"),
                "null_records": str(run_dir / "null_records.json"),
                "null_search_logs": str(run_dir / "null_search_logs.json"),
                "frontier_test": str(run_dir / "frontier_test.json"),
                "frontier_shift": str(run_dir / "frontier_shift.json"),
                "acceptance_summary": str(run_dir / "acceptance_summary.json"),
                "support_table": str(run_dir / "support_table.json"),
            },
            "robustness_artifacts": {
                "run_dir": str(robustness_dir),
                "candidate_records": str(robustness_dir / "candidate_records.json"),
                "null_records": str(robustness_dir / "null_records.json"),
                "frontier_test": str(robustness_dir / "frontier_test.json"),
                "frontier_shift": str(robustness_dir / "frontier_shift.json"),
                "acceptance_summary": str(robustness_dir / "acceptance_summary.json"),
                "support_table": str(robustness_dir / "support_table.json"),
                "robustness_comparison": str(robustness_dir / "robustness_comparison.json"),
            },
        }
        if (run_dir / "oracle_support.json").exists():
            setting_payload["oracle_artifacts"] = {
                "oracle_candidate_record": str(run_dir / "oracle_candidate_record.json"),
                "oracle_support": str(run_dir / "oracle_support.json"),
                "oracle_site_overlap": str(run_dir / "oracle_site_overlap.json"),
            }
        settings.append(setting_payload)
    return {
        "settings": settings,
        "final_claim_support_summary": str(output_dir / "final_claim_support_summary.json"),
        "figure_table_manifest": str(output_dir / "figure_table_manifest.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.final_manifest")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", default="artifacts/final_package")
    args = parser.parse_args(argv)

    run_dirs = [Path(item) for item in args.run_dir]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = [_setting_payload(run_dir) for run_dir in run_dirs]
    claim_support_summary = {
        "evidence_scope": _evidence_scope_summary(settings),
        "interpretive_caveats": _interpretive_caveats(settings),
        "settings": settings,
        "recommended_statuses": _recommended_claim_statuses(settings),
    }
    figure_table_manifest = _figure_table_manifest(run_dirs, output_dir)
    audit_artifact_manifest = _audit_artifact_manifest(run_dirs, output_dir)

    save_json(claim_support_summary, output_dir / "final_claim_support_summary.json")
    save_json(figure_table_manifest, output_dir / "figure_table_manifest.json")
    save_json(audit_artifact_manifest, output_dir / "audit_artifact_manifest.json")

    payload = {
        "status": "ok",
        "entrypoint": "python -m src.analysis.final_manifest",
        "n_settings": len(settings),
        "output_dir": str(output_dir),
        "final_claim_support_summary_path": str(output_dir / "final_claim_support_summary.json"),
        "figure_table_manifest_path": str(output_dir / "figure_table_manifest.json"),
        "audit_artifact_manifest_path": str(output_dir / "audit_artifact_manifest.json"),
        "paper_shape": claim_support_summary["recommended_statuses"]["paper_shape"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
