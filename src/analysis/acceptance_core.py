from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .frontier_core import FrontierResult, quantile_linear


@dataclass(frozen=True, slots=True)
class SupportResult:
    summary: dict[str, Any]
    support_table: list[dict[str, Any]]


def _group_vector(record: dict[str, Any], split: str, group_ids: list[str]) -> np.ndarray:
    contributions = {
        item["group_id"]: float(item["residual_bits"])
        for item in record["residual_contributions"][split]
    }
    return np.array([contributions[group_id] for group_id in group_ids], dtype=float)


def _selected_null_ids(frontier: FrontierResult) -> tuple[str, ...]:
    selected: list[str] = []
    for bin_summary in frontier.bin_summaries:
        if not bool(bin_summary["valid"]):
            continue
        for family_ids in bin_summary["selected_null_ids_by_family"].values():
            selected.extend(str(candidate_id) for candidate_id in family_ids)
    return tuple(sorted(set(selected)))


def _rebuild_bootstrap_frontier(
    *,
    frontier: FrontierResult,
    null_total_by_id: dict[str, float],
) -> FrontierResult:
    bin_summaries: list[dict[str, Any]] = []
    valid_bin_centers: list[float] = []
    valid_bin_quantiles: list[float] = []
    for bin_summary in frontier.bin_summaries:
        rebuilt = dict(bin_summary)
        if not rebuilt["valid"]:
            rebuilt["balanced_quantile"] = None
            rebuilt["isotonic_value"] = None
            bin_summaries.append(rebuilt)
            continue
        selected_values: list[float] = []
        for family_ids in rebuilt["selected_null_ids_by_family"].values():
            selected_values.extend(null_total_by_id[candidate_id] for candidate_id in family_ids)
        balanced_quantile = float(
            quantile_linear(np.array(selected_values, dtype=float), 0.05)
        )
        rebuilt["balanced_quantile"] = balanced_quantile
        valid_bin_centers.append(float(rebuilt["bin_center"]))
        valid_bin_quantiles.append(balanced_quantile)
        bin_summaries.append(rebuilt)

    if valid_bin_centers:
        xs = np.array(valid_bin_centers, dtype=float)
        ys = np.array(valid_bin_quantiles, dtype=float)
        from .frontier_core import _fit_isotonic_nonincreasing

        isotonic = _fit_isotonic_nonincreasing(xs, ys)
        domain = (float(xs[0]), float(xs[-1]))
        valid_index = 0
        for rebuilt in bin_summaries:
            if rebuilt["valid"]:
                rebuilt["isotonic_value"] = float(isotonic[valid_index])
                valid_index += 1
    else:
        xs = np.array([], dtype=float)
        ys = np.array([], dtype=float)
        isotonic = np.array([], dtype=float)
        domain = None

    return FrontierResult(
        split=frontier.split,
        bin_width_bits=frontier.bin_width_bits,
        min_family_count=frontier.min_family_count,
        available_families=frontier.available_families,
        valid_bin_centers=tuple(float(value) for value in xs.tolist()),
        valid_bin_quantiles=tuple(float(value) for value in ys.tolist()),
        isotonic_values=tuple(float(value) for value in isotonic.tolist()),
        domain=domain,
        bin_summaries=tuple(bin_summaries),
    )


def compute_support(
    *,
    candidate_records: list[dict[str, Any]],
    null_records: list[dict[str, Any]],
    frontier_test: FrontierResult,
    frontier_shift: FrontierResult,
    bootstrap_reps: int,
    bootstrap_seed: int,
    test_group_ids: list[str],
) -> SupportResult:
    candidate_records = list(candidate_records)
    null_by_id = {record["candidate_id"]: record for record in null_records}
    selected_null_ids = _selected_null_ids(frontier_test)
    selected_null_records = [null_by_id[candidate_id] for candidate_id in selected_null_ids]

    if test_group_ids:
        rng = np.random.default_rng(bootstrap_seed)
        sampled_indices = rng.integers(
            0,
            len(test_group_ids),
            size=(bootstrap_reps, len(test_group_ids)),
        )
        sample_counts = np.zeros((bootstrap_reps, len(test_group_ids)), dtype=int)
        for row_index, row in enumerate(sampled_indices):
            sample_counts[row_index] = np.bincount(row, minlength=len(test_group_ids))
    else:
        sample_counts = np.zeros((bootstrap_reps, 0), dtype=int)

    selected_null_matrix = (
        np.stack(
            [_group_vector(record, "test", test_group_ids) for record in selected_null_records],
            axis=0,
        )
        if selected_null_records and test_group_ids
        else np.zeros((len(selected_null_records), len(test_group_ids)), dtype=float)
    )
    selected_null_totals = (
        sample_counts @ selected_null_matrix.T
        if selected_null_records and test_group_ids
        else np.zeros((bootstrap_reps, len(selected_null_records)), dtype=float)
    )

    best_test_bits_per_example = min(
        float(record["test_total_bits_per_example"])
        for record in candidate_records
    )
    support_table: list[dict[str, Any]] = []
    bootstrap_ready_candidates = [
        record
        for record in candidate_records
        if float(record["test_total_bits_per_example"]) <= best_test_bits_per_example + 0.01
    ]

    candidate_group_vectors = {
        record["candidate_id"]: _group_vector(record, "test", test_group_ids)
        for record in bootstrap_ready_candidates
    }

    for record in candidate_records:
        structural_bits = float(record["code_lengths"]["total_structural_bits"])
        frontier_defined_test = frontier_test.defined_at(structural_bits)
        frontier_defined_shift = frontier_shift.defined_at(structural_bits)
        g_test = (
            None
            if not frontier_defined_test
            else frontier_test.evaluate(structural_bits) - float(record["residual_bits"]["test"])
        )
        g_shift = (
            None
            if not frontier_defined_shift
            else frontier_shift.evaluate(structural_bits) - float(record["residual_bits"]["shift"])
        )

        support_row = {
            "candidate_id": record["candidate_id"],
            "high_level_model_id": record["high_level_model_id"],
            "map_family_id": record["map_family_id"],
            "hyperparameter_id": record["hyperparameter_id"],
            "site_budget": record["site_budget"],
            "test_total_bits_per_example": float(record["test_total_bits_per_example"]),
            "structural_bits": structural_bits,
            "g_test": g_test,
            "g_shift": g_shift,
            "g_test_lcb95": None,
            "frontier_defined_test": frontier_defined_test,
            "frontier_defined_shift": frontier_defined_shift,
            "supported": False,
            "support_reason": None,
        }

        if not frontier_defined_test or not frontier_defined_shift:
            support_row["support_reason"] = "frontier_undefined"
            support_table.append(support_row)
            continue

        if float(record["test_total_bits_per_example"]) > best_test_bits_per_example + 0.01:
            support_row["support_reason"] = "not_within_best_bits"
            support_table.append(support_row)
            continue

        candidate_vector = candidate_group_vectors[record["candidate_id"]]
        candidate_totals = (
            sample_counts @ candidate_vector
            if len(test_group_ids)
            else np.zeros(bootstrap_reps, dtype=float)
        )
        bootstrap_gaps = np.zeros(bootstrap_reps, dtype=float)
        for replicate_index in range(bootstrap_reps):
            null_total_by_id = {
                selected_null_records[idx]["candidate_id"]: float(selected_null_totals[replicate_index, idx])
                for idx in range(len(selected_null_records))
            }
            bootstrap_frontier = _rebuild_bootstrap_frontier(
                frontier=frontier_test,
                null_total_by_id=null_total_by_id,
            )
            frontier_value = bootstrap_frontier.evaluate(structural_bits)
            if frontier_value is None:
                bootstrap_gaps[replicate_index] = float("-inf")
            else:
                bootstrap_gaps[replicate_index] = frontier_value - float(candidate_totals[replicate_index])

        g_test_lcb95 = float(quantile_linear(bootstrap_gaps, 0.05))
        support_row["g_test_lcb95"] = g_test_lcb95
        if g_test_lcb95 <= 0:
            support_row["support_reason"] = "test_gap_lcb_nonpositive"
            support_table.append(support_row)
            continue
        if g_shift is None or g_shift <= 0:
            support_row["support_reason"] = "shift_gap_nonpositive"
            support_table.append(support_row)
            continue

        support_row["supported"] = True
        support_row["support_reason"] = "supported"
        support_table.append(support_row)

    supported_rows = [row for row in support_table if row["supported"]]
    if supported_rows:
        best_supported_bits = min(row["test_total_bits_per_example"] for row in supported_rows)
        supported_class_ids = [
            row["candidate_id"]
            for row in supported_rows
            if row["test_total_bits_per_example"] <= best_supported_bits + 0.01
        ]
    else:
        best_supported_bits = None
        supported_class_ids = []

    for row in support_table:
        row["supported_class_member"] = row["candidate_id"] in supported_class_ids

    best_candidate_row = min(
        support_table,
        key=lambda row: (row["test_total_bits_per_example"], row["candidate_id"]),
    )
    frontier_defined_rows = [
        row
        for row in support_table
        if row["frontier_defined_test"] and row["frontier_defined_shift"]
    ]
    best_frontier_defined_row = (
        min(
            frontier_defined_rows,
            key=lambda row: (row["test_total_bits_per_example"], row["candidate_id"]),
        )
        if frontier_defined_rows
        else None
    )
    summary = {
        "n_candidates": len(candidate_records),
        "n_null_records": len(null_records),
        "best_candidate_test_bits_per_example": best_test_bits_per_example,
        "best_candidate_id": best_candidate_row["candidate_id"],
        "best_candidate_support_reason": best_candidate_row["support_reason"],
        "best_candidate_frontier_defined_test": best_candidate_row["frontier_defined_test"],
        "best_candidate_frontier_defined_shift": best_candidate_row["frontier_defined_shift"],
        "best_candidate_frontier_eligible": (
            best_candidate_row["frontier_defined_test"]
            and best_candidate_row["frontier_defined_shift"]
        ),
        "frontier_defined_candidate_count": len(frontier_defined_rows),
        "best_frontier_defined_candidate_id": (
            best_frontier_defined_row["candidate_id"]
            if best_frontier_defined_row is not None
            else None
        ),
        "best_frontier_defined_candidate_test_bits_per_example": (
            best_frontier_defined_row["test_total_bits_per_example"]
            if best_frontier_defined_row is not None
            else None
        ),
        "best_frontier_defined_candidate_g_test": (
            best_frontier_defined_row["g_test"]
            if best_frontier_defined_row is not None
            else None
        ),
        "best_frontier_defined_candidate_g_shift": (
            best_frontier_defined_row["g_shift"]
            if best_frontier_defined_row is not None
            else None
        ),
        "best_frontier_defined_candidate_within_best_bits": (
            best_frontier_defined_row is not None
            and best_frontier_defined_row["test_total_bits_per_example"]
            <= best_test_bits_per_example + 0.01
        ),
        "n_supported": len(supported_rows),
        "supported_candidate_ids": [row["candidate_id"] for row in supported_rows],
        "supported_class_ids": supported_class_ids,
        "best_supported_bits_per_example": best_supported_bits,
        "control_calibration_changed_decision": any(
            row["support_reason"] not in {"supported", "not_within_best_bits"}
            for row in support_table
            if row["test_total_bits_per_example"] <= best_test_bits_per_example + 0.01
        ),
    }
    support_table.sort(
        key=lambda row: (row["test_total_bits_per_example"], row["candidate_id"])
    )
    return SupportResult(summary=summary, support_table=support_table)
