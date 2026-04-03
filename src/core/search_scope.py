from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.planted.high_level import HIGH_LEVEL_MODEL_ORDER, HIGH_LEVEL_MODELS
from src.planted.readouts import MAP_FAMILY_HYPERGRIDS, parse_map_family_grid
from src.planted.scoring import SITE_BUDGET_CHOICES


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")
    return value


def _choice_coverage(
    selected: tuple[object, ...],
    locked: tuple[object, ...],
) -> bool:
    return len(selected) == len(locked) and set(selected) == set(locked)


def candidate_pool_scope_from_extras(extras: dict[str, Any]) -> dict[str, Any]:
    search_config = _ensure_mapping(
        extras.get("candidate_pool", {}),
        context="extras.candidate_pool",
    )

    raw_high_level_models = search_config.get(
        "high_level_models",
        list(HIGH_LEVEL_MODEL_ORDER),
    )
    if not isinstance(raw_high_level_models, list):
        raise TypeError("candidate_pool.high_level_models must be a list")
    high_level_model_ids = tuple(str(model_id) for model_id in raw_high_level_models)
    for model_id in high_level_model_ids:
        if model_id not in HIGH_LEVEL_MODELS:
            raise ValueError(f"Unknown high-level model id: {model_id}")

    raw_site_budgets = search_config.get("site_budgets", list(SITE_BUDGET_CHOICES))
    if not isinstance(raw_site_budgets, list):
        raise TypeError("candidate_pool.site_budgets must be a list")
    site_budgets = tuple(int(value) for value in raw_site_budgets)
    for site_budget in site_budgets:
        if site_budget not in SITE_BUDGET_CHOICES:
            raise ValueError(f"Unsupported site budget: {site_budget}")

    map_family_grid = parse_map_family_grid(search_config.get("map_families"))
    locked_high_level_model_ids = tuple(HIGH_LEVEL_MODEL_ORDER)
    locked_site_budgets = tuple(SITE_BUDGET_CHOICES)
    locked_map_family_grid = dict(MAP_FAMILY_HYPERGRIDS)

    configured_hyperparameter_cells = sum(len(values) for values in map_family_grid.values())
    locked_hyperparameter_cells = sum(
        len(values) for values in locked_map_family_grid.values()
    )
    configured_candidate_cells = (
        len(high_level_model_ids) * len(site_budgets) * configured_hyperparameter_cells
    )
    locked_candidate_cells = (
        len(locked_high_level_model_ids)
        * len(locked_site_budgets)
        * locked_hyperparameter_cells
    )

    covers_locked_high_level_models = _choice_coverage(
        high_level_model_ids,
        locked_high_level_model_ids,
    )
    covers_locked_site_budgets = _choice_coverage(site_budgets, locked_site_budgets)
    covers_locked_map_family_hypergrid = (
        set(map_family_grid) == set(locked_map_family_grid)
        and all(
            _choice_coverage(
                tuple(map_family_grid[map_family_id]),
                tuple(locked_map_family_grid[map_family_id]),
            )
            for map_family_id in locked_map_family_grid
        )
    )
    covers_full_locked_candidate_pool = (
        covers_locked_high_level_models
        and covers_locked_site_budgets
        and covers_locked_map_family_hypergrid
    )

    missing_high_level_models = [
        model_id
        for model_id in locked_high_level_model_ids
        if model_id not in high_level_model_ids
    ]
    missing_site_budgets = [
        site_budget
        for site_budget in locked_site_budgets
        if site_budget not in site_budgets
    ]
    missing_map_family_hyperparameters: dict[str, list[object]] = {}
    for map_family_id in locked_map_family_grid:
        missing_values = [
            value
            for value in locked_map_family_grid[map_family_id]
            if value not in map_family_grid.get(map_family_id, ())
        ]
        if missing_values:
            missing_map_family_hyperparameters[map_family_id] = missing_values

    return {
        "scope_label": (
            "full_locked_candidate_pool"
            if covers_full_locked_candidate_pool
            else "reduced_locked_slice"
        ),
        "covers_full_locked_candidate_pool": covers_full_locked_candidate_pool,
        "covers_locked_high_level_models": covers_locked_high_level_models,
        "covers_locked_site_budgets": covers_locked_site_budgets,
        "covers_locked_map_family_hypergrid": covers_locked_map_family_hypergrid,
        "configured_high_level_model_ids": list(high_level_model_ids),
        "locked_high_level_model_ids": list(locked_high_level_model_ids),
        "configured_site_budgets": list(site_budgets),
        "locked_site_budgets": list(locked_site_budgets),
        "configured_map_family_grid": {
            map_family_id: list(values)
            for map_family_id, values in map_family_grid.items()
        },
        "locked_map_family_grid": {
            map_family_id: list(values)
            for map_family_id, values in locked_map_family_grid.items()
        },
        "configured_candidate_cells": configured_candidate_cells,
        "locked_candidate_cells": locked_candidate_cells,
        "candidate_cell_coverage": configured_candidate_cells / locked_candidate_cells,
        "missing_high_level_models": missing_high_level_models,
        "missing_site_budgets": missing_site_budgets,
        "missing_map_family_hyperparameters": missing_map_family_hyperparameters,
    }


def attach_recorded_candidate_counts(
    scope_summary: dict[str, Any],
    *,
    recorded_candidate_records: int | None,
    recorded_unevaluable_candidate_cells: int | None,
) -> dict[str, Any]:
    payload = dict(scope_summary)
    payload["recorded_candidate_records"] = recorded_candidate_records
    payload["recorded_unevaluable_candidate_cells"] = recorded_unevaluable_candidate_cells
    if recorded_candidate_records is None or recorded_unevaluable_candidate_cells is None:
        payload["recorded_total_candidate_cells"] = None
        payload["configured_candidate_cells_match_recorded_total"] = None
        return payload
    recorded_total_candidate_cells = (
        recorded_candidate_records + recorded_unevaluable_candidate_cells
    )
    payload["recorded_total_candidate_cells"] = recorded_total_candidate_cells
    payload["configured_candidate_cells_match_recorded_total"] = (
        int(payload["configured_candidate_cells"]) == recorded_total_candidate_cells
    )
    return payload


def save_candidate_pool_scope(scope_summary: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(scope_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
