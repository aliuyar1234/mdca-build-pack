from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from src.core.codebooks import (
    QUANTIZED_CODEBOOK_ID,
    codebook_id_from_extras,
    recode_scored_record,
    validate_codebook_id,
)
from src.core.config import RunConfig, load_run_config, save_run_config
from src.planted.scoring import candidate_table_rows

from .acceptance_core import compute_support
from .frontier_core import (
    FrontierResult,
    build_balanced_frontier,
    load_json,
    maybe_render_frontier_plot,
    save_json,
)


def _n_train_tuples(split_manifest: list[dict[str, Any]]) -> int:
    return sum(1 for row in split_manifest if str(row["split"]) == "train")


def _rewrite_records(
    *,
    records: list[dict[str, Any]],
    n_train_tuples: int,
    codebook_id: str,
) -> list[dict[str, Any]]:
    rewritten = [
        recode_scored_record(
            record=record,
            n_train_tuples=n_train_tuples,
            codebook_id=codebook_id,
        )
        for record in records
    ]
    rewritten.sort(key=lambda record: (float(record["test_total_bits"]), str(record["candidate_id"])))
    return rewritten


def _load_primary_summary(run_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    acceptance_path = run_dir / "acceptance_summary.json"
    support_path = run_dir / "support_table.json"
    frontier_test_path = run_dir / "frontier_test.json"
    frontier_shift_path = run_dir / "frontier_shift.json"
    if acceptance_path.exists():
        payload["acceptance_summary"] = load_json(acceptance_path)
    if support_path.exists():
        payload["support_table"] = load_json(support_path)
    if frontier_test_path.exists():
        payload["frontier_test"] = load_json(frontier_test_path)
    if frontier_shift_path.exists():
        payload["frontier_shift"] = load_json(frontier_shift_path)
    return payload


def _support_row_lookup(support_table: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]): row for row in support_table}


def _build_comparison_summary(
    *,
    primary_run_dir: Path,
    primary_candidate_records: list[dict[str, Any]],
    primary_summary: dict[str, Any],
    rewritten_candidate_records: list[dict[str, Any]],
    rewritten_support: dict[str, Any],
    rewritten_frontier_test: FrontierResult,
    rewritten_frontier_shift: FrontierResult,
    codebook_id: str,
) -> dict[str, Any]:
    primary_acceptance = dict(primary_summary.get("acceptance_summary", {}))
    primary_support_rows = _support_row_lookup(primary_summary.get("support_table", []))
    rewritten_support_rows = _support_row_lookup(rewritten_support["support_table"])
    primary_best_candidate = primary_candidate_records[0]["candidate_id"] if primary_candidate_records else None
    rewritten_best_candidate = (
        rewritten_candidate_records[0]["candidate_id"] if rewritten_candidate_records else None
    )
    candidate_support_changes: list[dict[str, Any]] = []
    for candidate_id in sorted(set(primary_support_rows) | set(rewritten_support_rows)):
        primary_row = primary_support_rows.get(candidate_id, {})
        rewritten_row = rewritten_support_rows.get(candidate_id, {})
        if bool(primary_row.get("supported")) == bool(rewritten_row.get("supported")) and str(
            primary_row.get("support_reason")
        ) == str(rewritten_row.get("support_reason")):
            continue
        candidate_support_changes.append(
            {
                "candidate_id": candidate_id,
                "primary_supported": bool(primary_row.get("supported", False)),
                "robustness_supported": bool(rewritten_row.get("supported", False)),
                "primary_support_reason": primary_row.get("support_reason"),
                "robustness_support_reason": rewritten_row.get("support_reason"),
            }
        )

    return {
        "primary_run_dir": str(primary_run_dir),
        "codebook_id": codebook_id,
        "primary": {
            "n_supported": primary_acceptance.get("n_supported"),
            "supported_class_ids": primary_acceptance.get("supported_class_ids"),
            "control_calibration_changed_decision": primary_acceptance.get(
                "control_calibration_changed_decision"
            ),
            "best_candidate_id": primary_best_candidate,
        },
        "robustness": {
            "n_supported": rewritten_support["summary"]["n_supported"],
            "supported_class_ids": rewritten_support["summary"]["supported_class_ids"],
            "control_calibration_changed_decision": rewritten_support["summary"][
                "control_calibration_changed_decision"
            ],
            "best_candidate_id": rewritten_best_candidate,
            "test_valid_bins": sum(
                1 for item in rewritten_frontier_test.bin_summaries if bool(item["valid"])
            ),
            "shift_valid_bins": sum(
                1 for item in rewritten_frontier_shift.bin_summaries if bool(item["valid"])
            ),
        },
        "best_candidate_changed": primary_best_candidate != rewritten_best_candidate,
        "support_changed": primary_acceptance.get("supported_class_ids", []) != rewritten_support["summary"][
            "supported_class_ids"
        ],
        "candidate_support_changes": candidate_support_changes,
    }


def _rewritten_config(config: RunConfig, *, codebook_id: str, source_run_dir: Path) -> RunConfig:
    extras = dict(config.extras)
    extras["codebook"] = codebook_id
    extras["robustness_source_run_dir"] = str(source_run_dir)
    return replace(
        config,
        variant=f"{config.variant}_{codebook_id}",
        description=f"{config.description} ({codebook_id} codebook)",
        extras=extras,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.robustness_codebook")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--codebook", default=QUANTIZED_CODEBOOK_ID)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    codebook_id = validate_codebook_id(args.codebook)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / f"robustness_{codebook_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_run_config(run_dir / "config_snapshot.yaml")
    split_manifest = load_json(run_dir / "split_manifest.json")
    n_train = _n_train_tuples(split_manifest)
    candidate_records = load_json(run_dir / "candidate_records.json")
    null_records = load_json(run_dir / "null_records.json")

    rewritten_candidates = _rewrite_records(
        records=candidate_records,
        n_train_tuples=n_train,
        codebook_id=codebook_id,
    )
    rewritten_nulls = _rewrite_records(
        records=null_records,
        n_train_tuples=n_train,
        codebook_id=codebook_id,
    )
    rewritten_candidate_table = candidate_table_rows(rewritten_candidates)

    rewritten_config = _rewritten_config(
        config,
        codebook_id=codebook_id,
        source_run_dir=run_dir,
    )
    save_run_config(rewritten_config, output_dir / "config_snapshot.yaml")
    save_json(split_manifest, output_dir / "split_manifest.json")
    save_json(rewritten_candidates, output_dir / "candidate_records.json")
    save_json(rewritten_candidate_table, output_dir / "candidate_table.json")
    save_json(rewritten_nulls, output_dir / "null_records.json")

    frontier_test = build_balanced_frontier(
        null_records=rewritten_nulls,
        split="test",
        bin_width_bits=config.method.null_bin_width_bits,
        min_family_count=config.method.null_min_family_count,
        balance_seed=config.method.frontier_balance_seed,
    )
    frontier_shift = build_balanced_frontier(
        null_records=rewritten_nulls,
        split="shift",
        bin_width_bits=config.method.null_bin_width_bits,
        min_family_count=config.method.null_min_family_count,
        balance_seed=config.method.frontier_balance_seed,
    )
    save_json(frontier_test.to_dict(), output_dir / "frontier_test.json")
    save_json(frontier_shift.to_dict(), output_dir / "frontier_shift.json")
    plot_paths = {
        "test": maybe_render_frontier_plot(
            run_dir=output_dir,
            split="test",
            candidate_records=rewritten_candidates,
            null_records=rewritten_nulls,
            frontier=frontier_test,
        ),
        "shift": maybe_render_frontier_plot(
            run_dir=output_dir,
            split="shift",
            candidate_records=rewritten_candidates,
            null_records=rewritten_nulls,
            frontier=frontier_shift,
        ),
    }

    test_group_ids = sorted(
        {
            str(record["group_id"])
            for record in split_manifest
            if str(record["split"]) == "test"
        }
    )
    support = compute_support(
        candidate_records=rewritten_candidates,
        null_records=rewritten_nulls,
        frontier_test=frontier_test,
        frontier_shift=frontier_shift,
        bootstrap_reps=config.method.bootstrap_n_reps,
        bootstrap_seed=config.seeds.bootstrap_seed,
        test_group_ids=test_group_ids,
    )
    save_json(support.summary, output_dir / "acceptance_summary.json")
    save_json(support.support_table, output_dir / "support_table.json")

    primary_summary = _load_primary_summary(run_dir)
    comparison = _build_comparison_summary(
        primary_run_dir=run_dir,
        primary_candidate_records=candidate_records,
        primary_summary=primary_summary,
        rewritten_candidate_records=rewritten_candidates,
        rewritten_support={"summary": support.summary, "support_table": support.support_table},
        rewritten_frontier_test=frontier_test,
        rewritten_frontier_shift=frontier_shift,
        codebook_id=codebook_id,
    )
    save_json(comparison, output_dir / "robustness_comparison.json")

    payload = {
        "status": "ok",
        "entrypoint": "python -m src.analysis.robustness_codebook",
        "source_run_dir": str(run_dir),
        "run_dir": str(output_dir),
        "codebook_id": codebook_id,
        "n_train_tuples": n_train,
        "primary_codebook_id": codebook_id_from_extras(config.extras),
        "candidate_records_path": str(output_dir / "candidate_records.json"),
        "null_records_path": str(output_dir / "null_records.json"),
        "frontier_test_path": str(output_dir / "frontier_test.json"),
        "frontier_shift_path": str(output_dir / "frontier_shift.json"),
        "acceptance_summary_path": str(output_dir / "acceptance_summary.json"),
        "support_table_path": str(output_dir / "support_table.json"),
        "robustness_comparison_path": str(output_dir / "robustness_comparison.json"),
        "plot_paths": plot_paths,
        "n_supported": support.summary["n_supported"],
        "supported_class_ids": support.summary["supported_class_ids"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
