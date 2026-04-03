from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.planted_followup_summary")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    candidate_records = _load_json(run_dir / "candidate_records.json")
    null_records = _load_json(run_dir / "null_records.json")
    frontier_test = _load_json(run_dir / "frontier_test.json")
    frontier_shift = _load_json(run_dir / "frontier_shift.json")
    support_table = _load_json(run_dir / "support_table.json")
    oracle_support = _load_json(run_dir / "oracle_support.json")
    oracle_overlap = _load_json(run_dir / "oracle_site_overlap.json")

    def budget_summary(site_budget: int) -> dict[str, Any]:
        rows = [row for row in support_table if int(row["site_budget"]) == site_budget]
        candidate_rows = [
            record for record in candidate_records if int(record["site_budget"]) == site_budget
        ]
        if not rows:
            return {
                "site_budget": site_budget,
                "n_candidates": 0,
                "n_supported": 0,
                "best_candidate_id": None,
                "best_test_total_bits_per_example": None,
            }
        best_row = min(rows, key=lambda row: (float(row["test_total_bits_per_example"]), str(row["candidate_id"])))
        return {
            "site_budget": site_budget,
            "n_candidates": len(candidate_rows),
            "n_supported": sum(1 for row in rows if bool(row["supported"])),
            "best_candidate_id": best_row["candidate_id"],
            "best_test_total_bits_per_example": best_row["test_total_bits_per_example"],
            "best_support_reason": best_row["support_reason"],
        }

    per_budget_summary = [budget_summary(site_budget) for site_budget in sorted({int(row["site_budget"]) for row in support_table})]
    per_family_null_counts_test = [
        {
            "bin_start": item["bin_start"],
            "bin_end": item["bin_end"],
            "bin_center": item["bin_center"],
            "valid": item["valid"],
            "counts_by_family": item["counts_by_family"],
            "k_j": item["k_j"],
            "balanced_quantile": item["balanced_quantile"],
        }
        for item in frontier_test["bin_summaries"]
    ]
    per_family_null_counts_shift = [
        {
            "bin_start": item["bin_start"],
            "bin_end": item["bin_end"],
            "bin_center": item["bin_center"],
            "valid": item["valid"],
            "counts_by_family": item["counts_by_family"],
            "k_j": item["k_j"],
            "balanced_quantile": item["balanced_quantile"],
        }
        for item in frontier_shift["bin_summaries"]
    ]
    low_complexity_h_true_other = [
        item
        for item in oracle_overlap["all_h_true_other_candidates"]
        if float(item["test_total_bits_per_example"]) < 40.0
    ]

    payload = {
        "run_dir": str(run_dir),
        "n_candidate_records": len(candidate_records),
        "n_null_records": len(null_records),
        "per_budget_summary": per_budget_summary,
        "oracle_support_row": oracle_support["support_row"],
        "oracle_site_overlap": oracle_overlap["oracle_site_overlap"],
        "low_complexity_h_true_other_candidates": low_complexity_h_true_other,
        "frontier_test_bin_counts": per_family_null_counts_test,
        "frontier_shift_bin_counts": per_family_null_counts_shift,
    }
    output_path = run_dir / "followup_summary.json"
    _save_json(output_path, payload)
    print(
        json.dumps(
            {
                "status": "ok",
                "entrypoint": "python -m src.analysis.planted_followup_summary",
                "run_dir": str(run_dir),
                "followup_summary_path": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
