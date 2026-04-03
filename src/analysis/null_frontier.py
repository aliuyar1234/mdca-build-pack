from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.core.config import load_run_config

from .frontier_core import (
    build_balanced_frontier,
    load_json,
    maybe_render_frontier_plot,
    save_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.null_frontier")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = load_run_config(run_dir / "config_snapshot.yaml")
    candidate_records = load_json(run_dir / "candidate_records.json")
    null_records = load_json(run_dir / "null_records.json")

    frontier_test = build_balanced_frontier(
        null_records=null_records,
        split="test",
        bin_width_bits=config.method.null_bin_width_bits,
        min_family_count=config.method.null_min_family_count,
        balance_seed=config.method.frontier_balance_seed,
    )
    frontier_shift = build_balanced_frontier(
        null_records=null_records,
        split="shift",
        bin_width_bits=config.method.null_bin_width_bits,
        min_family_count=config.method.null_min_family_count,
        balance_seed=config.method.frontier_balance_seed,
    )

    save_json(frontier_test.to_dict(), run_dir / "frontier_test.json")
    save_json(frontier_shift.to_dict(), run_dir / "frontier_shift.json")

    plot_paths = {
        "test": maybe_render_frontier_plot(
            run_dir=run_dir,
            split="test",
            candidate_records=candidate_records,
            null_records=null_records,
            frontier=frontier_test,
        ),
        "shift": maybe_render_frontier_plot(
            run_dir=run_dir,
            split="shift",
            candidate_records=candidate_records,
            null_records=null_records,
            frontier=frontier_shift,
        ),
    }

    summary = {
        "status": "ok",
        "entrypoint": "python -m src.analysis.null_frontier",
        "run_dir": str(run_dir),
        "n_candidates": len(candidate_records),
        "n_null_records": len(null_records),
        "test_valid_bins": sum(1 for item in frontier_test.bin_summaries if item["valid"]),
        "shift_valid_bins": sum(1 for item in frontier_shift.bin_summaries if item["valid"]),
        "test_domain": list(frontier_test.domain) if frontier_test.domain is not None else None,
        "shift_domain": list(frontier_shift.domain) if frontier_shift.domain is not None else None,
        "frontier_test_path": str(run_dir / "frontier_test.json"),
        "frontier_shift_path": str(run_dir / "frontier_shift.json"),
        "plot_paths": plot_paths,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
