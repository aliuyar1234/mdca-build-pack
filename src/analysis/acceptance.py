from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.core.config import load_run_config

from .acceptance_core import compute_support
from .frontier_core import FrontierResult, load_json, save_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.acceptance")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = load_run_config(run_dir / "config_snapshot.yaml")
    candidate_records = load_json(run_dir / "candidate_records.json")
    null_records = load_json(run_dir / "null_records.json")
    frontier_test = FrontierResult.from_dict(load_json(run_dir / "frontier_test.json"))
    frontier_shift = FrontierResult.from_dict(load_json(run_dir / "frontier_shift.json"))
    split_manifest = load_json(run_dir / "split_manifest.json")
    test_group_ids = sorted(
        {
            str(record["group_id"])
            for record in split_manifest
            if str(record["split"]) == "test"
        }
    )

    support = compute_support(
        candidate_records=candidate_records,
        null_records=null_records,
        frontier_test=frontier_test,
        frontier_shift=frontier_shift,
        bootstrap_reps=config.method.bootstrap_n_reps,
        bootstrap_seed=config.seeds.bootstrap_seed,
        test_group_ids=test_group_ids,
    )
    save_json(support.summary, run_dir / "acceptance_summary.json")
    save_json(support.support_table, run_dir / "support_table.json")

    payload = {
        "status": "ok",
        "entrypoint": "python -m src.analysis.acceptance",
        "run_dir": str(run_dir),
        "n_supported": support.summary["n_supported"],
        "supported_class_ids": support.summary["supported_class_ids"],
        "control_calibration_changed_decision": support.summary[
            "control_calibration_changed_decision"
        ],
        "acceptance_summary_path": str(run_dir / "acceptance_summary.json"),
        "support_table_path": str(run_dir / "support_table.json"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
