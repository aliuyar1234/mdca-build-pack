# Non-Vacuous Causal Abstractions

Code, manuscript, and curated summary artifacts for the paper:

**What Survives Control Calibration? A Full-Scope Negative Result for a Locked Minimum-Description Acceptance Criterion**

Ali Uyar  
Independent Researcher

## Summary

This repository studies a focused mechanistic-interpretability question:

**What should count as evidence that a neural network implements a proposed high-level algorithm?**

The project evaluates a locked family of top-down causal abstractions scored by structural description length plus held-out residual description length. A candidate is accepted only if it beats matched spurious controls at comparable complexity, clears a held-out test criterion, and retains a positive shift gap without refitting.

The current source-of-truth conclusion is a **full-scope negative result**: under the locked criterion family, no supported abstraction class is certified in any of the three evaluation settings.

## Main Findings

- The final paper result is based on **full-scope reruns**, not reduced search slices.
- Across the planted symbolic setting, the miniature IOI transformer, and GPT-2-small IOI, the criterion certifies **no supported abstraction class**.
- Matched null calibration materially changes decisions in all settings, so the criterion is not redundant with simple code-length ranking.
- The negative result is **specific to this locked criterion family and evaluation protocol**. It is not a universal impossibility claim about mechanistic interpretability.

## Repository Contents

- [paper/](paper)  
  arXiv-ready manuscript source, figures, tables, helper scripts, and compiled PDF
- [src/](src)  
  experiment and analysis code
- [configs/](configs)  
  explicit experiment configurations for planted, mini-IOI, and GPT-2 IOI runs
- [tests/](tests)  
  regression and validation tests
- [docs/METHOD_SPEC.md](docs/METHOD_SPEC.md)  
  locked method specification
- [docs/REPRODUCIBILITY_AND_ENV.md](docs/REPRODUCIBILITY_AND_ENV.md)  
  environment and reproducibility notes
- [artifacts/final_package/](artifacts/final_package)  
  reduced-scope summary package retained for audit history
- [artifacts/final_package_full_locked/](artifacts/final_package_full_locked)  
  final full-scope summary package used by the paper

## Paper

The publication package is tracked under [paper/](paper).

Key files:

- [paper/main.pdf](paper/main.pdf)
- [paper/main.tex](paper/main.tex)
- [paper/refs.bib](paper/refs.bib)
- [paper/README.md](paper/README.md)

The final full-scope paper-facing summaries are:

- [artifacts/final_package_full_locked/final_claim_support_summary.json](artifacts/final_package_full_locked/final_claim_support_summary.json)
- [artifacts/final_package_full_locked/figure_table_manifest.json](artifacts/final_package_full_locked/figure_table_manifest.json)
- [artifacts/final_package_full_locked/audit_artifact_manifest.json](artifacts/final_package_full_locked/audit_artifact_manifest.json)

## Running the Code

Typical entrypoints:

- `pytest -q`
- `python -m src.experiments.planted --config configs/planted/full_locked.yaml`
- `python -m src.experiments.mini_ioi --config configs/mini_ioi/full_locked.yaml`
- `python -m src.experiments.gpt2_ioi --config configs/gpt2_ioi/full_locked_cuda.yaml`
- `python -m src.analysis.null_frontier --run-dir results/<run_id>`
- `python -m src.analysis.acceptance --run-dir results/<run_id>`

## Reproducibility Policy

This repository keeps the **paper-facing code, manuscript, and curated final summary artifacts** in version control.

The following are intentionally excluded from the public Git history:

- raw generated run directories under `results/`
- review bundles and scratch exports under `artifacts/review/`
- internal planning / lab-notebook style documents
- temporary cache files and LaTeX build byproducts

This keeps the public repository focused on the manuscript, reproducible code, and the final evidence summaries rather than on a full local experiment dump.

## Status

The current public repo state reflects the cleaned publication-oriented layout used for paper release preparation:

- manuscript promoted to `paper/`
- curated summary artifacts retained
- raw results excluded from Git by default
- tests passing locally on the tracked codebase

## Citation

If you cite this work before a final arXiv identifier or venue record is available, please cite the manuscript title and author information from [paper/main.pdf](paper/main.pdf).

## License

This repository is released under the [MIT License](LICENSE).
