# Non-Vacuous Causal Abstractions

Code and paper materials for:

**What Survives Control Calibration? A Full-Scope Negative Result for a Locked Minimum-Description Acceptance Criterion**

Author: **Ali Uyar**  
Affiliation: **Independent Researcher**

## Overview

This repository studies a narrow mechanistic-interpretability question:

**What should count as evidence that a neural network implements a proposed high-level algorithm?**

The project evaluates a locked family of top-down causal abstractions scored by structural description length plus held-out residual description length, and accepts candidates only when they beat matched spurious controls at comparable complexity under a held-out test and shift protocol.

The current source-of-truth outcome is a full-scope negative result: under the locked criterion family, no supported abstraction class is certified in the planted symbolic setting, the miniature IOI transformer, or GPT-2-small IOI.

## Repository Layout

- `paper/`  
  arXiv-ready manuscript source, figures, tables, and compiled PDF
- `src/`  
  experiment and analysis code
- `configs/`  
  explicit configuration files for planted, mini-IOI, and GPT-2 IOI runs
- `tests/`  
  regression and validation tests
- `docs/`  
  public-facing technical notes that remain useful outside the internal research log
- `artifacts/final_package/`  
  reduced-scope summary package kept for audit history
- `artifacts/final_package_full_locked/`  
  final full-scope paper-summary package

## Paper

The publication package lives in [paper/](paper).

Key files:

- [paper/main.tex](paper/main.tex)
- [paper/main.pdf](paper/main.pdf)
- [paper/refs.bib](paper/refs.bib)

## Reproducibility

The repository keeps the curated paper-facing summary artifacts in Git and treats raw run directories as generated outputs.

Typical local commands:

- `pytest -q`
- `python -m src.experiments.planted --config configs/planted/full_locked.yaml`
- `python -m src.experiments.mini_ioi --config configs/mini_ioi/full_locked.yaml`
- `python -m src.experiments.gpt2_ioi --config configs/gpt2_ioi/full_locked_cuda.yaml`
- `python -m src.analysis.null_frontier --run-dir results/<run_id>`
- `python -m src.analysis.acceptance --run-dir results/<run_id>`

## Git Hygiene

This public repo is intentionally cleaned for publication:

- internal planning / lab-notebook docs are gitignored locally
- review bundles and scratch exports under `artifacts/review/` are gitignored
- raw generated experiment outputs under `results/` are gitignored
- the tracked publication materials live in `paper/`

## Notes

- The negative result is about this locked criterion family and evaluation setup, not a universal impossibility claim about mechanistic interpretability.
- A software license has not been added automatically in this cleanup pass, because that should reflect your intended public-use terms.
