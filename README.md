# What Survives Control Calibration? A Full-Scope Negative Result for a Locked Minimum-Description Acceptance Criterion

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](paper/main.pdf)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-source-1D4ED8?style=flat-square&logo=latex&logoColor=white)](paper/main.tex)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square)](LICENSE)
[![Scope](https://img.shields.io/badge/Scope-Full--Scope%20Negative%20Result-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *What Survives Control Calibration? A Full-Scope Negative Result for a Locked Minimum-Description Acceptance Criterion*

This repository accompanies a disciplined negative-result paper in mechanistic interpretability. It asks a basic methodological question: what should count as evidence that a neural network implements a proposed high-level algorithm rather than merely admitting a good post-hoc fit? The paper evaluates a deliberately rigid criterion family — top-down causal abstractions scored by structural description length plus held-out residual code length, with acceptance gated on matched-null control calibration, a grouped-bootstrap test gap, and a no-refit shift gap — on three settings: a planted symbolic generator, a miniature learned IOI transformer, and GPT-2-small IOI. The central result is a full-scope negative finding: under the locked criterion, no supported abstraction class is certified in any setting.

## Abstract

Mechanistic interpretability still lacks clear acceptance criteria for when a neural network should count as implementing a proposed high-level algorithm rather than merely admitting a good post-hoc fit. We evaluate a deliberately rigid criterion that scores candidate causal abstractions by structural description length plus held-out residual code length and accepts them only if they beat matched spurious controls at comparable complexity, clear a grouped-bootstrap test-gap bound, and retain a positive shift gap without refitting. The paper uses only the final full-scope reruns that restore the entire locked candidate pool in three settings: a planted symbolic generator (S1), a miniature learned IOI transformer (S2), and GPT-2-small IOI (S3). No supported abstraction class is certified under either the primary or quantized robustness codebook. The negative result is nevertheless informative rather than empty. Null calibration changes decisions in all three settings; the exact planted oracle abstraction in S1 is frontier-defined yet still yields *g_test* = 0 and *g_shift* = 0; the closest frontier-defined candidates in S2 and S3 remain negative on both test and shift; and S3 retains many logged unevaluable high-complexity cells. We extract three design lessons for future evidence standards: control calibration is necessary but insufficient, support criteria must report frontier-domain exclusions separately from null-gap failure, and full configured-versus-realized candidate-pool accounting is part of the scientific claim. The result is a full-scope negative finding about this locked criterion family, not a universal impossibility theorem for mechanistic interpretability.

## Main Result

The headline is a structured negative result across three settings, based on final full-scope reruns that restore the entire locked candidate pool:

| Setting         | Cells (rec./unev.) | Valid bins (test/shift) | Best cand. frontier? | Δ to closest frontier-defined | *g_test* | *g_shift* | Supported classes |
| --------------- | ------------------ | ----------------------- | -------------------- | ----------------------------- | -------- | --------- | ----------------- |
| S1 planted      | 120 / 0            | 3 / 3                   | No                   | 2.008                         | 0.000    | 0.000     | 0                 |
| S2 mini-IOI     | 116 / 4            | 3 / 3                   | No                   | 0.066                         | -0.155   | -15.582   | 0                 |
| S3 GPT-2 IOI    | 92 / 28            | 2 / 2                   | No                   | 0.198                         | -2.340   | -37.016   | 0                 |

Claim posture under the locked criterion:

| Claim | Short form                                                              | Status               |
| ----- | ----------------------------------------------------------------------- | -------------------- |
| C1    | Fit-alone is fragile in the locked family.                              | weakened             |
| C2    | The criterion recovers the planted true abstraction.                    | unsupported          |
| C3    | Null calibration is necessary.                                          | supported            |
| C4    | Supported classes beat matched nulls on test.                           | unsupported          |
| C5    | Accepted classes keep a positive shift gap.                             | unsupported          |
| C6    | Main conclusions are stable under a robustness code.                    | supported            |
| C7    | Output is a supported class, not necessarily a unique interpretation.   | partially supported  |
| C8    | The failure record supports a publishable negative-result paper.        | supported            |

The failure pattern is structured, not generic. Null calibration materially changes decisions in all three settings; the global best candidate is frontier-ineligible in every setting; S2 and S3 fail more strongly because their closest frontier-defined candidates are already negative on both test and shift; and S3 additionally exposes a breadth problem — many high-complexity cells are configured but unevaluable.

## Contributions

1. We specify and fully execute a locked control-calibrated acceptance criterion for top-down causal abstractions, including candidate search, structural and residual code lengths, matched null families, balanced null frontiers, grouped-bootstrap test gaps, and no-refit shift gaps.
2. We evaluate the full configured candidate pool in three settings (S1 planted symbolic, S2 miniature learned IOI transformer, S3 GPT-2-small IOI) with explicit accounting for realized versus unevaluable cells, instead of drawing conclusions from a reduced search slice.
3. We show that no supported abstraction class is certified under either the primary or quantized robustness codebook, and that the planted oracle abstraction in S1 also fails under the locked criterion.
4. We distill methodological lessons from the failure: null calibration is necessary but insufficient, eligibility geometry must be surfaced separately from null-gap failure, and candidate-pool coverage belongs in the scientific claim rather than in implementation footnotes.

## Scope

This paper is a limits result for a particular acceptance family, not a universal impossibility claim.

- one locked candidate family: four high-level models, three site budgets, ten map-family/hyperparameter cells (120 configured cells per setting)
- one locked acceptance rule: best-bits class, positive grouped-bootstrap lower bound on held-out test gap, positive no-refit shift gap
- three settings: planted symbolic (S1), miniature learned IOI transformer (S2), GPT-2-small IOI (S3)
- two codebooks: primary BIC-style parameter code, and a quantized robustness codebook
- only the final full-scope reruns that restore the entire configured candidate pool; earlier reduced-scope pilots are not used as paper evidence

The negative result is specific to this locked criterion family and evaluation protocol. It does not claim that mechanistic interpretability in general is impossible.

## Paper

- Compiled PDF: [`paper/main.pdf`](paper/main.pdf)
- LaTeX source: [`paper/main.tex`](paper/main.tex)
- Bibliography: [`paper/refs.bib`](paper/refs.bib)
- Tables and figures: [`paper/tables/`](paper/tables/), [`paper/figures/`](paper/figures/)
- Paper data snapshot: [`paper/paper_data.json`](paper/paper_data.json)

## Repository Layout

- [`paper/`](paper/) — arXiv-ready manuscript source, figures, tables, and compiled PDF
- [`src/`](src/) — experiment and analysis code
- [`configs/`](configs/) — explicit experiment configurations for planted, mini-IOI, and GPT-2 IOI runs
- [`tests/`](tests/) — regression and validation tests
- [`artifacts/final_package_full_locked/`](artifacts/final_package_full_locked/) — final full-scope summary package used by the paper
- [`artifacts/final_package/`](artifacts/final_package/) — reduced-scope summary package retained for audit history
- [`docs/`](docs/) — method specification and reproducibility notes

Generated run directories, review bundles, and temporary build byproducts are intentionally excluded from version control.

## Reproducibility

- [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md) — locked method specification
- [`docs/REPRODUCIBILITY_AND_ENV.md`](docs/REPRODUCIBILITY_AND_ENV.md) — environment and reproducibility notes
- [`artifacts/final_package_full_locked/final_claim_support_summary.json`](artifacts/final_package_full_locked/final_claim_support_summary.json) — final claim-support summary
- [`artifacts/final_package_full_locked/figure_table_manifest.json`](artifacts/final_package_full_locked/figure_table_manifest.json) — figure and table manifest
- [`artifacts/final_package_full_locked/audit_artifact_manifest.json`](artifacts/final_package_full_locked/audit_artifact_manifest.json) — audit artifact manifest

Typical entrypoints:

```bash
pytest -q
python -m src.experiments.planted --config configs/planted/full_locked.yaml
python -m src.experiments.mini_ioi --config configs/mini_ioi/full_locked.yaml
python -m src.experiments.gpt2_ioi --config configs/gpt2_ioi/full_locked_cuda.yaml
python -m src.analysis.null_frontier --run-dir results/<run_id>
python -m src.analysis.acceptance   --run-dir results/<run_id>
```

## Citation

```bibtex
@unpublished{uyar2026mdca,
  author = {Uyar, Ali},
  title  = {What Survives Control Calibration? A Full-Scope Negative Result for a Locked Minimum-Description Acceptance Criterion},
  year   = {2026},
  note   = {Independent research}
}
```
