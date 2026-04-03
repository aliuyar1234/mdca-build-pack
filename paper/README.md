# Paper

This directory is the tracked publication package for the project.

## Contents

- `main.tex` — manuscript source
- `main.pdf` — latest arXiv-ready compiled manuscript
- `refs.bib` — bibliography
- `neurips_2026.sty` — local venue style file used for the manuscript build
- `figures/` — paper figures
- `tables/` — paper tables
- `scripts/` — helper script used to generate figures/tables from summary data
- `paper_data.json` — compact data snapshot used by the paper assets

## Build

From this directory:

1. `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
2. `bibtex main`
3. `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
4. `pdflatex -interaction=nonstopmode -halt-on-error main.tex`

The repository gitignores LaTeX build byproducts, but keeps `main.pdf` as the current public manuscript artifact.
