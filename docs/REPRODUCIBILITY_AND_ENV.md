# REPRODUCIBILITY_AND_ENV

## Expected environment
- Python 3.11
- PyTorch 2.x
- Hugging Face Transformers for GPT-2-small
- Hydra or OmegaConf for configs
- pytest for tests
- numpy / scipy / scikit-learn (for isotonic regression if convenient)
- matplotlib / pandas for analysis

## Dependency expectations
Keep dependencies minimal. Do not add heavy experiment-tracking or trainer frameworks unless clearly justified.

## Seed policy
- Global run seed required for every run
- Separate seeds for:
  - dataset generation
  - model initialization
  - candidate search restarts
  - bootstrap resampling
- Save all seeds with results

## Config policy
- Every run must have an explicit config file
- No hidden defaults that change scientific behavior
- Codebook variant, null-family selection, and split IDs must be config-visible

## Dataset assumptions
- S1 and S2 are generated from latent tuples
- S3 uses tokenizer-validated single-token names and fixed-length prompt families
- Split manifests are first-class artifacts

## Hardware assumptions
- S1 and S2 should run on a single workstation
- S3 should be feasible on a single modern GPU or CPU+GPU workflow at GPT-2-small scale
- If S3 compute is too slow, reduce restart counts but keep null/candidate tuning matched

## Experiment tracking expectations
For every run, save:
- config snapshot
- git commit hash or placeholder
- seed bundle
- split manifest hash
- candidate table
- null table
- frontier summary
- acceptance summary
- plots generated
- stderr/stdout logs if possible

## Artifact naming conventions
Use:
- `results/<timestamp>_<setting>_<run_name>/...`
- `artifacts/<timestamp>_<setting>_<figure_or_table_name>.<ext>`

## What must be saved
Minimum:
- raw candidate rows before filtering
- raw null rows before balancing
- balanced null bin summary
- all code-length components
- support decisions with reasons
- grouped bootstrap outputs

## Minimal rerun instructions
A rerun is minimally reproducible if another session can:
1. load the saved config,
2. rebuild the same splits,
3. regenerate the same candidate and null tables,
4. reproduce the same support decisions up to normal stochastic tolerance.

## What “minimally reproducible” means here
Not byte-for-byte identical neural training in all cases, but:
- same split manifests,
- same formulas,
- same codebook outputs from the same fitted parameters,
- same support logic,
- same qualitative claim status.
