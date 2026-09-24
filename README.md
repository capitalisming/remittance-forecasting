# MIDI 2026 — adaptive forecasting of remittances under delayed publication

Working repo for a submission to MIDI 2026 (14th Machine Intelligence and Digital Interaction,
Springer LNNS, deadline 2026-10-08). Studies whether the value of online adaptation survives
realistic publication delay, using monthly remittance inflows to the Kyrgyz Republic (NBKR data,
2005–2026) as the test case — including the 2022 forecasting failure by the National Bank and the
World Bank, and a live retroactive redefinition of the series in 2024.

## Layout

- `src/` — pipeline code
  - `parse_nbkr.py`, `build_vintages.py` — parse and validate the 99 archived NBKR file versions
  - `realtime.py`, `realtime_diagnostics.py` — "what was known at issue date v" real-time information sets
  - `build_covariates.py` — FX/Brent/IMOEX covariates (currently unused in the pilot, see `panel.py` note)
  - `panel.py` — feature/target panel per issue date, ground truth = first release
  - `experts.py` — seasonal naive, expanding-window ridge, rolling-window ridge (E1/E2)
  - `chronos_expert.py` — Chronos-Bolt zero-shot expert (E3)
  - `fixed_share.py` — recursive fixed-share aggregation (K=4)
  - `run_experiment_a.py`, `run_experiment_d.py` — main experiments
  - `bootstrap.py`, `run_bootstrap.py`, `run_bootstrap_shock_vs_all.py` — moving-block bootstrap CIs
- `tests/` — leakage and correctness tests (`pytest -q`)
- `data/raw/` — archived NBKR vintages (99 files) and raw covariate downloads
- `data/processed/` — parsed vintage tables, covariates, release table
- `results/` — experiment outputs

## Environment

Developed with a short-path venv at `C:\cvenv` (Windows `MAX_PATH` broke torch/regex DLL loading
when installed under a long nested path). See `requirements.txt` for pinned versions
(torch, chronos-forecasting, pandas, scikit-learn, pytest).

## Status

Data pipeline, three experts, fixed-share combiner, Experiments A/D, bootstrap CIs and ablations
are done. Manuscript draft (6–8 pages, Springer LNCS template) not yet started. Intermediate
results are also written up as a shared doc (see project notes) and were reviewed by a second
Claude session against a 20-section review protocol before drafting begins.
