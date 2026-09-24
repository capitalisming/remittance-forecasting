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
  - `tune_ridge.py` — grid search for ridge alpha (E1/E2), dev period only; writes `results/ridge_alpha_tuning.json`
  - `chronos_expert.py` — Chronos-Bolt zero-shot expert (E3)
  - `fixed_share.py` — recursive fixed-share aggregation (K=4)
  - `run_experiment_a.py`, `run_experiment_d.py` — main experiments
  - `build_expert_comparison.py`, `build_table1.py` — regenerate the expert-race table (rerun after any expert hyperparameter change)
  - `bootstrap.py`, `run_bootstrap.py`, `run_bootstrap_shock_vs_all.py` — moving-block bootstrap CIs
  - `check_revisions.py` — within-regime revision counts at various materiality thresholds (Section 3 of the paper)
- `tests/` — leakage and correctness tests (`pytest -q`)
- `data/raw/` — archived NBKR vintages (99 files) and raw covariate downloads
- `data/processed/` — parsed vintage tables, covariates, release table
- `results/` — experiment outputs

## Environment

Developed with a short-path venv at `C:\cvenv` (Windows `MAX_PATH` broke torch/regex DLL loading
when installed under a long nested path). See `requirements.txt` for pinned versions
(torch, chronos-forecasting, pandas, scikit-learn, pytest).

## Status

Data pipeline, three experts (ridge alpha for E1/E2 tuned via `tune_ridge.py`, not left as a
placeholder), fixed-share combiner, Experiments A/D, bootstrap CIs and ablations are done.
First manuscript draft (6–8 pages, Springer LNCS template) is written (`paper/main.tex`) and has
been through one full peer-review pass by a second Claude session against a 20-section review
protocol; that pass's two P0 findings (untuned ridge alpha, an undocumented materiality threshold
behind the "9 of 97 revisions" figure) have been addressed and all dependent results
(`results/*.csv`, `results/*.json`, and the paper's Tables 1–2 and in-text numbers) regenerated
from the corrected pipeline. Remaining before submission: confirm the MIDI 2026 deadline
(2026-10-08 as currently assumed), a real LaTeX compile against the Springer template (not
available locally — only structural self-checks have been run), and the P1/P2 items from that
review round (multiple-comparisons note, lag choice justification, pinned Chronos package/model
version).
