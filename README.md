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
from the corrected pipeline. All three P1/P2 items from that review round are done: a
multiple-comparisons caveat (Limitations), a lag-choice justification with a dev-period
sensitivity check (`tune_lags.py`, `results/lag_selection.json`) that keeps the frozen lag set, and
the exact Chronos model revision pinned in `chronos_expert.py`
(`MODEL_REVISION = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"`, obtained from a machine that could
reach `huggingface.co` — this sandbox couldn't) and cited in the paper's E3 paragraph.

`paper/main.tex` has now been compiled for real (pdflatex + bibtex, 3 passes, TeX Live 2023) against
the actual Springer `llncs` class v2.24 and `splncs04.bst` — not just structural self-checks. It
compiles clean: no undefined references, no missing-publisher bibtex warnings, no overfull table.
`llncs.cls` and `splncs04.bst` are Springer-licensed and are not checked into this repo
(gitignored under `paper/`) — fetch them yourself to recompile, e.g. from
`https://github.com/prosysscience/llncs`.

**Deadline and length confirmed against the MIDI 2026 site (2026-09-24):** submission deadline is
2026-10-08 as assumed; full papers are capped at **6–8pp in LNCS format** (short papers 4–6pp).
The draft was trimmed from 10pp to fit that cap — prose tightened across every section (no results,
tables, citations, or numeric findings were touched; verified by diffing every numeric token in
`paper/main.tex` before/after) — and now compiles to exactly **8 pages**. Remaining before
submission: the Chronos revision pin and P1/P2 items above are all done; nothing else is
currently blocking except a final proofread against the Springer template once MIDI 2026 confirms
which exact `.cls`/`.bst` it links from its submission page.
