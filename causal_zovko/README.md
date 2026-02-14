# Causal Zovko Pipeline (Sanofi 2019-10-01)

This folder reproduces Zovko-style lead-lag analysis and runs causal discovery.

## Folder layout
- `scripts/` executable Python scripts
- `data/` intermediate datasets
- `results/` CSV outputs for inference results
- `figures/` plots
- `notes/` logs and notes
- `docs/` consolidated documentation and path map

## Important: external data in EA_recherche
The following inputs are **outside** `causal_zovko/` and are read from sibling folders/files under the same `EA_recherche` root.

- `../euronextparis/EuronextParis/EuronextParis_20191001_FR0000120578/FR0000120578/OrderUpdate_20191001_FR0000120578.csv`
- `../msc_decoded/merged_intervals.csv`
- `../decoded_full_trade_information.csv`

All scripts now build default paths from `scripts/` to the project root (`EA_recherche`) with `pathlib.Path`, so they are robust to current working directory.

## Scripts (what each one does)
1. `scripts/build_rlop_vol.py`
  - Build event-level RLOP and volatility from `OrderUpdate`.
  - Save `data/rlop_events.csv`, `data/vol_events.csv`, and time-bucket files.

2. `scripts/build_multivariate_features.py`
  - Merge RLOP/volatility/trade/microstructure features by bucket.
  - Uses external decoded trades (`../decoded_full_trade_information.csv`).
  - Save `data/features_{freq}.csv`.

3. `scripts/build_causal_dataset.py`
  - Add lagged variables for causal modeling.
  - Save `data/causal_dataset_1min.csv` (or user-provided output).

4. `scripts/zovko_xcf.py`
  - Compute XCF with surrogate null band.
  - Save plot in `figures/` and summary in `results/`.

5. `scripts/pc_cit.py`
  - Custom PC + Cai-Li-Zhang CIT implementation.
  - Save graph edges/adjacency and graph image.

6. `scripts/pc_cit_package.py`
  - pgmpy PC with custom CIT test and temporal constraints.
  - Save graph edges/adjacency and graph image.

## Suggested run order
1) `build_rlop_vol.py`  
2) `build_multivariate_features.py`  
3) `build_causal_dataset.py`  
4) `zovko_xcf.py`  
5) `pc_cit.py` or `pc_cit_package.py`

## Notes on relative paths
- If you run scripts from anywhere inside the workspace, defaults still resolve correctly.
- You can always override with CLI args (`--orderupdate`, `--intervals`, `--trades`, etc.).
- Detailed path mapping is in `docs/DATA_PATHS.md`.
