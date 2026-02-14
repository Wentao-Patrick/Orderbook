# Scripts Overview

## build_rlop_vol.py
- Purpose: parse OrderUpdate, reconstruct best bid/ask, compute RLOP and volatility.
- Main outputs: `data/rlop_events.csv`, `data/vol_events.csv`, time-bucket CSV files.

## build_multivariate_features.py
- Purpose: aggregate RLOP/vol/trade/microstructure features by frequency bucket.
- Main outputs: `data/features_{freq}.csv`.

## build_causal_dataset.py
- Purpose: create lagged causal modeling table from selected feature columns.
- Main outputs: `data/causal_dataset_1min.csv` (default).

## zovko_xcf.py
- Purpose: estimate cross-correlation function (XCF) and null confidence band via surrogates.
- Main outputs: `figures/zovko_xcf_{side}_{bucket}.png`, `results/xcorr_summary_{side}_{bucket}.csv`.

## pc_cit.py
- Purpose: custom PC algorithm + Cai-Li-Zhang CIT with temporal orientation constraints.
- Main outputs: `results/pc_edges.csv`, `results/pc_adjmatrix.csv`, `figures/pc_graph.png`.

## pc_cit_package.py
- Purpose: pgmpy PC backend with custom CIT test and expert knowledge constraints.
- Main outputs: `results/pc_edges.csv`, `results/pc_adjmatrix.csv`, `figures/pc_graph.png`.
