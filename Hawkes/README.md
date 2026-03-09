# Hawkes Folder Layout

This directory is split by data frequency and use case:

- `hf_orderflow/`: High-frequency order-flow Hawkes workflows (buy/sell event streams).
- `daily_extreme_returns/`: Daily extreme-return Hawkes workflows (event extraction from daily close returns).

Each subfolder uses the same internal structure:

- `scripts/`: runnable Python scripts
- `data/input/`: source input data
- `data/derived/`: intermediate derived data
- `results/`: numeric outputs (CSV, bootstrap samples, summaries)
- `figures/`: plots and diagnostics
- `docs/`: notes and workflow documentation

