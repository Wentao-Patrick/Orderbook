# DATA_PATHS (EA_recherche)

This file lists all external data references used by scripts in `causal_zovko/scripts`.

## Project root convention
- `EA_recherche/` is treated as project root.
- Scripts compute paths from `Path(__file__).resolve()`:
  - `scripts/` -> `causal_zovko/` -> `EA_recherche/`

## External data (outside causal_zovko)
1. OrderUpdate (order book events)
   - Relative path from `causal_zovko/`:
     - `../euronextparis/EuronextParis/EuronextParis_20191001_FR0000120578/FR0000120578/OrderUpdate_20191001_FR0000120578.csv`
   - Used by:
     - `build_rlop_vol.py`
     - `build_multivariate_features.py`

2. Continuous session intervals
   - Relative path from `causal_zovko/`:
     - `../msc_decoded/merged_intervals.csv`
   - Used by:
     - `build_rlop_vol.py`
     - `build_multivariate_features.py`

3. Decoded trades
   - Relative path from `causal_zovko/`:
     - `../decoded_full_trade_information.csv`
   - Used by:
     - `build_multivariate_features.py`

## Internal data (inside causal_zovko)
- `data/rlop_events.csv`
- `data/vol_events.csv`
- `data/features_{freq}.csv`
- `data/causal_dataset_1min.csv`
- `results/*.csv`
- `figures/*.png`

## Override behavior
All scripts accept CLI arguments to override defaults.
Example:
- `python causal_zovko/scripts/build_rlop_vol.py --orderupdate <custom_path>`
