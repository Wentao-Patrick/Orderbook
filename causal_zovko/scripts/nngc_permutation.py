"""
Neural-network Granger causality (NNGC) with permutation significance test.

Method (per target y_t):
1) Baseline uses own lagged predictors of y (e.g., y_lag1, y_lag2, ...).
2) Candidate lagged features from other variables are added one-by-one (forward stepwise).
3) Improvement is delta = MSE_old - MSE_new on validation set.
4) For each candidate, permutation test shuffles candidate time order to build null improvements.
5) Accept candidate as causal only if:
   - delta > improvement_tol
   - permutation p-value < alpha

Default input/output locations (relative to EA_recherche root):
- input: causal_zovko/data/causal_dataset_1min.csv
- outputs:
    - causal_zovko/results/nngc_permutation_tests.csv
    - causal_zovko/results/nngc_edges.csv
    - causal_zovko/results/nngc_adjmatrix.csv
    - causal_zovko/figures/nngc_graph.png
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
CAUSAL_ZOVKO_DIR = SCRIPT_DIR.parent

# Easy-to-edit runtime defaults (no need to pass from CLI every time).
DEFAULT_DEVICE = "cuda"
DEFAULT_PERM_WORKERS = 8
SAVE_EVERY_TARGET = True

VARS_LIST = [
    "rlop_ask_mean",
    "rlop_bid_mean",
    "vol_bid_mean",
    "vol_ask_mean",
    "spread_mean",
    "imbalance_ob_mean",
    "imb_of",
    "rlop_ask_mean_lag1",
    "rlop_bid_mean_lag1",
    "vol_bid_mean_lag1",
    "vol_ask_mean_lag1",
    "spread_mean_lag1",
    "imbalance_ob_mean_lag1",
    "imb_of_lag1",
    "rlop_ask_mean_lag2",
    "rlop_bid_mean_lag2",
    "vol_bid_mean_lag2",
    "vol_ask_mean_lag2",
    "spread_mean_lag2",
    "imbalance_ob_mean_lag2",
    "imb_of_lag2",
    "rlop_ask_mean_lag5",
    "rlop_bid_mean_lag5",
    "vol_bid_mean_lag5",
    "vol_ask_mean_lag5",
    "spread_mean_lag5",
    "imbalance_ob_mean_lag5",
    "imb_of_lag5",
]


def parse_lag(name: str) -> int:
    match = re.search(r"_lag(\d+)\b", name)
    return int(match.group(1)) if match else 0


def base_name(name: str) -> str:
    return re.sub(r"_lag\d+\b", "", name)


def split_timewise(df: pd.DataFrame, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)
    train_df = df.iloc[: n - n_val].copy()
    val_df = df.iloc[n - n_val :].copy()
    return train_df, val_df


def mse_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true.astype(float) - y_pred.astype(float)
    return float(np.mean(diff * diff))


def build_mlp(input_dim: int, hidden: Tuple[int, ...]) -> nn.Module:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


def standardize_train_val(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    return X_train_scaled, X_val_scaled


def resolve_device(device_arg: str) -> torch.device:
    want = device_arg.strip().lower()
    if want == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            "cuda_available=False. "
            "Install a CUDA-enabled PyTorch build or run with --device cpu."
        )
    return torch.device(want)


def init_linear_weights(model: nn.Module, seed: int) -> None:
    # Initialize linear layers with a local RNG so parallel permutations do not race on torch global RNG.
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for layer in model.modules():
            if not isinstance(layer, nn.Linear):
                continue
            fan_in = int(layer.weight.shape[1])
            bound = np.sqrt(1.0 / max(1, fan_in))
            w = rng.uniform(-bound, bound, size=tuple(layer.weight.shape)).astype(np.float32)
            layer.weight.copy_(torch.from_numpy(w).to(layer.weight.device))
            if layer.bias is not None:
                b = rng.uniform(-bound, bound, size=tuple(layer.bias.shape)).astype(np.float32)
                layer.bias.copy_(torch.from_numpy(b).to(layer.bias.device))


def evaluate_mse(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    *,
    hidden: Tuple[int, ...],
    alpha: float,
    max_iter: int,
    random_state: int,
    device: torch.device,
    use_amp: bool,
    fit_pbar=None,
) -> float:
    use_amp = bool(use_amp and device.type == "cuda")

    y_train = train_df[target_col].to_numpy(dtype=float)
    y_val = val_df[target_col].to_numpy(dtype=float)

    if len(feature_cols) == 0:
        pred = np.full_like(y_val, float(np.mean(y_train)), dtype=float)
        if fit_pbar is not None:
            fit_pbar.update(1)
        return mse_numpy(y_val, pred)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)

    X_train, X_val = standardize_train_val(X_train, X_val)

    n_train = X_train.shape[0]
    n_inner_val = max(1, int(round(0.15 * n_train)))
    n_inner_val = min(n_inner_val, n_train - 1)
    split_idx = n_train - n_inner_val

    X_tr = X_train[:split_idx]
    y_tr = y_train[:split_idx]
    X_inner_val = X_train[split_idx:]
    y_inner_val = y_train[split_idx:]

    model = build_mlp(X_train.shape[1], hidden).to(device)
    init_linear_weights(model, random_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=alpha)
    criterion = nn.MSELoss()
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32, device=device)
    X_inner_val_t = torch.tensor(X_inner_val, dtype=torch.float32, device=device)
    y_inner_val_t = torch.tensor(y_inner_val.reshape(-1, 1), dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    shuffle_rng = np.random.default_rng(random_state)

    batch_size = min(64, len(X_tr_t))
    patience = 15
    best_val_loss = float("inf")
    best_state = None
    no_improve_epochs = 0

    for _ in range(max_iter):
        perm = torch.tensor(shuffle_rng.permutation(len(X_tr_t)), dtype=torch.long, device=device)
        X_tr_epoch = X_tr_t[perm]
        y_tr_epoch = y_tr_t[perm]

        model.train()
        for start in range(0, len(X_tr_epoch), batch_size):
            end = start + batch_size
            xb = X_tr_epoch[start:end]
            yb = y_tr_epoch[start:end]
            optimizer.zero_grad()
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                pred_b = model(xb)
                loss = criterion(pred_b, yb)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                val_pred = model(X_inner_val_t)
                val_loss = float(criterion(val_pred, y_inner_val_t).item())

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(X_val_t).cpu().numpy().reshape(-1)

    if fit_pbar is not None:
        fit_pbar.update(1)
    return mse_numpy(y_val, pred)


def permutation_improvement_once(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    current_features: List[str],
    candidate: str,
    mse_old: float,
    *,
    perm_seed: int,
    hidden: Tuple[int, ...],
    alpha: float,
    max_iter: int,
    random_state: int,
    device: torch.device,
    use_amp: bool,
    fit_pbar=None,
) -> float:
    local_rng = np.random.default_rng(perm_seed)
    train_perm = train_df.copy()
    val_perm = val_df.copy()

    train_perm[candidate] = local_rng.permutation(train_perm[candidate].to_numpy())
    val_perm[candidate] = local_rng.permutation(val_perm[candidate].to_numpy())

    mse_perm = evaluate_mse(
        train_perm,
        val_perm,
        target_col,
        current_features + [candidate],
        hidden=hidden,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        device=device,
        use_amp=use_amp,
        fit_pbar=fit_pbar,
    )
    return mse_old - mse_perm


def permutation_pvalue(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    current_features: List[str],
    candidate: str,
    mse_old: float,
    observed_improvement: float,
    *,
    n_perm: int,
    hidden: Tuple[int, ...],
    alpha: float,
    max_iter: int,
    random_state: int,
    rng: np.random.Generator,
    device: torch.device,
    use_amp: bool,
    perm_workers: int,
    show_progress: bool,
    progress_desc: str,
    fit_pbar=None,
) -> Tuple[float, float, float]:
    null_improvements = []
    perm_seeds = rng.integers(0, np.iinfo(np.int64).max, size=n_perm, dtype=np.int64)

    if perm_workers > 1 and n_perm > 1:
        max_workers = min(perm_workers, n_perm)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    permutation_improvement_once,
                    train_df,
                    val_df,
                    target_col,
                    current_features,
                    candidate,
                    mse_old,
                    perm_seed=int(seed),
                    hidden=hidden,
                    alpha=alpha,
                    max_iter=max_iter,
                    random_state=random_state,
                    device=device,
                    use_amp=use_amp,
                    fit_pbar=None,
                )
                for seed in perm_seeds
            ]
            completed_iter = as_completed(futures)
            if show_progress and tqdm is not None:
                completed_iter = tqdm(completed_iter, total=n_perm, desc=progress_desc, leave=False)
            for future in completed_iter:
                null_improvements.append(float(future.result()))
                if fit_pbar is not None:
                    fit_pbar.update(1)
    else:
        perm_iter = perm_seeds
        if show_progress and tqdm is not None:
            perm_iter = tqdm(perm_iter, desc=progress_desc, leave=False)

        for seed in perm_iter:
            improvement = permutation_improvement_once(
                train_df,
                val_df,
                target_col,
                current_features,
                candidate,
                mse_old,
                perm_seed=int(seed),
                hidden=hidden,
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
                device=device,
                use_amp=use_amp,
                fit_pbar=fit_pbar,
            )
            null_improvements.append(improvement)

    null_arr = np.asarray(null_improvements, dtype=float)
    count = int(np.sum(null_arr >= observed_improvement))
    pvalue = (count + 1.0) / (n_perm + 1.0)
    return float(pvalue), float(null_arr.mean()), float(null_arr.std(ddof=0))


def draw_causal_graph(edges: pd.DataFrame, out_path: Path) -> None:
    if edges.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No significant NN-GC edges", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    causes = sorted(edges["cause"].unique(), key=lambda c: (-parse_lag(c), base_name(c), c))
    effects = sorted(edges["effect"].unique())

    y_causes = np.linspace(1.0, 0.0, len(causes)) if len(causes) > 1 else np.array([0.5])
    y_effects = np.linspace(1.0, 0.0, len(effects)) if len(effects) > 1 else np.array([0.5])

    pos: Dict[str, Tuple[float, float]] = {}
    for idx, node in enumerate(causes):
        pos[node] = (0.1, float(y_causes[idx]))
    for idx, node in enumerate(effects):
        pos[node] = (0.9, float(y_effects[idx]))

    fig_h = max(6.0, 0.28 * (len(causes) + len(effects)))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    for node in causes:
        x, y = pos[node]
        ax.scatter(x, y, s=1400, c="#E3F2FD", edgecolors="#1E88E5", linewidths=1.2, zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=9)

    for node in effects:
        x, y = pos[node]
        ax.scatter(x, y, s=1400, c="#FFF3E0", edgecolors="#FB8C00", linewidths=1.2, zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=9)

    for _, row in edges.iterrows():
        src = row["cause"]
        dst = row["effect"]
        imp = float(row["improvement"])
        pval = float(row["pvalue"])
        lw = 0.8 + 3.0 * min(max(imp, 0.0), 1.0)

        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ax.annotate(
            "",
            xy=(x1 - 0.05, y1),
            xytext=(x0 + 0.05, y0),
            arrowprops=dict(arrowstyle="->", lw=lw, color="#616161", alpha=0.85),
            zorder=2,
        )

        xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        ax.text(xm, ym + 0.018, f"Δ={imp:.4g}, p={pval:.3g}", fontsize=7, color="#424242", ha="center")

    ax.text(0.1, 1.06, "Causes (lagged)", ha="center", va="bottom", fontsize=11, color="#0D47A1")
    ax.text(0.9, 1.06, "Effects (current)", ha="center", va="bottom", fontsize=11, color="#E65100")
    ax.set_title("NNGC causal graph (cause -> effect)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_adj_df(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty:
        return pd.DataFrame()
    causes = sorted(edges_df["cause"].unique(), key=lambda c: (-parse_lag(c), base_name(c), c))
    effects = sorted(edges_df["effect"].unique())
    adj_df = pd.DataFrame(0, index=causes, columns=effects, dtype=int)
    for _, row in edges_df.iterrows():
        adj_df.loc[row["cause"], row["effect"]] = 1
    return adj_df


def save_outputs(
    test_rows: List[dict],
    accepted_edges: List[dict],
    out_tests: Path,
    out_edges: Path,
    out_adj: Path,
    out_fig: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tests_df = pd.DataFrame(test_rows)
    edges_df = pd.DataFrame(accepted_edges)
    adj_df = build_adj_df(edges_df)

    tests_df.to_csv(out_tests, index=False)
    edges_df.to_csv(out_edges, index=False)
    adj_df.to_csv(out_adj)
    draw_causal_graph(edges_df, out_fig)
    return tests_df, edges_df, adj_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default=str(CAUSAL_ZOVKO_DIR / "data" / "causal_dataset_1min.csv"))
    parser.add_argument("--out_tests", default=str(CAUSAL_ZOVKO_DIR / "results" / "nngc_permutation_tests.csv"))
    parser.add_argument("--out_edges", default=str(CAUSAL_ZOVKO_DIR / "results" / "nngc_edges.csv"))
    parser.add_argument("--out_adj", default=str(CAUSAL_ZOVKO_DIR / "results" / "nngc_adjmatrix.csv"))
    parser.add_argument("--out_fig", default=str(CAUSAL_ZOVKO_DIR / "figures" / "nngc_graph.png"))
    parser.add_argument(
        "--vars",
        nargs="+",
        default=VARS_LIST,
        help="Feature columns to use. Default matches pc_cit.py VARS_LIST.",
    )

    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--hidden", default="32,16", help="MLP hidden layers, e.g. '64,32'")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization alpha for MLP")
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["auto", "cpu", "cuda"],
        help=f"Training device. Default is '{DEFAULT_DEVICE}'. Use 'auto' to fallback to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--perm_workers",
        type=int,
        default=DEFAULT_PERM_WORKERS,
        help=f"Number of parallel workers for permutation fits. Script default is {DEFAULT_PERM_WORKERS}.",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed precision on CUDA.",
    )

    parser.add_argument("--improvement_tol", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_perm", type=int, default=200)
    parser.add_argument("--max_candidates_per_target", type=int, default=-1)
    parser.add_argument(
        "--hide_progress",
        action="store_false",
        dest="show_progress",
        help="Disable progress bars during training/tests",
    )
    parser.set_defaults(show_progress=True)
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_tests = Path(args.out_tests)
    out_edges = Path(args.out_edges)
    out_adj = Path(args.out_adj)
    out_fig = Path(args.out_fig)

    out_tests.parent.mkdir(parents=True, exist_ok=True)
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    out_adj.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    hidden = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())
    device = resolve_device(args.device)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    perm_workers = max(1, int(args.perm_workers))

    df = pd.read_csv(in_csv)
    if "bucket" in df.columns:
        df["bucket"] = pd.to_datetime(df["bucket"], utc=True, errors="coerce")
        df = df.sort_values("bucket").reset_index(drop=True)

    if args.vars:
        if len(args.vars) == 1 and "," in args.vars[0]:
            vars_list = [v.strip() for v in args.vars[0].split(",") if v.strip()]
        else:
            vars_list = list(args.vars)
    else:
        vars_list = list(VARS_LIST)

    missing = [v for v in vars_list if v not in df.columns]
    if missing:
        print(f"Error: missing columns: {missing}")
        return

    df_model = df[vars_list].copy()
    lagged_cols = [c for c in vars_list if parse_lag(c) > 0]
    target_cols = [c for c in vars_list if parse_lag(c) == 0]

    train_df, val_df = split_timewise(df_model, args.val_ratio)
    rng = np.random.default_rng(args.random_state)

    print(f"training device: {device} | amp={use_amp} | perm_workers={perm_workers}")

    test_rows = []
    accepted_edges = []

    fit_pbar = None
    if args.show_progress and tqdm is not None:
        total_candidates = 0
        for target in target_cols:
            cands = [c for c in lagged_cols if base_name(c) != target]
            if args.max_candidates_per_target > 0:
                cands = cands[: args.max_candidates_per_target]
            total_candidates += len(cands)
        total_fits = len(target_cols) + total_candidates * (1 + args.n_perm)
        fit_pbar = tqdm(total=total_fits, desc="Model fits")

    target_iter = target_cols
    if args.show_progress and tqdm is not None:
        target_iter = tqdm(target_cols, desc="Targets")

    for target in target_iter:
        t0 = time.perf_counter()
        tests_before = len(test_rows)
        edges_before = len(accepted_edges)

        own_lags = sorted(
            [c for c in lagged_cols if base_name(c) == target],
            key=lambda name: (parse_lag(name), name),
        )
        candidates = sorted(
            [c for c in lagged_cols if base_name(c) != target],
            key=lambda name: (parse_lag(name), base_name(name), name),
        )
        if args.max_candidates_per_target > 0:
            candidates = candidates[: args.max_candidates_per_target]

        selected = own_lags.copy()

        candidate_iter = candidates
        if args.show_progress and tqdm is not None:
            candidate_iter = tqdm(candidates, desc=f"Candidates[{target}]", leave=False)

        mse_current = evaluate_mse(
            train_df,
            val_df,
            target,
            selected,
            hidden=hidden,
            alpha=args.l2,
            max_iter=args.max_iter,
            random_state=args.random_state,
            device=device,
            use_amp=use_amp,
            fit_pbar=fit_pbar,
        )

        step = 0
        for candidate in candidate_iter:
            step += 1
            mse_new = evaluate_mse(
                train_df,
                val_df,
                target,
                selected + [candidate],
                hidden=hidden,
                alpha=args.l2,
                max_iter=args.max_iter,
                random_state=args.random_state,
                device=device,
                use_amp=use_amp,
                fit_pbar=fit_pbar,
            )
            improvement = mse_current - mse_new

            pvalue, null_mean, null_std = permutation_pvalue(
                train_df,
                val_df,
                target,
                selected,
                candidate,
                mse_current,
                improvement,
                n_perm=args.n_perm,
                hidden=hidden,
                alpha=args.l2,
                max_iter=args.max_iter,
                random_state=args.random_state,
                rng=rng,
                device=device,
                use_amp=use_amp,
                perm_workers=perm_workers,
                show_progress=args.show_progress,
                progress_desc=f"Perm[{target} <- {candidate}]",
                fit_pbar=fit_pbar,
            )

            significant = bool((improvement > args.improvement_tol) and (pvalue < args.alpha))

            test_rows.append(
                {
                    "target": target,
                    "candidate": candidate,
                    "step": step,
                    "mse_old": mse_current,
                    "mse_new": mse_new,
                    "improvement": improvement,
                    "improvement_tol": args.improvement_tol,
                    "pvalue": pvalue,
                    "alpha": args.alpha,
                    "null_improvement_mean": null_mean,
                    "null_improvement_std": null_std,
                    "selected_before": "|".join(selected),
                    "accepted": significant,
                }
            )

            if significant:
                selected.append(candidate)
                mse_current = mse_new
                accepted_edges.append(
                    {
                        "cause": candidate,
                        "effect": target,
                        "improvement": improvement,
                        "pvalue": pvalue,
                    }
                )

        if SAVE_EVERY_TARGET:
            tests_df, edges_df, _ = save_outputs(
                test_rows,
                accepted_edges,
                out_tests,
                out_edges,
                out_adj,
                out_fig,
            )
            print(
                f"[target done] {target} | "
                f"new_tests={len(test_rows) - tests_before} | "
                f"new_edges={len(accepted_edges) - edges_before} | "
                f"elapsed={time.perf_counter() - t0:.1f}s | "
                f"saved_rows tests={len(tests_df)} edges={len(edges_df)}"
            )

    tests_df, edges_df, adj_df = save_outputs(
        test_rows,
        accepted_edges,
        out_tests,
        out_edges,
        out_adj,
        out_fig,
    )

    if fit_pbar is not None:
        fit_pbar.close()

    print(f"saved tests: {out_tests} rows={len(tests_df)}")
    print(f"saved edges: {out_edges} rows={len(edges_df)}")
    print(f"saved adj:   {out_adj}")
    print(f"saved fig:   {out_fig}")


if __name__ == "__main__":
    main()
