"""
Run pgmpy PC with custom CIT test and expert temporal constraints.

Default input/output locations (relative to EA_recherche root):
- input: causal_zovko/data/causal_dataset_1min.csv
- outputs:
    - causal_zovko/results/pc_edges.csv
    - causal_zovko/results/pc_adjmatrix.csv
    - causal_zovko/figures/pc_graph.png
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pgmpy.estimators import PC, ExpertKnowledge


SCRIPT_DIR = Path(__file__).resolve().parent
CAUSAL_ZOVKO_DIR = SCRIPT_DIR.parent


class PCSafe(PC):
    """
    PC variant that fills missing separating sets to avoid KeyError
    when orienting colliders under early stopping / max_cond_vars limits.
    """

    def orient_colliders(self, skel, separating_sets, temporal_ordering=None):
        nodes = list(skel.nodes())
        sep = dict(separating_sets)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                key = frozenset((nodes[i], nodes[j]))
                if key not in sep:
                    sep[key] = set()
        return super().orient_colliders(skel, sep, temporal_ordering)

# -----------------------------
# Utils
# -----------------------------

VARS_LIST = [
    "rlop_ask_mean","rlop_bid_mean","vol_bid_mean","vol_ask_mean","spread_mean","imbalance_ob_mean","imb_of",
    "rlop_ask_mean_lag1","rlop_bid_mean_lag1","vol_bid_mean_lag1","vol_ask_mean_lag1","spread_mean_lag1","imbalance_ob_mean_lag1","imb_of_lag1"
]


def rank_uniform(x):
    r = pd.Series(x).rank(method="average").to_numpy()
    return r / (len(r) + 1.0)


def parse_lag(name: str) -> int:
    m = re.search(r"_lag(\d+)\b", name)
    if m:
        return int(m.group(1))
    return 0


# -----------------------------
# Rosenblatt transform estimator
# -----------------------------
def kernel_cdf_estimate(x, Z, h=None):
    n = len(x)
    if Z is None or (isinstance(Z, np.ndarray) and Z.size == 0):
        return rank_uniform(x)

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    if h is None:
        rng0 = np.random.default_rng(0)
        idx = rng0.choice(n, size=min(n, 200), replace=False)
        Zs = Z[idx]
        dists = np.sqrt(((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(axis=2))
        if np.any(dists > 0):
            med = np.median(dists[dists > 0])
            h = med if med > 0 else 1.0
        else:
            h = 1.0

    diffs = Z[:, None, :] - Z[None, :, :]
    d2 = np.sum(diffs**2, axis=2)
    W = np.exp(-d2 / (2 * h * h))
    W_sum = W.sum(axis=1, keepdims=True)
    W = W / W_sum

    indicators = (x[None, :] <= x[:, None]).astype(float)
    return (W * indicators).sum(axis=1)


# -----------------------------
# rho-hat statistic + A2 null
# -----------------------------
_C0 = 1.0 / (13.0 * np.exp(-3.0) - 40.0 * np.exp(-2.0) + 13.0 * np.exp(-1.0))


def rho_hat_stat(U, V, W):
    U = np.asarray(U).reshape(-1)
    V = np.asarray(V).reshape(-1)
    W = np.asarray(W).reshape(-1)
    n = U.shape[0]

    du = np.abs(U[:, None] - U[None, :])
    dv = np.abs(V[:, None] - V[None, :])
    dw = np.abs(W[:, None] - W[None, :])

    EU = np.exp(-du)
    EV = np.exp(-dv)
    EW = np.exp(-dw)

    AU = (
        EU
        + np.exp(-U)[:, None] + np.exp(U - 1.0)[:, None]
        + np.exp(-U)[None, :] + np.exp(U - 1.0)[None, :]
        + 2.0 * np.exp(-1.0) - 4.0
    )
    AV = (
        EV
        + np.exp(-V)[:, None] + np.exp(V - 1.0)[:, None]
        + np.exp(-V)[None, :] + np.exp(V - 1.0)[None, :]
        + 2.0 * np.exp(-1.0) - 4.0
    )

    return float(_C0 * np.sum(AU * AV * EW) / (n * n))


def precompute_null_rho(n: int, sims: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    null_rhos = np.empty(sims, dtype=float)
    for b in range(sims):
        Us = rng.uniform(0.0, 1.0, size=n)
        Vs = rng.uniform(0.0, 1.0, size=n)
        Ws = rng.uniform(0.0, 1.0, size=n)
        null_rhos[b] = rho_hat_stat(Us, Vs, Ws)
    return null_rhos


def cit_stat_pvalue(x, y, Z, null_rhos):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = len(x)

    U = kernel_cdf_estimate(x, Z)
    V = kernel_cdf_estimate(y, Z)

    # W: use first column of Z (your convention)
    if Z is None or (isinstance(Z, np.ndarray) and Z.size == 0):
        W = np.zeros(n, dtype=float)
    else:
        Z = np.asarray(Z)
        if Z.ndim == 2:
            if Z.shape[1] == 0:
                W = np.zeros(n, dtype=float)
            else:
                W = rank_uniform(Z[:, 0].reshape(-1))
        else:
            W = rank_uniform(Z.reshape(-1))

    obs = rho_hat_stat(U, V, W)
    cnt = int(np.sum(null_rhos >= obs))
    pval = (cnt + 1) / (len(null_rhos) + 1)
    return obs, pval


# -----------------------------
# Drawing (same style as yours)
# -----------------------------
def draw_graph_from_dir_adj(nodes, dir_adj, title, out_path):
    """
    Draw CPDAG-like graph from dir_adj, but DO NOT draw isolated nodes (degree 0).

    dir_adj encoding:
      0: no edge
      1: undirected edge (both directions == 1)
      2: directed edge i->j (i,j==2 and j,i==0)
    """
    n = len(nodes)
    if n == 0:
        print("No nodes to draw.")
        return

    dir_adj = np.asarray(dir_adj)
    if dir_adj.shape != (n, n):
        raise ValueError(f"dir_adj shape {dir_adj.shape} does not match nodes length {n}")

    # ---- 1) filter isolated nodes (no incident edges in either direction) ----
    # node i is kept if it has any neighbor j such that dir_adj[i,j] != 0 or dir_adj[j,i] != 0
    has_edge = (dir_adj != 0) | (dir_adj.T != 0)
    deg = has_edge.sum(axis=1)  # includes self? diagonal is false anyway
    keep_idx = np.where(deg > 0)[0].tolist()

    if len(keep_idx) == 0:
        print("All nodes are isolated (no edges). Nothing to draw.")
        return

    nodes2 = [nodes[i] for i in keep_idx]
    dir_adj2 = dir_adj[np.ix_(keep_idx, keep_idx)]
    m = len(nodes2)

    # ---- 2) circular layout on remaining nodes ----
    theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
    pos = {nodes2[i]: (np.cos(theta[i]), np.sin(theta[i])) for i in range(m)}

    plt.figure(figsize=(10, 10))
    node_size = 3500
    node_radius_pts = float(np.sqrt(node_size / np.pi))
    arrow_shrink = node_radius_pts + 7.0

    # ---- 3) draw edges once per unordered pair ----
    for i in range(m):
        for j in range(i + 1, m):
            a, b = nodes2[i], nodes2[j]
            x1, y1 = pos[a]
            x2, y2 = pos[b]

            if dir_adj2[i, j] == 2 and dir_adj2[j, i] == 0:
                # i -> j
                plt.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        mutation_scale=24,
                        lw=2.0,
                        color="tab:red",
                        shrinkA=arrow_shrink,
                        shrinkB=arrow_shrink,
                    ),
                    zorder=5,
                )
            elif dir_adj2[j, i] == 2 and dir_adj2[i, j] == 0:
                # j -> i
                plt.annotate(
                    "", xy=(x1, y1), xytext=(x2, y2),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        mutation_scale=24,
                        lw=2.0,
                        color="tab:red",
                        shrinkA=arrow_shrink,
                        shrinkB=arrow_shrink,
                    ),
                    zorder=5,
                )
            elif dir_adj2[i, j] == 1 and dir_adj2[j, i] == 1:
                # undirected
                plt.plot([x1, x2], [y1, y2], color="gray", lw=1.0, zorder=1)

    # ---- 4) draw nodes ----
    for node, (x, y) in pos.items():
        plt.scatter([x], [y], s=node_size, color="skyblue", edgecolor="black", zorder=3)
        plt.text(x, y, node, ha="center", va="center", fontsize=9, weight="bold", zorder=4)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()
    print(f"Saved graph image to: {out_path} (nodes drawn: {m}/{n}, isolated removed: {n-m})")


def pdag_to_dir_adj(nodes, pdag_edges):
    """
    pgmpy PDAG stores undirected edge as BOTH (u,v) and (v,u).
    directed edge as only (u,v).
    Convert to:
      0 none, 1 undirected (both==1), 2 directed i->j (i,j==2 and j,i==0)
    """
    idx = {n: i for i, n in enumerate(nodes)}
    p = len(nodes)
    dir_adj = np.zeros((p, p), dtype=int)

    edge_set = set(pdag_edges)
    for u, v in list(edge_set):
        if (v, u) in edge_set:
            iu, iv = idx[u], idx[v]
            dir_adj[iu, iv] = 1
            dir_adj[iv, iu] = 1

    for u, v in list(edge_set):
        if (v, u) not in edge_set:
            iu, iv = idx[u], idx[v]
            dir_adj[iu, iv] = 2
            dir_adj[iv, iu] = 0

    return dir_adj


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(CAUSAL_ZOVKO_DIR / "data" / "causal_dataset_1min.csv"))
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--max_cond", type=int, default=11)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", default="stable", choices=["original", "stable", "parallel"])
    ap.add_argument("--out_edges", default=str(CAUSAL_ZOVKO_DIR / "results" / "pc_edges.csv"))
    ap.add_argument("--out_adj", default=str(CAUSAL_ZOVKO_DIR / "results" / "pc_adjmatrix.csv"))
    ap.add_argument("--out_png", default=str(CAUSAL_ZOVKO_DIR / "figures" / "pc_graph.png"))
    ap.add_argument("--vars", nargs="+", default=VARS_LIST)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, index_col=0)

    if args.vars:
        if len(args.vars) == 1 and "," in args.vars[0]:
            vars_list = [v.strip() for v in args.vars[0].split(",")]
        else:
            vars_list = args.vars  
        missing = [v for v in vars_list if v not in df.columns]
        if missing:
            print(f"Error: missing columns: {missing}")
            return
        df = df[vars_list]

    cols = list(df.columns)
    lags = {c: parse_lag(c) for c in cols}

    # ---------- ExpertKnowledge ----------
    # forbid same-lag edges (no same-layer causality / association)
    forbidden = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if lags[cols[i]] == lags[cols[j]]:
                forbidden.add((cols[i], cols[j]))
                forbidden.add((cols[j], cols[i]))

    # temporal tiers: past -> present (larger lag is more past)
    unique_lags = sorted(set(lags.values()), reverse=True)  # e.g. 5,1,0
    temporal_order = []
    for L in unique_lags:
        tier = [c for c in cols if lags[c] == L]
        if tier:
            temporal_order.append(tier)

    expert = ExpertKnowledge(
        forbidden_edges=forbidden,
        temporal_order=temporal_order,
    )

    # ---------- CIT null once ----------
    n = df.shape[0]
    print(f"Precomputing null distribution once (n={n}, sims={args.sims}) ...")
    null_rhos = precompute_null_rho(n=n, sims=args.sims, seed=args.seed)

    # ---------- CIT ci_test for pgmpy ----------
    def cit_ci_test(X, Y, Z, data, boolean=True, significance_level=0.01, **kwargs):
        x = data[X].to_numpy()
        y = data[Y].to_numpy()
        if Z is None or len(Z) == 0:
            Zmat = np.empty((len(x), 0))
        else:
            Zmat = data[list(Z)].to_numpy()

        stat, pval = cit_stat_pvalue(x, y, Zmat, null_rhos)

        if boolean:
            # pgmpy expects True => independent
            return pval >= significance_level
        else:
            return stat, pval

    # ---------- Run PC (library) ----------
    est = PCSafe(df)
    # enforce_expert_knowledge=True => skeleton stage prunes forbidden edges early
    pdag = est.estimate(
        variant=args.variant,
        ci_test=cit_ci_test,
        significance_level=args.alpha,
        max_cond_vars=args.max_cond,
        expert_knowledge=expert,
        enforce_expert_knowledge=True,
        show_progress=True,
    )

    # ---------- Convert to your matrix format ----------
    dir_adj = pdag_to_dir_adj(cols, list(pdag.edges()))

    # adjacency (undirected view) for saving
    adj = np.zeros((len(cols), len(cols)), dtype=int)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if dir_adj[i, j] != 0 or dir_adj[j, i] != 0:
                adj[i, j] = 1
                adj[j, i] = 1
    np.fill_diagonal(adj, 0)

    # edges CSV
    edges = []
    p = len(cols)
    for i in range(p):
        for j in range(i + 1, p):
            if dir_adj[i, j] == 2 and dir_adj[j, i] == 0:
                edges.append({"from": cols[i], "to": cols[j], "type": "directed"})
            elif dir_adj[j, i] == 2 and dir_adj[i, j] == 0:
                edges.append({"from": cols[j], "to": cols[i], "type": "directed"})
            elif dir_adj[i, j] == 1 and dir_adj[j, i] == 1:
                edges.append({"from": cols[i], "to": cols[j], "type": "undirected"})

    Path(args.out_edges).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edges).to_csv(args.out_edges, index=False)

    Path(args.out_adj).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(adj, index=cols, columns=cols).to_csv(args.out_adj)

    print("saved:", args.out_edges, args.out_adj)

    title = f"PC (pgmpy) + CIT (no same-layer; alpha={args.alpha}, max_cond={args.max_cond})"
    draw_graph_from_dir_adj(cols, dir_adj, title, args.out_png)


if __name__ == "__main__":
    main()
