"""
Run custom PC + Cai-Li-Zhang CIT on lag-augmented features.

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
from tqdm import tqdm
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
CAUSAL_ZOVKO_DIR = SCRIPT_DIR.parent

# -----------------------------
# Utils
# -----------------------------

VARS_LIST = [
    "rlop_ask_mean","rlop_bid_mean","vol_bid_mean","vol_ask_mean","spread_mean","imbalance_ob_mean","imb_of",
    "rlop_ask_mean_lag1","rlop_bid_mean_lag1","vol_bid_mean_lag1","vol_ask_mean_lag1","spread_mean_lag1","imbalance_ob_mean_lag1","imb_of_lag1",
    "rlop_ask_mean_lag2","rlop_bid_mean_lag2","vol_bid_mean_lag2","vol_ask_mean_lag2","spread_mean_lag2","imbalance_ob_mean_lag2","imb_of_lag2",
    "rlop_ask_mean_lag5","rlop_bid_mean_lag5","vol_bid_mean_lag5","vol_ask_mean_lag5","spread_mean_lag5","imbalance_ob_mean_lag5","imb_of_lag5",
]

def rank_uniform(x):
    r = pd.Series(x).rank(method="average").to_numpy()
    return r / (len(r) + 1.0)


def parse_lag(name: str) -> int:
    """
    Parse lag from column name.
    - 'xxx_lag5' -> 5
    - 'xxx_lag1' -> 1
    - otherwise -> 0
    """
    m = re.search(r"_lag(\d+)\b", name)
    if m:
        return int(m.group(1))
    return 0


def base_name(name: str) -> str:
    return re.sub(r"_lag\d+\b", "", name)


# -----------------------------
# Rosenblatt transform estimator
# -----------------------------
def kernel_cdf_estimate(x, Z, h=None):
    """
    x: (n,)
    Z: (n,k) or (n,) or None
    Return: U_i = \hat F_{X|Z}(x_i | z_i) in [0,1]
    """
    n = len(x)
    if Z is None or (isinstance(Z, np.ndarray) and Z.size == 0):
        return rank_uniform(x)

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    # bandwidth by median heuristic
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

    indicators = (x[None, :] <= x[:, None]).astype(float)  # [i,j]=1(x_j <= x_i)
    Fhat = (W * indicators).sum(axis=1)
    return Fhat


# -----------------------------
# rho-hat statistic (paper style)
# -----------------------------
_C0 = 1.0 / (13.0 * np.exp(-3.0) - 40.0 * np.exp(-2.0) + 13.0 * np.exp(-1.0))


def rho_hat_stat(U, V, W):
    """
    \hat{rho} = c0 * n^{-2} * sum_{i,j} A(U_i,U_j)*A(V_i,V_j)*exp(-|W_i-W_j|)
    with A(u_i,u_j) defined as in your implementation.
    """
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

    stat = _C0 * np.sum(AU * AV * EW) / (n * n)
    return float(stat)


def precompute_null_rho(n: int, sims: int, seed: int = 42) -> np.ndarray:
    """
    A2 optimization: precompute null distribution once.
    Under H0: U,V,W ~ i.i.d Uniform(0,1), mutually independent.
    """
    rng = np.random.default_rng(seed)
    null_rhos = np.empty(sims, dtype=float)
    for b in range(sims):
        Us = rng.uniform(0.0, 1.0, size=n)
        Vs = rng.uniform(0.0, 1.0, size=n)
        Ws = rng.uniform(0.0, 1.0, size=n)
        null_rhos[b] = rho_hat_stat(Us, Vs, Ws)
    return null_rhos


def cit_pvalue(X, Y, Z=None, *, null_rhos: np.ndarray):
    """
    CIT p-value by comparing obs rho to precomputed null_rhos.
    NOTE: W uses only 1D "Z first column" uniformized rank (your convention).
    """
    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    n = len(X)

    U = kernel_cdf_estimate(X, Z)
    V = kernel_cdf_estimate(Y, Z)

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
    return pval, obs


# -----------------------------
# Background knowledge: time orientation (cross-layer only)
# -----------------------------
def enforce_time_background_crosslayer_only(dir_adj: np.ndarray, lags: list[int]) -> np.ndarray:
    """
    Cross-layer hard rule:
      if lag[i] > lag[j] => i (more past) -> j (more present)
      if lag[i] < lag[j] => j -> i
      if equal => do nothing (but we already removed same-lag edges)
    """
    p = dir_adj.shape[0]

    def has_edge(i, j):
        return (dir_adj[i, j] != 0) or (dir_adj[j, i] != 0)

    def force_orient(src, dst):
        if not has_edge(src, dst):
            return
        dir_adj[src, dst] = 2
        dir_adj[dst, src] = 0

    for i in range(p):
        for j in range(i + 1, p):
            if not has_edge(i, j):
                continue
            if lags[i] == lags[j]:
                continue
            if lags[i] > lags[j]:
                force_orient(i, j)  # past -> present
            else:
                force_orient(j, i)

    return dir_adj


# -----------------------------
# Meek rules R1-R4
# -----------------------------
def apply_meek_rules(dir_adj: np.ndarray):
    """
    Representation:
      0: none
      1: undirected (both directions == 1)
      2: directed i->j (i,j==2 and j,i==0)

    Rules:
      R1: a->b - c and a not adj c  => b->c
      R2: a - b and (a => b directed path) => a->b   (generalized, any length)
    #   R3: a - b and a-c->b, a-d->b with c not adj d => a->b
    #   R4: a - b and a - c, a->d, c->d, d->b and c not adj b => a->b
    """
    n = dir_adj.shape[0]

    def adjacent(x, y):
        return (dir_adj[x, y] != 0) or (dir_adj[y, x] != 0)

    def is_undirected(x, y):
        return dir_adj[x, y] == 1 and dir_adj[y, x] == 1

    def is_directed(x, y):
        return dir_adj[x, y] == 2 and dir_adj[y, x] == 0

    def orient(x, y):
        # orient x -> y if currently undirected
        if is_undirected(x, y):
            dir_adj[x, y] = 2
            dir_adj[y, x] = 0
            return True
        return False

    def compute_reachability():
        """
        reach[i,j] = True if there exists a directed path i => j
        Using transitive closure on current directed edges.
        """
        directed = (dir_adj == 2) & (dir_adj.T == 0)  # i->j
        reach = directed.astype(bool).copy()
        # Warshall-style closure (vectorized)
        for k in range(n):
            reach |= (reach[:, [k]] & reach[[k], :])
        return reach

    changed = False

    # -----------------
    # R1
    # -----------------
    for a in range(n):
        for b in range(n):
            if not is_directed(a, b):
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                if is_undirected(b, c) and (not adjacent(a, c)):
                    changed |= orient(b, c)

    # -----------------
    # R2 (GENERALIZED): a - b and (a => b) => a -> b
    # -----------------
    reach = compute_reachability()
    for a in range(n):
        for b in range(a + 1, n):
            if not is_undirected(a, b):
                continue
            if reach[a, b]:
                changed |= orient(a, b)
            elif reach[b, a]:
                changed |= orient(b, a)

    # # -----------------
    # # R3
    # # -----------------
    # for a in range(n):
    #     for b in range(n):
    #         if a == b or not is_undirected(a, b):
    #             continue
    #         cand = []
    #         for c in range(n):
    #             if c == a or c == b:
    #                 continue
    #             if is_undirected(a, c) and is_directed(c, b):
    #                 cand.append(c)
    #         if len(cand) < 2:
    #             continue
    #         found = False
    #         for i in range(len(cand)):
    #             for j in range(i + 1, len(cand)):
    #                 c, d = cand[i], cand[j]
    #                 if not adjacent(c, d):
    #                     found = True
    #                     break
    #             if found:
    #                 break
    #         if found:
    #             changed |= orient(a, b)

    # # -----------------
    # # R4
    # # a - b and a - c and a->d and c->d and d->b and c not adj b => a->b
    # # -----------------
    # for a in range(n):
    #     for b in range(n):
    #         if a == b or not is_undirected(a, b):
    #             continue
    #         for c in range(n):
    #             if c in (a, b):
    #                 continue
    #             if not is_undirected(a, c):
    #                 continue
    #             if adjacent(c, b):
    #                 continue
    #             for d in range(n):
    #                 if d in (a, b, c):
    #                     continue
    #                 if is_directed(a, d) and is_directed(c, d) and is_directed(d, b):
    #                     changed |= orient(a, b)
    #                     break

    return dir_adj, changed
  

def meek_closure(dir_adj: np.ndarray, max_iter: int = 1000):
    for _ in range(max_iter):
        dir_adj, changed = apply_meek_rules(dir_adj)
        if not changed:
            break
    return dir_adj


# -----------------------------
# PC algorithm (NO same-layer edges) with pcalg-like stopping
# -----------------------------
def pc_algorithm_no_same_layer_edges(
    data: pd.DataFrame,
    alpha=0.05,
    max_cond_set=5,
    sims=2000,
    seed=42,
):
    """
    Changes vs your previous version:
      1) Enumeration: for each remaining edge (i,j), try conditioning sets from BOTH sides:
           S ⊆ Adj(i)\{j}  and  S ⊆ Adj(j)\{i}
      2) Early stop: DO NOT stop when a level l removes no edges.
         Stop when max_degree <= l  (no pair has enough neighbors to form size-l conditioning set),
         or when l > max_cond_set.
    """
    cols = list(data.columns)
    p = len(cols)
    lags = [parse_lag(c) for c in cols]

    # adjacency init: only allow edges across different lags
    adj = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            if lags[i] != lags[j]:
                adj[i, j] = 1
                adj[j, i] = 1

    sep_sets = {(i, j): set() for i in range(p) for j in range(p) if i < j}

    n = data.shape[0]
    print(f"Precomputing null distribution once (n={n}, sims={sims}) ...")
    null_rhos = precompute_null_rho(n=n, sims=sims, seed=seed)

    from itertools import combinations

    # skeleton
    for l in range(0, max_cond_set + 1):
        # pcalg-like stop: if max degree <= l, no adj(i)\{j} can reach size l
        degrees = adj.sum(axis=1)
        if degrees.max() <= l:
            print(f"[STOP] max_degree={degrees.max()} <= l={l}. No more tests possible.")
            break

        pairs = [(i, j) for i in range(p) for j in range(i + 1, p) if adj[i, j] == 1]
        if len(pairs) == 0:
            print("[STOP] No edges remain.")
            break

        print(f"--- l={l} | edges={len(pairs)} | max_degree={degrees.max()} ---")

        for i, j in tqdm(pairs, desc=f"l={l}"):
            if adj[i, j] == 0:
                continue

            removed = False
            # try conditioning sets from both endpoints (i-side then j-side)
            for side, other in ((i, j), (j, i)):
                neighbors = [k for k in range(p) if adj[side, k] == 1 and k != other]
                if len(neighbors) < l:
                    continue

                for cond in combinations(neighbors, l):
                    Z = data.iloc[:, list(cond)].to_numpy() if len(cond) > 0 else np.empty((n, 0))
                    pval, _ = cit_pvalue(
                        data.iloc[:, i].to_numpy(),
                        data.iloc[:, j].to_numpy(),
                        Z,
                        null_rhos=null_rhos,
                    )
                    if pval > alpha:
                        adj[i, j] = 0
                        adj[j, i] = 0
                        sep_sets[(min(i, j), max(i, j))] = set(cond)
                        removed = True
                        break

                if removed:
                    break

    # initialize directed adjacency
    dir_adj = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if adj[i, j] == 1:
                dir_adj[i, j] = 1

    # orient colliders
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j] == 1:
                continue  # adjacent => not (i,j) in collider check
            for k in range(p):
                if k == i or k == j:
                    continue
                if adj[i, k] == 1 and adj[j, k] == 1:
                    if k not in sep_sets.get((i, j), set()):
                        dir_adj[i, k] = 2
                        dir_adj[k, i] = 0
                        dir_adj[j, k] = 2
                        dir_adj[k, j] = 0

    # background knowledge: cross-layer must be past -> present
    dir_adj = enforce_time_background_crosslayer_only(dir_adj, lags)

    # Meek closure (R1-R4)
    dir_adj = meek_closure(dir_adj)

    return cols, lags, adj, dir_adj


# -----------------------------
# Drawing (same style)
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
    lags2 = [parse_lag(x) for x in nodes2]

    # ---- 2) layered layout: causes (larger lag, more past) on top ----
    # deterministic node ordering in each layer for readability
    def base_name(name: str) -> str:
        return re.sub(r"_lag\d+\b", "", name)

    unique_lags = sorted(set(lags2), reverse=True)
    layers = {}
    for L in unique_lags:
        idxs = [i for i in range(m) if lags2[i] == L]
        idxs = sorted(idxs, key=lambda i: (base_name(nodes2[i]), nodes2[i]))
        layers[L] = idxs

    x_gap = 2.0
    y_gap = 1.8
    pos_idx = {}
    for layer_rank, L in enumerate(unique_lags):
        idxs = layers[L]
        k = len(idxs)
        if k == 1:
            xs = np.array([0.0])
        else:
            xs = np.linspace(-(k - 1) / 2.0, (k - 1) / 2.0, k) * x_gap
        y = (len(unique_lags) - 1 - layer_rank) * y_gap
        for p_i, idx in enumerate(idxs):
            pos_idx[idx] = (float(xs[p_i]), float(y))

    # keep labels short and explicit
    labels = {}
    for i, node in enumerate(nodes2):
        b = base_name(node)
        L = lags2[i]
        labels[i] = f"{b}\nlag{L}"

    fig_w = max(11.0, 2.2 * max(len(v) for v in layers.values()))
    fig_h = max(7.0, 2.4 * len(unique_lags))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    node_size = 4200
    node_radius_pts = float(np.sqrt(node_size / np.pi))
    arrow_shrink = node_radius_pts + 5.0

    # subtle layer guides
    for layer_rank, L in enumerate(unique_lags):
        y = (len(unique_lags) - 1 - layer_rank) * y_gap
        ax.axhline(y=y, color="#E6E6E6", lw=0.9, zorder=0)
        ax.text(
            x=-0.35 - 0.5 * x_gap * max(len(v) for v in layers.values()),
            y=y + 0.16,
            s=f"lag{L}",
            fontsize=9,
            color="#555555",
            ha="left",
            va="bottom",
            zorder=0,
        )

    # ---- 3) draw edges once per unordered pair ----
    for i in range(m):
        for j in range(i + 1, m):
            x1, y1 = pos_idx[i]
            x2, y2 = pos_idx[j]
            same_layer = (lags2[i] == lags2[j])
            # deterministic mild curvature for readability
            sign = 1.0 if ((i + j) % 2 == 0) else -1.0
            base_rad = 0.16 if same_layer else 0.08
            rad = sign * base_rad

            if dir_adj2[i, j] == 2 and dir_adj2[j, i] == 0:
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        mutation_scale=20,
                        lw=2.0,
                        color="#D64550",
                        shrinkA=arrow_shrink,
                        shrinkB=arrow_shrink,
                        connectionstyle=f"arc3,rad={rad}",
                    ),
                    zorder=2,
                )
            elif dir_adj2[j, i] == 2 and dir_adj2[i, j] == 0:
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x2, y2),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        mutation_scale=20,
                        lw=2.0,
                        color="#D64550",
                        shrinkA=arrow_shrink,
                        shrinkB=arrow_shrink,
                        connectionstyle=f"arc3,rad={-rad}",
                    ),
                    zorder=2,
                )
            elif dir_adj2[i, j] == 1 and dir_adj2[j, i] == 1:
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-",
                        lw=1.25,
                        color="#7F7F7F",
                        alpha=0.95,
                        shrinkA=arrow_shrink,
                        shrinkB=arrow_shrink,
                        connectionstyle=f"arc3,rad={0.6 * rad}",
                    ),
                    zorder=1,
                )

    # ---- 4) draw nodes ----
    cmap = plt.cm.Blues
    lag_rank = {L: r for r, L in enumerate(unique_lags)}
    denom = max(1, len(unique_lags) - 1)
    for i in range(m):
        x, y = pos_idx[i]
        L = lags2[i]
        c = cmap(0.35 + 0.5 * (lag_rank[L] / denom))
        ax.scatter([x], [y], s=node_size, color=c, edgecolor="#1F1F1F", linewidth=1.1, zorder=3)
        ax.text(x, y, labels[i], ha="center", va="center", fontsize=9, weight="bold", zorder=4)

    ax.set_title(title + "\n(cause on top, effect on bottom)", fontsize=14, pad=14)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved graph image to: {out_path} (nodes drawn: {m}/{n}, isolated removed: {n-m})")


def collapse_to_variable_graph(cols, dir_adj, *, drop_self: bool = False):
    """
    Collapse lag-level directed edges into variable-level directed edges.
    Rule: if ANY directed edge between lagged nodes indicates A -> B, then keep variable edge A -> B.
    """
    cols = list(cols)
    dir_adj = np.asarray(dir_adj)
    n = len(cols)

    base_vars = []
    seen = set()
    for c in cols:
        b = base_name(c)
        if b not in seen:
            seen.add(b)
            base_vars.append(b)

    var_index = {v: i for i, v in enumerate(base_vars)}
    var_adj = np.zeros((len(base_vars), len(base_vars)), dtype=int)
    edges_set = set()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not (dir_adj[i, j] == 2 and dir_adj[j, i] == 0):
                continue

            src = base_name(cols[i])
            dst = base_name(cols[j])
            if drop_self and src == dst:
                continue
            edges_set.add((src, dst))

    edges = []
    for src, dst in sorted(edges_set):
        var_adj[var_index[src], var_index[dst]] = 1
        edges.append({"from": src, "to": dst, "type": "directed"})

    return base_vars, var_adj, edges


def draw_variable_causal_graph_lr(
    edges_df: pd.DataFrame,
    title: str,
    out_path: str,
    all_nodes=None,
):
    """
    Draw variable-level causal graph with each variable shown once.
    Layout:
      left   : source-like nodes (out>0, in=0)
      middle : mixed / isolated nodes
      right  : sink-like nodes (in>0, out=0)
    """
    if all_nodes is None:
        if edges_df.empty:
            all_nodes = []
        else:
            all_nodes = sorted(set(edges_df["from"]).union(set(edges_df["to"])))
    else:
        all_nodes = sorted(list(all_nodes))

    if len(all_nodes) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No variables to draw", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved graph image to: {out_path} (no variables)")
        return

    out_deg = {n: 0 for n in all_nodes}
    in_deg = {n: 0 for n in all_nodes}
    for _, row in edges_df.iterrows():
        src = str(row["from"])
        dst = str(row["to"])
        if src in out_deg:
            out_deg[src] += 1
        if dst in in_deg:
            in_deg[dst] += 1

    left_nodes = sorted([n for n in all_nodes if out_deg[n] > 0 and in_deg[n] == 0])
    right_nodes = sorted([n for n in all_nodes if in_deg[n] > 0 and out_deg[n] == 0])
    middle_nodes = sorted([n for n in all_nodes if n not in left_nodes and n not in right_nodes])

    def spread_y(k):
        if k <= 1:
            return np.array([0.5]) if k == 1 else np.array([])
        return np.linspace(1.0, 0.0, k)

    pos = {}
    for i, node in enumerate(left_nodes):
        pos[node] = (0.1, float(spread_y(len(left_nodes))[i]))
    for i, node in enumerate(middle_nodes):
        pos[node] = (0.5, float(spread_y(len(middle_nodes))[i]))
    for i, node in enumerate(right_nodes):
        pos[node] = (0.9, float(spread_y(len(right_nodes))[i]))

    fig_h = max(6.0, 0.34 * len(all_nodes))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    for node in left_nodes:
        x, y = pos[node]
        ax.scatter(x, y, s=1500, c="#E3F2FD", edgecolors="#1E88E5", linewidths=1.2, zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=9)
    for node in middle_nodes:
        x, y = pos[node]
        ax.scatter(x, y, s=1500, c="#E8F5E9", edgecolors="#2E7D32", linewidths=1.2, zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=9)
    for node in right_nodes:
        x, y = pos[node]
        ax.scatter(x, y, s=1500, c="#FFF3E0", edgecolors="#FB8C00", linewidths=1.2, zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=9)

    for _, row in edges_df.iterrows():
        src = str(row["from"])
        dst = str(row["to"])
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        if src == dst:
            ax.annotate(
                "",
                xy=(x0 - 0.03, y0 + 0.02),
                xytext=(x0 + 0.03, y0 + 0.02),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.7,
                    color="#616161",
                    alpha=0.9,
                    connectionstyle="arc3,rad=1.4",
                ),
                zorder=2,
            )
            continue

        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.7,
                color="#616161",
                alpha=0.85,
                connectionstyle="arc3,rad=0.06",
            ),
            zorder=2,
        )

    ax.text(0.1, 1.06, "Causes", ha="center", va="bottom", fontsize=11, color="#0D47A1")
    ax.text(0.5, 1.06, "Mixed/Neutral", ha="center", va="bottom", fontsize=11, color="#1B5E20")
    ax.text(0.9, 1.06, "Effects", ha="center", va="bottom", fontsize=11, color="#E65100")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved graph image to: {out_path} (variables={len(all_nodes)}, edges={len(edges_df)})")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(CAUSAL_ZOVKO_DIR / "data" / "causal_dataset_1min.csv"))
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--max_cond", type=int, default=8)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
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

    cols, lags, adj, dir_adj = pc_algorithm_no_same_layer_edges(
        df, alpha=args.alpha, max_cond_set=args.max_cond, sims=args.sims, seed=args.seed
    )

    # Collapse lag-level graph to variable-level graph.
    base_vars, var_adj, edges = collapse_to_variable_graph(cols, dir_adj, drop_self=False)

    Path(args.out_edges).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edges).to_csv(args.out_edges, index=False)

    Path(args.out_adj).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(var_adj, index=base_vars, columns=base_vars).to_csv(args.out_adj)

    print("saved:", args.out_edges, args.out_adj)

    title = f"PC Variable Causal Graph (cause left -> effect right; alpha={args.alpha}, max_cond={args.max_cond})"
    draw_variable_causal_graph_lr(pd.DataFrame(edges), title, args.out_png, all_nodes=base_vars)


if __name__ == "__main__":
    main()
