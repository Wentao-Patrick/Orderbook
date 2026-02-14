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
    "rlop_ask_mean_lag1","rlop_bid_mean_lag1","vol_bid_mean_lag1","vol_ask_mean_lag1","spread_mean_lag1","imbalance_ob_mean_lag1","imb_of_lag1"
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

    # edges list from dir_adj
    edges = []
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
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

    title = f"PC Graph (no same-layer; alpha={args.alpha}, max_cond={args.max_cond})"
    draw_graph_from_dir_adj(cols, dir_adj, title, args.out_png)


if __name__ == "__main__":
    main()
