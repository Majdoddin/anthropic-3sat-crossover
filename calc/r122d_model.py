# One-level grid constructions for the (K, lambda)-clique sum at K = g^2 (g = 3, 4), stated for ANY upper-bound
# function w with omega(1,k,1) <= w(k) on [0,1] (omega <= w(1)).  Units: exponents of N = 2^lambda; e = exponent / C(K,2).
#
# Matrix costs.  Omega_w(a,b,c) = (M - m) + m w(s/m) with (M, m, s) the sorted dims: omega is symmetric in its three
# arguments, and <N^M, N^s, N^m> is N^{M-m} copies of <N^m, N^s, N^m>.  Non-decreasing in w (pointwise).
#
# Constructions (Section 2.4 and Appendix A of the paper):
#   tree : best subset-merge contraction tree with rectangular MM per step; bw(K) = width lower bound for all trees.
#   U    : ungrouped eight-square expansion per 3-bit block (as in Theorem 2.7, with the bypass), entropy sum_r H(a_r)/3 at rate 1.
#   G    : grouped (Lemma 2 / eq. (20)) with naming log C(t,i) bits, rect cost rho(a) = min(2, 1 + a + H(a)/3).
#   E    : grouped counted by ranks (App. H eigenbasis; grouping counted by ranks, Section 2.3), rho_E(a) = max_{z <= min(a, 7/8)} H(z)/3 + 1 + (L56-1) z.
#   B    : Appendix B pieces on m-bit blocks, m -> infinity (continuum model of Section 2.3): (rho, kappa) = (sqrt(2+u), sqrt(2-u)).
# Per term: rows or columns merged (global choice); top instance = g merged nodes (triangle at g = 3, K4 at g = 4);
# row tables built by a sequential merge inside the row with the bypass (y inner, theta on one side) or fusing.
import itertools
import numpy as np
from math import comb

L56 = np.log2(56) / 3


def H(a):
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = -(a * np.log2(np.where(a > 0, a, 1)) + (1 - a) * np.log2(np.where(a < 1, 1 - a, 1)))
    return v


# ---------------- w functions ----------------
class W:
    """An upper-bound function for omega(1,k,1), k in [0,1]; name for printing."""
    def __init__(self, f, name):
        self.f, self.name = f, name

    def __call__(self, k):
        return self.f(np.clip(np.asarray(k, dtype=float), 0.0, 1.0))


def w_const2():
    return W(lambda k: np.full_like(k, 2.0), 'omega=2 (w = 2)')


def w_curve(expr, alpha, omega, name):
    # expr: python expression in k valid for k >= alpha; w = 2 below alpha; w(1) capped by omega
    code = compile(expr, '<w>', 'eval')
    def f(k):
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.where(k <= alpha, 2.0, eval(code, {'np': np}, {'k': np.maximum(k, 1e-9)}))
        return np.minimum(v, np.maximum(2.0, omega))  # omega(1,k,1) <= omega for k <= 1 (monotone in k)
    return W(f, name)


def w_table(pts, omega, name):
    ks, ws = map(np.array, zip(*pts))
    def f(k):
        return np.where(k <= ks[0], 2.0, np.interp(np.minimum(k, 1.0), ks, ws))
    return W(f, name)


def w_chord(alpha, omega, name=None):
    return W(lambda k: np.where(k <= alpha, 2.0, 2 + (k - alpha) / (1 - alpha) * (omega - 2)),
             name or f'chord alpha={alpha}, omega={omega}')


def Om(a, b, c, w):
    d = np.sort(np.stack(np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(c, float))), axis=0)
    s, m, M = d[0], d[1], d[2]
    safe = np.where(m > 0, m, 1.0)
    v = np.where(m > 0, (M - m) + m * w(s / safe), M)
    gen = getattr(w, 'general', None)
    if gen is not None:
        v = np.minimum(v, gen(M, m, s))
    return v


def avdbz_general(M, m, s):
    """AvdBZ arXiv 2610.30054 Thm 1.2 (not checked by us, not in values.yaml): for
    a >= b >= c >= 0, omega(a,b,c) <= a + b if c(a+b) <= ab, else (ab+bc+ca)^2/(4abc)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        alt = (M * m + m * s + s * M) ** 2 / (4 * M * m * s)
    return np.where(s * (M + m) <= M * m, M + m, np.where(s > 0, alt, M + m))


def with_general(w, gen, name):
    w2 = W(w.f, name)
    w2.general = gen
    return w2


# ---------------- trees ----------------
def bw(K):
    # min over ternary trees of the max over tree edges of k(K-k) (k = stars on the smaller side): every ternary tree
    # has an edge with ceil(K/3) <= k <= K/2 ... and a tree with all edges k <= ceil(K/3) exists -> bw = k0 (K - k0).
    k0 = -(-K // 3)
    return k0 * (K - k0)


def tree(K, w):
    # f[n] = best cost to build the merged table of an n-set of nodes (complement K - n); all nodes alike.
    f = {1: float(K - 1)}
    arg = {1: None}
    for n in range(2, K):
        c = K - n
        best = (np.inf, None)
        for x in range(1, n // 2 + 1):
            y = n - x
            step = float(Om(x * c, x * y, y * c, w))
            v = max(f[x], f[y], step, (x + y) * c)
            if v < best[0]:
                best = (v, x)
        f[n], arg[n] = best
    best = (np.inf, None)
    for x in range(1, K // 2 + 1):
        v = max(f[x], f[K - x], x * (K - x))
        if v < best[0]:
            best = (v, x)
    return best[0], best[1], f, arg


# ---------------- grid: rectangles, edges, top ----------------
def grid(g):
    pairs = list(itertools.combinations(range(g), 2))
    rects = [(P, C) for P in range(len(pairs)) for C in range(len(pairs))]
    return pairs, rects


def edges_from(rho, kap, g):
    # rho, kap: arrays shape (nP, nC) of per-rectangle edge contributions (rows merged / columns merged)
    E = g + rho.sum(axis=1)        # row pair P: g column side edges + its C(g,2) rectangles
    F = g + kap.sum(axis=0)
    return E, F


def top_cost(E, g, w):
    pairs = list(itertools.combinations(range(g), 2))
    e = {p: E[i] for i, p in enumerate(pairs)}
    ed = lambda i, j: e[(min(i, j), max(i, j))]
    if g == 2:
        return E[0]
    if g == 3:
        return float(Om(E[0], E[1], E[2], w))
    if g == 4:
        return float(_k4_top(np.asarray(E, float)[None, :], w)[0])
    raise ValueError(g)


_PAIRS4 = list(itertools.combinations(range(4), 2))
_PI4 = {p: i for i, p in enumerate(_PAIRS4)}


def _e4(i, j):
    return _PI4[(min(i, j), max(i, j))]


_K4_TREES = []   # (merge1 (a,b,c) edge-index lists, merge2, final dot edge list)
for (i, j, k, l) in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
    _K4_TREES.append(((([_e4(i, k), _e4(i, l)], [_e4(i, j)], [_e4(j, k), _e4(j, l)]),
                       ([_e4(k, i), _e4(k, j)], [_e4(k, l)], [_e4(l, i), _e4(l, j)])),
                      [_e4(i, k), _e4(i, l), _e4(j, k), _e4(j, l)]))
for (i, j, k, l) in itertools.permutations(range(4)):
    if i > j:
        continue
    _K4_TREES.append(((([_e4(i, k), _e4(i, l)], [_e4(i, j)], [_e4(j, k), _e4(j, l)]),
                       ([_e4(i, l), _e4(j, l)], [_e4(i, k), _e4(j, k)], [_e4(k, l)])),
                      [_e4(i, l), _e4(j, l), _e4(k, l)]))


_MA, _MB, _MC, _DOT = [], [], [], []
for (m1, m2), dot in _K4_TREES:
    for (A, Bi, C) in (m1, m2):
        _MA.append(A); _MB.append(Bi); _MC.append(C)
    _DOT.append(dot)


def _sel(lists, n):
    M_ = np.zeros((len(lists), n))
    for r, L in enumerate(lists):
        M_[r, L] = 1
    return M_


_SA, _SB, _SC, _SD = _sel(_MA, 6), _sel(_MB, 6), _sel(_MC, 6), _sel(_DOT, 6)


def _k4_top(E, w):
    """E: (n, 6) edge sizes of a K4 (pairs in lexicographic order); min over contraction trees (3 pairings, 12
    caterpillars) of the max step cost.  Vectorised over n."""
    om = Om(E @ _SA.T, E @ _SB.T, E @ _SC.T, w)          # (n, 30): two merges per tree
    c = np.maximum(np.maximum(om[:, 0::2], om[:, 1::2]), E @ _SD.T)
    return c.min(1)


# ---------------- per-rectangle contributions ----------------
def rho_G(a):
    return np.minimum(2.0, 1 + a + H(a) / 3)


ZE = np.linspace(0, 7 / 8, 701)
_fE = H(ZE) / 3 + 1 + (L56 - 1) * ZE


def rho_E(a):
    a = np.asarray(a, float)
    idx = np.clip(np.searchsorted(ZE, np.minimum(a, 7 / 8), side='right') - 1, 0, len(ZE) - 1)
    cm = np.maximum.accumulate(_fE)
    return cm[idx]


def rho_EM(a):
    # E-mixed: rank-grouped representation for a <= 1/2, Lemma-2 named representation (or raw) above
    a = np.asarray(a, float)
    return np.where(a <= 0.5, rho_E(a), rho_G(a))


def contributions(kind, p, g):
    n = comb(g, 2)
    if kind == 'B':
        u = np.clip(p, -2, 2).reshape(n, n)
        return np.sqrt(2 + u), np.sqrt(2 - u), 0.0
    a = np.clip(p, 0, 1).reshape(n, n)
    if kind == 'G':
        return rho_G(a), rho_G(1 - a), 0.0
    if kind == 'E':
        return rho_E(a), rho_E(1 - a), 0.0
    if kind == 'U':
        return 1 + a, 2 - a, float(H(a).sum() / 3)
    if kind == 'EM':
        return rho_EM(a), rho_EM(1 - a), 0.0
    raise ValueError(kind)


def term_top(kind, p, g, w):
    rho, kap, ent = contributions(kind, p, g)
    E, F = edges_from(rho, kap, g)
    return ent + min(top_cost(E, g, w), top_cost(F, g, w))


def maxmin_top(kind, g, w, starts=60, seed=0, extra=(), de=3, de_iter=600):
    """max over term profiles of min(rows, columns) of the top cost (+ entropy for U). Numerical (Powell, restarts)."""
    from scipy.optimize import minimize
    n = comb(g, 2) ** 2
    lo, hi = (-2.0, 2.0) if kind == 'B' else (0.0, 1.0)
    mid = 0.0 if kind == 'B' else 0.5
    rng = np.random.default_rng(seed)
    pats = patterns(kind, g)
    pv = sorted(((term_top(kind, p, g, w), i) for i, p in enumerate(pats)), reverse=True)
    cands = [np.full(n, mid)] + [pats[i] for _, i in pv[:8]] + [np.asarray(e, float) for e in extra]
    cands += [rng.uniform(lo, hi, n) for _ in range(starts)]
    best = (-np.inf, None)
    f = lambda p: -term_top(kind, np.clip(p, lo, hi), g, w)
    if g == 3:
        from scipy.optimize import differential_evolution
        for sd in range(de):
            r = differential_evolution(f, [(lo, hi)] * n, seed=seed * 10 + sd, maxiter=de_iter, popsize=25, tol=1e-10,
                                       polish=False)
            cands.append(r.x)
    for x0 in cands:
        r = minimize(f, x0, method='Powell', bounds=[(lo, hi)] * n, options={'xtol': 1e-5, 'ftol': 1e-9, 'maxfev': 20000})
        v = -r.fun
        if v > best[0]:
            best = (v, np.clip(r.x, lo, hi))
    return best


def patterns(kind, g):
    """Structured profiles: sets SP of row pairs and SC of column pairs; rect (P,C) gets 'mid' if P in SP and C in SC,
    rows-expensive if only P in SP, columns-expensive if only C in SC, and one of the three values otherwise."""
    n = comb(g, 2)
    hiR, hiC, mid = (2.0, -2.0, 0.0) if kind == 'B' else (1.0, 0.0, 0.5)
    out = []
    for mP in range(1, 2 ** n):
        for mC in range(1, 2 ** n):
            for rest in (hiR, hiC, mid):
                p = np.full((n, n), rest)
                for P in range(n):
                    for C in range(n):
                        inP, inC = (mP >> P) & 1, (mC >> C) & 1
                        if inP and inC: p[P, C] = mid
                        elif inP: p[P, C] = hiR
                        elif inC: p[P, C] = hiC
                out.append(p.ravel())
    return out


# ---------------- row tables (rows merged; columns symmetric) ----------------
BG = np.linspace(0, 1, 11)
PG = np.linspace(0, 1, 11)


def step_cost(OX, OY, I0, kept, classes, sizeX, sizeY, w):
    """One merge step X . Y inside a row.  kept: weight of blocks kept as (x, y) (x to X, y to Y); classes: list of
    (weight W, compressed size r): bypass amount beta (inner +beta, theta of beta*r split phi to X / 1-phi to Y, the
    side receiving theta is re-indexed: its size + its share), fused W - beta (x to X, y to Y, compressed after).
    Returns (min cost, compressed output size)."""
    OX = OX + kept; OY = OY + kept
    grids = [BG * Wc for Wc, _ in classes]
    best = np.inf
    for betas in itertools.product(*grids) if classes else [()]:
        th = sum(b * r for b, (_, r) in zip(betas, classes))
        fu = sum(Wc - b for b, (Wc, _) in zip(betas, classes))
        inner = I0 + sum(betas)
        for ph in PG:
            ox = OX + fu + ph * th; oy = OY + fu + (1 - ph) * th
            c = max(float(Om(ox, inner, oy, w)), sizeX + ph * th, sizeY + (1 - ph) * th)
            best = min(best, c)
    out = OX + OY + sum(Wc * r for Wc, r in classes)
    return best, out


def row_table_cost(g, w, groups):
    """Sequential merge v1, ..., vg of one row.  groups[s] = (kept, classes) per activated rectangle at step s+1
    (all s(g-1) rectangles of the step alike).  Returns the max step cost (one sub-table; naming counted outside)."""
    K = g * g
    size = float(K - 1)           # table of v1
    worst = size
    for s in range(1, g):
        n_act = s * (g - 1)
        fut = g - s - 1
        OX = size - (s * 1.0) - n_act * 1.0      # remove from X's outer: side strings to v_{s+1} and the raw x's
        OY = (g - 1) + fut + fut * (g - 1)       # Y = v_{s+1}: col sides, sides to future nodes, raw diags to future
        kept, classes = groups[s - 1]
        cl = [(n_act * Wc, r) for Wc, r in classes if Wc > 1e-12]
        c, out = step_cost_vec(OX, OY, float(s), n_act * kept, cl, size, float(K - 1), w)
        worst = max(worst, c)
        size = out
    return worst, size


def rect_blocks(kind, v):
    # per-rectangle (kept, classes) for rows merged at parameter v (a for G/U, zeta for E, rho for B)
    if kind in ('G', 'U'):
        return v, [(1 - v, 1.0)]
    if kind == 'E':
        return 0.0, [(1 - v, 1.0), (v, L56)]
    if kind == 'B':
        return 0.0, [(1.0, v)]
    raise ValueError(kind)


def tables_max(kind, g, w, ngrid=11):
    """Upper bound on the row-table cost over all term profiles and both directions (grouped: + naming of the row's
    rectangles, equal split within a step group maximises it).  Max over per-step parameters on a grid."""
    if kind in ('G', 'U'):
        vals = np.linspace(0, 1, ngrid)
    elif kind == 'E':
        vals = np.linspace(0, 7 / 8, ngrid)
    else:
        vals = np.linspace(0, 2, ngrid)
    best = (-np.inf, None)
    for vs in itertools.product(vals, repeat=g - 1):
        groups = [rect_blocks(kind, v) for v in vs]
        c, _ = row_table_cost(g, w, groups)
        ent = 0.0
        if kind == 'G':
            ent = sum(s * (g - 1) * float(H(v)) / 3 for s, v in zip(range(1, g), vs))
        if kind == 'E':
            ent = sum(s * (g - 1) * float(H(v)) / 3 for s, v in zip(range(1, g), vs))
        if c + ent > best[0]:
            best = (c + ent, vs)
    return best


# ---------------- joint per-term cost: min over direction of max(top, row tables) ----------------
_B11 = np.linspace(0, 1, 11)
_PH = np.linspace(0, 1, 11)


def step_cost_vec(OX, OY, I0, kept, classes, sizeX, sizeY, w):
    OX = OX + kept; OY = OY + kept
    if not classes:
        return float(max(Om(OX, I0, OY, w), sizeX, sizeY)), OX + OY
    mesh = np.meshgrid(*([_B11] * len(classes)), _PH, indexing='ij')
    th = sum(m * Wc * r for m, (Wc, r) in zip(mesh[:-1], classes))
    fu = sum((1 - m) * Wc for m, (Wc, r) in zip(mesh[:-1], classes))
    inner = I0 + sum(m * Wc for m, (Wc, r) in zip(mesh[:-1], classes))
    ph = mesh[-1]
    ox = OX + fu + ph * th; oy = OY + fu + (1 - ph) * th
    c = np.maximum(Om(ox, inner, oy, w), np.maximum(sizeX + ph * th, sizeY + (1 - ph) * th))
    return float(c.min()), OX + OY + sum(Wc * r for Wc, r in classes)


def rect_rows(kind, v):
    """(kept, classes, entropy, rho) of one rectangle with rows merged; v = a (G, U, E) or rho (B)."""
    if kind == 'G':
        r = 1 + v + float(H(v)) / 3
        if r >= 2:
            return 1.0, [], 0.0, 2.0
        return v, [(1 - v, 1.0)], float(H(v)) / 3, r
    if kind == 'U':
        return v, [(1 - v, 1.0)], 0.0, 1 + v
    if kind == 'E':   # upper bound: entropy at zeta = min(a, 1/2), blocks at zeta = min(a, 7/8)
        z = min(v, 7 / 8)
        return 0.0, [(1 - z, 1.0), (z, L56)], float(H(min(v, 0.5))) / 3, float(rho_E(v))
    if kind == 'B':
        return 0.0, [(1.0, v)], 0.0, v
    if kind == 'EM':
        if v <= 0.5:
            return 0.0, [(1 - v, 1.0), (v, L56)], float(H(v)) / 3, float(rho_E(v))
        return rect_rows('G', v)


def row_cost_one(g, w, kind, vals_by_pair):
    """One row (or column) node of the merged instance. vals_by_pair[(c1, c2)] = list of the g-1 rectangle params of
    that pair of nodes in the row. Min over merge orders (sequential); returns (cost incl. naming, table size)."""
    K = g * g
    ent = 0.0
    info = {}
    for pr, vs in vals_by_pair.items():
        L = [rect_rows(kind, v) for v in vs]
        info[pr] = L
        ent += sum(x[2] for x in L)
    best = np.inf
    for order in itertools.permutations(range(g)):
        if g > 2 and order[0] > order[1]:
            continue
        size = float(K - 1); worst = size
        for s in range(1, g):
            y = order[s]
            act = [info[(min(order[p], y), max(order[p], y))] for p in range(s)]
            act = [x for L in act for x in L]
            n_act = len(act)
            fut = g - s - 1
            OX = size - s - n_act
            OY = (g - 1) + fut + fut * (g - 1)
            kept = sum(x[0] for x in act)
            # aggregate classes by r (G/U: r = 1; E: r = 1 and L56; B: per-rect r -> proportional class sum)
            if kind == 'B':
                Wt = sum(x[1][0][0] for x in act); Wr = sum(x[1][0][0] * x[1][0][1] for x in act)
                classes = [(Wt, Wr / Wt)] if Wt > 0 else []
            else:
                agg = {}
                for x in act:
                    for Wc, r in x[1]:
                        agg[r] = agg.get(r, 0.0) + Wc
                classes = [(Wc, r) for r, Wc in agg.items() if Wc > 1e-12]
            c, out = step_cost_vec(OX, OY, float(s), kept, classes, size, float(K - 1), w)
            worst = max(worst, c); size = out
        best = min(best, worst)
    return best + ent, size + ent


def tables_dir(kind, P, g, w):
    """P: (nP, nC) params for rows merged (a or rho). Max over the g row nodes."""
    pairs = list(itertools.combinations(range(g), 2))
    pidx = {p: i for i, p in enumerate(pairs)}
    worst = 0.0
    for i in range(g):
        vb = {}
        for C, cp in enumerate(pairs):
            vb[cp] = [P[pidx[(min(i, j), max(i, j))], C] for j in range(g) if j != i]
        c, _ = row_cost_one(g, w, kind, vb)
        worst = max(worst, c)
    return worst


def term_cost(kind, p, g, w, tables=True):
    n = comb(g, 2)
    rho, kap, ent = contributions(kind, p, g)
    E, F = edges_from(rho, kap, g)
    tE, tF = top_cost(E, g, w), top_cost(F, g, w)
    if not tables:
        return ent + min(tE, tF)
    if kind == 'B':
        PR, PC = rho, kap.T
    else:
        a = np.clip(p, 0, 1).reshape(n, n)
        PR, PC = a, (1 - a).T
    cE = max(tE, tables_dir(kind, PR, g, w))
    cF = max(tF, tables_dir(kind, PC, g, w))
    return ent + min(cE, cF)


def maxmin_joint(kind, g, w, starts, seed=0, maxfev=3000, extra=()):
    from scipy.optimize import minimize
    n = comb(g, 2) ** 2
    lo, hi = (-2.0, 2.0) if kind == 'B' else (0.0, 1.0)
    mid = 0.0 if kind == 'B' else 0.5
    rng = np.random.default_rng(seed)
    cands = [np.full(n, mid)] + [np.asarray(e, float) for e in extra] + [rng.uniform(lo, hi, n) for _ in range(starts)]
    f = lambda q: -term_cost(kind, np.clip(q, lo, hi), g, w)
    best = (-np.inf, None)
    for x0 in cands:
        r = minimize(f, x0, method='Powell', bounds=[(lo, hi)] * n, options={'xtol': 1e-4, 'ftol': 1e-7, 'maxfev': maxfev})
        if -r.fun > best[0]:
            best = (-r.fun, np.clip(r.x, lo, hi))
    return best


def tables_crude(kind, g, w, ngrid=11):
    """Valid upper bound on the row-table cost for every term profile and both directions: the sub-table cost depends
    on the profile only through per-step aggregated sums (kept weight / class weights), so take the max over those on
    a grid (each step group uniform), plus the largest possible naming entropy ((g-1) C(g,2) rectangles x 1/3 for
    G and E, 0 for U and B).  Sequential merge order only (an upper bound)."""
    if kind in ('G', 'U'):
        vals = np.linspace(0, 1, ngrid)
    elif kind == 'E':
        vals = np.linspace(0, 7 / 8, ngrid)
    else:
        vals = np.linspace(0, 2, ngrid)
    worst = (-np.inf, None)
    for vs in itertools.product(vals, repeat=g - 1):
        c, _ = row_table_cost(g, w, [rect_blocks(kind, v) for v in vs])
        if c > worst[0]:
            worst = (c, vs)
    ent = (g - 1) * comb(g, 2) / 3 if kind in ('G', 'E') else 0.0
    return worst[0] + ent, worst[1]
