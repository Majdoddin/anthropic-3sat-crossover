"""Exponent per edge bit e(K) of the known one-level shape against the tree (Sections 2.4-2.5, Table 3 of the paper).

(1) a x b grids (the model of r122d_model.py, generalised from g x g): a rows of b nodes; each rectangle (row pair,
    column pair) expanded by App. B pieces (B, m -> infinity) or the rank-counted grouped expansion (E); per term rows
    or columns merged. Rows merged: top = the a merged row nodes, edge = b column edges + sum of the rectangles' rho;
    row tables by a sequential merge with the bypass (r122d_model.step_cost_vec). Columns merged: the transposed grid.
    Used at 2 x 3 (K = 6: top a dot product one way, a triangle the other).
(2) K = 16 two levels: the 4 x 4 grid with the K4 on top contracted by the K = 4 B construction instead of a tree.
    The K = 4 construction for a K4 with unequal edge sizes: choose the pair of opposite edges expanded (3 ways); the
    shorter diagonal is padded to the longer (upper bound); per term rows or columns merged; tables by the same
    merge step. At unit sizes it reproduces the g = 2 value of r122d_model.py (checked below).
Max-min over term profiles is a numerical search (lower estimate of the max; inner K = 4 max over u on a grid).
Results to calc/r123_s1_grid_jobs.jsonl and calc/r123_s1_grid_out.txt."""
import itertools, json, os, sys, time
from math import comb
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r122d_model as M
import r122d_values as RV
from resumable import run_jobs

WS, _ = RV.load()


# ---------------- (1) a x b grid ----------------
def group_cost(nin, nother, w, kind, vals_by_pair):
    """r122d_model.row_cost_one with a row of nin nodes and nother other rows (K = nin (nother + 1))."""
    K = nin * (nother + 1)
    ent, info = 0.0, {}
    for pr, vs in vals_by_pair.items():
        L = [M.rect_rows(kind, v) for v in vs]
        info[pr] = L
        ent += sum(x[2] for x in L)
    best = np.inf
    for order in itertools.permutations(range(nin)):
        if nin > 2 and order[0] > order[1]:
            continue
        size = float(K - 1); worst = size
        for s in range(1, nin):
            y = order[s]
            act = [x for p in range(s) for x in info[(min(order[p], y), max(order[p], y))]]
            fut = nin - s - 1
            OX = size - s - len(act)
            OY = nother + fut + fut * nother
            kept = sum(x[0] for x in act)
            if kind == 'B':
                Wt = sum(x[1][0][0] for x in act); Wr = sum(x[1][0][0] * x[1][0][1] for x in act)
                classes = [(Wt, Wr / Wt)] if Wt > 0 else []
            else:
                agg = {}
                for x in act:
                    for Wc, r in x[1]:
                        agg[r] = agg.get(r, 0.0) + Wc
                classes = [(Wc, r) for r, Wc in agg.items() if Wc > 1e-12]
            c, out = M.step_cost_vec(OX, OY, float(s), kept, classes, size, float(K - 1), w)
            worst = max(worst, c); size = out
        best = min(best, worst)
    return best + ent


def top_nodes(Ed, n, w, k4=None):
    """Ed: dict pair -> edge size on n merged nodes."""
    pairs = list(itertools.combinations(range(n), 2))
    E = np.array([Ed[p] for p in pairs])
    if n == 2:
        return float(E[0])
    if n == 3:
        return float(M.Om(E[0], E[1], E[2], w))
    if n == 4:
        return k4(E, w) if k4 else float(M._k4_top(E[None, :], w)[0])
    raise ValueError(n)


def dir_cost(kind, P, a, b, w, k4=None, tables=True):
    """P[(rowpair index, colpair index)] params for 'rows merged' of an a x b grid (rows have b nodes)."""
    rp = list(itertools.combinations(range(a), 2)); cp = list(itertools.combinations(range(b), 2))
    ri = {p: i for i, p in enumerate(rp)}
    rho = np.array([[M.rect_rows(kind, P[i, j])[3] for j in range(len(cp))] for i in range(len(rp))])
    ent = sum(M.rect_rows(kind, P[i, j])[2] for i in range(len(rp)) for j in range(len(cp)))
    Ed = {p: b + rho[ri[p]].sum() for p in rp}
    top = top_nodes(Ed, a, w, k4)
    if not tables:
        return top
    worst = 0.0
    for i in range(a):
        vb = {c: [P[ri[(min(i, j), max(i, j))], C] for j in range(a) if j != i] for C, c in enumerate(cp)}
        worst = max(worst, group_cost(b, a - 1, w, kind, vb))
    # naming entropy of grouped kinds is per rectangle, counted once at the top (as r122d_model.term_cost: ent + ...)
    return max(top + 0.0, worst), ent


def term_cost_ab(kind, p, a, b, w, k4=None):
    nP, nC = comb(a, 2), comb(b, 2)
    if kind == 'B':
        u = np.clip(np.asarray(p, float), -2, 2).reshape(nP, nC)
        PR, PC = np.sqrt(2 + u), np.sqrt(2 - u).T
    else:
        x = np.clip(np.asarray(p, float), 0, 1).reshape(nP, nC)
        PR, PC = x, (1 - x).T
    cE, eE = dir_cost(kind, PR, a, b, w, k4)
    cF, eF = dir_cost(kind, PC, b, a, w, k4)
    if kind in ('G', 'U'):
        raise ValueError('entropy convention of G/U not ported')
    # r122d_model.term_cost adds the entropy of the rows-direction expansion (0 for B; for E the rect_rows entropy enters
    # through group_cost already); keep r122d_model's convention: ent of contributions() is 0 for B and E.
    return min(cE, cF)


def maxmin_ab(kind, a, b, w, seed, starts, maxfev, k4=None, de=True):
    from scipy.optimize import minimize, differential_evolution
    n = comb(a, 2) * comb(b, 2)
    lo, hi = (-2.0, 2.0) if kind == 'B' else (0.0, 1.0)
    f = lambda q: -term_cost_ab(kind, np.clip(q, lo, hi), a, b, w, k4)
    rng = np.random.default_rng(seed)
    cands = [np.full(n, (lo + hi) / 2)] + [rng.uniform(lo, hi, n) for _ in range(starts)]
    if de:
        r = differential_evolution(f, [(lo, hi)] * n, seed=seed, maxiter=200 if n <= 6 else 40, popsize=15,
                                   tol=1e-9, polish=False)
        cands.append(r.x)
    best = (-np.inf, None)
    for x0 in cands:
        r = minimize(f, x0, method='Powell', bounds=[(lo, hi)] * n,
                     options={'xtol': 1e-4, 'ftol': 1e-8, 'maxfev': maxfev})
        if -r.fun > best[0]:
            best = (-r.fun, np.clip(r.x, lo, hi))
    return best


# ---------------- (2) the K = 4 B construction on a K4 with unequal edges ----------------
_U = np.linspace(-2, 2, 41)
_P4 = list(itertools.combinations(range(4), 2))
_I4 = {p: i for i, p in enumerate(_P4)}


def _e(E, i, j):
    return E[_I4[(min(i, j), max(i, j))]]


def k4_B(E, w):
    """Upper bound (per u on a grid) for the K = 4 B construction on a K4 with edge sizes E (pairs lexicographic).
    Nodes relabelled 00, 01, 10, 11 = (p, q, r, s); diagonals (p,s), (q,r); rows {p,q}, {r,s}; columns {p,r}, {q,s}."""
    best = np.inf
    for (p, q, r, s) in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)]:   # the three pairings as diagonals (p,s), (q,r)
        r0, r1 = _e(E, p, q), _e(E, r, s)
        c0, c1 = _e(E, p, r), _e(E, q, s)
        D = max(_e(E, p, s), _e(E, q, r))
        worst = -np.inf
        for u in _U:
            rho, kap = np.sqrt(2 + u), np.sqrt(2 - u)
            # rows merged: row {p,q}: x = p (outer c0), y = q (outer c1), inner r0; row {r,s}: inner r1
            rows = c0 + c1 + rho * D
            for I0 in (r0, r1):
                cst, _ = M.step_cost_vec(c0, c1, I0, 0.0, [(D, rho)], I0 + c0 + D, I0 + c1 + D, w)
                rows = max(rows, cst)
            cols = r0 + r1 + kap * D
            for I0 in (c0, c1):
                cst, _ = M.step_cost_vec(r0, r1, I0, 0.0, [(D, kap)], I0 + r0 + D, I0 + r1 + D, w)
                cols = max(cols, cst)
            worst = max(worst, min(rows, cols))
        best = min(best, worst)
    return float(best)


def k4_best(E, w):
    return min(k4_B(E, w), float(M._k4_top(np.asarray(E, float)[None, :], w)[0]))


# ---------------- jobs ----------------
def job(j):
    w = WS[j['w']]
    t = time.time()
    if j['what'] == 'k4check':
        return {'k4_B_unit': k4_B(np.ones(6), w)}
    if j['what'] == 'tree':
        return {'tree': M.tree(j['K'], w)[0], 'bw': M.bw(j['K'])}
    k4 = k4_best if j.get('top') == 'k4method' else None
    v, p = maxmin_ab(j['kind'], j['a'], j['b'], w, j['seed'], j['starts'], j['maxfev'], k4, de=j.get('de', True))
    return {'value': v, 'profile': np.round(p, 4).tolist(), 'secs': time.time() - t}


if __name__ == '__main__':
    # consistency: the a x b code at 3 x 3 must equal r122d_model.term_cost on random profiles
    rng = np.random.default_rng(1)
    for kind in ('B', 'E'):
        for _ in range(3):
            lo, hi = (-2, 2) if kind == 'B' else (0, 1)
            p = rng.uniform(lo, hi, 9)
            x, y = term_cost_ab(kind, p, 3, 3, WS['om2']), M.term_cost(kind, p, 3, WS['om2'])
            assert abs(x - y) < 1e-9, (kind, x, y)
    jobs = []
    for wk in ('om2', 'current'):
        jobs.append({'what': 'k4check', 'w': wk})
        for K in (4, 6, 9, 16):
            jobs.append({'what': 'tree', 'w': wk, 'K': K})
        for kind in ('B', 'E'):
            jobs.append({'what': 'grid', 'w': wk, 'kind': kind, 'a': 2, 'b': 3, 'seed': 1, 'starts': 20, 'maxfev': 4000})
        jobs.append({'what': 'grid', 'w': wk, 'kind': 'B', 'a': 4, 'b': 4, 'seed': 1, 'starts': 3, 'maxfev': 3000,
                     'de': False})
        jobs.append({'what': 'grid', 'w': wk, 'kind': 'B', 'a': 4, 'b': 4, 'seed': 1, 'starts': 3, 'maxfev': 3000,
                     'de': False, 'top': 'k4method'})
    res = run_jobs(jobs, job, os.path.join(HERE, 'r123_s1_grid_jobs.jsonl'), processes=3)
    with open(os.path.join(HERE, 'r123_s1_grid_out.txt'), 'w') as out:
        out.write('R123 S1: one-level shapes vs trees (exponents of N; e = value / C(K,2)); search values are lower '
                  'estimates of the max over term profiles\n')
        for j in jobs:
            r = res[json.dumps(j, sort_keys=True)]
            if j['what'] == 'grid':
                K = j['a'] * j['b']
                out.write(f"{j['w']:8s} K={K:2d} {j['a']}x{j['b']} {j['kind']} top={j.get('top', 'tree'):9s}: "
                          f"{r['value']:.4f} (e = {r['value'] / comb(K, 2):.4f}) profile {r['profile']}\n")
            else:
                out.write(f"{j['w']:8s} {j}: {r}\n")
    print(open(os.path.join(HERE, 'r123_s1_grid_out.txt')).read())
