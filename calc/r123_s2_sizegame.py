"""The table-size game at K = 4, 6, 9 (Section 4.6, Table 8 and Appendix C of the paper).

Model (the size game of Section 4.6, generalised): the K-clique's edges are raw (cost 1 on every cut they cross) or grouped into
crossings: a crossing is 4 nodes (a, b, c, d) with the two edges (a, d), (b, c); its 2|2 splits are rows {a,b}|{c,d},
cols {a,c}|{b,d} (both edges cross) and internal {a,d}|{b,c} (neither crosses). A crossing is expanded per block into
pieces; a term is a product of pieces; the adversary mixes the pieces of each crossing (one mixture per crossing); the
term picks ANY contraction tree of the K nodes; its table size on a tree edge = the term's log-rank across that cut =
sum over raw edges crossing + sum over crossings of the mixture-averaged piece cost on the split the cut induces.
Value = max over mixtures of min over trees of the largest table (leaf tables included: K - 1). Entropy and bilinear
cost ignored, so for a piece family the value is a lower bound for every construction with those pieces (sizes only).
Piece families (per crossing, costs on rows / cols / internal / 1|3 splits):
  raw   : 2, 2, 0, 1.
  ideal : two pieces (2, 0, 2) and (0, 2, 2), 1|3 cost 1 (mixture t: 2t, 2(1-t), 2, 1). Allowed by every known
          constraint (pigeonhole, subadditivity, complement symmetry, the 4-party rank inequality r_BC <= r_AB r_AC,
          Song-Chen-Sun-Hu 2021 (arXiv 2105.07679), which forces internal >= |rows - cols| = 2); their existence is the
          open K = 4 question (window [3, 3.547]).
  B     : App. B pieces, rows sqrt(2+u), cols sqrt(2-u), internal set to the least value the inequality allows,
          |rows - cols|, 1|3 cost max(0, max(rows, cols, internal) - 1) (least allowed by subadditivity).
Min over trees by a DP over subsets: g(S) = cost of building the table of S = min over S = A + B of
max(cut(A), cut(B), g(A), g(B), cut(S)); value = min over (S, complement) of max(g(S), g(complement)).
The max over mixtures is a numerical search (lower estimate of the max); for the 2 x 3 ideal family also on a grid
with the Lipschitz bound (each cut is 2-Lipschitz per mixture coordinate), which makes it an upper bound."""
import itertools, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from resumable import run_jobs


def grid_layout(a, b):
    """a x b grid, node (i, j) = i b + j; crossings = rectangles (rows i<i', cols j<j'): a=(i,j), b=(i,j'),
    c=(i',j), d=(i',j'); diagonals (a,d), (b,c); rows {a,b}|{c,d} = the two grid rows."""
    K = a * b
    n = lambda i, j: i * b + j
    cross = []
    for i, i2 in itertools.combinations(range(a), 2):
        for j, j2 in itertools.combinations(range(b), 2):
            cross.append((n(i, j), n(i, j2), n(i2, j), n(i2, j2)))
    diag = {frozenset((A, D)) for A, B, C, D in cross} | {frozenset((B, C)) for A, B, C, D in cross}
    raw = [e for e in itertools.combinations(range(K), 2) if frozenset(e) not in diag]
    return K, raw, cross


class Game:
    def __init__(self, K, raw, cross):
        self.K, self.raw, self.cross = K, raw, cross
        masks = np.arange(1 << K)
        bit = lambda v: (masks >> v) & 1
        self.rawcut = sum((bit(u) ^ bit(v)) for u, v in raw).astype(float)
        # split type of each crossing under each mask: 0 none, 1 rows, 2 cols, 3 internal, 4 one-vs-three
        self.split = []
        for A, B, C, D in cross:
            xa, xb, xc, xd = bit(A), bit(B), bit(C), bit(D)
            cnt = xa + xb + xc + xd
            t = np.zeros(1 << K, dtype=np.int8)
            t[(cnt == 1) | (cnt == 3)] = 4
            two = cnt == 2
            t[two & (xa == xb)] = 1
            t[two & (xa == xc)] = 2
            t[two & (xa == xd)] = 3
            self.split.append(t)
        full = (1 << K) - 1
        # subsets in order of size, with their proper splits (A contains the lowest bit of S)
        self.order = sorted(range(1, full + 1), key=lambda s: bin(s).count('1'))

    def cut(self, costs):
        """costs: list per crossing of (rows, cols, internal, single) mixture-averaged."""
        c = self.rawcut.copy()
        for t, (r, q, i, s) in zip(self.split, costs):
            c += np.choose(t, [0.0, r, q, i, s])
        return c

    def value(self, costs):
        cut = self.cut(costs)
        K, full = self.K, (1 << self.K) - 1
        g = np.full(1 << K, np.inf)
        for S in self.order:
            if S & (S - 1) == 0:
                g[S] = max(cut[S], K - 1.0)   # the input node table
                continue
            low = S & -S
            best = np.inf
            A = (S - 1) & S
            while A:
                if A & low:
                    B = S ^ A
                    v = max(g[A], g[B])
                    if v < best:
                        best = v
                A = (A - 1) & S
            g[S] = max(best, cut[S])
        # unrooted: root at a split (S, full - S) with S containing node 0
        best = np.inf
        S = full
        A = (S - 1) & S
        while A:
            if A & 1:
                v = max(g[A], g[full ^ A])
                best = min(best, v)
            A = (A - 1) & S
        return best


def fam_costs(kind, x):
    if kind == 'raw':
        return (2.0, 2.0, 0.0, 1.0)
    if kind == 'ideal':
        t = float(np.clip(x, 0, 1))
        return (2 * t, 2 * (1 - t), 2.0, 1.0)
    if kind == 'B':
        u = float(np.clip(x, -2, 2))
        p, q = np.sqrt(2 + u), np.sqrt(2 - u)
        r = abs(p - q)
        return (p, q, r, max(0.0, max(p, q, r) - 1))
    raise ValueError(kind)


def maxmin(game, kind, seed, starts):
    from scipy.optimize import minimize, differential_evolution
    m = len(game.cross)
    lo, hi = (-2.0, 2.0) if kind == 'B' else (0.0, 1.0)
    f = lambda x: -game.value([fam_costs(kind, xi) for xi in x])
    r = differential_evolution(f, [(lo, hi)] * m, seed=seed, maxiter=60, popsize=12, tol=1e-9, polish=False)
    best = (-r.fun, r.x)
    rng = np.random.default_rng(seed)
    for x0 in [r.x, np.full(m, (lo + hi) / 2)] + [rng.uniform(lo, hi, m) for _ in range(starts)]:
        s = minimize(f, x0, method='Powell', bounds=[(lo, hi)] * m, options={'xtol': 1e-4, 'ftol': 1e-9, 'maxfev': 3000})
        if -s.fun > best[0]:
            best = (-s.fun, s.x)
    return best


def job(j):
    t0 = time.time()
    K, raw, cross = grid_layout(j['a'], j['b'])
    G = Game(K, raw, cross)
    if j['kind'] == 'raw':
        return {'value': G.value([fam_costs('raw', 0)] * len(cross)), 'secs': time.time() - t0}
    if j.get('grid'):
        # upper bound for the ideal family: max over a grid of step h, plus 2 * (h/2) per coordinate (Lipschitz)
        h = 1.0 / j['grid']
        pts = np.linspace(0, 1, j['grid'] + 1)
        best = -np.inf
        for x in itertools.product(pts, repeat=len(cross)):
            best = max(best, G.value([fam_costs('ideal', xi) for xi in x]))
        return {'grid_max': best, 'upper': best + 2 * (h / 2) * len(cross), 'secs': time.time() - t0}
    v, x = maxmin(G, j['kind'], j['seed'], j['starts'])
    return {'value': v, 'profile': np.round(x, 4).tolist(), 'secs': time.time() - t0}


if __name__ == '__main__':
    jobs = []
    for a, b in ((2, 2), (2, 3), (3, 3)):
        jobs.append({'a': a, 'b': b, 'kind': 'raw'})
        for kind in ('ideal', 'B'):
            jobs.append({'a': a, 'b': b, 'kind': kind, 'seed': 1, 'starts': 4 if a * b < 9 else 1})
    jobs.append({'a': 2, 'b': 3, 'kind': 'ideal', 'grid': 40})
    res = run_jobs(jobs, job, os.path.join(HERE, 'r123_s2_sizegame_jobs.jsonl'), processes=2)
    with open(os.path.join(HERE, 'r123_s2_sizegame_out.txt'), 'w') as out:
        out.write('R123 S2 size game (exponents of N, sizes only; e = value / C(K,2); search = lower estimate of '
                  'the max over mixtures unless "upper")\n')
        for j in jobs:
            r = res[json.dumps(j, sort_keys=True)]
            K = j['a'] * j['b']; C2 = K * (K - 1) // 2
            v = r.get('value', r.get('upper'))
            out.write(f"K={K} {j['a']}x{j['b']} {j['kind']:5s} {'grid' if j.get('grid') else ''}: {json.dumps(r)}  "
                      f"(e = {v / C2:.4f})\n")
    print(open(os.path.join(HERE, 'r123_s2_sizegame_out.txt')).read())
