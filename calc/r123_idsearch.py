"""Identity search of Section 4.7 (after Proposition 4.23), stage 1. Strings are single bits (N = 2).

Question: is the crossing f (product of equalities along the edge matching E) a sum of L terms per merge,
f = sum_k sum_{l<L} T_{k,l}, where T_{k,l} is a product of 2x2 factors along the merge-k pairing Pi_k (pairs of strings
that merge k puts into one block, so T_{k,l} has rank 1 across every cut of merge k)?
2D sanity check: the rectangle (strings x, y, u, v; E = xv, yu; rows xy, uv; columns xu, yv) has the App. H identity
[x=v][y=u] = [x=u][y=v] - [x!=y](-1)^x [u!=v](-1)^u, so L = 1 must be solvable there.
Z3 configuration: strings X(i,j) at a_i and Y(i,j) at b_i (i != j in Z3); E pairs X(i,j)-X(j,i), Y(i,j)-Y(j,i);
Pi_k pairs X(i,j) with Y(i+k, j+k).
Least squares from random starts; results appended to calc/r123_idsearch_out.jsonl (resumable)."""
import sys, json, string
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, 'calc')
from resumable import run_jobs


def config(name):
    if name == '2d':
        n = 4  # x y u v
        E = [(0, 3), (1, 2)]
        merges = [[(0, 1), (2, 3)], [(0, 2), (1, 3)]]
        return n, E, merges
    sid = {}
    for side in 'XY':
        for i in range(3):
            for j in range(3):
                if i != j:
                    sid[(side, i, j)] = len(sid)
    E = [(sid[('X', i, j)], sid[('X', j, i)]) for i in range(3) for j in range(i + 1, 3)] + \
        [(sid[('Y', i, j)], sid[('Y', j, i)]) for i in range(3) for j in range(i + 1, 3)]
    merges = [[(sid[('X', i, j)], sid[('Y', (i + k) % 3, (j + k) % 3)]) for i in range(3) for j in range(3) if i != j]
              for k in range(3)]
    return len(sid), E, merges


def product(n, pairs, mats):
    letters = string.ascii_letters
    ops, args = [], []
    for (p, q), M in zip(pairs, mats):
        ops.append(letters[p] + letters[q]); args.append(M)
    return np.einsum(','.join(ops) + '->' + letters[:n], *args).reshape(-1)


def job(j):
    name, L, seed = j['config'], j['L'], j['seed']
    n, E, merges = config(name)
    f = product(n, E, [np.eye(2)] * len(E))
    npair = len(merges[0])
    nvar = len(merges) * L * npair * 4

    def model(z):
        z = z.reshape(len(merges), L, npair, 2, 2)
        return sum(product(n, merges[k], list(z[k, l])) for k in range(len(merges)) for l in range(L))

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(5):
        r = least_squares(lambda z: model(z) - f, rng.normal(size=nvar), method='lm', max_nfev=4000)
        res = float(np.linalg.norm(r.fun) / np.linalg.norm(f))
        if best is None or res < best[0]:
            best = (res, r.x)
    return {'rel_residual': best[0], 'solution': best[1].round(6).tolist() if best[0] < 1e-8 else None}


if __name__ == '__main__':
    jobs = [{'config': '2d', 'L': 1, 'seed': s} for s in range(3)] + \
           [{'config': 'z3', 'L': L, 'seed': s} for L in (1, 2, 3) for s in range(3)]
    res = run_jobs(jobs, job, 'calc/r123_idsearch_out.jsonl', processes=3)
    for j in jobs:
        k = json.dumps(j, sort_keys=True)
        print(j, 'rel residual %.2e' % res[k]['rel_residual'])
