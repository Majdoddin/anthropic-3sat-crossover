# Section 4.6 (after Proposition 4.22): lower bound for the three-cut size game at K4 over ALL per-block piece
# families, assuming the rank triangle inequality (each log flattening rank of a piece
# is <= the sum of the other two). Any family for matching k must contain a piece P^A = (2, qA, rA) full on its
# cut A and a piece P^B = (pB, 2, rB) full on its cut B (pigeonhole, App. H), with qA + rA >= 2, pB + rB >= 2
# (triangle); raw = (2,2,0) is the case P^A = P^B. Extra pieces only add adversary options, and naming entropy
# only adds cost, so min over (P^A, P^B) of the adversary's LP value is a lower bound on the size exponent of
# every per-block construction that uses all three cuts.
import itertools, numpy as np
from scipy.optimize import linprog
CROSS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
def value(fams):
    opts = fams; nvar = sum(len(o) for o in opts) + 1
    offs = np.cumsum([0] + [len(o) for o in opts])[:-1]
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for c in range(3):
        row = np.zeros(nvar); row[-1] = 1.0
        for k in range(3):
            A, B = CROSS[k]
            for j, (p, q, r) in enumerate(opts[k]):
                row[offs[k]+j] -= p if c == A else (q if c == B else r)
        A_ub.append(row); b_ub.append(0.0)
    for k in range(3):
        row = np.zeros(nvar); row[offs[k]:offs[k]+len(opts[k])] = 1; A_eq.append(row); b_eq.append(1.0)
    cobj = np.zeros(nvar); cobj[-1] = -1
    res = linprog(cobj, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                  bounds=[(0, None)]*(nvar-1) + [(None, None)])
    return -res.fun
def fam(qA, rA, pB, rB): return [(2.0, qA, rA), (pB, 2.0, rB)]
grid = np.linspace(0, 2, 9)
pairs = [(q, r) for q in grid for r in grid if q + r >= 2 - 1e-12]
print(f"admissible (q, r) pairs on the grid: {len(pairs)}")
# (i) cyclically symmetric families
best = (9, None)
for (qA, rA), (pB, rB) in itertools.product(pairs, pairs):
    f = fam(qA, rA, pB, rB); v = value([f, f, f])
    if v < best[0]: best = (v, (qA, rA, pB, rB))
print(f"(i) symmetric families: min over families of adversary value = {best[0]:.4f} at (qA,rA,pB,rB) = {best[1]}")
# (ii) independent families per matching, random search + local coordinate descent
rng = np.random.default_rng(7)
bestv, bestf = 9, None
for trial in range(3000):
    F = [fam(*pairs[rng.integers(len(pairs))], *pairs[rng.integers(len(pairs))]) for _ in range(3)]
    v = value(F)
    if v < bestv: bestv, bestf = v, F
improved = True
while improved:
    improved = False
    for k in range(3):
        for (qA, rA), (pB, rB) in itertools.product(pairs, pairs):
            F = list(bestf); F[k] = fam(qA, rA, pB, rB); v = value(F)
            if v < bestv - 1e-9: bestv, bestf, improved = v, F, True
print(f"(ii) independent families (random + coordinate descent): min value = {bestv:.4f}")
print(f"     families: {bestf}")
# (iii) two-cut reference: M1, M2 raw, M3 ideal rho = 1/2 pieces (2,0,2), (0,2,2)
print(f"(iii) reference, r122b layout with ideal pieces: {value([[(2,2,0)], [(2,2,0)], [(2,0,2),(0,2,2)]]):.4f}")
