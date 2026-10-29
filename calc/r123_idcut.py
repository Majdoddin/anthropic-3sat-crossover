"""Section 4.7 of the paper: rigorous lower bound on the number of merge-local product terms in f = sum_k sum_l T_{k,l} (Z3 config).
For a cut Q of the 12 strings, rank_Q(f) = N^{e(Q)} (e = E-pairs crossing Q) and a merge-k-local product term has
rank_Q <= N^{p_k(Q)} (p_k = Pi_k-pairs crossing Q). So sum_k L_k N^{p_k(Q)} >= N^{e(Q)} for every Q."""
import sys
sys.path.insert(0, 'calc')
from r123_idsearch import config
n, E, merges = config('z3')
best = {}
for Q in range(1, 2 ** (n - 1)):
    side = [(Q >> s) & 1 for s in range(n)]
    cross = lambda pairs: sum(side[p] != side[q] for p, q in pairs)
    e, ps = cross(E), tuple(cross(m) for m in merges)
    key = max(ps)
    if e - key > best.get('gap', -99):
        best = {'gap': e - key, 'e': e, 'p': ps, 'Q': [s for s in range(n) if side[s]]}
print(best)
for N in (2, 8):
    g = best
    print(f'N = {N}: total number of local product terms >= N^e / max_k N^p_k / 1 = {N ** g["e"] / max(N ** p for p in g["p"]):.0f}'
          f' (sum over the three merges)')
