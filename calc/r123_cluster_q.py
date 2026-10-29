"""Clustering lemma (Lemma 3.8 = Lemma C, Proposition 3.9 of the paper): the threshold log2 m* when the top level of Theorem A(c) uses
Lemma 8 (ii) with the clustering of Lemma C, replacing the crude 'log2 m* = 2 log2 K' of r123_density.py.

Top level (theta = 2, kappa = 1/ln 2, k = 3): exponent <= 2km [e (1 + 2 kappa zeta (1 + eps)) + 2 kappa (1 + eps)/K + nu e]
must stay below 2km * bar. The recursion (r123_levers.run, free schedule) is run to the target bar * f (f < 1), which
leaves room for the top factor; then zeta_top is the largest zeta that fits, s the least integer with
2 zeta^2 s >= ln(2 s K^2), g = s K, and (T1): m* = (4 + 2 ln2 g^2 (log2(4 k g^2) + 1)) / eps.
Also printed: the largest C0 for which (T2) C0 K^2 log2(2 K k m) + e K^2 + g log2 K <= nu 2 e k m holds at m = m*
(log2 of it). Main terms of the recursion as in r123_levers (lower bound side of the paper's own levels)."""
import sys, json, math
sys.path.insert(0, 'calc')
from resumable import run_jobs
from r123_levers import run

K3 = 3
KAPPA = 1 / math.log(2)
NU = 1e-3
FS = [0.999, 0.998, 0.995, 0.99, 0.98, 0.97, 0.95, 0.92, 0.9]
X0S = list(range(4, 11))
EPSS = [0.25, 0.5, 1.0, 2.0]


def job(j):
    r = run(j['bar'] * j['f'], x0=j['x0'])
    return None if r is None else {'levels': r[0], 'log2K': r[1], 'e': r[2]}


def s_least(x, zeta):
    """least integer s >= 1 with 2 zeta^2 s >= ln(2 s K^2), K = 2^x."""
    s = 1
    while 2 * zeta ** 2 * s < math.log(2) * (1 + math.log2(s) + 2 * x):
        s = max(s + 1, math.ceil(math.log(2) * (1 + math.log2(s) + 2 * x) / (2 * zeta ** 2)))
    return s


def top(bar, x, e, eps):
    """(log2 m*, zeta, log2 s, log2 C0max) or None."""
    room = bar / e - 1 - NU - 2 * KAPPA * (1 + eps) * 2.0 ** -x / e
    if room <= 0:
        return None
    zeta = min(1.0, room / (2 * KAPPA * (1 + eps)))
    s = s_least(x, zeta)
    lg = math.log2(s) + x
    lm = math.log2(2 * math.log(2)) + 2 * lg + math.log2(math.log2(4 * K3) + 2 * lg + 1) - math.log2(eps)
    # (T2) at m = 2^lm, in log2: C0 <= (nu 2 e k m - e K^2 - g log2 K) / (K^2 log2(2Kkm)); the subtracted terms are
    # below 2^-60 of the first at these sizes, so drop them
    lc0 = math.log2(NU * 2 * e * K3) + lm - 2 * x - math.log2(1 + x + math.log2(K3) + lm)
    return lm, zeta, math.log2(s), lc0


if __name__ == '__main__':
    deltas = [1.0, 1.35, 2.0, 3.0, 4.26]
    bars = {D: min(0.5283, 0.711 / D) / 6 for D in deltas}
    jobs = [{'delta': D, 'bar': bars[D], 'f': f, 'x0': x0} for D in deltas for f in FS for x0 in X0S]
    res = run_jobs(jobs, job, 'calc/r123_cluster_q_jobs.jsonl', processes=3)
    with open('calc/r123_cluster_q_out.txt', 'w') as out:
        out.write(f'theta = 2, k = {K3}, nu = {NU}; f = recursion target / bar; m* from (T1); C0max from (T2)\n')
        for D in deltas:
            bar = bars[D]
            best, base = None, None
            for j in jobs:
                if j['delta'] != D:
                    continue
                r = res[json.dumps(j, sort_keys=True)]
                if r is None:
                    continue
                if j['f'] == FS[0] and (base is None or r['log2K'] < base):
                    base = r['log2K']
                for eps in EPSS:
                    t = top(bar, r['log2K'], r['e'], eps)
                    if t and (best is None or t[0] < best[0][0]):
                        best = (t, j, r, eps)
            (lm, zeta, ls, lc0), j, r, eps = best
            out.write(f'\n== density {D}: bar {bar:.4f}\n'
                      f'  best: f {j["f"]}, x0 {j["x0"]}, levels {r["levels"]}, log2 K = {r["log2K"]:.0f}, e = {r["e"]:.5f}; '
                      f'top zeta = {zeta:.4g} (log2 {math.log2(zeta):.2f}), eps = {eps}, log2 s = {ls:.1f}\n'
                      f'  log2 m* = {lm:.0f}, q = log2 m*/log2 K = {lm / r["log2K"]:.4f}; log2 C0max = {lc0:.1f}\n'
                      f'  (crude 2 log2 K at f = {FS[0]}: {2 * base:.0f})\n')
