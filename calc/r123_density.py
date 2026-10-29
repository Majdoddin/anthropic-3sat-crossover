"""Table 4 of the paper, columns (P), (G), (X): main-term thresholds of Theorem A(c) (bound in m) against the clause-parameterised
competitor (Peng-Sheng-Xiao 2^{0.5283 m}, bar e < 0.5283/6 at every density) and the general n-parameterised one
(2^{0.711 n}, bar e < 0.711/(6 Delta)); the bar is the larger (weaker competitor... no: the theorem must beat the
better competitor, so bar = min of the two). Uses calc/r123_levers.py (run, LEVERS); resumable jobs."""
import sys, json
sys.path.insert(0, 'calc')
from resumable import run_jobs
from r123_levers import run, LEVERS

USE = ['paper: schedule 2^{-l-5}, tau 1, x0 18', 'free slack schedule (DP), tau 1, x0 18',
       'free schedule, tau 1, best x0', 'free schedule, tau 9, best x0, rho = 1/2 (ideal pieces)']


def job(j):
    kw = dict(dict(LEVERS)[j['lever']]); kw['x0'] = 18 if j['x0'] is None else j['x0']
    r = run(j['bar'], **kw)
    return None if r is None else {'levels': r[0], 'log2K': r[1], 'e': r[2]}


if __name__ == '__main__':
    deltas = [1.0, 1.35, 2.0, 3.0, 4.26]
    bars = {D: min(0.5283, 0.711 / D) / 6 for D in deltas}
    jobs = [{'delta': D, 'bar': bars[D], 'lever': L, 'x0': x0} for D in deltas for L in USE
            for x0 in (range(3, 25) if dict(LEVERS)[L].get('x0') == 'best' else [None])]
    res = run_jobs(jobs, job, 'calc/r123_density_jobs.jsonl', processes=3)
    with open('calc/r123_density_out.txt', 'w') as out:
        for D in deltas:
            out.write(f'\n== density {D}: bar e < {bars[D]:.4f} ({"clause competitor 0.5283 m" if 0.5283 <= 0.711 / D else "variable competitor 0.711 n"})\n')
            for L in USE:
                cand = [(j['x0'], res[json.dumps(j, sort_keys=True)]) for j in jobs if j['delta'] == D and j['lever'] == L]
                cand = [(x0, r) for x0, r in cand if r]
                x0, r = min(cand, key=lambda t: t[1]['log2K'])
                out.write(f'  {L:58s} x0 {x0 if x0 is not None else 18:2d}: levels {r["levels"]}, log2 K = {r["log2K"]:8.0f}; '
                          f'log2 m* >= {9 * r["log2K"]:9.0f} (q = 9), {2 * r["log2K"]:8.0f} (q = 2)\n')
