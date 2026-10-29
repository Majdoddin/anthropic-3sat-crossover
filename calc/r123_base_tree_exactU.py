# Exact edge-bit total (eq. (8), column (U) of Table 4): the clause network has U = 2(km - n) = (6 - 2/Delta) m bits, not the bound 2km = 6m; bar = min(0.5283, 0.711/Delta)/(6 - 2/Delta). Copy of r123_base_tree.py otherwise; results in r123_base_tree_exactU_out.txt.
"""Base case by a contraction tree instead of brute force (Lemma 3.7, column (T) of Table 4 of the paper).
Brute force at level 0 costs 2^{e0 K0^2 lambda} with e0 = 1/2 (Anthropic's Theorem A). A tree: bisect the K0 nodes; each half's
table over the K0^2/4 crossing strings is one product of its two quarters' tables, outer 2^{lambda K0^2/8} per side,
inner 2^{lambda K0^2/16} (inner/outer = 1/2), cost 2^{lambda (K0^2/8) omega(1,1/2,1)}; then the final sum over the
crossing strings costs 2^{lambda K0^2/4}. So e0 = omega(1,1/2,1)/8: 1/4 if alpha >= 1/2 (values.yaml: AvdBZ/Uznanski
preprints), 2.042994/8 = 0.25537 on VXXZ's peer-reviewed curve. Lower-order terms (quarters, polynomial factors)
ignored as in r123_levers. Main-term thresholds, same model as r123_density.py ('free schedule, tau 1, best x0')."""
import sys, json
sys.path.insert(0, 'calc')
from resumable import run_jobs
from r123_levers import run

E0S = [('brute force (paper)', 0.5), ('tree, VXXZ curve', 2.042994 / 8), ('tree, alpha >= 1/2', 0.25)]


def job(j):
    r = run(j['bar'], x0=j['x0'], e0=j['e0'])
    return None if r is None else {'levels': r[0], 'log2K': r[1], 'e': r[2]}


if __name__ == '__main__':
    deltas = [1.0, 2.0, 3.0, 4.26]
    bars = {D: min(0.5283, 0.711 / D) / (6 - 2 / D) for D in deltas}
    jobs = [{'delta': D, 'bar': bars[D], 'e0': e0, 'x0': x0} for D in deltas for _, e0 in E0S for x0 in range(3, 25)]
    res = run_jobs(jobs, job, 'calc/r123_base_tree_exactU_jobs.jsonl', processes=3)
    with open('calc/r123_base_tree_exactU_out.txt', 'w') as out:
        for D in deltas:
            out.write(f'\n== density {D}: bar e < {bars[D]:.4f}\n')
            for name, e0 in E0S:
                cand = [(j['x0'], res[json.dumps(j, sort_keys=True)]) for j in jobs if j['delta'] == D and j['e0'] == e0]
                x0, r = min([(x, r) for x, r in cand if r], key=lambda t: t[1]['log2K'])
                out.write(f'  {name:24s} e0 = {e0:.5f}: x0 {x0:2d}, levels {r["levels"]}, log2 K = {r["log2K"]:8.0f}; '
                          f'log2 m* >= {9 * r["log2K"]:8.0f} (q = 9), {2 * r["log2K"]:7.0f} (q = 2 crude)\n')
    print(open('calc/r123_base_tree_exactU_out.txt').read())
