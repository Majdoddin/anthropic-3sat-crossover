"""K = 16 rerun of r123_s1_grid.py's 4 x 4 jobs (Section 2.4, Table 3 of the paper). The first run's Powell search stayed at the uniform profile
(49.94 = the lower end of Table 9); here the starts are the structured profiles of r122d_model (M.patterns: row-pair / column-pair set
patterns) ranked by their joint cost, then Powell. Still a lower estimate of the max over term profiles; the
certified one-level range (Table 9) is [49.94, 57.94] at omega = 2. Output calc/r123_s1_k16_out.txt (+ _jobs.jsonl)."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r122d_model as M
import r123_s1_grid as S
from resumable import run_jobs


def job(j):
    from scipy.optimize import minimize
    w = S.WS[j['w']]; k4 = S.k4_best if j['top'] == 'k4method' else None
    t = time.time()
    f = lambda q: S.term_cost_ab('B', np.clip(q, -2, 2), 4, 4, w, k4)
    pats = M.patterns('B', 4)
    vals = sorted(((f(p), i) for i, p in enumerate(pats)), reverse=True)
    best = (vals[0][0], pats[vals[0][1]])
    for v, i in vals[:j['nstart']]:
        r = minimize(lambda q: -f(q), pats[i], method='Powell', bounds=[(-2, 2)] * 36,
                     options={'xtol': 1e-3, 'ftol': 1e-7, 'maxfev': j['maxfev']})
        if -r.fun > best[0]:
            best = (-r.fun, np.clip(r.x, -2, 2))
    return {'pattern_best': vals[0][0], 'value': best[0], 'profile': np.round(best[1], 3).tolist(),
            'secs': time.time() - t}


if __name__ == '__main__':
    jobs = [{'w': w, 'top': top, 'nstart': 4, 'maxfev': 3000} for w in ('om2', 'current') for top in ('tree', 'k4method')]
    res = run_jobs(jobs, job, os.path.join(HERE, 'r123_s1_k16_jobs.jsonl'), processes=4)
    with open(os.path.join(HERE, 'r123_s1_k16_out.txt'), 'w') as out:
        for j in jobs:
            r = res[json.dumps(j, sort_keys=True)]
            out.write(f"{j['w']:8s} K=16 4x4 B top={j['top']:9s}: best pattern {r['pattern_best']:.4f}, after Powell "
                      f"{r['value']:.4f} (e = {r['value'] / 120:.4f}); profile {r['profile']}\n")
    print(open(os.path.join(HERE, 'r123_s1_k16_out.txt')).read())
